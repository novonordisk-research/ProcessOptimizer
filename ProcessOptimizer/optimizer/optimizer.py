import sys
import warnings
from math import log
from numbers import Number

import numpy as np

from joblib import Parallel, delayed

from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import clone
from sklearn.base import is_regressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import check_random_state

from ..acquisition import _gaussian_acquisition
from ..acquisition import gaussian_acquisition_1D
from ..learning import GaussianProcessRegressor
from ..space import Categorical
from ..space import Space
from ..space.constraints import Constraints
from ..utils import check_x_in_space
from ..utils import cook_estimator
from ..utils import create_result
from ..utils import has_gradients
from ..utils import is_listlike
from ..utils import is_2Dlistlike
from ..utils import normalize_dimensions



class Optimizer(object):
    """Run bayesian optimisation loop.

    An `Optimizer` represents the steps of a bayesian optimisation loop. To
    use it you need to provide your own loop mechanism. The various
    optimisers provided by `ProcessOptimizer` use this class under the hood.

    Use this class directly if you want to control the iterations of your
    bayesian optimisation loop.

    Parameters
    ----------
    * `dimensions` [list, shape=(n_dims,)]:
        List of search space dimensions.
        Each search dimension can be defined either as

        - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
          dimensions),
        - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
          dimensions),
        - as a list of categories (for `Categorical` dimensions), or
        - an instance of a `Dimension` object (`Real`, `Integer` or
          `Categorical`).

    * `base_estimator` ["GP", "RF", "ET", "GBRT" or sklearn regressor, default="GP"]:
        Should inherit from `sklearn.base.RegressorMixin`.
        In addition the `predict` method, should have an optional `return_std`
        argument, which returns `std(Y | x)`` along with `E[Y | x]`.
        If base_estimator is one of ["GP", "RF", "ET", "GBRT"], a default
        surrogate model of the corresponding type is used corresponding to what
        is used in the minimize functions.

    * `n_random_starts` [int, default=10]:
        DEPRECATED, use `n_initial_points` instead.

    * `n_initial_points` [int, default=10]:
        Number of evaluations of `func` with initialization points
        before approximating it with `base_estimator`. Points provided as
        `x0` count as initialization points. If len(x0) < n_initial_points
        additional points are sampled at random.

    * `lhs` [bool, default = False]:
        If set to true the optimizer will use latin hypercube sampling for the first n_initial_points

    * `acq_func` [string, default=`"gp_hedge"`]:
        Function to minimize over the posterior distribution. Can be either

        - `"LCB"` for lower confidence bound.
        - `"EI"` for negative expected improvement.
        - `"PI"` for negative probability of improvement.
        - `"gp_hedge"` Probabilistically choose one of the above three
          acquisition functions at every iteration.
            - The gains `g_i` are initialized to zero.
            - At every iteration,
                - Each acquisition function is optimised independently to
                  propose an candidate point `X_i`.
                - Out of all these candidate points, the next point `X_best` is
                  chosen by $softmax(\eta g_i)$
                - After fitting the surrogate model with `(X_best, y_best)`,
                  the gains are updated such that $g_i -= \mu(X_i)$
        - `"EIps" for negated expected improvement per second to take into
          account the function compute time. Then, the objective function is
          assumed to return two values, the first being the objective value and
          the second being the time taken in seconds.
        - `"PIps"` for negated probability of improvement per second. The
          return type of the objective function is assumed to be similar to
          that of `"EIps

    * `acq_optimizer` [string, `"sampling"` or `"lbfgs"`, default=`"auto"`]:
        Method to minimize the acquistion function. The fit model
        is updated with the optimal value obtained by optimizing `acq_func`
        with `acq_optimizer`.

        - If set to `"auto"`, then `acq_optimizer` is configured on the
          basis of the base_estimator and the space searched over.
          If the space is Categorical or if the estimator provided based on
          tree-models then this is set to be "sampling"`.
        - If set to `"sampling"`, then `acq_func` is optimized by computing
          `acq_func` at `n_points` randomly sampled points.
        - If set to `"lbfgs"`, then `acq_func` is optimized by
              - Sampling `n_restarts_optimizer` points randomly.
              - `"lbfgs"` is run for 20 iterations with these points as initial
                points to find local minima.
              - The optimal of these local minima is used to update the prior.

    * `random_state` [int, RandomState instance, or None (default)]:
        Set random state to something other than None for reproducible
        results.

    * `acq_func_kwargs` [dict]:
        Additional arguments to be passed to the acquistion function.

    * `acq_optimizer_kwargs` [dict]:
        Additional arguments to be passed to the acquistion optimizer.
        options are:
        - "length_scale_bounds" [list] a list of tuples with lower and upper bound
        -  "length_scale" [list] a list of floats
        - "n_restarts_optimizer" [int]
        - "n_jobs" [int]
        
    * `n_objectives` [int, default=1]:
        Number of objectives to be optimized. 
        When n_objectives>1 the optimizer will fit models for each objective and the Pareto front can be approximated using NSGA2



    Attributes
    ----------
    * `Xi` [list]:
        Points at which objective has been evaluated.
    * `yi` [scalar]:
        Values of objective at corresponding points in `Xi`.
    * `models` [list]:
        Regression models used to fit observations and compute acquisition
        function.
    * `space`
        An instance of `ProcessOptimizer.space.Space`. Stores parameter search space used
        to sample points, bounds, and type of parameters.

    """

    def __init__(self, dimensions, base_estimator="gp",
                 n_random_starts=None, n_initial_points=10, lhs=False,
                 acq_func="gp_hedge",
                 acq_optimizer="auto",
                 random_state=None, acq_func_kwargs=None,
                 acq_optimizer_kwargs=None, n_objectives=1):
        self.rng = check_random_state(random_state)
        
        
        # Set the number of objectives
        self.n_objectives= n_objectives

        # Configure acquisition function

        # Store and creat acquisition function set
        self.acq_func = acq_func
        self.acq_func_kwargs = acq_func_kwargs

        allowed_acq_funcs = ["gp_hedge", "EI", "LCB", "PI", "EIps", "PIps" , ""]
        if self.acq_func not in allowed_acq_funcs:
            raise ValueError("expected acq_func to be in %s, got %s" %
                             (",".join(allowed_acq_funcs), self.acq_func))

        # treat hedging method separately
        if self.acq_func == "gp_hedge":
            self.cand_acq_funcs_ = ["EI", "LCB", "PI"]
            self.gains_ = np.zeros(3)
        else:
            self.cand_acq_funcs_ = [self.acq_func]

        if acq_func_kwargs is None:
            acq_func_kwargs = dict()
        self.eta = acq_func_kwargs.get("eta", 1.0)

        # Configure counters of points

        # Check `n_random_starts` deprecation first
        if n_random_starts is not None:
            warnings.warn(("n_random_starts will be removed in favour of "
                           "n_initial_points."),
                          DeprecationWarning)
            n_initial_points = n_random_starts

        if n_initial_points < 0:
            raise ValueError(
                "Expected `n_initial_points` >= 0, got %d" % n_initial_points)
        self._n_initial_points = n_initial_points
        self.n_initial_points_ = n_initial_points

        # record other arguments
        if acq_optimizer_kwargs is None:
            acq_optimizer_kwargs = dict()

        self._length_scale_bounds = acq_optimizer_kwargs.get(
            "length_scale_bounds", None)
        self._length_scale = acq_optimizer_kwargs.get(
            "length_scale", None)
        self.n_points = acq_optimizer_kwargs.get("n_points", 10000)
        self.n_restarts_optimizer = acq_optimizer_kwargs.get(
            "n_restarts_optimizer", 5)
        n_jobs = acq_optimizer_kwargs.get("n_jobs", 1)
        self.n_jobs = n_jobs
        self.acq_optimizer_kwargs = acq_optimizer_kwargs

        # Configure estimator
        self._check_length_scale_bounds(dimensions, self._length_scale_bounds)
        # build base_estimator if doesn't exist
        if isinstance(base_estimator, str):
            base_estimator = cook_estimator(
                base_estimator, space=dimensions,
                random_state=self.rng.randint(0, np.iinfo(np.int32).max), length_scale_bounds=self._length_scale_bounds, length_scale=self._length_scale)

        # check if regressor
        if not is_regressor(base_estimator) and base_estimator is not None:
            raise ValueError(
                "%s has to be a regressor." % base_estimator)

        # treat per second acqusition function specially
        is_multi_regressor = isinstance(base_estimator, MultiOutputRegressor)
        if "ps" in self.acq_func and not is_multi_regressor:
            self.base_estimator_ = MultiOutputRegressor(base_estimator)
        else:
            self.base_estimator_ = base_estimator

        # Configure optimizer

        # decide optimizer based on gradient information
        if acq_optimizer == "auto":
            if has_gradients(self.base_estimator_):
                acq_optimizer = "lbfgs"
            else:
                acq_optimizer = "sampling"

        if acq_optimizer not in ["lbfgs", "sampling"]:
            raise ValueError("Expected acq_optimizer to be 'lbfgs' or "
                             "'sampling', got {0}".format(acq_optimizer))

        if (not has_gradients(self.base_estimator_) and
                acq_optimizer != "sampling"):
            raise ValueError("The regressor {0} should run with "
                             "acq_optimizer"
                             "='sampling'.".format(type(base_estimator)))
        self.acq_optimizer = acq_optimizer

        # Configure search space

        # normalize space if GP regressor
        if isinstance(self.base_estimator_, GaussianProcessRegressor):
            dimensions = normalize_dimensions(dimensions)
        self.space = Space(dimensions)

        if isinstance(self.base_estimator_, GaussianProcessRegressor) and self.space.is_categorical:
                  raise ValueError("GaussianProcessRegressor on a purely categorical space"
                                 " is not supported. Please use another base estimator")          
        # Latin hypercube sampling

        self._lhs = lhs
        if lhs:
            self._lhs_samples = self.space.lhs(n_initial_points)

        # Default is no constraints
        self._constraints = None
        # record categorical and non-categorical indices
        self._cat_inds = []
        self._non_cat_inds = []
        for ind, dim in enumerate(self.space.dimensions):
            if isinstance(dim, Categorical):
                self._cat_inds.append(ind)
            else:
                self._non_cat_inds.append(ind)

        # Initialize storage for optimization

        self.models = []
        self.Xi = []
        self.yi = []

        # Initialize cache for `ask` method responses

        # This ensures that multiple calls to `ask` with n_points set
        # return same sets of points. Reset to {} at every call to `tell` and `set_constraints`.
        self.cache_ = {}

    def copy(self, random_state=None):
        """Create a shallow copy of an instance of the optimizer.

        Parameters
        ----------
        * `random_state` [int, RandomState instance, or None (default)]:
            Set the random state of the copy.
        """

        optimizer = Optimizer(
            dimensions=self.space.dimensions,
            base_estimator=self.base_estimator_,
            n_initial_points=self.n_initial_points_,
            lhs=self._lhs,
            acq_func=self.acq_func,
            acq_optimizer=self.acq_optimizer,
            acq_func_kwargs=self.acq_func_kwargs,
            acq_optimizer_kwargs=self.acq_optimizer_kwargs,
            random_state=random_state, n_objectives=self.n_objectives
        )

        # It is important to copy the constraints so that a call to '_tell()' will create a valid _next_x
        optimizer._constraints = self._constraints
        if self._lhs:
            optimizer._lhs_samples = self._lhs_samples

        if hasattr(self, "gains_"):
            optimizer.gains_ = np.copy(self.gains_)

        if self.Xi:
            optimizer._tell(self.Xi, self.yi)

        return optimizer

    def ask(self, n_points=None, strategy="stbr", space_fill=None):
        """Query point or multiple points at which objective should be evaluated.

        * `n_points` [int or None, default=None]:
            Number of points returned by the ask method.
            If the value is None, a single point to evaluate is returned.
            Otherwise a list of points to evaluate is returned of size
            n_points. This is useful if you can evaluate your objective in
            parallel, and thus obtain more objective function evaluations per
            unit of time.

        * `strategy` [string, default=`"cl_min"`]:
            Method to use to sample multiple points (see also `n_points`
            description). This parameter is ignored if n_points = None.
            Supported options are `"cl_min"`, `"cl_mean"` or `"cl_max"`.
            
            -if set to `"stbr"` then steinerberger sampling is used
              after first point.
            - If set to `"cl_min"`, then constant liar strtategy is used
               with lie objective value being minimum of observed objective
               values. `"cl_mean"` and `"cl_max"` means mean and max of values
               respectively. For details on this strategy see:

               https://hal.archives-ouvertes.fr/hal-00732512/document

               With this strategy a copy of optimizer is created, which is
               then asked for a point, and the point is told to the copy of
               optimizer with some fake objective (lie), the next point is
               asked from copy, it is also told to the copy with fake
               objective and so on. The type of lie defines different
               flavours of `cl_x` strategies.


         * space_fill [string or None, default="None"]
           Method used for space filling after initialization. This parameter is ignored if Space_fill = None.
           Supported options are `"rand"`, `"lhs"` or `"stbr"`.
         
           - If set to "rand", then random points from the parameter space is returned
           
           - If set to "lhs", then points sampled by latin hypercube from the parameter space is returned
           
           - If set to "stbr", then the Steinerberger sampling medthod is used. For details on this method see:

             https://arxiv.org/abs/1902.03269
        """

        if not ((isinstance(n_points, int) and n_points > 0)  or n_points is None) :
            raise ValueError(
                "n_points should be int > 0, got " + str(n_points)
            )

        # These are the only filling strategies which are supported
        
        supported_fill_strategies = ["rand", "lhs", "stbr",None]

        if space_fill not in supported_fill_strategies:
            raise ValueError(
                "Expected space filling to be one of " +
                str(supported_fill_strategies) + ", " + "got %s" % space_fill
            )

        
        if space_fill=="rand":

            return self.space.rvs(n_samples=n_points)
        
        elif space_fill=="lhs": 
            
            return self.space.lhs(n_points)
        
        elif space_fill=="stbr":
            
            # Steienerberger sampling can not be used from an empty Xi set
            if self.Xi == []:
                raise ValueError(
                    "Steinerberger sampling requires initial points but got [] " 
                    )

    
            
            if n_points is None:
                # Returns a single Steinerberger point
                X=self.stbr_scipy()
            else:
                # Returns 'n_points' Steinerberger points
                X=self.stbr_scipy(n_points=n_points)
        
            return X
                
                
        if n_points is None or n_points==1:
            return self._ask()

        supported_strategies = ["cl_min", "cl_mean", "cl_max", "stbr"]

        if not (isinstance(n_points, int) and n_points > 0):
            raise ValueError(
                "n_points should be int > 0, got " + str(n_points)
            )

        if strategy not in supported_strategies:
            raise ValueError(
                "Expected parallel_strategy to be one of " +
                str(supported_strategies) + ", " + "got %s" % strategy
            )

        # Caching the result with n_points not None. If some new parameters
        # are provided to the ask, the cache_ is not used.
        if (n_points, strategy) in self.cache_:
            return self.cache_[(n_points, strategy)]

        # Copy of the optimizer is made in order to manage the
        # deletion of points with "lie" objective (the copy of
        # oiptimizer is simply discarded)
        opt = self.copy(random_state=self.rng.randint(0,
                                                      np.iinfo(np.int32).max))

        X = []
        for i in range(n_points):
            if i>0 and strategy == "stbr" and self._n_initial_points <1:
                x = opt.stbr_scipy()[0]
            else:
                x = opt.ask()
            X.append(x)

            ti_available = "ps" in self.acq_func and len(opt.yi) > 0
            ti = [t for (_, t) in opt.yi] if ti_available else None

            if strategy == "cl_min":
                y_lie = np.min(opt.yi, axis=0).tolist() if opt.yi else np.zeros(opt.n_objectives).tolist()  # CL-min lie
                if opt.n_objectives == 1 and not opt.yi:
                    y_lie= y_lie[0]
                t_lie = np.min(ti) if ti is not None else log(
                    sys.float_info.max)
            elif strategy == "cl_mean":
                y_lie = np.mean(opt.yi, axis=0).tolist() if opt.yi else np.zeros(opt.n_objectives).tolist()  # CL-mean lie
                if opt.n_objectives == 1 and not opt.yi:
                    y_lie= y_lie[0]
                t_lie = np.mean(ti) if ti is not None else log(
                    sys.float_info.max)
            else:
                y_lie = np.max(opt.yi, axis=0).tolist() if opt.yi else np.zeros(opt.n_objectives).tolist() # CL-max lie
                if opt.n_objectives == 1 and not opt.yi:
                    y_lie= y_lie[0]
                t_lie = np.max(ti) if ti is not None else log(
                    sys.float_info.max)

            # Lie to the optimizer.
            if "ps" in self.acq_func:
                # Use `_tell()` instead of `tell()` to prevent repeated
                # log transformations of the computation times.
                opt._tell(x, (y_lie, t_lie))
            else:
                opt._tell(x, y_lie)

        self.cache_ = {(n_points, strategy): X}  # cache_ the result

        return X

    def _ask(self):
        """Suggest next point at which to evaluate the objective.

        Return a random point while not at least `n_initial_points`
        observations have been `tell`ed, after that `base_estimator` is used
        to determine the next point.
        """
        
        
        if self._n_initial_points > 0 or self.base_estimator_ is None:
            # this will not make a copy of `self.rng` and hence keep advancing
            # our random state.

            assert not (
                self._constraints and self._lhs), "Constraints can't be used while latin hypercube sampling is not exhausted"

            if self._constraints:
                # We use another sampling method when constraints are added
                return self._constraints.rvs(random_state=self.rng)[0]
            elif self._lhs:
                # The samples are evaluated starting form lhs_samples[0]
                return self._lhs_samples[len(self._lhs_samples)-self._n_initial_points]
            else:
                return self.space.rvs(random_state=self.rng)[0]
        else:
            if not self.models:
                raise RuntimeError("Random evaluations exhausted and no "
                                   "model has been fit.")

            next_x = self._next_x
            
            min_delta_x = min([self.space.distance(next_x, xi)
                           for xi in self.Xi])
            
            
            
            if abs(min_delta_x) <= 1e-8:
                warnings.warn("The objective has been evaluated "
                              "at this point before.")
                

            # return point computed from last call to tell()
            return next_x

    def tell(self, x, y, fit=True):
        """Record an observation (or several) of the objective function.

        Provide values of the objective function at points suggested by `ask()`
        or other points. By default a new model will be fit to all
        observations. The new model is used to suggest the next point at
        which to evaluate the objective. This point can be retrieved by calling
        `ask()`.

        To add observations without fitting a new model set `fit` to False.

        To add multiple observations in a batch pass a list-of-lists for `x`
        and a list of scalars for `y`.

        Parameters
        ----------
        * `x` [list or list-of-lists]:
            Point at which objective was evaluated.

        * `y` [scalar or list]:
            Value of objective at `x`.

        * `fit` [bool, default=True]
            Fit a model to observed evaluations of the objective. A model will
            only be fitted after `n_initial_points` points have been told to
            the optimizer irrespective of the value of `fit`.
        """
        check_x_in_space(x, self.space)
        self._check_y_is_valid(x, y)

        # take the logarithm of the computation times
        if "ps" in self.acq_func:
            if is_2Dlistlike(x):
                y = [[val, log(t)] for (val, t) in y]
            elif is_listlike(x):
                y = list(y)
                y[1] = log(y[1])

        return self._tell(x, y, fit=fit)

    def _tell(self, x, y, fit=True):
        """Perform the actual work of incorporating one or more new points.
        See `tell()` for the full description.

        This method exists to give access to the internals of adding points
        by side stepping all input validation and transformation."""

        if "ps" in self.acq_func:
            if is_2Dlistlike(x):
                self.Xi.extend(x)
                self.yi.extend(y)
                self._n_initial_points -= len(y)
            elif is_listlike(x):
                self.Xi.append(x)
                self.yi.append(y)
                self._n_initial_points -= 1
                
                
         # If we have been handed a batch of multiobjective points  
        elif self.n_objectives> 1 and is_2Dlistlike(x) and is_2Dlistlike(y):               
            self.Xi.extend(x)
            self.yi.extend(y)
            self._n_initial_points -= len(y)
            
            
        # If we have been handed a single multiobjective point    
        elif self.n_objectives> 1 and is_listlike(x) and is_listlike(y):
            self.Xi.append(x)
            self.yi.append(y)
            self._n_initial_points -= 1
            
        # if we have been handed a batch of single objective points
        elif is_listlike(y) and is_2Dlistlike(x) and self.n_objectives==1:
            self.Xi.extend(x)
            self.yi.extend(y)
            self._n_initial_points -= len(y)
            
        # if we have been handed a single point with a single objective        
        elif is_listlike(x) and self.n_objectives==1:
            self.Xi.append(x)
            self.yi.append(y)
            self._n_initial_points -= 1
        else:
            raise ValueError("Type of arguments `x` (%s) and `y` (%s) "
                             "not compatible when number of objectives is (%s)." % (type(x), type(y),self.n_objectives))
 
        # optimizer learned something new - discard cache
        self.cache_ = {}

        # after being "told" n_initial_points we switch from sampling
        # random points to using a surrogate model(s)
        if (fit and self._n_initial_points <= 0 and
                self.base_estimator_ is not None):
            transformed_bounds = np.array(self.space.transformed_bounds)
            est = clone(self.base_estimator_)


            # If the problem containts multiblie objectives a model has to be fitted for each objective
            if self.n_objectives > 1:

                # fit an estimator to each objective
                obj_models=[]
                for i in range(self.n_objectives):
                    est = clone(self.base_estimator_)
                    y_list= [item[i] for item in self.yi]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        est.fit(self.space.transform(self.Xi), y_list)
                    obj_models.append(est)
                
                # Append all objective functions
                self.models.append(obj_models)
                    

                # Setting the probability of using Steinerberger for the next point (exploration)
                # The probability for using the NSGAII algorithm is 1-prob_stbr (exploitation)
                prob_stbr=0.25
                
                # Simulate a random number 
                random_uniform_number= np.random.uniform()

                # The random number decides what strategy to use for the next point
                if random_uniform_number < prob_stbr:
                    
                    # self._next_x is found via stbr_scipy
                    next_x=self.stbr_scipy()
                    self._next_x=next_x[0]

                else:

                    # The Pareto front is approximated using the NSGAII algorithm
                    pop, logbook, front = self.NSGAII()
            
                    # The best point in the Pareto front is found (the point furthest from existing measurements)
                    next_x= self.best_Pareto_point( pop, front)
                    self._next_x = self.space.inverse_transform(
                    next_x.reshape((1, -1)))[0]




                
                
                
                


            if self.n_objectives == 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    est.fit(self.space.transform(self.Xi), self.yi)
                    
                if hasattr(self, "next_xs_") and self.acq_func == "gp_hedge":
                    self.gains_ -= est.predict(np.vstack(self.next_xs_))
                self.models.append(est)
    
                # even with BFGS as optimizer we want to sample a large number
                # of points and then pick the best ones as starting points
                if self._constraints:
                    # We use another sampling method if constraints have been added
                    X = self.space.transform(self._constraints.rvs(
                        n_samples=self.n_points, random_state=self.rng))
                else:
                    X = self.space.transform(self.space.rvs(
                        n_samples=self.n_points, random_state=self.rng))
    
                self.next_xs_ = []
                for cand_acq_func in self.cand_acq_funcs_:
                    values = _gaussian_acquisition(
                        X=X, model=est, y_opt=np.min(self.yi),
                        acq_func=cand_acq_func,
                        acq_func_kwargs=self.acq_func_kwargs)
                    # Find the minimum of the acquisition function by randomly
                    # sampling points from the space. If constraints are present
                    # we use this strategy
                    if self.acq_optimizer == "sampling" or self._constraints:
                        next_x = X[np.argmin(values)]
    
                    # Use BFGS to find the mimimum of the acquisition function, the
                    # minimization starts from `n_restarts_optimizer` different
                    # points and the best minimum is used
                    elif self.acq_optimizer == "lbfgs":
                        x0 = X[np.argsort(values)[:self.n_restarts_optimizer]]
    
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            results = Parallel(n_jobs=self.n_jobs)(
                                delayed(fmin_l_bfgs_b)(
                                    gaussian_acquisition_1D, x,
                                    args=(est, np.min(self.yi), cand_acq_func,
                                          self.acq_func_kwargs),
                                    bounds=self.space.transformed_bounds,
                                    approx_grad=False,
                                    maxiter=20)
                                for x in x0)
    
                        cand_xs = np.array([r[0] for r in results])
                        cand_acqs = np.array([r[1] for r in results])
                        next_x = cand_xs[np.argmin(cand_acqs)]
    
                    # lbfgs should handle this but just in case there are
                    # precision errors.
                    if not self.space.is_categorical:
                        next_x = np.clip(
                            next_x, transformed_bounds[:, 0],
                            transformed_bounds[:, 1])
                    self.next_xs_.append(next_x)
    
                if self.acq_func == "gp_hedge":
                    logits = np.array(self.gains_)
                    logits -= np.max(logits)
                    exp_logits = np.exp(self.eta * logits)
                    probs = exp_logits / np.sum(exp_logits)
                    next_x = self.next_xs_[np.argmax(self.rng.multinomial(1,
                                                                          probs))]
                else:
                    next_x = self.next_xs_[0]
    
                # note the need for [0] at the end
                self._next_x = self.space.inverse_transform(
                    next_x.reshape((1, -1)))[0]
        # Pack results

        return create_result(self.Xi, self.yi, self.space, self.rng,
                             models=self.models)

    def _check_y_is_valid(self, x, y):
        """Check if the shape and types of x and y are consistent."""

        if "ps" in self.acq_func:
            if is_2Dlistlike(x):
                if not (np.ndim(y) == 2 and np.shape(y)[1] == 2):
                    raise TypeError("expected y to be a list of (func_val, t)")
            elif is_listlike(x):
                if not (np.ndim(y) == 1 and len(y) == 2):
                    raise TypeError("expected y to be (func_val, t)")
                    
       # Check batch tell with multiobjective          
        if is_2Dlistlike(x) and is_2Dlistlike(y) and self.n_objectives > 1:
                for y_values in y:
                    for y_value in y_values:
                        if not isinstance(y_value, Number):
                            raise ValueError("expected y to be a list of lists of scalars")
                    
        # Check batch tell with single objective
        elif is_listlike(y) and is_2Dlistlike(x) and self.n_objectives ==1:
                for y_value in y:
                        if not isinstance(y_value, Number):
                            raise ValueError("expected y to be a list of scalars")
                    
        
        # Check single tell with multiobjective
        elif is_listlike(y):
                # Check if the observation has the correct number of objectives
                if not len(y)==self.n_objectives:
                        raise ValueError("y does not have the correct number of objective scores")    
                # Check if all objective scores are numbers            
                for y_value in y:
                        if not isinstance(y_value, Number):
                            raise ValueError("expected y to be a list of scalars")  


        # Check single tell with single objective
        elif is_listlike(x):
            if not isinstance(y, Number):
                raise ValueError("`func` should return a scalar")
                        
                
                

        else:
            raise ValueError("Type of arguments `x` (%s) and `y` (%s) "
                             "not compatible." % (type(x), type(y)))

    def run(self, func, n_iter=1):
        """Execute ask() + tell() `n_iter` times"""
        for _ in range(n_iter):
            x = self.ask()
            self.tell(x, func(x))

        return create_result(self.Xi, self.yi, self.space, self.rng,
                             models=self.models)

    def set_constraints(self, constraints):
        ''' Sets the constraints for the optimizer

        Parameters
        ----------
        * `constraints` [list] or [Constraints]:
            Can either be a list of Constraint objects or a Constraints object
        '''
        if (self._n_initial_points >
                0 and self._lhs):
            raise RuntimeError(
                "Can't set constraints while latin hypercube sampling points are not exhausted.")

        if constraints:
            if isinstance(constraints, Constraints):
                # If constraints is a Constraints object we simply add it
                self._constraints = constraints
            else:
                # If it is a list of constraints we initialize a Constraints object.
                self._constraints = Constraints(constraints, self.space)
        else:
            self._constraints = None

        self.update_next()

    def remove_constraints(self):
        ''' Sets constraints to None'''
        self.set_constraints(None)

    def get_constraints(self):
        '''Returns constraints'''
        return self._constraints

    def update_next(self):
        ''' Updates the value returned by opt.ask(). Useful if a parameter was updated after ask was called.'''
        self.cache_ = {}
        # Ask for a new next_x. Usefull if new constraints have been added or lenght_scale has been tweaked.
        # We only need to overwrite _next_x if it exists.
        if hasattr(self, '_next_x'):
            opt = self.copy(random_state=self.rng)
            self._next_x = opt._next_x

    def get_result(self):
        '''Returns the same result that would be returned by opt.tell() but without calling tell'''
        return create_result(self.Xi, self.yi, self.space, self.rng,
                             models=self.models)

    def _check_length_scale_bounds(self, dimensions, bounds):
        ''' Checks if length scale bounds are of in correct format'''
        space = Space(dimensions)
        n_dims = space.n_dims
        if bounds:
            if isinstance(bounds, list):
                # Check that number of bounds is the same as number of dimensions
                if not len(bounds) == n_dims:
                    raise ValueError("Number of bounds (%s) must be same as number of dimensions (%s)" % (
                        len(bounds), n_dims))
                for i in range(len(bounds)):
                    if not isinstance(bounds[i], tuple):
                        raise TypeError("Each bound must be of type tuple, got %s" %
                                        (type(bounds[i])))
                    if not len(bounds[i]) == 2:
                        raise ValueError("Each bound-tuple must have length 2, got %s" %
                                         (len(bounds[i])))
            else:
                raise TypeError("Expected bounds to be of type list, got %s" %
                                (type(bounds)))
    


    
    
    def stbr_scipy(self, n_points=1):
        from scipy.optimize import minimize
        # Suggestion for improvemenet. Seperate categorical and real/integer space.
        # Only solve Steinerberger minimization in real/integer sub space.
        
        
        # Get bounds for variables in transformed space
        # Needs to be a tuple of tuples with 0 and 1 for each transformed dimension
        bounds=[]
        for i in range(self.space.transformed_n_dims):
            bounds.append((0.0,1.0))
        bounds = tuple(tuple(sub) for sub in bounds) 

        # Copy of the optimizer to not append new points to original  self.Xi
        # Set base estimator to GP so transform always nomalizes
        copy=Optimizer(self.space,'GP', n_objectives=self.n_objectives, n_initial_points=999)
        for i in range(len(self.Xi)):
            if self.n_objectives == 1:
                copy.tell(self.Xi[i], 0)
            elif self.n_objectives > 1:
                copy.Xi.append(self.Xi[i])
                copy.yi.append(np.zeros(self.n_objectives).tolist())
                

        # Initialize list with Steinerberger points
        X=[]
        # In each loop calculate the next Steinerberger point
        for i in range(n_points):    
            # lists with local minimum location and the function value of the Steinerberger sum at that point
            loc_min=[]
            fun_val=[]
            # We use 20 lhs point as initial guesses for minimization
            x0= copy.space.lhs(20)
            x0=copy.space.transform(x0)
            
            # Loop over each initial guess and find a local minimum
            for j in range(len(x0)):
                res=minimize(copy.stbr_fun, x0=x0[j], bounds=bounds)
                loc_min.append(res.x)
                fun_val.append(res.fun)

            # Decide the global minimum as the local minimum with the lowest function value
            glob_min=loc_min[np.argmin(fun_val)]
            next_X=np.asarray(glob_min).reshape(1,copy.space.transformed_n_dims)
            # Transform back to original space
            next_X=copy.space.inverse_transform(next_X)

            # This deals with categorical variables, since they are not suited for Steinerberger
            for dim, n in zip(self.space.dimensions, range(self.space.n_dims)):
                    # Check if dimension is categorical
                    if isinstance(dim, Categorical):
                        # Make array with categories for that dimension
                        categories=dim.categories
                        # Make array with all instances of categories observed
                        instances= np.array(copy.Xi)[:,n]
                        # Calculate the number of instances of each category
                        cat_population=np.zeros(len(categories))
                        for category,i in zip(categories,range(len(categories))):
                            cat_population[i]=(instances.tolist()).count(category)
                        # Find the category with the lowest population
                        least_populated=categories[np.argmin(cat_population)]
                        # Set the category to the lowest populated category
                        next_X[0][n]=least_populated       
            
            # Append point to list of new Steinerberger points
            X.append(next_X[0])
            # Append to Xi of copy optimizer
            copy.Xi.append(next_X[0])
        
        return X
    
    
    # This function returns the Steinerberger sum for a given x
    def stbr_fun(self, x):
        # parameter to ensure that log argument is non-zero
        eta=10**-8
        # Initialize Steinerberger sum
        stbr_sum=0
        # Transform initial points to [0,1]^d space
        Xi=self.space.transform(self.Xi)
        # for loop over all existing points
        for i in range(len(Xi)):
            # Calculate the factors in the Steinerberger term in each dimension
            stbr_vector=(1 - np.log(2 * np.sin(np.pi * abs(x-Xi[i])) + eta) )
            # Calculate the Steinerberger term by multiplying factor from each dimension
            stbr_term=np.prod(stbr_vector)
            # Add term to Steinerberger sum
            stbr_sum = stbr_sum + stbr_term
            
        return stbr_sum
        
        
    # This function returns the objective scores at x estimated by the models
    def __ObjectiveGP(self, x):

        #Fator = 1.0e10
        F = [None] * self.n_objectives
        xx = np.asarray(x).reshape(1, -1)
    
        #Constraints = 0.0
        #for cons in self.constraints:
        #    y = cons['fun'](x)
        #    if cons['type'] == 'eq':
        #        Constraints += np.abs(y)
        #    elif cons['type'] == 'ineq':
        #        if y < 0:
        #            Constraints -= y
    
        for i in range(self.n_objectives):

            F[i] = self.models[-1][i].predict(xx)[0]
    
        return F

# This function returns the point in the Pareto front, which is deemed the best (furthest away from existing observations)
    def best_Pareto_point(self, pop, front, q=0.5):
        
        Population = np.asarray(pop)

        IndexF, FatorF = self.__LargestOfLeast(front, self.yi)

        IndexPop, FatorPop = self.__LargestOfLeast(Population,
                                                   self.space.transform(self.Xi).tolist())

        Fator = q * FatorF + (1- q) * FatorPop
        Index_try = np.argmax(Fator)
    
        best_point = Population[Index_try]
    
        return best_point
    
 # This function is used  in best_Pareto_point. For each point it gives a relative distance to the closest point
    def __LargestOfLeast(self, front, F):
            NF = len(front)
            MinDist = np.empty(NF)
            for i in range(NF):
                MinDist[i] = self.__MinimalDistance(front[i], F)
            ArgMax = np.argmax(MinDist)
    
            Mean = MinDist.mean()
            Std = np.std(MinDist)
            if Std == 0: #Quick workaround to avoid divide by zero, when 0 or 1 datapoints. Investigate alternatives in future.
                return ArgMax, (MinDist-Mean)
            else:
                return ArgMax, (MinDist-Mean)/(Std)
        
    # This fuction returns the minimal distance between a point and a list of points
    @staticmethod
    def __MinimalDistance(X, Y):
        Y=np.asarray(Y)
        N = len(X)
        Npts = len(Y)
        DistMin = float('inf')
        for i in range(Npts):
            Dist = 0.
            for j in range(N):
                Dist += (X[j]-Y[i, j])**2
            Dist = np.sqrt(Dist)
            if Dist < DistMin:
                DistMin = Dist
        return DistMin
    
    # This function calls NSGAII to estimate the Pareto Front
    def NSGAII(self, plot=False, MU=40):

        from ._NSGA2 import NSGAII
        pop, logbook, front = NSGAII(self.n_objectives,
                self.__ObjectiveGP,
                np.array(self.space.transformed_bounds),
                MU=MU)
        
        if plot==True and self.n_objectives == 2:
            print("plotting not yet implemented directly")
            
        return pop, logbook, front

