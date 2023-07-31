from copy import deepcopy
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize as sp_minimize

from joblib import dump as dump_
from joblib import load as load_


__all__ = (
    "load",
    "dump",
)


def create_result(Xi, yi, space=None, rng=None, specs=None, models=None):
    """
    Initialize an `OptimizeResult` object.

    Parameters
    ----------
    * `Xi` [list of lists, shape=(n_iters, n_features)]:
        Location of the minimum at every iteration.

    * `yi` [array-like, shape=(n_iters,)]:
        Minimum value obtained at every iteration.

    * `space` [Space instance, optional]:
        Search space.

    * `rng` [RandomState instance, optional]:
        State of the random state.

    * `specs` [dict, optional]:
        Call specifications.

    * `models` [list, optional]:
        List of fit surrogate models.

    Returns
    -------
    * `res` [`OptimizeResult`, scipy object]:
        OptimizeResult instance with the required information.

       or if the optimizer is multiobjective:
     * `results` [list of `OptimizeResult`, scipy object]:
        OptimizeResult instance with the required information.
    """
    res = OptimizeResult()
    yi = np.asarray(yi)
    if np.ndim(yi) == 1:
        best = np.argmin(yi)
        res.x = Xi[best]
        res.fun = yi[best]
        res.func_vals = yi
        res.x_iters = Xi
        res.models = models
        res.space = space
        res.random_state = rng
        res.specs = specs
        return res
    models = np.asarray(models)
    results = []
    for i in range(yi.shape[1]):
        res = OptimizeResult()
        yi_single = np.ravel(yi[:, i])
        best = np.argmin(yi_single)
        res.x = Xi[best]
        res.fun = yi_single[best]
        res.func_vals = yi_single
        res.x_iters = Xi
        if models.size == 0:
            res.models = models
        else:
            res.models = models[:, i]
        res.space = space
        res.random_state = rng
        res.specs = specs
        results.append(res)
    return results


def eval_callbacks(callbacks, result):
    """Evaluate list of callbacks on result.

    The return values of the `callbacks` are ORed together to give the
    overall decision on whether or not the optimization procedure should
    continue.

    Parameters
    ----------
    * `callbacks` [list of callables]:
        Callbacks to evaluate.

    * `result` [`OptimizeResult`, scipy object]:
        Optimization result object to be stored.

    Returns
    -------
    * `decision` [bool]:
        Decision of the callbacks whether or not to keep optimizing
    """
    stop = False
    if callbacks:
        for c in callbacks:
            decision = c(result)
            if decision is not None:
                stop = stop or decision

    return stop


def dump(res, filename, store_objective=True, **kwargs):
    """
    Store an ProcessOptimizer optimization result into a file.

    Parameters
    ----------
    * `res` [`OptimizeResult`, scipy object]:
        Optimization result object to be stored.

    * `filename` [string or `pathlib.Path`]:
        The path of the file in which it is to be stored. The compression
        method corresponding to one of the supported filename extensions ('.z',
        '.gz', '.bz2', '.xz' or '.lzma') will be used automatically.

    * `store_objective` [boolean, default=True]:
        Whether the objective function should be stored. Set `store_objective`
        to `False` if your objective function (`.specs['args']['func']`) is
        unserializable (i.e. if an exception is raised when trying to serialize
        the optimization result).

        Notice that if `store_objective` is set to `False`, a deep copy of the
        optimization result is created, potentially leading to performance
        problems if `res` is very large. If the objective function is not
        critical, one can delete it before calling `ProcessOptimizer.dump()`
        and thus avoid deep copying of `res`.

    * `**kwargs` [other keyword arguments]:
        All other keyword arguments will be passed to `joblib.dump`.
    """
    if store_objective:
        dump_(res, filename, **kwargs)

    elif "func" in res.specs["args"]:
        # If the user does not want to store the objective and it is indeed
        # present in the provided object, then create a deep copy of it and
        # remove the objective function before dumping it with joblib.dump.
        res_without_func = deepcopy(res)
        del res_without_func.specs["args"]["func"]
        dump_(res_without_func, filename, **kwargs)

    else:
        # If the user does not want to store the objective and it is already
        # missing in the provided object, dump it without copying.
        dump_(res, filename, **kwargs)


def load(filename, **kwargs):
    """
    Reconstruct a ProcessOptimizer optimization result from a file
    persisted with ProcessOptimizer.dump.

    Notice that the loaded optimization result can be missing
    the objective function (`.specs['args']['func']`) if
    `ProcessOptimizer.dump` was called with `store_objective=False`.

    Parameters
    ----------
    * `filename` [string or `pathlib.Path`]:
        The path of the file from which to load the optimization result.

    * `**kwargs` [other keyword arguments]:
        All other keyword arguments will be passed to `joblib.load`.

    Returns
    -------
    * `res` [`OptimizeResult`, scipy object]:
        Reconstructed OptimizeResult instance.
    """
    return load_(filename, **kwargs)


def is_listlike(x):
    return isinstance(x, (list, tuple))


def is_2Dlistlike(x):
    return np.all([is_listlike(xi) for xi in x])


def check_x_in_space(x, space):
    if is_2Dlistlike(x):
        if not np.all([p in space for p in x]):
            raise ValueError("Not all points are within the bounds of the space.")
        if any([len(p) != len(space.dimensions) for p in x]):
            raise ValueError("Not all points have the same dimensions as the space.")
    elif is_listlike(x):
        if x not in space:
            raise ValueError(
                "Point (%s) is not within the bounds of "
                "the space (%s)." % (x, space.bounds)
            )
        if len(x) != len(space.dimensions):
            raise ValueError(
                "Dimensions of point (%s) and space (%s) do not match"
                % (x, space.bounds)
            )


def expected_minimum(
    res,
    n_random_starts=20,
    random_state=None,
    return_std=False,
    minmax="min",
):
    """
    Compute the minimum over the predictions of the last surrogate model.

    Note that the returned minimum may not necessarily be an accurate
    prediction of the minimum of the true objective function.

    Parameters
    ----------
    * `res`  [`OptimizeResult`, scipy object]:
        The optimization result returned by a `ProcessOptimizer` minimizer.

    * `n_random_starts` [int, default=20]:
        The number of random starts for the minimization of the surrogate
        model.

    * 'return_std' [Boolean, default=True]:
        Whether the function should return the standard deviation or not.

    * `random_state` [int, RandomState instance, or None (default)]:
        Set random state to something other than None for reproducible
        results.

    * `minmax` [str, default='min']:
        Whether the function should return the expected minimun (intended use)
        or the expected maximum (edge use case).

    Returns
    -------
    * `x` [list]: location of the minimum (or maximum).

    * `fun` [float]: the surrogate function value at the minimum (or maximum).
    """
    if type(res) == list:
        raise ValueError("expected_minimum does not support multiobjective results")

    def func(x):
        reg = res.models[-1]
        if minmax == "min":
            return reg.predict(x.reshape(1, -1))[0]
        elif minmax == "max":
            return -1 * reg.predict(x.reshape(1, -1))[0]
        else:
            raise ValueError(
                "Expected minmax to be in ['min','max'], got %s" % (minmax)
            )

    xs = [res.x]
    if n_random_starts > 0:
        xs.extend(res.space.rvs(n_random_starts, random_state=random_state))
    xs = res.space.transform(xs)
    best_x = None
    best_fun = np.inf

    for x0 in xs:
        r = sp_minimize(func, x0=x0, bounds=res.space.transformed_bounds)
        if r.fun < best_fun:
            best_x = res.space.inverse_transform(np.array(r.x).reshape(1, -1))[0]
            best_fun = r.fun

    if minmax == "min":
        if return_std == True:
            std_estimate = res.models[-1].predict(
                res.space.transform([best_x]).reshape(1, -1), return_std=True
            )[1][0]
            return [v for v in best_x], [best_fun, std_estimate]
        else:
            return [v for v in best_x], best_fun

    elif minmax == "max":
        if return_std == True:
            std_estimate = res.models[-1].predict(
                res.space.transform([best_x]).reshape(1, -1), return_std=True
            )[1][0]
            return [v for v in best_x], [-best_fun, std_estimate]
        else:
            return [v for v in best_x], -best_fun

    else:
        raise ValueError("Expected acq_func to be in ['min','max'], got %s" % (minmax))


def expected_minimum_random_sampling(
    res,
    n_random_starts=100000,
    random_state=None,
    return_std=False,
    minmax="min",
):
    """Minimum search by doing naive random sampling, Returns the parameters
    that gave the minimum function value. Can be used when the space
    contains any categorical values.
    .. note::
        The returned minimum may not necessarily be an accurate
        prediction of the minimum of the true objective function.
    Parameters
    ----------
    res : `OptimizeResult`, scipy object
        The optimization result returned by a `skopt` minimizer.
    n_random_starts : int, default=100000
        The number of random starts for the minimization of the surrogate
        model.
    random_state : int, RandomState instance, or None (default)
        Set random state to something other than None for reproducible
        results.
    Returns
    -------
    x : list]
        location of the minimum.
    fun : float
        the surrogate function value at the minimum.
    """

    # sample points from search space
    random_samples = res.space.rvs(n_random_starts, random_state=random_state)

    # make estimations with surrogate
    model = res.models[-1]
    y_random = model.predict(res.space.transform(random_samples))
    if minmax == "min":
        index_best_objective = np.argmin(y_random)
    elif minmax == "max":
        index_best_objective = np.argmax(y_random)
    else:
        raise ValueError("Expected minmax to be in ['min','max'], got %s" % (minmax))

    extreme_x = random_samples[index_best_objective]
    if return_std == True:
        std_estimate = res.models[-1].predict(
            res.space.transform([extreme_x]).reshape(1, -1), return_std=True
        )[1][0]
        return extreme_x, [y_random[index_best_objective], std_estimate]
    else:
        return extreme_x, y_random[index_best_objective]


def dimensions_aslist(search_space):
    """Convert a dict representation of a search space into a list of
    dimensions, ordered by sorted(search_space.keys()).

    Parameters
    ----------
    search_space : dict
        Represents search space. The keys are dimension names (strings)
        and values are instances of classes that inherit from the class
        ProcessOptimizer.space.Dimension (Real, Integer or Categorical)
        Example:
            {'name1': Real(0,1), 'name2': Integer(2,4), 'name3': Real(-1,1)}

    Returns
    -------
    params_space_list: list of ProcessOptimizer.space.Dimension instances.
        Example output with example inputs:
            [Real(0,1), Integer(2,4), Real(-1,1)]
    """
    params_space_list = [search_space[k] for k in sorted(search_space.keys())]
    return params_space_list


def point_asdict(search_space, point_as_list):
    """Convert the list representation of a point from a search space
    to the dictionary representation, where keys are dimension names
    and values are corresponding to the values of dimensions in the list.

    Counterpart to parameters_aslist.

    Parameters
    ----------
    search_space : dict
        Represents search space. The keys are dimension names (strings)
        and values are instances of classes that inherit from the class
        ProcessOptimizer.space.Dimension (Real, Integer or Categorical)
        Example:
            {'name1': Real(0,1), 'name2': Integer(2,4), 'name3': Real(-1,1)}

    point_as_list : list
        list with parameter values.The order of parameters in the list
        is given by sorted(params_space.keys()).
        Example:
            [0.66, 3, -0.15]

    Returns
    -------
    params_dict: dictionary with parameter names as keys to which
        corresponding parameter values are assigned.
        Example output with inputs:
            {'name1': 0.66, 'name2': 3, 'name3': -0.15}
    """
    params_dict = {k: v for k, v in zip(sorted(search_space.keys()), point_as_list)}
    return params_dict


def point_aslist(search_space, point_as_dict):
    """Convert a dictionary representation of a point from a search space to
    the list representation. The list of values is created from the values of
    the dictionary, sorted by the names of dimensions used as keys.

    Counterpart to parameters_asdict.

    Parameters
    ----------
    search_space : dict
        Represents search space. The keys are dimension names (strings)
        and values are instances of classes that inherit from the class
        ProcessOptimizer.space.Dimension (Real, Integer or Categorical)
        Example:
            {'name1': Real(0,1), 'name2': Integer(2,4), 'name3': Real(-1,1)}

    point_as_dict : dict
        dict with parameter names as keys to which corresponding
        parameter values are assigned.
        Example:
            {'name1': 0.66, 'name2': 3, 'name3': -0.15}

    Returns
    -------
    point_as_list: list with point values.The order of
        parameters in the list is given by sorted(params_space.keys()).
        Example output with example inputs:
            [0.66, 3, -0.15]
    """
    point_as_list = [point_as_dict[k] for k in sorted(search_space.keys())]
    return point_as_list


def y_coverage(res, return_plot=False, random_state=None, horizontal=False):
    """
    A function to calculate the expected range of observable function values
    given a result instans. This can be compared with the actual observed
    range of function values.
    """
    assert len(res.func_vals) != 0, "train model before using this function"
    observed_min = res.func_vals.min()
    observed_max = res.func_vals.max()
    min_x, expected_min = expected_minimum(res, random_state=random_state)
    max_x, expected_max = expected_minimum(res, random_state=random_state, minmax="max")

    if return_plot:
        reg = res.models[-1]
        min_x = res.space.transform(
            [
                min_x,
            ]
        )
        max_x = res.space.transform(
            [
                max_x,
            ]
        )
        sampled_mins = reg.sample_y(min_x, n_samples=5000, random_state=random_state)[0]
        sampled_maxs = reg.sample_y(max_x, n_samples=5000, random_state=random_state)[0]
        extreme_min = sampled_mins.min()
        extreme_max = sampled_maxs.max()
        bins = np.linspace(extreme_min, extreme_max, 30)
        colors = ["#B8DE29FF", "#453781FF"]

        if horizontal:
            fig, ax = plt.subplots()
            ax.hist(
                [sampled_mins, sampled_maxs],
                bins,
                label=["expected min", "expected max"],
                orientation="horizontal",
                density=True,
                color=colors,
            )
            ax.set_xlabel(
                '"Plausibility" of achieving/realizing ' "given function value"
            )
            ax.set_ylabel("Function Value")
            for i in range(len(res.func_vals)):
                if i == 0:
                    ax.axhline(
                        y=res.func_vals[i],
                        xmin=0.6,
                        xmax=1,
                        color="darkorange",
                        alpha=0.5,
                        label="Observed points",
                    )
                else:
                    ax.axhline(
                        y=res.func_vals[i],
                        xmin=0.6,
                        xmax=1,
                        color="darkorange",
                        alpha=0.5,
                    )
            ax.legend(loc="best", shadow=True)
            ax.set_xticks([])
            plt.show()

        else:
            fig, ax = plt.subplots()
            ax.hist(
                [sampled_mins, sampled_maxs],
                bins,
                label=["expected min", "expected max"],
                density=True,
                color=colors,
            )
            ax.set_xlabel("Function value")
            ax.set_ylabel(
                '"Plausibility" of achieving/realizing ' "given function value"
            )
            for i in range(len(res.func_vals)):
                if i == 0:
                    ax.axvline(
                        x=res.func_vals[i],
                        ymin=0.3,
                        ymax=0.7,
                        color="darkorange",
                        alpha=0.5,
                        label="Observed points",
                    )
                else:
                    ax.axvline(
                        x=res.func_vals[i],
                        ymin=0.3,
                        ymax=0.7,
                        color="darkorange",
                        alpha=0.5,
                    )
            ax.legend(loc="best", shadow=True)
            ax.set_yticks([])
            plt.show()

    return (observed_min, observed_max), (expected_min, expected_max)
