from typing import Iterable, Any

import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import torch
import warnings
# from botorch.exceptions import OptimizationWarning as MasterOptimizationWarning

from botorch.models import MultiTaskGP
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.acquisition import qLogExpectedImprovement

from gpytorch.likelihoods import _GaussianLikelihoodBase, GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.likelihoods.noise_models import MultitaskHomoskedasticNoise
from gpytorch.kernels import RBFKernel, IndexKernel, MultitaskKernel, ScaleKernel
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import GP, ExactGP
from gpytorch.distributions import base_distributions, MultivariateNormal, MultitaskMultivariateNormal

# Adding this temporarily to see whether acquisition for TLBO is defined and optimized properly by default IdentityMCObjective.
# ~~~~~~ Seems we need GenericMCObjective to only optimize the acquisition function wrt to the test objective ~~~~~~
from botorch.acquisition.objective import (
    GenericMCObjective
)

from ProcessOptimizer.space import Space

#botorch.settings.debug(state=True)

MC_SAMPLES = 1024
BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 512

class MTSuggestor():
    def __init__(self, space: Space, n_objectives: int, rng: np.random.Generator, **kwargs):
        self.space = space
        self.n_objectives = n_objectives
        self.rng = rng
        self.kwargs = kwargs

        # Beginning of defaults
        if self.kwargs.get('surrogate_kwargs') is None:
            self.kwargs['surrogate_kwargs'] = {}

        # If task feature is not specified by the user
        if self.kwargs.get('surrogate_kwargs').get('task_feature') is None:

            # Defaults to a single-task problem if task feature is not clear
            if not self.space.is_partly_task:
                
                self.tasks = 1
                self.kwargs['surrogate_kwargs'] = {}
                self.kwargs['surrogate_kwargs']['type'] = None
                self.kwargs['surrogate_kwargs']['likelihood'] = None
                # define kernel parameters
                self.kwargs['surrogate_kwargs']['mean_module'] = None
                self.kwargs['surrogate_kwargs']['covar_module'] = None
                self.kwargs['surrogate_kwargs']['task_module'] = None

            # Finds task feature
            elif self.space.num_task_dims == 1:
                self.tasks = self.space.num_tasks
                self.kwargs['surrogate_kwargs']['task_feature'] = self.space.task_feature
                self.kwargs['surrogate_kwargs']['type'] = 'multitask'

            # If multiple task features raise error 
            elif self.space.num_task_dims > 1: 
                raise NotImplementedError("More than one task feature, e.g., '[0, 1, {1}]', provided in self.space. Please combine tasks into a single list as MTSuggestor does not currently support multiple task features.")

        task_feature = self.kwargs.get('surrogate_kwargs').get('task_feature')

        # Decide between multioutput, or single-output multitask optimization
        if self.kwargs.get('surrogate_kwargs').get('type') is None:
            if self.space.dimensions[task_feature].active_task is None:
                self.kwargs['surrogate_kwargs']['type'] = 'multioutput'
            else: 
                self.kwargs['surrogate_kwargs']['type'] = 'multitask'

        optimization_type = self.kwargs.get('surrogate_kwargs').get('type')
        
        if optimization_type == 'multitask':

            if self.kwargs.get('surrogate_kwargs').get('test_task') is None:
                self.kwargs['surrogate_kwargs']['test_task'] = self.space.active_task

            if self.kwargs.get('surrogate_kwargs').get('mean_module') is None: 
                self.kwargs['surrogate_kwargs']['mean_module'] = ConstantMean()

            if self.kwargs.get('surrogate_kwargs').get('covar_module') is None:

                non_task_dims = self.space.n_dims - self.space.num_task_dims

                non_task_indices = [i for i in range(self.space.n_dims)]
                non_task_indices.remove(task_feature)

                non_task_bound_widths = [self.space.bounds[i][1] - self.space.bounds[i][0] for i in non_task_indices]
                
                base_kernel = RBFKernel(ard_num_dims=non_task_dims)
                
                # List/array of length scales relative to size of boundaries
                if self.kwargs.get('surrogate_kwargs').get('covar_relative_lengthscale') is None:
                    self.kwargs['surrogate_kwargs']['covar_relative_lengthscale'] = 0.2

                rel_lengthscale = self.kwargs.get('surrogate_kwargs').get('covar_relative_lengthscale')
                
                base_kernel.lengthscale = torch.tensor([[width*rel_lengthscale for width in non_task_bound_widths]])

                self.kwargs['surrogate_kwargs']['covar_module'] = ScaleKernel(base_kernel)                

            if self.kwargs.get('surrogate_kwargs').get('likelihood') is None: 

                self.kwargs['surrogate_kwargs']['likelihood'] = GaussianLikelihood()

        if self.kwargs.get('acq_func_kwargs') is None:
            self.kwargs['acq_func_kwargs'] = {}
        
        if self.kwargs.get('acq_func_kwargs').get('xi') is None:
            self.kwargs['acq_func_kwargs']['xi'] = 0.01

        if self.kwargs.get('acq_func_kwargs').get('acq') is None:
            self.kwargs['acq_func_kwargs']['acq'] = 'qLogExpectedImprovement'


    
    def acquisition_function(self, model: GP,
                sampler: SobolQMCNormalSampler) -> Any:
        acq_constraints = self.kwargs.get('acq_func_kwargs').get('acquisition_constraints')
        eta = self.kwargs.get('acq_func_kwargs').get('constraint_approx_temperature')
        if eta is None:
            eta = 0.01

        acq = self.kwargs.get('acq_func_kwargs').get('acq')
        task_feature = self.kwargs.get('surrogate_kwargs').get('task_feature')

        # Fixing the checker for qLogExpectedImprovement (and setting this to default)
        if (acq == 'qLogExpectedImprovement' or acq is None):
            acq_func = qLogExpectedImprovement(
                model=model,
                best_f=self.best_f,
                sampler=sampler,
                constraints=acq_constraints,
                eta=eta,
                # For now we're going to have to assume tasks are 0 or 1, with the active task being labelled 1. 
                objective=GenericMCObjective(lambda samples, **kwargs: samples[..., 1])
            )
            return acq_func
        elif (acq == 'qExpectedImprovement'):
            acq_func = qExpectedImprovement(
                model=model,
                best_f=self.best_f,
                sampler=sampler,
                constraints=acq_constraints,
                # eta=eta,
                # For now we're going to have to assume tasks are 0 or 1, with the active task being labelled 1. 
                objective=GenericMCObjective(lambda samples, **kwargs: samples[..., 1])
            )
            return acq_func
        else:
            raise NotImplementedError(f"The acquisition function '{acq}' is not currently supported by MTSuggestor.")

    def suggest(self, Xi: Iterable[Iterable], Yi: Iterable, n_asked: int = 1) -> np.ndarray:

        # Fit the models
        mll, model = self.initialize_model(Xi, Yi)

        with warnings.catch_warnings():
            
            # TODO: add a kwarg to decide whether to suppress warnings (and/or optimization errors?) in optimize_acqf. 
            
            # warnings.filterwarnings("ignore", category=OptimizationWarning)

            warnings.filterwarnings("ignore")
        
            fit_gpytorch_mll(mll, max_retries=30)

        task_feature = self.kwargs.get('surrogate_kwargs').get('task_feature')
        test_task = self.kwargs.get('surrogate_kwargs').get('test_task')
 
        self.active_task_indices = np.where(np.stack(Xi)[:, task_feature] == test_task)[0]

        if not len(self.active_task_indices) == 0:
            self.best_f = max(np.stack(Yi)[self.active_task_indices])
        else:
            raise ValueError('Suggestion cannot be made as no data from the active task has been provided.')

        # Create acquisition function with all necessary parameters
        acq_func = self.acquisition_function(
            model=model,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        )
        
        # optimize and get new candidate
        candidates = self.optimize_acqf_and_get_observation(acq_func, n_asked).numpy()

        return candidates
        
    def initialize_model(
            self,
            Xi: Iterable[Iterable],
            Yi: Iterable):

        # define likelihood
        likelihood = self.kwargs.get('surrogate_kwargs').get('likelihood')
        # define kernel parameters
        mean_module = self.kwargs.get('surrogate_kwargs').get('mean_module')
        covar_module = self.kwargs.get('surrogate_kwargs').get('covar_module')
        
        train_x = torch.tensor(np.stack(Xi))
        train_y = torch.tensor(Yi).unsqueeze(-1)

        # edge case for single task
        if self.tasks == 1:            
            model = SingleTaskGP(
                train_X=train_x,
                train_Y=train_y,
                likelihood=likelihood,
                mean_module=mean_module,
                covar_module=covar_module
            )

        else:
            if self.kwargs.get('surrogate_kwargs').get('type') == 'multioutput':
                if not isinstance(likelihood, MultitaskGaussianLikelihood):
                    raise ValueError("Only MultitaskGaussianLikelihood is supported with this type of model. Use 'likelihood: None' in suggestor factory")

                if self.kwargs.get('surrogate_kwargs').get('likelihood') == 'hadamard':
                    likelihood = HadamardGaussianLikelihood()

                    
                model = KroneckerMultiTaskGP(
                    train_X=train_x,
                    train_Y=train_y,
                    data_covar_module = covar_module,
                    )

            elif self.kwargs.get('surrogate_kwargs').get('type') == 'multitask':
                    
                model = MultiTaskGP(
                    train_X = train_x,
                    train_Y = train_y,
                    task_feature = self.kwargs.get('surrogate_kwargs').get('task_feature'),
                    likelihood = likelihood,
                    mean_module = mean_module,
                    covar_module = covar_module)
                
            else:
                raise ValueError('Unsupported surrogate model type.')

        with warnings.catch_warnings():
            
            # TODO: add a kwarg to decide whether to suppress warnings (and/or optimization errors?) in optimize_acqf. 
            
            # warnings.filterwarnings("ignore", category=OptimizationWarning)

            warnings.filterwarnings("ignore")

            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
        return mll, model

    def optimize_acqf_and_get_observation(self, acq_func, n_asked: int = 1, num_restarts: int = 10):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation.
        """

        # For multitask examples - bounds need to not include task parameter for BoTorch. 
        # Notice we've assumed task has a scalar label (there are not more than one task dimensions)
        if self.space.is_partly_task == True:
            n_dim = self.space.n_dims - self.space.num_task_dims
        else:
            n_dim = self.space.n_dims

        non_task_indices = [i for i in range(self.space.n_dims)]
        non_task_indices.remove(self.kwargs.get('surrogate_kwargs').get('task_feature'))

        lower_bounds = [self.space.bounds[i][0] for i in non_task_indices]
        upper_bounds = [self.space.bounds[i][1] for i in non_task_indices]
        
        bounds = torch.tensor([lower_bounds, upper_bounds])

        with warnings.catch_warnings():
            
            # TODO: add a kwarg to decide whether to suppress warnings (and/or optimization errors?) in optimize_acqf. 
            
            # warnings.filterwarnings("ignore", category=OptimizationWarning)

            warnings.filterwarnings("ignore")
            
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=n_asked,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES
                # options={},
            )   
        
        return candidates.detach()

    def get_best_f(self, Xi: Iterable[Iterable], Yi: Iterable):

        task_feature = self.kwargs.get('surrogate_kwargs').get('task_feature')
        test_task = self.kwargs.get('surrogate_kwargs').get('test_task')
        tasks = self.space.all_tasks
        
        self.active_task_indices = np.where(np.stack(Xi)[:, task_feature] == test_task)[0]

        if not len(self.active_task_indices) == 0:
            self.best_f = max(np.stack(Yi)[self.active_task_indices])
        else:
            self.best_f = np.nan

        return self.best_f

    def preprocess_tell(self, x: Any, y: Any, task: Any):

        task_feature = self.kwargs['surrogate_kwargs']['task_feature']
        all_tasks_in_space = self.space.all_tasks

        # Preprocessing converts x to a 2D numpy array.
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        elif isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        else:
            x = np.array(x)

        nrows_x = x.shape[0]
        ncols_x = x.shape[1]

        nrows_y = len(y)

        if not nrows_x == nrows_y:
            raise ValueError(f"x has {nrows_x} entries while y has {nrows_y}.")

        if ncols_x == self.space.n_dims:
            
            x_tasks = np.unique(x[:, task_feature])
            is_x_tasks_unique = len(x_tasks) == 1
            all_tasks_valid = np.all(np.isin(x_tasks, all_tasks_in_space))
            
            if not all_tasks_valid:
                raise ValueError(f"Task feature entries in column {task_feature} of x are not valid tasks, i.e., not in {all_tasks_in_space}.")
                
            if not task is None: 
                all_tasks_equal = np.all(x[:, task_feature] == task)
                if not all_tasks_equal:
                    if is_x_tasks_unique:
                        warnings.warn(f"Task feature value is different in column {task_feature} of x and 'task' argument. Defaulting to value in x.")
                    else:
                        warnings.warn(f"Task features in column {task_feature} of x have different values between entries. Ignoring 'task' argument")

        elif ncols_x == self.space.n_dims - self.space.num_task_dims:
            if task is None: 
                raise ValueError("No task specified.")
            else: 
                if not np.isin(task, all_tasks_in_space):
                    raise ValueError(f"task argument {task} not a valid task in space, i.e., not in {all_tasks_in_space}.")
                
                task_array = np.full((nrows_x, 1), task)
                
                x_cols_before_task = x[:, :task_feature]
                x_cols_after_task = x[:, task_feature:]
                
                x = np.hstack((x_cols_before_task, task_array, x_cols_after_task))
        
        return x, y
        

    def plot_objective(
        self,
        Xi,
        Yi,
        levels=10,
        size=2,
        grid_resolution=101,
        show_confidence=True,
        z_scale='linear',
        title=None,
        plot_options=None,
        **kwargs
    ):
        """Pairwise dependence plot for test task in TLBO. 
        Currently supporting real dimensions only. 
        Currently performing a grid search over posterior---very inefficient for high dimensional tasks.
        """
        if Xi is None: 
            warnings.warn("No data told to the XpyriMentor, plotting the prior if this is supported.")
        
        mml, model = self.initialize_model(Xi=Xi, Yi=-1*np.array(Yi))
        
        model.eval()
        
        task_feature = self.kwargs.get('surrogate_kwargs').get('task_feature')
        test_task = self.kwargs.get('surrogate_kwargs').get('test_task')
        
        non_task_indices = [i for i in range(self.space.n_dims)]
        non_task_indices.remove(task_feature)
        
        lower_bounds = [self.space.bounds[i][0] for i in non_task_indices]
        upper_bounds = [self.space.bounds[i][1] for i in non_task_indices]
        
        X_linspaces = np.array([np.linspace(lower_bounds[i], upper_bounds[i], grid_resolution, axis=0) for i in non_task_indices])
        X_grid = np.meshgrid(*X_linspaces, indexing='ij')
        X_grid = np.stack(X_grid, axis=-1).reshape(-1, len(X_linspaces))
        
        X_grid_with_task = np.insert(X_grid, task_feature, np.ones((grid_resolution**len(non_task_indices), ))*test_task, axis=1)
        
        X_tensor = torch.tensor(X_grid_with_task)

        # Calculating whole posterior scales poorly with dimension of feature space. Could improve this in future.
        with torch.no_grad():
            mean = model.posterior(X_tensor).mean
        idx = mean.argmin()

        # Expected minimum according to posterior
        x_star = X_tensor[idx]
        
        x_star_without_task = X_tensor[idx][non_task_indices]
        
        dims = len(non_task_indices)

        if dims == 2:
            size=size*1.75
        
        fig, axes = plt.subplots(dims, dims, figsize=(size*dims, size*dims))
        
        for i in non_task_indices:
            for j in non_task_indices:
                ax = axes[i, j]
                ax.clear()
        
                if i == j:
                    # Diagonal: 1D marginal
                    X_grid_ij = np.tile(x_star, (grid_resolution, 1))
                    
                    X_grid_ij[:, i] = X_linspaces[i]
        
                    X_tensor_ij = torch.tensor(X_grid_ij)
                    with torch.no_grad():
                        posterior = model.posterior(X_tensor_ij)
                        mean_ij = posterior.mean.numpy().squeeze()
                        std_ij = np.sqrt(posterior.variance.numpy().squeeze()) if hasattr(posterior, 'variance') else np.zeros_like(mean_ij)
        
                    ax.plot(X_linspaces[i], mean_ij, 'darkgreen')

                    ax.set_xlim(0.0, 15.0)

                    ax.axvline(x_star[i], linestyle="--", color="k", lw=1)
                    
                    ax.fill_between(X_linspaces[i], mean_ij - 1.96*std_ij, mean_ij + 1.96*std_ij,                                               alpha=0.5, color="green", edgecolor="green")
                    ax.set_title(f'Feature dimension {i}')
                    ax.set_xlabel(f'x_{i}')
                    ax.set_ylabel('Posterior 95% credible interval')
        
                elif i > j:
                    # Lower triangle: 2D contour
                    grid_points = []
                    
                    for xi in X_linspaces[i]:
                        for xj in X_linspaces[j]:
                            point = torch.Tensor(np.copy(x_star))
                            point[i] = xi
                            point[j] = xj
                            grid_points.append(point)
                    
                    X_grid_ij = np.array(grid_points)
        
                    X_tensor_ij = torch.tensor(X_grid_ij)
                    with torch.no_grad():
                        posterior = model.posterior(X_tensor_ij)
                        mean = posterior.mean.numpy().squeeze()
        
                    Z = mean.reshape(grid_resolution, grid_resolution)
                    cs = ax.contourf(
                        X_linspaces[j],
                        X_linspaces[i],
                        Z, 
                        levels = levels,
                        cmap='viridis_r')  # note swapped axes for correct orientation
                    fig.colorbar(cs, ax=axes[0][-1], orientation="horizontal", label=f"Expected value plot ({i}, {j})")
                    ax.set_xlabel(f"x_{j}")
                    ax.set_ylabel(f"x_{i}")

                    test_samples_i = [sample[i] for sample in Xi if sample[task_feature] == test_task]
                    test_samples_j = [sample[j] for sample in Xi if sample[task_feature] == test_task]

                    training_samples_i = [sample[i] for sample in Xi if sample[task_feature] != test_task]
                    training_samples_j = [sample[j] for sample in Xi if sample[task_feature] != test_task]

                    # Test samples added to contour plot
                    ax.scatter(
                    test_samples_j,
                    test_samples_i,
                    c="darkorange",
                    s=20,
                    lw=0.0,
                    zorder=10,
                    clip_on=False,
                    )

                    # Training samples added to contour plot
                    ax.scatter(
                    training_samples_j,
                    training_samples_i,
                    c="lightgrey",
                    s=20,
                    lw=0.0,
                    zorder=9,
                    clip_on=False,
                    )

                    # Expected minimum label
                    ax.scatter(
                    x_star[j],
                    x_star[i],
                    c="k",
                    s=30,
                    marker="D",
                    lw=0.0,
                    zorder=8,
                    clip_on=False,
                    )
        
                else:
                    # Upper triangle: hide axis
                    ax.axis('off')
        
                ax.label_outer()

        # For legend display for test data
        legend_test_points = mpl.lines.Line2D([], [], color="darkorange", marker=".", markersize=9, lw=0.0)
        # Training data
        legend_training_points = mpl.lines.Line2D([], [], color="lightgrey", marker=".", markersize=9, lw=0.0)
        # Expected minimum x_star in 2D plots
        legend_hp = mpl.lines.Line2D([], [], color="k", marker="D", markersize=5, lw=0.0)
        # Expected minimum x_star in 1D plots
        legend_hl = mpl.lines.Line2D([], [], linestyle="--", color="k", marker="", lw=1)
        # Confidence set plots
        legend_ci = mpl.patches.Patch(color="green", alpha=0.5)

        axes[0][-1].legend(
            handles=[legend_test_points, legend_training_points, (legend_hp, legend_hl), legend_ci],
            labels=["Test task data", "Training task data", "Expected minimum", "95% credibility interval"],
            loc="upper center",
            handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)},
        )
        
        plt.tight_layout()
        plt.show()
            

class MTMeanGPModel(ExactGP):
    """multi-output regressor --> corresponds to the KroneckerMultiTasksGP in Botorch"""

    def __init__(self, train_x, train_y, likelihood, num_tasks_):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = MultitaskMean(
            ConstantMean(), num_tasks=num_tasks_
        )
        self.covar_module = MultitaskKernel(
            RBFKernel(), num_tasks=num_tasks_, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)

class MTGPModel(ExactGP):
    "corresponds to the MultiTaskGP object in Botorch"
    def __init__(self, train_x, train_y, likelihood, num_tasks_):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = RBFKernel()
        self.task_covar_module = IndexKernel(
            num_tasks=num_tasks_, rank=1)

    def forward(self,x, task):
        mean_x = self.mean_module(x)

        # Feature covariance
        covar_x = self.covar_module(x)
        # Task covariance
        covar_task = self.task_covar_module(task)
        covar = covar_x.mul(covar_task)

        return MultivariateNormal(mean_x, covar)


class HadamardGaussianLikelihood(_GaussianLikelihoodBase):
  # see the feature request and related discussion: https://github.com/cornellius-gp/gpytorch/pull/2481
    """
    Likelihood for input-wise homo-skedastic noise, and task-wise
    hetero-skedastic, i.e. we learn a different (constant) noise level for each fidelity.

    Args:
        num_of_tasks : number of tasks in the multi output GP
        noise_prior : any prior you want to put on the noise
        noise_constraint : constraint to put on the noise
    """
    def __init__(
        self,
        num_tasks,
        noise_prior=None,
        noise_constraint=None,
        batch_shape=torch.Size(),
        **kwargs,
    ):
        noise_covar = MultitaskHomoskedasticNoise(
            num_tasks=num_tasks,
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
            batch_shape=batch_shape,
        )
        self.num_tasks = num_tasks
        super().__init__(noise_covar=noise_covar)

    @property
    def noise(self) -> torch.Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: torch.Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> torch.Tensor:
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: torch.Tensor) -> None:
        self.noise_covar.initialize(raw_noise=value)

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
        # params contains training data
        task_idxs = params[0][-1]
        noise_base_covar_matrix = self.noise_covar(*params, shape=base_shape, **kwargs)
        # initialize masking
        mask = torch.zeros(size=noise_base_covar_matrix.shape)
        # for each task create a masking
        for task_num in range(self.num_tasks):
            # create vector of indexes
            task_idx_diag = (task_idxs == task_num).int().reshape(-1).diag()
            mask[..., task_num, :, :] = task_idx_diag
        # multiply covar by masking
        # there seems to be problems when base_shape is singleton, so we need to squeeze
        if base_shape == torch.Size([1]):
            noise_base_covar_matrix = noise_base_covar_matrix.squeeze(-1).mul(mask.squeeze(-1))
            noise_covar_matrix = noise_base_covar_matrix.unsqueeze(-1).sum(dim=1)
        else:
            noise_covar_matrix = noise_base_covar_matrix.mul(mask).sum(dim=1)
        return noise_covar_matrix

    def forward(
        self,
        function_samples: torch.Tensor,
        *params: Any,
        **kwargs: Any,
    ) -> base_distributions.Normal:
        noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diag()
        return base_distributions.Normal(function_samples, noise.sqrt())

    def marginal(self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> MultivariateNormal:
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs).squeeze(0)
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar)