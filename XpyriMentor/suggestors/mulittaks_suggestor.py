from abc import ABC, abstractmethod
from typing import Iterable, Any

import numpy as np
import torch

import botorch
from botorch.models import MultiTaskGP
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import Normalize

from gpytorch.likelihoods import Likelihood, _GaussianLikelihoodBase, _MultitaskGaussianLikelihoodBase, GaussianLikelihood, MultitaskGaussianLikelihood,
from gpytorch.likelihoods.noise_models import MultitaskHomoskedasticNoise
from gpytorch.kernels import RBFKernel, IndexKernel, MultitaskKernel
from gpytorch.means import ZeroMean, ConstantMean, MultitaskMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import GP
from gpytorch.module import Module
from gpytorch.distributions import base_distributions, MultivariateNormal

from XpyriMentor.suggestors.optimization_utils import fit_gpytorch_mll_wth_stopper
from ProcessOptimizer.space import Space

botorch.settings.debug(state=True)

MC_SAMPLES = 1024
BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 512

class BoTorchMTSuggestor(ABC):
    """
    Base suggestor class for BoTorch acquisition functions.
    Subclasses must implement the acquisition_function method with their specific parameters.
    """

    def __init__(self, space: Space, n_objectives: int, rng: np.random.Generator, **kwargs):
        self.space = space
        self.n_objectives = n_objectives
        self.rng = rng
        self.kwargs = kwargs

        self.tasks = 1 if self.kwargs.get('model').get('tasks') is None else self.kwargs.get('model').get('tasks')

    @abstractmethod
    def acquisition_function(self, model: GP, **kwargs) -> Any:
        """
        Create and return the acquisition function.

        Args:
            model: The fitted GP model
            **kwargs: Additional arguments that might be needed by specific acquisition functions

        Returns:
            A BoTorch acquisition function
        """
        pass

    def suggest(self, Xi: Iterable[Iterable], Yi: Iterable, n_asked: int = -1) -> np.ndarray:
        Xi = self.space.transform(Xi)
        Yi = - np.array(Yi)

        # Fit the models
        mll, model = self.initialise_model(Xi, Yi)
        fit_gpytorch_mll_wth_stopper(mll)

        # Create acquisition function with all necessary parameters
        acq_func = self.acquisition_function(
            model=model,
            X_baseline=torch.tensor(Xi, dtype=torch.float64),
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        )

        # optimize and get new candidate
        candidates = self.optimize_acqf_and_get_observation(acq_func).numpy()
        candidates = self.space.inverse_transform(candidates)
        return candidates
    def initialise_model(
            self,
            Xi: Iterable[Iterable],
            Yi: Iterable)

        train_x = torch.tensor(Xi, dtype=torch.float64)
        train_y = torch.tensor(Yi, dtype=torch.float64).reshape(-1, 1)

        # define likelihood
        likelihood = self.kwargs.get('model').get('likelihood')
        # define kernel parameters
        mean_module = self.kwargs.get('model').get('mean_module')
        covar_module = self.kwargs.get('model').get('covar_module')
        task_module = self.kwargs.get('model').get('task_module')

        if self.tasks == 1:
            model = SingleTaskGP(
                train_X=train_x,
                train_Y=train_y,
                likelihood=likelihood,
                mean_module=mean_module,
                covar_module=covar_module
            )
        else:
            if self.kwargs.get('model').get('type') == 'multioutput':
                if isinstance(likelihood, MultitaskGaussianLikelihood):
                    raise ValueError('Only MultitaskGaussianLikelihood is supported with this type of model.' +
                                     'Use 'likelihood: None' in suggestor factory')
                model = KroneckerMultiTaskGP(
                    train_X=train_x,
                    train_Y=train_y,
                    data_covar_module = covar_module,
                    )

            elif self.kwargs.get('model').get('type') == 'multitask':
                train_indices = torch.arange(0, self.tasks).expand(train_y.shape)
                train_indices = train_indices.T.flatten()
                train_x = torch.cat([train_x, train_indices], -1)
                train_y = train_y.T.flatten()

                if self.kwargs.get('model').get('likelihood') == 'hadamard':
                    likelihood = HadamardGaussianLikelihood()
                model = MultiTaskGP(
                    train_X = train_x,
                    train_Y = train_y,
                    task_feature = self.kwargs.get('model').get('task_feature'),
                    likelihood = likelihood,
                    mean_module = mean_module,
                    covar_module = covar_module,
                    )
            else:
                raise ValueError('Unsupported model type.')


        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def optimize_acqf_and_get_observation(self, acq_func):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation.
        Note: assumes that the input space has been normalized to [0, 1]^d.
        """

        n_dim = self.space.n_dims
        bounds = torch.tensor([[0.0] * n_dim, [1.0] * n_dim])
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for initialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )
        return candidates.detach()

class MTMeanGPModel(gpytorch.models.ExactGP):
    " multi-output regressor --> corresponds to the KroneckerMultiTasksGP in Botorch
    def __init__(self, train_x, train_y, likelihood, num_tasks_):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks_
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks_, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

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

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)


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