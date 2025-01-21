from abc import ABC, abstractmethod
from typing import Iterable, Any

import numpy as np
import torch
import botorch
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import Normalize
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from ProcessOptimizer.space import Space

botorch.settings.debug(state=True)

MC_SAMPLES = 1024
BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 512


class BoTorchSuggestor(ABC):
    """
    Base suggestor class for BoTorch acquisition functions.

    Subclasses must implement the acquisition_function method with their specific parameters.
    """

    def __init__(self, space: Space, n_objectives: int, rng: np.random.Generator, **kwargs):
        self.space = space
        self.n_objectives = n_objectives
        self.rng = rng
        self.kwargs = kwargs

    @abstractmethod
    def acquisition_function(self, model: SingleTaskGP, **kwargs) -> Any:
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
        mll, model = self.initialize_model(Xi, Yi)
        fit_gpytorch_mll(mll)

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

    def initialize_model(self, Xi: Iterable[Iterable], Yi: Iterable) -> SingleTaskGP:
        # Convert Xi to torch tensor and float64
        Xi = torch.tensor(Xi, dtype=torch.float64)
        Yi = torch.tensor(Yi, dtype=torch.float64).reshape(-1, 1)
        model = SingleTaskGP(
            train_X=torch.Tensor(Xi),
            train_Y=torch.Tensor(Yi),
            likelihood=GaussianLikelihood(),
            mean_module=ZeroMean(),
            covar_module=RBFKernel()
        )
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


class BoTorch_qLogNoisyEI(BoTorchSuggestor):
    """
    Implementation of qLogNoisyExpectedImprovement acquisition function.
    """

    def acquisition_function(
            self,
            model: SingleTaskGP,
            X_baseline: torch.Tensor,
            sampler: SobolQMCNormalSampler,
            **kwargs
    ) -> qLogNoisyExpectedImprovement:
        eta = self.kwargs.get('acq_func_kwargs').get("xi")
        if eta is None:
            eta = 0.01

        return qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=X_baseline,
            sampler=sampler,
            eta=eta
        )


class BoTorch_LogEI(BoTorchSuggestor):
    """
    Implementation of LogExpectedImprovement acquisition function.
    """

    def acquisition_function(self, model: SingleTaskGP, **kwargs) -> LogExpectedImprovement:
        best_f = torch.max(model.train_targets)
        return LogExpectedImprovement(
            model=model,
            best_f=best_f,
        )
