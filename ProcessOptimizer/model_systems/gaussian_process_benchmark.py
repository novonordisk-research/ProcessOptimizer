import torch
import numpy as np

from typing import List, Optional, Union, Dict
from .model_system import ModelSystem
from ..space import Real


class GPExperiment:
    """Base Class for 1D Gaussian Process experiment."""

    def __init__(
            self,
            lower_bound: Optional[float],
            upper_bound: Optional[float],
            tasks: Optional[int] = 1,
            signal_variance: Optional[float] = 1.5,
            lengthscale: Optional[float] = 0.75,
            seed: Optional[int] = 42,
    ):
        """
        Parameters
        ----------
        *  'lower_bound' [Optional[float]]:
            lower bound of 1D domain of Gaussian Process.
        * 'upper_bound' [Optional[float]]:
            upper bound of 1D domain of Gaussian Process.
        * 'tasks' [Optional[int]]:
            number of tasks generated from GP
        * 'seed' [Optional[int], default=42]:
            seed for numpy random number generator
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.tasks = tasks
        self.signal_variance = signal_variance
        self.lengthscale = lengthscale
        self._rng = np.random.RandomState(seed)
        self.experiment = self._initiate_experiment()

    def _get_domain(self, tolist: bool = True) -> Union[torch.Tensor, List[Real]]:
        """Generates the domain of the Gaussian Process
        Parameters
        ----------
        * `gridsize` [int]:
            Number of grid points. If None, use 'grid' value from the constructor.
        * `tolist` [bool]:
            Whether to return the domain as a list. True by default to ensure compatibility with ProcessOptimizer.

        Returns
        -------
        * `domain` Union[torch.Tensor, List[Real]]:
            The domain of the Gaussian Process used by GP score.

        """
        delta, delta_grid_size = 0.01, 0.001
        basic_grid = torch.arange(self.lower_bound, self.upper_bound + delta, delta)
        gridsize = delta_grid_size / np.abs(self.lower_bound - self.upper_bound)
        grid_domain = torch.linspace(self.lower_bound, self.upper_bound, int(gridsize))

        domain_join = torch.cat((basic_grid, grid_domain))
        domain, _ = torch.sort(domain_join)
        return domain.tolist() if tolist else domain

    def _get_exp_kernel(self, signal_variance, lengthscale):
        "RBF kernel for Gaussian Process."
        return lambda a, b: signal_variance * (
                -0.5 * (a.reshape(-1, 1) - b.reshape(1, -1)).pow(2) / (lengthscale ** 2)).exp()

    def _sample_from_gp(self, domain, kernel) -> torch.Tensor:
        """
         Sample trajectories of the Gaussian Process.
         The total number of trajectories is equal to number of tasks.
        """
        K = kernel(domain, domain)
        # sample from N(0, K):
        # L: LL' = K
        # f = L @ z, z ~ N(0, I)
        d, V = torch.linalg.eigh(K)  # computes the eigenvalue decomposition, d - eig values, V - eigen vectors
        L = d.clamp_min(0.0).sqrt() * V
        f = L @ torch.randn(L.shape[0], self.tasks)
        return f.squeeze()

    def _initiate_experiment(self) -> Dict:
        domain = self._get_domain(False)
        task_values = self._sample_from_gp(domain, kernel=self._get_exp_kernel(self.signal_variance, self.lengthscale))
        experiment = {'domain': domain, 'f_values': task_values}

        return experiment

    def get_experiment(self):
        return [self.experiment['domain'].tolist(),
                self.experiment['f_values'].tolist()]

    @property
    def domain(self):
        return self.experiment['domain'].tolist()

    @property
    def f_values(self):
        return self.experiment['f_values'].tolist()

    def score(self, x: Optional[float]) -> float:
        toll = 2.0e-05
        closest_candidate = torch.min(torch.abs(self.experiment['domain'] - x))
        if closest_candidate <= toll:
            idx = torch.argmin(torch.abs(self.experiment['domain'] - x))
            return self.experiment['f_values'][idx].tolist()
        else:
            l, u = self._find_closest_pair_idx(x)
            weight = (x - self.experiment['domain'][l]) / (self.experiment['domain'][u] - self.experiment['domain'][l])
            score = torch.lerp(self.experiment['f_values'][l], self.experiment['f_values'][u], weight)
            return score.tolist()

    def _find_closest_pair_idx(self, x) -> Tuple[int, int]:
        closest_candidate_idx = torch.argmin(torch.abs(self.experiment['domain'] - x))
        closest_candidate = self.experiment['domain'][closest_candidate_idx]

        is_smaller = closest_candidate < x
        is_lower_boundary = closest_candidate == self.lower_bound

        if is_lower_boundary or is_smaller:
            return (closest_candidate_idx, closest_candidate_idx + 1)
        else:
            return (closest_candidate_idx - 1, closest_candidate_idx)

    def get_sample(self, size, noise_scale=0.1):
        indices = np.random.choice(range(self.experiment['domain'].shape[0]), size, replace=False)
        x = torch.Tensor(self.experiment['domain'][indices])
        y = torch.Tensor(self.experiment['f_values'][indices, :]) + noise_scale * torch.randn(
            self.experiment['f_values'][indices, :].size())  # sigma * N(0, 1) + m = N(m, sigma^2)
        return indices, x, y

    @property
    def max(self):
        if self.tasks == 1:
            return torch.max(self.experiment['f_values']).item()
        else:
            return torch.max(self.experiment['f_values'].T, 1)[0].tolist()

    @property
    def min(self):
        if self.tasks == 1:
            return torch.min(self.experiment['f_values']).item()
        else:
            return torch.min(self.experiment['f_values'].T, 1)[0].tolist()


def create_gp_experiment(
        lower_bound: Optional[float],
        upper_bound: Optional[float],
        tasks: Optional[int] = 1,
        signal_variance: Optional[float] = 1.5,
        lengthscale: Optional[float] = 0.75,
        grid: Optional[int] = 200, noise: bool = True) -> ModelSystem:
    gp_experiment = GPExperiment(lower_bound, upper_bound, tasks, signal_variance, lengthscale)
    return ModelSystem(
        gp_experiment.score,
        [Real(lower_bound, upper_bound, name='domain')],
        noise_model=None,
        true_max=gp_experiment.max,
        true_min=gp_experiment.min)