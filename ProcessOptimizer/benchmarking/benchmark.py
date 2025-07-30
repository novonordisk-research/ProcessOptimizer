from __future__ import annotations
import functools
from dataclasses import dataclass, field
from typing import Iterable

from ..model_systems import get_model_system, ModelSystem
from ProcessOptimizer import Optimizer, XpyriMentor
from ProcessOptimizer.utils import expected_minimum


@dataclass
class BenchmarkInstance:
    model_system_name: str
    xpyrimentor_definition: dict
    experimental_budget: int
    expected_random_runtime: float
    seed: int
    noise_level: float
    validate: bool = False
    # Results:
    estimated_optima: list[tuple[float, float, float]] = field(init=False, repr=False)
    number_of_evaluations: int | None = None
    success: bool | None = None
    # Internal variables:
    success_level: float = field(init=False, repr=False)
    model: ModelSystem = field(init=False, repr=False)
    xpyrimentor: XpyriMentor = field(init=False, repr=False)

    def __init__(
        self,
        model_system_name: str,
        xpyrimentor_definition: dict,
        experimental_budget: int,
        expected_random_runtime: float,
        seed: int,
        noise_level: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the benchmark instance.

        Needs the following parameters:
        * `model_system_name` [str]:
            Name of the model system to use.
        * `xpyrimentor_definition` [dict]:
            Definition of the XpyriMentor object to use.
        * `experimental_budget` [int]:
            Maximum number of evaluations to run before stopping.
        * `expected_random_runtime` [float]:
            How "hard" the system is to optimize. This is the expected number of random
            parameter sets you need to evaluate to find a good one.
        * `seed` [int]:
            Random seed to use.
        """
        self.__dict__.update(
            {
                "model_system_name": model_system_name,
                "xpyrimentor_definition": xpyrimentor_definition,
                "experimental_budget": experimental_budget,
                "expected_random_runtime": expected_random_runtime,
                "seed": seed,
                "noise_level": noise_level,
            }
        )
        self.__dict__.update(kwargs)
        self.success_level = find_limits(
            model_system_name, expected_random_runtime, self.noise_level
        )
        self.model = get_model_system(model_system_name, seed=seed)
        self.model.noise_size *= noise_level
        self.xpyrimentor = XpyriMentor(
            self.model.space, self.xpyrimentor_definition, seed=seed
        )
        self.estimated_optima = []

    @property
    def model_system(self) -> ModelSystem:
        model_system = get_model_system(self.model_system_name, seed=self.seed)
        model_system.noise_size = model_system.noise_size * self.noise_level
        return model_system

    @property
    def optimizer(self) -> Optimizer:
        # This is a bit of a hack, but it works for now. The optimizer is the last
        # suggestor in the suggestor list of the sequential strategizer.
        return self.xpyrimentor.suggestor.suggestors[-1][1].optimizer

    def tell(self, x: Iterable, y: float) -> tuple[float, float, float]:
        """
        Tell the xpyrimentor about a new observation.

        Parameters:
        -----------
        x : Iterable
            The input parameters for the observation.
        y : float
            The output value for the observation.

        Returns:
        --------
        A tuple containing the estimated location, value, and standard deviation of the
        estimated optimum after the new point has been added. This is also appended to
        `self.estimated_optima`, so a full history of the estimated optima is
        maintained.
        """
        self.xpyrimentor.tell(x, [y])
        optimizer: Optimizer = self.optimizer
        optimizer.Xi = self.xpyrimentor.Xi
        optimizer.yi = self.xpyrimentor.yi
        optimizer.update_next()
        optimizer.add_observational_noise()
        result = optimizer.get_result()
        result_location, [result_value, result_std] = expected_minimum(
            result, return_std=True
        )
        optimizer.remove_observational_noise()
        self.estimated_optima.append((result_location, result_value, result_std))
        return (result_location, result_value, result_std)

    def run(self):
        self.success = False
        while len(self.xpyrimentor.Xi) < self.experimental_budget:
            x = self.xpyrimentor.ask()
            y = self.model.get_score(x)
            minimum_location, minimum_value, minimum_std = self.tell(x, y)
            if (minimum_value + 2 * minimum_std) < self.success_level and self.validate:
                result = self.model.get_score(minimum_location)
                minimum_location, minimum_value, minimum_std = self.tell(
                    [minimum_location], result
                )
            if (minimum_value + 2 * minimum_std) < self.success_level:
                true_quality = find_pesimistic_value(self.model, minimum_location)
                if true_quality < self.success_level:
                    self.success = True
                break
        self.number_of_evaluations = len(self.xpyrimentor.Xi)

    def report(self) -> dict[str, Any]:
        """
        Report the results of the benchmark instance.

        Returns:
        -------
        A dictionary with the results of the benchmark instance.
        """
        # Converting sampled points to a list of list of floats or strings. Numpy arrays are not directly serializable.
        x = [
            [point if isinstance(point, str) else float(point) for point in x.tolist()]
            for x in self.xpyrimentor.Xi
        ]
        return {
            "model_system_name": self.model_system_name,
            "expected_random_runtime": self.expected_random_runtime,
            "experimental_budget": self.experimental_budget,
            "noise_level": self.noise_level,
            "n_initial_points": instance.xpyrimentor.suggestor.suggestors[0][0],
            "n_replicates": instance.xpyrimentor.suggestor.suggestors[1][0],
            "acq_func_kwargs": instance.optimizer.acq_func_kwargs,
            "noise_level_bounds": self.optimizer.base_estimator_.noise_level_bounds,
            "length_scale_bounds": self.optimizer.base_estimator_.kernel.get_params()[
                "k2__length_scale_bounds"
            ],
            "xpyrimentor_definition": self.xpyrimentor_definition,
            "seed": self.seed,
            "validate": self.validate,
            "success_level": float(self.success_level),
            "number_of_evaluations": self.number_of_evaluations,
            "x": x,
            "y": [float(y) for y in instance.xpyrimentor.yi],
            "estimated_optima": self.estimated_optima,
            "success": self.success,
        }


@functools.cache
def find_limits(
    model_system_name: str,
    expected_random_runtime: float,
    noise_level: float,
):
    seed = 42
    random_scaling = 100
    model_system = get_model_system(model_system_name, seed=seed)
    model_system.noise_size = model_system.noise_size * noise_level
    sampler = XpyriMentor(
        space=model_system.space, suggestor={"suggestor_name": "GoldenRatio"}, seed=seed
    )
    estimated_points = [
        (point, find_pesimistic_value(model_system, point))
        for point in sampler.ask(expected_random_runtime * random_scaling)
    ]
    # Sort the points by score, and find the point that corresponds to the expected
    # random runtime
    estimated_points.sort(key=lambda x: x[1])
    limit_point = estimated_points[int(random_scaling)]
    return limit_point[1]


def find_pesimistic_value(model_system: ModelSystem, x: Iterable):
    """
    Find the value that is 2 standard deviations above the true value at `x`.
    """
    model_system = model_system.copy()  # Copy to avoid changing the original
    # Set the noise model so that we always return two standard deviations above the true
    # value.
    model_system.noise_model.noise_types["constant"] = lambda: 2
    model_system.noise_model.noise_type = "constant"
    return model_system.get_score(x)
