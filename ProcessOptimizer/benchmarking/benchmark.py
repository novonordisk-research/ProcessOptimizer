from __future__ import annotations
import functools
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ProcessOptimizer.model_systems import get_model_system, ModelSystem
from ProcessOptimizer import Optimizer, XpyriMentor
from ProcessOptimizer.XpyriMentor.suggestors import ConstantSuggestor
from ProcessOptimizer.utils import expected_minimum


@dataclass
class BenchmarkInstance:
    """
    A benchmark instance for evaluating the performance of a suggestor.

    The evaluation is done in a constant target manner, and the number of evaluations,
    whether it reached the target is saved.

    Parameters
    ----------
    model_system_name [str]: The name of the model system to use. Only internal model
        systems are usable, since the optimisation target is calculated and cached, and
        caching doesn't work on instantiated model systems.
    suggestor_definition [dict[str, Any]]: The definition of the suggestor to use. This
        is fed to the suggestor factory.
    experimental_budget [int]: The maximum number of evaluations to run before stopping.
    expected_random_runtime [float]: How hard the optimization is. Randomly sampling the
        model system until you find an acceptable configuration takes this many times,
        on average. On initialization of the `BenchmarkInstance`,
        `100*expected_random_runtime` random points in the search space are evaluated on
        what objective would be reached 95% of the times there (2 standard deviations).
        The 100th lowest objective value is used as the target for the optimization.
    seed [int]: The random seed to use for the optimization. This ensures that the
        optimization is reproducible.
    noise_level [float]: The noise level to use for the optimization. This is multiplied
        with the noise size of the model system, so it can be used to scale the noise
        size up or down.
    validate [bool]: Whether to validate the optimization result. If True, when the
        benchmark has found a point that is below the success level, it will
        re-evaluate the point to ensure that it is indeed below the success level.

    Results
    -------
    estimated_optima [list[tuple[float, float, float]]]: A list of tuples containing the
        estimated location, value, and standard deviation of the estimated optimum after
        each point has been added.
    number_of_evaluations [int | None]: The number of evaluations that were done during
        the optimization. This is None if the optimization was not run.
    success [bool | None]: Whether the optimization was successful. This is None if the
        optimization was not run.

    Internal variables
    -------------------
    success_level [float]: The target objective value that the optimization should reach.
        This is calculated on initialization of the `BenchmarkInstance`.
    model_system [ModelSystem]: The model system to use for the optimization. This is a copy of
        the model system with the noise size scaled by the noise level.
    xpyrimentor [XpyriMentor]: The XpyriMentor instance to use for the optimization.
        This is initialized with the model system and the suggestor definition.
    """

    model_system_name: str
    suggestor_definition: dict[str, Any]
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
    model_system: ModelSystem = field(init=False, repr=False)
    xpyrimentor: XpyriMentor = field(init=False, repr=False)

    def __init__(
        self,
        model_system_name: str,
        suggestor_definition: dict[str, Any],
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
            Name of the model system to use. Has to be a name of an internal
            ModelSystem. Examples are 'hart3' or 'hart6'. You can see the full list in
            ProcessOptimizer\\model_systems\\model_system_getter.py
        * `suggestor_definition` [dict]:
            Definition of the suggestor object to use.
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
                "suggestor_definition": suggestor_definition.copy(),
                "experimental_budget": experimental_budget,
                "expected_random_runtime": expected_random_runtime,
                "seed": seed,
                "noise_level": noise_level,
            }
        )
        self.__dict__.update(kwargs)
        # Translating the difficulty level (expected random runtime) to a value of the
        # objective.
        self.success_level = find_limits(
            model_system_name, expected_random_runtime, self.noise_level
        )
        # TODO: Nicer error if the model system is not found
        self.model_system = get_model_system(model_system_name, seed=seed)
        self.model_system.noise_size *= noise_level
        self.xpyrimentor = XpyriMentor(
            self.model_system.space, self.suggestor_definition, seed=seed
        )
        self.estimated_optima = []  # This will end up with one point per evaluation.
        # Each entry is the estimated optimum when we only have the points suggested "so far".

    @property
    def optimizer(self) -> Optimizer:
        # This is a bit of a hack, but it works for now. The optimizer is the last
        # suggestor in the suggestor list of the sequential strategizer.
        return self.xpyrimentor.suggestor.suggestors[-1][1].optimizer

    def tell_benchmark(self, x: Iterable, y: float) -> tuple[float, float, float]:
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
        # To estimate the optimum, we need to update the optimizer with the new data.
        # We then add observational noise to the optimizer to get the correct standard
        # deviation, find the position, value, and standard deviation of the expected
        # minimum, and tell the optimizer to stop including observational noise.
        # Updating with the points so far makes this not a pure function. But since
        # the OptimizerSuggestor does the same, we should be good.
        optimizer: Optimizer = self.optimizer
        optimizer.Xi = self.xpyrimentor.Xi
        optimizer.yi = self.xpyrimentor.yi
        optimizer.update_next()
        optimizer.add_observational_noise()
        result = optimizer.get_result()
        result_location, [result_value, result_std] = expected_minimum(
            result, return_std=True, random_state=np.random.RandomState(self.seed)
        )
        optimizer.remove_observational_noise()
        self.estimated_optima.append((result_location, result_value, result_std))
        return (result_location, result_value, result_std)

    def run(self):
        self.success = False
        while len(self.xpyrimentor.Xi) < self.experimental_budget:
            x = self.xpyrimentor.ask()
            y = self.model_system.get_score(x)
            minimum_location, minimum_value, minimum_std = self.tell_benchmark(x, y)
            # If the validation is enabled, we do an extra experiment when it seems good
            # enough, to be more sure that we have a good point.
            if self.validate and (minimum_value + 2 * minimum_std) < self.success_level:
                result = self.model_system.get_score(minimum_location)
                minimum_location, minimum_value, minimum_std = self.tell_benchmark(
                    [minimum_location], result
                )
            if (minimum_value + 2 * minimum_std) < self.success_level:
                true_quality = find_pessimistic_value(
                    self.model_system, minimum_location
                )
                if true_quality < self.success_level:
                    self.success = True
                break
        self.number_of_evaluations = len(self.xpyrimentor.Xi)

    @property
    def replicate_suggestors(self) -> list[tuple[int, ConstantSuggestor]]:
        """All replicate suggestors and how many points each consume."""
        return [
            suggestor
            for suggestor in self.xpyrimentor.suggestor.suggestors
            if isinstance(suggestor[1], ConstantSuggestor)
        ]

    @property
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
        # This format assumes we have an Optimizer. If we don't, we should probably just
        # return the suggestor definition, it contains all information about the
        # suggestor.
        return {
            "model_system_name": self.model_system_name,
            "expected_random_runtime": self.expected_random_runtime,
            "experimental_budget": self.experimental_budget,
            "noise_level": self.noise_level,
            "n_initial_points": self.xpyrimentor.suggestor.suggestors[0][0],
            "n_replicates": [number for number, _ in self.replicate_suggestors],
            "acq_func_kwargs": self.optimizer.acq_func_kwargs,
            "noise_level_bounds": self.optimizer.base_estimator_.noise_level_bounds,
            "length_scale_bounds": self.optimizer.base_estimator_.kernel.get_params()[
                "k2__length_scale_bounds"
            ],
            "suggestor_definition": self.suggestor_definition,
            "seed": self.seed,
            "validate": self.validate,
            "success_level": float(self.success_level),
            "number_of_evaluations": self.number_of_evaluations,
            "x": x,
            "y": [float(y) for y in self.xpyrimentor.yi],
            "estimated_optima": self.estimated_optima,
            "success": self.success,
        }


@functools.cache
def find_limits(
    model_system_name: str,
    expected_random_runtime: float,
    noise_level: float,
):
    """
    Find the limits for the given model system, expected random runtime, and noise level.

    The limit is the objective value that a point in the `model_system.space` has a
    probability of `1/expected_random_runtime` of having a pessimistic objective value
    lower than.

    The pessimistic objective value of a point is 2 standard deviations above the mean
    value.
    """
    # Special seed since multiple benchmark instances has the same limit.
    seed = 42
    random_scaling = 100
    model_system = get_model_system(model_system_name, seed=seed)
    model_system.noise_size = model_system.noise_size * noise_level
    # Use Generalised golden ratio sampler to find points in the space.
    sampler = XpyriMentor(
        space=model_system.space, suggestor={"suggestor_name": "GoldenRatio"}, seed=seed
    )
    # Find the values for sampled points
    estimated_points = [
        find_pessimistic_value(model_system, point)
        for point in sampler.ask(expected_random_runtime * random_scaling)
    ]
    # Sort the objective values of the points
    estimated_points.sort()
    limit_point = estimated_points[int(random_scaling)]
    return limit_point


def find_pessimistic_value(model_system: ModelSystem, x: Iterable):
    """
    Find the value that is 2 standard deviations above the true value at `x`.

    We use this for our goal since a solution is only relevant in production if the
    quality is good enough the vast majority of the time. So the 2.2th percentile of the
    quality has to be above a certain threshold. Since we are minimizing, this turns
    into a demand on the value two standard deviations above the mean quality.
    """
    model_system = model_system.copy()  # Copy to avoid changing the original
    # Set the noise model so that we always return two standard deviations above the true
    # value.
    model_system.noise_model.possible_noise_types["pessimistic"] = lambda: 2
    model_system.noise_model.noise_type = "pessimistic"
    return model_system.get_score(x)
