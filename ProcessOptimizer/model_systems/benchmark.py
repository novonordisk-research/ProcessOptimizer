from __future__ import annotations
import functools
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from . import get_model_system, ModelSystem
from ProcessOptimizer import Optimizer
from ProcessOptimizer.utils import expected_minimum
from XpyriMentor import XpyriMentor


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
            **kwargs
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
        self.__dict__.update({
            "model_system_name": model_system_name,
            "xpyrimentor_definition":xpyrimentor_definition,
            "experimental_budget":experimental_budget,
            "expected_random_runtime":expected_random_runtime,
            "seed":seed,
            "noise_level":noise_level,
        })
        self.__dict__.update(kwargs)
        self.success_level = find_limits(
            model_system_name, expected_random_runtime, self.noise_level
        )
        self.model = get_model_system(model_system_name, seed=seed)
        self.xpyrimentor = XpyriMentor(self.model.space, self.xpyrimentor_definition, seed=seed)

    @property
    def model_system(self) -> ModelSystem:
        model_system = get_model_system(self.model_system_name, seed=self.seed)
        model_system.noise_size = model_system.noise_size*self.noise_level
        return model_system

    def run(self) -> BenchmarkInstance:
        """
        Run the benchmark instance, save the number of evaluations and whether the
        success level was reached, and return the instance.
        """
        success = False
        while len(self.xpyrimentor.Xi) < self.experimental_budget:
            x = self.xpyrimentor.ask()
            y = self.model.get_score(x)
            self.xpyrimentor.tell(x, [y])
            # We could restrict testing to only if the point is considered good, but it
            # doesn't seem to matter much for the runtime.
            minimum_location, minimum_value = self.find_estimated_optimum()
            if minimum_value<self.success_level and self.validate:
                result = self.model.get_score(minimum_location)
                self.xpyrimentor.tell(minimum_location, result)
                minimum_location, minimum_value = self.find_estimated_optimum()
            if minimum_value<self.success_level:
                # Insert validation here
                true_quality = find_pesimistic_value(self.model, minimum_location)
                if true_quality<self.success_level:
                    success = True
                break
        self.number_of_evaluations = len(self.xpyrimentor.Xi)
        self.success = success
        return self
    
    def find_estimated_optimum(self) -> float:
        """
        Find the parameter set that is estimated to be the optimum, and the value that is
        2 standard deviations above the true value at that point.
        """
        # This is a bit of a hack, but it works for now. The optimizer is the second
        # suggestor in the suggestor list of the sequential strategizer. We need the
        # optimizer since we need to know the expected minimum and the model uncertainty
        # there.
        optimizer: Optimizer = self.xpyrimentor.suggestor.suggestors[-1][1].optimizer
        optimizer.Xi = self.xpyrimentor.Xi
        optimizer.yi = self.xpyrimentor.yi
        optimizer.update_next()
        result = optimizer.get_result()
        result_location, [result_value, result_std] = expected_minimum(result, return_std=True)
        return (result_location, result_value + 2*result_std)


@functools.cache
def find_limits(
        model_system_name: str,
        expected_random_runtime: float,
        noise_level: float,
    ):
    seed = 42
    random_scaling = 100
    model_system = get_model_system(model_system_name, seed=seed)
    model_system.noise_size = model_system.noise_size*noise_level
    sampler = XpyriMentor(
        space=model_system.space, suggestor={"suggestor_name": "GoldenRatio"}, seed=seed
    )
    estimated_points = [
        (point, find_pesimistic_value(model_system, point))
        for point in sampler.ask(expected_random_runtime*random_scaling) 
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
    model_system = model_system.copy() # Copy to avoid changing the original
    # Set the noise model to be constant, which means always return two standard deviations
    # above the true value.
    model_system.noise_model.noise_type = "constant"
    return model_system.get_score(x)