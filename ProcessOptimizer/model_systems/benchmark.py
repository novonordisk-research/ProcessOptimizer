from __future__ import annotations
import functools
from dataclasses import dataclass, field
from typing import Any, Iterable


from . import get_model_system, ModelSystem
from ProcessOptimizer import Optimizer
from ProcessOptimizer.utils import expected_minimum
from XpyriMentor import XpyriMentor


@dataclass
class BenchmarkInstance:
    model_system_name: str
    experimental_budget: int
    xpyrimentor_definition: dict
    seed: int
    validate: bool = False
    expected_random_runtime: float = 1000.0
    noise_level: float = 1.0
    number_of_evaluations: int | None = None
    success: bool | None = None
    success_level: float = field(init=False, repr=False)
    model: ModelSystem = field(init=False, repr=False)
    xpyrimentor: XpyriMentor = field(init=False, repr=False)
    optimizer: Optimizer = field(init=False, repr=False)

    def __init__(
            self,
            model_system_name: str,
            expected_random_runtime: float,
            noise_level: float,
            seed: int,
            **kwargs
        ):
        self.__dict__.update({
            "model_system_name": model_system_name,
            "expected_random_runtime":expected_random_runtime,
            "noise_level":noise_level,
            "seed":seed,
        })
        self.__dict__.update(kwargs)
        self.success_level = find_limits(
            model_system_name, expected_random_runtime, noise_level
        )
        self.model = get_model_system(model_system_name, seed=seed)
        self.xpyrimentor = XpyriMentor(self.model.space, self.xpyrimentor_definition, seed=seed)
        self.optimizer = self.xpyrimentor.suggestor.suggestors[1][1].optimizer

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
            self.optimizer.Xi = self.xpyrimentor.Xi
            self.optimizer.yi = self.xpyrimentor.yi
            self.optimizer.update_next()
            result = self.optimizer.get_result()
            result_location, [result_value, result_std] = expected_minimum(result, return_std=True)
            if result_value + 2*result_std < self.success_level: # Include modelled noise
                # Insert validation here
                true_quality = find_pesimistic_value(self.model, result_location)
                if true_quality<self.success_level:
                    success = True
                break
        self.number_of_evaluations = len(self.xpyrimentor.Xi)
        self.success = success
        return self

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
    Find the value that is `std` standard above below the true value at `x`.
    """
    model_system = model_system.copy() # Copy to avoid changing the original
    # Set the noise model to be constant, which means always return two standard deviations
    # above the true value.
    model_system.noise_model.noise_type = "constant"
    return model_system.get_score(x)