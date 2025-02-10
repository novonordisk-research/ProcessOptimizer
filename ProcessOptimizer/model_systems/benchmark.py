from __future__ import annotations
import functools
from dataclasses import dataclass, field


from . import get_model_system, ModelSystem
from ...XpyriMentor import XpyriMentor

def run_test_optimization(test: TestResult) -> None:
    """
    Run an optimization test and modifies the input object with the number of evaluations
    done and whether the optimization was successful

    Parameters
    ----------
    test : TestResult
        The test to run
    """
    model_system = get_model_system(test.model_system_name, seed=test.seed)
    model_system.noise_size = model_system.noise_size*test.noise_level
    optimizer = XpyriMentor(
        model_system.space,test.xpyrimentor_definition,seed=test.seed
    )
    finished = False
    while len(optimizer.Xi) < test.experimental_budget:
        x = optimizer.ask()
        y = model_system.get_score(x) # include model noise
        optimizer.tell(x, y)
        if y < test.success_level:
            # Insert validation here
            finished = True
            break
        optimizer.tell(x, y)
    test.number_of_evaluations = len(optimizer.Xi)
    test.success = finished

@dataclass
class TestResult:
    model_system_name: str
    experimental_budget: int
    xpyrimentor_definition: dict
    seed: int
    validate: bool 
    expected_random_runtime: float = 1000.0
    noise_level: float = 1.0
    success_level: float = field(init=False)
    number_of_evaluations: int | None = None
    success: bool | None = None

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
        self.__dict__["success_level"] = find_limits(self.model_system, expected_random_runtime, seed)

    @property
    def model_system(self) -> ModelSystem:
        model_system = get_model_system(self.model_system_name, seed=self.seed)
        model_system.noise_size = model_system.noise_size*self.noise_level
        return model_system

@functools.cache()
def find_limits(
        model_system: ModelSystem,
        expected_random_runtime: float,
        noise_level: float,
        seed: int,
    ):
    random_scaling = 100
    n_noise_points = 1000
    # Save the noise size for later
    noise_level = model_system.noise_size
    # Find the noiseless score on a set of space filling points
    model_system.noise_size = 0.0
    sampler = XpyriMentor(model_system.space, seed=seed) # TODO: Use golden ratio sampling instead
    estimated_points = [
        (point, model_system.get_score(point))
        for point in sampler.ask(expected_random_runtime*random_scaling) 
    ]
    # Sort the points by score, and find the point that corresponds to the expected
    # random runtime
    estimated_points.sort(key=lambda x: x[1])
    limit_point = estimated_points[int(random_scaling)]
    # Find the 95th percentile of the noise at the limit point
    model_system.noise_size = noise_level
    noise_points = [model_system.noise_model.get_noise(*limit_point) for _ in range(n_noise_points)]
    noise_points.sort()
    return limit_point[1] + noise_points[int(0.95*n_noise_points)]