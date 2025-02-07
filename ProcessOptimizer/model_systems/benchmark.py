from __future__ import annotations
from dataclasses import dataclass


from . import get_model_system
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
    model_system = get_model_system(test["model_system_name"], seed=test["seed"])
    model_system.noise_size = model_system.noise_size*test["noise_level"]
    objective_range = model_system.true_max - model_system.true_min
    target = model_system.true_min + test["target_level"] * objective_range
    optimizer = XpyriMentor(
        model_system.space,test.xpyrimentor_definition,seed=test["seed"]
    )
    finished = False
    for _ in range(test["experiment_budget"]):

        x = optimizer.ask()
        y = model_system.get_score(x) # include model noise
        if y < target:
            # We have found a point that is close enough to the true minimum
            # There should be some more logic here, to check if the point is acutally
            # good enough, or whether it was just luck.
            finished = True
            break
        optimizer.tell(x, y)
    TestResult.number_of_evaluations = len(optimizer.Xi)
    TestResult.success = finished

@dataclass
class TestResult:
    model_system_name: str
    expected_random_runtime: float
    success_chance: float # What the chance of being under the target is in a successful point
    noise_level: float
    target_level: float
    experimental_budget: int
    xpyrimentor_definition: dict
    seed: int
    validate: bool
    number_of_evaluations: int | None = None
    success: bool | None = None

    def __init__(self, **kwargs):
        success_level = find_limits(model_system_name)
        self.__dict__.update(kwargs)