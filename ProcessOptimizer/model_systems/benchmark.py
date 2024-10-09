from typing import Union

from . import get_model_system
from ProcessOptimizer import Optimizer


def run_test_optimization(
        test: dict[str, Union[str, float, int]]
) -> tuple[dict, int, bool]:
    """
    Run an optimization test and return the number of evaluations done and whether the
    optimization was successful

    Parameters
    ----------
    test : dict
        A dictionary containing the test parameters.

    Returns
    -------
    tuple
        A tuple containing the input test (for reference), the number of evaluations done
        and whether the optimization was successful.
    """
    model_system = get_model_system(test["model_system_name"], seed = test["seed"])
    model_system.noise_size = model_system.noise_size*test["noise_level"]
    objective_range = model_system.true_max - model_system.true_min
    target = model_system.true_min + test["target_level"] * objective_range
    optimizer = Optimizer(
        dimensions=model_system.space,
        n_initial_points=test["n_initial_points"]
    )
    for _ in range(test["experiment_budget"]):
        finished = False
        x = optimizer.ask()
        y = model_system.get_score(x)
        if y < target:
            # We have found a point that is close enough to the true minimum
            # There should be some more logic here, to check if the point is acutally
            # good enough, or whether it was just luck.
            finished = True
            break
        optimizer.tell(x, y)
    return (test, len(optimizer.Xi), finished)
