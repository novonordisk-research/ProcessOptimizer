import numpy as np

from .model_system import ModelSystem
from ..space import Real


# Defining the Branin-Hoo benchmark function


def branin_score(x):
    """
    The Branin-Hoo function.

    Intended parameter Space during benchmark use:
        [(-5.0, 10.0), (0.0, 15.0)]
        If x contains more than two dimensions the excess are ignored.

    Global minimum value, f(x*): 0.3979
    Global mimimum location, x*: (-pi, 12.275), (+pi, 2.275), and (9.425, 2.475)
    Local minima locations, x**: None.
    Global maximum value, f(x+): 308.13
    Global maximum location, x+: (-5.0, 0.0)

    More details: <http://www.sfu.ca/~ssurjano/branin.html>

    Parameters
    ----------
    * 'x' [array of floats of length >=2]:
        The point to evaluate the function at.

    Returns
    -------
    * 'score' [float]:
        The score of the system at x.
    """
    # Define the constants that are canonically used with this function.
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5.0 / np.pi
    r = 6
    s = 10
    t = 1.0 / (8 * np.pi)
    return (
        a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s
    )


# Create model system

def create_branin(noise: bool = True) -> ModelSystem:
    """
    Create the Branin-Hoo model system.

    Parameters
    ----------
    * `noise` [bool]:
        Whether to add noise to the system.

    Returns
    -------
    * `branin` [ModelSystem]:
        The Branin-Hoo model
    """
    if noise:
        return ModelSystem(
            branin_score,
            [Real(-5, 10, name="x1"), Real(0, 15, name="x2")],
            noise_model={"model_type": "proportional", "noise_size": 0.1},
            true_max=308.13,
            true_min=0.397887,
        )
    else:
        return ModelSystem(
            branin_score,
            [Real(-5, 10, name="x1"), Real(0, 15, name="x2")],
            noise_model=None,
            true_max=308.13,
            true_min=0.397887,
        )
