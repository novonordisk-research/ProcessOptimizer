import numpy as np

from .model_system import ModelSystem


def poly2_score(x):
    """
    A simple 2D polynomial with one minimum.

    Intended parameter Space during benchmark use:
        [(-1.0, 1.0), (-1.0, 1.0)]
        If x contains more than two dimensions the excess are ignored.

    Global minimum value, f(x*): -2.0512
    Global mimimum location, x*: (0.6667, -0.4833)
    Global maximum value, f(x+): -1.270
    Global maximum location, x+: (-1.0, -1.0)

    Parameters
    ----------
    * 'x' [array of floats of length >=2]:
        The point to evaluate the function at.

    Returns
    -------
    * 'score' [float]:
        The score of the system at x.
    """
    return -(
        2
        - 0.2 * ((x[0] - 0.3) ** 2 + (x[1] - 0.1) ** 2)
        + 0.05 * x[0]
        - 0.1 * x[1]
        - 0.2 * x[0] * x[1]
    )


def create_poly2(noise: bool = True) -> ModelSystem:
    noise_model = "constant" if noise else None
    return ModelSystem(
        poly2_score,
        [(-1.0, 1.0), (-1.0, 1.0)],
        noise_model=noise_model,
        true_max=-1.270,
        true_min=-2.0512,
    )
