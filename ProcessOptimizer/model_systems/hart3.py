import numpy as np

from .model_system import ModelSystem
from ..space import Real


def hart3_score(x):
    """
    The three dimensional Hartmann function.

    Intended parameter Space during benchmark use:
        The unit hypercube [(0.0, 1.0), (0.0, 1.0), etc.] in three dimensions.
        If x contains more than three dimensions the excess are ignored.

    Global minimum value, f(x*): -3.863
    Global mimimum location, x*: (0.1146, 0.5556, 0.8525)
    Local minima locations, x**: Four exist, not stated here.
    Global maximum value, f(x+): 0.0000
    Global maximum location, x+: (1, 1, 0)

    More details: <https://www.sfu.ca/~ssurjano/hart3.html>

    Parameters
    ----------
    * 'x' [array of floats of length >=3]:
        The point to evaluate the function at.

    Returns
    -------
    * 'score' [float]:
        The score of the system at x.
    """
    # Define the constants that are canonically used with this function.
    alpha = np.asarray([1.0, 1.2, 3.0, 3.2])
    P = 10**-4 * np.asarray(
        [[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]]
    )
    A = np.asarray(
        [[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]]
    )
    return -np.sum(alpha * np.exp(-np.sum(A * (np.array(x) - P) ** 2, axis=1)))


hart3 = ModelSystem(
    hart3_score,
    [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    noise_model="constant",
    true_max=0.0,
    true_min=-3.863,
)

hart3_no_noise = ModelSystem(
    hart3_score,
    [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
    noise_model=None,
    true_max=0.0,
    true_min=-3.863,
)
