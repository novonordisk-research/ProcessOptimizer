import numpy as np

from .model_system import ModelSystem


def hart6_score(x):
    """
    The six dimensional Hartmann function.

    Intended parameter Space during benchmark use:
        The unit hypercube [(0.0, 1.0), (0.0, 1.0), etc.] in six dimensions.
        If x contains more than six dimensions the excess are ignored.

    Global minimum value, f(x*): -3.3224
    Global mimimum location, x*: (0.2017, 0.1500, 0.4769, 0.2753, 0.3117, 0.6573)
    Local minima locations, x**: Six exist, not stated here.
    Global maximum value, f(x+): 0.0000
    Global maximum location, x+: (1, 1, 0, 1, 1, 1), but close to the same in
                                 all other corners of the space

    More details: <http://www.sfu.ca/~ssurjano/hart6.html>

    Parameters
    ----------
    * 'x' [array of floats of length >=6]:
        The point to evaluate the function at.

    Returns
    -------
    * 'score' [float]:
        The score of the system at x.
    """
    # Define the constants that are canonically used with this function.
    alpha = np.asarray([1.0, 1.2, 3.0, 3.2])
    P = 10**-4 * np.asarray(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    A = np.asarray(
        [
            [10.0, 3.0, 17.0, 3.50, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ]
    )
    return -np.sum(alpha * np.exp(-np.sum(A * (np.array(x) - P) ** 2, axis=1)))


def create_hart6(noise: bool = True) -> ModelSystem:
    noise_model = "constant" if noise else None
    return ModelSystem(
        hart6_score,
        [(0.0, 1.0) for _ in range(6)],
        noise_model=noise_model,
        true_max=0.0,
        true_min=-3.3224,
    )
