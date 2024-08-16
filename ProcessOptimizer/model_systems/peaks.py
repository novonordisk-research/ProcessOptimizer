import numpy as np

from .model_system import ModelSystem


def peaks_score(x):
    """
    The peaks function widely used in MATLAB.

    Intended parameter Space during benchmark use:
        [(-3.0, 3.0), (-3.0, 3.0)].
        If x contains more than two dimensions the excess are ignored.

    Global minimum value, f(x*): -6.5511
    Global mimimum location, x*: (0.228, -1.626)
    Local minima locations, x**: (-1.348, 0.205)
    Global maximum value, f(x+): 8.106
    Global maximum location, x+: (-0.010, 1.581)

    Parameters
    ----------
    * 'x' [array of floats of length >=2]:
        The point to evaluate the function at.

    Returns
    -------
    * 'score' [float]:
        The score of the system at x.
    """
    score = (
        3 * (1 - x[0]) ** 2 * np.exp(-x[0] ** 2 - (x[1] + 1) ** 2)
        - 10 * (x[0] / 5 - x[0] ** 3 - x[1] ** 5) * np.exp(-x[0] ** 2 - x[1] ** 2)
        - 1 / 3 * np.exp(-((x[0] + 1) ** 2) - x[1] ** 2)
    )
    return score


def create_peaks(noise: bool = True) -> ModelSystem:
    noise_model = "constant" if noise else None
    return ModelSystem(
        peaks_score,
        [(-3.0, 3.0), (-3.0, 3.0)],
        noise_model=noise_model,
        true_max=8.106,
        true_min=-6.5511,
    )
