# -*- coding: utf-8 -*-
"""A collection of benchmark problems."""

from typing import Callable, List, Union
import numpy as np

from .model_system import ModelSystem


def bench1(x):
    """A benchmark function for test purposes.

        f(x) = x ** 2

    It has a single minima with f(x*) = 0 at x* = 0.
    """
    return x[0] ** 2


def bench1_with_time(x):
    """Same as bench1 but returns the computation time (constant)."""
    return x[0] ** 2, 2.22


def bench2(x):
    """A benchmark function for test purposes.

        f(x) = x ** 2           if x < 0
               (x-5) ** 2 - 5   otherwise.

    It has a global minima with f(x*) = -5 at x* = 5.
    """
    if x[0] < 0:
        return x[0] ** 2
    else:
        return (x[0] - 5) ** 2 - 5


def bench3(x):
    """A benchmark function for test purposes.

        f(x) = sin(5*x) * (1 - tanh(x ** 2))

    It has a global minima with f(x*) ~= -0.9 at x* ~= -0.3.
    """
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2))


def bench4(x):
    """A benchmark function for test purposes.

        f(x) = float(x) ** 2

    where x is a string. It has a single minima with f(x*) = 0 at x* = "0".
    This benchmark is used for checking support of categorical variables.
    """
    return float(x[0]) ** 2


def bench5(x):
    """A benchmark function for test purposes.

        f(x) = float(x[0]) ** 2 + x[1] ** 2

    where x is a string. It has a single minima with f(x) = 0 at x[0] = "0"
    and x[1] = "0"
    This benchmark is used for checking support of mixed spaces.
    """
    return float(x[0]) ** 2 + x[1] ** 2


def poly2(x):
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


poly2_model_system = ModelSystem(
    poly2,
    [(-1.0, 1.0), (-1.0, 1.0)],
    noise_model="constant",
    true_max=-1.270,
    true_min=-2.0512,
)


def peaks(x):
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


peaks_model_system = ModelSystem(
    peaks,
    [(-3.0, 3.0), (-3.0, 3.0)],
    noise_model="constant",
    true_max=8.106,
    true_min=-6.5511,
)
