# -*- coding: utf-8 -*-
"""A collection of benchmark problems."""

from typing import Callable, List, Union
import numpy as np


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


def branin(x):
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
    * 'a, b, c, r, s, t' [all float]:
        Default values typically used with this function. Changing any of these
        values will change the relative shape of the function in the space.

    Returns
    -------
    * 'score' [float]:
        The score of the system at x.
    """
    # Define the constants that are canonically used with this function.
    a=1
    b=5.1/(4*np.pi**2)
    c=5./np.pi
    r=6
    s=10
    t=1./(8*np.pi)
    return (a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 +
            s * (1 - t) * np.cos(x[0]) + s)


def hart6(x):
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
    alpha=np.asarray([1.0, 1.2, 3.0, 3.2])
    P=10**-4 * np.asarray([[1312, 1696, 5569, 124, 8283, 5886],
                           [2329, 4135, 8307, 3736, 1004, 9991],
                           [2348, 1451, 3522, 2883, 3047, 6650],
                           [4047, 8828, 8732, 5743, 1091, 381]])
    A=np.asarray([[10, 3, 17, 3.50, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    return -np.sum(alpha * np.exp(-np.sum(A * (np.array(x) - P)**2, axis=1)))


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
        - 0.2*((x[0] - 0.3)**2 + (x[1] - 0.1)**2) 
        + 0.05*x[0] 
        - 0.1*x[1] 
        - 0.2*x[0]*x[1]
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
        3*(1- x[0])**2 * np.exp(-x[0]**2 - (x[1]+1)**2)
        - 10*(x[0]/5 - x[0]**3 - x[1]**5) * np.exp(-x[0]**2 - x[1]**2)
        - 1/3*np.exp(-(x[0]+1)**2 - x[1]**2)
    )
    return score