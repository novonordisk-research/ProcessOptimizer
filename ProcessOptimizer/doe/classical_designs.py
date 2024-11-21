# import math
# import warnings

import numpy as np

# from .doe_transform import doe_to_real_space

# In here goes some classical designs.
# They could be adapted from https://github.com/tisimst/pyDOE


def fullfact(levels):
    """
    Create a general full-factorial design

    Parameters
    ----------
    levels : array-like
        An array of integers that indicate the number of levels of each input
        design factor.

    Returns
    -------
    mat : 2d-array
        The design matrix with coded levels 0 to k-1 for a k-level factor

    Example
    -------
    ::
        >>> fullfact([2, 4, 3])
        array([[ 0.,  0.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 1.,  1.,  0.],
               [ 0.,  2.,  0.],
               [ 1.,  2.,  0.],
               [ 0.,  3.,  0.],
               [ 1.,  3.,  0.],
               [ 0.,  0.,  1.],
               [ 1.,  0.,  1.],
               [ 0.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 0.,  2.,  1.],
               [ 1.,  2.,  1.],
               [ 0.,  3.,  1.],
               [ 1.,  3.,  1.],
               [ 0.,  0.,  2.],
               [ 1.,  0.,  2.],
               [ 0.,  1.,  2.],
               [ 1.,  1.,  2.],
               [ 0.,  2.,  2.],
               [ 1.,  2.,  2.],
               [ 0.,  3.,  2.],
               [ 1.,  3.,  2.]])

    This function is from https://github.com/tisimst/pyDOE
    Copyright (c) 2014, Abraham D. Lee
    License: BSD 3-Clause License
    License link: https://github.com/tisimst/pyDOE/blob/master/LICENSE
    """
    n = len(levels)  # number of factors
    nb_lines = np.prod(levels)  # number of trial conditions
    H = np.zeros((nb_lines, n))

    level_repeat = 1
    range_repeat = np.prod(levels)
    for i in range(n):
        range_repeat //= levels[i]
        lvl = []
        for j in range(levels[i]):
            lvl += [j]*level_repeat
        rng = lvl*range_repeat
        level_repeat *= levels[i]
        H[:, i] = rng

    return H
