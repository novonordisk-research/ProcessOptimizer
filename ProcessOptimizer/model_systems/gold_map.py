from typing import Sequence
import numpy as np

from . import ModelSystem


def score(coordinates: Sequence[float]):
    """
    Modified Branin-Hoo function. It has three approximate local minima.

    It returns a value between 0 and -3.1, interpreted as the negative value of
    amount of gold found at the given coordinates, in mg. The negative value is
    to make it play nicely together with ProcessOptimizer, which is a minimizer.

    More details: <http://www.sfu.ca/~ssurjano/branin.html>
    """
    x = 10 - coordinates[0]
    y = coordinates[1]
    gold_found = ((y - 1 / 8 * x**2 + 1.6 * x - 6) ** 2 + 10 * np.cos(x) - 299) / 100
    return gold_found


gold_map = ModelSystem(
    score,
    space=[(0.0, 15.0), (0.0, 15.0)],
    noise_model=None,
    true_min=-3.09,
)
