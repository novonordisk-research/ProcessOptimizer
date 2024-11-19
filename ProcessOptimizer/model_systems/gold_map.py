from numbers import Number
from typing import Sequence
import numpy as np

from . import ModelSystem


def score(coordinates: Sequence[Number]):
    """
    Modified Branin-Hoo function. It has three approximate local minima.

    It returns a value between 0 and -3.1, interpreted as the negative value of
    amount of gold found at the given coordinates, in mg. The negative value is
    to make it play nicely together with ProcessOptimizer, which is a
    minimizer.

    More details: <http://www.sfu.ca/~ssurjano/branin.html>
    """
    x = 10 - coordinates[0]
    y = coordinates[1]
    gold_found = (
        (y - 1 / 8 * x**2 + 1.6 * x - 6) ** 2 + 10 * np.cos(x) - 299
    ) / 100
    return gold_found


def create_gold_map() -> ModelSystem:
    """Create the gold map model system."""
    return ModelSystem(
        score,
        space=[(0.0, 15.0), (0.0, 15.0)],
        noise_model=None,
        true_min=-3.09,
    )


def create_distance_map(
        camp_coordinates: Sequence[Number] = (4, 10)
) -> ModelSystem:
    """
    Create the mode system that returns the distance from the camp in gold map.

    Parameters
    ----------
    * `camp_coordinates` [Sequence[Number]]:
        The coordinates of the camp in the gold map. Default is (4, 10).
    """
    def distance(coordinates: Sequence[float]):
        return np.sqrt(
            (coordinates[0] - camp_coordinates[0])**2
            + (coordinates[1] - camp_coordinates[1])**2
        )
    return ModelSystem(
        distance,
        space=[(0.0, 15.0), (0.0, 15.0)],
        noise_model=None,
        true_min=0,
    )