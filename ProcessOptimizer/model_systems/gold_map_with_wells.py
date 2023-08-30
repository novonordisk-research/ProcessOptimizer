from collections.abc import Iterable

import numpy as np

from . import ModelSystem
from .gold_map import score as gold_map_score

well_list = [
    {"x": 3, "y": 10, "width": 1, "gold": 3},
    {"x": 11, "y": 4, "width": 1, "gold": 2},
]


def score(coordinates: Iterable[float]):
    """Modified Branin-Hoo function, with 'narrow wells' added.'"""
    gold_found = gold_map_score(coordinates)

    def normal_distribution(x, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * (x / sigma) ** 2)

    for well in well_list:
        dist = np.sqrt(
            (coordinates[0] - well["x"]) ** 2 + (coordinates[1] - well["y"]) ** 2
        )
        gold_found -= well["gold"] * normal_distribution(dist, well["width"])
    return gold_found


gold_map_with_wells = ModelSystem(
    score,
    space=[(0.0, 15.0), (0.0, 15.0)],
    noise_model=None,
)
