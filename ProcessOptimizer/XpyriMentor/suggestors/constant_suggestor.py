from __future__ import annotations
from typing import Any, Iterable

import numpy as np
from ProcessOptimizer.space import Space


class ConstantSuggestor:
    """
    A suggestor that always returns the same point.
    """

    def __init__(self, space: Space, point: list[Any]):
        """
        Initialize the suggestor.

        Parameters:
        * `space` [Space]:
            The search space.
        * `point` [list]:
            The point to suggest. It has to be a point in the search space.
        """
        self.space = space
        self.point = np.array(point)

    def suggest(
        self, Xi: Iterable[Iterable], Yi: Iterable, n_points_to_suggest: int = 1
    ) -> np.ndarray:
        """
        Suggest a new point.

        Parameters:
        * `Xi` [Iterable[Iterable]]:
            The input is a list of already evaluated points. Not used in this suggestor.
            Present for consistency with other suggestors and XPyriMentor.
        * `Yi` [Iterable]:
            The results of the evaluations of `Xi`. Not used in this suggestor.
            Present for consistency with other suggestors and XPyriMentor.
        * `n_points_to_suggest` [int]:
            The number of suggested points to return. Must be a positive integer.
        Returns:
        -------
        A np.ndarray of size `n_points_to_suggest` x `n_dim`, where `n_dim` is the number of
        dimensions in the search space. The points are all the same, equal to `self.point`.
        """
        return np.tile(self.point, (n_points_to_suggest, 1))

    @classmethod
    def create_from_definition(
        cls,
        space: Space,
        suggestor_factory,
        definition: dict[str, Any],
        n_objectives,
        rng: np.random.Generator,
    ) -> ConstantSuggestor:
        return ConstantSuggestor(space=space, point=definition["point"])
