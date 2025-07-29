from __future__ import annotations
from typing import Any, Union, Iterable

import numpy as np
from ProcessOptimizer.space import Space
from ProcessOptimizer.utils import is_listlike


class ConstantSuggestor:
    """
    A suggestor that always returns the same point.
    """

    def __init__(
        self,
        space: Space,
        point: Union[list, float] = 0.5,
        convert: bool = True,
        **kwargs,  # To catch any additional keyword arguments
    ):
        """
        Initialize the suggestor.

        Parameters:
        * `space` [Space]:
            The search space.
        * `point` [list or float]:
            The point to suggest. If not a list, it is converted to a list with
            `space.n_dims` identical elements. Default is 0.5, which, if `convert` is
            `True`, makes the suggestor return the center point of `space`.
        * `convert` [bool]:
            If `True` (default), the point is converted to a point in the space with
            `space.sample`
        """
        self.space = space
        if not is_listlike(point):
            point = [point] * space.n_dims
        if convert:
            point = space.sample(point)
        self.point = np.array(point)

    def suggest(
        self, Xi: Iterable[Iterable], Yi: Iterable, n_asked: int = 1
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
        * `n_asked` [int]:
            The number of suggested points to return. Must be a positive integer.
        Returns:
        -------
        A np.ndarray of size `n_asked` x `n_dim`, where `n_dim` is the number of
        dimensions in the search space. The points are all the same, equal to `self.point`.
        """
        return np.tile(self.point, (n_asked, 1))

    @classmethod
    def create_from_definition(
        cls,
        space: Space,
        suggestor_factory,
        definition: dict[str, Any],
        n_objectives,
        rng: np.random.Generator,
    ) -> ConstantSuggestor:
        return ConstantSuggestor(space=space, **definition)
