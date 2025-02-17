from typing import Union, Iterable

import numpy as np
from ProcessOptimizer.space import Space
from ProcessOptimizer.utils import is_listlike


class ConstantSuggestor():
    """
    A suggestor that always returns the same point.
    """
    def __init__(self, space: Space, point: Union[list, float] = 0.5, convert: bool = True):
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
        return np.tile(self.point, (n_asked, 1))