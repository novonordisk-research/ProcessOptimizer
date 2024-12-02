import logging
import warnings
from typing import Any, Union

import numpy as np
from ProcessOptimizer.space import space_factory, Space
from ProcessOptimizer.utils.get_rng import get_random_generator

from .suggestors import BatchSuggestor, DefaultSuggestor, Suggestor, suggestor_factory

logger = logging.getLogger(__name__)

DEFAULT_SUGGESTOR = {
    "type": "InitialPoint",
    "initial_suggestor": {"type": "Default"},
    "ultimate_suggestor": {"type": "Default"},
    "n_initial_points": 5
}


class Expyrementor:
    """
    Expyrementor class for optimization experiments. This class is used to manage the
    optimization process, including the search space, the suggestor, and the already
    evaluated points. The ask-tell interface is used to interact with the optimization
    process. The Expyrementor class is stateful and keeps track of the already evaluated
    points and scores.
    """
    def __init__(
        self,
        space: Union[Space, list],
        suggestor: Union[Suggestor, dict, None] = None,
        n_objectives: int = 1,
        seed: Union[int, np.random.RandomState, np.random.Generator, None] = 42
    ):
        """
        Initialize the Expyrementor with the search space and the suggestor. The suggestor
        can be a Suggestor object, a dictionary with the suggestor configuration, or None.
        If the suggestor is None, the default suggestor is used. The seed is used to
        initialize the random number generator.
        """
        space = space_factory(space)
        rng = get_random_generator(seed)
        suggestor = suggestor_factory(space, suggestor, n_objectives, rng)
        if isinstance(suggestor, DefaultSuggestor):
            logger.debug("Replacing DefaultSuggestor with InitialPointSuggestor")
            suggestor = suggestor_factory(space, DEFAULT_SUGGESTOR, n_objectives, rng=rng)
        self.suggestor = suggestor
        self._suggested_experiments: list[list] = []
        self.Xi: list[list] = []
        # This is a list of points in the search space. Each point is a list of values for
        # each dimension of the search space.
        self.yi: list = []
        # We are agnostic to the type of the scores. They can be a float for single
        # objective optimization or a list of floats for multiobjective optimization.
        pass

    def ask(self, n: int = 1):
        if n > 1 and not isinstance(self.suggestor, BatchSuggestor):
            warnings.warn(
                "No batch suggestor is defined. Asking for multiple suggestions might not work."
            )
        if len(self._suggested_experiments) < n:
            if isinstance(self.suggestor, BatchSuggestor):
                self.suggestor.n_given = n - len(self._suggested_experiments)
            self._suggested_experiments.extend(self.suggestor.suggest(self.Xi, self.yi))
        if len(self._suggested_experiments) < n:
            raise ValueError("The suggestor did not return enough suggestions.")
        return self._suggested_experiments[:n]

    def tell(self, x: list, y: Any):
        self.Xi.append(x)
        self.yi.append(y)
        if len(x) > len(self._suggested_experiments):
            self._suggested_experiments = []
        else:
            self._suggested_experiments = self._suggested_experiments[len(x):]
