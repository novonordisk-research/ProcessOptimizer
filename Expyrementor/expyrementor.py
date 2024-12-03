import logging
from typing import Any, Union

import numpy as np
from ProcessOptimizer.space import space_factory, Space
from ProcessOptimizer.utils import is_2Dlistlike
from ProcessOptimizer.utils.get_rng import get_random_generator

from .suggestors import DefaultSuggestor, Suggestor, suggestor_factory

logger = logging.getLogger(__name__)

DEFAULT_SUGGESTOR = {
    "name": "InitialPoint",
    "initial_suggestor": {"name": "Default"},
    "ultimate_suggestor": {"name": "Default"},
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
            suggestor = suggestor_factory(
                space, DEFAULT_SUGGESTOR.copy(), n_objectives, rng=rng
            )
        self.suggestor = suggestor
        self.Xi: list[list] = []
        # This is a list of points in the search space. Each point is a list of values for
        # each dimension of the search space.
        self.yi: list = []
        # We are agnostic to the type of the scores. They can be a float for single
        # objective optimization or a list of floats for multiobjective optimization.
        pass

    def ask(self, n: int = 1):
        """
        Ask the suggestor for new points to evaluate. The number of points to ask is
        specified by the argument n. The method returns a list of new points to evaluate.
        """
        return self.suggestor.suggest(Xi=self.Xi, Yi=self.yi, n_asked=n)

    def tell(self, x: list, y: Any):
        if is_2Dlistlike(x):
            # If x is a list of points, we assume that y is a list of scores of the same
            # length, and we add the members of x and y to the lists Xi and yi.
            self.Xi.extend(x)
            self.yi.extend(y)
        else:
            # If x is a single point, we assume that y is a single score, and we add x
            # and y to the lists Xi and yi.
            self.Xi.append(x)
            self.yi.append(y)

    def __str__(self):
        return f"Expyrementor with suggestor {self.suggestor}"
