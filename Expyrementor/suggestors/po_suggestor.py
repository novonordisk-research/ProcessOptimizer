import logging

import numpy as np
from ProcessOptimizer import Optimizer
from ProcessOptimizer.space import Space

logger = logging.getLogger(__name__)


class POSuggestor:
    def __init__(self, space: Space, rng: np.random.Generator, **kwargs):
        self.optimizer = Optimizer(
            dimensions=space,
            random_state=np.random.RandomState(int(rng.random() * (2**32 - 1))),
            # ProcessOptimizer uses a legacy random state object, so we need to convert
            # the numpy random generator to a numpy random state object.
            **kwargs,
        )
        self.n_given = 1

    def suggest(self, Xi: list[list], yi: list, n_asked: int = 1) -> list[list]:
        if Xi != self.optimizer.Xi or yi != self.optimizer.yi:
            self.optimizer.Xi = Xi.copy()
            self.optimizer.yi = yi.copy()
            self.optimizer.update_next()
        point = self.optimizer.ask(n_asked)
        logger.debug(
            "Given Xi = %s and yi = %s, POSugggestor suggests the point: %s",
            Xi,
            yi,
            point,
        )
        return [point]
