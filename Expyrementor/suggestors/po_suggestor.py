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

    def suggest(self, Xi: list[list], yi: list) -> list[list]:
        if Xi != self.optimizer.Xi or yi != self.optimizer.yi:
            self.optimizer.Xi = Xi
            self.optimizer.yi = yi
            self.optimizer.update_next()
        point = self.optimizer.ask()
        logger.debug(
            "Given Xi = %s and yi = %s, POSugggestor suggests the point: %s",
            Xi,
            yi,
            point,
        )
        return [point]
