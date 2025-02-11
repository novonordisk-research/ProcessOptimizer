from typing import Iterable

import numpy as np
from ProcessOptimizer.space import Space

class GoldenRatioSuggestor():
    def __init__(self,space: Space,rng: np.random.Generator):
        self.space = space
        self.rng = rng
        self.offset = self.rng.random()

    @staticmethod
    def phi(d):
        """
        Calculate the generalized golden ratio for a given dimensionality d.

        It holds that phi(d)**(d+1) = 1 + phi(d).
        """
        x = 2.0
        for _ in range(10): 
            x = pow(1+x,1/(d+1)) 
        return x

    def suggest(
            self, Xi: Iterable[Iterable], _: Iterable, n_asked: int = 1
    ) -> np.ndarray:
        d = self.space.n_dims
        g = self.phi(d)
        alpha = np.fromiter((pow(1/g, j+1) %1 for j in range(d)), dtype=float)
        offset = np.array([self.offset]*d + len(Xi)*alpha)
        x = np.fromiter(
            ((offset + alpha*(i+1)) %1 for i in range(n_asked)),
            dtype = np.dtype((float,d)),
        )
        return self.space.sample(x)
