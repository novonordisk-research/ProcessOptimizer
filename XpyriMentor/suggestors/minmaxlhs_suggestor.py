from typing import Iterable

import numpy as np
from skopt.sampler import Lhs
from ProcessOptimizer.space import Space

from .suggestor import IncompatibleNumberAsked


class MinMaxLHSSuggestor():
    def __init__(
            self,
            space: Space,
            rng: np.random.Generator,
            n_points: int = 5
    ):
        self.space = space
        self.rng = rng
        self.n_points = n_points
        self.cache = self.find_lhs_points()

    def find_lhs_points(self) -> np.ndarray:
        lhs = Lhs(criterion="maximin", iterations=10000)
        x = lhs.generate(self.space.bounds, self.n_points)
        return x

    def suggest(
        self, Xi: Iterable[Iterable], Yi: Iterable, n_asked: int = 1
    ) -> np.ndarray:
        return self.cache[len(Xi):len(Xi) + n_asked]
    
    def __str__(self):
        return f"Latin Hypercube Suggestor with {self.n_points} points"

    def __repr__(self):
        return f"LHSSuggestor(space={self.space}, rng={self.rng}, n_points={self.n_points})"
