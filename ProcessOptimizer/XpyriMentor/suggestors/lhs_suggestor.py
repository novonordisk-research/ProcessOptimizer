from __future__ import annotations
from typing import Any, Iterable

import numpy as np
from ProcessOptimizer.space import Space

from .suggestor import IncompatibleNumberAsked


class LHSSuggestor:
    def __init__(self, space: Space, rng: np.random.Generator, n_points: int = 5):
        self.space = space
        self.rng = rng
        self.n_points = n_points
        self.cache = self.find_lhs_points()

    def find_lhs_points(self) -> np.ndarray:
        # Create a list of evenly distributed points in the range [0, 1] to sample from
        sample_indices = (np.arange(self.n_points) + 0.5) / self.n_points
        permuted_sample_indices = np.array(
            [self.rng.permutation(sample_indices) for _ in range(self.space.n_dims)],
            dtype=object,
        )
        # Permuted sample indices are now a list of n_dims arrays, each containing the
        # same n_points indices in a different order. We need to transpose this list to
        # get an array of n_points arrays, each containing the indices for one point.
        samples_indexed = permuted_sample_indices.T
        return self.space.sample(samples_indexed)

    def suggest(
        self, Xi: Iterable[Iterable], Yi: Iterable, n_asked: int = 1
    ) -> np.ndarray:
        if n_asked + len(Xi) > self.n_points:
            raise IncompatibleNumberAsked(
                "The number of points requested is greater than the number of points "
                "in the LHS cache."
            )
        return self.cache[len(Xi) : len(Xi) + n_asked]

    def __str__(self):
        return f"Latin Hypercube Suggestor with {self.n_points} points"

    def __repr__(self):
        return f"LHSSuggestor(space={self.space}, rng={self.rng}, n_points={self.n_points})"

    @classmethod
    def create_from_definition(
        cls,
        space: Space,
        suggestor_factory: callable,
        definition: dict[str, Any],
        n_objectives: int = 1,
        rng: np.random.Generator | None = None,
    ) -> LHSSuggestor:
        return LHSSuggestor(
            space=space,
            rng=rng,
            **definition,
        )
