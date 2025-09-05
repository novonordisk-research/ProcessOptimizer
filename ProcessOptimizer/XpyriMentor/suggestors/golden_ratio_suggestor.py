from __future__ import annotations
from typing import Iterable

import numpy as np
from ProcessOptimizer.space import Space


class GoldenRatioSuggestor:
    """
    A quasi-random, low-discrepancy sequence suggestor based on the generalized golden
    ratio.

    From https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    """

    def __init__(
        self,
        space: Space,
        rng: np.random.Generator,
        **kwargs,  # To catch any additional keyword arguments that might have been added
    ):
        self.space = space
        self.rng = rng
        self.offset = rng.uniform(0, 1, size=space.n_dims)

    @staticmethod
    def phi(d: int) -> float:
        """
        Calculate the generalized golden ratio for a given dimensionality d.

        phi(d) is the unique positive real root of the polynomial equation
        x**(d+1) = 1 + x. If follows that phi(d)**(d+1) = 1 + phi(d).
        """
        x = 2.0
        for _ in range(10):
            x = pow(1 + x, 1 / (d + 1))
        # The above loop converges to the root, as per the source (extreme learning
        # link) in the class docstring. Any irrational number works, and the slight
        # suboptimality from not having an exact value is not an issue for our use. We
        # could do it more directly, eg. with np.polynomial.Polynomial().roots, but this
        # gives us a list of roots, where we then have to find the unique
        # positive real root.
        return x

    def suggest(
        self, Xi: Iterable[Iterable], Yi: Iterable, n_points_to_suggest: int = 1
    ) -> np.ndarray:
        """
        Suggests a new point.

        Parameters
        ----------
        * Xi [`Iterable[Iterable]`]:
            The input is a list of already evaluated points. Only the number of
            points is used, not the actual values.
        * Yi [`Iterable`]:
            The results of the evaluations of `Xi`. Not used in this suggestor.
            Present for consistency with other suggestors and XPyriMentor.
        * n_points_to_suggest [`int`]:
            The number of suggested points to return. Must be a positive integer.
        Returns
        -------
        A np.ndarray of size `n_points_to_suggest` x `n_dim`, where `n_dim` is the number of
        dimensions in the search space. The points are sampled from the search space
        using a quasi-random sequence based on the generalized golden ratio.
        """
        d = self.space.n_dims
        g = self.phi(d)
        alpha = np.fromiter((pow(1 / g, j + 1) % 1 for j in range(d)), dtype=float)
        # Disregarding the already sampled points
        offset = np.array(self.offset + len(Xi) * alpha)
        x = np.fromiter(
            ((offset + alpha * (i + 1)) % 1 for i in range(n_points_to_suggest)),
            dtype=np.dtype((float, d)),
        )
        return self.space.sample(x)

    @classmethod
    def create_from_definition(
        cls,
        space: Space,
        suggestor_factory,
        definition,
        n_objectives,
        rng: np.random.Generator,
    ) -> GoldenRatioSuggestor:
        return GoldenRatioSuggestor(space=space, rng=rng)
