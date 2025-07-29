from __future__ import annotations
from typing import Iterable

import numpy as np
from ProcessOptimizer.space import Space


class DefaultSuggestor:
    """
    Default suggestor class. It should only be used as a placeholder for use in
    strategizers. It should be replaced with the appropriate suggestor before use.
    """

    def __init__(
        self, space: Space, n_objectives: int, rng: np.random.Generator, **kwargs
    ):
        # Space and random number generator are stored for use when replacing the
        # DefaultSuggestor.
        self.space = space
        self.n_objectives = n_objectives
        self.rng = rng

    def suggest(
        self, Xi: Iterable[Iterable], Yi: Iterable, n_asked: int = -1
    ) -> np.ndarray:
        """
        SHOULD NOT BE CALLED!

        DefaultSuggestor only exists to act as a placeholder to be replaced with the
        appropriate Suggestor. If its suggest method is called, this has not happened,
        which is an error.
        """
        raise NoDefaultSuggestorError("Default suggestor should not be used.")

    @classmethod
    def create_from_definition(
        cls,
        space: Space,
        suggestor_factory,
        definition,
        n_objectives: int,
        rng: np.random.Generator,
    ) -> DefaultSuggestor:
        return cls(
            space=space, n_objectives=n_objectives, rng=rng or np.random.default_rng()
        )


class NoDefaultSuggestorError(NotImplementedError):
    """Raised when a DefaultSuggestor is used when it should have been replaced."""
