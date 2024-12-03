import numpy as np
from ProcessOptimizer.space import Space


class DefaultSuggestor():
    """
    Default suggestor class. It should only be used as a placeholder for use in
    strategizers. It should be replaced with the appropriate suggestor before use.
    """

    def __init__(self, space: Space, n_objectives: int, rng: np.random.Generator, **kwargs):
        # Space and random number generator are stored for use when replacing the
        # DefaultSuggestor.
        self.space = space
        self.n_objectives = n_objectives
        self.rng = rng

    def suggest(self, Xi: list[list], Yi: list, n_asked: int = -1) -> list[list]:
        raise NoDefaultSuggestorError("Default suggestor should not be used.")


class NoDefaultSuggestorError(NotImplementedError):
    """ Raised when a DefaultSuggestor is used when it should have been replaced."""
