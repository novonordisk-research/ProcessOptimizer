import numpy as np
from ProcessOptimizer.space import Space

from .suggestor import BatchSuggestor


class DefaultSuggestor(BatchSuggestor):
    """
    Default suggestor class. It should only be used as a placeholder for use in
    strategizers. It should be replaced with the appropriate suggestor before use.
    """

    def __init__(self, space: Space, rng: np.random.Generator):
        # Space and random number generator are stored for use when replacing the
        # DefaultSuggestor.
        self.space = space
        self.rng = rng

    def suggest(self, _, __):
        raise NoDefaultSuggestorError("Default suggestor should not be used.")


class NoDefaultSuggestorError(NotImplementedError):
    """ Raised when a DefaultSuggestor is used when it should have been replaced."""
