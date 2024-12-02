import warnings

import numpy as np

from .default_suggestor import DefaultSuggestor, NoDefaultSuggestorError
from .suggestor import Suggestor
from .strategizer import Strategizer


class RandomStragegizer(Strategizer):
    def __init__(
            self, suggestors: list[tuple[float, Suggestor]], n_objectives, rng: np.random.Generator
        ):
        self.total = sum(item[0] for item in suggestors)
        if float(self.total) != 1.0 and float(self.total) != 100.0:
            warnings.warn(
                "Probabilities do not sum to 1.0 or 100.0. They will be normalized."
            )
        if any(
            isinstance(item[1], DefaultSuggestor) for item in suggestors
        ):
            raise NoDefaultSuggestorError(
                "No DefaultSuggestor defined for RandomStrategizer."
            )
        self.suggestors = suggestors
        self.rng = rng

    def next_suggestor(self, _, __) -> Suggestor:
        random_number = self.rng.random()*self.total
        for weight, suggestor in self.suggestors:
            if random_number < weight:
                return suggestor
            random_number -= weight
