import warnings

import numpy as np

from .default_suggestor import DefaultSuggestor, NoDefaultSuggestorError
from .suggestor import Suggestor


class RandomStragegizer():
    def __init__(
        self, suggestors: list[tuple[float, Suggestor]], rng: np.random.Generator
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

    def suggest(self, Xi: list[list], Yi: list, n_asked: int = 1) -> np.ndarray:
        # Creating n_asked random indices in the range [0, total)
        selector_indices = [
            relative_index*self.total for relative_index in self.rng.random(size=n_asked)
        ]
        suggested_points = []
        # Iterating over the suggestors, finding the number of suggested points for each
        # suggestor, and suggesting them.
        # Note that this will order the points in the order of the suggestors, the
        # suggestion will start with points from the first suggestor, then the second,
        # etc.
        for weight, suggestor in self.suggestors:
            selector_indices = [index-weight for index in selector_indices]
            n_suggested = sum(index < 0 for index in selector_indices)
            if n_suggested > 0:
                suggested_points.extend(suggestor.suggest(Xi, Yi, int(n_suggested)))
            selector_indices = [index for index in selector_indices if index >= 0]
        return np.array(suggested_points, dtype=object)

    def __str__(self):
        return "Random Strategizer with suggestors: " + ", ".join(
            suggestor.__class__.__name__ for _, suggestor in self.suggestors
        )