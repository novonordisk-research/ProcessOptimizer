import numpy as np
from Expyrementor.suggestors import RandomStragegizer, Suggestor


class MockSuggestor:
    def __init__(self, suggestions: list):
        self.suggestions = suggestions
        self.last_input = {}

    def suggest(self, Xi, Yi, n_asked=1):
        self.last_input = {"Xi": Xi, "Yi": Yi}
        return self.suggestions*n_asked


def test_random_strategizer():
    suggestor = RandomStragegizer(
        suggestors=[(0.8, MockSuggestor([[1]])), (0.2, MockSuggestor([[2]]))],
        n_objectives=1,
        rng=np.random.default_rng(1)
    )
    assert isinstance(suggestor, Suggestor)
    # np.random.default_rng(1).random() gives 0.5118216247148916, 0.9504636963259353,
    # and 0.14415961271963373 on the first three calls, so the first three calls
    # should return the suggestors with weights 0.8, 0.2, and 0.8, respectively.
    assert suggestor.suggest([], []) == [[1]]
    assert suggestor.suggest([], []) == [[2]]
    assert suggestor.suggest([], []) == [[1]]


def test_random_multiple_ask():
    suggestor = RandomStragegizer(
        suggestors=[(0.8, MockSuggestor([[1]])), (0.2, MockSuggestor([[2]]))],
        n_objectives=1,
        rng=np.random.default_rng(1)
    )
    assert suggestor.suggest([], [], n_asked=2) == [[1], [2]]
    assert suggestor.suggest([], [], n_asked=3) == [[1], [1], [2]]
