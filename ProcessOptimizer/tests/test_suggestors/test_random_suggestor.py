import numpy as np
import pytest
from Expyrementor.suggestors import RandomStragegizer, Suggestor


class MockSuggestor:
    def __init__(self, suggestions: list):
        self.suggestions = suggestions
        self.last_input = {}

    def suggest(self, Xi, yi):
        self.last_input = {"Xi": Xi, "yi": yi}
        return self.suggestions


def test_random_strategizer():
    suggestor = RandomStragegizer(
        suggestors=[(0.8, MockSuggestor([])), (0.2, MockSuggestor([]))],
        n_objectives=1,
        rng=np.random.default_rng(1)
    )
    assert isinstance(suggestor, Suggestor)
    # np.random.default_rng(1).random() gives 0.5118216247148916, 0.9504636963259353,
    # and 0.14415961271963373 on the first three calls, so the first three calls
    # should return the suggestors with weights 0.8, 0.2, and 0.8, respectively.
    assert suggestor.next_suggestor([], []) == suggestor.suggestors[0][1]
    assert suggestor.next_suggestor([], []) == suggestor.suggestors[1][1]
    assert suggestor.next_suggestor([], []) == suggestor.suggestors[0][1]
