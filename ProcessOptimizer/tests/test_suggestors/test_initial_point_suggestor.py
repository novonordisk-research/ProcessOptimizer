import numpy as np
import pytest
from Expyrementor.suggestors import InitialPointSuggestor, DefaultSuggestor, POSuggestor


class MockSuggestor:
    def __init__(self, suggestions: list):
        self.suggestions = suggestions
        self.last_input = {}

    def suggest(self, Xi, yi):
        self.last_input = {"Xi": Xi, "yi": yi}
        return self.suggestions


def test_initialization():
    space = [[0, 1], [0, 1]]
    suggestor = InitialPointSuggestor(
        initial_suggestor=DefaultSuggestor(space, rng=np.random.default_rng(1)),
        ultimate_suggestor=DefaultSuggestor(space, rng=np.random.default_rng(2)),
    )
    assert suggestor.n_initial_points == 5
    # Initial suggestor is a POSuggestor with more than 5 initial points. This means than
    # we will only use the LHS part of this ProcessOptimizer.
    assert isinstance(suggestor.initial_suggestor, POSuggestor)
    assert suggestor.initial_suggestor.optimizer._n_initial_points == 5
    # Ultimate suggestor is a POSuggestor with no initial points, since
    # InitialPointSuggestor handles the initial points.
    assert isinstance(suggestor.ultimate_suggestor, POSuggestor)
    assert suggestor.ultimate_suggestor.optimizer._n_initial_points == 0


def test_suggestor_switch():
    suggestor = InitialPointSuggestor(
        initial_suggestor=MockSuggestor([1]),
        ultimate_suggestor=MockSuggestor([2]),
        n_initial_points=3,
    )
    assert suggestor.next_suggestor([], []) == suggestor.initial_suggestor
    assert suggestor.next_suggestor([1, 2], []) == suggestor.initial_suggestor
    assert suggestor.next_suggestor([1, 2, 3], []) == suggestor.ultimate_suggestor


def test_too_may_suggested_point():
    suggestor = InitialPointSuggestor(
        initial_suggestor=MockSuggestor([1, 2]),
        ultimate_suggestor=MockSuggestor([4]),
        n_initial_points=3,
    )
    with pytest.raises(ValueError):
        suggestor.next_suggestor([1, 2], [])
