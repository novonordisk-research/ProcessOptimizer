from Expyrementor.suggestors import (
    CachingStrategizer, DefaultSuggestor, IncompatibleNumberAsked
)
import pytest


class MockSuggestor:
    def __init__(self, suggestions: list):
        self.suggestions = suggestions
        self.last_input = {}

    def suggest(self, Xi, Yi, n_asked=-1):
        self.last_input = {"Xi": Xi, "Yi": Yi}
        return self.suggestions


def test_initialization():
    suggestor = CachingStrategizer(MockSuggestor([[1]]))
    assert suggestor.ask_times_left == 1


def test_suggest():
    suggestor = CachingStrategizer(MockSuggestor([[1]]))
    assert suggestor.suggest([], []) == [[1]]


def test_ask_times():
    suggestor = CachingStrategizer(MockSuggestor([[1]]))
    assert suggestor.suggest([], []) == [[1]]
    with pytest.raises(IncompatibleNumberAsked):
        suggestor.suggest([], [])

    suggestor = CachingStrategizer(MockSuggestor([[1]]), ask_times=2)
    assert suggestor.suggest([], []) == [[1]]
    assert suggestor.suggest([], []) == [[1]]
    with pytest.raises(IncompatibleNumberAsked):
        suggestor.suggest([], [])


def test_cache():
    suggestor = CachingStrategizer(MockSuggestor([[1], [2]]))
    assert suggestor.suggest([], []) == [[1]]
    suggestor.suggestor = MockSuggestor([])
    assert suggestor.suggest([], []) == [[2]]
    with pytest.raises(IncompatibleNumberAsked):
        suggestor.suggest([], [])


def test_num_asked():
    suggestor = CachingStrategizer(MockSuggestor([[1], [2], [3]]), ask_times=2)
    assert suggestor.suggest([], [], n_asked=2) == [[1], [2]]
    assert suggestor.suggest([], [], n_asked=2) == [[3], [1]]
    assert suggestor.suggest([], [], n_asked=2) == [[2], [3]]
    with pytest.raises(IncompatibleNumberAsked):
        suggestor.suggest([], [], n_asked=2)


def test_default_suggestor():
    with pytest.raises(ValueError):
        CachingStrategizer(DefaultSuggestor(space=[[0, 1]], n_objectives=1, rng=None))
