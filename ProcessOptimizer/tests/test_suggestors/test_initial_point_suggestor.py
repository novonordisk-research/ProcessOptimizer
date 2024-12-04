import numpy as np
from Expyrementor.suggestors import (
    suggestor_factory,
    InitialPointStrategizer,
    DefaultSuggestor,
    POSuggestor,
    Suggestor,
    LHSSuggestor,
)
from ProcessOptimizer.space import space_factory


class MockSuggestor:
    def __init__(self, suggestions: list):
        self.suggestions = suggestions
        self.last_input = {}

    def suggest(self, Xi, Yi, n_asked=1):
        self.last_input = {"Xi": Xi, "Yi": Yi}
        return self.suggestions[:n_asked]


def test_initialization():
    space = space_factory([[0, 1], [0, 1]])
    suggestor = InitialPointStrategizer(
        initial_suggestor=DefaultSuggestor(space, n_objectives=1, rng=np.random.default_rng(1)),
        ultimate_suggestor=DefaultSuggestor(space, n_objectives=1, rng=np.random.default_rng(2)),
    )
    assert isinstance(suggestor, Suggestor)
    assert suggestor.n_initial_points == 5
    # Initial suggestor is a POSuggestor with more than 5 initial points. This means than
    # we will only use the LHS part of this ProcessOptimizer.
    assert isinstance(suggestor.initial_suggestor, LHSSuggestor)
    assert suggestor.initial_suggestor.n_points == 5
    # Ultimate suggestor is a POSuggestor with no initial points, since
    # InitialPointSuggestor handles the initial points.
    assert isinstance(suggestor.ultimate_suggestor, POSuggestor)
    assert suggestor.ultimate_suggestor.optimizer._n_initial_points == 0


def test_factory():
    space = space_factory([[0, 1], [0, 1]])
    suggestor = suggestor_factory(
        space=space,
        definition={"name": "InitialPoint", "n_initial_points": 10},
    )
    assert isinstance(suggestor, InitialPointStrategizer)
    assert suggestor.n_initial_points == 10
    assert isinstance(suggestor.initial_suggestor, LHSSuggestor)
    assert suggestor.initial_suggestor.n_points == 10
    assert isinstance(suggestor.ultimate_suggestor, POSuggestor)
    assert suggestor.ultimate_suggestor.optimizer._n_initial_points == 0


def test_suggestor_switch():
    suggestor = InitialPointStrategizer(
        initial_suggestor=MockSuggestor([[1]]),
        ultimate_suggestor=MockSuggestor([[2]]),
        n_initial_points=3,
    )
    assert suggestor.suggest([], []) == [[1]]
    assert suggestor.suggest([1], []) == [[1]]
    assert suggestor.suggest([1, 2], []) == [[1]]
    assert suggestor.suggest([1, 2, 3], []) == [[2]]


def test_bridging_the_switch():
    suggestor = InitialPointStrategizer(
        initial_suggestor=MockSuggestor([[1], [1]]),
        ultimate_suggestor=MockSuggestor([[2]]),
        n_initial_points=2,
    )
    assert suggestor.suggest([], [], n_asked=2) == [[1], [1]]
    assert suggestor.suggest([1], [], n_asked=2) == [[1], [2]]


def test_multiple_initial():
    suggestor = InitialPointStrategizer(
        initial_suggestor=MockSuggestor([[1], [1]]),
        ultimate_suggestor=MockSuggestor([[2]]),
        n_initial_points=2,
    )
    assert suggestor.suggest([], []) == [[1]]
    assert suggestor.suggest([1], []) == [[1]]
    assert suggestor.suggest([1, 1], []) == [[2]]
