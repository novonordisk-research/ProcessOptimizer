import numpy as np
import pytest
from ProcessOptimizer.space import space_factory
from XpyriMentor.suggestors import (
    DefaultSuggestor,
    IncompatibleNumberAsked,
    LHSSuggestor,
    POSuggestor,
    SequentialStrategizer,
    Suggestor,
    suggestor_factory
)


class MockSuggestor:
    def __init__(self, suggestions: list):
        self.suggestions = suggestions
        self.last_input = {}

    def suggest(self, Xi, Yi, n_asked=1):
        self.last_input = {"Xi": Xi, "Yi": Yi}
        return self.suggestions*n_asked


def test_initialization():
    suggestor = SequentialStrategizer(
        suggestors=[(5, MockSuggestor([[1]])), (-1, MockSuggestor([[2]]))],
    )
    assert suggestor.suggestors[0][0] == 5
    assert suggestor.suggestors[1][0] == float("inf")
    assert suggestor.suggestors[0][1].suggest(Xi=None, Yi=None, n_asked=1) == [[1]]


def test_protocol():
    suggestor = SequentialStrategizer(
        suggestors=[(5, MockSuggestor([[1]])), (-1, MockSuggestor([[2]]))],
    )
    assert isinstance(suggestor, Suggestor)


def test_factory():
    suggestor = {
        "name": "Sequential",
        "suggestors": [
            {"suggestor_budget": 5, "suggestor": MockSuggestor([[1]])},
            {"suggestor_budget": -1, "suggestor": MockSuggestor([[2]])},
        ],
    }
    suggestor = suggestor_factory(
        space=None,
        definition=suggestor,
    )
    assert isinstance(suggestor, SequentialStrategizer)


def test_budget():
    suggestor = SequentialStrategizer(
        suggestors=[(5, MockSuggestor([[1]])), (-1, MockSuggestor([[2]]))],
    )
    assert suggestor.suggestors[0][0] == 5
    assert suggestor.suggestors[1][0] == float("inf")
    suggestor = SequentialStrategizer(
        suggestors=[(5, MockSuggestor([[1]])), (10, MockSuggestor([[2]]))],
    )
    assert suggestor.suggestors[0][0] == 5
    assert suggestor.suggestors[1][0] == 10
    with pytest.raises(IncompatibleNumberAsked):
        suggestor.suggest([[1]], [1], n_asked=16)
    with pytest.raises(ValueError):
        SequentialStrategizer(
            suggestors=[(-1, MockSuggestor([[1]])), (5, MockSuggestor([[2]]))],
        )


def test_suggest():
    suggestor_1 = MockSuggestor([[1]])
    suggestor_2 = MockSuggestor([[2]])
    suggestor_3 = MockSuggestor([[3]])
    suggestor = SequentialStrategizer(
        suggestors=[(3, suggestor_1), (2, suggestor_2), (-1, suggestor_3)],
    )
    assert suggestor.suggest([], []) == [[1]]
    assert all(suggestor.suggest([], [], n_asked=2) == [[1], [1]])
    assert all(suggestor.suggest([], [], n_asked=3) == [[1], [1], [1]])
    assert all(suggestor.suggest([], [], n_asked=4) == [[1], [1], [1], [2]])
    assert suggestor_1.last_input == {"Xi": [], "Yi": []}
    assert suggestor_2.last_input == {"Xi": [], "Yi": []}
    assert suggestor.suggest([[1]]*2, [1, 1]) == [[1]]
    assert all(suggestor.suggest([[1]]*2, [1, 1], n_asked=2) == [[1], [2]])
    assert all(suggestor.suggest([[1]]*2, [1, 1], n_asked=4) == [[1], [2], [2], [3]])
    assert suggestor_1.last_input == {"Xi": [[1], [1]], "Yi": [1, 1]}
    assert suggestor_2.last_input == {"Xi": [[1], [1]], "Yi": [1, 1]}
    assert suggestor_3.last_input == {"Xi": [[1], [1]], "Yi": [1, 1]}
    assert suggestor.suggest([[1]]*100, [1]*100) == [[3]]
    assert suggestor_3.last_input == {"Xi": [[1]]*100, "Yi": [1]*100}


def test_default_suggestors():
    rng = np.random.default_rng(1)
    space = space_factory([[0, 1], [0, 1]])
    suggestor = SequentialStrategizer(
        suggestors=[
            (3, DefaultSuggestor(space=space, n_objectives=1, rng=rng)),
            (2, DefaultSuggestor(space=space, n_objectives=1, rng=rng)),
            (-1, DefaultSuggestor(space=space, n_objectives=2, rng=rng))
        ],
    )
    assert isinstance(suggestor.suggestors[0][1], LHSSuggestor)
    assert suggestor.suggestors[0][1].n_points == 3
    assert isinstance(suggestor.suggestors[1][1], POSuggestor)
    assert suggestor.suggestors[1][1].optimizer.n_objectives == 1
    assert isinstance(suggestor.suggestors[2][1], POSuggestor)
    assert suggestor.suggestors[2][1].optimizer.n_objectives == 2


def test_incompatible_n_points():
    class NPointsSuggestor:
        def __init__(self, n_points):
            self.n_points = n_points

        def suggest(self, Xi, Yi, n_asked=1):
            return [[self.n_points]]*n_asked
    with pytest.warns(UserWarning):
        SequentialStrategizer(suggestors=[(5, NPointsSuggestor(10))])


def test_default_n_points():
    space = space_factory([[0, 1], [0, 1]])
    suggestor_definition = {
        "name": "Sequential",
        "suggestors": [{"suggestor_budget": 7, "name": "LHS"}],
    }
    suggestor = suggestor_factory(
        space=space,
        definition=suggestor_definition,
    )
    assert suggestor.suggestors[0][1].n_points == 7
