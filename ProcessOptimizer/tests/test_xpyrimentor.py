import pytest
from ProcessOptimizer import XpyriMentor
from ProcessOptimizer.XpyriMentor.suggestors import OptimizerSuggestor, LHSSuggestor, SequentialStrategizer


class MockSuggestor:
    def __init__(self, suggestions: list):
        self.suggestions = suggestions
        self.last_input = {}

    def suggest(self, Xi, Yi, n_asked=1):
        self.last_input = {"Xi": Xi, "Yi": Yi}
        return self.suggestions[:n_asked]


def test_initialization():
    space = [[0, 1], [0, 1]]
    exp = XpyriMentor(space)
    assert exp.Xi == []
    assert exp.yi == []
    assert isinstance(exp.suggestor, SequentialStrategizer)
    assert isinstance(exp.suggestor.suggestors[0][1], LHSSuggestor)
    assert exp.suggestor.suggestors[0][0] == 5
    assert isinstance(exp.suggestor.suggestors[1][1], OptimizerSuggestor)
    assert exp.suggestor.suggestors[1][0] == float("inf")


def test_tell_single_objective():
    space = [[0, 1], [0, 1]]
    exp = XpyriMentor(space)
    exp.tell([0.5, 0.5], 1)
    assert exp.Xi == [[0.5, 0.5]]
    assert exp.yi == [1]
    exp.tell([0.6, 0.6], 2)
    assert exp.Xi == [[0.5, 0.5], [0.6, 0.6]]
    assert exp.yi == [1, 2]


def test_tell_multiple_objectives():
    space = [[0, 1], [0, 1]]
    exp = XpyriMentor(space)
    exp.tell([0.5, 0.5], [1, 2])
    assert exp.Xi == [[0.5, 0.5]]
    assert exp.yi == [[1, 2]]
    exp.tell([0.6, 0.6], [2, 3])
    assert exp.Xi == [[0.5, 0.5], [0.6, 0.6]]
    assert exp.yi == [[1, 2], [2, 3]]


def test_ask_single_return():
    space = [[0, 1], [0, 1]]
    exp = XpyriMentor(space)
    exp.suggestor = MockSuggestor([[0.5, 0.5]])
    assert exp.ask() == [[0.5, 0.5]]


def test_ask_multiple_returns():
    space = [[0, 1], [0, 1]]
    exp = XpyriMentor(space)
    exp.suggestor = MockSuggestor([[0.5, 0.5], [0.6, 0.6]])
    # exp will now get two suggestions from the suggestor, and only return the first one
    assert exp.ask() == [[0.5, 0.5]]
    # We will now replace the suggestor
    exp.suggestor = MockSuggestor([[0.7, 0.7]])
    # exp has now used all of the suggestions from the first suggestor, and will get a
    # new suggestion from the second suggestor
    assert exp.ask() == [[0.7, 0.7]]


def test_ask_passes_on_values():
    space = [[0, 1], [0, 1]]
    exp = XpyriMentor(space)
    exp.suggestor = MockSuggestor([[0.5, 0.5]])
    exp.tell([0.6, 0.6], 2)
    exp.ask()
    assert exp.suggestor.last_input == {"Xi": [[0.6, 0.6]], "Yi": [2]}
    exp.tell([0.7, 0.7], 3)
    exp.ask()
    assert exp.suggestor.last_input == {"Xi": [[0.6, 0.6], [0.7, 0.7]], "Yi": [2, 3]}


def test_ask_multiple():
    space = [[0, 1], [0, 1]]
    exp = XpyriMentor(space)
    exp.suggestor = MockSuggestor([[0.5, 0.5], [0.6, 0.6]])
    assert exp.ask(2) == [[0.5, 0.5], [0.6, 0.6]]


def test_warning_if_raw_POSuggestor():
    with pytest.warns(UserWarning):
        XpyriMentor([[0, 1], [0, 1]], suggestor={"suggestor_name": "Optimizer"})
