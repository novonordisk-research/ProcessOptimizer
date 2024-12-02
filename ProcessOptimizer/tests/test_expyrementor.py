from Expyrementor.expyrementor import Expyrementor
from Expyrementor.suggestors import InitialPointSuggestor, POSuggestor


class MockSuggestor:
    def __init__(self, suggestions: list):
        self.suggestions = suggestions
        self.last_input = {}

    def suggest(self, Xi, yi):
        self.last_input = {"Xi": Xi, "yi": yi}
        return self.suggestions


def test_initialization():
    space = [[0, 1], [0, 1]]
    exp = Expyrementor(space)
    assert exp.Xi == []
    assert exp.yi == []
    assert isinstance(exp.suggestor, InitialPointSuggestor)
    assert exp.suggestor.n_initial_points == 5
    assert isinstance(exp.suggestor.initial_suggestor, POSuggestor)
    assert exp.suggestor.initial_suggestor.optimizer._n_initial_points >= 5
    assert isinstance(exp.suggestor.ultimate_suggestor, POSuggestor)
    assert exp.suggestor.ultimate_suggestor.optimizer._n_initial_points == 0


def test_tell_single_objective():
    space = [[0, 1], [0, 1]]
    exp = Expyrementor(space)
    exp.tell([0.5, 0.5], 1)
    assert exp.Xi == [[0.5, 0.5]]
    assert exp.yi == [1]
    exp.tell([0.6, 0.6], 2)
    assert exp.Xi == [[0.5, 0.5], [0.6, 0.6]]
    assert exp.yi == [1, 2]


def test_tell_multiple_objectives():
    space = [[0, 1], [0, 1]]
    exp = Expyrementor(space)
    exp.tell([0.5, 0.5], [1, 2])
    assert exp.Xi == [[0.5, 0.5]]
    assert exp.yi == [[1, 2]]
    exp.tell([0.6, 0.6], [2, 3])
    assert exp.Xi == [[0.5, 0.5], [0.6, 0.6]]
    assert exp.yi == [[1, 2], [2, 3]]


def test_ask_single_return():
    space = [[0, 1], [0, 1]]
    exp = Expyrementor(space)
    exp.suggestor = MockSuggestor([[0.5, 0.5]])
    assert exp.ask() == [0.5, 0.5]


def test_ask_multiple_returns():
    space = [[0, 1], [0, 1]]
    exp = Expyrementor(space)
    exp.suggestor = MockSuggestor([[0.5, 0.5], [0.6, 0.6]])
    # exp will now get two suggestions from the suggestor, and only return the first one
    assert exp.ask() == [0.5, 0.5]
    # We will now replace the suggestor
    exp.suggestor = MockSuggestor([[0.7, 0.7]])
    # exp will now return the second suggestion from the first suggestor
    assert exp.ask() == [0.6, 0.6]
    # exp has now used all of the suggestions from the first suggestor, and will get a
    # new suggestion from the second suggestor
    assert exp.ask() == [0.7, 0.7]


def test_ask_passes_on_values():
    space = [[0, 1], [0, 1]]
    exp = Expyrementor(space)
    exp.suggestor = MockSuggestor([[0.5, 0.5]])
    exp.tell([0.6, 0.6], 2)
    exp.ask()
    assert exp.suggestor.last_input == {"Xi": [[0.6, 0.6]], "yi": [2]}
    exp.tell([0.7, 0.7], 3)
    exp.ask()
    assert exp.suggestor.last_input == {"Xi": [[0.6, 0.6], [0.7, 0.7]], "yi": [2, 3]}
