import numpy as np
import pytest
import warnings
from XpyriMentor.suggestors import (
    RandomStragegizer,
    Suggestor,
    suggestor_factory,
    POSuggestor,
    LHSSuggestor,
    DefaultSuggestor
)
from XpyriMentor.suggestors.default_suggestor import NoDefaultSuggestorError
from ProcessOptimizer.space import space_factory


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
        rng=np.random.default_rng(1)
    )
    assert isinstance(suggestor, Suggestor)
    # np.random.default_rng(1).random() gives 0.5118216247148916, 0.9504636963259353,
    # and 0.14415961271963373 on the first three calls, so the first three calls
    # should return the suggestors with weights 0.8, 0.2, and 0.8, respectively.
    assert suggestor.suggest([], []) == [[1]]
    assert suggestor.suggest([], []) == [[2]]
    assert suggestor.suggest([], []) == [[1]]


def test_factory():
    space = space_factory([[0, 1], [0, 1]])
    suggestor = suggestor_factory(
        space=space,
        definition={"suggestor_name": "Random", "suggestors": [
            {"suggestor_usage_ratio": 0.8, "suggestor_name": "PO"},
            {"suggestor_usage_ratio": 0.2, "suggestor_name": "LHS"},]},
    )
    assert isinstance(suggestor, RandomStragegizer)
    assert len(suggestor.suggestors) == 2
    assert suggestor.suggestors[0][0] == 0.8
    assert suggestor.suggestors[1][0] == 0.2
    assert isinstance(suggestor.suggestors[0][1], POSuggestor)
    assert isinstance(suggestor.suggestors[1][1], LHSSuggestor)


def test_random_multiple_ask():
    suggestor = RandomStragegizer(
        suggestors=[(0.8, MockSuggestor([[1]])), (0.2, MockSuggestor([[2]]))],
        rng=np.random.default_rng(1)
    )
    assert all(suggestor.suggest([], [], n_asked=2) == [[1], [2]])
    assert all(suggestor.suggest([], [], n_asked=3) == [[1], [1], [2]])


def test_default_suggestor():
    with pytest.raises(NoDefaultSuggestorError):
        RandomStragegizer(
            suggestors=[
                (0.8, MockSuggestor([[1]])),
                (0.2, DefaultSuggestor(space=[], n_objectives=1, rng=None))
            ],
            rng=np.random.default_rng(1)
        )


def test_wrong_sum():
    with pytest.warns(UserWarning):
        # Warning if the sum of usage ratios is not 1 or 100
        RandomStragegizer(
            suggestors=[(0.8, MockSuggestor([[1]])), (0.3, MockSuggestor([[2]]))],
            rng=np.random.default_rng(1)
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # No warnings if the sum of usage ratios is 1
        RandomStragegizer(
            suggestors=[(0.8, MockSuggestor([[1]])), (0.2, MockSuggestor([[2]]))],
            rng=np.random.default_rng(1)
        )
        # No warnings if the sum of usage ratios is 100
        RandomStragegizer(
            suggestors=[(80, MockSuggestor([[1]])), (20, MockSuggestor([[2]]))],
            rng=np.random.default_rng(1)
        )


def test_random_with_suggestor_given():
    space = space_factory([[0, 1], [0, 1]])
    suggestor = suggestor_factory(
        space=space,
        definition={"suggestor_name": "Random", "suggestors": [
            {"suggestor_usage_ratio": 0.8, "suggestor": MockSuggestor([[1]])},
            {"suggestor_usage_ratio": 0.2, "suggestor": MockSuggestor([[2]])},]},
    )
    assert isinstance(suggestor, RandomStragegizer)
    assert len(suggestor.suggestors) == 2
    assert suggestor.suggestors[0][0] == 0.8
    assert suggestor.suggestors[1][0] == 0.2
    assert isinstance(suggestor.suggestors[0][1], MockSuggestor)
    assert isinstance(suggestor.suggestors[1][1], MockSuggestor)


def test_random_with_suggestor_given_wrong_keys():
    space = space_factory([[0, 1], [0, 1]])
    with pytest.raises(ValueError):
        suggestor_factory(
            space=space,
            definition={"name": "Random", "suggestors": [
                {"suggestor_usage_ratio": 0.8, "suggestor": MockSuggestor([[1]])},
                {
                    "suggestor_usage_ratio": 0.2,
                    "suggestor": MockSuggestor([[2]]),
                    "additional_key": "Can't have this key",
                },]},
        )