import numpy as np
import pytest
from ProcessOptimizer.space import space_factory
from ProcessOptimizer.XpyriMentor.suggestors import LHSSuggestor, Suggestor, suggestor_factory, IncompatibleNumberAsked


def test_initializaton():
    suggestor = LHSSuggestor(
        space=space_factory([[1, 2], [1, 2]]),
        rng=np.random.default_rng(1),
    )
    assert isinstance(suggestor, Suggestor)
    assert suggestor.n_points == 5
    assert len(suggestor.cache) == 5


def test_factory():
    space = space_factory([[1, 2], [1, 2]])
    suggestor = suggestor_factory(
        space=space,
        definition={"suggestor_name": "LHS", "n_points": 10},
    )
    assert isinstance(suggestor, LHSSuggestor)
    assert suggestor.n_points == 10


def test_suggest():
    space = space_factory([[0, 10], [0.0, 1.0], ["cat", "dog"]])
    suggestor = LHSSuggestor(
        space, rng=np.random.default_rng(1), n_points=5
    )
    suggestions = suggestor.suggest([], [])
    assert len(suggestions) == 1
    assert len(suggestions[0]) == 3
    assert suggestions[0] in space
    suggestions = suggestor.suggest([], [], n_asked=5)
    # Testing that the values for each dimension is regularly spaced over the range
    assert set(suggestion[0] for suggestion in suggestions) == {1, 3, 5, 7, 9}
    assert set(suggestion[1] for suggestion in suggestions) == {0.1, 0.3, 0.5, 0.7, 0.9}
    assert set(suggestion[2] for suggestion in suggestions) == {"cat", "dog"}


def test_suggest_too_many():
    space = space_factory([[0, 10], [0.0, 1.0], ["cat", "dog"]])
    suggestor = LHSSuggestor(
        space, rng=np.random.default_rng(1), n_points=5
    )
    for n_told in range(1, 5):
        told = suggestor.suggest([], [], n_asked=n_told)
        suggestor.suggest(told, [0]*n_told, n_asked=1)
    with pytest.raises(IncompatibleNumberAsked):
        suggestor.suggest([], [], n_asked=6)
    with pytest.raises(IncompatibleNumberAsked):
        suggestor.suggest([[1, 0.0, "cat"]], [1], n_asked=5)


def test_n():
    space = space_factory([[0, 10], [0.0, 1.0], ["cat", "dog"]])
    for n in range(1, 10):
        suggestor = LHSSuggestor(space, rng=np.random.default_rng(1), n_points=n)
        assert suggestor.n_points == n
        assert len(suggestor.cache) == n
