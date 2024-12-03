import numpy as np
from ProcessOptimizer.space import space_factory
from Expyrementor.suggestors import LHSSuggestor


def test_initializaton():
    LHSSuggestor(space=[], rng=None, n_points=5)


def test_suggest():
    space = space_factory([[0, 10], [0.0, 1.0], ["cat", "dog"]])
    suggestor = LHSSuggestor(space, rng=np.random.default_rng(1), n_points=5)
    suggestions = suggestor.suggest([], [])
    assert len(suggestions) == 5
    assert set(suggestion[0] for suggestion in suggestions) == {1, 3, 5, 7, 9}
    assert set(suggestion[1] for suggestion in suggestions) == {0.1, 0.3, 0.5, 0.7, 0.9}
    assert set(suggestion[2] for suggestion in suggestions) == {"cat", "dog"}
