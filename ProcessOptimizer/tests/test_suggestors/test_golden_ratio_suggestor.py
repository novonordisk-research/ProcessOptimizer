import numpy as np
from ProcessOptimizer.space import space_factory
from XpyriMentor.suggestors import GoldenRatioSuggestor, Suggestor, suggestor_factory

def test_initializaton():
    suggestor = GoldenRatioSuggestor(
        space=space_factory([[1, 2], [1, 2]]),
        rng=np.random.default_rng(1),
    )
    assert isinstance(suggestor, Suggestor)

def test_factory():
    space = space_factory([[1, 2], [1, 2]])
    suggestor = suggestor_factory(
        space=space,
        definition={"suggestor_name": "GoldenRatio"},
    )
    assert isinstance(suggestor, GoldenRatioSuggestor)

def test_suggest():
    space = space_factory([[0, 10], [0.0, 1.0], ["cat", "dog"]])
    suggestor = GoldenRatioSuggestor(
        space, rng=np.random.default_rng(1)
    )
    suggestions = suggestor.suggest([], [])
    assert len(suggestions) == 1
    assert suggestions[0] in space
    suggestions = suggestor.suggest([], [], n_asked=5)
    assert len(suggestions) == 5
    for suggestion in suggestions:
        assert suggestion in space
        assert len(suggestion) == 3