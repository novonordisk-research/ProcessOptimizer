import numpy as np
from ProcessOptimizer.space import space_factory
from ProcessOptimizer.XpyriMentor.suggestors import (
    CreatableSuggestor,
    GoldenRatioSuggestor,
    Suggestor,
)


def test_initialization():
    space = space_factory([[0, 1], [0, 1]])
    suggestor = GoldenRatioSuggestor(space=space, rng=np.random.default_rng(1))
    assert isinstance(suggestor, GoldenRatioSuggestor)


def test_protocol():
    space = space_factory([[0, 1], [0, 1]])
    suggestor = GoldenRatioSuggestor(space=space, rng=np.random.default_rng(1))
    assert isinstance(suggestor, Suggestor)
    assert isinstance(suggestor, CreatableSuggestor)


def test_create_from_definition():
    space = space_factory([[0, 1], [0, 1]])
    suggestor = GoldenRatioSuggestor.create_from_definition(
        space=space,
        suggestor_factory=None,
        definition={},
        n_objectives=1,
        rng=np.random.default_rng(1),
    )
    assert isinstance(suggestor, GoldenRatioSuggestor)


def test_suggest():
    space = space_factory([[0, 10], [0.0, 1.0], ["cat", "dog"]])
    suggestor = GoldenRatioSuggestor(space, rng=np.random.default_rng(1))
    suggestions = suggestor.suggest([], [])
    assert len(suggestions) == 1
    assert suggestions[0] in space
    suggestions = suggestor.suggest([], [], n_points_to_suggest == 5)
    assert len(suggestions) == 5
    for suggestion in suggestions:
        assert suggestion in space
        assert len(suggestion) == 3
