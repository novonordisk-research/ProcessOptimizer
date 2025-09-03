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
