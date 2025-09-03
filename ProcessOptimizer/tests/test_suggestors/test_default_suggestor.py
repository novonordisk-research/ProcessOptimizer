import numpy as np
from ProcessOptimizer.space import space_factory
from ProcessOptimizer.XpyriMentor.suggestors import (
    CreatableSuggestor,
    DefaultSuggestor,
    Suggestor,
)


def test_initialization():
    space = space_factory([[0, 1], [0, 1]])
    suggestor = DefaultSuggestor(
        space=space, n_objectives=1, rng=np.random.default_rng(1)
    )
    assert isinstance(suggestor, DefaultSuggestor)


def test_protocol():
    space = space_factory([[0, 1], [0, 1]])
    suggestor = DefaultSuggestor(
        space=space, n_objectives=1, rng=np.random.default_rng(1)
    )
    assert isinstance(suggestor, Suggestor)
    assert isinstance(suggestor, CreatableSuggestor)


def test_create_from_definition():
    space = space_factory([[0, 1], [0, 1]])
    suggestor = DefaultSuggestor.create_from_definition(
        space=space,
        suggestor_factory=None,
        definition={},
        n_objectives=1,
        rng=np.random.default_rng(1),
    )
    assert isinstance(suggestor, DefaultSuggestor)
