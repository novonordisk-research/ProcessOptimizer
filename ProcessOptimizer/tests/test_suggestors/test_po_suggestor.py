import numpy as np
from ProcessOptimizer.XpyriMentor.suggestors import (
    CreatableSuggestor,
    OptimizerSuggestor,
    suggestor_factory,
    Suggestor,
)
from ProcessOptimizer.space import space_factory


def test_initialization():
    space = space_factory([[0, 1], [0, 1]])
    suggestor = OptimizerSuggestor(space, rng=np.random.default_rng(1))
    assert isinstance(suggestor, OptimizerSuggestor)
    assert isinstance(suggestor, Suggestor)
    assert isinstance(suggestor, CreatableSuggestor)
    assert suggestor.optimizer._n_initial_points == 0
    nonstandard_suggestor = OptimizerSuggestor(
        space, rng=np.random.default_rng(1), n_initial_points=5
    )
    assert nonstandard_suggestor.optimizer._n_initial_points == 5


def test_factory():
    space = space_factory([[0, 1], [0, 1]])
    suggestor = suggestor_factory(
        space=space,
        definition={"suggestor_name": "Optimizer"},
    )
    assert isinstance(suggestor, OptimizerSuggestor)
    assert suggestor.optimizer._n_initial_points == 0


def test_suggest():
    space = space_factory([[0, 1], [0, 1]])
    suggestor = OptimizerSuggestor(space, rng=np.random.default_rng(1))
    suggestions = suggestor.suggest([[1, 1]], [1])
    assert len(suggestions) == 1
    assert len(suggestions[0]) == 2
    assert suggestions[0] in space
    suggestions = suggestor.suggest([[1, 1]], [1], n_asked=5)
    assert len(suggestions) == 5
    for suggestion in suggestions:
        assert suggestion in space
