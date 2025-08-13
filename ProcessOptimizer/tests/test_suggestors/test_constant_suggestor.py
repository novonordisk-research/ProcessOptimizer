from ProcessOptimizer.space import space_factory
from ProcessOptimizer.XpyriMentor.suggestors import (
    CreatableSuggestor,
    ConstantSuggestor,
    Suggestor,
    suggestor_factory,
)


def test_initialization():
    space = space_factory([[0, 1], [0, 1]])
    suggestor = ConstantSuggestor(space=space, point=[0.5, 0.5])
    assert isinstance(suggestor, ConstantSuggestor)


def test_protocol():
    space = space_factory([[0, 1], [0, 1]])
    suggestor = ConstantSuggestor(space=space, point=[0.5, 0.5])
    assert isinstance(suggestor, Suggestor)
    assert isinstance(suggestor, CreatableSuggestor)


def test_given_value():
    space = space_factory([[0.0, 1.0]])
    suggestor = ConstantSuggestor(space=space, point=[0.2])
    assert suggestor.suggest(Xi=[], Yi=[], n_asked=1) == [[0.2]]


def test_different_given_values():
    space = space_factory([[0.0, 1.0], [0.0, 1.0]])
    suggestor = ConstantSuggestor(space=space, point=[0.2, 0.8])
    assert all(suggestor.suggest(Xi=[], Yi=[], n_asked=1)[0] == [0.2, 0.8])


def test_factory():
    space = space_factory([[-10.0, 10.0], [0, 10]])
    suggestor = suggestor_factory(
        space=space,
        definition={
            "suggestor_name": "Constant",
            "point": [0.2, 0.8],
        },
    )
    assert isinstance(suggestor, ConstantSuggestor)
    assert all(suggestor.suggest(Xi=[], Yi=[], n_asked=1)[0] == [0.2, 0.8])
