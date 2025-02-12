import numpy as np

from ProcessOptimizer.space import space_factory
from XpyriMentor.suggestors import ConstantSuggestor, Suggestor, suggestor_factory

def test_initializaton():
    suggestor = ConstantSuggestor(space=space_factory([[1, 2], [1, 2]]))
    assert isinstance(suggestor, Suggestor)

def test_factory():
    space = space_factory([[1, 2], [1, 2]])
    suggestor = suggestor_factory(
        space=space,
        definition={"suggestor_name": "Constant"},
    )
    assert isinstance(suggestor, ConstantSuggestor)

def test_factory_settings():
    space = space_factory([[1, 2], [1, 2]])
    suggestor = suggestor_factory(
        space=space,
        definition={"suggestor_name": "Constant", "point": [1, 1]},
    )
    assert isinstance(suggestor, ConstantSuggestor)
    assert (suggestor.suggest([1,2], []) == np.array([[2,2]])).all()
    suggestor = suggestor_factory(
        space=space,
        definition={"suggestor_name": "Constant", "convert": False, "point": [1, 1]},
    )
    assert isinstance(suggestor, ConstantSuggestor)
    assert (suggestor.suggest([1,2], []) == np.array([[1,1]])).all()

def test_ask_multiple():
    space = space_factory([[1, 10], [1, 10]])
    suggestor = ConstantSuggestor(space)
    suggestions = suggestor.suggest([], [], n_asked=5)
    assert len(suggestions) == 5
    for suggestion in suggestions:
        assert suggestion in space
        assert len(suggestion) == 2
        assert (suggestion == suggestions[0]).all()