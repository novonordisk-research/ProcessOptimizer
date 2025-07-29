import logging
from typing import Any, Union, Optional

import numpy as np
from ProcessOptimizer.space import Space

from .constant_suggestor import ConstantSuggestor
from .default_suggestor import DefaultSuggestor
from .golden_ratio_suggestor import GoldenRatioSuggestor
from .lhs_suggestor import LHSSuggestor
from .po_suggestor import OptimizerSuggestor
from .random_strategizer import RandomStragegizer
from .sequential_strategizer import SequentialStrategizer
from .suggestor import CreatableSuggestor, Suggestor

logger = logging.getLogger(__name__)

SUGGESTORS = {
    "Constant": ConstantSuggestor,
    "Default": DefaultSuggestor,
    "LHS": LHSSuggestor,
    "PO": OptimizerSuggestor,
    "Optimizer": OptimizerSuggestor,
    "Random": RandomStragegizer,
    "Sequential": SequentialStrategizer,
    "GoldenRatio": GoldenRatioSuggestor,
}


def suggestor_factory(
    space: Space,
    definition: Union[Suggestor, dict[str, Any], None],
    n_objectives: int = 1,
    rng: Optional[np.random.Generator] = None,
    suggestors: dict[str, CreatableSuggestor] | None = None,
) -> Suggestor:
    """
    Create a suggestor from a definition dictionary.

    Definition is either a suggestor instance, a dict that specifies the suggestor type
    and its parameters, or None.

    If definiton is a suggestor instance it is returned as is.

    If definition is a dict, it is used to create a suggestor. The dictionary must have
    a 'name' key that specifies the type of suggestor. The other keys depend on the
    suggestor type. It can be recursive if the suggestor is a strategizer, that is, a
    suggestor that delegates ask() to other suggestors.

    If definition is None, a DefaultSuggestor is created. This is useful as a
    placeholder in strategizers, and should be replaced with a real suggestor before
    use.
    """
    if suggestors is None:
        suggestors = SUGGESTORS
    if isinstance(definition, Suggestor):
        return definition
    if rng is None:
        rng = np.random.default_rng(1)
    if not definition:  # If definition is None or empty, return DefaultSuggestor.
        suggestor_type = "Default"
    else:
        try:
            suggestor_type = definition.pop("suggestor_name")
        except KeyError as e:
            raise ValueError(
                f"Missing 'suggestor_name' key in suggestor definition: {definition}"
            ) from e
    if suggestor_type not in suggestors:
        raise ValueError(f"Unknown suggestor name: {suggestor_type}")
    return suggestors[suggestor_type].create_from_definition(
        space=space,
        suggestor_factory=suggestor_factory,
        definition=definition,
        n_objectives=n_objectives,
        rng=rng,
    )
