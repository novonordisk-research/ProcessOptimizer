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

SUGGESTORS: dict[str, CreatableSuggestor] = {
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
    suggestors: Optional[dict[str, CreatableSuggestor]] = None,
) -> Suggestor:
    """
    Create a suggestor from a definition dictionary.

    Parameters
    ----------
    * space [`Space`]:
        The search space.
    * definition [`Union[Suggestor, dict[str, Any], None]`]:
        The definition of the suggestor to create.

        If it is a suggestor instance, it is returned as is. This is useful for creating
        suggestors that are composed of other suggestors (strategizers).

        If it is a dict, it is used to create the suggestor. The dictionary must have
        the key `suggestor_name` that specifies the type of suggestor. The other keys
        depend on the suggestor type. It can be recursive if the suggestor is a
        strategizer, that is, a suggestor that delegates ask() to other suggestors.

        If it is None, a DefaultSuggestor is created. This is useful as a placeholder in
        strategizers, and should be replaced with a real suggestor before use.
    * n_objectives [`int`]:
        The number of objectives for the suggestor.
    * rng [`Optional[np.random.Generator]`]:
        The random number generator to use. If None, a reproducible RNG is created.
    * suggestors [`dict[str, CreatableSuggestor] | None`]:
        A dictionary of available suggestors. The built-in suggestors will be added.
    """
    if suggestors is None:
        suggestors = {}
    # For the keys not present in suggestors, add the built-in suggestors
    for key, value in SUGGESTORS.items():
        suggestors.setdefault(key, value)
        # This also returns the value, which we don't use
    if isinstance(definition, Suggestor):
        return definition
    if rng is None:
        rng = np.random.default_rng(1)
    if not definition:  # If definition is None or empty, return DefaultSuggestor.
        definition = {"suggestor_name": "Default"}
    definition = definition.copy()
    # Copying to preserve original before we start popping keys
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
