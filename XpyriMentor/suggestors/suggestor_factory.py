import logging
from typing import Any, Union, Optional

import numpy as np
from ProcessOptimizer.space import Space

from .default_suggestor import DefaultSuggestor
from .golden_ratio_suggestor import GoldenRatioSuggestor
from .lhs_suggestor import LHSSuggestor
from .po_suggestor import POSuggestor
from .random_strategizer import RandomStragegizer
from .sequential_strategizer import SequentialStrategizer
from .suggestor import Suggestor

logger = logging.getLogger(__name__)


def suggestor_factory(
    space: Space,
    definition: Union[Suggestor, dict[str, Any], None],
    n_objectives: int = 1,
    rng: Optional[np.random.Generator] = None,
    n_points: Optional[int] = None
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
    if isinstance(definition, Suggestor):
        return definition
    if rng is None:
        rng = np.random.default_rng(1)
    elif not definition:  # If definition is None or empty, return DefaultSuggestor.
        logger.debug("Creating DefaultSuggestor")
        return DefaultSuggestor(space, n_objectives, rng)
    try:
        definition = definition.copy()
        suggestor_type = definition.pop("suggestor_name")
    except KeyError as e:
        raise ValueError(
            f"Missing 'suggestor_name' key in suggestor definition: {definition}"
        ) from e
    if suggestor_type == "Default" or suggestor_type is None:
        logger.debug("Creating DefaultSuggestor")
        return DefaultSuggestor(space, n_objectives, rng)
    elif suggestor_type == "PO":
        logger.debug("Creating POSuggestor")
        return POSuggestor(
            space=space,
            n_objectives=n_objectives,
            rng=rng,
            **definition,
        )
    elif suggestor_type == "RandomStrategizer" or suggestor_type == "Random":
        logger.debug("Creating RandomStrategizer")
        suggestors = []
        for suggestor in definition["suggestors"]:
            usage_ratio = suggestor.pop("suggestor_usage_ratio")
            # Note that we are removing the key usage_ratio from the suggestor
            # definition. If any suggestor uses this key, it will have to be redefined in
            # the suggestor definition.
            if "suggestor" in suggestor:
                if len(suggestor) > 1:
                    raise ValueError(
                        "If a suggestor definition for a RandomStrategizer has a "
                        "'suggestor' key, it should only have that key and "
                        "'usage_ratio', but it has the keys `usage_ratio`, "
                        f"{suggestor.keys()}."
                    )
                suggestor = suggestor["suggestor"]
            suggestors.append((
                usage_ratio, suggestor_factory(space, suggestor, n_objectives, rng)
            ))
        return RandomStragegizer(suggestors=suggestors, rng=rng)
    elif suggestor_type == "LHS":
        if "n_points" not in definition and n_points is not None:
            definition["n_points"] = n_points
        logger.debug("Creating a LHSSuggestor.")
        return LHSSuggestor(
            space=space,
            rng=rng,
            **definition,
        )
    elif suggestor_type == "Sequential":
        logger.debug("Creating SequentialStrategizer")
        suggestors = []
        for suggestor in definition["suggestors"]:
            suggestor: dict = suggestor.copy()
            n = suggestor.pop("suggestor_budget")
            if "suggestor" in suggestor:
                if len(suggestor) > 1:
                    raise ValueError(
                        "If a suggestor definition for a SequentialStrategizer has a "
                        "'suggestor' key, it should only have that key and "
                        "'suggestor_budget', but it has the keys `suggestor_budget`, "
                        f"{', '.join(suggestor.keys())}."
                    )
                suggestor = suggestor["suggestor"]
            suggestors.append((
                n,
                suggestor_factory(space, suggestor, n_objectives, rng, n_points = n)
            ))
        return SequentialStrategizer(suggestors)
    elif suggestor_type == "GoldenRatio":
        logger.debug("Creating GoldenRatioSuggestor")
        return GoldenRatioSuggestor(
            space=space,
            rng=rng,
            **definition,
        )
    else:
        raise ValueError(f"Unknown suggestor name: {suggestor_type}")
