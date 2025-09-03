from __future__ import annotations
import warnings
from typing import Any, Callable, Iterable

import numpy as np
from ProcessOptimizer.space import Space

from .default_suggestor import DefaultSuggestor, NoDefaultSuggestorError
from .suggestor import Suggestor


class RandomStrategizer:
    """
    Strategizer that randomly selects one of its child suggestors to suggest each
    point, based on a usage ratio for each suggestor.

    When creating from a definition dictionary, the dictionary should have the key
    `"suggestors"`, and the corresponding value should be a list of dictionaries
    (child suggestor dictionaries), each representing a suggestor and its configuration.
    Each child suggestor dictionary should have the key `"suggestor_usage_ratio"` (the
    usage ratio for the suggestor). In addition, each child suggestor dictionary should
    either only have the key `"suggestor"`, with the value being interpretetable by the
    `suggestor_factory`, or it should be directly interpretable by the suggestor factory
    when the key `"suggestor_usage_ratio"` has been removed.

    Example of a valid definition, using different approaches:
    definition = {
        "suggestor_name": "Random",
        "suggestors": [
            {
                "suggestor_usage_ratio": 70,
                "suggestor": {
                    "suggestor_name": "LHS",
                    "n_points": 5,
                },
            },
            {
                "suggestor_usage_ratio": 20,
                "suggestor": {
                    "suggestor_name": "PO",
                    "n_initial_points": 10,
                },
            },
            {
                "suggestor_usage_ratio": 10,
                "suggestor": ConstantSuggestor(space, [42]),
            },
        ]
    }
    """

    def __init__(
        self, suggestors: list[tuple[float, Suggestor]], rng: np.random.Generator
    ):
        self.total = sum(item[0] for item in suggestors)
        if float(self.total) != 1.0 and float(self.total) != 100.0:
            warnings.warn(
                "Probabilities do not sum to 1.0 or 100.0. They will be normalized."
            )
        if any(isinstance(item[1], DefaultSuggestor) for item in suggestors):
            raise NoDefaultSuggestorError(
                "No DefaultSuggestor defined for RandomStrategizer."
            )
        self.suggestors = suggestors
        self.rng = rng

    def suggest(
        self, Xi: Iterable[Iterable], Yi: Iterable, n_points_to_suggest: int = 1
    ) -> np.ndarray:
        # Creating n_points_to_suggest random indices in the range [0, total)
        selector_indices = [
            relative_index * self.total
            for relative_index in self.rng.random(size=n_points_to_suggest)
        ]
        suggested_points: list[np.ndarray] = []
        # Iterating over the suggestors, finding the number of suggested points for each
        # suggestor, and suggesting them.
        # Note that this will order the points in the order of the suggestors, the
        # suggestion will start with points from the first suggestor, then the second,
        # etc.
        for weight, suggestor in self.suggestors:
            selector_indices = [index - weight for index in selector_indices]
            n_suggested = sum(index < 0 for index in selector_indices)
            if n_suggested > 0:
                suggested_points.extend(suggestor.suggest(Xi, Yi, int(n_suggested)))
            selector_indices = [index for index in selector_indices if index >= 0]
        return np.array(suggested_points, dtype=object)

    def __str__(self):
        return "Random Strategizer with suggestors: " + ", ".join(
            suggestor.__class__.__name__ for _, suggestor in self.suggestors
        )

    @classmethod
    def create_from_definition(
        cls,
        space: Space,
        suggestor_factory: Callable[..., Suggestor],
        definition: dict[str, Any],
        n_objectives: int,
        rng: np.random.Generator,
    ) -> RandomStrategizer:
        suggestors = []
        # To make the child suggestors independent, we spawn a random number generator
        # for each. This allows them to be fully deterministic based on the original
        # seed, while what they suggest is not affected by how many times the other
        # sibling suggestors have been used. If this was not here, for two child
        # suggestors `A` and `B`, `[A.suggest(), B.suggest(), A.suggest()]` would risk
        # yielding different points than `[A.suggest(), A.suggest(), B.suggest()]`.
        child_rngs = rng.spawn(len(definition["suggestors"]))
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
                        "'suggestor_usage_ratio', but it has the keys "
                        f"`suggestor_usage_ratio`, {suggestor.keys()}."
                    )
                suggestor = suggestor["suggestor"]
            suggestors.append(
                (
                    usage_ratio,
                    suggestor_factory(space, suggestor, n_objectives, child_rngs.pop()),
                )
            )
        return cls(suggestors=suggestors, rng=rng)
