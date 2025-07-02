import math
import warnings
from typing import Iterable

import numpy as np

from .default_suggestor import DefaultSuggestor
from .lhs_suggestor import LHSSuggestor
from .po_suggestor import OptimizerSuggestor
from .suggestor import IncompatibleNumberAsked, Suggestor


class SequentialStrategizer():
    """
    Stratgizer that uses a sequence of suggestors, each with a budget of suggestions to
    make.

    It uses the suggestors in order, skipping suggestors with a budget of suggestions
    that have already been made.
    """
    def __init__(self, suggestors: list[tuple[int, Suggestor]]):
        """
        Initialize the strategizer with a list of suggestors and their budgets.

        Parameters
        ----------
        * suggestors [`list[tuple[int, Suggestor]]`]:
            A list of tuples where the first element is the number of suggestions the
            suggestor can make (its budget) and the second element is the suggestor. A
            negative number of suggestions is interpreted as infinity, and can only be
            used as budget for the last suggestor.

            If the first suggestor is a DefaultSuggestor, it will be replaced with a
            LHSSuggestor with the same budget. If any other suggestor is a
            DefaultSuggestor, it will be replaced with a POSuggestor.
        """
        for n, (budget, suggestor) in enumerate(suggestors):
            if isinstance(suggestor, DefaultSuggestor):
                if n == 0:
                    suggestors[n] = (
                        budget,
                        LHSSuggestor(
                            space=suggestor.space, rng=suggestor.rng, n_points=budget
                        )
                    )
                else:
                    suggestors[n] = (
                        budget,
                        OptimizerSuggestor(
                            space=suggestor.space,
                            rng=suggestor.rng,
                            n_objectives=suggestor.n_objectives,
                        ))
            if budget < 0:
                # Interpret negative budgets as infinity
                suggestors[n] = (float("inf"), suggestors[n][1])
            if hasattr(suggestors[n][1], "n_points"):
                if suggestors[n][1].n_points != suggestors[n][0]:
                    warnings.warn(
                        f"Budget of {suggestors[n][0]} points does not match number of "
                        "points for suggestor of type "
                        f"{suggestors[n][1].__class__.__name__}."
                    )
            if math.isinf(suggestors[n][0]):
                if n < len(suggestors) - 1:
                    raise ValueError(
                        "Only the last suggestor can have an infinite budget."
                    )
        self.suggestors = suggestors

    def suggest(self, Xi: Iterable[Iterable], Yi: Iterable, n_asked: int = 1, active_task: bool=False):
        # We will skip as many points as we have already been told about.
        number_left_to_skip = len(Xi)  # Running tally of points to skip.
        number_left_to_find = n_asked  # Running tally of points to find.
        # Both of these will be decremented as we go through the suggestors.
        suggestions = []
        for budget, suggestor in self.suggestors:
            if number_left_to_skip >= budget:
                # If we have been told enough points, we have to skip this suggestor.
                number_left_to_skip -= budget
                continue
            elif number_left_to_skip + number_left_to_find >= budget:
                # If we need more points than the suggestor can give us, we take all the
                # points the suggestor can give us and continue with the next suggestor.
                number_from_this_suggestor = budget - number_left_to_skip
                number_left_to_skip = 0
            else:
                # If we need fewer points than the suggestor can give us, we take the
                # points we need and stop.
                number_from_this_suggestor = number_left_to_find
            suggestions.extend(suggestor.suggest(Xi, Yi, number_from_this_suggestor, active_task))
            number_left_to_find -= number_from_this_suggestor
            if number_left_to_find == 0:
                # If we have already found all the points we need, we can stop.
                break
        if len(suggestions) < n_asked:
            raise IncompatibleNumberAsked("Not enough suggestions")
        return np.array(suggestions, dtype=object)

    def __str__(self):
        return "Sequential Strategizer with suggestors: " + ", ".join(
            suggestor.__class__.__name__ for _, suggestor in self.suggestors
        )

    def __repr__(self):
        suggestor_list_str = ", ".join(
            f"({budget}, {suggestor.__class__.__name__}(...))" for budget, suggestor in self.suggestors
        )
        return (
            f"SequentialStrategizer(suggestors=[{suggestor_list_str}]"
        )
