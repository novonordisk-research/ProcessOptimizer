import math
import warnings
from typing import Iterable

import numpy as np

from .default_suggestor import DefaultSuggestor
from .lhs_suggestor import LHSSuggestor
from .po_suggestor import POSuggestor
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
            suggestor can make and the second element is the suggestor. A negative number
            of suggestions is interpreted as infinity, aand can only be used as the last
            budget.

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
                        POSuggestor(
                            space=suggestor.space,
                            rng=suggestor.rng,
                            n_objectives=suggestor.n_objectives,
                        ))
            if budget <= 0:
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
                    raise ValueError("Infinite budget must be the last budget")
        self.suggestors = suggestors

    def suggest(self, Xi: Iterable[Iterable], Yi: Iterable, n_asked: int = 1):
        number_to_skip = len(Xi)
        number_left_to_find = n_asked
        suggestions = []
        for budget, suggestor in self.suggestors:
            if number_left_to_find == 0:
                break
            if number_to_skip >= budget:
                number_to_skip -= budget
                continue
            if number_to_skip + number_left_to_find >= budget:
                suggestions.extend(suggestor.suggest(Xi, Yi, budget - number_to_skip))
                number_left_to_find -= budget - number_to_skip
                number_to_skip = 0
            else:
                suggestions.extend(suggestor.suggest(Xi, Yi, number_left_to_find))
                number_left_to_find = 0

        if len(suggestions) < n_asked:
            raise IncompatibleNumberAsked("Not enough suggestions")
        return np.array(suggestions, dtype = object)

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
