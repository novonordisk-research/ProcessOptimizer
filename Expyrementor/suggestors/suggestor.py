from typing import Protocol, runtime_checkable

import numpy as np
from ProcessOptimizer.space import Space


@runtime_checkable  # Need to be runtime checkable for the factory to work
class Suggestor(Protocol):
    """
    Protocol for suggestors. Suggestors are used to suggest new points to evaluate in the
    optimization process. Suggestors should be stateless and only depend on the search
    space and the already evaluated points. In particular, consecutive calls to the
    suggest method with the same input should ideally return the same output, or at least
    output the same number of points.
    """
    def __init__(self, space: Space, n_objectives: int, rng: np.random.Generator, **kwargs):
        """
        Initialize the suggestor with the search space. Suggestors can take other input
        arguments as needed.
        """
        pass

    def suggest(self, Xi: list[list], Yi: list) -> list[list]:
        """
        Suggest a new point to evaluate. The input is a list of already evaluated points
        and their corresponding scores. The output is a list of new points to evaluate.
        The list can have the length of 1 or more.
        """
        pass


class BatchSuggestor(Suggestor):
    """
    Protocol for batch suggestors. Batch suggestors are suggestors that can be told how
    many suggestions to make in a single call. This will usually be something building on
    top of a regular suggestor, like constant liers or Krigging belivers.
    """
    n_given: int
