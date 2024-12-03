from typing import Protocol, runtime_checkable


@runtime_checkable  # Need to be runtime checkable for the factory to work
class Suggestor(Protocol):
    """
    Protocol for suggestors. Suggestors are used to suggest new points to evaluate in the
    optimization process. Suggestors should be stateless and only depend on the search
    space and the already evaluated points. In particular, consecutive calls to the
    suggest method with the same input should ideally return the same output, or at least
    output the same number of points.
    """
    def __init__(self, **kwargs):
        """
        Initialize the suggestor with the search space. Suggestors can take other input
        arguments as needed.
        """
        pass

    def suggest(self, Xi: list[list], Yi: list, n_asked: int) -> list[list]:
        """
        Suggest a new point to evaluate. The input is a list of already evaluated points
        and their corresponding scores. The output is a list of new points to evaluate.
        The list can have the length of 1 or more.
        """
        pass


class IncompatibleNumberAsked(ValueError):
    """
    Exception raised when a suggestor is asked to suggest more points than it can suggest.
    """
    pass
