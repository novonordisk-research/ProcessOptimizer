from typing import Iterable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable  # Need to be runtime checkable for the factory to work
class Suggestor(Protocol):
    """
    Protocol for suggestors. Suggestors are used to suggest new points to evaluate in the
    optimization process. Suggestors should be stateless and only depend on the search
    space and the already evaluated points. In particular, consecutive calls to the
    suggest method with the same input should ideally return the same output.
    """
    def __init__(self, **kwargs):
        """
        Initialize the suggestor with the search space. Suggestors can take other input
        arguments as needed.

        The input key `suggestor` and any keyword starting with `suggestor_` are
        reserved and should not be used in __init__ of suggestors. They might be removed
        from the definition dict by the factory before passing it to the suggestor.
        """
        pass

    def suggest(self, Xi: Iterable[Iterable], Yi: Iterable, n_asked: int) -> np.ndarray:
        """
        Suggest a new point to evaluate.

        Parameters
        ----------
        * Xi [`Iterable[Iterable]`]:
            The input is a list of already evaluated points.
        * Yi [`Iterable`]:
            The results of the evaulations of `Xi`.
        * n_asked [`int`]:
            The number of suggested points to return

        Returns
        ----------
        A np.ndarray of size `n_asked` x `n_dim`, where `n_dim` is the number of
        dimenstion in the search space.
        """
        pass


class IncompatibleNumberAsked(ValueError):
    """
    Exception raised when a suggestor is asked to suggest more points than it can suggest.
    """
    pass
