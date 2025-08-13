from __future__ import annotations

from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from ProcessOptimizer.space import Space

import numpy as np


@runtime_checkable  # Need to be runtime checkable for the factory to work
class Suggestor(Protocol):
    """
    Protocol for suggestors. Suggestors are used to suggest new points to evaluate in the
    optimization process. Suggestors should be stateless and only depend on the search
    space and the already evaluated points. In particular, consecutive calls to the
    suggest method with the same input should ideally return the same output.
    """

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
        dimensions in the search space. The dtype is `object`, to allow for any both
        categorical and numerical dimensions without converting everything to str.
        """
        pass


@runtime_checkable  # Need to be runtime checkable for the factory to work
class CreatableSuggestor(Protocol):
    """
    Suggestors that are creatable from the factory from dicts
    """

    @classmethod
    def create_from_definition(
        cls,
        space: Space,
        suggestor_factory: Callable[..., Suggestor],
        definition: dict[str, Any],
        n_objectives: int,
        rng: np.random.Generator,
    ) -> Suggestor: ...


class IncompatibleNumberAsked(ValueError):
    """
    Exception raised when a suggestor is asked to suggest more points than it can suggest.
    """

    pass
