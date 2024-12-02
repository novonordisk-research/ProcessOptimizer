import logging
from abc import ABC, abstractmethod

from .suggestor import Suggestor

logger = logging.getLogger(__name__)


# Strategizer is nearly superfluous, but having it done the same way helps with
# readability, and we also ensure that the logging is consistent.
class Strategizer(ABC):
    """
    A strategizer is a suggestor that uses other suggestors to suggest points.
    """
    @abstractmethod
    def next_suggestor(self, Xi: list[list], Yi: list) -> Suggestor:
        """ Select the next suggestor to use."""
        pass

    def suggest(self, Xi: list[list], Yi: list) -> list[list]:
        suggestor = self.next_suggestor(Xi, Yi)
        suggestion = suggestor.suggest(Xi, Yi)
        logger.debug(
            "Strategizer of type %s is using suggestor %s of type %s. It suggested: %s",
            type(self),
            suggestor,
            type(suggestor),
            suggestion,
        )
        return suggestion
