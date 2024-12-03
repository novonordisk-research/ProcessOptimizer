import logging

from .caching_strategizer import CachingStrategizer
from .default_suggestor import DefaultSuggestor
from .lhs_suggestor import LHSSuggestor
from .po_suggestor import POSuggestor
from .suggestor import Suggestor

logger = logging.getLogger(__name__)


class InitialPointStrategizer():
    """
    A strategizer that uses one suggestor for a fixed number of initial points and then
    switches to another suggestor.

    Be careful when using a suggestor that suggest multiple points at once as the
    inital suggestor. If it suggests more points than the number of remaining initial
    points, its suggestions will be truncated so it only returns the remaining initial
    points. This can be a problem with design of experiments suggestors, for example, as
    all of their suggested points are needed for the analysis.
    """
    def __init__(
            self,
            initial_suggestor: Suggestor,
            ultimate_suggestor: Suggestor,
            n_initial_points: int = 5,
    ):
        """
        Parameters
        ----------
        initial_suggestor : Suggestor
            The suggestor to use for the initial points. Default is a POSuggestor with
            n_initial_points initial points.
        ultimate_suggestor : Suggestor
            The suggestor to use after the initial points have been suggested. Default
            is a POSuggestor with no initial points.
        n_initial_points : int
            The number of initial points to suggest with the initial suggestor. Default
            is 5.
        """
        if isinstance(initial_suggestor, DefaultSuggestor):
            # If the initial suggestor is the default suggestor, we replace it with a
            # POSuggestor. It should be a much simpler suggestor, e.g. a Latin Hypercube
            # Sampling suggestor. Replace the POSuggestor when available.
            logger.debug(
                "Initial suggestor is DefaultSuggestor, replacing with "
                "cached LHSSuggestor."
            )
            lhs_suggestor = LHSSuggestor(
                space=initial_suggestor.space,
                rng=initial_suggestor.rng,
                n_points=n_initial_points,
            )
            initial_suggestor = CachingStrategizer(lhs_suggestor)
        if isinstance(ultimate_suggestor, DefaultSuggestor):
            logger.debug(
                "Ultimate suggestor is DefaultSuggestor, replacing with POSuggestor."
            )
            ultimate_suggestor = POSuggestor(
                ultimate_suggestor.space, n_initial_points=0, rng=ultimate_suggestor.rng
            )
        self.n_initial_points = n_initial_points
        self.initial_suggestor = initial_suggestor
        self.ultimate_suggestor = ultimate_suggestor

    def suggest(self, Xi: list[list], Yi: list, n_asked: int = 1) -> list[list]:
        n_initial_points = max(self.n_initial_points - len(Xi), 0)
        n_ultimate_points = n_asked - n_initial_points
        suggestions = []
        if n_initial_points > 0:
            suggestions.extend(self.initial_suggestor.suggest(Xi, Yi, n_initial_points))
        if n_ultimate_points > 0:
            suggestions.extend(self.ultimate_suggestor.suggest(
                Xi, Yi, n_ultimate_points)
            )
        return suggestions
