import logging

from .default_suggestor import DefaultSuggestor
from .po_suggestor import POSuggestor
from .suggestor import Suggestor
from .strategizer import Strategizer

logger = logging.getLogger(__name__)


class InitialPointSuggestor(Strategizer):
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
                "Initial suggestor is DefaultSuggestor, replacing with POSuggestor"
            )
            initial_suggestor = POSuggestor(
                initial_suggestor.space,
                n_initial_points=n_initial_points,
                rng=initial_suggestor.rng,
            )
        if isinstance(ultimate_suggestor, DefaultSuggestor):
            logger.debug(
                "Ultimate suggestor is DefaultSuggestor, replacing with POSuggestor"
            )
            ultimate_suggestor = POSuggestor(
                ultimate_suggestor.space, n_initial_points=0, rng=ultimate_suggestor.rng
            )
        self.n_initial_points = n_initial_points
        self.initial_suggestor = initial_suggestor
        self.ultimate_suggestor = ultimate_suggestor

    def next_suggestor(self, Xi, yi):
        remaining_initial_points = self.n_initial_points - len(Xi)
        if remaining_initial_points > 0:
            logger.debug("Only %s points told, using initial suggestor", len(Xi))
            suggestion = self.initial_suggestor.suggest(Xi, yi)
            if len(suggestion) > remaining_initial_points:
                raise ValueError(
                    "Initial suggestor suggested %s points, but since %s points have"
                    "already been told, it should suggest at most %s points",
                    len(suggestion),
                    len(Xi),
                    remaining_initial_points,
                )
            return self.initial_suggestor
        else:
            logger.debug("Using ultimate suggestor")
            return self.ultimate_suggestor
