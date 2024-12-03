from .default_suggestor import DefaultSuggestor
from .suggestor import Suggestor, IncompatibleNumberAsked


class CachingStrategizer():
    """
    Cache the suggestions of a suggestor and return them when asked for new points.

    The caching strategizer will ask the suggestor for new points a fixed number of times
    and store the suggestions. When asked for new points, it will return the stored
    suggestions until they run out, at which point it will ask the suggestor for new
    points again.

    The caching strategizer can be used to convert a suggestor that suggests a fixed
    number of points into a suggestor that can suggest a variable number of points.
    Examples would be fitting an LHS suggestor into an iterative optimization process.
    """
    def __init__(self, suggestor: Suggestor, ask_times: int = 1, **_):
        """
        Initialize the caching strategizer with the suggestor to cache and the number of
        times to ask the suggestor for new points.

        Parameters
        ----------
        suggestor : Suggestor
            The suggestor to cache.
        ask_times : int, optional
            The number of times to ask the suggestor for new points, by default 1.
            If it is smaller than 0, the suggestor will be asked for new points
            indefinitely.
        """
        if isinstance(suggestor, DefaultSuggestor):
            raise ValueError("CachingStrategizer has no default suggestor defined.")
        self.suggestor = suggestor
        self.ask_times_left = ask_times
        self.cache = []

    def suggest(self, Xi: list[list], Yi: list, n_asked: int = 1) -> list[list]:
        suggested_points = []
        while len(suggested_points) < n_asked:
            if n_asked <= len(self.cache):
                n_points_from_cache = n_asked - len(suggested_points)
                suggested_points.extend(self.cache[:n_points_from_cache])
                self.cache = self.cache[n_points_from_cache:]
            else:
                if self.ask_times_left != 0:  # Not <= 0 to allow for infinite asking
                    # Use the exisiting cache, ask the suggestor for new points, put
                    # them in the cache, and decrement the number of times left to ask.
                    suggested_points.extend(self.cache)
                    self.cache = self.suggestor.suggest(Xi, Yi, -1)
                    self.ask_times_left -= 1
                else:
                    raise IncompatibleNumberAsked(
                        "CachingStrategizer ran out of ask times."
                    )
        return suggested_points
