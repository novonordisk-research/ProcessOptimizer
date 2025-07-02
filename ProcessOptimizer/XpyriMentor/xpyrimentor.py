import copy
import logging
from typing import Any, Iterable, Union, Optional
import warnings

import torch
import numpy as np
from ProcessOptimizer.space import space_factory, Space
from ProcessOptimizer.utils import is_2Dlistlike
from ProcessOptimizer.utils.get_rng import get_random_generator

from .suggestors import DefaultSuggestor, Suggestor, suggestor_factory, OptimizerSuggestor

logger = logging.getLogger(__name__)

DEFAULT_SUGGESTOR = {
    "suggestor_name": "Sequential",
    "suggestors": [
        {"suggestor_budget": 5, "suggestor_name": "Default"},
        {"suggestor_budget": -1, "suggestor_name": "Default"},
    ],
}


class XpyriMentor:
    """
    XpyriMentor class for optimization experiments. This class is used to manage the
    optimization process, including the search space, the suggestor, and the already
    evaluated points. The ask-tell interface is used to interact with the optimization
    process. The XpyriMentor class is stateful and keeps track of the already evaluated
    points and scores.
    """
    def __init__(
        self,
        space: Union[Space, list],
        suggestor: Union[Suggestor, dict, None] = None,
        n_objectives: int = 1,
        seed: Union[int, np.random.RandomState, np.random.Generator, None] = 42,
        active_task: bool = False
    ):
        """
        Initialize the XpyriMentor with the search space and the suggestor. The suggestor
        can be a Suggestor object, a dictionary with the suggestor configuration, or None.
        If the suggestor is None, the default suggestor is used. The seed is used to
        initialize the random number generator.
        """
        space = space_factory(space)
        rng = get_random_generator(seed)
        suggestor = suggestor_factory(space, suggestor, n_objectives, rng, active_task)
        if isinstance(suggestor, DefaultSuggestor):
            logger.debug("Replacing DefaultSuggestor with InitialPointSuggestor")
            suggestor = suggestor_factory(
                space, copy.deepcopy(DEFAULT_SUGGESTOR), n_objectives, rng=rng, active_task=active_task
            )
        if isinstance(suggestor, OptimizerSuggestor):
            warnings.warn(
                "POSuggestor is not recommended for use as a base. Use "
                "InitialPointSuggestor with a POSuggestor as ultimate_suggestor "
                "instead. Unless explicitly set, n_initial_points in POSuggestor will "
                "be set to 0."
            )
        self.suggestor = suggestor
        self.active_task_flag = active_task
        self.Xi: list[np.ndarray] = []
        # This is a list of points in the search space. Each point is a list of values for
        # each dimension of the search space.
        self.yi: list = []
        # We are agnostic to the type of the scores. They can be a float for single
        # objective optimization or a list of floats for multiobjective optimization.
        pass

    def ask(self, n: int = 1, active_task: Optional[bool]=None) -> np.ndarray:
        """
        Ask the suggestor for new points to evaluate. The number of points to ask is
        specified by the argument n. The method returns a list of new points to evaluate.
        """
        return self.suggestor.suggest(Xi=self.Xi, Yi=self.yi, n_asked=n)

    def tell(self, x: Iterable, y: Any, task: Any=None) -> None:

        # Test whether the input is valid for the suggestor
        if hasattr(self.suggestor, "preprocess_tell"):
            x, y = self.suggestor.preprocess_tell(x, y, task)
                    
        if is_2Dlistlike(x):
            # If x is a list of points, we assume that y is an iterable of scores of the same
            # length, and we add the members of x and y to the iterables Xi and yi.
            self.Xi.extend(x)
            self.yi.extend(y)
        else:
            # If x is a single point, we assume that y is a single score, and we add x
            # and y to the lists Xi and yi.
            self.Xi.append(x)
            self.yi.append(y)

        if hasattr(self.suggestor, 'get_best_f'): 
            self.best_f = self.suggestor.get_best_f(self.Xi, self.yi)
            
    def plot_objective(
        self,
        levels=10,
        size=2,
        grid_resolution=101,
        show_confidence=True,
        z_scale='linear',
        title=None,
        plot_options=None,
    ):
        if hasattr(self.suggestor, 'plot_objective'):
            self.suggestor.plot_objective(
                           Xi=self.Xi,
                           Yi=self.yi,
                           levels=levels, 
                           size=size, 
                           grid_resolution=grid_resolution, 
                           show_confidence=show_confidence, 
                           z_scale=z_scale, 
                           title=title, 
                           plot_options=plot_options)
        else:
            raise NotImplementedError(f"Method plot_objective not supported for suggestor of type {type(self.suggestor)}")

    def __str__(self):
        return f"XpyriMentor with a {self.suggestor.__class__.__name__} suggestor"
