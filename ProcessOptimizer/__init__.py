"""
`ProcessOptimizer`, is a simple and efficient library to minimize (very)
expensive and noisy black-box functions. It implements several methods for
sequential model-based optimization. `ProcessOptimizer` is reusable in many
contexts and accessible.

## Install

pip install ProcessOptimizer

## Development

The library is still experimental and under heavy development.

"""

from . import acquisition
from .model_systems import benchmarks, ModelSystem
from . import callbacks
from . import learning
from . import optimizer

from . import space
from .learning import cook_estimator
from .optimizer import dummy_minimize
from .optimizer import forest_minimize
from .optimizer import gbrt_minimize
from .optimizer import gp_minimize
from .optimizer import Optimizer
from .searchcv import BayesSearchCV
from .space import Categorical, Integer, Space, space_factory, Real
from .utils import dump
from .utils import expected_minimum
from .utils import expected_minimum_random_sampling
from .utils import load
from .utils import create_result, y_coverage
from .plots import plot_objective, plot_objectives, plot_objective_1d
from .plots import plot_evaluations, plot_convergence
from .plots import plot_Pareto, plot_expected_minimum_convergence

__version__ = "1.0.1"


__all__ = (
    "acquisition",
    "benchmarks",
    "ModelSystem",
    "callbacks",
    "learning",
    "optimizer",
    "plots",
    "space",
    "gp_minimize",
    "dummy_minimize",
    "forest_minimize",
    "gbrt_minimize",
    "Optimizer",
    "dump",
    "load",
    "cook_estimator",
    "create_result",
    "expected_minimum",
    "expected_minimum_random_sampling",
    "BayesSearchCV",
    "Categorical",
    "Integer",
    "Space",
    "space_factory",
    "Real",
    "plot_objective",
    "plot_objectives",
    "plot_objective_1d",
    "plot_evaluations",
    "plot_convergence",
    "plot_Pareto",
    "y_coverage",
    "plot_expected_minimum_convergence",
)
