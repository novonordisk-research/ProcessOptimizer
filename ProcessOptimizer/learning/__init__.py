"""Machine learning extensions for model-based optimization."""

from .cook_estimator import cook_estimator
from .forest import RandomForestRegressor
from .forest import ExtraTreesRegressor
from .gaussian_process import GaussianProcessRegressor
from .gbrt import GradientBoostingQuantileRegressor
from .has_gradients import has_gradients
from .use_named_args import use_named_args

__all__ = (
    "cook_estimator",
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "GradientBoostingQuantileRegressor",
    "GaussianProcessRegressor",
    "has_gradients",
    "use_named_args",
)
