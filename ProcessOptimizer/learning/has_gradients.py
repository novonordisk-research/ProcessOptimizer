from . import ExtraTreesRegressor
from . import GradientBoostingQuantileRegressor
from . import RandomForestRegressor
from .gaussian_process.kernels import HammingKernel


def has_gradients(estimator):
    """
    Check if an estimator's ``predict`` method provides gradients.

    Parameters
    ----------
    estimator: sklearn BaseEstimator instance.
    """
    tree_estimators = (
        ExtraTreesRegressor,
        RandomForestRegressor,
        GradientBoostingQuantileRegressor,
    )

    # cook_estimator() returns None for "dummy minimize" aka random values only
    if estimator is None:
        return False

    if isinstance(estimator, tree_estimators):
        return False

    categorical_gp = False
    if hasattr(estimator, "kernel"):
        params = estimator.get_params()
        categorical_gp = isinstance(estimator.kernel, HammingKernel) or any(
            [isinstance(params[p], HammingKernel) for p in params]
        )

    return not categorical_gp
