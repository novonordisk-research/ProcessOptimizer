import numpy as np
from sklearn.base import is_regressor
from sklearn.ensemble import GradientBoostingRegressor

from .forest import ExtraTreesRegressor
from .gaussian_process import GaussianProcessRegressor
from .gbrt import GradientBoostingQuantileRegressor
from .forest import RandomForestRegressor
from .gaussian_process.kernels import ConstantKernel
from .gaussian_process.kernels import HammingKernel
from .gaussian_process.kernels import Matern
from ..space import Space, normalize_dimensions


def cook_estimator(
    base_estimator,
    space=None,
    length_scale_bounds=None,
    length_scale=None,
    **kwargs,
):
    """
    Cook a default estimator.

    For the special base_estimator called "DUMMY" the return value is None.
    This corresponds to sampling points at random, hence there is no need
    for an estimator.

    Parameters
    ----------
    * `base_estimator` ["GP", "RF", "ET", "GBRT", "DUMMY"
                        or sklearn regressor, default="GP"]:
        Should inherit from `sklearn.base.RegressorMixin`.
        In addition the `predict` method should have an optional `return_std`
        argument, which returns `std(Y | x)`` along with `E[Y | x]`.
        If base_estimator is one of ["GP", "RF", "ET", "GBRT", "DUMMY"], a
        surrogate model corresponding to the relevant `X_minimize` function
        is created.

    * `space` [Space instance]:
        Has to be provided if the base_estimator is a gaussian process.
        Ignored otherwise.
    * `length_scale_bounds` [list of tuples]:
        the length scale bounds for the matern kernel
    * `length_scale_bounds` [list of floats]:
        the length scales for the Matern or Hamming kernel
    * `kwargs` [dict]:
        Extra parameters provided to the base_estimator at init time.
    """

    if isinstance(base_estimator, str):
        base_estimator = base_estimator.upper()
        if base_estimator not in ["GP", "ET", "RF", "GBRT", "DUMMY"]:
            raise ValueError(
                "Valid strings for the base_estimator parameter "
                "are: 'RF', 'ET', 'GP', 'GBRT' or 'DUMMY' not "
                "%s." % base_estimator
            )
    elif not is_regressor(base_estimator):
        raise ValueError("base_estimator has to be a regressor.")

    if base_estimator == "GP":
        if space is not None:
            space = Space(space)
            space = Space(normalize_dimensions(space.dimensions))
            n_dims = space.transformed_n_dims
            is_cat = space.is_categorical

        else:
            raise ValueError("Expected a Space instance, not None.")

        cov_amplitude = ConstantKernel(1.0, (0.01, 1000))

        if not length_scale:
            length_scale = np.ones(n_dims)
        if not length_scale_bounds:
            length_scale_bounds = [(0.1, 1)] * n_dims

        # Transform lengthscale bounds:
        length_scale_bounds_transformed = []
        length_scale_transformed = []
        for i in range(len(space.dimensions)):
            for j in range(space.dimensions[i].transformed_size):
                length_scale_bounds_transformed.append(length_scale_bounds[i])
                length_scale_transformed.append(length_scale[i])

        # only special if *all* dimensions are categorical
        if is_cat:
            other_kernel = HammingKernel(length_scale=length_scale_transformed)
        else:
            other_kernel = Matern(
                length_scale=length_scale_transformed,
                length_scale_bounds=length_scale_bounds_transformed,
                nu=2.5,
            )

        base_estimator = GaussianProcessRegressor(
            kernel=cov_amplitude * other_kernel,
            normalize_y=True,
            noise="gaussian",
            n_restarts_optimizer=4,
        )
    elif base_estimator == "RF":
        base_estimator = RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
    elif base_estimator == "ET":
        base_estimator = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=3)
    elif base_estimator == "GBRT":
        gbrt = GradientBoostingRegressor(n_estimators=30, loss="quantile")
        base_estimator = GradientBoostingQuantileRegressor(base_estimator=gbrt)

    elif base_estimator == "DUMMY":
        return None

    base_estimator.set_params(**kwargs)
    return base_estimator
