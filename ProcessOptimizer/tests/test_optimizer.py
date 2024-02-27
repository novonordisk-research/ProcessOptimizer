import numpy as np
import pytest

from sklearn.multioutput import MultiOutputRegressor
from numpy.testing import (
    assert_array_equal,
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

from math import isclose

from ProcessOptimizer import gp_minimize
from ProcessOptimizer.model_systems.benchmarks import bench1, bench1_with_time
from ProcessOptimizer.model_systems import branin_no_noise
from ProcessOptimizer.model_systems.model_system import ModelSystem
from ProcessOptimizer.learning import (
    ExtraTreesRegressor,
    GaussianProcessRegressor,
    GradientBoostingQuantileRegressor,
    RandomForestRegressor
    )
from ProcessOptimizer.optimizer import Optimizer
from ProcessOptimizer.utils import expected_minimum
from scipy.optimize import OptimizeResult

# Introducing branin function as a test function from the Branin no noise ModelSystem
branin = branin_no_noise.get_score


TREE_REGRESSORS = (
    ExtraTreesRegressor(random_state=2),
    RandomForestRegressor(random_state=2),
    GradientBoostingQuantileRegressor(random_state=2),
)
ACQ_FUNCS_MIXED = ["EI"]  # , "EIps" removed
ESTIMATOR_STRINGS = [
    "GP",
    "RF",
    "ET",
    "GBRT",
    "DUMMY",
    "gp",
    "rf",
    "et",
    "gbrt",
    "dummy",
]


@pytest.mark.fast_test
def test_multiple_asks():
    # calling ask() multiple times without a tell() inbetween should
    # be a "no op"
    base_estimator = ExtraTreesRegressor(random_state=2)
    opt = Optimizer(
        [(-2.0, 2.0)],
        base_estimator,
        n_initial_points=1,
        acq_optimizer="sampling",
    )

    opt.run(bench1, n_iter=3)
    # tell() computes the next point ready for the next call to ask()
    # hence there are three after three iterations
    assert_equal(len(opt.models), 3)
    assert_equal(len(opt.Xi), 3)
    opt.ask()
    assert_equal(len(opt.models), 3)
    assert_equal(len(opt.Xi), 3)
    assert_equal(opt.ask(), opt.ask())


@pytest.mark.fast_test
def test_invalid_tell_arguments():
    base_estimator = ExtraTreesRegressor(random_state=2)
    opt = Optimizer(
        [(-2.0, 2.0)],
        base_estimator,
        n_initial_points=2,
        acq_optimizer="sampling",
    )

    # can't have single point and multiple values for y
    assert_raises(ValueError, opt.tell, [1.0], [1.0, 1.0])


@pytest.mark.fast_test
def test_invalid_tell_arguments_list():
    base_estimator = ExtraTreesRegressor(random_state=2)
    opt = Optimizer(
        [(-2.0, 2.0)],
        base_estimator,
        n_initial_points=2,
        acq_optimizer="sampling",
    )

    assert_raises(ValueError, opt.tell, [[1.0], [2.0]], [1.0, None])


@pytest.mark.fast_test
def test_bounds_checking_1D():
    low = -2.0
    high = 2.0
    base_estimator = ExtraTreesRegressor(random_state=2)
    opt = Optimizer(
        [(low, high)], base_estimator, n_initial_points=2, acq_optimizer="sampling"
    )

    assert_raises(ValueError, opt.tell, [high + 0.5], 2.0)
    assert_raises(ValueError, opt.tell, [low - 0.5], 2.0)
    # feed two points to tell() at once
    assert_raises(ValueError, opt.tell, [high + 0.5, high], (2.0, 3.0))
    assert_raises(ValueError, opt.tell, [low - 0.5, high], (2.0, 3.0))


@pytest.mark.fast_test
def test_bounds_checking_2D():
    low = -2.0
    high = 2.0
    base_estimator = ExtraTreesRegressor(random_state=2)
    opt = Optimizer(
        [(low, high), (low + 4, high + 4)],
        base_estimator,
        n_initial_points=2,
        acq_optimizer="sampling",
    )

    assert_raises(ValueError, opt.tell, [high + 0.5, high + 4.5], 2.0)
    assert_raises(ValueError, opt.tell, [low - 0.5, low - 4.5], 2.0)

    # first out, second in
    assert_raises(ValueError, opt.tell, [high + 0.5, high + 0.5], 2.0)
    assert_raises(ValueError, opt.tell, [low - 0.5, high + 0.5], 2.0)


@pytest.mark.fast_test
def test_bounds_checking_2D_multiple_points():
    low = -2.0
    high = 2.0
    base_estimator = ExtraTreesRegressor(random_state=2)
    opt = Optimizer(
        [(low, high), (low + 4, high + 4)],
        base_estimator,
        n_initial_points=2,
        acq_optimizer="sampling",
    )

    # first component out, second in
    assert_raises(
        ValueError,
        opt.tell,
        [(high + 0.5, high + 0.5), (high + 0.5, high + 0.5)],
        [2.0, 3.0],
    )
    assert_raises(
        ValueError,
        opt.tell,
        [(low - 0.5, high + 0.5), (low - 0.5, high + 0.5)],
        [2.0, 3.0],
    )


@pytest.mark.fast_test
def test_dimension_checking_1D():
    low = -2
    high = 2
    opt = Optimizer([(low, high)])
    with pytest.raises(ValueError) as e:
        # within bounds but one dimension too high
        opt.tell([low + 1, low + 1], 2.0)
    assert "Dimensions of point " in str(e.value)


@pytest.mark.fast_test
def test_dimension_checking_2D():
    low = -2
    high = 2
    opt = Optimizer([(low, high), (low, high)])
    # within bounds but one dimension too little
    with pytest.raises(ValueError) as e:
        opt.tell([low + 1], 2.0)
    assert "Dimensions of point " in str(e.value)
    # within bounds but one dimension too much
    with pytest.raises(ValueError) as e:
        opt.tell([low + 1, low + 1, low + 1], 2.0)
    assert "Dimensions of point " in str(e.value)


@pytest.mark.fast_test
def test_dimension_checking_2D_multiple_points():
    low = -2
    high = 2
    opt = Optimizer([(low, high), (low, high)])
    # within bounds but one dimension too little
    with pytest.raises(ValueError) as e:
        opt.tell([[low + 1], [low + 1, low + 2], [low + 1, low + 3]], 2.0)
    assert "dimensions as the space" in str(e.value)
    # within bounds but one dimension too much
    with pytest.raises(ValueError) as e:
        opt.tell(
            [[low + 1, low + 1, low + 1], [low + 1, low + 2], [low + 1, low + 3]], 2.0
        )
    assert "dimensions as the space" in str(e.value)


@pytest.mark.fast_test
def test_returns_result_object():
    base_estimator = ExtraTreesRegressor(random_state=2)
    opt = Optimizer(
        [(-2.0, 2.0)],
        base_estimator,
        n_initial_points=2,
        acq_optimizer="sampling",
    )
    result = opt.tell([1.5], 2.0)

    assert isinstance(result, OptimizeResult)
    assert_equal(len(result.x_iters), len(result.func_vals))
    assert_equal(np.min(result.func_vals), result.fun)


@pytest.mark.fast_test
@pytest.mark.parametrize("base_estimator", TREE_REGRESSORS)
def test_acq_optimizer(base_estimator):
    with pytest.raises(ValueError) as e:
        Optimizer(
            [(-2.0, 2.0)],
            base_estimator=base_estimator,
            n_initial_points=2,
            acq_optimizer="lbfgs",
        )
    assert "should run with acq_optimizer='sampling'" in str(e.value)


@pytest.mark.fast_test
@pytest.mark.parametrize("acq_func", ACQ_FUNCS_MIXED)
def test_optimizer_copy(acq_func):
    # Checks that the base estimator, the objective and target values
    # are copied correctly.

    base_estimator = ExtraTreesRegressor(random_state=2)
    opt = Optimizer(
        [(-2.0, 2.0)],
        base_estimator,
        acq_func=acq_func,
        n_initial_points=2,
        acq_optimizer="sampling",
    )

    # run three iterations so that we have some points and objective values
    if "ps" in acq_func:
        opt.run(bench1_with_time, n_iter=3)
    else:
        opt.run(bench1, n_iter=3)

    opt_copy = opt.copy()

    copied_estimator = opt_copy.base_estimator_

    if "ps" in acq_func:
        assert isinstance(copied_estimator, MultiOutputRegressor)
        # check that the base_estimator is not wrapped multiple times
        is_multi = isinstance(copied_estimator.estimator, MultiOutputRegressor)
        assert not is_multi
    else:
        assert not isinstance(copied_estimator, MultiOutputRegressor)

    assert_array_equal(opt_copy.Xi, opt.Xi)
    assert_array_equal(opt_copy.yi, opt.yi)


@pytest.mark.parametrize("base_estimator", ESTIMATOR_STRINGS)
def test_exhaust_initial_calls(base_estimator):
    # check a model is fitted and used to make suggestions after we added
    # at least n_initial_points via tell()
    opt = Optimizer(
        [(-2.0, 2.0)],
        base_estimator,
        n_initial_points=2,
        acq_optimizer="sampling",
        random_state=1,
    )
    X_start = opt.ask(2)

    x0 = X_start[0]  # random point
    x1 = X_start[1]  # random point
    assert x0 != x1
    # first call to tell()
    r1 = opt.tell(x1, 3.0)
    assert len(r1.models) == 0
    x2 = opt.ask()  # random point (NOT in the case of LHS)
    if opt._lhs == False:
        assert x1 != x2
        # second call to tell()
        r2 = opt.tell(x2, 4.0)
        if base_estimator.lower() == "dummy":
            assert len(r2.models) == 0
        else:
            assert len(r2.models) == 1
        # this is the first non-random point
        x3 = opt.ask()
        assert x2 != x3
        x4 = opt.ask()
        r3 = opt.tell(x3, 1.0)
        # no new information was added so should be the same, unless we are using
        # the dummy estimator which will forever return random points and never
        # fits any models
        if base_estimator.lower() == "dummy":
            assert x3 != x4
            assert len(r3.models) == 0
        else:
            assert x3 == x4
            assert len(r3.models) == 2

    elif opt._lhs == True:
        assert x1 == x2
        # second call to tell()
        r2 = opt.tell(x0, 4.0)
        if base_estimator.lower() == "dummy":
            assert len(r2.models) == 0
        else:
            assert len(r2.models) == 1
        # this is the first non-LHS point
        x3 = opt.ask()
        assert x2 != x3
        x4 = opt.ask()
        r3 = opt.tell(x3, 1.0)
        # no new information was added so should be the same, unless we are using
        # the dummy estimator which will forever return random points and never
        # fits any models
        if base_estimator.lower() == "dummy":
            assert x3 != x4
            assert len(r3.models) == 0
        else:
            assert x3 == x4
            assert len(r3.models) == 2


@pytest.mark.fast_test
def test_optimizer_base_estimator_string_invalid():
    with pytest.raises(ValueError) as e:
        Optimizer([(-2.0, 2.0)], base_estimator="rtr", n_initial_points=1)
    assert "'RF', 'ET', 'GP', 'GBRT' or 'DUMMY'" in str(e.value)


@pytest.mark.fast_test
@pytest.mark.parametrize("base_estimator", ESTIMATOR_STRINGS)
def test_optimizer_base_estimator_string_smoke(base_estimator):
    opt = Optimizer(
        [(-2.0, 2.0)],
        base_estimator=base_estimator,
        n_initial_points=2,
        acq_func="EI",
    )
    opt.run(func=lambda x: x[0] ** 2, n_iter=3)


def test_defaults_are_equivalent():
    # check that the defaults of Optimizer reproduce the defaults of
    # gp_minimize
    space = [(-5.0, 10.0), (0.0, 15.0)]
    # opt = Optimizer(space, 'ET', acq_func="EI", random_state=1)
    opt = Optimizer(space, random_state=1)

    for n in range(12):
        x = opt.ask()
        res_opt = opt.tell(x, branin(x))

    # res_min = forest_minimize(branin, space, n_calls=12, random_state=1)
    res_min = gp_minimize(branin, space, n_calls=12, random_state=1)

    assert res_min.space == res_opt.space
    # tolerate small differences in the points sampled
    assert np.allclose(res_min.x_iters, res_opt.x_iters)  # , atol=1e-5)
    assert np.allclose(res_min.x, res_opt.x)  # , atol=1e-5)

    # TODO: Test that LHS can be parsed to optimizer


@pytest.mark.fast_test
def test_parsing_lhs():
    opt = Optimizer([(1, 3)], "GP", lhs=True)
    assert opt._lhs == True
    opt = Optimizer([(1, 3)], "GP")
    assert opt._lhs == True
    opt = Optimizer([(1, 3)], "GP", lhs=False)
    assert opt._lhs == False


@pytest.mark.fast_test
def test_iterating_ask_tell_lhs():
    opt = Optimizer([(1, 6)], "GP", lhs=True)
    samples = opt._lhs_samples
    # Check that opt.ask gos through all the lhs_samples
    assert opt.ask() == samples[0]
    opt.tell([1], 0)  # sample 0
    assert opt.ask() == samples[1]
    opt.tell([1], 0)  # sample 1
    # Assert that the next 3 points are the same
    assert opt.ask(n_points=3) == samples[2:5]  # samples 2,3 and 4
    opt.tell([[2], [2], [2]], [0, 0, 0])
    assert opt.ask() == samples[5]  # sample 5


@pytest.mark.slow_test
def test_add_remove_modelled_noise():
    """
    Tests whether the addition of white noise leads to predictions closer to
    known true values of experimental noise (iid gaussian noise)"""

    # Define objective function
    def flat_score(x):
        return 42

    # Set noise and model system
    noise_size = 0.45
    flat_space = [(-1.0, 1.0)]
    flat_noise = {"model_type": "constant", "noise_size": noise_size}
    # Build ModelSystem object
    model = ModelSystem(score=flat_score, space=flat_space, noise_model=flat_noise)
    # Instantiate Optimizer
    opt = Optimizer(flat_space, "GP", lhs=False, n_initial_points=1, random_state=42)
    # Make 20 dispersed points on X
    next_x = np.linspace(-1, 1, 20).tolist()
    x = []
    y = []
    # sample noisy experiments, 20 in each x-value
    for _ in range(20):
        for xx in next_x:
            x.append([xx])
            y.append(model.get_score([xx]))
    # Fit the model
    res = opt.tell(x, y)
    _, [_, res_std_no_white] = expected_minimum(res, return_std=True)
    # Add moddeled experimental noise
    opt_noise = opt.copy()
    opt_noise.add_modelled_noise()
    res_noise = opt_noise.get_result()
    _, [_, res_std_white] = expected_minimum(res_noise, return_std=True)
    # Test modelled noise is added and predicts know noise within tolerance 10%
    assert res_std_no_white < res_std_white
    assert isclose(noise_size, res_std_white, rel_tol=0.1)
    # Test function to remove experimental noise and regain "old" noise level
    opt_noise.remove_modelled_noise()
    res_noise = opt_noise.get_result()
    _, [_, res_std_reset] = expected_minimum(res_noise, return_std=True)
    assert isclose(res_std_no_white, res_std_reset, rel_tol=0.001)


@pytest.mark.fast_test
def test_estimate_single_x():
    x = [1, 1]
    regressor = GaussianProcessRegressor(noise=0, alpha=0)
    opt = Optimizer(
        [(-2.0, 2.0), (-3.0, 3.0)],
        base_estimator=regressor,
        n_initial_points=1,
    )
    opt.tell(x, 2)
    estimate_list = opt.estimate(x)[0]
    assert_almost_equal(estimate_list.Y.mean, 2)
    assert_almost_equal(estimate_list[0].mean, 2)  # test that indexing works
    assert_almost_equal(estimate_list.Y.std, 0)
    assert_almost_equal(estimate_list.mean, 2)
    assert_almost_equal(estimate_list[1], 2)  # test that indexing works
    assert_almost_equal(estimate_list.std, 0)


@pytest.mark.fast_test
def test_estimate_uncertainty():
    x = [1, 1]
    regressor = GaussianProcessRegressor(noise=1)
    opt = Optimizer(
        [(-2.0, 2.0), (-3.0, 3.0)],
        base_estimator=regressor,
        n_initial_points=1,
    )
    opt.tell(x, 2)
    estimate_list = opt.estimate(x)[0]
    assert_almost_equal(estimate_list.Y.std, (1/2)**(1/2))
    # Why is the expected value 1, and not 2, and why is the standard
    # deviation sqrt(1/2)? This would be the case if there were two
    # observations, one with Y = 0, and one with Y = 2. Is that what
    # GaussianProcessRegressor does if there is noise and only one observation?


@pytest.mark.fast_test
def test_estimate_multiple_x():
    x_list = [[1, 1], [0, 0], [1, -1]]
    y_list = [2, -1, 5]
    regressor = GaussianProcessRegressor(noise=0, alpha=0)
    opt = Optimizer(
        [(-2.0, 2.0), (-3.0, 3.0)],
        base_estimator=regressor,
        n_initial_points=3,
    )
    opt.tell(x_list, y_list)
    etimate_list = opt.estimate(x_list)
    assert_almost_equal(etimate_list[0].Y.mean, y_list[0])
    assert_almost_equal(etimate_list[0].mean, y_list[0])
    assert_almost_equal(etimate_list[1].Y.mean, y_list[1])
    assert_almost_equal(etimate_list[1].mean, y_list[1])
    assert_almost_equal(etimate_list[2].Y.mean, y_list[2])
    assert_almost_equal(etimate_list[2].mean, y_list[2])


@pytest.mark.fast_test
def test_estimate_multiple_y():
    x_list = [[1, 1], [0, 0], [1, -1]]
    y_list = [[2, 2], [-1, 0], [-3, 5]]
    regressor = GaussianProcessRegressor(noise=0, alpha=0)
    opt = Optimizer(
        [(-2.0, 2.0), (-3.0, 3.0)],
        base_estimator=regressor,
        n_initial_points=3,
        n_objectives=2
    )
    opt.tell(x_list, y_list)
    estimate_list = opt.estimate(x_list)
    assert_almost_equal(estimate_list[0].Y1.mean, y_list[0][0])
    assert_almost_equal(estimate_list[0].Y2.mean, y_list[0][1])
    assert_almost_equal(estimate_list[1].Y1.mean, y_list[1][0])
    assert_almost_equal(estimate_list[1].Y2.mean, y_list[1][1])
    assert_almost_equal(estimate_list[2].Y1.mean, y_list[2][0])
    assert_almost_equal(estimate_list[2].Y2.mean, y_list[2][1])


@pytest.mark.fast_test
def test_estimate_named_objective():
    x = [1, 1]
    y = 2
    opt = Optimizer(
        [(-2.0, 2.0), (-3.0, 3.0)],
        n_initial_points=1,
        objective_name_list=["foo"]
    )
    opt.tell(x, y)
    estimate = opt.estimate(x)[0]
    assert_almost_equal(estimate.foo.mean, y)
    assert_almost_equal(estimate.mean, y)


@pytest.mark.fast_test
def test_estimate_named_objectives():
    x = [1, 1]
    y = [2, -2]
    opt = Optimizer(
        [(-2.0, 2.0), (-3.0, 3.0)],
        n_initial_points=1,
        n_objectives=2,
        objective_name_list=["foo", "bar"]
    )
    opt.tell([x], [y])
    estimate = opt.estimate(x)[0]
    assert_almost_equal(estimate.foo.mean, y[0])
    assert_almost_equal(estimate.bar.mean, y[1])
