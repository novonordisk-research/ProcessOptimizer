import pytest
import numpy as np

from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_raises
from numpy.testing import assert_equal


from ProcessOptimizer import Optimizer

np.random.seed(0)


@pytest.mark.fast_test
def test_empty_Steinerberger():
    opt = Optimizer([[0, 1], [0, 1]], n_initial_points=0)
    assert_raises(ValueError, opt.ask, n_points=1, strategy="stbr_full")


@pytest.mark.fast_test
def test_Steinerberger_full():
    opt = Optimizer([[0, 1.0], [0, 1.0]], n_initial_points=1, random_state=42)
    x = [[1.0, 0.0]]
    y = [1]
    opt.tell(x, y)
    stbr_points = opt.stbr_scipy(n_points=5)
    stbr_full_points = opt.ask(n_points=5, strategy="stbr_full")
    assert_equal(len(stbr_full_points), 5)
    for x in stbr_full_points:
        assert_equal(opt.space.__contains__(x), True)
    assert_array_almost_equal(stbr_points, stbr_full_points)


@pytest.mark.fast_test
def test_Steinerberger_fill():
    opt = Optimizer([[0, 1.0], [0, 1.0]], n_initial_points=1, random_state=42)
    x = [[1.0, 0.0]]
    y = [1]
    opt.tell(x, y)
    stbr_fill_points = opt.ask(n_points=5, strategy="stbr_fill")
    new_point = opt.ask()
    opt.tell(new_point, 1)
    stbr_points = opt.stbr_scipy(n_points=4)
    new_points = stbr_points
    new_points.insert(0, new_point)
    assert_equal(len(stbr_fill_points), 5)
    for x in stbr_fill_points:
        assert_equal(opt.space.__contains__(x), True)
    assert_array_almost_equal(stbr_fill_points, new_points)


@pytest.mark.fast_test
def test_Steinerberger_initial():
    opt = Optimizer(
        [[0, 1.0], [0, 2.0]], n_initial_points=8, lhs=True, random_state=42
    )
    x = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
    y = [1, 1, 1]
    opt.tell(x, y)
    init_points = opt.ask(5)
    init_stbr_fill_points = opt.ask(5, strategy="stbr_fill")
    init_stbr_full_points = opt.ask(5, strategy="stbr_full")
    assert_array_almost_equal(init_points, init_stbr_fill_points)
    assert_array_almost_equal(init_points, init_stbr_full_points)
    assert_array_almost_equal(init_stbr_fill_points, init_stbr_full_points)
