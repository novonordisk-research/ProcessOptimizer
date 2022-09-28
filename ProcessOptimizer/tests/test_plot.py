import pytest

import numpy as np
from numpy.testing import assert_array_equal

import json

import ProcessOptimizer as Optimizer
from ProcessOptimizer.utils import cook_estimator
from ProcessOptimizer.learning.gaussian_process.kernels import Matern
from ProcessOptimizer.plots import plot_Pareto_bokeh


def build_Pareto_opt():
    space = [(-150.0, 50.0), (25.0, 60.0)]
    kernel_intern = 1**2 * Matern(
        length_scale=np.ones(len(space)), length_scale_bounds=[0.1, 3.0], nu=2.5
    )
    base_est = cook_estimator(
        base_estimator="GP",
        space=space,
        noise="gaussian",
        kernel=kernel_intern,
    )
    x = [
        [-23.4, 50.0],
        [2.9, 54.0],
        [-143.6, 26.0],
        [30.3, 45.0],
        [-76.2, 47.0],
        [-37.1, 40.0],
        [-117.2, 52.0],
        [-129.9, 33.0],
        [43.0, 56.0],
        [-89.9, 29.0],
        [-49.8, 36.0],
        [-9.8, 38.0],
        [16.6, 31.0],
        [-103.5, 59.0],
        [-63.5, 43.0],
        [0.0, 45.0],
        [-149.4, 38.0],
        [-149.4, 30.0],
        [-149.4, 58.0],
        [-145.5, 41.0],
        [-32.2, 27.0],
        [-90.8, 25.0],
        [-60.5, 25.0],
        [-117.2, 25.0],
        [-44.9, 53.0],
        [-143.6, 35.0],
        [-98.6, 49.0],
    ]
    y = [
        [0.7023567972530045, 0.9],
        [0.7255723960012899, 0.89],
        [0.7882291119285338, 0.67],
        [0.7465162574651626, 0.89],
        [0.6733502917851264, 0.89],
        [0.7219637413765442, 0.83],
        [0.6493506493506493, 0.89],
        [0.7142857142857143, 0.76],
        [0.7533902561526871, 0.89],
        [0.782200590996002, 0.71],
        [0.7480053191489362, 0.79],
        [0.758981278461798, 0.82],
        [0.8632265490120852, 0.74],
        [0.6581834137779728, 0.89],
        [0.6856620447965869, 0.87],
        [0.716674629718108, 0.88],
        [0.6712410501193318, 0.82],
        [0.739888194672805, 0.72],
        [0.6340707341130055, 0.88],
        [0.6549265026924757, 0.84],
        [0.8770220229974663, 0.67],
        [0.855188141391106, 0.65],
        [0.8873989351212779, 0.65],
        [0.8296460176991151, 0.65],
        [0.6876528117359413, 0.88],
        [0.6905017646156207, 0.76],
        [0.6613756613756614, 0.88],
    ]

    opt = Optimizer.Optimizer(
        dimensions=space,
        base_estimator=base_est,
        n_initial_points=15,
        lhs=True,
        acq_func="EI",
        n_objectives=2,
    )
    for c, v in list(zip(x, y)):
        opt.tell(c, v)

    return opt, np.array(x), np.array(y)


def test_plot_Pareto_bokeh_return_data():

    opt, x, y = build_Pareto_opt()
    output = plot_Pareto_bokeh(opt, return_data=True)
    assert_array_equal(x, output[0])
    assert_array_equal(y, output[1])


def test_plot_Pareto_bokeh_return_htmlString():
    opt, x, y = build_Pareto_opt()
    html = plot_Pareto_bokeh(opt, return_type_bokeh="htmlString")
    assert isinstance(html, str)


def validateJSON(jsonData):
    try:
        json.loads(json.dumps(jsonData))
    except:
        return False
    return True


def test_plot_Pareto_bokeh_return_json():
    opt, x, y = build_Pareto_opt()
    json_object = plot_Pareto_bokeh(opt, return_type_bokeh="json")
    assert validateJSON(json_object)
