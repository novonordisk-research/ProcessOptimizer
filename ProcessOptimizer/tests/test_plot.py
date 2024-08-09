import pytest

import numpy as np
import matplotlib as mpl
from numpy.testing import assert_array_equal

import json

import ProcessOptimizer as Optimizer
from ProcessOptimizer.learning import cook_estimator
from ProcessOptimizer.learning.gaussian_process.kernels import Matern
from ProcessOptimizer.plots import (
    plot_Pareto_bokeh,
    plot_objective,
    plot_objective_1d,
    plot_brownie_bee_frontend,
)
from ProcessOptimizer.space import Categorical


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


def test_plot_objective():
    # Integration test of the plot_objective function
    dimensions = [
        "Mass added [kg]",
        "Temperature [C]",
        "Conc. [M]",
        "Viscosity",
        "Categoric choice",
    ]
    space = [
        (15.0, 25.0),
        (70.0, 90.0),
        (0.03, 0.15),
        (70.0, 167.0),
        ["A", "B"],
    ]
    opt = Optimizer.Optimizer(
        space,
        acq_func="EI",
        n_initial_points=5,
    )
    x = [
        [18, 84, 0.042, 102.8, "A"],
        [24, 80, 0.09, 102.8, "B"],
        [20, 76, 0.066, 102.8, "A"],
        [16, 72, 0.111, 102.8, "A"],
        [22, 88, 0.138, 102.8, "B"],
        [24.5, 70, 0.145, 102.8, "A"],
        [19, 79, 0.124, 102.8, "B"],
        [15.5, 80, 0.100, 106, "B"],
        [24, 86, 0.111, 106, "B"],
        [18, 79, 0.124, 109, "A"],
        [16, 75, 0.145, 109, "B"],
        [17.9, 74, 0.150, 122.9, "A"],
        [22.6, 89, 0.075, 122.9, "B"],
        [23, 70, 0.145, 118.6, "A"],
        [23, 84, 0.042, 118.6, "B"],
    ]
    y = [54, 53, 38, 30, 65, 42, 60, 77, 66, 75, 61, 57, 87, 47, 67]
    res = opt.tell(x, y)
    plot_options = {
        "interpolation": "bicubic",
        "uncertain_color": [1, 1, 1],
        "colormap": "viridis_r",
        "normalize_uncertainty": (
            lambda x, global_min, global_max:
                (x - global_min)/ (global_max - global_min)
        ),
    }
    axes = plot_objective(
        res,
        pars="expected_minimum",
        show_confidence=True,
        usepartialdependence=False,
        dimensions=dimensions,
        plot_options=plot_options,
    )
    fig = axes[0][0].figure
    assert axes.shape == (5, 5)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_objective_1d():
    # Integration test of the plot_objective_1d function
    dimensions = [
        "Mass added [kg]",
        "Temperature [C]",
        "Conc. [M]",
        "Viscosity",
        "Categoric choice",
    ]
    space = [
        (15.0, 25.0),
        (70.0, 90.0),
        (0.03, 0.15),
        (70.0, 167.0),
        ["A", "B"],
    ]
    opt = Optimizer.Optimizer(
        space,
        acq_func="EI",
        n_initial_points=5,
    )
    x = [
        [18, 84, 0.042, 102.8, "A"],
        [24, 80, 0.09, 102.8, "B"],
        [20, 76, 0.066, 102.8, "A"],
        [16, 72, 0.111, 102.8, "A"],
        [22, 88, 0.138, 102.8, "B"],
        [24.5, 70, 0.145, 102.8, "A"],
        [19, 79, 0.124, 102.8, "B"],
        [15.5, 80, 0.100, 106, "B"],
        [24, 86, 0.111, 106, "B"],
        [18, 79, 0.124, 109, "A"],
        [16, 75, 0.145, 109, "B"],
        [17.9, 74, 0.150, 122.9, "A"],
        [22.6, 89, 0.075, 122.9, "B"],
        [23, 70, 0.145, 118.6, "A"],
        [23, 84, 0.042, 118.6, "B"],
    ]
    y = [54, 53, 38, 30, 65, 42, 60, 77, 66, 75, 61, 57, 87, 47, 67]
    res = opt.tell(x, y)
    axes = plot_objective_1d(
        res,
        pars="expected_minimum",
        show_confidence=True,
        usepartialdependence=False,
        dimensions=dimensions,
    )
    fig = axes[0][0].figure
    assert axes.shape == (2, 3)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_brownie_bee():
    # Integration test of the plot function for Brownie Bee
    
    max_q = 10
    # Helper function to convery y-values to stars
    def star_score(Y, max_stars, scale_min, scale_max):
        # Get the scale of the system
        y_scale = scale_max - scale_min
        # Map Y into the scale 0 to max_stars
        score = round((scale_max - Y) / y_scale * max_stars)
        # If Y is outside the system min/max values (can happen due to noise)
        # we clamp to the scale.
        score = min(max(score, 0), max_stars)        
        return score
    
    space = [
        (15.0, 25.0),
        (70.0, 90.0),
        (0.03, 0.15),
        (70.0, 167.0),
        ["A", "B"],
    ]
    opt = Optimizer.Optimizer(
        space,
        acq_func="EI",
        n_initial_points=5,
    )
    x = [
        [18, 84, 0.042, 102.8, "A"],
        [24, 80, 0.09, 102.8, "B"],
        [20, 76, 0.066, 102.8, "A"],
        [16, 72, 0.111, 102.8, "A"],
        [22, 88, 0.138, 102.8, "B"],
        [24.5, 70, 0.145, 102.8, "A"],
        [19, 79, 0.124, 102.8, "B"],
        [15.5, 80, 0.100, 106, "B"],
        [24, 86, 0.111, 106, "B"],
        [18, 79, 0.124, 109, "A"],
        [16, 75, 0.145, 109, "B"],
        [17.9, 74, 0.150, 122.9, "A"],
        [22.6, 89, 0.075, 122.9, "B"],
        [23, 70, 0.145, 118.6, "A"],
        [23, 84, 0.042, 118.6, "B"],
    ]
    y = [54, 53, 38, 30, 65, 42, 60, 77, 66, 75, 61, 57, 87, 47, 67]
    y_star = [-star_score(yy, max_q, 0, 100) for yy in y]
    res = opt.tell(x, y_star)
    fig_list = plot_brownie_bee_frontend(res, max_quality=max_q)
    
    # Check we built a list of figures
    for fig in fig_list:
        assert isinstance(fig, mpl.figure.Figure)
    # Check that we return the right number of figures in total
    assert len(fig_list) == len(space)+1
   
    # Check that the y-axis uses the correct limits
    limits = fig_list[0].gca().get_ylim()
    assert (limits[0] == 0) & (limits[1] == max_q*1.02)
    
    # Check that each x-axis of the factor plots uses the correct limits
    for i, limits in enumerate(space):
        xlim = fig_list[i].gca().get_xlim()
        # Categorical factors are special in their encoding
        if isinstance(opt.space.dimensions[i], Categorical):
            assert xlim[0] == -0.2
            assert xlim[1] == len(opt.space.dimensions[i].categories)-0.8
        else:
            # Numeric factors should simply map to the space bounds
            assert (xlim[0] == space[i][0]) & (xlim[1] == space[i][1])
    
    # Check that the histogram of expected outputs uses a sensible x-axis
    xlim = fig_list[-1].gca().get_xlim()
    assert (xlim[0] >= 0) & (xlim[1] <= max_q*1.02)
    


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
