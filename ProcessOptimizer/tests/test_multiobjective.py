import pytest
import numpy as np

from numpy.testing import assert_raises
from numpy.testing import assert_equal


from ProcessOptimizer import Optimizer
from ProcessOptimizer.model_systems import get_model_system


@pytest.mark.fast_test
def test_multiobjective_tell():
    opt = Optimizer([[0.0, 1.0], [0.0, 1.0]], n_objectives=2)
    x = [0, 0]
    assert_raises(ValueError, opt.tell, x, 1)
    opt.tell(x, [1, 1])
    assert_raises(ValueError, opt.tell, x, [1, 1, 1])

    assert_raises(ValueError, opt.tell, [x, x], [[1], [1]])
    opt.tell([x, x], [[1, 1], [1, 1]])
    assert_raises(ValueError, opt.tell, [x, x], [[1, 1, 1], [1, 1, 1]])


@pytest.mark.fast_test
def test_singleobjective_tell():
    opt = Optimizer([[0.0, 1.0], [0.0, 1.0]], n_objectives=1)
    x = [0, 0]
    opt.tell(x, 1)
    assert_raises(ValueError, opt.tell, x, [1, 1])

    opt.tell([x, x], [1, 1])
    assert_raises(ValueError, opt.tell, [x, x], [[1, 1], [1, 1]])


@pytest.mark.fast_test
def test_Pareto_in_space():
    opt = Optimizer(
        [[0.0, 1.0], [0.0, 1.0]], n_objectives=2, n_initial_points=1
    )
    x = [0, 0]
    opt.tell(x, [1, 1])

    # Calculate Pareto front
    pop, logbook, front = opt.NSGAII()
    pop = np.asarray(pop)
    # Assert that Pareto points are in space
    for x in pop:
        assert_equal(opt.space.__contains__(x), True)


@pytest.mark.fast_test
def test_Pareto_reproducible():
    gold_model_system = get_model_system('gold_map')
    distance_model_system = get_model_system('distance_map', camp_coordinates=(4,10))

    opt = Optimizer(gold_model_system.space, n_initial_points=4, n_objectives=2)

    gold_model_system.noise_model.set_seed(40)
    distance_model_system.noise_model.set_seed(40)

    for i in range(10):
        new_dig_site = opt.ask()
        gold_found = gold_model_system.get_score(new_dig_site)
        distance = distance_model_system.get_score(new_dig_site)
        opt.tell(new_dig_site, [gold_found, distance])
    
    # Calculate Pareto front twice and compare
    pop1, logbook, front1 = opt.NSGAII()
    pop2, logbook, front2 = opt.NSGAII()
    assert_equal(pop1, pop2)
    assert_equal(front1, front2)