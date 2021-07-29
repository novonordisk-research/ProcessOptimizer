import pytest
import numpy as np

from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_raises
from numpy.testing import assert_equal



from ProcessOptimizer import Optimizer



@pytest.mark.fast_test
def test_multiobjective_tell():
    opt = Optimizer([[0.,1.],[0.,1.]], n_objectives=2)
    x= [0,0]
    assert_raises(ValueError, opt.tell, x, 1)
    opt.tell(x, [1,1])
    assert_raises(ValueError, opt.tell, x, [1,1,1])
    
    assert_raises(ValueError, opt.tell, [x,x], [[1],[1]])
    opt.tell([x,x], [[1,1],[1,1]])
    assert_raises(ValueError, opt.tell, [x,x], [[1,1,1],[1,1,1]])



@pytest.mark.fast_test
def test_singleobjective_tell():
    opt = Optimizer([[0.,1.],[0.,1.]], n_objectives=1)
    x= [0,0]
    opt.tell(x, 1)
    assert_raises(ValueError, opt.tell, x, [1,1])
    
    opt.tell([x,x], [1,1])
    assert_raises(ValueError, opt.tell, [x,x], [[1,1],[1,1]])



@pytest.mark.fast_test
def test_Pareto_in_space():
    opt = Optimizer([[0.,1.],[0.,1.]], n_objectives=2, n_initial_points=1)
    x= [0,0]
    opt.tell(x, [1,1]);
    
    # Calculate Pareto front
    pop, logbook, front = opt.NSGAII()
    pop = np.asarray(pop)
    # Assert that Pareto points are in space
    for x in pop:
        assert_equal(opt.space.__contains__(x), True)




