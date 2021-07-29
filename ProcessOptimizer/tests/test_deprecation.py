import pytest

from ProcessOptimizer import Optimizer
from ProcessOptimizer.learning import ExtraTreesRegressor



def test_n_random_starts_Optimizer():
    # n_random_starts got renamed in v0.4
    et = ExtraTreesRegressor(random_state=2)
    with pytest.deprecated_call():
        Optimizer([(0, 1.)], et, n_random_starts=10, acq_optimizer='sampling')
