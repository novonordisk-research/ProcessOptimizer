import unittest

import numpy as np
from ProcessOptimizer.model_systems import branin, branin_no_noise, hart3_no_noise
from ProcessOptimizer.space import Real


class TestModelSystem(unittest.TestCase):
    # Make a test for the branin ModelSystem
    def test_branin(self):
        assert branin.noise_size == 0.02
        assert len(branin.space.dimensions) == 2
        assert branin.space.dimensions[0].name == "x1"
        assert branin.space.dimensions[1].name == "x2"
        assert isinstance(branin.space.dimensions[0], Real)
        assert isinstance(branin.space.dimensions[0], Real)
        assert branin.space.dimensions[0].bounds == (-5, 10)
        assert branin.space.dimensions[1].bounds == (0, 15)
        self.assertAlmostEqual(branin.get_score([0, 0]), 55.60820698386535, places=5)
        # This also tests the seeding of the random state
        xstars = np.asarray([(-np.pi, 12.275), (+np.pi, 2.275), (9.42478, 2.475)])
        f_at_xstars = np.asarray([branin_no_noise.get_score(xstar) for xstar in xstars])
        for value in f_at_xstars:
            self.assertAlmostEqual(value, 0.397887, places=5)

    def test_hart3(self):
        assert hart3_no_noise.noise_size == 0
        assert hart3_no_noise.space.bounds == [(0, 1), (0, 1), (0, 1)]
