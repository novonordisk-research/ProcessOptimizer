import unittest

import numpy as np
from ProcessOptimizer.model_systems import (
    branin,
    branin_no_noise,
    hart3_no_noise,
    hart6_no_noise,
    poly2_no_noise,
    peaks_no_noise,
    ProportionalNoise,
)
from ProcessOptimizer import Real


class TestIndividualModelSystem(unittest.TestCase):
    # Make a test for the branin ModelSystem
    def test_branin(self):
        assert branin.noise_size == 0.1
        assert type(branin.noise_model) == ProportionalNoise
        assert len(branin.space.dimensions) == 2
        assert branin.space.dimensions[0].name == "x1"
        assert branin.space.dimensions[1].name == "x2"
        assert isinstance(branin.space.dimensions[0], Real)
        assert isinstance(branin.space.dimensions[0], Real)
        assert branin.space.dimensions[0].bounds == (-5, 10)
        assert branin.space.dimensions[1].bounds == (0, 15)
        self.assertAlmostEqual(branin.get_score([0, 0]), 57.29640398152322, places=5)
        # This also tests the seeding of the random state
        xstars = np.asarray([(-np.pi, 12.275), (+np.pi, 2.275), (9.42478, 2.475)])
        f_at_xstars = np.asarray([branin_no_noise.get_score(xstar) for xstar in xstars])
        for value in f_at_xstars:
            self.assertAlmostEqual(value, 0.397887, places=5)

    def test_hart3(self):
        assert hart3_no_noise.noise_size == 0
        assert hart3_no_noise.space.bounds == [(0, 1), (0, 1), (0, 1)]

    def test_hart6(self):
        assert hart6_no_noise.noise_size == 0
        assert hart6_no_noise.space.bounds == [
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
        ]
        x_test = np.asarray((0.20169, 0.15001, 0.476874, 0.275332, 0.311652, 0.6573))
        self.assertAlmostEqual(hart6_no_noise.get_score(x_test), -3.32237, places=5)

    def test_poly2(self):
        assert poly2_no_noise.noise_size == 0
        assert poly2_no_noise.space.bounds == [(-1, 1), (-1, 1)]
        x_test = np.asarray((0.6667, -0.4833))
        self.assertAlmostEqual(poly2_no_noise.get_score(x_test), -2.0512, places=4)
        x_test_max = np.asarray((-1.0, -1.0))
        self.assertAlmostEqual(poly2_no_noise.get_score(x_test_max), -1.270, places=3)

    def test_peaks(self):
        assert peaks_no_noise.noise_size == 0
        assert peaks_no_noise.space.bounds == [(-3.0, 3.0), (-3.0, 3.0)]
        x_test = np.asarray((0.228, -1.626))
        self.assertAlmostEqual(peaks_no_noise.get_score(x_test), -6.5511, places=4)
        x_test_max = np.asarray((-0.010, 1.581))
        self.assertAlmostEqual(peaks_no_noise.get_score(x_test_max), 8.106, places=3)
