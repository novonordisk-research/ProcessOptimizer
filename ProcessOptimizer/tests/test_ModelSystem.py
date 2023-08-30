import unittest

from ProcessOptimizer.model_systems import ModelSystem


class TestModelSystem(unittest.TestCase):
    def test_default_noise(self):
        test_system = ModelSystem(
            score=lambda x: x,
            space=[],
            noise_model="constant",
            true_max=4.7,
            true_min=-1,
        )
        # If no noise sie is given, noise size should be 1 % of the span of the score.
        assert test_system.noise_size == 0.057

        test_system = ModelSystem(
            score=lambda x: x,
            space=[],
            noise_model={"model_type": "constant"},
            true_max=4.7,
            true_min=-1,
        )

        assert test_system.noise_size == 0.057

        test_system = ModelSystem(
            score=lambda x: x,
            space=[],
            noise_model={"model_type": "constant", "noise_size": 1},
            true_max=4.7,
            true_min=-1,
        )
        # If noise size is given, this should be used, even if it is the same as the default.
        assert test_system.noise_size == 1
