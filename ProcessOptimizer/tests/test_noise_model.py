from typing import List

import numpy as np
from ProcessOptimizer.model_systems.noise_models import NoiseModel, ConstantNoise, ProportionalNoise, DataDependentNoise, ZeroNoise, SumNoise, noise_model_factory, parse_noise_model
from scipy.stats import norm, uniform
import pytest


@pytest.fixture
def signal_list():
    return [1,7,43,1212.21]

# If we are to fit the distributions, more data is needed
@pytest.fixture
def long_signal_list():
    return [1]*1000 + [10]*1000

# Setting the seed for the random number generator used so the tests are reproducible.
def setup_function():
    np.random.seed(42)

# Function for fitting a distribution and evaulating its mean and standard deviation.
def evaluate_random_dist(noise_list: List[float], size: float=1):
    (mean, stddev) = norm.fit(noise_list)
    assert np.allclose(mean, 0, atol=0.1*size)
    assert np.allclose(stddev, size, atol=0.1*size)

def test_noise_abstract():
    # Tests that the abstract class can not be instantiated.
    with pytest.raises(TypeError):
        NoiseModel()

def test_constant_noise(signal_list):
    noise_model = ConstantNoise()
    noise_model._noise_distribution = lambda: 2
    noise_list = [noise_model.get_noise(None, signal) for signal in signal_list]
    assert all([noise == 2 for noise in noise_list])

@pytest.mark.parametrize("noise_level",(1,2,3))
def test_proportional_noise(signal_list, noise_level):
    noise_model = ProportionalNoise(noise_size=1)
    noise_model._noise_distribution = lambda: noise_level
    noise_list = [noise_model.get_noise(None, signal) for signal in signal_list]
    rel_noise_list = [noise/signal for (signal, noise) in zip(signal_list, noise_list)]
    assert np.allclose(rel_noise_list, noise_level)

def test_proportional_noise_default(long_signal_list):
    noise_model = ProportionalNoise()
    noise_list = [noise_model.get_noise(None, signal) for signal in long_signal_list]
    rel_noise_list = [noise/signal
                       for (signal, noise) in zip(long_signal_list, noise_list)]
    evaluate_random_dist(rel_noise_list, size=0.1)

def test_proportional_noise_given_size(long_signal_list):
    noise_model = ProportionalNoise(noise_size = 0.1)
    noise_list = [noise_model.get_noise(None, signal) for signal in long_signal_list]
    rel_noise_list = [noise/signal 
                      for (signal,noise) in zip(long_signal_list, noise_list)]      
    evaluate_random_dist(rel_noise_list, size=0.1)

def test_random_noise(long_signal_list):
    noise_model = ConstantNoise()
    noise_list = [noise_model.get_noise(None, signal) for signal in long_signal_list]
    evaluate_random_dist(noise_list)

@pytest.mark.parametrize("size", (1, 2, 47))
def test_noise_size_constant(size, long_signal_list):
    noise_model = ConstantNoise(noise_size=size)
    noise_list = [noise_model.get_noise(None, signal) for signal in long_signal_list]
    evaluate_random_dist(noise_list,size)

@pytest.mark.parametrize("size", (1, 2, 47))
def test_noise_size_proportional(size, long_signal_list):
    noise_model = ProportionalNoise(noise_size=size)
    noise_list = [noise_model.get_noise(None, signal) for signal in long_signal_list]
    rel_noise_list = [noise/signal
                       for (signal, noise) in zip(long_signal_list, noise_list)]
    evaluate_random_dist(rel_noise_list, size)

@pytest.mark.parametrize("input", (1, 2, 3))
def test_data_dependent_noise(signal_list, input):
    # noise_choice returns a noise model that gives the same noise as the input data.
    def noise_choice(X):
        local_noise_model = ConstantNoise()
        local_noise_model._noise_distribution = lambda: X
        return local_noise_model
    noise_model = DataDependentNoise(noise_function=noise_choice)
    noise_list = [noise_model.get_noise(input, signal) for signal in signal_list]  
    assert [noise==signal for (signal, noise) in zip(signal_list, noise_list)]

def test_zero_noise(signal_list):
    noise_model = ZeroNoise()
    noise_list = [noise_model.get_noise(None, signal) for signal in signal_list]    
    assert [noise==0 for noise in noise_list]

# Testing that the examples in the docstring of DataDependentNoise work
@pytest.mark.parametrize("magnitude",(1,2,3))
def test_noise_model_example_1(long_signal_list, magnitude):
    # the following two lines are taken from the docstring of DataDependentNoise
    noise_choice = lambda X: ConstantNoise(noise_size=X, seed=None)
    noise_model = DataDependentNoise(noise_function=noise_choice)
    data = [magnitude]*len(long_signal_list)
    noise_list = [noise_model.get_noise(x, signal) 
                  for (x, signal) in zip(data,long_signal_list)]
    evaluate_random_dist(noise_list, magnitude)

def test_noise_model_example_2(long_signal_list):
    # the following two lines are taken from the docstring of DataDependentNoise
    noise_choice = lambda X: ZeroNoise() if X[0]==0 else ConstantNoise(seed=None)
    noise_model = DataDependentNoise(noise_function=noise_choice)
    X = [0, 10, 5]
    noise_list = [noise_model.get_noise(X, signal) for signal in long_signal_list]
    (mean, stddev) = norm.fit(noise_list)
    # This yields the zero noise model, where the fitted parameters are exactly zero
    assert mean==0
    assert stddev==0
    X = [1, 27, 53.4]
    noise_list = [noise_model.get_noise(X, signal) for signal in long_signal_list]
    evaluate_random_dist(noise_list)

def test_sum_noise(signal_list):
    noise_model = SumNoise(noise_model_list=[
        "constant",
        {"model_type" : "proportional", "noise_size" : 1}
    ])
    noise_model.noise_model_list[0]._noise_distribution = lambda: 2
    noise_model.noise_model_list[1]._noise_distribution = lambda: 3
    noise_list = [noise_model.get_noise(None, signal) for signal in signal_list]
    true_noise_list = [2 + 3 * signal for signal in signal_list]
    assert noise_list == true_noise_list

# _sample_noise of SumNoise should not be accesible, since it is a composite NoiseModel,
# and should ot have "its own" noise; it does not need it to retrun the sum of its
# underlying noise models.
def test_sum_noise_raw_noise_error():
    noise_model = SumNoise(noise_model_list=[ConstantNoise()])
    with pytest.raises(TypeError):
        noise_model._sample_noise

# _sample_noise of DataDependentNoise should not be accesible, since it is a composite
# NoiseModel, and should ot have "its own" noise.
def test_data_dependent_raw_noise_error():
    noise_choise = lambda: ConstantNoise()
    noise_model = DataDependentNoise(noise_function=noise_choise)
    with pytest.raises(TypeError):
        noise_model._sample_noise

def test_factory_constant():
    noise_model = noise_model_factory(model_type="constant")
    assert isinstance(noise_model,ConstantNoise)

def test_factory_proportional():
    noise_model = noise_model_factory(model_type="proportional")
    assert isinstance(noise_model,ProportionalNoise)

def test_factory_zero():
    noise_model = noise_model_factory(model_type="zero")
    assert isinstance(noise_model,ZeroNoise)

def test_factory_other():
    with pytest.raises(ValueError):
        noise_model_factory(model_type="not_implemented")

def test_factory_size():
    noise_model = noise_model_factory(model_type="constant", noise_size=3)
    assert noise_model.noise_size == 3

def test_parse_model():
    noise_model = parse_noise_model(ConstantNoise())
    assert isinstance(noise_model,ConstantNoise)

def test_pars_str():
    noise_model = parse_noise_model("constant")
    assert isinstance(noise_model,ConstantNoise)

def test_parse_dict():
    noise_model = parse_noise_model({"model_type": "constant"})
    assert isinstance(noise_model,ConstantNoise)

def test_uniform_noise(long_signal_list):
    noise_model = ConstantNoise()
    noise_model.set_noise_type("uniform")
    noise_list = [noise_model.get_noise(None, Y) for Y in long_signal_list]
    (start,width) = uniform.fit(noise_list)
    assert np.allclose(start, -1, atol=0.1)
    assert np.allclose(width, 2, atol=0.1)

def test_unknown_distribution():
    noise_model = ConstantNoise()
    with pytest.raises(ValueError):
        noise_model.set_noise_type("not_implemented")

# set_noise_model_list did not reset the list. This test verifies that this bug has been
# fixed.
def test_reset_model_list():
    noise_model = SumNoise(noise_model_list=[ConstantNoise(noise_size=100)])
    noise_model.set_noise_model_list([ConstantNoise()])
    noise_list = [noise_model.get_noise(None, None) for _ in range(30)]
    # It is very unlikely that a normally distributed noise with size 1 is above 10. It
    # is unlikely none of 30 normally distributed noise reading with size 100 is above
    # 10
    assert all([noise<10 for noise in noise_list])

def test_noise_seeding():
    # Test that two noise models created with no seeding give different values
    noise_model_1 = ConstantNoise(seed=None)
    noise_model_2 = ConstantNoise(seed=None)
    assert noise_model_1.get_noise(None, 0) != noise_model_2.get_noise(None, 0)
    # Test that two noise models created with the same seed give the same value
    noise_model_1 = ConstantNoise(seed=1)
    noise_model_2 = ConstantNoise(seed=1)
    assert noise_model_1.get_noise(None, 0) == noise_model_2.get_noise(None, 0)

def test_noise_seed_reset():
    # Test that you can reset the seed of a noise model after drawing noise
    noise_model = ConstantNoise(seed=1)
    # Sample noise value one time and store
    first_noise = noise_model.get_noise(None, 0)
    # Advance the noise sampling
    second_noise = noise_model.get_noise(None, 0)
    # Reset to the same seed
    noise_model.set_seed(1)
    reset_noise = noise_model.get_noise(None, 0)
    assert first_noise == reset_noise
    assert first_noise != second_noise