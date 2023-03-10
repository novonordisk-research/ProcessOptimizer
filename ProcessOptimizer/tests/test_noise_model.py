import numpy as np
from ProcessOptimizer.model_systems.noise_models import NoiseModel, AdditiveNoise, MultiplicativeNoise, DataDependentNoise, ZeroNoise
from  scipy.stats import norm
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
def evaluate_random_dist(noise_list: list[float],size: float=1):
    (mean, spread) = norm.fit(noise_list)
    assert np.allclose(mean,0,atol=0.1*size)
    assert np.allclose(spread,size,atol=0.1*size)

def test_noise_abstract():
    # Tests that the abstract class can not be instantiated.
    with pytest.raises(TypeError):
        NoiseModel()

def test_additive_noise(signal_list):
    noise_model = AdditiveNoise(noise_dist=lambda: 2)
    noisy_list = [noise_model.apply(None,signal) for signal in signal_list]
    noise_list = [noise - signal for (signal,noise) in zip(signal_list, noisy_list)]
    assert all([noise == 2 for noise in noise_list])

@pytest.mark.parametrize("noise_level",(1,2,3))
def test_multiplicative_noise(signal_list, noise_level):
    noise_model = MultiplicativeNoise(noise_dist=lambda: noise_level)
    noisy_list = [noise_model.apply(None,signal) for signal in signal_list]
    noise_list = [noisy/signal-1 for (signal,noisy) in zip(signal_list, noisy_list)]
    assert np.allclose(noise_list,noise_level/100 )

def test_random_noise(long_signal_list):
    noise_model = AdditiveNoise()
    noise_list = [noise_model.apply(None,signal) - signal for signal in long_signal_list]
    evaluate_random_dist(noise_list)

@pytest.mark.parametrize("size",(1,2,47))
def test_noise_size_additive(size, long_signal_list):
    noise_model = AdditiveNoise(noise_size=size)
    noise_list = [noise_model.apply(None,signal) - signal for signal in long_signal_list]
    evaluate_random_dist(noise_list,size)

@pytest.mark.parametrize("size",(1,2,47))
def test_noise_size_multiplicative(size, long_signal_list):
    noise_model = MultiplicativeNoise(noise_size=size)
    relative_noise_list = [(noise_model.apply(None,signal)-signal)/signal
                           for signal in long_signal_list]
    evaluate_random_dist(relative_noise_list,size)


@pytest.mark.parametrize("a,b,c",[(1,2,3),(4,2,-1),(10,0,-10)])
def test_noise_model_composition(a,b,c, signal_list):
    noise_model = AdditiveNoise(
        underlying_noise_model=MultiplicativeNoise(
            underlying_noise_model=AdditiveNoise(
                noise_dist=lambda: a),
            noise_dist=lambda: b),
        noise_dist=lambda: c)
    noisy_list = [noise_model.apply(None,signal) for signal in signal_list]
    true_list = [(signal + a) * (1+b/100) + c for signal in signal_list]
    assert all([noisy == true for (true, noisy) in zip(true_list,noisy_list)])

@pytest.mark.parametrize("input",(1,2,3))
def test_data_dependent_noise(signal_list, input):
    # noice_choice returns a noice model that gives the same noise as the input data
    noise_choice = lambda X: AdditiveNoise(noise_dist = lambda: X)
    noise_model = DataDependentNoise(noise_models=noise_choice)
    noisy_list = [noise_model.apply(input,signal) for signal in signal_list]
    assert [noisy==input + signal for (signal,noisy) in zip(signal_list,noisy_list)]

def test_zero_noice(signal_list):
    noise_model = ZeroNoise()
    noisy_list = [noise_model.apply(None,signal) for signal in signal_list]
    assert [noisy==signal for (signal,noisy) in zip(signal_list,noisy_list)]