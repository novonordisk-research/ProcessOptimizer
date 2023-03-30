import numpy as np
from ProcessOptimizer.model_systems import ModelSystem
from ProcessOptimizer.model_systems.benchmarks import poly_2d
from ProcessOptimizer.space import Real

# This system is intended to run in the domain x[0] in [-1,1] and x[1] in [-1,1]. This function
# has its minimum in this domain at (0.6667,-0.4833), with a value of 
# about -2.0512, and its maximum in this domain at (-1,-1) with a value of -1.27
def score(x, rng=np.random.default_rng(), noise_std=0.02):
    """Calculate the score of the model system

    Parameters:
    * x [array of length 2 containing numbers between -1 and 1]:
        The point in the parameter space at which to calculate the score. 
    * rng [np.random Generator]:
        Random number generator used to generate noise. 
    * noise_std [float]: 
        Standard deviation of the noise which is added to the result of the polynomial. 
    """
    value = poly_2d(x) + rng.normal(scale=noise_std)
    return value

# The location of the minimum in the parameter space:
true_min_loc = [2/3, -0.4833]
# The true minimum value
true_min = score(true_min_loc, noise_std=0)

# Define a the domain as a Space object
space = []
space += [Real(-1, 1, name='x1')]
space += [Real(-1, 1, name='x2')]

# Create model system
polynomial_system_2d = ModelSystem(score, space, true_min)