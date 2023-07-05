import numpy as np
from .model_system import ModelSystem
from .benchmarks import branin
from ..space import Real


# Define the relevant parameter space
space = []
space += [Real(-5,10,name='x1')]
space += [Real(0,15,name='x2')]

# Set the true minimum value
true_min = 0.397887

# Create model system
branin_hoo = ModelSystem(score, space, true_min)