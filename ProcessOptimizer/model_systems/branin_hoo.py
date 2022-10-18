from ProcessOptimizer.model_systems import ModelSystem
from ProcessOptimizer.model_systems.benchmarks import branin
from ProcessOptimizer.space import Real

# Take the score function from the benchmarks file
score = branin

# Define the relevant parameter space
space = []
space += [Real(-5,10,name='x1')]
space += [Real(0,15,name='x2')]

# Set the true minimum value
true_min = 0.397887

# Create model system
branin_hoo = ModelSystem(score, space, true_min)