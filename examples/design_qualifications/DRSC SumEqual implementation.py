# -*- coding: utf-8 -*-
"""
A script demonstrating the features of the DRSC algorithm for use with the
SumEquals constraint in ProcessOptimizer.

Author: Morten Bormann Nielsen, Danish Technological Institute
December 2025
"""


import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from ProcessOptimizer import Optimizer
from ProcessOptimizer.space import Real
from ProcessOptimizer.space.constraints import SumEquals

from ProcessOptimizer.model_systems import hart3
from ProcessOptimizer.plots import plot_objective_1d

#%% Setup of SumEquals sampling for very oblong space

# The DRSC generator for SumEquals sampling is able to generate valid points
# extremely fast
space = [
    (52.9, 96.3),
    (0.1, 0.5),
    (3.6, 6.6),
    (1.0, 5.0),
    (1.0, 35.0),        
]
# Define the constraint as all factor settings adding up to 100
cons = [SumEquals(dimensions=[0, 1, 2, 3, 4], value=100.0)]
opt = Optimizer(space, lhs=False, n_initial_points=10)
opt.set_constraints(cons)

# Extract the generator itself
constraint = opt.get_constraints().sum_equals[0]
drsc_gen = constraint.get_drsc_generator(opt.space)
# Demonstrate speed of algorithm by generating 10,000 points. This call takes
# about 12 seconds on an Intel i7-12850HX.
start_time = time.time()
x_simplex = drsc_gen.generate_sample(10000)
print("Generated 10,000 points in the space in %s seconds" % (time.time() - start_time))

#%% Demonstrate typical use for a real user with fewer points

# Define the space
space = [
    (52.9, 96.3),
    (0.1, 0.5),
    (3.6, 6.6),
    (1.0, 5.0),
    (1.0, 35.0),        
]
# Set up the constraint
cons = [SumEquals(dimensions=[0, 1, 2, 3, 4], value=100.0)]
opt = Optimizer(space, lhs=False, n_initial_points=10, random_state=31031988)
opt.set_constraints(cons)
# Ask for settings for the initial experiments
print(time.strftime("Starting opt.ask calculation with DRSC method at:") + " " + time.strftime("%H:%M:%S"))
x = opt.ask(10, strategy="cl_min")
print(time.strftime("Finished opt.ask calculation with DRSC method at:") + " " + time.strftime("%H:%M:%S"))

#%% Demonstrate that different seeds lead to different points

opt1 = Optimizer(space, lhs=False, n_initial_points=10, random_state=1)
opt1.set_constraints(cons)
opt2 = Optimizer(space, lhs=False, n_initial_points=10, random_state=2)
opt2.set_constraints(cons)

x1 = opt1.ask(10, strategy="cl_min")
x2 = opt2.ask(10, strategy="cl_min")

x!=x2

#%% Demonstrate that identical seeds lead to identical points

constraint1 = opt1.get_constraints()
constraint2 = opt2.get_constraints()

x1 = constraint1.sumequal_sampling(n_samples=1000, random_state=31031988)
x2 = constraint2.sumequal_sampling(n_samples=1000, random_state=31031988)

x1 == x2

#%% Demonstrate how to use SumEqual constraints when categoricals are present
space = [
    ("A", "B"),
    (52.9, 96.3),
    (0.1, 0.5),
    (3.6, 6.6),
    (1.0, 5.0),
    (1.0, 35.0),
    ("C", "D", "E"),
    (1, 5),
]
cons = [SumEquals(dimensions=[1, 2, 3, 4, 5], value=100.0)]
opt = Optimizer(space, lhs=False, n_initial_points=10)
opt.set_constraints(cons)

print(time.strftime("Starting opt.ask calculation with DRSC method at:") + " " + time.strftime("%H:%M:%S"))
x = opt.ask(10, strategy="cl_min")
print(time.strftime("Finished opt.ask calculation with DRSC method at:") + " " + time.strftime("%H:%M:%S"))

#%% Use the new sampling method with simulated data

# Build Hartmann 3D ModelSystem object
hart3_model = hart3.create_hart3(noise=True)
# Define space
space = [
    Real(0., 1., name='x0'),
    Real(0., 1., name='x1'),
    Real(0., 1., name='x2'),
]

cons = [SumEquals(dimensions=[0, 1, 2], value=1.0)]
opt = Optimizer(space, lhs=False, n_initial_points=20)
opt.set_constraints(cons)

# Run optimization, first through the initial points, then using EI  
for _ in range(30):
    x = opt.ask(5, strategy="cl_min")
    y = [hart3_model.get_score(x) for x in x]
    res = opt.tell(x, y)

# Show the system
plot_objective_1d(res, pars="expected_minimum")

# Show the location of the sampled points in this experiment
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d')
cmap = mpl.colormaps["viridis"].resampled(len(opt.Xi))

idx = np.arange(len(opt.Xi))
x = np.array(opt.Xi)
y = np.array(opt.yi)

ax.scatter(x[:,0], x[:, 1], x[:, 2], c=idx, cmap=cmap, alpha=1)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)


#%% Demonstrate the use of a non-linear SumEquals constraint 

# Define space
space = [
    Real(0., 1., name='x0'),
    Real(0., 1., name='x1'),
    Real(0., 1., name='x2'),
]

sum_value = 1.0

# Nonlinear constraint: x0 * x1 < 0.1
# Convention: function returns value that should be <= 0 when SATISFIED
def product_constraint(x):
    """x[0] * x[1] <= 0.1"""
    return x[0] * x[1] - 0.1

# Create SumEquals constraint with nonlinear constraint
cons = [
    SumEquals(
        dimensions=[0, 1, 2],
        value=sum_value,
        nonlinear_constraints=[product_constraint]
    )
]

# Create optimizer
opt = Optimizer(space, lhs=False, n_initial_points=10)
opt.set_constraints(cons)

# Test sampling
constraint = opt.get_constraints().sum_equals[0]
drsc_gen = constraint.get_drsc_generator(opt.space)

# Generate samples
samples = drsc_gen.generate_sample(1000)

# Convert to original space and verify constraints
samples_original = samples * sum_value

x = samples_original
# Create a plot showing where the samples lie in the space
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2],)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)


#%% Demonstrate the use of three constrained dimensions and a categorical

# Sample 1000 points and plot them to show it works
space = [
    (0., 10.),
    (0., 10.),
    (0., 10.),
    ("A", "B", "C"),
]

cons = [SumEquals(dimensions=[0, 1, 2], value=15.0)]
opt = Optimizer(space, lhs=False, n_initial_points=100)
opt.set_constraints(cons)

constraint = opt.get_constraints().sum_equals[0]

x = opt.ask(100, strategy="cl_min")
colors = []
for xx in x:
    if xx[3] == "A":
        colors.append("r")
    elif xx[3] == "B":
        colors.append("g")
    else:
        colors.append("b")
    
# Create a plot showing the distribution of points in this space
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d')
[ax.scatter(x[0], x[1], x[2], c=colors[i]) for i, x in enumerate(x)]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0, 10)
