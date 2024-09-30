# Feature demonstration

This folder contains Jupyter notebooks that demonstrate certain features or workflow
from the `ProcessOptimizer` package.

Content here (both in this file and in the folder) is given in no particular order,
but hopefully with self-explanatory names.

## Multiobjective optimization

Many traditional optimization methods only allow for one optimization target. If you have more than one optimization target, e.g., quality and price, this is not sufficient. If you don't want to make a single synthetic objective that takes into account all relevant objectives, [here](multiobjective.ipynb) is a notebook that describes how you can use `ProcessOptimizer` to map out the connection between the different objectives and find the Pareto front.

## Independent variables

Control parameters are the independent variables that can be controlled in the
optimization. [Here](control_parameters.ipynb) is a notebook that
demonstrate how the control parameter settings are defined.

Control parameter settings can be [sampled](sampling_control_parameters.ipynb)
in different ways.
