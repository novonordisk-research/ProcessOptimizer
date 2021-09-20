"""
`ProcessOptimizer`, is a simple and efficient library to minimize (very)
expensive and noisy black-box functions. It implements several methods for
sequential model-based optimization. `ProcessOptimizer` is reusable in many
contexts and accessible.

## Install

```
pip install ProcessOptimizer
```

## Getting started

Find the minimum of the noisy function `f(x)` over the range `-2 < x < 2`
with `ProcessOptimizer`:

```python
import numpy as np
from ProcessOptimizer import gp_minimize

def f(x):
    return (np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) *
            np.random.randn() * 0.1)

res = gp_minimize(f, [(-2.0, 2.0)])
```

## Development

The library is still experimental and under heavy development.

"""

from . import acquisition
from . import benchmarks
from . import callbacks
from . import learning
from . import optimizer

from . import space
from .optimizer import Optimizer
from .space import Space
from .utils import dump
from .utils import expected_minimum
from .utils import expected_minimum_random_sampling
from .utils import load
from .utils import cook_estimator, create_result, y_coverage
from .plots import plot_objective, plot_objectives
from .plots import plot_evaluations, plot_convergence
from .plots import plot_Pareto, plot_expected_minimum_convergence

__version__ = "0.7.1"


__all__ = (
    "acquisition",
    "benchmarks",
    "callbacks",
    "learning",
    "optimizer",
    "plots",
    "space",
    "Optimizer",
    "dump",
    "load",
    "cook_estimator",
    "create_result",
    "expected_minimum",
    "expected_minimum_random_sampling",
    "Space",
    "plot_objective",
    "plot_objectives",
    "plot_evaluations",
    "plot_convergence",
    "plot_Pareto",
    "y_coverage",
    "plot_expected_minimum_convergence"
)
