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
from .optimizer import dummy_minimize
from .optimizer import forest_minimize
from .optimizer import gbrt_minimize
from .optimizer import gp_minimize
from .optimizer import Optimizer
from .searchcv import BayesSearchCV
from .space import Space
from .utils import dump
from .utils import expected_minimum
from .utils import expected_minimum_random_sampling
from .utils import load

__version__ = "0.7.1"


__all__ = (
    "acquisition",
    "benchmarks",
    "callbacks",
    "learning",
    "optimizer",
    "plots",
    "space",
    "gp_minimize",
    "dummy_minimize",
    "forest_minimize",
    "gbrt_minimize",
    "Optimizer",
    "dump",
    "load",
    "expected_minimum",
    "expected_minimum_random_sampling",
    "BayesSearchCV",
    "Space"
)
