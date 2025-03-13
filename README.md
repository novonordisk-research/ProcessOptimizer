<div align="center">
<pre>
  _____                              ____        _   _           _              
 |  __ \                            / __ \      | | (_)         (_)             
 | |__) | __ ___   ___ ___  ___ ___| |  | |_ __ | |_ _ _ __ ___  _ _______ _ __ 
 |  ___/ '__/ _ \ / __/ _ \/ __/ __| |  | | '_ \| __| | '_ ` _ \| |_  / _ \ '__|
 | |   | | | (_) | (_|  __/\__ \__ \ |__| | |_) | |_| | | | | | | |/ /  __/ |   
 |_|   |_|  \___/ \___\___||___/___/\____/| .__/ \__|_|_| |_| |_|_/___\___|_|   
                                          | |                                   
                                          |_|                                   
</pre>
<a href="https://badge.fury.io/py/ProcessOptimizer"><img src="https://badge.fury.io/py/ProcessOptimizer.svg" alt="PyPI version"></a>
<a href="https://github.com/novonordisk-research/ProcessOptimizer/actions"><img src="https://github.com/novonordisk-research/ProcessOptimizer/actions/workflows/python-package-tests.yml/badge.svg" alt="Tests"></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/built%20with-Python3-green.svg" alt="built with Python3" /></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue" alt="Runs on" /></a>
<a href="https://github.com/novonordisk-research/ProcessOptimizer/blob/develop/LICENSE.md"><img src="https://img.shields.io/pypi/l/ProcessOptimizer" alt="PyPI - License" /></a>
<a href="https://scikit-optimize.github.io/stable/"><img src="https://img.shields.io/badge/BuildOn-Scikit--Optimize-brightgreen" alt="Scikit-Optimize" /></a>
<a href="https://doi.org/10.5281/zenodo.5155295"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.5155295.svg" alt="DOI"></a>

[![Downloads](https://static.pepy.tech/personalized-badge/processoptimizer?period=total&units=international_system&left_color=brightgreen&right_color=orange&left_text=Downloads)](https://pepy.tech/project/processoptimizer)
</div>

----------

## Table of Contents
 * [ProcessOptimizer](https://github.com/novonordisk-research/ProcessOptimizer/blob/develop/README.md#processoptimizer)
 * [Installation](https://github.com/novonordisk-research/ProcessOptimizer/blob/develop/README.md#installation)
 * [How to get started](https://github.com/novonordisk-research/ProcessOptimizer/blob/develop/README.md#how-to-get-started)
 * [Examples](https://github.com/novonordisk-research/ProcessOptimizer/blob/develop/README.md#examples)
 * [Contributions](https://github.com/novonordisk-research/ProcessOptimizer/blob/develop/README.md#contributions)
 * [Related work](https://github.com/novonordisk-research/ProcessOptimizer/blob/develop/README.md#related-work)
 * [Citation](https://github.com/novonordisk-research/ProcessOptimizer/blob/develop/README.md#Citation)
 

## ProcessOptimizer

ProcessOptimizer is intended and tailored for optimization of real world processes. This could e.g. be some complex chemical reaction where no reliable analytical model mapping input variables to the output is readily available. Functionality includes Bayesian optimization, space filling, Design of Experiment algorithms, multi-objective optimization, and more. <br/>
<br/> 
ProcessOptimizer is tailored to perform well when observables have non-neglible noise but where the underlying relations between factors and responses follow regular real world behavior. 
 <br/>

## Installation

ProcessOptimizer can be installed using `pip install ProcessOptimizer`<br/>
The repository and examples can be found at https://github.com/novonordisk-research/ProcessOptimizer<br/>
ProcessOptimizer can also be installed by running `pip install -e .` in top directory of the cloned repository.

## How to get started
Below is an illustrative example of minimization of the 2-dimensional [Booth function](https://www.sfu.ca/~ssurjano/booth.html) using the `ProcessOptimizer` package. Notice that in real world applications, we would not know this function beforehand, i.e., it would be "black-box" (and typically we would also have more than 2 input factors). <br/>
In this example, uniformly distributed random noise between 0-5% of the function value is added using `np.random`. The function is defined as follows:
```python
import numpy as np

def Booth(x0, x1):
    booth = (x0 + 2 * x1 - 7)**2 + (2 * x0 + x1 - 5)**2
    noise = 1 + 0.05 * (2 * np.random.rand() - 1)
    return (booth * noise)
```
<!--
Below is an image of the Booth function.

![Model for "unknown" truth](/media/Booth_function.png?raw=True)
-->

You are given the task of finding the minimum of the function without knowing its analytical form. You can perform "experiments", where you provide `x0` and `x1` and obtain the noisy value of the function. You want to do as few experiments as possible.
<br/>
Working with the ProcessOptimizer package, you define the experimental `Space` and create an `Optimizer` object. In this specific case, we have two continous numerical dimensions both ranging from 0.0 to 5.0. <br/>

```python
import ProcessOptimizer as po

SPACE = po.Space([[0.0, 5.0], [0.0, 5.0]])   

```
The `Optimizer` defined below uses `"GP"` (Gaussian Process) for Bayesian optimization. Before the Bayesian part of the optimization begins, of number of initial "experiments" (`n_initial_points`) is run to obtain some initial data. After these initial "experiments" and every time new data is added afterwards, a Gaussian Process regression model is fitted to the data we have obtained so far. Based on this model (and an acquisition function that determines our search strategy), the optimizer suggests the next point to evaluate.

```python
opt = po.Optimizer(SPACE, base_estimator = "GP", n_initial_points = 2)
```
The optimizer can be used in steps by calling the `.ask()` function, evaluating the function at the given point and using `.tell()` to feed back the result to the `Optimizer`. In practise it would work like this. First ask the optimizer for the next point to perform an experiment:
```python
opt.ask()
>>> [3.75, 3.75]
```
Now go to the laboratory or wherever the experiment can be performed and use the values obtained when calling `ask()`. In this example the experiment can simply be performed by evaluating the Booth function using the values above:
```python
Booth(3.75, 3.75)
>>> 59.313996676981354
```
When a result has been obtained the user needs to tell the output to the `Optimizer`. This is done using the `.tell()` function:
```python
opt.tell([3.75, 3.75], 59.313996676981354)
result = opt.get_result()

po.plot_objective(result)
```
The `result` returned by `tell` contains a model of the Gaussian Process predicted mean. This model can be plotted using `plot_objective(result)`. Below is a gif of the search after 6 initial points and until 20 points have been sampled in total. The orange dots visualise each evaluation of the function and the red dot shows the position of the expected minimum. Besides the 2D color plot, there are also 1D plots for each input variable. These show how the function depend on each input variable with other input variables kept constant at the expected minimum.

![Progression of several ask and tells to processoptimizer](/media/BO_GIF.gif?raw=True?width="500" "Finding the minimum in the Booth function")

Notice that this is an optimization tool and not a modelling tool. This means that the optimizer finds an approximate solution for the global minimum quickly. It does, however, not guarantee that the obtained model is accurate on the entire domain.<br/>
<!--
The best observation against the number of observations can be plotted with `plot_convergence(result)`:

```python
po.plot_convergence(result)
```

![BayesianOptimization in action](/media/Convergence_plot.png?raw=True)
-->

A full minimal example of use can be found below:

```python
import numpy as np
import ProcessOptimizer as po

def Booth(x0, x1):
    booth = (x0 + 2 * x1 - 7)**2 + (2 * x0 + x1 - 5)**2
    noise = 1 + 0.05 * (2 * np.random.rand() - 1)
    return (booth * noise)

SPACE = po.Space([[0.0, 5.0], [0.0, 5.0]])
opt = po.Optimizer(SPACE,
                   base_estimator = "GP",
                   n_initial_points = 2)

for i in range(20):
    x = opt.ask()
    y = Booth(*x)
    opt.tell(x, y)

result = opt.get_result()
po.plot_objective(result)
```

## Examples
An introductory walkthough of the package can be found [here](https://github.com/novonordisk-research/ProcessOptimizer/blob/develop/examples/walkthrough/readme.md)<br>
Various examples on use and functionality can be found [here](https://github.com/novonordisk-research/ProcessOptimizer/tree/develop/examples).

## Contributions

Feel free to play around with algorithm. Should you encounter errors while using ProcessOptimizer, please report them
at https://github.com/novonordisk-research/ProcessOptimizer/issues. <br>
To help solve the issues, please: <br>

- Provide minimal amount of code to reproduce the error
- State versions of ProcesOptimizer, sklearn, numpy, ...
- Describe the expected behavior of the code <br>

If you would like to contribute by making anything from documentation to feature-additions, THANK YOU. Please open a pull request 
marked as *WIP* as early as possible and describe the issue you seek to solve and outline your planned solution. <br>
<!-- Pull requests to the develop branch will be automatically tested using pytest and flake8. We'll be happy to help solving potential
issues that could arise here.
-->

## Related work

ProcessOptimizer is a fork of [scikit-optimize](/https://scikit-optimize.github.io/stable/). ProcessOptimizer will fundamentally function like scikit-optimize, 
yet developments are focussed on bringing improvements to help optimizing real world processes, like chemistry or baking. 

[Brownie Bee](https://browniebee.io/) is a web-based platform for Bayesian process optimization intended for non-coders. It uses ProcessOptimizer as the underlying optimization engine.

## Citation

If you use the package in relation to published works, please cite: https://doi.org/10.5281/zenodo.5155295 and https://pubs.acs.org/doi/full/10.1021/acs.jcim.4c02240<br>
Please also cite the underlaying package (scikit-optimize).

<!--
## PyPi

If you have not packaged before check out https://packaging.python.org/tutorials/packaging-projects/
To upload a new version to PyPi do the following in the root folder of the project:

- In terminal run the command "pytest" and make sure there are no errors
- Change version number in setup.py
- Change version number in ProcessOptimizer/\_\_init\_\_.py
- Remember to `pip install twine` if running in a new virtual env. (You might also have to `pip install build`)
- Run `python -m build`
- Run `python -m twine upload dist/*` (make sure that /dist only contains relevant version)
- (Remember that pypi has changed the way it handles credentials, you might have to state username: [dunderscore]token[dunderscore] and then use your token value (incl pypi-prefix) as password. As stated here https://pypi.org/help/#apitoken
-->
