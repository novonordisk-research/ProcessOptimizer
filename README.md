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
<a href="https://badge.fury.io/py/ProcessOptimizer"><img src="https://badge.fury.io/py/ProcessOptimizer.svg" alt="PyPI version" height="18"></a>
<a href="https://github.com/novonordisk-research/ProcessOptimizer/actions"><img src="https://github.com/novonordisk-research/ProcessOptimizer/workflows/Python%20package/badge.svg" alt="Tests" height="18"></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/built%20with-Python3-green.svg" alt="built with Python3" /></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue" alt="Runs on" /></a>
<a href="https://github.com/novonordisk-research/ProcessOptimizer/blob/develop/LICENSE.md"><img src="https://img.shields.io/pypi/l/ProcessOptimizer" alt="PyPI - License" /></a>
<a href="https://scikit-optimize.github.io/stable/"><img src="https://img.shields.io/badge/BuildOn-Scikit--Optimize-brightgreen" alt="Scikit-Optimize" /></a>

[![Downloads](https://static.pepy.tech/personalized-badge/processoptimizer?period=total&units=international_system&left_color=brightgreen&right_color=orange&left_text=Downloads)](https://pepy.tech/project/processoptimizer)
</div>

----------
This readme.md is work in progress

## Table of Contents
 * [ProcessOptimizer](https://github.com/novonordisk-research/ProcessOptimizer/blob/develop/README.md#processoptimizer)
 * [Installation](https://github.com/novonordisk-research/ProcessOptimizer/blob/develop/README.md#installation)
 * [Contributions](https://github.com/novonordisk-research/ProcessOptimizer/blob/develop/README.md#contributions)
 * [Related work](https://github.com/novonordisk-research/ProcessOptimizer/blob/develop/README.md#related-work)
 * [PyPi](https://github.com/novonordisk-research/ProcessOptimizer/blob/develop/README.md#pypi)
 

## ProcessOptimizer

ProcessOptimizer is a fork of scikit-optimize. ProcessOptimizer will fundamentally function like scikit-optimize, 
yet developments are focussed on bringing improvements to help optimizing real world processes, like chemistry or baking.
For examples on use, checkout https://github.com/novonordisk-research/ProcessOptimizer/tree/develop/examples.

## Installation

ProcessOptimizer can be installed using `pip install ProcessOptimizer`
The repository and examples can be found at https://github.com/novonordisk-research/ProcessOptimizer
ProcessOptimizer can also be installed by running `pip install -e .` in top directory of the cloned repository.

## Contributions

Feel free to play around with algorithm. Should you encounter errors while using ProcessOptimizer, please report them
at https://github.com/novonordisk-research/ProcessOptimizer/issues. <br>
To help solve the issues, please: <br>

- Provide minimal amount of code to reproduce the error
- State versions of ProcesOptimizer, sklearn, numpy, ...
- Describe the expected nehavior of the code <br>

If you would like to contribute by making anything from documentation to feature-additions, THANK YOU. Please open a pull request 
marked as *WIP* as early as possible and describe the issue you seek to solve and outline your planned solution.

## Related work

We are currently building a GUI to offer the power of Bayesian Process Optimization to non-coders. Stay tuned.

## PyPi

If you have not packaged before check out https://packaging.python.org/tutorials/packaging-projects/
To upload a new version to PyPi do the following in the root folder of the project:

- In terminal run the command "pytest" and make sure there are no errors
- Change version number in setup.py
- Change version number in ProcessOptimizer/\_\_init\_\_.py
- Remember to `pip install twine` if running in a new virtual env
- Run `python setup.py sdist bdist_wheel`
- Run `python -m twine upload dist/*` (make sure that /dist only contains relevant version)
