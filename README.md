# ProcessOptimizer

ProcessOptimizer is a fork of scikit-optimize, that focuses on optimizing real world processes, like chemistry or baking.
For examples on use checkout https://scikit-optimize.github.io/.

## Installation

ProcessOptimizer can be installed using "pip install ProcessOptimizer"
The repository and examples can be found at https://github.com/novonordisk-research/ProcessOptimizer
ProcessOptimizer can also be installed by running `pip install -e .` in top directory of the downloaded repository.

## PyPi

If you have not packaged before check out https://packaging.python.org/tutorials/packaging-projects/
To upload a new version to PyPi do the following in the root folder:

- In terminal run the command "pytest" and make sure there are no errors
- Change version number in setup.py
- Change version number in ProcessOptimizer/\_\_init\_\_.py
- Remember to `pip install twine` if running in a new virtual env
- run `python setup.py sdist bdist_wheel`
- run `python -m twine upload dist/*` (make sure that /dist only contains relevant version)
