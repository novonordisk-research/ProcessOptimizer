try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="ProcessOptimizer",
    version="1.0.1",
    description="Sequential model-based optimization toolbox \
    (forked from scikit-optimize)",
    url="https://github.com/novonordisk-research/ProcessOptimizer",
    license="BSD",
    author="Novo Nordisk, Research & Early Development",
    packages=[
        "ProcessOptimizer",
        "ProcessOptimizer.learning",
        "ProcessOptimizer.model_systems",
        "ProcessOptimizer.model_systems.data",
        "ProcessOptimizer.optimizer",
        "ProcessOptimizer.space",
        "ProcessOptimizer.utils",
        "ProcessOptimizer.learning.gaussian_process",
    ],
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "bokeh",
        "scikit-learn>=0.24.2",
        "six",
        "deap",
        "pyYAML",
    ],
    extras_require={
        "browniebee": [
            "bokeh==3.4.1",
            "deap==1.4.1",
            "matplotlib==3.8.4",
            "numpy==1.26.4",
            "pyYAML==6.0.1",
            "scikit-learn==1.4.2",
            "scipy==1.13.0",
            "six==1.16.0",
        ]
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
)
