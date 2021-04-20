try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='ProcessOptimizer',
      version='0.6.2',
      description='Sequential model-based optimization toolbox (forked from scikit-optimize)',
      url='https://github.com/novonordisk-research/ProcessOptimizer',
      license='BSD',
      author='Novo Nordisk, Research & Early Development',
      packages=['ProcessOptimizer', 'ProcessOptimizer.learning', 'ProcessOptimizer.optimizer', 'ProcessOptimizer.space',
                'ProcessOptimizer.learning.gaussian_process'],
      install_requires=['numpy', 'matplotlib', 'scipy',
                        'scikit-learn', 'bokeh', 'tornado', 
						'six', 'deap'],
      long_description=long_description,
      long_description_content_type='text/markdown'
      )
