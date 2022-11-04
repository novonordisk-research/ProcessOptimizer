try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='ProcessOptimizer',
      version='0.7.6',
      description='Sequential model-based optimization toolbox \
    (forked from scikit-optimize)',
      url='https://github.com/novonordisk-research/ProcessOptimizer',
      license='BSD',
      author='Novo Nordisk, Research & Early Development',
      packages=[
          'ProcessOptimizer',
          'ProcessOptimizer.learning',
          'ProcessOptimizer.model_system',
          'ProcessOptimizer.optimizer',
          'ProcessOptimizer.space',
          'ProcessOptimizer.learning.gaussian_process'
          ],
      install_requires=['numpy', 'matplotlib', 'scipy', 'bokeh',
                        'scikit-learn>=0.24.2', 'six', 'deap', 'pyYAML'],
      extras_require={
          "browniebee": ['numpy==1.23.3',
                         'matplotlib==3.5.3',
                         'scipy==1.9.1',
                         'scikit-learn==1.1.2',
                         'six==1.16.0',
                         'deap==1.3.3',
                         'pyYAML==6.0',
                         'bokeh==2.4.3',
                         'tornado==6.2',]
          },
      long_description=long_description,
      long_description_content_type='text/markdown'
      )
