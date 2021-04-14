try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='ProcessOptimizer',
      version='0.6.1',
      description='Sequential model-based optimization toolbox (forked from scikit-optimize)',
      url='https://github.com/novonordisk-research/ProcessOptimizer',
      license='BSD',
      author='Novo Nordisk, Research & Early Development',
      packages=['ProcessOptimizer', 'ProcessOptimizer.learning', 'ProcessOptimizer.optimizer', 'ProcessOptimizer.space',
                'ProcessOptimizer.learning.gaussian_process'],
      install_requires=['numpy', 'matplotlib', 'scipy',
                        'scikit-learn', 'bokeh', 'tornado', 
						'six', 'deap']
      )
