# Release history

## Version 0.8.1 [unpublished]

### Changes

- Added additonal model systems to the list of benchmarks and made their
  structure more consistent.
- Added seeding to the noise models used for benchmarking to ensure 
  reproducible results when benchmarking.

### Bugfixes

## Version 0.8.0

### Changes

- Default acquisition function changed to expected improvement (EI)
- Updated list of contributors
- Minor addition of guidance in plot_objectives()
- Implemented a major new constraint type called SumEquals. This constraint is
  designed to be used for mixture experiments where a (sub)set of factors must
  sum to a specific number.
- Add a module to add noise to model systems
- QoL opt.space.names added as property
- Changed default behavior of plot_objective to show uncertainty in 1D plots

### Bugfixes

-

## Version 0.7.8

### Changes

-

### Bugfixes

- ParetoFront did not show full recipe for model points
- Replaced friedman_mse with squared_error

## Version 0.7.7

### Changes

- Changed look of uncertainty-plots in plot_objective
- Added plot to only show 1d plots
- Align code in GPR module to reflect sklearn. While still supporting SKlearn
  0.24.2, we have some parallel code between our local GPR and the original from
  sklearn.

### Bugfixes

- Model systems should now be imported as intended.

## Version 0.7.6

### Changes

- Add Bokeh version of Pareto plot
- Make a bleeding edge installable and a stable
- Add Bokeh to list of requiered packages
- Change the call of gaussian filter for a helper illustration function
- Model systems added to help benchmark performance or teach

### Bugfixes

- Remove call of plot_width and plot_height in bokeh

## Version 0.7.5

### Changes

- Allow user defined bounds on noise level of WhiteKernel

### Bugfixes

## Version 0.7.4

### Changes

- Added the option to display uncertainty in 2d plots in plot_objective
- Initial efforts to streamline the input-structure of plot options
- Additional options for plots

### Bugfixes

## Version 0.7.3

### Changes

- Dependence modul now consistenty returns arrays instead of lists
- House-keeping on Github (contribution guidelines etc)

### Bugfixes

- Bokeh_plot is repaired after we started returning the std to plots
- LHS is rewritten to ensure consistent returns in between real and integer
  dimensions (integer types are ensure to return values "close" to those of a
  corresponding real dimension)

## Version 0.7.2

### Changes

- New plot-type to envision model coverage
- Kriging Believer now supports multiobjective opt
- Examples pruned to better reflect the purpose of ProcessOptimizer as a tool
  for optimizing real world physical/chemical processes
- Expected_minimum can now return both maximum and minimum and can return the
  expected std in the points. Works for both numerical and categorical dimensions
- QoL improvements with easy impoart of most used features through \_\_init\_\_
  .py
- Add possibility to show ~95% credibility_intervals in plot_objective

### Bugfixes

- More linting
- Supports Scikit-Learn 1.0.0
- plot_pareto works with partially categorical spaces
- Consolidate tests

## Version 0.7.1

### Changes

- Kriging Believer added for batch-mode optimization
- Added Interactive Pareto plotting

### Bugfixes

- LHS fixed to ensure randomization of dimensions
- More linting

## Version 0.7.0

### Changes

- Added plot_expected_minimum_convergence
- Numerous format changes to satisfy Flake8
- Fixed deprecation warning from Numpy
- Merge keywords for batch-optimization
- Set LHS=True as default
- Extensive changes in test-suite

### Bugfixes

- expected_minimum_sampling refactored
- Fix URL to present images on pypi
- Fixed Bokeh-plot to optional dependency
- Dependencies set (Bokeh optional, pyYAML added)

## Version 0.6.4

### Changes

- Add plot_expected_minimum_convergence
- Recode expected_minimum_random_sampling and move to utils.py
- Update Readme to add illustrations on pypi

## Version 0.6.3

### Changes

- Improve documentation in README.md
- Reset normalize_y to True and update requirements.txt

## Version 0.6.2

### Changes

- Automatic testing when commiting to develop is implemented
- Slight adjustment to documentation
- Unneccesary files pruned
- Visual change to changelog

### Bugfixes

- test_deadline_stopper fixed as it was giving unreproducible results
- Fixed Steinerberger error

## Version 0.6.1

### Changes

- Two additional examples added
- Tests are primarily from numpy instead of sklearn
- Changed plot_objective to use same color scale for each individual dependence plot and added colorbar
- Added plot_objectives, which plots all individual objective functions for multiobjective optimization
- Added a title parameter to plot_objective
- More informative ReadMe.md started

### Bugfixes

- Example on visualize_results.ipynb corrected to avoid warnings
- DEAP added as dependency for Pareto optimization
- normalize_y temporarely set to False until real fix from sklearn
- Handled a case in which feeding multible datapoints to the model would fail.
- Added functionality to create_result to create a list of results in case of multiobjective optimization
- Added a ValueError to warn users against using GP in entirely categorical spaces

## Version 0.6.0

### Changes

- README changed to reflect move of repo to NN-research
- Steinerberger sampling added for improves spacefilling and explorative mode
- Multiobjective optimization added by NSGA and Pareto front
- Unused folders trimmed
- Added example notebooks on new functionality

## Version 0.5.1

### Bugfixes

- Removed check for numpy version in constraints.py
- Updated example/constraints.ipynb

## Version 0.5.0

### Changes

- Remove dependency on scipy>=0.14.0 \*
- Remove dependency on scikit-learn==0.21.0 \*^
- Remove dependency on bokeh==1.4.0 \*
- Remove dependency on tornado==5.1.1 \*
- \*from setup.py
- ^from requirements.txt
- Change gpr (as in skopt #943) to reflect changes in sklearn gpr-module
  (relates to normalilzation)
- Change searchCV to reflect skopt #939 and #904 (relates to np.mask and imports)
- Changes in tests (skopt#939 and #808). Extensive changes in tests!
- Change in Bokeh_plot.py to fix bug when Bokeh>=2.2.0

- TODO: look more into implemented normalizations in skopt.

## Version 0.4.9

### Bugfixes

- Version number increased due to reupload to pypi.

## Version 0.4.8

### Bugfixes

- Locked SKlearn to version 0.21.0 to avoid install errors.

## Version 0.4.7

### Bugfixes

- Changed bokeh version to 1.4.0

## Version 0.4.6

### Bugfixes

- ProcessOptimizer.\_\_version\_\_ shows correct version.
- Removed \_version.py as we dont use versioneer anymore.
- Version needs to be changed manually in \_\_init\_\_ .py from now on.

## Version 0.4.5

- Wrong upload. Please don't use this version

## Version 0.4.4

### New features

- Latin hypercube sampling

### Bugfixes

- Progress is now correctly showed in bokeh.

## Version 0.4.3

### Bugfixes

- Lenght scale bounds and length scales were not transformed properly.

## Version 0.4.2

### New features

- optimizer.update_next() added
- Added option to change length scale bounds
- Added optimizer.get_result()
- Added exploration example notebook
- Added length scale bounds example notebook

## Version 0.4.1

### New features

- Draw upper confidence limit in bokeh.
- Colorbar in bokeh
- Same color mapping button in bokeh

## Version 0.4.0

Merged darnr's scikit-optimize fork into ProcessOptimizer. Here is their changelog:

### New features

- `plot_regret` function for plotting the cumulative regret;
  The purpose of such plot is to access how much an optimizer
  is effective at picking good points.
- `CheckpointSaver` that can be used to save a
  checkpoint after each iteration with skopt.dump
- `Space.from_yaml()`
  to allow for external file to define Space parameters

### Bug fixes

- Fixed numpy broadcasting issues in gaussian_ei, gaussian_pi
- Fixed build with newest scikit-learn
- Use native python types inside BayesSearchCV
- Include fit_params in BayesSearchCV refit

### Maintenance

- Added `versioneer` support, to reduce changes with new version of the `skopt`

### Bug fixes

- Separated `n_points` from `n_jobs` in `BayesSearchCV`.
- Dimensions now support boolean np.arrays.

### Maintenance

- `matplotlib` is now an optional requirement (install with `pip install 'scikit-optimize[plots]'`)

High five!

### New features

- Single element dimension definition, which can be used to
  fix the value of a dimension during optimization.
- `total_iterations` property of `BayesSearchCV` that
  counts total iterations needed to explore all subspaces.
- Add iteration event handler for `BayesSearchCV`, useful
  for early stopping inside `BayesSearchCV` search loop.
- added `utils.use_named_args` decorator to help with unpacking named dimensions
  when calling an objective function.

### Bug fixes

- Removed redundant estimator fitting inside `BayesSearchCV`.
- Fixed the log10 transform for Real dimensions that would lead to values being
  out of bounds.

## Version 0.3.3

### New features

- Added text describing progress in bokeh

### Changes

- Changed plot size in bokeh
- ProcessOptimizer now requires tornado 5.1.1

## Version 0.3.0

### New features

- Added constrained parameters

## Version 0.2.0

### New features

- Interactive bokeh GUI for plotting the objective function

## Version 0.0.2

### New features

- Support for using categorical values when plotting objective.

## Version 0.0.1

### New features

- Support for not using partial dependence when plotting objective.
- Support for choosing the values of other parameters when calculating dependence plots
- Support for choosing other minimum search algorithms for the red lines and dots in objective plots
