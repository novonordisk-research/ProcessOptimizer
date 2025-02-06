# Release history

## Version 1.0.3 [unpublished]

### Changes

- 

### Bugfixes

-  

## Version 1.0.2

### Changes

- 

### Bugfixes

- Fixed a bug where sum_equals constraints would break if used with dimensions other than 
  an increasing list from 0. E.g. constraining dimensions [0, 1, 2] would work, but 
  constraining [1, 2, 3] would not. 
  

## Version 1.0.1 (October 2024)

### Changes

- Added a plotting function that supports the Brownie Bee user interface (see browniebee.io).
- Made small improvements to the plot_objective_1d function, and added it to __init.py__ for ProcessOptimizer.
- Added documentation for multiobjective optimization

### Bugfixes

-

## Version 1.0.0 (September 2024)

### Changes

- Documentation about the features in control parameters and sampling control parameters.
- ModelSystems moved to creator systems, so they are only created when you ask for them. You now need to use `ProcessOptimizer.model_systems.get_model_system(model_system_name)` to create them. This has two advantages: If you change a model system, it doesn't affect a new instance of it. And ProcessOptimizer should import faster, since fewer objects are created i memory.
- Radius changed in certain Bokeh plots
- Default Pareto plot has more points on Pareto-front (40 -> 100)

### Bugfixes

- Setting noise-size in zero-noise models will now raise an error
- Multible imports of a model system will now provide separate instances

## Version 0.9.5 (May 2024)

### Changes

- Updated package requirements for Brownie Bee user interface.
- Examples reworked.
- opt.estimate() implemented - Works in non-transformed space and on all objectives.

### Bugfixes

- Fix that categorical dimensions with more than two levels induces error when used 
  together with SumEqual constraint.
- Fix that Bokeh has changed naming convention related to sizes of circles in their
  plots from "size" to "radius".

## Version 0.9.4

### Changes

-

### Bugfixes

- Fix dependency on deprecated Matrix from scipy in favour of a numpy solution
- Ensure prober warning/Errors when users try to combine constraints with operations that
  doesn't support constraints.

## Version 0.9.3

### Changes

-

### Bugfixes

- Fix install bug that precluded installation of 0.9.2

## Version 0.9.2

### Changes

- Update ipynb file showing example of color_pH modelsystem. (Very minor)

### Bugfixes

- Fixed a bug in `expected_minimum` where SumEquals constraint values were not rescaled
  correctly during normalization.

## Version 0.9.1

### Changes

- Added colorpH as ModelSystem
- Exposed Integer, Real, Categorical and ModelSystem as direct imports of ProcessOptimizer.

### Bugfixes

- Changed the result object to store information about constraints (when present).
- Updated `expected_minimum` to make sure that the returned result location respects
  SumEquals constraints if these were used during the optimization.

## Version 0.9.0

### Changes

-

### Bugfixes

- Fix install issue in 0.8.2. Th installed package could not be imported.

## Version 0.8.2

### Changes

- **BREAKING**: `cook_estimator`, `has_gradients`, and `use_named_args` are moved from
  `utils` to `learning`.
- **BREAKING**: `normalize_dimensions` is moved from `utils` to `space`.
- **BREAKING**: `branin`, `hart3`, `hart6`, `poly2`, and `peaks` has been changed to noisy and noiseless `ModelSystem`s.
- Implemented tests for ModelSystems.
- `gold_map` exist as `ModelSystem`.
- Sampling consolidated. There is now only one sampling function per `Dimension`.
  Different sampling types (at the moment, random value sampling and Latin Hypercube
  Sampling (LHS)) are handled through `Space`, referencing the sampling functions of the
  `Dimesion`s
- LHS now allows for arbitrary seeding, or for random seeding, better supporting
  benchmarking. The algorithm still uses a fixed seed by default.
- The module `space` uses `np.random.default_rng` as a random number generator, instead
  of the deprecated `np.random.RandomState`. A bridging strategy allows it to still
  accept `RandomState`s, but it will tranform them to `default_rng` for internal use.
  The rest of the codebase still uses `RandomState`.
- **BREAKING**: LHS now respects priors. This means that performing LHS on a space with
  a log-normal `Real` `Dimension`, or a `Categorical` `Dimension` with informative
  priors will give different results in this release than in previous releases.
- **BREAKING**: The mechanism for seeding pseudorandom generators in the `space` module
  have changed, meaning that, while the results are reproducible within a release, they
  will not be the same as in old releases.
- Bokeh is now (again) a required installation.

### Bugfixes

- Fixes to `DataDependentNoise` and `SumNoise` to avoid highly correlated
  noise of the underlying noise models.
- Switched to local imports internally to avoid circular import errors.
- `NoiseModel._noise_distribution` is now a method, to allow changes of
  `self._rng` to affect `self._noise_distribution` automatically.
- Removed max_features='auto' to avoid using hardcoded variables to external functions

## Version 0.8.1

### Changes

- Added additonal model systems to the list of benchmarks and made their
  structure more consistent.
- Added seeding to the noise models used for benchmarking to ensure
  reproducible results when benchmarking.
- Allow addition or removal of modelled noise to the optimizer object. This
  is to allow user to predict the full outcome space of a given new exp.

### Bugfixes

- Fix a small number of deprecationwarnings.

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
