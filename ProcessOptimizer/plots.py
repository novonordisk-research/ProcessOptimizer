"""Plotting functions."""
from itertools import count
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy.optimize import OptimizeResult
from scipy.stats.mstats import mquantiles
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from ProcessOptimizer import expected_minimum, expected_minimum_random_sampling
from .space import Categorical, Integer
from .optimizer import Optimizer

import json
import bokeh.models as bh_models
import bokeh.plotting as bh_plotting
import bokeh.io as bh_io
import bokeh.embed as bh_embed
import bokeh.resources as bh_resources

def plot_convergence(*args, **kwargs):
    """Plot one or several convergence traces.

    Parameters
    ----------
    * `args[i]` [`OptimizeResult`, list of `OptimizeResult`, or tuple]:
        The result(s) for which to plot the convergence trace.

        - if `OptimizeResult`, then draw the corresponding single trace;
        - if list of `OptimizeResult`, then draw the corresponding convergence
          traces in transparency, along with the average convergence trace;
        - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
          an `OptimizeResult` or a list of `OptimizeResult`.

    * `ax` [`Axes`, optional]:
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    * `true_minimum` [float, optional]:
        The true minimum value of the function, if known.

    * `yscale` [None or string, optional]:
        The scale for the y-axis.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    # <3 legacy python
    ax = kwargs.get("ax", None)
    true_minimum = kwargs.get("true_minimum", None)
    yscale = kwargs.get("yscale", None)

    if ax is None:
        ax = plt.gca()

    ax.set_title("Convergence plot")
    ax.set_xlabel("Number of calls $n$")
    ax.set_ylabel(r"$\min f(x)$ after $n$ calls")
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    colors = cm.viridis(np.linspace(0.25, 1.0, len(args)))

    for results, color in zip(args, colors):
        if isinstance(results, tuple):
            name, results = results
        else:
            name = None

        if isinstance(results, OptimizeResult):
            n_calls = len(results.x_iters)
            mins = [
                np.min(results.func_vals[:i]) for i in range(1, n_calls + 1)
            ]
            ax.plot(
                range(1, n_calls + 1),
                mins,
                c=color,
                marker=".",
                markersize=12,
                lw=2,
                label=name,
            )

        elif isinstance(results, list):
            n_calls = len(results[0].x_iters)
            iterations = range(1, n_calls + 1)
            mins = [
                [np.min(r.func_vals[:i]) for i in iterations] for r in results
            ]

            for m in mins:
                ax.plot(iterations, m, c=color, alpha=0.2)

            ax.plot(
                iterations,
                np.mean(mins, axis=0),
                c=color,
                marker=".",
                markersize=12,
                lw=2,
                label=name,
            )

    if true_minimum:
        ax.axhline(
            true_minimum, linestyle="--", color="r", lw=1, label="True minimum"
        )

    if true_minimum or name:
        ax.legend(loc="best")

    return ax


def plot_regret(*args, **kwargs):
    """Plot one or several cumulative regret traces.

    Parameters
    ----------
    * `args[i]` [`OptimizeResult`, list of `OptimizeResult`, or tuple]:
        The result(s) for which to plot the cumulative regret trace.

        - if `OptimizeResult`, then draw the corresponding single trace;
        - if list of `OptimizeResult`, then draw the corresponding cumulative
        regret traces in transparency, along with the average cumulative regret
        trace;
        - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
          an `OptimizeResult` or a list of `OptimizeResult`.

    * `ax` [`Axes`, optional]:
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    * `true_minimum` [float, optional]:
        The true minimum value of the function, if known.

    * `yscale` [None or string, optional]:
        The scale for the y-axis.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    # <3 legacy python
    ax = kwargs.get("ax", None)
    true_minimum = kwargs.get("true_minimum", None)
    yscale = kwargs.get("yscale", None)

    if ax is None:
        ax = plt.gca()

    ax.set_title("Cumulative regret plot")
    ax.set_xlabel("Number of calls $n$")
    ax.set_ylabel(r"$\sum_{i=0}^n(f(x_i) - optimum)$ after $n$ calls")
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    colors = cm.viridis(np.linspace(0.25, 1.0, len(args)))

    if true_minimum is None:
        results = []
        for res in args:
            if isinstance(res, tuple):
                res = res[1]

            if isinstance(res, OptimizeResult):
                results.append(res)
            elif isinstance(res, list):
                results.extend(res)
        true_minimum = np.min([np.min(r.func_vals) for r in results])

    for results, color in zip(args, colors):
        if isinstance(results, tuple):
            name, results = results
        else:
            name = None

        if isinstance(results, OptimizeResult):
            n_calls = len(results.x_iters)
            regrets = [
                np.sum(results.func_vals[:i] - true_minimum)
                for i in range(1, n_calls + 1)
            ]
            ax.plot(
                range(1, n_calls + 1),
                regrets,
                c=color,
                marker=".",
                markersize=12,
                lw=2,
                label=name,
            )

        elif isinstance(results, list):
            n_calls = len(results[0].x_iters)
            iterations = range(1, n_calls + 1)
            regrets = [
                [np.sum(r.func_vals[:i] - true_minimum) for i in iterations]
                for r in results
            ]

            for cr in regrets:
                ax.plot(iterations, cr, c=color, alpha=0.2)

            ax.plot(
                iterations,
                np.mean(regrets, axis=0),
                c=color,
                marker=".",
                markersize=12,
                lw=2,
                label=name,
            )

    if name:
        ax.legend(loc="best")

    return ax


def _format_scatter_plot_axes(ax, space, ylabel, dim_labels=None):
    # Work out min, max of y axis for the diagonal so we can adjust
    # them all to the same value
    diagonal_ylim = (
        np.min([ax[i, i].get_ylim()[0] for i in range(space.n_dims)]),
        np.max([ax[i, i].get_ylim()[1] for i in range(space.n_dims)]),
    )

    if dim_labels is None:
        dim_labels = space.names
    # Axes for categorical dimensions are really integers; we have to
    # label them with the category names
    iscat = [isinstance(dim, Categorical) for dim in space.dimensions]

    # Deal with formatting of the axes
    for i in range(space.n_dims):  # rows
        for j in range(space.n_dims):  # columns
            ax_ = ax[i, j]

            if j > i:
                ax_.axis("off")
            elif i > j:  # off-diagonal plots
                # plots on the diagonal are special, like Texas. They have
                # their own range so do not mess with them.
                if not iscat[i]:  # bounds not meaningful for categoricals
                    ax_.set_ylim(*space.dimensions[i].bounds)
                if iscat[j]:
                    # partial() avoids creating closures in a loop
                    ax_.xaxis.set_major_formatter(
                        FuncFormatter(
                            partial(_cat_format, space.dimensions[j])
                        )
                    )
                else:
                    ax_.set_xlim(*space.dimensions[j].bounds)
                if j == 0:  # only leftmost column (0) gets y labels
                    ax_.set_ylabel(dim_labels[i])
                    if iscat[i]:  # Set category labels for left column
                        ax_.yaxis.set_major_formatter(
                            FuncFormatter(
                                partial(_cat_format, space.dimensions[i])
                            )
                        )
                else:
                    ax_.set_yticklabels([])

                # for all rows except ...
                if i < space.n_dims - 1:
                    ax_.set_xticklabels([])
                # ... the bottom row
                else:
                    [labl.set_rotation(45) for labl in ax_.get_xticklabels()]
                    ax_.set_xlabel(dim_labels[j])

                # configure plot for linear vs log-scale
                if space.dimensions[j].prior == "log-uniform":
                    ax_.set_xscale("log")
                else:
                    ax_.xaxis.set_major_locator(
                        MaxNLocator(6, prune="both", integer=iscat[j])
                    )

                if space.dimensions[i].prior == "log-uniform":
                    ax_.set_yscale("log")
                else:
                    if iscat[i]:
                        # Do not remove first/last tick for categoric factors
                        ax_.yaxis.set_major_locator(
                            MaxNLocator(6, prune=None, integer=iscat[i])
                        )
                    else:
                        ax_.yaxis.set_major_locator(
                            MaxNLocator(6, prune="both", integer=iscat[i])
                        )

            else:  # diagonal plots
                ax_.set_ylim(*diagonal_ylim)
                ax_.yaxis.tick_right()
                ax_.yaxis.set_label_position("right")
                ax_.yaxis.set_ticks_position("both")
                ax_.set_ylabel(ylabel)

                ax_.xaxis.tick_top()
                ax_.xaxis.set_label_position("top")
                [labl.set_rotation(45) for labl in ax_.get_xticklabels()]
                ax_.set_xlabel(dim_labels[j])

                if space.dimensions[i].prior == "log-uniform":
                    ax_.set_xscale("log")
                else:
                    ax_.xaxis.set_major_locator(
                        MaxNLocator(6, prune="both", integer=iscat[i])
                    )
                    if iscat[i]:
                        ax_.xaxis.set_major_formatter(
                            FuncFormatter(
                                partial(_cat_format, space.dimensions[i])
                            )
                        )

    return ax

def _format_1d_dependency_axes(ax, space, ylabel, dim_labels=None):

    if dim_labels is None:
        dim_labels = space.names
    # Figure out where we have categorical factors
    iscat = [isinstance(dim, Categorical) for dim in space.dimensions]

    if space.n_dims < 3:
        nrows = 1
        ncols = space.n_dims
    else:
        nrows, ncols = ax.shape
    
    for i in range(nrows):
        for j in range(ncols):
            # Build a handle to the present subplot
            if space.n_dims == 1:
                ax_ = ax
            elif space.n_dims == 2:
                ax_ = ax[j]
            else:
                ax_ = ax[i, j]
            
            # Figure out what dimension number we are plotting from the indices
            if space.n_dims < 3:
                n = j
            else:
                n = np.ravel_multi_index(
                    np.array([[i],[j]]), 
                    ax.shape
                )
                n = n[0]
            
            # Turn off axes that do not contain a plot
            if n >= space.n_dims:
                ax_.axis("off")
            else:
                # Fix formatting of the y-axis
                ax_.yaxis.set_major_locator(
                    MaxNLocator(6, prune="both")
                )
                ax_.tick_params(axis="y", direction="inout")
                # Leftmost plot in each row:
                if j == 0:
                    ax_.set_ylabel(ylabel)
                # Rightmost plot in each row:                                      
                elif j == ncols-1:
                    ax_.set_ylabel(ylabel)
                    ax_.yaxis.set_label_position("right")
                    ax_.yaxis.tick_right()
                    ax2_ = ax_.secondary_yaxis("left")
                    ax2_.tick_params(axis="y", direction="inout")
                    ax2_.set_yticklabels([])
                    ax2_.yaxis.set_major_locator(ax_.yaxis.get_major_locator())
                else:
                    ax_.set_yticklabels([])
                    ax_.tick_params(axis="y", direction="inout")
                    ax2_ = ax_.secondary_yaxis("right")
                    ax2_.tick_params(axis="y", direction="inout")
                    ax2_.yaxis.set_major_locator(ax_.yaxis.get_major_locator())
                    if n < space.n_dims-1:
                        ax2_.set_yticklabels([])
                    else:
                        ax2_.set_ylabel(ylabel)
                
                # Fix formatting of the x-axis
                [labl.set_rotation(20) for labl in ax_.get_xticklabels()]
                ax_.set_xlabel(dim_labels[n])
                if space.dimensions[n].prior == "log-uniform":
                    ax_.set_xscale("log")
                else:
                    ax_.xaxis.set_major_locator(
                        MaxNLocator(5, prune="both", integer=iscat[n])
                    )
                    if iscat[n]:
                        # Axes for categorical dimensions are really integers; 
                        # we have to label them with the category names
                        ax_.xaxis.set_major_formatter(
                            FuncFormatter(
                                partial(_cat_format, space.dimensions[n])
                            )
                        )
    
    return ax

def dependence(
    space,
    model,
    i,
    j=None,
    sample_points=None,
    n_samples=250,
    n_points=40,
    x_eval=None,
    return_std=False,
):
    """
    Calculate the dependence for dimensions `i` and `j` with
    respect to the objective value, as approximated by `model`.
    If x_eval is set to "None" partial dependence will be calculated.

    The dependence plot shows how the value of the dimensions
    `i` and `j` influence the `model`.

    Parameters
    ----------
    * `space` [`Space`]
        The parameter space over which the minimization was performed.

    * `model`
        Surrogate model for the objective function.

    * `i` [int]
        The first dimension for which to calculate the partial dependence.

    * `j` [int, default=None]
        The second dimension for which to calculate the partial dependence.
        To calculate the 1D partial dependence on `i` alone set `j=None`.

    * `sample_points` [np.array, shape=(n_points, n_dims), default=None]
        Only used when `x_eval=None`, i.e in case partial dependence should
        be calculated.
        Randomly sampled and transformed points to use when averaging
        the model function at each of the `n_points` when using partial
        dependence.

    * `n_samples` [int, default=100]
        Number of random samples to use for averaging the model function
        at each of the `n_points` when using partial dependence. Only used
        when `sample_points=None` and `x_eval=None`.

    * `n_points` [int, default=40]
        Number of points at which to evaluate the partial dependence
        along each dimension `i` and `j`.

    * `x_eval` [list, default=None]
        x_eval is a list of parameter values. In case this list is
        parsed dependence will be calculated using these values
        instead of using partial dependence.

    * `return_std` [Boolian, default=False]
        Whether to return the standard deviation matrix for 2D dependence. Note that it
        is always returned for 1D dependence.

    Returns
    -------
    For 1D partial dependence:

    * `xi`: [np.array]:
        The points at which the partial dependence was evaluated.
    * `yi`: [np.array]:
        The value of the model at each point `xi`.
    * `stddevs`: [np.array]:
        The standard deviation of the model at each point `xi`.


    For 2D partial dependence:

    * `xi`: [np.array, shape=n_points]:
        The points at which the partial dependence was evaluated.
    * `yi`: [np.array, shape=n_points]:
        The points at which the partial dependence was evaluated.
    * `zi`: [np.array, shape=(n_points, n_points)]:
        The value of the model at each point `(xi, yi)`.
    * `std`: [np.array, shape=(n_points, n_points)]:
        The standard deviation of `zi` at each point `(xi, yi)`.

    For Categorical variables, the `xi` (and `yi` for 2D) returned are
    the indices of the variable in `Dimension.categories`.
    """
    # The idea is to step through one dimension and evaluating the model with
    # that dimension fixed.
    # (Or step through 2 dimensions when i and j are given.)
    # Categorical dimensions make this interesting, because they are one-
    # hot-encoded, so there is a one-to-many mapping of input dimensions
    # to transformed (model) dimensions.

    # If we havent parsed an x_eval list we use random sampled values instead
    if x_eval is None:
        sample_points = space.transform(space.rvs(n_samples=n_samples))
    else:
        sample_points = space.transform([x_eval])

    # dim_locs[i] is the (column index of the) start of dim i in sample_points.
    # This is usefull when we are using one hot encoding,
    # i.e using categorical values.
    dim_locs = np.cumsum([0] + [d.transformed_size for d in space.dimensions])

    if j is None:
        # We sample evenly instead of randomly. This is necessary when using
        # categorical values
        xi, xi_transformed = _evenly_sample(space.dimensions[i], n_points)
        yi = []
        stddevs = []
        for x_ in xi_transformed:
            rvs_ = np.array(sample_points)  # copy
            # We replace the values in the dimension that we want to keep fixed
            rvs_[:, dim_locs[i] : dim_locs[i + 1]] = x_
            # In case of `x_eval=None` rvs conists of random samples.
            # Calculating the mean of these samples is how partial dependence
            # is implemented.
            funcvalue, stddev = model.predict(rvs_, return_std = True)
            yi.append(np.mean(funcvalue))
            stddevs.append(np.mean(stddev))
        # Convert yi and stddevs from lists to numpy arrays
        yi = np.array(yi)
        stddevs = np.array(stddevs)
        
        return xi, yi, stddevs

    else:
        xi, xi_transformed = _evenly_sample(space.dimensions[j], n_points)
        yi, yi_transformed = _evenly_sample(space.dimensions[i], n_points)

        # stddev structure is made regardless of whether it is returned,
        # since this is cheap and makes the code simpler.
        zi = []
        stddev_matrix = []
        for x_ in xi_transformed:
            value_row = []
            stddev_row = []
            for y_ in yi_transformed:
                rvs_ = np.array(sample_points)  # copy
                rvs_[:, dim_locs[j] : dim_locs[j + 1]] = x_
                rvs_[:, dim_locs[i] : dim_locs[i + 1]] = y_
                funcvalue, stddev = model.predict(rvs_, return_std = True)
                value_row.append(np.mean(funcvalue))
                stddev_row.append(np.mean(stddev))
            zi.append(value_row)
            stddev_matrix.append(stddev_row)
        if return_std:
            return xi, yi, np.array(zi).T, np.array(stddev_matrix).T
        else:
            return xi, yi, np.array(zi).T


def plot_objective(
    result,
    levels=10,
    n_points=40,
    n_samples=250,
    size=2,
    zscale="linear",
    dimensions=None,
    usepartialdependence=False,
    pars="result",
    expected_minimum_samples=None,
    title=None,
    show_confidence=True,
    plot_options=None,
):
    """Pairwise dependence plot of the objective function.

    The diagonal shows the dependence for dimension `i` with
    respect to the objective function. The off-diagonal shows the
    dependence for dimensions `i` and `j` with
    respect to the objective function. The objective function is
    approximated by `result.model.`

    Pairwise scatter plots of the points at which the objective
    function was directly evaluated are shown on the off-diagonal.
    A red point indicates per default the best observed minimum, but
    this can be changed by changing argument ´pars´.

    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The result for which to create the scatter plot matrix.

    * `levels` [int, default=10]
        Number of levels to draw on the contour plot, passed directly
        to `plt.contour()`.

    * `n_points` [int, default=40]
        Number of points at which to evaluate the partial dependence
        along each dimension.

    * `n_samples` [int, default=250]
        Number of random samples to use for averaging the model function
        at each of the `n_points`.

    * `size` [float, default=2]
        Height (in inches) of each facet.

    * `zscale` [str, default='linear']
        Scale to use for the z axis of the contour plots. Either 'linear'
        or 'log'.

    * `dimensions` [list of str, default=None] Labels of the dimension
        variables. `None` defaults to `space.dimensions[i].name`, or
        if also `None` to `['X_0', 'X_1', ..]`.

    * `usepartialdependence` [bool, default=false] Whether to use partial
        dependence or not when calculating dependence. If false plot_objective
        will parse values to the dependence function, defined by the 
        pars argument

    * `pars` [str, default = 'result' or list of floats] Defines the values
    for the red
        points in the plots, and if partialdependence is false, this argument
        also defines values for all other parameters when calculating
        dependence.
        Valid strings:
            'result' - Use best observed parameters
            'expected_minimum' - Parameters that gives the best minimum
                Calculated using scipy's minimize method. This method
                currently does not work with categorical values.
            'expected_minimum_random' - Parameters that gives the best minimum
                when using naive random sampling. Works with categorical values
            '[x[0], x[1], ..., x[n]] - Parameter to show depence from a given x

    * `expected_minimum_samples` [float, default = None] Determines how many
    points should be evaluated to find the minimum when using
    'expected_minimum' or 'expected_minimum_random'

    * `title` [str, default=None]
        String to use as title of the figure

    * `show_confidence` [bool, default=true] Whether or not to show a credible
        range around the mean estimate on the 1d-plots in the diagonal. The
        credible range is given as 1.96 times the std in the point.

    * `plot_options` [dict or None, default None] A dict of the options for the plot. If,
        none, the defaults are used for all values.
        Possible keys:
        * interpolation [string, default ""] Which interpolation to use. If empty,
            contour plots are used. Note that contour plots are not compatible with
            showing uncertainty.
        * uncertain_color [color, default [0, 0, 0]] The color of the maximally
            uncertain data point.
        * colormap [string, default 'viridis_r'] The colormap to use in the 2D plots.
        * normalize_uncertainty [function with three inputs, default
            `lambda x, global_min, global_max: (x-global_min)/(global_max-global_min)`]
            The normalisation function for the uncertainty.
            x is a numpy array containing the standard deviations, global_min and 
            global_max are the min and max standard deviations for all 2D plots.
            Output must be a numpy array of the same dimensions as x with values between 
            0 and 1. If not, it raises an error, but only if any concrete outputs are
            outside this range. Output values of 0 shows the color for the expected 
            value modeled objective function. Output values of 1 corresponds to the 
            uncertain_color. 
            To not show any uncertainty, use `lambda x, min, max: 0*x`.
            Another use is to visually deemphasize points with medium uncertainty by e.g
            `lambda x, global_min, global_max: ((x-global_min)/(global_max-global_min))**(1/2)`.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    # Here we define the values for which to plot the red dot (2d plot) and
    # the red dotted line (1d plot). These same values will be used for
    # evaluating the plots when calculating dependence. (Unless partial
    # dependence is to be used instead).
    if not plot_options:
        plot_options = {}
    default_plot_type = {
       "interpolation": "",
       "uncertain_color": [0, 0, 0],
       "colormap" : "viridis_r",
       "normalize_uncertainty": (
           lambda x, global_min, global_max: (x-global_min)/(global_max-global_min)
       ),
    }
    for k,v in default_plot_type.items():
        if k not in plot_options.keys():
            plot_options[k] = v
    # zscale and levels really belong in plot_options, but for backwards 
    # compatibility they are given as separate arguments.
    plot_options["zscale"] = zscale
    plot_options["levels"] = levels
    space = result.space
    # Check if we have any categorical dimensions, as this influences the plots
    # on the diagonal (1D dependency)
    is_cat = [isinstance(dim, Categorical) for dim in space.dimensions]
    
    if isinstance(pars, str):
        if pars == "result":
            # Using the best observed result
            x_vals = result.x
        elif pars == "expected_minimum":

            # Do a gradient based minimum search using scipys own minimizer
            if expected_minimum_samples:
                # If a value for expected_minimum_samples has been parsed
                x_vals, _ = expected_minimum(
                    result,
                    n_random_starts=expected_minimum_samples,
                    random_state=None,
                )
            else:  # Use standard of 20 random starting points
                x_vals, _ = expected_minimum(
                    result, n_random_starts=20, random_state=None
                )
        elif pars == "expected_minimum_random":
            # Do a minimum search by evaluating the function with n_samples
            # sample values
            if expected_minimum_samples:
                # If a value for expected_minimum_samples has been parsed
                x_vals, _ = expected_minimum_random_sampling(
                    result,
                    n_random_starts=expected_minimum_samples,
                    random_state=None,
                )
            else:
                # Use standard of 10^n_parameters. Note this becomes very slow
                # for many parameters
                x_vals, _ = expected_minimum_random_sampling(
                    result, n_random_starts=100000, random_state=None
                )
        else:
            raise ValueError(
                "Argument ´pars´ must be a valid string \
            (´result´)"
            )
    elif isinstance(pars, list):
        assert len(pars) == len(
            result.x
        ), "Argument ´pars´ of type list \
        must have same length as number of features"
        # Using defined x_values
        x_vals = pars
    else:
        raise ValueError("Argument ´pars´ must be a string or a list")

    if usepartialdependence:
        x_eval = None
    else:
        x_eval = x_vals
    rvs_transformed = space.transform(space.rvs(n_samples=n_samples))
    samples, minimum, _ = _map_categories(space, result.x_iters, x_vals)
    
    # Build slightly larger plots when we have just two dimensions
    if space.n_dims == 2:
        size = size*1.75
        
    fig, ax = plt.subplots(
        space.n_dims,
        space.n_dims,
        figsize=(size * space.n_dims, size * space.n_dims),
    )
    
    # Generate consistent padding for axis labels and ticks
    l_pad = 0.7/fig.get_figwidth()
    r_pad = 1 - l_pad
    b_pad = 0.7/fig.get_figheight()
    t_pad = 1 - 0.7/fig.get_figheight()
    
    if space.n_dims <= 3:
        h_pad = 0.2
        w_pad = 0.2
    else:
        h_pad = 0.1
        w_pad = 0.1
    
    fig.subplots_adjust(
        left=l_pad, right=r_pad, bottom=b_pad, top=t_pad, hspace=h_pad, wspace=w_pad
    )    

    if title is not None:
        fig.suptitle(title)

    val_min_1d = float("inf")
    val_max_1d = -float("inf")
    val_min_2d = float("inf")
    val_max_2d = -float("inf")
    stddev_min_2d = float("inf")
    stddev_max_2d = -float("inf")

    plots_data = []
    # Gather all data relevant for plotting
    for i in range(space.n_dims):
        row = []
        for j in range(space.n_dims):

            if j > i:
                # We only plot the lower left half of the grid,
                # to avoid duplicates.
                break

            # The diagonal of the plot
            elif i == j:
                xi, yi, stddevs = dependence(
                    space,
                    result.models[-1],
                    i,
                    j=None,
                    sample_points=rvs_transformed,
                    n_points=n_points,
                    x_eval=x_eval,
                )
                row.append({"xi": xi, "yi": yi, "std": stddevs})

                
                if show_confidence:
                    yi_low_bound = yi - 1.96 * stddevs
                    yi_high_bound = yi + 1.96 * stddevs
                else:
                    yi_low_bound = yi
                    yi_high_bound = yi
                if np.min(yi_low_bound) < val_min_1d:
                    val_min_1d = np.min(yi_low_bound)
                if np.max(yi_high_bound) > val_max_1d:
                    val_max_1d = np.max(yi_high_bound)



            # lower triangle
            else:
                xi, yi, zi, stddevs = dependence(
                    space,
                    result.models[-1],
                    i,
                    j,
                    rvs_transformed,
                    n_points,
                    x_eval=x_eval,
                    return_std=True,
                )
                # print('filling with i, j = ' + str(i) + str(j))
                row.append({"xi": xi, "yi": yi, "zi": zi, "std": stddevs})

                if np.min(zi) < val_min_2d:
                    val_min_2d = np.min(zi)
                if np.max(zi) > val_max_2d:
                    val_max_2d = np.max(zi)
                if np.min(stddevs) < stddev_min_2d:
                    stddev_min_2d = np.min(stddevs)
                if np.max(stddevs) > stddev_max_2d:
                    stddev_max_2d = np.max(stddevs)

        plots_data.append(row)
    
    # Build all the plots of in the figure
    for i in range(space.n_dims):
        for j in range(space.n_dims):

            if j > i:
                # We only plot the lower left half of the grid,
                # to avoid duplicates.
                break

            # The diagonal of the plot, showing the 1D (partial) dependence for each of the n parameters
            elif i == j:

                xi = plots_data[i][j]["xi"]
                yi = plots_data[i][j]["yi"]
                stddevs = plots_data[i][j]["std"]
                
                # Check if we are about to plot a categoric factor
                if is_cat[i]:                    
                    # Expand the x-axis for this factor so we can see the first
                    # and the last category
                    ax[i, i].set_xlim(np.min(xi)-0.2, np.max(xi)+0.2)
                    # Use same y-axis as all other 1D plots
                    ax[i, i].set_ylim(val_min_1d-abs(val_min_1d)*.02, 
                                      val_max_1d+abs(val_max_1d)*.02)
                    if show_confidence:
                        # Create one uniformly colored bar for each category.
                        # Edgecolor ensures we can see the bar when plotting 
                        # at best obeservation, as stddev is often tiny there
                        ax[i, i].bar(
                            xi,
                            2*1.96*stddevs,
                            width=0.2,
                            bottom=yi-1.96*stddevs,
                            alpha=0.5,
                            color="green",
                            edgecolor="green",
                            zorder=1,
                        )
                        # Also add highlight the best point/expected minimum
                        ax[i, i].scatter(
                            minimum[i],
                            yi[int(minimum[i])],
                            c="k",
                            s=20,
                            marker="D",
                            zorder=0,
                        )
                    else:
                        # Simply show the mean value
                        ax[i, i].scatter(
                            xi,
                            yi,
                            c="red",
                            s=80,
                            marker="_",
                            zorder=1,
                        )
                        # Also add highlight the best/expected minimum
                        ax[i, i].scatter(
                            minimum[i],
                            yi[int(minimum[i])],
                            c="k",
                            s=20,
                            marker="D",
                            zorder=0,
                        )
                
                # For non-categoric factors
                else:
                    ax[i, i].set_xlim(np.min(xi), np.max(xi))
                    ax[i, i].set_ylim(val_min_1d-abs(val_min_1d)*.02, 
                                      val_max_1d+abs(val_max_1d)*.02)
                    ax[i, i].axvline(minimum[i], linestyle="--", color="k", lw=1)
                    if show_confidence:
                        ax[i, i].fill_between(xi,
                                              y1=(yi - 1.96*stddevs),
                                              y2=(yi + 1.96*stddevs),
                                              alpha=0.5,
                                              color="green",
                                              edgecolor="green",
                                              linewidth=0.0,
                            )
                    else:
                        ax[i, i].plot(
                            xi,
                            yi,
                            color="red",
                            lw=1,
                            zorder=0,
                        )

            # lower triangle
            elif i > j:
                _2d_dependency_plot(
                    data=plots_data[i][j],
                    axes=ax[i][j],
                    samples=(samples[:, j], samples[:, i]),
                    highlighted=(minimum[j], minimum[i]),
                    limits={
                        "z_min" : val_min_2d,
                        "z_max" : val_max_2d,
                        "stddev_min": stddev_min_2d,
                        "stddev_max": stddev_max_2d
                    },
                    options = plot_options
                )

                if [i, j] == [1, 0]:
                    # Add a colorbar for the 2D plot value scale
                    import matplotlib as mpl

                    norm_color = mpl.colors.Normalize(
                        vmin=val_min_2d, vmax=val_max_2d
                    )
                    cb = ax[0][-1].figure.colorbar(
                        mpl.cm.ScalarMappable(norm=norm_color, cmap=plot_options["colormap"]),
                        ax=ax[0][-1],
                        location="top",
                        fraction=0.1,
                        label="Score",
                    )
                    cb.ax.locator_params(nbins=8)
                    
                    # Add a legend for the various figure contents
                    if isinstance(pars, str):
                        if pars == "result":
                            highlight_label = "Best data point"
                        elif pars == "expected_minimum":
                            highlight_label = "Expected minimum"
                        elif pars == "expected_minimum_random":
                            highlight_label = "Simulated minimum"
                    elif isinstance(pars, list):
                        # The case where the user specifies [x[0], x[1], ...]
                        highlight_label = "Point: " + str(pars)
                    # Legend icon for data points
                    legend_data_point = mpl.lines.Line2D(
                        [],
                        [],
                        color="darkorange",
                        marker=".",
                        markersize=9,
                        lw=0.0,
                    )
                    # Legend icon for the highlighted point in the 2D plots
                    legend_hp = mpl.lines.Line2D(
                        [],
                        [],
                        color="k",
                        marker="D",
                        markersize=5,
                        lw=0.0,
                    )
                    # Legend icon for the highlighted value in the 1D plots
                    legend_hl = mpl.lines.Line2D(
                        [],
                        [],
                        linestyle="--",
                        color="k",
                        marker="",
                        lw=1,
                    )
                    if show_confidence:
                        # Legend icon for the 95 % credibility interval
                        legend_fill = mpl.patches.Patch(
                            color="green",
                            alpha=0.5,
                        )
                        if usepartialdependence:
                            ci_label = "Est. 95 % credibility interval"
                        else:
                            ci_label = "95 % credibility interval"
                        ax[0][-1].legend(
                            handles=[legend_data_point, (legend_hp, legend_hl), legend_fill],
                            labels=["Data points", highlight_label, ci_label],
                            loc="upper center",
                            handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)},
                        )
                    else:
                        # Legend icon for the model mean function
                        legend_mean = mpl.lines.Line2D(
                            [],
                            [],
                            linestyle="-",
                            color="red",
                            marker="",
                            lw=1,
                        )
                        ax[0][-1].legend(
                            handles=[legend_data_point, (legend_hp, legend_hl), legend_mean],
                            labels=["Data points", highlight_label, "Model mean function"],
                            loc="upper center",
                            handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)},
                        )
                    

    if usepartialdependence:
        ylabel = "Partial dependence"
    else:
        ylabel = "Dependence"

    return _format_scatter_plot_axes(
        ax, space, ylabel=ylabel, dim_labels=dimensions
    )

def _2d_dependency_plot(data, axes, samples, highlighted, limits, options = {}):
    if "zscale" in options.keys():
        if options["zscale"] == "log":
            locator = LogLocator()
        elif options["zscale"] == "linear":
            locator = None
        else:
            raise ValueError(
                "Valid values for zscale are 'linear' and 'log',"
                " not '%s'." % options["zscale"]
            )
    else :
        raise ValueError("No zscale given for 2D dependency plot")
    xi = data["xi"]
    yi = data["yi"]
    zi = data["zi"]
    if not options["interpolation"]:
        axes.contourf(
            xi,
            yi,
            zi,
            options["levels"],
            locator=locator,
            cmap=options["colormap"],
            vmin=limits["z_min"],
            vmax=limits["z_max"],
        )
    else:
        # Normalising z and stddev to scale beteween 0 and 1, needed for more manual plot
        zi = (zi-limits["z_min"])/(limits["z_max"]-limits["z_min"])
        #Converting numerical z values to RGBA values to be able to change alpha
        zi = plt.get_cmap(options["colormap"])(zi)
        stddev = data["std"]
        stddev = options["normalize_uncertainty"](
            stddev,
            limits["stddev_min"],
            limits["stddev_max"])
        if stddev.max() > 1:
            raise ValueError(
                "Normalization of uncertainty resulted in values above one")
        if stddev.min() < 0:
            raise ValueError(
                "Normalizetion of uncertainty resulted in values below zero")
        # Setting the alpha (opacity) to be inversly proportional to uncertainty, so
        # the background color shines thorugh in uncertain areas.
        for i in range(zi.shape[0]):
            for j in range(zi.shape[1]):
                zi[i,j,3] = 1-stddev[i,j]
        axes.set_facecolor(options["uncertain_color"])
        axes.imshow(
            zi,
            interpolation=options["interpolation"],
            origin="lower",
            extent=(min(xi),max(xi),min(yi),max(yi))
        )
        # imshow assumes square pixels, so it sets aspect to 1. We do not want that.
        axes.set_aspect('auto')
    axes.scatter(
        samples[0],
        samples[1],
        c="darkorange",
        s=20,
        lw=0.0,
        zorder=10,
        clip_on=False,
    )
    axes.scatter(
        highlighted[0],
        highlighted[1],
        c="k",
        s=30,
        marker="D",
        lw=0.0,
        zorder=9,
        clip_on=False,
    )

def plot_objective_1d(
    result,
    n_points=60,
    n_samples=250,
    size=2.5,
    dimensions=None,
    usepartialdependence=False,
    pars="result",
    expected_minimum_samples=None,
    title=None,
    show_confidence=True,
):
    """Single factor dependence plot of the objective function.

    The plot shows the dependence for each dimension `i` with
    respect to the objective function. The objective function is
    approximated by `result.model.` A vertical line indicates per default
    the best observed data point, but this can be changed via the
    argument 'pars'.

    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The result for which to create the scatter plot matrix.

    * `n_points` [int, default=40]
        Number of points at which to evaluate the partial dependence
        along each dimension.

    * `n_samples` [int, default=250]
        Number of random samples to use for averaging the model function
        at each of the `n_points`.

    * `size` [float, default=2.5]
        Height (in inches) of each facet.

    * `dimensions` [list of str, default=None] 
        Labels of the dimension variables. `None` defaults to 
        `space.dimensions[i].name`, or if also `None` to `['X_0', 'X_1', ..]`.

    * `usepartialdependence` [bool, default=false] 
        Whether to use partial dependence or not when calculating dependence. 
        If false then the function will parse values to the dependence 
        function, defined by the pars argument

    * `pars` [str, default = 'result' or list of floats, ints and/or strings] 
        Defines the nature of the highlighted setting in the points and if 
        usepartialdependence is false, this argument also defines values for 
        all other factors when calculating dependence.
        Valid strings:
            'result' - Use best observed factor settings in the input data.
            'expected_minimum' - Use factor settings expected to give the best 
                minimum calculated using scipy's minimize method.
            'expected_minimum_random' - Use factor settings that gives the best
                minimum after carrying out naive random sampling. Works with 
                categorical values.
            '[x[0], x[1], ..., x[n]] - Parameter to show depence at the factor
                settings provided in this list. Depending on the system, this 
                list can contain a mixture of floats, ints and strings

    * `expected_minimum_samples` [float, default = None] 
        Determines how many points should be evaluated to find the minimum when
        using 'expected_minimum' or 'expected_minimum_random'.

    * `title` [str, default=None]
        Alternative string to use as the title of the legend. If left as None 
        the title is dynamically generated based on the number of factors in 
        the model (N) after the template "Dependency plot across N factors"

    * `show_confidence` [bool, default=true] 
        Whether or not to show a 95 % credibility range for the model values
        for each function (when not using partial dependence). The range is
        defined by 1.96 times the std in each point when sampling from the
        model. When using partial dependence the range is not a credibility 
        range but is defined as 1.96 times the std for random sampling across
        the parameter space. It is labelled as an estimated credibility range.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    # Here we define the value to highlight in each dimension. These 
    # same values will be used for evaluating the plots when calculating 
    # dependence. (Unless partial dependence is to be used instead).
    
    space = result.space
    # Check if we have any categorical dimensions, as this influences the plots
    is_cat = [isinstance(dim, Categorical) for dim in space.dimensions]
    
    if isinstance(pars, str):
        if pars == "result":
            # Using the best observed result
            x_vals = result.x
        elif pars == "expected_minimum":
            # Do a gradient based minimum search using scipys own minimizer
            if expected_minimum_samples:
                # If a value for expected_minimum_samples has been parsed
                x_vals, _ = expected_minimum(
                    result,
                    n_random_starts=expected_minimum_samples,
                    random_state=None,
                )
            else:  # Use standard of 20 random starting points
                x_vals, _ = expected_minimum(
                    result,
                    n_random_starts=20,
                    random_state=None,
                )
        elif pars == "expected_minimum_random":
            # Do a minimum search by evaluating the function with n_samples
            # sample values
            if expected_minimum_samples:
                # If a value for expected_minimum_samples has been parsed
                x_vals, _ = expected_minimum_random_sampling(
                    result,
                    n_random_starts=expected_minimum_samples,
                    random_state=None,
                )
            else:
                # Use standard of 10^n_parameters. Note this becomes very slow
                # for many parameters
                x_vals, _ = expected_minimum_random_sampling(
                    result,
                    n_random_starts=10 ** space.n_dims,
                    random_state=None,
                )
        else:
            raise ValueError(
                "Argument ´pars´ must be a valid string \
            (´result´)"
            )
    elif isinstance(pars, list):
        assert len(pars) == len(
            result.x
        ), "Argument ´pars´ of type list \
        must have same length as number of features"
        # Using defined x_values
        x_vals = pars
    else:
        raise ValueError("Argument ´pars´ must be a string or a list")

    if usepartialdependence:
        x_eval = None
    else:
        x_eval = x_vals
    rvs_transformed = space.transform(space.rvs(n_samples=n_samples))
    samples, minimum, _ = _map_categories(space, result.x_iters, x_vals)
    
    # Build a figure using the smallest possible N by N tiling
    ncols = int(np.ceil(np.sqrt(space.n_dims)))
    nrows = int(np.ceil(space.n_dims/ncols))
    # Build slightly larger figures when we have few factors
    if ncols == 1:
        size = size*1.5
    elif ncols == 2:
        size = size*4/3
    
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(size * ncols, size * nrows),        
    )
    
    # Generate consistent padding for axis labels and ticks
    l_pad = 0.5/fig.get_figwidth()
    r_pad = 1 - l_pad
    b_pad = 0.6/fig.get_figheight()
    t_pad = 1 - 0.8/fig.get_figheight()
    if nrows > 1:
        h_pad = 1 / (t_pad/(b_pad*nrows) - 1)
    else:
        h_pad = 0

    if ncols == 1:
        # One factor is a very special edge-case
        fig.subplots_adjust(
            left=0.18, right=0.95, bottom=b_pad, top=t_pad, hspace=0.0, wspace=0.0
        )
    else:
        fig.subplots_adjust(
            left=l_pad, right=r_pad, bottom=b_pad, top=t_pad, hspace=h_pad, wspace=0.0
        )
    
    if title is None:
        title = "Dependency plot across " + str(space.n_dims) + " factors"

    val_min_1d = float("inf")
    val_max_1d = -float("inf")

    plots_data = []
    # Gather all data relevant for plotting
    for i in range(space.n_dims):
        row = []
        xi, yi, stddevs = dependence(
            space,
            result.models[-1],
            i,
            j=None,
            sample_points=rvs_transformed,
            n_points=n_points,
            x_eval=x_eval,
        )
        row.append({"xi": xi, "yi": yi, "std": stddevs})
        
        if show_confidence:
            yi_low_bound = yi - 1.96 * stddevs
            yi_high_bound = yi + 1.96 * stddevs
        else:
            yi_low_bound = yi
            yi_high_bound = yi
        if np.min(yi_low_bound) < val_min_1d:
            val_min_1d = np.min(yi_low_bound)
        if np.max(yi_high_bound) > val_max_1d:
            val_max_1d = np.max(yi_high_bound)

        plots_data.append(row)
    
    # Build all the plots in the figure
    for n in range(space.n_dims):
        # Generate a handle to the subplot we are targeting
        if space.n_dims == 1:
            ax_ = ax
        elif space.n_dims == 2:
            ax_ = ax[n]
        else:
            i, j = np.unravel_index(n, ax.shape)
            ax_ = ax[i, j]
        # Get data to plot in this subplot
        xi = plots_data[n][0]["xi"]
        yi = plots_data[n][0]["yi"]
        stddevs = plots_data[n][0]["std"]        
        
        # Set y-axis limits with a small buffer
        ax_.set_ylim(
            val_min_1d-abs(val_min_1d)*.02,
            val_max_1d+abs(val_max_1d)*.02
        )        
        
        # Enter here when we plot a categoric factor
        if is_cat[n]:                    
            # Expand the x-axis for this factor so we can see the first
            # and the last category
            ax_.set_xlim(np.min(xi)-0.2, np.max(xi)+0.2)
            
            if show_confidence:
                # Create one uniformly colored bar for each category.
                # Edgecolor ensures we can see the bar when plotting 
                # at best obeservation, as stddev is often tiny there
                ax_.bar(
                    xi,
                    2*1.96*stddevs,
                    width=0.2,
                    bottom=yi-1.96*stddevs,
                    alpha=0.5,
                    color="green",
                    edgecolor="green",
                    zorder=1,
                )
            else:
                # Show the mean value
                ax_.scatter(
                    xi,
                    yi,
                    c="red",
                    s=80,
                    marker="_",
                    zorder=1,
                )
        
        # For non-categoric factors
        else:
            ax_.set_xlim(np.min(xi), np.max(xi))
            
            if show_confidence:
                ax_.fill_between(
                    xi,
                    y1=(yi - 1.96*stddevs),
                    y2=(yi + 1.96*stddevs),
                    alpha=0.5,
                    color="green",
                    edgecolor="green",
                    linewidth=0.0,
                )
            else:
                ax_.plot(
                    xi,
                    yi,
                    color="red",
                    lw=1,
                    zorder=0,
                )
        
        # Highlight the point defined by 'pars'
        ax_.axvline(minimum[n], linestyle="--", color="r", lw=2, zorder=6)
        
        # Add a legend to the figure
        if n == 0:
            if isinstance(pars, str):
                if pars == "result":
                    highlight_label = "Best data point"
                elif pars == "expected_minimum":
                    highlight_label = "Expected minimum"
                elif pars == "expected_minimum_random":
                    highlight_label = "Simulated minimum"
                elif isinstance(pars, list):
                    # The case where the user specifies [x[0], x[1], ...]
                    highlight_label = "Point: " + str(pars)
            # Legend icon for the highlighted value
            legend_hl = mpl.lines.Line2D(
                [],
                [],
                linestyle="--",
                color="r",
                marker="",
                lw=2,
            )

            if show_confidence:
                # Legend icon for the 95 % credibility interval
                legend_fill = mpl.patches.Patch(
                    color="green",
                    alpha=0.5,
                )
                if usepartialdependence:
                    ci_label = "Est. 95 % credibility interval"
                else:
                    ci_label = "95 % credibility interval"
                # Build legend
                ax_.figure.legend(
                    handles=[legend_hl, legend_fill],
                    labels=[highlight_label, ci_label],
                    title=title,
                    framealpha=1,
                    loc="upper center",
                    handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)},
                )
            else:
                # Legend icon for the model mean function
                legend_mean = mpl.lines.Line2D(
                    [],
                    [],
                    linestyle="-",
                    color="red",
                    marker="",
                    lw=1,
                )
                # Build legend
                ax_.figure.legend(
                    handles=[legend_hl, legend_mean],
                    labels=[highlight_label, "Model mean function"],
                    title=title,
                    framealpha=1,
                    loc="upper center",
                    handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)},
                )        

    if usepartialdependence:
        ylabel = "Partial dependence"
    else:
        ylabel = "Dependence"

    return _format_1d_dependency_axes(
        ax, space, ylabel=ylabel, dim_labels=dimensions
    )

def plot_brownie_bee_frontend(
    result,
    n_points=60,
    n_samples=250,
    size=3,
    max_quality=5,
):
    """Single factor dependence plot of the model intended for use with the 
    Brownie Bee user interface.

    Each plot shows how quality depends on the dimension `i` when all other 
    factor values are locked to those of the expected minimum. A vertical line 
    indicates the location of the expected minimum for each factor.

    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The result for which to create the plots.

    * `n_points` [int, default=60]
        Number of points at which to evaluate the partial dependence
        along each dimension.

    * `n_samples` [int, default=250]
        Number of random samples to use for averaging the model function
        at each of the `n_points`.

    * `size` [float, default=3]
        Height (in inches) of each returned figure.

    * `max_quality` [int, default=5] 
        The maximal quality intended for the setup of Brownie Bee. Quality is
        assumed to be measured on a scale from 0 to this number, and the y-axis
        of each plot is scaled to reflect this. If the uncertainty reaches above
        this number, the y-axis is expanded to accomodate this.

    Returns
    -------
    * `plot_list`: [`Figures`]:
        A list of individual matplotlib figure handles, one for each dimension
        present in 'result' and a last one representing a histogram of samples
        drawn at the expected minimum.
    """
   
    space = result.space
    # Check if we have any categorical dimensions, as this influences the plots
    is_cat = [isinstance(dim, Categorical) for dim in space.dimensions]
    # Check if we have any integer dimensions, as this influences the plots
    is_int = [isinstance(dim, Integer) for dim in space.dimensions]
    # Identify the location of the expected minimum, and its mean and std
    x_eval, [res_mean, res_std] = expected_minimum(
        result,
        n_random_starts=20,
        random_state=None,
        return_std=True,
    )
        
    rvs_transformed = space.transform(space.rvs(n_samples=n_samples))
    _, minimum, _ = _map_categories(space, result.x_iters, x_eval)
    
    # Gather all data relevant for plotting
    plots_data = []
    # Starting point for our maximum visible quality
    plot_max_q = max_quality
    
    for i in range(space.n_dims):
        row = []
        xi, yi, stddevs = dependence(
            space,
            result.models[-1],
            i,
            j=None,
            sample_points=rvs_transformed,
            n_points=n_points,
            x_eval=x_eval,
        )
        row.append({"xi": xi, "yi": yi, "std": stddevs})
        plots_data.append(row)
        # Check if our quality interval goes above our desired max_quality view
        if -np.min(yi-1.96*stddevs) > plot_max_q:
            plot_max_q = -np.min(yi-1.96*stddevs)
    
    # Create the list to store figure handles
    figure_list = []  
    
    # Expand maximal quality value if needed to show upper bound on uncertainty
    max_quality = plot_max_q
    
    # Build all the plots in the figure
    for n in range(space.n_dims):
        # Prepare a figure    
        fig, ax_ = plt.subplots(
            figsize=(size, size),
            dpi=200,
        )
        # Set the padding
        fig.subplots_adjust(
            left=0.12, right=0.93, bottom=0.2, top=0.95, hspace=0.0, wspace=0.0
        )

        # Get data to plot in this subplot
        xi = plots_data[n][0]["xi"]
        yi = plots_data[n][0]["yi"]
        stddevs = plots_data[n][0]["std"]        
        
        # Set y-axis limits with a small buffer
        ax_.set_ylim(0, max_quality*1.02)
        
        # Enter here when we plot a categoric factor
        if is_cat[n]:
            # Expand the x-axis for this factor so we can see the first
            # and the last category
            ax_.set_xlim(np.min(xi)-0.2, np.max(xi)+0.2)            
            
            # Create one uniformly colored bar for each category.
            # Edgecolor ensures we can see the bar when plotting 
            # at best obeservation, as stddev is often tiny there
            ax_.bar(
                xi,
                2*1.96*stddevs,
                width=0.2,
                bottom=(-yi-1.96*stddevs),
                alpha=0.5,
                color="green",
                edgecolor="green",
            )

            # Adjust font size according to the number of labesl
            if len(xi) < 3:
                [labl.set_fontsize(10) for labl in ax_.get_xticklabels()]
            elif len(xi) < 6:
                [labl.set_fontsize(7) for labl in ax_.get_xticklabels()]
            else:
                [labl.set_fontsize(5) for labl in ax_.get_xticklabels()]
        
        # For non-categoric factors
        else:
            ax_.set_xlim(np.min(xi), np.max(xi))
            # Show the uncertainty
            ax_.fill_between(
                xi,
                y1=-(yi - 1.96*stddevs),
                y2=-(yi + 1.96*stddevs),
                alpha=0.5,
                color="green",
                edgecolor="green",
                linewidth=0.0,
            )
        
        # Highlight the expected minimum
        ax_.axvline(minimum[n], linestyle="--", color="r", lw=2, zorder=6)        
        
        # Fix formatting of the y-axis with ticks from 0 to our max quality
        ax_.yaxis.set_major_locator(MaxNLocator("auto", integer=True))
        ax_.tick_params(axis="y", direction="inout")
        
        if space.dimensions[n].prior == "log-uniform":
            ax_.set_xscale("log")
        else:
            ax_.xaxis.set_major_locator(
                MaxNLocator("auto", prune=None, integer=(is_cat[n] | is_int[n]))
            )
            if is_cat[n]:
                # Axes for categorical dimensions are really integers; 
                # we have to label them with the category names
                ax_.xaxis.set_major_formatter(
                    FuncFormatter(
                        partial(_cat_format, space.dimensions[n])
                    )
                )
                # Rotate the labels if we have many of them to help it fit
                if len(xi) > 3:                    
                    plt.xticks(rotation=45)
        
        # Add the figure to the output list
        figure_list.append(fig)
    
    # Prepare a figure for a histogram of expected quality
    fig, ax_ = plt.subplots(
        figsize=(size, size),
        dpi=200,
    )
    # Set the padding
    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.2, top=0.95, hspace=0.0, wspace=0.0
    )
    # Plot in the interval between 0 and our max quality
    xi = np.linspace(0, max_quality*1.02, 250)
    # Create histogram y-values
    yi = norm.pdf(xi, -res_mean, res_std)
    # Build the plot
    ax_.fill_between(
        xi,
        y1=np.zeros((len(xi),)),
        y2=yi,
        alpha=0.5,
        color="blue",
        edgecolor="blue",
        linewidth=0.0,
    )
    # Cosmetics
    ax_.get_yaxis().set_visible(False)
    ax_.set_ylim(0, max(yi)*1.05)
    # Fix formatting of the x-axis with ticks from 0 to our max quality
    ax_.set_xlim(0, max_quality*1.02)
    ax_.xaxis.set_major_locator(
        MaxNLocator("auto", prune=None, integer=True)
    )
    
    # Add the figure to the output list
    figure_list.append(fig)
    
    return figure_list

def plot_objectives(
    results,
    levels=10,
    n_points=40,
    n_samples=250,
    size=2,
    zscale="linear",
    dimensions=None,
    usepartialdependence=False,
    pars="result",
    expected_minimum_samples=None,
    titles=None,
    show_confidence=True,
    plot_options=None,
):
    """Pairwise dependence plots of each of the objective functions.
    Parameters
    ----------
    * `results` [list of `OptimizeResult`]
        The list of results for which to create the objective plots.

    * `titles` [list of str, default=None]
        The list of strings of the names of the objectives used as titles in
        the figures

    """
    assert type(results) is list, (
        'I expected a list of results from a multiobjective optimization. '
        'You might have wanted plot_objective() (singular)')

    if titles is None:
        for result in results:
            plot_objective(
                result,
                levels=levels,
                n_points=n_points,
                n_samples=n_samples,
                size=size,
                zscale=zscale,
                dimensions=dimensions,
                usepartialdependence=usepartialdependence,
                pars=pars,
                expected_minimum_samples=expected_minimum_samples,
                title=None,
                show_confidence=show_confidence,
                plot_options=plot_options,
            )
        return
    else:
        for k in range(len(results)):
            plot_objective(
                results[k], 
                levels=levels,
                n_points=n_points,
                n_samples=n_samples,
                size=size,
                zscale=zscale,
                dimensions=dimensions,
                usepartialdependence=usepartialdependence,
                pars=pars,
                expected_minimum_samples=expected_minimum_samples,
                title=titles[k],
                show_confidence=show_confidence,
                plot_options=plot_options,
            )
        return


def plot_evaluations(result, bins=20, dimensions=None):
    """Visualize the order in which points where sampled.

    The scatter plot matrix shows at which points in the search
    space and in which order samples were evaluated. Pairwise
    scatter plots are shown on the off-diagonal for each
    dimension of the search space. The order in which samples
    were evaluated is encoded in each point's color.
    The diagonal shows a histogram of sampled values for each
    dimension. A red point indicates the found minimum.

    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The result for which to create the scatter plot matrix.

    * `bins` [int, bins=20]:
        Number of bins to use for histograms on the diagonal.

    * `dimensions` [list of str, default=None] Labels of the dimension
        variables. `None` defaults to `space.dimensions[i].name`, or
        if also `None` to `['X_0', 'X_1', ..]`.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    if type(result) is list:
        result = result[0]

    space = result.space
    # Convert categoricals to integers, so we can ensure consistent ordering.
    # Assign indices to categories in the order they appear in the Dimension.
    # Matplotlib's categorical plotting functions are only present in v 2.1+,
    # and may order categoricals differently in different plots anyway.
    samples, minimum, iscat = _map_categories(space, result.x_iters, result.x)
    order = range(samples.shape[0])
    fig, ax = plt.subplots(
        space.n_dims,
        space.n_dims,
        figsize=(2 * space.n_dims, 2 * space.n_dims),
    )

    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.1, wspace=0.1
    )

    for i in range(space.n_dims):
        for j in range(space.n_dims):
            if i == j:
                if iscat[j]:
                    bins_ = len(space.dimensions[j].categories)
                elif space.dimensions[j].prior == "log-uniform":
                    low, high = space.bounds[j]
                    bins_ = np.logspace(np.log10(low), np.log10(high), bins)
                else:
                    bins_ = bins
                ax[i, i].hist(
                    samples[:, j],
                    bins=bins_,
                    range=None if iscat[j] else space.dimensions[j].bounds,
                )

            # lower triangle
            elif i > j:
                ax[i, j].scatter(
                    samples[:, j],
                    samples[:, i],
                    c=order,
                    s=40,
                    lw=0.0,
                    cmap="viridis",
                    zorder=10,
                    clip_on=False
                )
                ax[i, j].scatter(minimum[j], minimum[i], c=["r"], s=20, lw=0.0, zorder=10, clip_on=False)

    return _format_scatter_plot_axes(
        ax, space, ylabel="Number of samples", dim_labels=dimensions
    )


def _map_categories(space, points, minimum):
    """
    Map categorical values to integers in a set of points.

    Returns
    -------
    * `mapped_points` [np.array, shape=points.shape]:
        A copy of `points` with categoricals replaced with their indices in
        the corresponding `Dimension`.

    * `mapped_minimum` [np.array, shape=(space.n_dims,)]:
        A copy of `minimum` with categoricals replaced with their indices in
        the corresponding `Dimension`.

    * `iscat` [np.array, shape=(space.n_dims,)]:
       Boolean array indicating whether dimension `i` in the `space` is
       categorical.
    """
    points = np.asarray(points, dtype=object)  # Allow slicing, preserve cats
    iscat = np.repeat(False, space.n_dims)
    min_ = np.zeros(space.n_dims)
    pts_ = np.zeros(points.shape)
    for i, dim in enumerate(space.dimensions):
        if isinstance(dim, Categorical):
            iscat[i] = True
            catmap = dict(zip(dim.categories, count()))
            pts_[:, i] = [catmap[cat] for cat in points[:, i]]
            min_[i] = catmap[minimum[i]]
        else:
            pts_[:, i] = points[:, i]
            min_[i] = minimum[i]
    return pts_, min_, iscat


def _evenly_sample(dim, n_points):
    """Return `n_points` evenly spaced points from a Dimension.

    Parameters
    ----------
    * `dim` [`Dimension`]
        The Dimension to sample from.  Can be categorical; evenly-spaced
        category indices are chosen in order without replacement (result
        may be smaller than `n_points`).

    * `n_points` [int]
        The number of points to sample from `dim`.

    Returns
    -------
    * `xi`: [np.array]:
        The sampled points in the Dimension.  For Categorical
        dimensions, returns the index of the value in
        `dim.categories`.

    * `xi_transformed`: [np.array]:
        The transformed values of `xi`, for feeding to a model.
    """
    cats = np.array(getattr(dim, "categories", []), dtype=object)
    if len(cats):  # Sample categoricals while maintaining order
        xi = np.linspace(0, len(cats) - 1, min(len(cats), n_points), dtype=int)
        xi_transformed = dim.transform(cats[xi])
    else:
        bounds = dim.bounds
        xi = np.linspace(bounds[0], bounds[1], n_points)
        xi_transformed = dim.transform(xi)
    return xi, xi_transformed


def _cat_format(dimension, x, _):
    """Categorical axis tick formatter function.  Returns the name of category
    `x` in `dimension`.  Used with `matplotlib.ticker.FuncFormatter`."""
    x = min(max(int(x), 0), len(dimension.categories)-1)
    label = str(dimension.categories[x])
    # If longer than 10 characters, try to break on spaces
    if len(label) > 10:
        if ' ' in label:
            # Break label at a space near the middle
            spaces = [i for i in range(len(label)) if label[i] == ' ']
            middle_space = spaces[len(spaces)//2]
            label = label[:middle_space] + '\n' + label[middle_space+1:]
        else:
            # If no spaces, abbreviate to first 7 characters
            label = label[:7] + '...'
    return label


def plot_expected_minimum_convergence(
    result, figsize=(15, 15), random_state=None, sigma=0.5
):
    """
    A function to perform a retrospective analysis of all the data points by
    building successive models and predicting the mean of the functional value
    of the surogate model in the expected minimum together with a measure for
    the variability in the suggested mean value. This code "replays" the
    entire optimization, hence it builds quiet many models and can, thus, seem
    slow. 
    TODO: Consider allowing user to subselect in data, to e.g. perform
    the analysis using every n-th datapoint, or only performing the analysis
    for the last n datapoints.

    Args:
        result ([`OptimizeResult`]): The result returned from a opt.tell
        command or from using get_result() or create_result()
        figsize (`tuple`, optional): [Gives the desired size of the resulting
        chart]. Defaults to (15, 15).
        random_state ([`Int`], optional): [abbility to pass random state to
        ensure reproducibility in plotting]. Defaults to None.
        sigma (`float`, optional): [Sigma gives a slight gaussian smoothing to
        the depiction of the credible interval around the expected minimum
        value.]. Defaults to 0.5.

    Returns:
        [ax]: [description]
    """

    estimated_mins_x = []
    estimated_mins_y = []
    quants_list = []
    distances = []
    for i in range(len(result.x_iters)):
        # Build an optimizer with as close to those used during the data
        # generating as possible. TODO: add more details on the optimizer build
        _opt = Optimizer(
            result.space.bounds, n_initial_points=1, random_state=random_state
        )
        # Tell the available data
        if i == 0:
            _result_internal = _opt.tell(
                result.x_iters[: i + 1][0], result.func_vals[: i + 1].item()
            )
        else:
            _result_internal = _opt.tell(
                result.x_iters[: i + 1], result.func_vals[: i + 1].tolist()
            )
        # Ask for the expected minimum in the result
        _exp = expected_minimum(_result_internal, random_state=random_state)
        # Append expected minimum to list. To plot it later
        estimated_mins_x.append(_exp[0])
        estimated_mins_y.append(_exp[1])
        # Transform x-value into transformed space and sample n times
        # Make 95% quantiles of samples
        transformed_point = _opt.space.transform([_exp[0],])
        samples_of_y = _result_internal.models[-1].sample_y(
            transformed_point, n_samples=10000, random_state=random_state
        )
        quants = mquantiles(samples_of_y.flatten(), [0.025, 0.975])
        quants_list.append(quants)

        # Calculate distance in x-space from last "believed" expected_min
        if i == 0:
            distancefromlast = _opt.space.distance(
                estimated_mins_x[-1], result.x
            )  # opt.Xi[0]
            distances.append(distancefromlast)
        else:
            distancefromlast = _opt.space.distance(
                estimated_mins_x[-1], estimated_mins_x[-2]
            )
            distances.append(distancefromlast)

        # Smoothing quantiles for graphically pleasing plot
        quant_max_smooth = gaussian_filter1d(
            [i[1] for i in quants_list], sigma=sigma
        )
        quant_min_smooth = gaussian_filter1d(
            [i[0] for i in quants_list], sigma=sigma
        )

    # Do the actual plotting
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=figsize)
    ax1.fill_between(
        list(range(1, len(result.x_iters) + 1)),
        y1=quant_min_smooth,
        y2=quant_max_smooth,
        alpha=0.5,
        color="grey",
    )
    ax1.plot(list(range(1, len(result.x_iters) + 1)), estimated_mins_y)
    ax1.set_ylabel('expected "y"-value @ expected min')

    ax2.plot(list(range(1, len(result.x_iters) + 1)), distances)
    ax2.set_ylabel("euclidian distance in x space from previous expected min")
    ax2.set_xticks(list(range(1, len(result.x_iters) + 1)))

    plt.xlabel("Number of calls $n$")
    return fig


def plot_Pareto(
    optimizer,
    figsize=(15, 15),
    objective_names=None,
    dimensions=None,
    return_data=False,
):
    """Interactive plot of the Pareto front implemented in 2 and 3 dimensions

    The plot shows all observations and the estimated Pareto front in the
    objective space. By hovering over each point it is possible to see the
    corresponding values of the point in the input space.

    Parameters
    ----------
    * `optimizer` [`Optimizer`]
        The optimizer containing data and the model

    * `figsize` ['tuple', default=(15,15)]
        Size of figure

    * `objective_names` [list, default=None]
        List of objective names. Used for plots. If None the objectives
        will be named "Objective 1", "Objective 2"...

    * `dimensions` [list, default=None]
        List of dimension names. Used for plots. If None the dimensions
        will be named "x_1", "x_2"...
        
    * `return_data` [bool, default=False]
        Whether to return data or not. If True the function will return
        all data for observation and estimated Pareto front, dimensions
        and objectives_names


    if return_data is true Returns
    -------
    * `np.array(optimizer.Xi)`: [numpy.ndarray]:
        Observations
    * `np.array(optimizer.yi)`: [numpy.ndarray]:
        Observed objective scores
    * `pop`: [numpy.ndarray]:
        Pareto front
    * `front`: [numpy.ndarray]:
        Pareto front objective scores
    * `dimensions`: [list]:
        Names of dimensions
    * `objective_names`: [list]:
        Objective names
    """

    def update_annot(ind, vals, sc):

        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos

        exp_params = vals[ind["ind"][0]]
        values = [str(number) for number in exp_params]
        strings = []
        for dim, value in zip(dimensions, values):
            strings.append(dim + ": " + value + str("\n"))
        strings[-1] = strings[-1].replace("\n", "")
        string = "".join(map(str, strings))

        annot.set_text(string)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event, vals=None, sc=None):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind, vals, sc)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    if optimizer.models == []:
        raise ValueError("No models have been fitted yet")

    if optimizer.n_objectives == 1:
        raise ValueError(
            "Pareto_plot is not possible with single objective optimization"
        )

    if optimizer.n_objectives > 3:
        raise ValueError("Pareto_plot is not possible with >3 objectives")

    if dimensions == None:
        dimensions = optimizer.space.names

    if len(dimensions) != len(optimizer.space.dimensions):
        raise ValueError(
            "Number of dimensions specified does not match the number of"
            "dimensions in the optimizers space"
        )

    pop, logbook, front = optimizer.NSGAII(MU=40)

    pop = np.asarray(pop)
    pop = np.asarray(
        optimizer.space.inverse_transform(
            pop.reshape(len(pop), optimizer.space.transformed_n_dims)
        )
    )

    if optimizer.n_objectives == 2:

        fig, ax = plt.subplots(figsize=figsize)
        plt.title("Pareto Front in Objective Space")

        if objective_names == None:
            objective_names = ["Objective 1", "Objective 2"]

        if len(objective_names) != 2:
            raise ValueError(
                "Number of objective_names is not equal to number of objectives"
            )

        plt.xlabel(objective_names[0])
        plt.ylabel(objective_names[1])

        all_points = np.concatenate([np.array(optimizer.yi), front])
        colors = ["black"] * len(optimizer.yi) + ["red"] * len(front)
        sc = plt.scatter(all_points[:, 0], all_points[:, 1], s=8, c=colors)

        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        annot.set_visible(False)
        fig.canvas.mpl_connect(
            "motion_notify_event",
            lambda event: hover(
                event,
                vals=np.concatenate([np.array(optimizer.Xi), pop]),
                sc=sc,
            ),
        )

        colors = ["black", "red"]
        texts = ["Observations", "Estimated Pareto Front"]
        patches = [
            plt.plot(
                [],
                [],
                marker="o",
                ms=6,
                ls="",
                mec=None,
                color=colors[i],
                label="{:s}".format(texts[i]),
            )[0]
            for i in range(len(texts))
        ]
        plt.legend(
            handles=patches,
            bbox_to_anchor=(0.5, 0.5),
            loc="best",
            ncol=1,
            numpoints=1,
        )

    if optimizer.n_objectives == 3:

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")
        plt.title("Pareto Front in Objective Space")

        if objective_names == None:
            objective_names = ["Objective 1", "Objective 2", "Objective 3"]

        if len(objective_names) != 3:
            raise ValueError(
                "Number of objective_names is not equal to number of objectives"
            )

        ax.set_xlabel(objective_names[0])
        ax.set_ylabel(objective_names[1])
        ax.set_zlabel(objective_names[2])
        ax.view_init(30, 240)

        all_points = np.concatenate([np.array(optimizer.yi), front])
        colors = ["black"] * len(optimizer.yi) + ["red"] * len(front)
        sc = ax.scatter(
            all_points[:, 0], all_points[:, 1], all_points[:, 2], s=8, c=colors
        )

        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        annot.set_visible(False)

        fig.canvas.mpl_connect(
            "motion_notify_event",
            lambda event: hover(
                event,
                vals=np.concatenate([np.array(optimizer.Xi), pop]),
                sc=sc,
            ),
        )

        colors = ["black", "red"]
        texts = ["Observations", "Estimated Pareto Front"]
        patches = [
            plt.plot(
                [],
                [],
                marker="o",
                ms=6,
                ls="",
                mec=None,
                color=colors[i],
                label="{:s}".format(texts[i]),
            )[0]
            for i in range(len(texts))
        ]
        plt.legend(
            handles=patches,
            bbox_to_anchor=(0.5, 0.5),
            loc="best",
            ncol=1,
            numpoints=1,
        )
    plt.show()

    if return_data is True:
        return (
            np.array(optimizer.Xi),
            np.array(optimizer.yi),
            pop,
            front,
            dimensions,
            objective_names,
        )

def plot_Pareto_bokeh(
    optimizer,
    objective_names=None,
    dimensions=None,
    return_data=False,
    show_browser=False,
    return_type_bokeh = None,
    filename='ParetoPlot',
):
    """Interactive bokeh plot of the Pareto front implemented in two dimensions

    The plot shows all observations and the estimated Pareto front in the
    objective space. By hovering over each point it is possible to see the
    corresponding values of the point in the input space.
    Data and bokeh objects are exclusively returned.<MON: What does this mean?>

    Parameters
    ----------
    * `optimizer` [`Optimizer`]
        The optimizer containing data and the model.

    * `objective_names` [list, default=None]
        List of objective names. Used for plots. If None the objectives
        will be named "Objective 1", "Objective 2"...

    * `dimensions` [list, default=None]
        List of dimension names. Used for plots. If None the dimensions
        will be named "X_1", "X_2"...
        
    * `return_data` [bool, default=False]
        Whether to return data or not. If True the function will return
        all data for observation and estimated Pareto front, dimensions
        and objectives_names.
        
    * `show_browser` [bool, default=False]
        Whether to open the new plot in the browser or not. If True new
        HTML-file is opened in the default browser.

    * `return_type_bokeh` ["file", "htmlString", "embed", or "json", default="file"]
        Determine how the bokeh plot is returned. Can be either
        
        - '"file"' for a HTML-file returned to the present working directory
          with the name 'filename'.html
        - '"htmlString"' for a string containing the HTML code
        - '"embed"' for <script> and <div> components for embeding. See
        https://docs.bokeh.org/en/latest/_modules/bokeh/embed/standalone.html#components
        for more information.
        - '"json"' for a json-item that can be supplied to BokehJS that
        https://docs.bokeh.org/en/latest/docs/user_guide/embed.html#json-items
        for more information.
        
    * `filename` [str, default='ParetoPlot']
        The filename to apply to the generated HTML-file.


    if return_data is True and return_type_bokeh is 'file' Returns
    -------
    * `np.array(optimizer.Xi)`: [numpy.ndarray]:
        Points at which the objectives have been evaluated.
    * `np.array(optimizer.yi)`: [numpy.ndarray]:
        Values of the objectives at corresponding points in 'Xi'.
    * `pop`: [numpy.ndarray]:
        Points on the Pareto front.
    * `front`: [numpy.ndarray]:
        Objective scores along the Pareto front points in 'pop'.
    * `dimensions`: [list]:
        Names of dimensions.
    * `objective_names`: [list]:
        Objective names.

    if return_type_bokeh is 'embed' and return_data is False Returns
    -------
    * `script`: [str]:
        Script part to be embeded acording to 
        https://docs.bokeh.org/en/latest/docs/user_guide/embed.html#components
    * `div`: [str]:
        <div> tag[s] part to be embeded acording to 
        https://docs.bokeh.org/en/latest/docs/user_guide/embed.html#components
    """
    if not optimizer.models:
        raise ValueError("No models have been fitted yet.")
    
    if dimensions == None:
        dimensions = [
            "X_(%i)" % i if d.name is None else d.name
            for i, d in enumerate(optimizer.space.dimensions)
        ]

    if len(dimensions) != len(optimizer.space.dimensions):
        raise ValueError(
            "Number of dimensions specified does not match the number of"
            "dimensions in the optimizer space."
        )

    if return_type_bokeh not in ['htmlString', 'embed', 'file', 'json', None]:
        raise NameError(f"'{return_type_bokeh}' is an unsupported return type for bokeh plot.")
    
    # Get objective names
    if objective_names:
        obj1 = objective_names[0]
        obj2 = objective_names[1]
    else:
        obj1 = 'Objective 1'
        obj2 = 'Objective 2'
        objective_names = [obj1, obj2]
    
    # Obtain the 'recipe' from the optimizer
    pop, logbook, front = optimizer.NSGAII(MU=100)
    pop = np.asarray(pop)
    pop = np.asarray(
        optimizer.space.inverse_transform(
            pop.reshape(len(pop), optimizer.space.transformed_n_dims)
        )
    )
    
    # Collect data for observed and Pareto-front into dicts
    data_observed_dict = {obj1: [i[0] for i in optimizer.yi],
                     obj2: [i[1] for i in optimizer.yi]}
    
    for i, dim in enumerate(dimensions):
        data_observed_dict[dim] =  [x[i] for x in optimizer.Xi]

    data_calculated_dict = {
        'front_x': np.unique(front, axis=0)[:,0].tolist(),
        'front_y': np.unique(front, axis=0)[:,1].tolist()}
    for i, dim in enumerate(dimensions):
        data_calculated_dict[dim] = pop[np.unique(front, axis=0, return_index=True)[1],i].tolist()
    
    # Create Tooltip strings for the observed data points
    Tooltips_observed = '''<div><font size="2"><b>Settings for this point are:</b></font></div>'''
    for dim in dimensions:
        Tooltips_observed += '''<div><font size="2">'''+dim+''' = @{'''+dim+'''}{0.0}</font></div>'''
    Tooltips_observed += '''<div><font size="2"><b>Scores for this point are:</b></font></div>'''
    Tooltips_observed += '''<div><font size="2">[ @{'''+obj1+'''}{0.00} , @{'''+obj2+'''}{0.00} ]</font></div>'''
    
    # Create Tooltip strings for calculated points on the Pareto-front
    Tooltips_recipe = '''<div><font size="2"><b>Settings for this point are:</b></font></div>'''
    for dim in dimensions:
        Tooltips_recipe += '''<div><font size="2">'''+dim+''' = @{'''+dim+'''}{0.0}</font></div>'''
    Tooltips_recipe += '''<div><font size="2"><b>Predicted scores for this point are:</b></font></div>'''
    Tooltips_recipe += '''<div><font size="2">[ @{front_x}{0.00} , @{front_y}{0.00} ]</font></div>'''
    
    # Load data into bokeh object
    source_observed = bh_models.ColumnDataSource(data_observed_dict)
    source_calculated = bh_models.ColumnDataSource(data_calculated_dict)

    # Find bounds for the zoom of the figure
    xlimitmax=max(max(data_calculated_dict['front_x']), max(data_observed_dict[obj1]))*1.02
    ylimitmax=max(max(data_calculated_dict['front_y']), max(data_observed_dict[obj2]))*1.02
    xlimitmin=min(min(data_calculated_dict['front_x']), min(data_observed_dict[obj1]))*0.98
    ylimitmin=min(min(data_calculated_dict['front_y']), min(data_observed_dict[obj2]))*0.98

    # Create figure
    p = bh_plotting.figure(
            title="Multiobjective score-score plot", 
            tools="pan,box_zoom,wheel_zoom,reset",
            active_scroll="wheel_zoom", 
            x_axis_label=list(data_observed_dict.keys())[0], 
            y_axis_label=list(data_observed_dict.keys())[1],
            x_range=bh_models.Range1d(xlimitmin,xlimitmax,bounds=(xlimitmin,xlimitmax)),
            y_range=bh_models.Range1d(ylimitmin,ylimitmax,bounds=(ylimitmin,ylimitmax)),
            width_policy = 'max',
            height_policy = 'max',
            )

    # Plot observed data and create Tooltip
    r1 = p.circle(
            list(data_observed_dict.keys())[0],
            list(data_observed_dict.keys())[1],
            radius=0.2,
            source=source_observed,
            legend_label="Observed datapoints",
            fill_alpha=0.4,
            )
    p.add_tools(
            bh_models.HoverTool(
                renderers=[r1],
                tooltips=Tooltips_observed,
                point_policy='snap_to_data',
                line_policy="none",
                )
            )

    # Plot Pareto-front and create Tooltip
    r2 = p.circle(
            list(data_calculated_dict.keys())[0],
            list(data_calculated_dict.keys())[1],
            radius=0.2,
            source=source_calculated,
            color="red",
            legend_label="Estimated Pareto front",
            )
    p.add_tools(
            bh_models.HoverTool(
                renderers=[r2],
                tooltips=Tooltips_recipe,
                point_policy='snap_to_data',
                line_policy="none",
                )
            )
    
    # plot settings
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.toolbar.logo = None

    # Save plot as HTML-file
    
    bh_plotting.reset_output()
    bh_plotting.output_file(filename+".html", title='Multiobjective score-score plot')

    if return_type_bokeh == 'file':
        bh_plotting.save(p)
    if show_browser and return_type_bokeh != 'file':
        raise ValueError("Cannot show the plot if 'return_type_bokeh' is not set to 'file'.")
    elif show_browser:
        bh_io.show(p)
        

    if return_data is True and return_type_bokeh in ['htmlString', 'embed', 'json']:
        raise ValueError("Cannot ruturn data and bokeh object at the same time")
    elif return_data is True and return_type_bokeh == 'file':
        return (
            np.array(optimizer.Xi),
            np.array(optimizer.yi),
            pop,
            front,
            dimensions,
            objective_names,
        )
    elif return_data is True and return_type_bokeh == None:
        return (
            np.array(optimizer.Xi),
            np.array(optimizer.yi),
            pop,
            front,
            dimensions,
            objective_names,
        )
    elif not return_data is True and return_type_bokeh in ['htmlString', 'embed','json']:
        if return_type_bokeh == 'htmlString':
            html = bh_embed.file_html(p, bh_resources.CDN, 'Multiobjective score-score plot')
            return html
        elif return_type_bokeh == 'embed':
            script, div = bh_embed.components(p)
            return (script, div)
        elif return_type_bokeh == 'json':
            json_item = bh_embed.json_item(p)
            return json_item

