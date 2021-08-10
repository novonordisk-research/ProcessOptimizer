"""Plotting functions."""
from itertools import count
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy.optimize import OptimizeResult
from scipy.stats.mstats import mquantiles
from scipy.ndimage.filters import gaussian_filter1d
from ProcessOptimizer import expected_minimum, expected_minimum_random_sampling
from .space import Categorical
from .optimizer import Optimizer


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
        dim_labels = [
            "$X_{%i}$" % i if d.name is None else d.name
            for i, d in enumerate(space.dimensions)
        ]
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


def dependence(
    space,
    model,
    i,
    j=None,
    sample_points=None,
    n_samples=250,
    n_points=40,
    x_eval=None,
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

    Returns
    -------
    For 1D partial dependence:

    * `xi`: [np.array]:
        The points at which the partial dependence was evaluated.

    * `yi`: [np.array]:
        The value of the model at each point `xi`.

    For 2D partial dependence:

    * `xi`: [np.array, shape=n_points]:
        The points at which the partial dependence was evaluated.
    * `yi`: [np.array, shape=n_points]:
        The points at which the partial dependence was evaluated.
    * `zi`: [np.array, shape=(n_points, n_points)]:
        The value of the model at each point `(xi, yi)`.

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
        for x_ in xi_transformed:
            rvs_ = np.array(sample_points)  # copy
            # We replace the values in the dimension that we want to keep fixed
            rvs_[:, dim_locs[i] : dim_locs[i + 1]] = x_
            # In case of `x_eval=None` rvs conists of random samples.
            # Calculating the mean of these samples is how partial dependence
            # is implemented.
            yi.append(np.mean(model.predict(rvs_)))

        return xi, yi

    else:
        xi, xi_transformed = _evenly_sample(space.dimensions[j], n_points)
        yi, yi_transformed = _evenly_sample(space.dimensions[i], n_points)

        zi = []
        for x_ in xi_transformed:
            row = []
            for y_ in yi_transformed:
                rvs_ = np.array(sample_points)  # copy
                rvs_[:, dim_locs[j] : dim_locs[j + 1]] = x_
                rvs_[:, dim_locs[i] : dim_locs[i + 1]] = y_
                row.append(np.mean(model.predict(rvs_)))
            zi.append(row)

        return xi, yi, np.array(zi).T


def plot_objective(
    result,
    levels=10,
    n_points=40,
    n_samples=250,
    size=2,
    zscale="linear",
    dimensions=None,
    usepartialdependence=True,
    pars="result",
    expected_minimum_samples=None,
    title=None,
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
    * `usepartialdependence` [bool, default=false] Wether to use partial
        dependence or not when calculating dependence. If false plot_objective
        will parse values to the dependence function,
        defined by the pars argument

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

    * `expected_minimum_samples` [float, default = None] Determines how many
    points should be evaluated to find the minimum when using
    'expected_minimum' or 'expected_minimum_random'

    * `title` [str, default=None]
        String to use as title of the figure


    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    # Here we define the values for which to plot the red dot (2d plot) and
    # the red dotted line (1d plot). These same values will be used for
    # evaluating the plots when calculating dependence. (Unless partial
    # dependence is to be used instead).
    space = result.space
    if isinstance(pars, str):
        if pars == "result":
            # Using the best observed result
            x_vals = result.x
        elif pars == "expected_minimum":
            if result.space.is_partly_categorical:
                # space is also categorical
                raise ValueError(
                    text="expected_minimum does not support any \
                categorical values"
                )
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

    if zscale == "log":
        locator = LogLocator()
    elif zscale == "linear":
        locator = None
    else:
        raise ValueError(
            "Valid values for zscale are 'linear' and 'log',"
            " not '%s'." % zscale
        )

    fig, ax = plt.subplots(
        space.n_dims,
        space.n_dims,
        figsize=(size * space.n_dims, size * space.n_dims),
    )

    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.1, wspace=0.1
    )

    if title is not None:
        fig.suptitle(title)

    val_min_1d = float("inf")
    val_max_1d = -float("inf")
    val_min_2d = float("inf")
    val_max_2d = -float("inf")

    plots_data = []

    for i in range(space.n_dims):
        row = []
        for j in range(space.n_dims):

            if j > i:
                # We only plot the lower left half of the grid,
                # to avoid duplicates.
                break

            elif i == j:
                xi, yi = dependence(
                    space,
                    result.models[-1],
                    i,
                    j=None,
                    sample_points=rvs_transformed,
                    n_points=n_points,
                    x_eval=x_eval,
                )
                row.append({"xi": xi, "yi": yi})

                if np.min(yi) < val_min_1d:
                    val_min_1d = np.min(yi)
                if np.max(yi) > val_max_1d:
                    val_max_1d = np.max(yi)

            # lower triangle
            else:
                xi, yi, zi = dependence(
                    space,
                    result.models[-1],
                    i,
                    j,
                    rvs_transformed,
                    n_points,
                    x_eval=x_eval,
                )
                # print('filling with i, j = ' + str(i) + str(j))
                row.append({"xi": xi, "yi": yi, "zi": zi})

                if np.min(zi) < val_min_2d:
                    val_min_2d = np.min(zi)
                if np.max(zi) > val_max_2d:
                    val_max_2d = np.max(zi)

        plots_data.append(row)

    for i in range(space.n_dims):
        for j in range(space.n_dims):

            if j > i:
                # We only plot the lower left half of the grid,
                # to avoid duplicates.
                break

            elif i == j:

                xi = plots_data[i][j]["xi"]
                yi = plots_data[i][j]["yi"]

                ax[i, i].plot(xi, yi)
                ax[i, i].set_ylim(val_min_1d, val_max_1d)
                ax[i, i].axvline(minimum[i], linestyle="--", color="r", lw=1)

            # lower triangle
            elif i > j:

                xi = plots_data[i][j]["xi"]
                yi = plots_data[i][j]["yi"]
                zi = plots_data[i][j]["zi"]

                ax[i, j].contourf(
                    xi,
                    yi,
                    zi,
                    levels,
                    locator=locator,
                    cmap="viridis_r",
                    vmin=val_min_2d,
                    vmax=val_max_2d,
                )
                ax[i, j].scatter(
                    samples[:, j], samples[:, i], c="darkorange", s=10, lw=0.0
                )
                ax[i, j].scatter(minimum[j], minimum[i], c=["r"], s=20, lw=0.0)

                if [i, j] == [1, 0]:
                    import matplotlib as mpl

                    norm = mpl.colors.Normalize(
                        vmin=val_min_2d, vmax=val_max_2d
                    )
                    cb = ax[0][-1].figure.colorbar(
                        mpl.cm.ScalarMappable(norm=norm, cmap="viridis_r"),
                        ax=ax[0][-1],
                    )
                    cb.ax.locator_params(nbins=8)

    if usepartialdependence:
        ylabel = "Partial dependence"
    else:
        ylabel = "Dependence"
    return _format_scatter_plot_axes(
        ax, space, ylabel=ylabel, dim_labels=dimensions
    )


def plot_objectives(results, titles=None):
    """Pairwise dependence plots of each of the objective functions.
    Parameters
    ----------
    * `results` [list of `OptimizeResult`]
        The list of results for which to create the objective plots.

    * `titles` [list of str, default=None]
        The list of strings of the names of the objectives used as titles in
        the figures

    """

    if titles is None:
        for result in results:
            plot_objective(result, title=None)
        return
    else:
        for k in range(len(results)):
            plot_objective(results[k], title=titles[k])
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
                )
                ax[i, j].scatter(minimum[j], minimum[i], c=["r"], s=20, lw=0.0)

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
    return str(dimension.categories[int(x)])


def plot_expected_minimum_convergence(
    result, figsize=(15, 15), random_state=None, sigma=0.5
):
    """
    A function to perform a retrospective analysis of all the data points by
    building successive models and predicting the mean of the functional value
    of the surogate model in the expected minimum together with a measure for
    the variability in the suggested mean value. This code "replays" the
    entire optimization, hence it builds quiet many models and can, thus, seem
    slow. TODO: Consider allowing user to subselect in data, to e.g. perform
    the analysis using every n-th datapoint, or only performing the analysis
    for the last n datapoints.
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
        will be named "1_1", "x_2"...
        
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
        values = ["%.3f" % number for number in exp_params]
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
        dimensions = [
            "$X_{%i}$" % i if d.name is None else d.name
            for i, d in enumerate(optimizer.space.dimensions)
        ]

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
