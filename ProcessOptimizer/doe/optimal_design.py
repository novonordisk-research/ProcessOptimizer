import math
import warnings

import numpy as np
import patsy

from .doe_transform import doe_to_real_space
from ..space import Categorical, Integer, Real, Space

# Defining a number of helper functions for the optimal design of experiments


def hit_and_run(x0, constraint_matrix, bounds, n_samples, thin=1):
    """A basic implementation of the hit and run sampler

    :param x0: The starting value of sampler.
    :param constraint_matrix: A matrix of constraints in the form Ax <= b.
    :param bounds: A vector of bounds in the form Ax <= b.
    :param n_samples: The numbers of samples to return.
    :param thin: The thinning factor. Retain every 'thin' sample
        (e.g. if thin=2, retain every 2nd sample)

    This function is from https://github.com/statease/dexpy
    Copyright 2016 Stat-Ease, Inc.
    License: Apache License, Version 2.0
    License link: https://github.com/statease/dexpy/blob/master/LICENSE
    """
    x = np.copy(x0)
    p = len(x)

    out_samples = np.zeros((n_samples, p))

    for i in range(0, n_samples):
        thin_count = 0

        while thin_count < thin:
            thin_count = thin_count + 1

            random_dir = np.random.normal(0.0, 1.0, p)
            random_dir = random_dir / np.linalg.norm(random_dir)

            denom = constraint_matrix.dot(random_dir)
            intersections = (bounds - constraint_matrix.dot(x)) / denom
            t_low = np.max(intersections[denom < 0])
            t_high = np.min(intersections[denom > 0])

            u = np.random.uniform(0, 1)
            random_distance = t_low + u * (t_high - t_low)
            x_new = x + random_distance * random_dir

        out_samples[i,] = x_new
        x = x_new

    return out_samples


def bootstrap(factor_names, model, run_count):
    """Create a minimal starting design that is non-singular.

    This function is modified from https://github.com/statease/dexpy
    Copyright 2016 Stat-Ease, Inc.
    License: Apache License, Version 2.0
    License link: https://github.com/statease/dexpy/blob/master/LICENSE
    """
    md = patsy.ModelDesc.from_formula(model)
    model_size = len(md.rhs_termlist)
    if run_count == 0:
        run_count = model_size
    if model_size > run_count:
        raise ValueError(
            "Can't build a design of size {} "
            "for a model of rank {}. "
            "Model: '{}'".format(run_count, model_size, model)
        )

    factor_count = len(factor_names)
    x0 = np.zeros(factor_count)
    # add high/low bounds to constraint matrix
    constraint_matrix = np.zeros((factor_count * 2, factor_count))
    bounds = np.zeros(factor_count * 2)
    c = 0
    for f in range(factor_count):
        constraint_matrix[c][f] = -1
        bounds[c] = 1
        c += 1
        constraint_matrix[c][f] = 1
        bounds[c] = 1
        c += 1

    start_points = hit_and_run(x0, constraint_matrix, bounds, run_count)

    d = start_points

    d_dict = {}
    for i in range(0, factor_count):
        d_dict[factor_names[i]] = start_points[:, i]

    X = patsy.dmatrix(model, d_dict, return_type="matrix")

    return (d, X)


def update(XtXi, new_point, old_point):
    """rank-2 update of the variance-covariance matrix

    Equation (6) from Meyer and Nachtsheim :cite:`MeyerNachtsheim1995`.

    This function is from https://github.com/statease/dexpy
    Copyright 2016 Stat-Ease, Inc.
    License: Apache License, Version 2.0
    License link: https://github.com/statease/dexpy/blob/master/LICENSE
    """
    F2 = np.vstack((new_point, old_point))
    F1 = F2.T.copy()
    F1[:, 1] *= -1
    FD = np.dot(F2, XtXi)
    I2x2 = np.identity(2) + np.dot(FD, F1)
    Inverse2x2 = np.linalg.inv(I2x2)
    F2x2FD = np.dot(np.dot(F1, Inverse2x2), FD)
    return XtXi - np.dot(XtXi, F2x2FD)


def expand_point(design_point, code):
    """Converts a point in factor space to conform with the X matrix.

    This function is from https://github.com/statease/dexpy
    Copyright 2016 Stat-Ease, Inc.
    License: Apache License, Version 2.0
    License link: https://github.com/statease/dexpy/blob/master/LICENSE
    """
    return np.array(eval(code, {}, design_point))


def delta(X, XtXi, row, new_point):
    """Calculates the change in D-optimality from exchanging a point.

    This is equation (1) in Meyer and Nachtsheim :cite:`MeyerNachtsheim1995`.

    This function is from https://github.com/statease/dexpy
    Copyright 2016 Stat-Ease, Inc.
    License: Apache License, Version 2.0
    License link: https://github.com/statease/dexpy/blob/master/LICENSE
    """
    old_point = X[row]

    added_variance = np.dot(new_point, np.dot(XtXi, new_point.T))
    removed_variance = np.dot(old_point, np.dot(XtXi, old_point.T))
    covariance = np.dot(new_point, np.dot(XtXi, old_point.T))
    return (
        1
        + (added_variance - removed_variance)
        + (covariance * covariance - added_variance * removed_variance)
    )


def make_model(factor_names, model_order, include_powers=True):
    """Creates patsy formula representing a given model order.

    :param factor_names: The names of the factors in the design.
    :type factor_names: list of str

    :param model_order: The order of the model.
    :type model_order: int

    :param include_powers: Whether to include squared and cubed terms.
    :type include_powers: bool, list of bool
    if True, include squared and cubed terms for all factors
    if include_powers is a list, it must be the same length as factor_names
    if include_powers is a list, include squared and cubed terms for factors
    where include_powers is True

    :return: The patsy formula representing the model.
    :rtype: str

    This function is inspired by similar function in
    https://github.com/statease/dexpy
    Copyright 2016 Stat-Ease, Inc.
    License: Apache License, Version 2.0
    License link: https://github.com/statease/dexpy/blob/master/LICENSE
    """

    if isinstance(include_powers, list) and len(include_powers) != len(
        factor_names
    ):
        raise ValueError(
            "The length of include_powers must be equal to the number of "
            "factors"
        )

    if model_order == 1:
        return "+".join(factor_names)

    elif model_order == 2:
        interaction_model = "({})**2".format("+".join(factor_names))
        if not include_powers:
            return interaction_model
        if isinstance(include_powers, list):
            terms = []
            for factor, include_power in zip(factor_names, include_powers):
                if include_power:
                    terms.append(f"pow({factor}, 2)")
            squared_terms = "+".join(terms)
        else:  # This is if include_powers is True
            squared_terms = "pow({}, 2)".format(",2)+pow(".join(factor_names))
        return f"{interaction_model}+{squared_terms}"

    elif model_order == 3:
        interaction_model = "({})**3".format("+".join(factor_names))
        if not include_powers:
            return interaction_model
        if isinstance(include_powers, list):
            terms = []
            for factor, include_power in zip(factor_names, include_powers):
                if include_power:
                    terms.append(f"pow({factor}, 2)")
                    terms.append(f"pow({factor}, 3)")
            power_terms = "+".join(terms)
            return f"{interaction_model}+{power_terms}"
        else:  # This is if include_powers is True
            squared_terms = "pow({}, 2)".format(",2)+pow(".join(factor_names))
            cubed_terms = "pow({}, 3)".format(",3)+pow(".join(factor_names))
            return "+".join([interaction_model, squared_terms, cubed_terms])

    else:
        raise Warning("Model order not supported")


def initial_guess_cat_vars(init_guess, space):
    """Function that changes initial values for the design points for 
    categorical variables.

    :param init_guess: The initial guess for the design points.
    :type init_guess: np.array

    :param space: The space of the factors.
    :type space: Space object from ProcessOptimizer

    :return: The initial guess for the design points with categorical variables
        changed to -1 and 1.
    :rtype: np.array
    """

    levels = []
    for factor in space.dimensions:
        if isinstance(factor, Categorical):
            if len(factor.categories) != 2:
                raise ValueError(
                    "Only 2 level categorical factors are supported"
                )
            levels.append(len(factor.categories))
        else:
            levels.append(None)

    for i, level in enumerate(levels):
        if level:
            if level == 2:
                init_guess[:, i] = np.where(init_guess[:, i] < 0, -1, 1)
            else:
                raise ValueError(
                    "Only 2 level categorical factors are supported"
                )
    return init_guess


# Main function for optimal design of experiments


def build_optimal_design(factor_names, **kwargs):
    """Builds an optimal design.

    This uses the Coordinate-Exchange algorithm from Meyer and Nachtsheim 1995
    :cite:`MeyerNachtsheim1995`.

    :param factor_names: The names of the factors in the design.
    :type factor_names: list of str

    :Keyword Arguments:
        * **order** (:class:`ModelOrder <dexpy.model.ModelOrder>`) -- \
            Builds a design for this order model. \
            Mutually exclusive with the **model** parameter.
        * **model** (`patsy formula <https://patsy.readthedocs.io>`_) -- \
            Builds a design for this model formula. \
            Mutually exclusive with the **order** parameter.
        * **run_count** (`integer`) -- \
            The number of runs to use in the design. This must be equal\
            to or greater than the rank of the model.
        * **space** (:class:`Space <ProcessOptimizer.space.Space>`) -- \
            The space of the factors. This is used to be more consistent \
            when using categorical variables.

    _______________________________________________________

    This function is adapted from https://github.com/statease/dexpy
    Copyright 2016 Stat-Ease, Inc.
    License: Apache License, Version 2.0
    License link: https://github.com/statease/dexpy/blob/master/LICENSE
    """

    factor_count = len(factor_names)
    model = kwargs.get("model", None)

    include_powers = kwargs.get("include_powers", True)
    space = kwargs.get("space", None)

    if model is None:
        order = kwargs.get("order", 2)
        model = make_model(factor_names, order, include_powers=include_powers)

    run_count = kwargs.get("run_count", 0)

    # first generate a valid starting design
    (design, X) = bootstrap(factor_names, model, run_count)

    if space is not None:
        design = initial_guess_cat_vars(design, space)

    # Enable conversion between design points and X matrix
    functions = []
    for _, subterms in X.design_info.term_codings.items():
        sub_funcs = []
        for subterm in subterms:
            for factor in subterm.factors:
                factor_info = X.design_info.factor_infos[factor]
                eval_code = factor_info.state["eval_code"]
                if eval_code[0] == "I":
                    eval_code = eval_code[1:]
                sub_funcs.append(eval_code)
        if not sub_funcs:
            functions.append("1")  # intercept
        else:
            functions.append("*".join(sub_funcs))

    full_func = "[" + ",".join(functions) + "]"
    code = compile(full_func, "<string>", "eval")

    # set up the algorithm parameters
    steps = 12
    low = -1
    high = 1

    XtXi = np.linalg.inv(np.dot(np.transpose(X), X))
    (_, d_optimality) = np.linalg.slogdet(XtXi)

    design_improved = True
    swaps = 0
    evals = 0
    min_change = 1.0 + np.finfo(float).eps
    while design_improved:
        design_improved = False
        for i in range(0, len(design)):
            design_point = {}
            for ii in range(0, factor_count):
                design_point[factor_names[ii]] = design[i, ii]
            for index_factor, f in enumerate(factor_names):
                original_value = design_point[f]
                original_expanded = X[i]
                best_step = -1
                best_point = []
                best_change = min_change

                # Make some requirements related to have categorical variables can be updated.
                # Maybe that can be made by a function that changes the number of steps
                # If it is a categorical variable, there should only be two steps

                for s in range(0, steps):

                    design_point[f] = low + ((high - low) / (steps - 1)) * s
                    new_point = expand_point(design_point, code)

                    change_in_d = delta(X, XtXi, i, new_point)
                    evals += 1

                    if change_in_d - best_change > np.finfo(float).eps:
                        best_point = new_point
                        best_step = s
                        best_change = change_in_d

                if best_step >= 0:
                    # update X with the best point
                    design_point[f] = (
                        low + ((high - low) / (steps - 1)) * best_step
                    )
                    design[i, index_factor] = design_point[f]
                    XtXi = update(XtXi, best_point, X[i])
                    X[i] = best_point

                    d_optimality -= math.log(best_change)
                    design_improved = True
                    swaps += 1

                else:
                    # restore the original design point value
                    design_point[f] = original_value
                    X[i] = original_expanded

    return design


# Integration with ProcessOptimizer


def sanitize_names_for_patsy(factor_names):
    """
    Sanitize factor names for use in patsy formulas.

    This function replaces spaces and mathematical symbols with underscores.
    It also removes special characters that are not allowed in patsy formulas.

    :param factor_names: The names of the factors in the design.
    :type factor_names: list of str

    :return: The sanitized factor names
    :rtype: list of str
    """

    chars_to_replace_with_underscore = [
        " ",
        "-",
        "+",
        "*",
        "/",
        ":",
        "^",
        "=",
        "~",
    ]
    chars_to_remove = ["$", "(", ")", "[", "]", "{", "}"]

    for i, name in enumerate(factor_names):
        for symbol in chars_to_replace_with_underscore:
            if symbol in name:
                warnings.warn(
                    (
                        "Warning: Factor names should not contain spaces or "
                        "mathematical symbols. Replacing with underscore"
                    )
                )
                factor_names[i] = name.replace(symbol, "_")
                name = factor_names[i]
        for symbol_rm in chars_to_remove:
            if symbol_rm in name:
                factor_names[i] = name.replace(symbol_rm, "")
                name = factor_names[i]

    return factor_names


def model_order_and_include_powers(design_type):
    """
    Determine the order and include_powers of the model based on the design
    type

    :param design_type: The design_type of design to create.
    :type design_type: str
    :options: 'linear', 'screening', 'response', 'optimization'

    :return: The order and include_powers of the model
    :rtype: int, bool
    """

    # Specify options for the design types
    design_types = ["linear", "screening", "response", "optimization"]
    model_orders = [1, 2, 2, 3]
    model_include_powers = [None, False, True, True]

    if design_type is not None and design_type not in design_types:
        raise ValueError(f"design_type must be one of {design_types}")

    # Determine the order and "include powers" of the model
    if design_type is not None:
        design_model_orders = dict(zip(design_types, model_orders))
        order = design_model_orders[design_type]
        include_powers_dict = dict(zip(design_types, model_include_powers))
        include_powers = include_powers_dict[design_type]
    else:
        order = None
        include_powers = None

    return order, include_powers


def generate_replicas_and_sort(
    design_points_real_space, n_replicates, sorting
):
    """
    Generate replicas and sort the design points

    :param design_points_real_space: The design points in real space
    :type design_points_real_space: np.array

    :param n_replicates: The number of replicates to include in the design
    :type n_replicates: postive int

    :param sorting: Whether to sort the design points in real space
    :type sorting: False, or str
    :options: False, "ascending", "randomized", "random_but_group_replicates"

    :return: The design points with replicas and sorted
    :rtype: np.array
    """
    sorting_options = [
        False,
        "ascending",
        "randomized",
        "random_but_group_replicates",
    ]

    if sorting not in sorting_options:
        raise ValueError(f"sorting must be one of {sorting_options}")

    # if sorting is False, just replicate the design points
    if sorting is False:
        design_points_rep_and_sort = np.tile(
            design_points_real_space, (n_replicates, 1)
        )
    # if sorting is "random_but_group_replicates", replicate the design points
    # and group the replicas
    # do this by extending the design points in the first dimension and
    # reshaping
    elif sorting == "random_but_group_replicates":
        design_points_mid_reps = np.tile(
            design_points_real_space, (1, n_replicates)
        )
        design_points_rep_and_sort = np.reshape(
            design_points_mid_reps,
            (
                len(design_points_real_space) * n_replicates,
                len(design_points_real_space[0]),
            ),
        )
    # if sorting is "ascending" or "randomized", replicate the design points
    # first and then sort them
    else:
        design_points_mid_reps = np.tile(
            design_points_real_space, (n_replicates, 1)
        )
        if sorting == "ascending":
            design_points_rep_and_sort = design_points_mid_reps[
                np.lexsort(np.fliplr(design_points_mid_reps).T)
            ]
        elif sorting == "randomized":
            np.random.shuffle(design_points_mid_reps)
            design_points_rep_and_sort = design_points_mid_reps

    return design_points_rep_and_sort


def get_optimal_DOE(
    factor_space,
    budget,
    design_type=None,
    model=None,
    replicates=1,
    sorting=False,
):
    """
    A function that returns the d-optimal design of experiments
    It is non-deterministic and returns a new and perhaps different design
    each time it is called

    Inputs:

    :param factor_space: The space of the factors
    :type factor_space: dict
    Generated from the Space class in the ProcessOptimizer library

    :param budget: The number of runs in the design
    :type budget: int
    Must be at least the number of factors in the model

    :param design_type: The design_type of design to create.
    :type design_type: str
    :options: 'linear', 'screening', 'response', 'optimization'
    Mutually exclusive with the model parameter

    :param model: The model to use for the design. The default is None
    :type model: str in patsy formula format
    Mutually exclusive with the design_type parameter
    Used if you want to have some hand-curated contributions in the model
        e.g., a specific cross interaction between two factors

    :param replicates: The number of replicates to include in the design
    :type replicates: postive int
    Default is 1

    :param sorting: Whether to sort the design points in real space
    :type sorting: False, or str
    Can take False, "ascending", "randomized", "random_but_group_replicates",
    Default is False

    Outputs:

    :return: A design of experiments in real space
    :rtype: np.array

    Example:

    from ProcessOptimizer.space import Integer, Real, Space
    from ProcessOptimizer.doe import get_optimal_DOE
    factor_space = Space(dimensions=[Real(10, 40, name='ul_indicator'),
                                     Integer(20, 100, name='ul_base'),
                                     Integer(20, 100, name='ul_acid'),
                                     ])

    get_optimal_DOE(factor_space, 10, design_type='response')
    """
    # Only relevant if implementing for higher level categorical variables
    # categorical_options = 1

    for factor in factor_space.dimensions:
        if isinstance(factor, Categorical):
            # categorical_options *= len(factor.categories)
            # INITIALLY, only allow 2 level categorical factors
            if len(factor.categories) != 2:
                raise ValueError(
                    "Only 2 level categorical factors are supported"
                )

    # Only relevant if implementing for higher level categorical variables
    # if budget < categorical_options:
    #     raise ValueError(
    #        "The number of runs in the design must be at least the number of "
    #        "possible combinations of categorical variables categories: "
    #        f"{categorical_options}"
    #     )

    # Checking inputs
    # Making sure that factor names are valid for use in patsy
    factor_names_raw = factor_space.names

    factor_names = sanitize_names_for_patsy(factor_names_raw)

    if design_type is not None and model is not None:
        raise ValueError(
            "'design_type' and 'model' are mutually exclusive. "
            "Please choose one or the other"
        )

    order, include_powers = model_order_and_include_powers(design_type)

    # Build the optimal design
    design = build_optimal_design(
        factor_names,
        space=factor_space,
        run_count=budget,
        order=order,
        model=model,
        include_powers=include_powers,
    )

    # Transform the design into real space
    # This needs to be updated to work with categorical variables
    design_points_real_space = doe_to_real_space(design, factor_space)

    # Generate replicas and sort the design points
    design_points_with_reps_and_sort = generate_replicas_and_sort(
        design_points_real_space, replicates, sorting
    )

    return (design_points_with_reps_and_sort, factor_names)
