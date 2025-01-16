import math

import numpy as np
import patsy
import scipy

from ProcessOptimizer.space import Categorical
from ProcessOptimizer.utils.get_rng import get_random_generator

from .doe_transform import doe_to_real_space
from .doe_utils import (generate_replicas_and_sort, round_design_point_values,
                        sanitize_names_for_patsy)

# Defining a number of helper functions for the optimal design of experiments


def hit_and_run(x0, constraint_matrix, bounds, n_samples, thin=1, seed=None):
    """A basic implementation of the hit and run sampler

    :param x0: The starting value of sampler.
    :param constraint_matrix: A matrix of constraints in the form Ax <= b.
    :param bounds: A vector of bounds in the form Ax <= b.
    :param n_samples: The numbers of samples to return.
    :param thin: The thinning factor. Retain every 'thin' sample
        (e.g. if thin=2, retain every 2nd sample)
    :param seed: possibility to specify a seed for random generator

    This function is modified from
    https://github.com/statease/dexpy -  Version 0.12
    Copyright 2016 Stat-Ease, Inc.
    License: Apache License, Version 2.0
    License link: https://github.com/statease/dexpy/blob/master/LICENSE
    """
    x = np.copy(x0)
    p = len(x)

    rng = get_random_generator(seed)

    out_samples = np.zeros((n_samples, p))

    for i in range(0, n_samples):
        thin_count = 0

        while thin_count < thin:
            thin_count = thin_count + 1

            random_dir = rng.normal(0.0, 1.0, p)
            random_dir = random_dir / np.linalg.norm(random_dir)

            denom = constraint_matrix.dot(random_dir)
            intersections = (bounds - constraint_matrix.dot(x)) / denom
            t_low = np.max(intersections[denom < 0])
            t_high = np.min(intersections[denom > 0])

            u = rng.uniform(0, 1)
            random_distance = t_low + u * (t_high - t_low)
            x_new = x + random_distance * random_dir

        out_samples[i,] = x_new
        x = x_new

    return out_samples


def bootstrap(factor_names, model, n_exp, **kwargs):
    """Create a minimal starting design that is non-singular.

    This function is adapted from
    https://github.com/statease/dexpy -  Version 0.12
    Copyright 2016 Stat-Ease, Inc.
    License: Apache License, Version 2.0
    License link: https://github.com/statease/dexpy/blob/master/LICENSE
    """
    md = patsy.ModelDesc.from_formula(model)
    model_size = len(md.rhs_termlist)
    if n_exp == 'Min':
        n_exp = model_size
    if model_size > n_exp:
        raise ValueError(
            "Can't build a design of size {} "
            "for a model of rank {}. "
            "Model: '{}'".format(n_exp, model_size, model)
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

    start_points = hit_and_run(x0, constraint_matrix, bounds, n_exp, **kwargs)

    d = start_points

    d_dict = {}
    for i in range(0, factor_count):
        d_dict[factor_names[i]] = start_points[:, i]

    X = patsy.dmatrix(model, d_dict, return_type="matrix")

    return (d, X)


def update(XtXi, new_point, old_point):
    """rank-2 update of the variance-covariance matrix

    Equation (6) from Meyer and Nachtsheim :cite:`MeyerNachtsheim1995`.

    This function is adapted from
    https://github.com/statease/dexpy -  Version 0.12
    Copyright 2016 Stat-Ease, Inc.
    License: Apache License, Version 2.0
    License link: https://github.com/statease/dexpy/blob/master/LICENSE
    """
    F2 = np.vstack((new_point, old_point))
    F1 = F2.T.copy()
    F1[:, 1] *= -1
    FD = np.dot(F2, XtXi)
    I2x2 = np.identity(2) + np.dot(FD, F1)
    Inverse2x2 = scipy.linalg.inv(I2x2)
    F2x2FD = np.dot(np.dot(F1, Inverse2x2), FD)
    return XtXi - np.dot(XtXi, F2x2FD)


def expand_point(design_point, code):
    """Converts a point in factor space to conform with the X matrix.

    This function is from
    https://github.com/statease/dexpy -  Version 0.12
    Copyright 2016 Stat-Ease, Inc.
    License: Apache License, Version 2.0
    License link: https://github.com/statease/dexpy/blob/master/LICENSE
    """
    return np.array(eval(code, {}, design_point))


def delta(X, XtXi, row, new_point):
    """Calculates the change in D-optimality from exchanging a point.

    This is equation (1) in Meyer and Nachtsheim :cite:`MeyerNachtsheim1995`.

    This function is from https://github.com/statease/dexpy -  Version 0.12
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
    if True, include higher order terms (squared if `model_order` is 2,
        squared and cubed if it is 3) for all factors
    if include_powers is a list, it must be the same length as factor_names
    if include_powers is a list, include squared and cubed terms for factors
    where include_powers is True

    :return: The patsy formula representing the model.
    :rtype: str

    This function is inspired by similar function in
    https://github.com/statease/dexpy -  Version 0.12
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
        raise Warning(f"Model order {model_order} not supported")


def initial_values_cat_vars(init_guess, space):
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


def conversion_design_Xmatrix(X):
    """Function that enables conversion between design points and X matrix.

    :param X: The X matrix.
    :type X: patsy dmatrix

    :return: The code that enables conversion between design points and X
    matrix.
    :rtype: code object

    This function is adapted from
    https://github.com/statease/dexpy -  Version 0.12
    Copyright 2016 Stat-Ease, Inc.
    License: Apache License, Version 2.0
    License link: https://github.com/statease/dexpy/blob/master/LICENSE
    """

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
    return code


def optimize_design(X, design, factor_names, code, **kwargs):
    """
    Optimize a design using the Coordinate-Exchange algorithm from Meyer and
    Nachtsheim 1995 :cite:`MeyerNachtsheim1995`.

    :param X: The X matrix.
    :type X: patsty dmatrix

    :param design: The design points.
    :type design: np.array

    :param factor_names: The names of the factors in the design.
    :type factor_names: list of str

    :param code: The code that enables conversion between design points and X
        matrix.
    :type code: code object

    :Keyword Arguments:
        * **high** (`float`) -- The high value for the design points. The
            default is 1.
        * **low** (`float`) -- The low value for the design points. The
            default is -1.
        * **res** (`integer`) -- The resolution of the design. This is the
            sampling resolution used when sampling the design space. The
            higher the resolution, the more accurate the design will be. The
            default is 11.
        * **space** (:class:`Space <ProcessOptimizer.space.Space>`) -- The
            space of the factors. This is needed when using categorical
            variables in a good way.

    :return: The optimized design.
    :rtype: np.array


    This function is adapted from
    https://github.com/statease/dexpy -  Version 0.12
    Copyright 2016 Stat-Ease, Inc.
    License: Apache License, Version 2.0
    License link: https://github.com/statease/dexpy/blob/master/LICENSE
    """

    high = kwargs.get("high", 1)
    low = kwargs.get("low", -1)
    steps0 = kwargs.get("res", 11)
    space = kwargs.get("space", None)

    min_change = 1.0 + np.finfo(float).eps

    XtXi = scipy.linalg.inv(np.dot(np.transpose(X), X))
    (_, d_optimality) = np.linalg.slogdet(XtXi)

    design_improved = True  # Specified to initiate the while loop

    while design_improved:
        design_improved = False
        for i in range(0, len(design)):
            design_point = {}
            for ii in range(0, len(factor_names)):
                design_point[factor_names[ii]] = design[i, ii]
            # This is probably where there should be a change if code should
            # be able to handle categorical variables with more than 2 levels
            for index_factor, f in enumerate(factor_names):
                original_value = design_point[f]
                original_expanded = X[i]
                best_step = -1
                best_point = []
                best_change = min_change
                # If it is a categorical variable, there should only be two
                # steps. This is to ensure that the design points are always
                # -1 and 1 in the categorical var. Only 2 level categorical
                # vars implemented
                steps = steps0
                if space is not None:
                    for factor in space.dimensions:
                        if factor.name == f and isinstance(
                            factor, Categorical
                        ):
                            steps = len(factor.categories)
                # Brute force sampling
                for s in range(0, steps):

                    design_point[f] = low + ((high - low) / (steps - 1)) * s
                    new_point = expand_point(design_point, code)

                    change_in_d = delta(X, XtXi, i, new_point)

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

                else:
                    # restore the original design point value
                    design_point[f] = original_value
                    X[i] = original_expanded

    return design


# Main function for optimal design of experiments
def build_optimal_design(factor_names, **kwargs):
    """Builds an optimal design.

    This uses the Coordinate-Exchange algorithm from Meyer and Nachtsheim 1995
    :cite:`MeyerNachtsheim1995`.

    :param factor_names: The names of the factors in the design.
    :type factor_names: list of str

    :Keyword Arguments:
        * **order** (`integer`) --
            Builds a design for this order model.
            Mutually exclusive with the **model** parameter.
        * **model** (`patsy formula <https://patsy.readthedocs.io>`_) --
            Builds a design for this model formula.
            Mutually exclusive with the **order** parameter.
        * **n_exp** (`integer`) --
            The number of runs to use in the design. This must be equal
            to or greater than the rank of the model.
        * **space** (:class:`Space <ProcessOptimizer.space.Space>`) --
            The space of the factors. This is used to be more consistent
            when using categorical variables.
        * **res** (`integer`) --
            The resolution of the design. This is the sampling resolution used
            when sampling the design space. The higher the resolution, the
            more accurate the design will be. The default is 11.
        * **seed** (`integer`) --
            The seed to use for the random number generator. This is useful if
            you want to reproduce the same design multiple times.

    _______________________________________________________

    This function is adapted from
    https://github.com/statease/dexpy -  Version 0.12
    Copyright 2016 Stat-Ease, Inc.
    License: Apache License, Version 2.0
    License link: https://github.com/statease/dexpy/blob/master/LICENSE
    """

    model = kwargs.get("model", None)
    include_powers = kwargs.get("include_powers", True)
    space = kwargs.get("space", None)
    res = kwargs.get("res", 11)

    if model is None:
        order = kwargs.get("order", 2)
        model = make_model(factor_names, order, include_powers=include_powers)

    n_exp = kwargs.get("n_exp", 'Min')
    nseed = kwargs.get("seed", None)

    # first generate a valid starting design
    (design, X) = bootstrap(factor_names, model, n_exp, seed=nseed)

    # The numbers in the design representing categorical factors should
    # initially be exactly at one of the levels

    if space is not None:
        design = initial_values_cat_vars(design, space)

    # Make sure that design point values are rounded to the resolution
    design = round_design_point_values(design, res)

    # Enable conversion between design points and X matrix
    code = conversion_design_Xmatrix(X)

    opt_design = optimize_design(
        X, design, factor_names, code, res=res, space=space
    )

    return opt_design


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
    design_types = {"linear": (1, None),
                    "screening": (2, False),
                    "response": (2, True),
                    "optimization": (3, True),
                    None: (None, None)}

    # Key is design type, value[0] is order, value[1] is include powers

    if design_type not in design_types:
        raise ValueError(f"design_type must be one of {design_types.keys()}")

    order = design_types[design_type][0]
    include_powers = design_types[design_type][1]

    return order, include_powers


def get_cat_var_levels(factor_space):
    """
    Determine the levels of the categorical variables in the factor space.
    Gives a list of the number of levels for each factor in the factor space.
    List element will be None if the factor is not categorical

    :param factor_space: The space of the factors
    :type factor_space: ProcessOptimizer.space.Space

    :return: The levels of the categorical variables in the factor space
    :rtype: list of int or None
    """

    var_is_cat = []

    for factor in factor_space.dimensions:
        if isinstance(factor, Categorical):
            # categorical_options *= len(factor.categories)
            # INITIALLY, only allow 2 level categorical factors
            var_is_cat.append(len(factor.categories))
            if len(factor.categories) != 2:
                raise ValueError(
                    "Only 2 level categorical factors are supported"
                )
        else:
            var_is_cat.append(None)
    return var_is_cat


def include_powers_to_list(include_powers, cat_var_levels):
    """
    Convert the include_powers parameter to a list if True and there are
    categorical variables

    :param include_powers: Whether to include squared and cubed terms
    :type include_powers: bool, list of bool

    :param cat_var_levels: The levels of the categorical variables in the
        factor space
    :type cat_var_levels: list of int or None

    :return: The include_powers parameter as a list
    :rtype: list of bool or bool
    """

    if include_powers is True and any(level for level in cat_var_levels):
        include_powers = [True] * len(cat_var_levels)
        for i, level in enumerate(cat_var_levels):
            if level:
                include_powers[i] = False

    return include_powers


def get_optimal_DOE(
    factor_space,
    budget='Min',
    design_type=None,
    model=None,
    replicates=1,
    sorting=False,
    res=11,
    seed=None,
    **kwargs,
):
    """
    A function that returns the d-optimal design of experiments
    It is non-deterministic and returns a new and perhaps different design
    each time it is called

    Inputs:

    :param factor_space: The space of the factors
    :type factor_space: dict
    Generated from the Space class

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

    :param res: The resolution of the design sampling. The default is 11.
    :type res: int

    :param seed: The seed to use for randomization
    :type seed: int, None or True (True=1)

    Outputs:

    :return: A design of experiments in real space and the factor names
    :rtype: (np.ndarray, list of str)

    Example:

    from ProcessOptimizer.space import Integer, Real, Space
    from ProcessOptimizer.doe import get_optimal_DOE
    factor_space = Space(dimensions=[Integer(10, 40, name='int_var1'),
                                     Integer(-40, 520, name='int_var2'),
                                     Real(0.4, 117.7, name='real_var1'),
                                     Categorical(['A', 'B'], name='cat_var1'),
                                    ])

    get_optimal_DOE(factor_space, 14, design_type='response')
    """

    # Check that "res" is an int and at least 2
    if not isinstance(res, int) or res < 2:
        raise ValueError("'res' must be an integer and at least 2")

    # Check if the factor space has any categorical variables
    cat_var_levels = get_cat_var_levels(factor_space)

    # Checking inputs
    # Making sure that factor names are valid for use in patsy
    factor_names_raw = factor_space.names
    factor_names = sanitize_names_for_patsy(factor_names_raw)

    if design_type is None and model is None:
        design_type = 'screening'

    if design_type is not None and model is not None:
        raise ValueError(
            "'design_type' and 'model' are mutually exclusive. "
            "Please choose one or the other"
        )

    order, include_powers = model_order_and_include_powers(design_type)

    include_powers = include_powers_to_list(include_powers, cat_var_levels)

    # Build the optimal design
    design = build_optimal_design(
        factor_names,
        space=factor_space,
        n_exp=budget,
        order=order,
        model=model,
        include_powers=include_powers,
        res=res,
        seed=seed,
        **kwargs,
    )

    # Transform the design into real space
    # This needs to be updated to work with categorical variables with
    # more than 2 levels
    corner_neg = [-1] * len(factor_names)
    corner_pos = [1] * len(factor_names)
    corner_points_optimal = np.array([corner_neg, corner_pos])

    design_points_real_space = doe_to_real_space(
        design, factor_space, corner_points=corner_points_optimal
    )

    # Generate replicas and sort the design points
    design_points_with_reps_and_sort = generate_replicas_and_sort(
        design_points_real_space, replicates, sorting=sorting, seed=seed
    )

    return (design_points_with_reps_and_sort, factor_names)
