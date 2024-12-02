import warnings

import numpy as np


def generate_replicas_and_sort(
    design_points_real_space, n_replicates, sorting=False
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

    # Make sure the design points are an array of objects so that we can avoid
    # issues with numpy trying to convert the elements to strings when we
    # categorical variables

    design_points_real_space = np.array(design_points_real_space, dtype=object)

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
        np.random.shuffle(design_points_mid_reps)
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
                        "Factor names should not contain spaces or "
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
