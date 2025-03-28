import warnings

import numpy as np

from ProcessOptimizer.utils.get_rng import get_random_generator


def generate_replicas_and_sort(
    design_points_real_space, n_replicates, sorting=False, seed=None
):
    """
    Generate replicas and sort the design points.

    :param design_points_real_space: The design points in real space
    :type design_points_real_space: np.array

    :param n_replicates: The number of replicates to include in the design
    :type n_replicates: postive int

    :param sorting: Whether to sort the design points in real space
    :type sorting: False, or str
    :options: False, "ascending", "randomized", "random_but_group_replicates"

    :param seed: The seed to use for randomization
    :type seed: int, None or True (True=1)

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

    rng = get_random_generator(seed)

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
    # Do this by extending the design points in the first dimension and
    # reshaping
    elif sorting == "random_but_group_replicates":
        design_points_mid_reps = np.tile(
            design_points_real_space, (1, n_replicates)
        )
        rng.shuffle(design_points_mid_reps)
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
            rng.shuffle(design_points_mid_reps)
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

    sanitized_factor_names = factor_names[:]
    for i, name in enumerate(sanitized_factor_names):
        for symbol in chars_to_replace_with_underscore:
            if symbol in name:
                warnings.warn(
                    (
                        "Factor names should not contain spaces or "
                        "mathematical symbols. Replacing with underscore"
                    )
                )
                sanitized_factor_names[i] = name.replace(symbol, "_")
                name = sanitized_factor_names[i]
        for symbol_rm in chars_to_remove:
            if symbol_rm in name:
                sanitized_factor_names[i] = name.replace(symbol_rm, "")
                name = sanitized_factor_names[i]
    if len(sanitized_factor_names) != len(set(sanitized_factor_names)):
        raise ValueError(
            "Duplicate factor names found after sanitation."
            " Factor names must be unique."
        )
    return sanitized_factor_names


def round_design_point_values(design_points, res):
    """
    Round the design points to the resolution specified.

    :param design_points: The design points to round
    :type design_points: np.array

    :param res: The resolution to round the design points to
    Specifies the number of bins.
    :type res: float

    :return: The rounded design points
    :rtype: np.array

    If the resolution is 3, the design points will be rounded to -1, 0, or 1.

    """

    points_res_scaled = (design_points + 1) / 2 * (res - 1)
    rounded_points_res_scaled = np.round(points_res_scaled)
    rounded_points = rounded_points_res_scaled / (res - 1) * 2 - 1
    return rounded_points
