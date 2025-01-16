import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ProcessOptimizer.space import normalize_dimensions


def doe_to_real_space(design, factor_space, corner_points=None):
    """
    Transform the design into real space.

    :param design: The design to transform.
    :type design: np.ndarray

    :param factor_space: The space object used to transform the design.
    :type factor_space: Space

    :param corner_points: The corner points of the design space, given as a
                          list of two lists. The two lists should each have
                          length len(factor_space), and contain the minimum
                          and maximum values for each dimension. If not
                          provided, the design will be transformed under the
                          assumption that min and max values of each dimension
                          in the design space are represented in the design
                          points.
                          Example: np.array([[-1, -1], [1, 1]])
    :type corner_points: np.ndarray or list of list, optional

    :return: The design points in real space.
    :rtype: np.ndarray
    """

    # This ensure that the transformation works no matter how the design is
    # expressed
    # E.g., with values from 0 to 1 or from -1 to 1.
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))

    if corner_points is not None:
        corner_points = np.asarray(corner_points)
        scaler.fit(corner_points)
        transformed_design = scaler.transform(design)
    else:
        transformed_design = scaler.fit_transform(design)

    space_transform = normalize_dimensions(factor_space)
    design_points_real_space = space_transform.inverse_transform(
        np.asarray(transformed_design)
    )
    return design_points_real_space
