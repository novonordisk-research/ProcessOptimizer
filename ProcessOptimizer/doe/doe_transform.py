import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ..space import normalize_dimensions


def doe_to_real_space(design, factor_space, corner_points=None):
    """
    Transform the design into real space

    Parameters:
    -----------
    design: np.array
        The design to transform.
    space: Space object from ProcessOptimizer
        The space object used to transform the design.
    corner_points: np.array or 2D list, optional
        The corner points of the design space. If not provided, the design
        will be transformed as if containing the corner points.
        Example: np.array([[0, 0], [1, 1]])

    Returns:
    --------
    design_points_real_space: np.array
        The design points in real space.
    """

    # This ensure that the transformation works no matter how the design is
    # expressed
    # E.g., with values from 0 to 1 or from -1 to 1.
    # This also means that you cannot effectively make a circumscribed central
    # composite design
    # It will be transformed into an inscribed central composite design
    scaler = MinMaxScaler(feature_range=(0., 1.))
    if corner_points is not None:
        corner_points = np.asarray(corner_points)
        scaler.fit_transform(corner_points)
        transformed_design = scaler.transform(design)
    else:
        transformed_design = scaler.fit_transform(design)

    space_transform = normalize_dimensions(factor_space)
    design_points_real_space = space_transform.inverse_transform(
        np.asarray(transformed_design)
    )
    return design_points_real_space
