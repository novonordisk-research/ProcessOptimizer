from . import Space, Categorical, Integer, Real, Dimension


def normalize_dimensions(dimensions):
    """Create a ``Space`` where all dimensions are normalized to unit range.

    This is particularly useful for Gaussian process based regressors and is
    used internally by ``gp_minimize``.

    Parameters
    ----------
    * `dimensions` [list, shape=(n_dims,)]:
        List of search space dimensions.
        Each search dimension can be defined either as

        - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
          dimensions),
        - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
          dimensions),
        - as a list of categories (for `Categorical` dimensions), or
        - an instance of a `Dimension` object (`Real`, `Integer` or
          `Categorical`).

         NOTE: The upper and lower bounds are inclusive for `Integer`
         dimensions.
    """
    space = Space(dimensions)
    transformed_dimensions = []
    if space.is_categorical:
        # recreate the space and explicitly set transform to "identity"
        # this is a special case for GP based regressors
        for dimension in space:
            transformed_dimensions.append(
                Categorical(
                    dimension.categories,
                    dimension.prior,
                    name=dimension.name,
                    transform="identity",
                )
            )

    else:
        for dimension in space.dimensions:
            if isinstance(dimension, Categorical):
                transformed_dimensions.append(dimension)
            # To make sure that GP operates in the [0, 1] space
            elif isinstance(dimension, Real):
                transformed_dimensions.append(
                    Real(
                        dimension.low,
                        dimension.high,
                        dimension.prior,
                        name=dimension.name,
                        transform="normalize",
                    )
                )
            elif isinstance(dimension, Integer):
                transformed_dimensions.append(
                    Integer(
                        dimension.low,
                        dimension.high,
                        name=dimension.name,
                        transform="normalize",
                    )
                )
            else:
                raise RuntimeError("Unknown dimension type " "(%s)" % type(dimension))

    return Space(transformed_dimensions)
