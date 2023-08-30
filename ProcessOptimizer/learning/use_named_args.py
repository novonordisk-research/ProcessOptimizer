from functools import wraps

from ..space import Dimension


def use_named_args(dimensions):
    """
    Wrapper / decorator for an objective function that uses named arguments
    to make it compatible with optimizers that use a single list of parameters.

    Your objective function can be defined as being callable using named
    arguments: `func(foo=123, bar=3.0, baz='hello')` for a search-space
    with dimensions named `['foo', 'bar', 'baz']`. But the optimizer
    will only pass a single list `x` of unnamed arguments when calling
    the objective function: `func(x=[123, 3.0, 'hello'])`. This wrapper
    converts your objective function with named arguments into one that
    accepts a list as argument, while doing the conversion automatically.

    The advantage of this is that you don't have to unpack the list of
    arguments `x` yourself, which makes the code easier to read and
    also reduces the risk of bugs if you change the number of dimensions
    or their order in the search-space.

    Example Usage
    -------------
    # Define the search-space dimensions. They must all have names!
    dim1 = Real(name='foo', low=0.0, high=1.0)
    dim2 = Real(name='bar', low=0.0, high=1.0)
    dim3 = Real(name='baz', low=0.0, high=1.0)

    # Gather the search-space dimensions in a list.
    dimensions = [dim1, dim2, dim3]

    # Define the objective function with named arguments
    # and use this function-decorator to specify the search-space dimensions.
    @use_named_args(dimensions=dimensions)
    def my_objective_function(foo, bar, baz):
        return foo ** 2 + bar ** 4 + baz ** 8

    # Now the function is callable from the outside as
    # `my_objective_function(x)` where `x` is a list of unnamed arguments,
    # which then wraps your objective function that is callable as
    # `my_objective_function(foo, bar, baz)`.
    # The conversion from a list `x` to named parameters `foo`, `bar`, `baz`
    # is done automatically.

    # Run the optimizer on the wrapped objective function which is called as
    # `my_objective_function(x)` as expected by `forest_minimize()`.
    result = forest_minimize(func=my_objective_function, dimensions=dimensions,
                             n_calls=20, base_estimator="ET", random_state=4)

    # Print the best-found results.
    print("Best fitness:", result.fun)
    print("Best parameters:", result.x)

    Parameters
    ----------
    * `dimensions` [list(Dimension)]:
        List of `Dimension`-objects for the search-space dimensions.

    Returns
    -------
    * `wrapped_func` [callable]
        Wrapped objective function.
    """

    def decorator(func):
        """
        This uses more advanced Python features to wrap `func` using a
        function-decorator, which are not explained so well in the
        official Python documentation.

        A good video tutorial explaining how this works is found here:
        https://www.youtube.com/watch?v=KlBPCzcQNU8

        Parameters
        ----------
        * `func` [callable]:
            Function to minimize. Should take *named arguments*
            and return the objective value.
        """

        # Ensure all dimensions are correctly typed.
        if not all(isinstance(dim, Dimension) for dim in dimensions):
            # List of the dimensions that are incorrectly typed.
            err_dims = list(
                filter(lambda dim: not isinstance(dim, Dimension), dimensions)
            )

            # Error message.
            msg = """All dimensions must be instances of the Dimension-class,
                  but found: {}"""
            msg = msg.format(err_dims)
            raise ValueError(msg)

        # Ensure all dimensions have names.
        if any(dim.name is None for dim in dimensions):
            # List of the dimensions that have no names.
            err_dims = list(filter(lambda dim: dim.name is None, dimensions))

            # Error message.
            msg = "All dimensions must have names, but found: {}"
            msg = msg.format(err_dims)
            raise ValueError(msg)

        @wraps(func)
        def wrapper(x):
            """
            This is the code that will be executed every time the
            wrapped / decorated `func` is being called.
            It takes `x` as a single list of parameters and
            converts them to named arguments and calls `func` with them.

            Parameters
            ----------
            * `x` [list]:
                A single list of parameters e.g. `[123, 3.0, 'linear']`
                which will be converted to named arguments and passed
                to `func`.

            Returns
            -------
            * `objective_value`
                The objective value returned by `func`.
            """

            # Ensure the number of dimensions match
            # the number of parameters in the list x.
            if len(x) != len(dimensions):
                msg = (
                    "Mismatch in number of search-space dimensions. "
                    "len(dimensions)=={} and len(x)=={}"
                )
                msg = msg.format(len(dimensions), len(x))
                raise ValueError(msg)

            # Create a dict where the keys are the names of the dimensions
            # and the values are taken from the list of parameters x.
            arg_dict = {dim.name: value for dim, value in zip(dimensions, x)}

            # Call the wrapped objective function with the named arguments.
            objective_value = func(**arg_dict)

            return objective_value

        return wrapper

    return decorator
