from abc import ABC, abstractmethod
from typing import Iterable, List, Union

import numbers
import numpy as np
import yaml


from .transformers import CategoricalEncoder
from .transformers import Normalize
from .transformers import Identity
from .transformers import Log10
from .transformers import Pipeline
from ..utils import get_random_generator

# helper class to be able to print [1, ..., 4] instead of [1, '...', 4]


class _Ellipsis:
    def __repr__(self):
        return "..."


def space_factory(input: Union["Space", List]) -> "Space":
    """Transforms a list of dimension definitions into a Space

    If the input is already a Space, that is returned.

    Parameters:
    * `input` (List, Space): The list of Dimension definitions. For more details,
        see documentation for check_dimensions.

    Returns:
    * `space`: The resulting Space.
    """
    if isinstance(input, Space):
        return input
    else:
        return Space(input)


def check_dimension(dimension, transform=None):
    """Turn a provided dimension description into a dimension object.

    Checks that the provided dimension falls into one of the
    supported types. For a list of supported types, look at
    the documentation of ``dimension`` below.

    If ``dimension`` is already a ``Dimension`` instance, return it.

    Parameters
    ----------
    * `dimension`:
        Search space Dimension.
        Each search dimension can be defined either as

        - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
          dimensions),
        - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
          dimensions),
        - as a list of categories (for `Categorical` dimensions), or
        - an instance of a `Dimension` object (`Real`, `Integer` or
          `Categorical`).

    * `transform` ["identity", "normalize", "onehot" optional]:
        - For `Categorical` dimensions, the following transformations are
          supported.

          - "onehot" (default) one-hot transformation of the original space.
          - "identity" same as the original space.

        - For `Real` and `Integer` dimensions, the following transformations
          are supported.

          - "identity", (default) the transformed space is the same as the
            original space.
          - "normalize", the transformed space is scaled to be between 0 and 1.

    Returns
    -------
    * `dimension`:
        Dimension instance.
    """
    if isinstance(dimension, Dimension):
        return dimension

    if not isinstance(dimension, (list, tuple, np.ndarray)):
        raise ValueError("Dimension has to be a list or tuple.")

    # A `Dimension` described by a single value is assumed to be
    # a `Categorical` dimension. This can be used in `BayesSearchCV`
    # to define subspaces that fix one value, e.g. to choose the
    # model type, see "sklearn-gridsearchcv-replacement.ipynb"
    # for examples.
    if len(dimension) == 1:
        return Categorical(dimension, transform=transform)

    if len(dimension) == 2:
        if any(
            [isinstance(d, (str, bool)) or isinstance(d, np.bool_) for d in dimension]
        ):
            return Categorical(dimension, transform=transform)
        elif all([isinstance(dim, numbers.Integral) for dim in dimension]):
            return Integer(*dimension, transform=transform)
        elif any([isinstance(dim, numbers.Real) for dim in dimension]):
            return Real(*dimension, transform=transform)
        else:
            raise ValueError(
                "Invalid dimension {}. Read the documentation for"
                " supported types.".format(dimension)
            )

    if len(dimension) == 3:
        if (any([isinstance(dim, (float, int)) for dim in dimension[:2]])
            and dimension[2] in ["uniform", "log-uniform"]):
            return Real(*dimension, transform=transform)
        else:
            return Categorical(dimension, transform=transform)

    if len(dimension) > 3:
        return Categorical(dimension, transform=transform)

    raise ValueError(
        "Invalid dimension {}. Read the documentation for "
        "supported types.".format(dimension)
    )


class Dimension(ABC):
    """Base class for search space dimensions."""

    prior = None

    def transform(self, X):
        """Transform samples form the original space to a warped space."""
        return self.transformer.transform(X)

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the
        original space.
        """
        return self.transformer.inverse_transform(Xt)

    @property
    def size(self):
        return 1

    @property
    def transformed_size(self):
        return 1

    @property
    def bounds(self):
        raise NotImplementedError

    @property
    def transformed_bounds(self):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if isinstance(value, str) or value is None:
            self._name = value
        else:
            raise ValueError("Dimension's name must be either string or None.")

    def sample(
        self, points: Union[float, Iterable[float]], allow_duplicates: bool = True
    ) -> np.ndarray:
        """Draw points from the dimension.



        Parameters
        ----------
        * `points` [float or list[float]]:
            A single point or a list of points to sample. All must be between 0 and 1.

        * `allow_duplicates` [bool, default=True]:
            If True, the output will have the same size as `points`. If False, each
            point in the output will be unique. This means that the output can be
            shorter than `points`.
        """
        if isinstance(points, (int, float)):  # If a single point is given, convert it
            # to a list.
            points = [points]
        if any([point < 0 or point > 1 for point in points]):
            raise ValueError("Sample points must be between 0 and 1.")
        sampled_points = self._sample(points)
        if not allow_duplicates:
            # np.unique sorts the inputs, which we do not want, so we have to reinvent
            # the wheel.
            seen = set()
            unique_points = []
            for point in sampled_points:
                if point not in seen:
                    unique_points.append(point)
                    seen.add(point)
            sampled_points = unique_points
        return np.array(sampled_points)

    @abstractmethod
    def _sample(self, points: Iterable[float]) -> np.array:
        """A reasonable mapping from the interval [0, 1] to the dimension, whatever that
        may mean. For example, for an integer dimension, the sampling should be give a
        higher integer for a higher value of the point, and each integer should be
        mapped to from an equally large interval.

        The mapping should be monotonic, but it does not have to be strictly monotonic.

        The mapping should respect (informative) priors. For example, if the prior is
        log-uniform, the mapping should be logarithmic, so that the interval that maps
        to [0.1, 1] is the same size as the interval that maps to [1, 10].
        """
        pass


class Real(Dimension):
    def __init__(self, low, high, prior="uniform", transform=None, name=None):
        """Search space dimension that can take on any real value.

        Parameters
        ----------
        * `low` [float]:
            Lower bound (inclusive).

        * `high` [float]:
            Upper bound (inclusive).

        * `prior` ["uniform" or "log-uniform", default="uniform"]:
            Distribution to use when sampling random points for this dimension.
            - If `"uniform"`, points are sampled uniformly between the lower
              and upper bounds.
            - If `"log-uniform"`, points are sampled uniformly between
              `log10(lower)` and `log10(upper)`.`

        * `transform` ["identity", "normalize", optional]:
            The following transformations are supported.

            - "identity", (default) the transformed space is the same as the
              original space.
            - "normalize", the transformed space is scaled to be between
              0 and 1.

        * `name` [str or None]:
            Name associated with the dimension, e.g., "learning rate".
        """
        if high <= low:
            raise ValueError(
                "the lower bound {} has to be less than the"
                " upper bound {}".format(low, high)
            )
        self.low = low
        self.high = high
        self.prior = prior
        self.name = name

        if transform is None:
            transform = "identity"

        self.transform_ = transform

        if self.transform_ not in ["normalize", "identity"]:
            raise ValueError(
                "transform should be 'normalize' or 'identity'"
                " got {}".format(self.transform_)
            )

        if self.transform_ == "normalize":
            # set upper bound to next float after 1. to make the numbers
            # inclusive of upper edge
            if self.prior == "uniform":
                self.transformer = Pipeline([Identity(), Normalize(low, high)])
            else:
                self.transformer = Pipeline(
                    [Log10(), Normalize(np.log10(low), np.log10(high))]
                )
        else:
            if self.prior == "uniform":
                self.transformer = Identity()
            else:
                self.transformer = Log10()

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and np.allclose([self.low], [other.low])
            and np.allclose([self.high], [other.high])
            and self.prior == other.prior
            and self.transform_ == other.transform_
        )

    def __repr__(self):
        return "Real(low={}, high={}, prior='{}', transform='{}')".format(
            self.low, self.high, self.prior, self.transform_
        )

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the
        orignal space.
        """
        return np.clip(
            super(Real, self).inverse_transform(Xt).astype(float), self.low, self.high
        )

    @property
    def bounds(self):
        return (self.low, self.high)

    def __contains__(self, point):
        return self.low <= point <= self.high

    @property
    def transformed_bounds(self):
        if self.transform_ == "normalize":
            return 0.0, 1.0
        else:
            if self.prior == "uniform":
                return self.low, self.high
            else:
                return np.log10(self.low), np.log10(self.high)

    def distance(self, a, b):
        """Compute distance between point `a` and `b`.

        Parameters
        ----------
        * `a` [float]
            First point.

        * `b` [float]
            Second point.
        """
        if not (a in self and b in self):
            raise RuntimeError(
                "Can only compute distance for values within "
                "the space, not %s and %s." % (a, b)
            )
        return abs(a - b)

    def _sample(self, point_list: Iterable[float]) -> np.ndarray:
        if self.prior == "uniform":
            sampled_points = [
                point * (self.high - self.low) + self.low for point in point_list
            ]
        else:
            log_sampled_points = [
                point * np.log(self.high / self.low) + np.log(self.low)
                for point in point_list
            ]
            sampled_points = np.exp(log_sampled_points)
        return sampled_points


class Integer(Dimension):
    def __init__(self, low, high, transform=None, name=None):
        """Search space dimension that can take on integer values.

        Parameters
        ----------
        * `low` [int]:
            Lower bound (inclusive).

        * `high` [int]:
            Upper bound (inclusive).

        * `transform` ["identity", "normalize", optional]:
            The following transformations are supported.

            - "identity", (default) the transformed space is the same as the
              original space.
            - "normalize", the transformed space is scaled to be between
              0 and 1.

        * `name` [str or None]:
            Name associated with dimension, e.g., "number of trees".
        """
        if high <= low:
            raise ValueError(
                "The lower bound {} has to be less than the "
                "upper bound {}".format(low, high)
            )
        self.low = low
        self.high = high
        self.name = name

        if transform is None:
            transform = "identity"

        self.transform_ = transform

        if transform not in ["normalize", "identity"]:
            raise ValueError(
                "Transform should be 'normalize' or 'identity' "
                "got {}".format(self.transform_)
            )
        if transform == "normalize":
            self.transformer = Normalize(low, high, is_int=True)
        else:
            self.transformer = Identity()

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and np.allclose([self.low], [other.low])
            and np.allclose([self.high], [other.high])
        )

    def __repr__(self):
        return "Integer(low={}, high={})".format(self.low, self.high)

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the
        orignal space.
        """
        # The concatenation of all transformed dimensions makes Xt to be
        # of type float, hence the required cast back to int.
        return super(Integer, self).inverse_transform(Xt).astype(np.int64)

    @property
    def bounds(self):
        return (self.low, self.high)

    def __contains__(self, point):
        return self.low <= point <= self.high

    @property
    def transformed_bounds(self):
        if self.transform_ == "normalize":
            return 0, 1
        else:
            return (self.low, self.high)

    def distance(self, a, b):
        """Compute distance between point `a` and `b`.

        Parameters
        ----------
        * `a` [int]
            First point.

        * `b` [int]
            Second point.
        """
        if not (a in self and b in self):
            raise RuntimeError(
                "Can only compute distance for values within "
                "the space, not %s and %s." % (a, b)
            )
        return abs(a - b)

    def _sample(self, point_list: Iterable[float]) -> np.ndarray:
        point_list = [
            point * (self.high + 1 - self.low) + self.low for point in point_list
        ]
        return np.floor(point_list).astype(int)


class Categorical(Dimension):
    def __init__(self, categories, prior=None, transform=None, name=None):
        """Search space dimension that can take on categorical values.

        Parameters
        ----------
        * `categories` [list, shape=(n_categories,)]:
            Sequence of possible categories.

        * `prior` [list, shape=(categories,), default=None]:
            Prior probabilities for each category. By default all categories
            are equally likely.

        * `transform` ["onehot", "identity", default="onehot"] :
            - "identity", the transformed space is the same as the original
              space.
            - "onehot", the transformed space is a one-hot encoded
              representation of the original space.

        * `name` [str or None]:
            Name associated with dimension, e.g., "colors".
        """
        if transform == "identity":
            self.categories = tuple([str(c) for c in categories])
        else:
            self.categories = tuple(categories)

        self.name = name

        if transform is None:
            transform = "onehot"
        self.transform_ = transform
        if transform not in ["identity", "onehot"]:
            raise ValueError(
                "Expected transform to be 'identity' or 'onehot' "
                "got {}".format(transform)
            )
        if transform == "onehot":
            self.transformer = CategoricalEncoder()
            self.transformer.fit(self.categories)
        else:
            self.transformer = Identity(dtype=type(categories[0]))

        self.prior = prior

        if prior is None:
            self.prior_ = np.tile(1.0 / len(self.categories), len(self.categories))
        else:
            self.prior_ = prior

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.categories == other.categories
            and np.allclose(self.prior_, other.prior_)
        )

    def __repr__(self):
        if len(self.categories) > 7:
            cats = self.categories[:3] + (_Ellipsis(),) + self.categories[-3:]
        else:
            cats = self.categories

        if self.prior is not None and len(self.prior) > 7:
            prior = self.prior[:3] + [_Ellipsis()] + self.prior[-3:]
        else:
            prior = self.prior

        return "Categorical(categories={}, prior={})".format(cats, prior)

    @property
    def transformed_size(self):
        if self.transform_ == "onehot":
            size = len(self.categories)
            # when len(categories) == 2, CategoricalEncoder outputs a
            # single value
            return size if size != 2 else 1
        return 1

    @property
    def bounds(self):
        return self.categories

    def __contains__(self, point):
        return point in self.categories

    @property
    def transformed_bounds(self):
        if self.transformed_size == 1:
            return (0.0, 1.0)
        else:
            return [(0.0, 1.0) for i in range(self.transformed_size)]

    def distance(self, a, b):
        """Compute distance between category `a` and `b`.

        As categories have no order the distance between two points is one
        if a != b and zero otherwise.

        Parameters
        ----------
        * `a` [category]
            First category.

        * `b` [category]
            Second category.
        """
        if not (a in self and b in self):
            raise RuntimeError(
                "Can only compute distance for values within "
                "the space, not {} and {}.".format(a, b)
            )
        return 1 if a != b else 0

    def _sample(self, point_list: Iterable[float]) -> np.ndarray:
        # XXX check that sum(prior) == 1
        cummulative_prior = np.cumsum(self.prior_)
        # For each point in point_list, find the index of the first element in cummulative_prior that is greater than the point
        # This is the index of the category that the point corresponds to
        category_index = [np.argmax(cummulative_prior > point) for point in point_list]
        return np.array([self.categories[index] for index in category_index])


class Space(object):
    """Search space."""

    def __init__(self, dimensions):
        """Initialize a search space from given specifications.

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
        self.dimensions = [check_dimension(dim) for dim in dimensions]

    def __eq__(self, other):
        return all([a == b for a, b in zip(self.dimensions, other.dimensions)])

    def __repr__(self):
        if len(self.dimensions) > 31:
            dims = self.dimensions[:15] + [_Ellipsis()] + self.dimensions[-15:]
        else:
            dims = self.dimensions
        return "Space([{}])".format(",\n       ".join(map(str, dims)))

    def __iter__(self):
        return iter(self.dimensions)

    def __len__(self):
        return len(self.dimensions)

    @property
    def is_real(self):
        """
        Returns true if all dimensions are Real
        """
        return all([isinstance(dim, Real) for dim in self.dimensions])

    @classmethod
    def from_yaml(cls, yml_path, namespace=None):
        """Create Space from yaml configuration file

        Parameters
        ----------
        * `yml_path` [str]:
            Full path to yaml configuration file, example YaML below:
            Space:
              - Integer:
                  low: -5
                  high: 5
              - Categorical:
                  categories:
                  - a
                  - b
              - Real:
                  low: 1.0
                  high: 5.0
                  prior: log-uniform
        * `namespace` [str, default=None]:
           Namespace within configuration file to use, will use first
             namespace if not provided

        Returns
        -------
        * `space` [Space]:
           Instantiated Space object
        """
        with open(yml_path, "rb") as f:
            config = yaml.safe_load(f)

        dimension_classes = {
            "real": Real,
            "integer": Integer,
            "categorical": Categorical,
        }

        # Extract space options for configuration file
        if isinstance(config, dict):
            if namespace is None:
                options = next(iter(config.values()))
            else:
                options = config[namespace]
        elif isinstance(config, list):
            options = config
        else:
            raise TypeError("YaML does not specify a list or dictionary")

        # Populate list with Dimension objects
        dimensions = []
        for option in options:
            key = next(iter(option.keys()))
            # Make configuration case insensitive
            dimension_class = key.lower()
            values = {k.lower(): v for k, v in option[key].items()}
            if dimension_class in dimension_classes:
                # Instantiate Dimension subclass and add it to the list
                dimension = dimension_classes[dimension_class](**values)
                dimensions.append(dimension)

        space = cls(dimensions=dimensions)

        return space

    def rvs(
        self,
        n_samples=1,
        random_state: Union[
            int, np.random.RandomState, np.random.Generator, None
        ] = None,
    ):
        """Draw random samples.

        The samples are in the original space. They need to be transformed
        before being passed to a model or minimizer by `space.transform()`.

        Parameters
        ----------
        * `n_samples` [int, default=1]:
            Number of samples to be drawn from the space.

        * `random_state` [int, np.random.RandomState, np.random.Generator, or None, default=None]:
            Set random state to something other than None for reproducible
            results.

        Returns
        -------
        * `points`: [list of lists, shape=(n_points, n_dims)]
           Points sampled from the space.
        """

        rng = get_random_generator(random_state)

        columns = []

        for dim in self.dimensions:
            index_array = rng.uniform(size=n_samples)
            columns.append(dim.sample(index_array))

        # Transpose
        rows = []
        for i in range(n_samples):
            r = []
            for j in range(self.n_dims):
                r.append(columns[j][i])

            rows.append(r)

        return rows

    def transform(self, X):
        """Transform samples from the original space into a warped space.

        Note: this transformation is expected to be used to project samples
              into a suitable space for numerical optimization.

        Parameters
        ----------
        * `X` [list of lists, shape=(n_samples, n_dims)]:
            The samples to transform.

        Returns
        -------
        * `Xt` [array of floats, shape=(n_samples, transformed_n_dims)]
            The transformed samples.
        """
        # Pack by dimension
        columns = []
        for dim in self.dimensions:
            columns.append([])

        for i in range(len(X)):
            for j in range(self.n_dims):
                columns[j].append(X[i][j])

        # Transform
        for j in range(self.n_dims):
            columns[j] = self.dimensions[j].transform(columns[j])

        # Repack as an array
        Xt = np.hstack([np.asarray(c).reshape((len(X), -1)) for c in columns])

        return Xt

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back to the
           original space.

        Parameters
        ----------
        * `Xt` [array of floats, shape=(n_samples, transformed_n_dims)]:
            The samples to inverse transform.

        Returns
        -------
        * `X` [list of lists, shape=(n_samples, n_dims)]
            The original samples.
        """
        # Inverse transform
        columns = []
        start = 0

        for j in range(self.n_dims):
            dim = self.dimensions[j]
            offset = dim.transformed_size

            if offset == 1:
                columns.append(dim.inverse_transform(Xt[:, start]))
            else:
                columns.append(dim.inverse_transform(Xt[:, start : start + offset]))

            start += offset

        # Transpose
        rows = []

        for i in range(len(Xt)):
            r = []
            for j in range(self.n_dims):
                r.append(columns[j][i])

            rows.append(r)

        return rows

    @property
    def n_dims(self):
        """The dimensionality of the original space."""
        return len(self.dimensions)

    @property
    def transformed_n_dims(self):
        """The dimensionality of the warped space."""
        return sum([dim.transformed_size for dim in self.dimensions])

    @property
    def bounds(self):
        """The dimension bounds, in the original space."""
        b = []

        for dim in self.dimensions:
            if dim.size == 1:
                b.append(dim.bounds)
            else:
                b.extend(dim.bounds)

        return b

    @property
    def names(self):
        """The names of the dimensions if given. Otherwise [X1, X2, ... Xn]"""
        labels = [
            "$X_{%i}$" % i if d.name is None else d.name
            for i, d in enumerate(self.dimensions)
        ]
        return labels

    def __contains__(self, point):
        """Check that `point` is within the bounds of the space."""
        for component, dim in zip(point, self.dimensions):
            if component not in dim:
                return False
        return True

    @property
    def transformed_bounds(self):
        """The dimension bounds, in the warped space."""
        b = []

        for dim in self.dimensions:
            if dim.transformed_size == 1:
                b.append(dim.transformed_bounds)
            else:
                b.extend(dim.transformed_bounds)

        return b

    @property
    def is_categorical(self):
        """Space contains exclusively categorical dimensions"""
        return all([isinstance(dim, Categorical) for dim in self.dimensions])

    @property
    def is_partly_categorical(self):
        """Space contains any categorical dimensions"""
        return any([isinstance(dim, Categorical) for dim in self.dimensions])

    def distance(self, point_a, point_b):
        """Compute the L1 (Manhattan or taxicab) distance between two points in this space.

        Parameters
        ----------
        * `a` [array]
            First point.

        * `b` [array]
            Second point.
        """
        distance = 0.0
        if len(self.dimensions) > 1:
            for a, b, dim in zip(point_a, point_b, self.dimensions):
                distance += dim.distance(a, b)

        if len(self.dimensions) == 1:
            distance += self.dimensions[0].distance(point_a[0], point_b[0])

        return distance

    def lhs(
        self,
        n: int,
        seed: Union[int, float, np.random.RandomState, np.random.Generator, None] = 42,
    ):
        """Returns n latin hypercube samples as a list of lists

        Parameters
        ----------
        * `n` [int]: The number of samples to generate.

        * `seed`
            [int, float, np.random.RandomState, np.random.Generator, or None, default=42]:
            The seed used by the random number generator. If None, the results are not reproducible.
        """
        rng = get_random_generator(seed)
        samples = []
        for i in range(self.n_dims):
            lhs_perm = []
            # Get evenly distributed samples from one dimension
            sample_indices = (np.arange(n) + 0.5) / n
            lhs_aranged = self.dimensions[i].sample(sample_indices)
            perm = rng.permutation(n)
            for p in perm:  # Random permutate the order of the samples
                lhs_perm.append(lhs_aranged[p])
            samples.append(lhs_perm)
        # Now we have a list of lists with samples for each dimension.
        # We need to transpose this so that we get a list of lists with
        # samples for all the dimensions
        transposed_samples = []
        for i in range(n):
            row = []
            for j in range(self.n_dims):
                row.append(samples[j][i])
            transposed_samples.append(row)
        return transposed_samples

    # TODO: Add a R1 QRS sampling method here
