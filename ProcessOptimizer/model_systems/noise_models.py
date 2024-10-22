from abc import ABC, abstractmethod
from typing import Callable, List, Union, Optional

import numpy as np

from ProcessOptimizer.utils import get_random_generator


class NoiseModel(ABC):
    """
    Abstract class that is the basis for noise models.
    """

    def __init__(
        self,
        noise_size: Optional[float],
        seed: Union[int, np.random.RandomState, np.random.Generator, None] = 42,
    ):
        """
        Parameters
        ----------
        * `noise_size` [Optional[float]]:
            The size (magnitude) of the noise. If ´None´, it signifies that
            the class is compound, and should not have its own noise signal,
            but refer to its compounding NoiseModels instead. An example is
            SumNoise, which sums the noise of a list of noise models, but does
            not add any noise by itself.

        * `seed` [Optional[int], default=42]:
            Seed to pass forward to the numpy random number generator that is
            used when providing samples with noise.
        """
        # We could just set noise_dist and let that have a size, but being able to
        # directly set the size is more intuitive, and that would be complicated if it
        # was just one variable.
        # Note that this has the potential for problems if _noise_distribution does not
        # have "size" 1, but as long as it is only set by set_noise_type(), it should be
        # safe.
        self.noise_size = noise_size
        self._rng = get_random_generator(seed)
        self.noise_type = "normal"

    @abstractmethod
    def get_noise(self, X, Y: float) -> float:
        pass

    @property
    def _sample_noise(self) -> float:
        """A raw noise value, to be used in the get_noise() function."""
        if self.noise_size is None:
            raise TypeError(
                'Method "raw_noise()" for NoiseModel class '
                f"{self.__class__.__name__} is not supposed to be called."
            )

        return self._noise_distribution() * self.noise_size

    @property
    def noise_type(self) -> str:
        return self._noise_type

    @noise_type.setter
    def noise_type(self, value: str):
        if value in ["normal", "Gaussian", "norm", "uniform"]:
            self._noise_type = value
        else:
            raise ValueError(f'Noise distribution "{value}" not recognised.')

    def set_seed(self, seed: Optional[int]):
        # Instantiate the random number generator again
        self._rng = np.random.default_rng(seed)

    @property
    def _noise_distribution(self) -> Callable[[], float]:
        if self.noise_type in ["normal", "Gaussian", "norm"]:
            return self._rng.normal
        elif self.noise_type == "uniform":
            return lambda: self._rng.uniform(low=-1, high=1)
        else:
            raise ValueError(f'Noise distribution "{self.noise_type}" not recognised.')

    def copy(self) -> "NoiseModel":
        """
        Create a copy of the noise model. This is necessary to avoid the same random
        seed being used in multiple noise models, which would make the noise correlated.
        """
        copy = self.__class__(noise_size=self.noise_size, seed=self._rng.spawn(1))
        # np.random.Generator.spawn() returns a new generator based on the old one, but
        # with a different seed. It is deterministic, but not identical to the old one.
        copy.noise_type = self.noise_type


class ConstantNoise(NoiseModel):
    """
    Noise model for noise that is independent of both the sampled point and the
    resulting score; The noise is constant. Another name is "additive noise", as the
    same noise is added to the score. This is typical for many measurements.

    Parameters
    ----------
    * `noise_size` [float, default=1]:
        The size (magnitude) of the noise.
    """

    def __init__(self, noise_size: float = 1, **kwargs):
        super().__init__(noise_size=noise_size, **kwargs)

    def get_noise(self, _, Y: float) -> float:
        return self._sample_noise


class ProportionalNoise(NoiseModel):
    """
    Noise model for noise proportional to the signal, but independent of the sampled
    point. Another name is "multiplicative noise", as the same noise is multiplied with
    the score. This is typical of some electronic measurements.

    Parameters
    ----------
    * `noise_size` [float, default=0.1]:
        The size of the noise relative to the signal.
    """

    def __init__(self, noise_size: float = 0.1, **kwargs):
        super().__init__(noise_size=noise_size, **kwargs)

    def get_noise(self, _, Y: float) -> float:
        return self._sample_noise * Y


class DataDependentNoise(NoiseModel):
    """
    Noise model for noise that depends on the input parameters.

    Parameters
    ----------
    * `noise_function` [(parameters) -> NoiseModel]:
        A function that takes a set of parameters, and returns a noise model to
        apply.
    * `overwrite_rng` [bool, default=True]:
        Whether to overwrite the random number generator of underlying noise models.
        This is necessary if the underlying noise models are created each time
        `noise_function()` is called, since they will otherwise have the same random
        seed. If the underlying noise models are created once, this can be set to
        False to not mess with the underlying noise models.

    Examples
    --------
    To make additive noise proportional to the input parameter (not to the score):
    ```
    noise_choice = lambda X: ConstantNoise(noise_size=X)
    noise_model = DataDependentNoise(noise_function=noise_choice)
    ```

    To add constant noise except if X[0] is 0:
    ```
    noise_choice = lambda X: ZeroNoise() if X[0]==0 else ConstantNoise()
    noise_model = DataDependentNoise(noise_function=noise_choice)
    ```
    """

    def __init__(
        self,
        noise_function: Callable[..., NoiseModel],
        overwrite_rng: bool = True,
        **kwargs,
    ):
        super().__init__(noise_size=None, **kwargs)
        self.noise_function = noise_function
        self.overwrite_rng = overwrite_rng

    def get_noise(self, X, Y: float) -> float:
        noise_model = self.noise_function(X)
        if self.overwrite_rng:
            noise_model._rng = self._rng
        return noise_model.get_noise(X, Y)


class ZeroNoise(NoiseModel):
    """
    Noise model for zero noise. Doesn't take any arguments. Exist for consistency,
    and to be used in data dependent noise models.
    """

    def __init__(self):
        super().__init__(noise_size=0)

    def get_noise(self, _, Y: float) -> float:
        return 0

    @property
    def noise_size(self):
        return 0

    @noise_size.setter
    def noise_size(self, value):
        if value != 0:
            raise ValueError("ZeroNoise should not have a noise size other than 0.")


class SumNoise(NoiseModel):
    """
    Noise model that returns the sum of two or more noise models. Can be used if the
    total noise is a sum of constant and proportional noise.

    Parameters
    ----------
    * `noise_model_list` [List[Union[dict, NoiseModel]]]:
        List of either noise models, or dicts containing at least the type of
        noise model to create.
    * `overwrite_rng` [bool, default=True]:
        Whether to overwrite the random number generator of underlying noise models.
        This is necessary if the underlying noise models are created with the same seed,
        which they are by default, since they will otherwise give correlated noise.

    """

    def __init__(
        self,
        noise_model_list: List[Union[str, dict, NoiseModel]],
        overwrite_rng: bool = True,
        **kwargs,
    ):
        super().__init__(noise_size=None, **kwargs)
        self.noise_model_list: List[NoiseModel]
        self.overwrite_rng = overwrite_rng
        self.set_noise_model_list(noise_model_list=noise_model_list)

    def set_noise_model_list(
        self, noise_model_list: List[Union[str, dict, NoiseModel]]
    ):
        self.noise_model_list = []
        for model_description in noise_model_list:
            self.noise_model_list.append(parse_noise_model(model_description))
        if self.overwrite_rng:
            for model in self.noise_model_list:
                model._rng = self._rng

    def get_noise(self, X, Y: float) -> float:
        noise_list = [model.get_noise(X, Y) for model in self.noise_model_list]
        return sum(noise_list)


def parse_noise_model(
    model: Union[str, dict, NoiseModel, None], **kwargs
) -> NoiseModel:
    if isinstance(model, NoiseModel):
        return model
    elif isinstance(model, str):
        return noise_model_factory(model_type=model, **kwargs)
    elif model is None:
        return noise_model_factory(model_type="zero")
    else:
        return noise_model_factory(**model)


def noise_model_factory(model_type: str, **kwargs) -> NoiseModel:
    if model_type == "constant":
        return ConstantNoise(**kwargs)
    elif model_type == "proportional":
        return ProportionalNoise(**kwargs)
    elif model_type in ["zero", "none"]:
        return ZeroNoise()
    else:
        raise ValueError(f"Noise model of type '{model_type}' not recognised")
