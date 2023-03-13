from abc import ABC, abstractmethod
from typing import Optional, Callable

from  scipy.stats import norm

class NoiseModel(ABC):
    """
    Abstract class that is the basis for noise models.
    
    Parameters:
    * `noise_size` [Option[float]]: The size of the noise. If ´None´, it signifies that
        the class is compound, and should not have its own noise signal, but refer to
        its compounding NoiseModels instead. An example is SumNoise, which sums the
        noise of a list of noise models, but doesn't add any noise by itself.
    """
    def __init__(
            self,
            noise_size: float,
            noise_dist: Callable[[],float] = norm.rvs):
        self.noise_size = noise_size
        self.noise_dist = noise_dist

    @abstractmethod
    def get_noise(self,X,Y: float) -> float:
        pass
    
    @property
    def raw_noise(self) -> float:
        """A raw noise value, to be used in the get_noise() function."""
        if self.noise_size is None:
            raise TypeError("Method \"raw_noise()\" for NoiseModel class "
                            f"{self.__class__.__name__} is not supposed to be called.")
        return self._noise_distribution()*self.noise_size
class AdditiveNoise(NoiseModel): # Should this be named ConstantNoise?
    """
    Noise model for constant noise.

    Parameters:
    * `noise_size` [float, default 1]: The size of the noise. 
    """
    def __init__(self, noise_size: float = 1, **kwargs):
        super().__init__(noise_size=noise_size, **kwargs)

    def get_noise(self,_,Y: float) -> float:
        return self.noise


    
class MultiplicativeNoise(NoiseModel): # Should this be named ProportionalNoise?
    """
    Noise model for noise proportional to the signal.

    Parameters:
    * `noise_size` [float, default 0.01]: The size of the noise realitve to the signal.
    """
    def __init__(self, **kwargs):
        if "noise_size" not in kwargs.keys() and "noise_dist" not in kwargs.keys():
            super().__init__(noise_size=0.01, noise_dist=norm.rvs,**kwargs)
        elif "noise_size" in kwargs.keys():
            super().__init__(**kwargs)
        else:
            # If the noise dist is given, but the noise size isn't, the expected
            # behavior is that the noise follows the noise dist.
            super().__init__(noise_size=1,**kwargs)
    
    def get_noise(self,_,Y: float) -> float:
        return self.raw_noise*Y
    

class DataDependentNoise(NoiseModel):
    """
    Noise model for noise that depends on the input parameters.

    Parameters:
    * `noise_models` [(parameters) -> NoiseModel]: A function that takes a set of
        parameters, and returns a noise model to apply.

    Examples:

    To make additive noise proportional to the input parameter (not to the score):
    ```
    noise_choice = lambda X: AdditiveNoise(noise_size=X)
    noise_model = DataDependentNoise(noise_models=noise_choice)
    ```

    To add additive noise except if X[0] is 0:
    ```
    noise_choice = lambda X: ZeroNoise() if X[0]==0 else AdditiveNoise()
    noise_model = DataDependentNoise(noise_models=noise_choice)
    ```
    """
    def __init__(self, noise_models: Callable[...,NoiseModel], **kwargs):
        self.noise_models = noise_models
        super().__init__(noise_size=0, **kwargs)
    
    def get_noise(self,X,Y: float) -> float:
        return self.noise_models(X).get_noise(X,Y)           
    
class ZeroNoise(NoiseModel):
    """Noise model for zero noise. Doesn't take any arguments. Exist for consistency,
    and to be used in data dependent noise models.
    """
    def __init__(self):
        super().__init__(noise_size=0)

    def get_noise(self,_,Y: float) -> float:
        return 0


def noise_model_factory(type: str, **kwargs)-> NoiseModel:
    if type == "additive":
        return AdditiveNoise(**kwargs)
    elif type == "multiplicative":
        return MultiplicativeNoise(**kwargs)
    elif type == "zero":
        return ZeroNoise()
    else:
        raise ValueError(f"Noise model of type '{type}' not recognised")