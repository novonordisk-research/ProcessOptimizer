from abc import ABC, abstractmethod
from typing import Callable, List, Union, Optional

from scipy.stats import norm, uniform

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
            noise_size: Optional[float]):
        # We could just set noise_dist and let that have a size, but being able to
        # directly set the size is more intuitive, and that would be complicated if it
        # was just one variable.
        # Note that this has the potential for problems if noise_dist does not have
        # "size" 1, but as long as it is only set by set_noise_type, it should be safe.
        self.noise_size = noise_size
        self._noise_distribution: Callable[[],float]
        self.set_noise_type("normal")

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
    
    def set_noise_type(self,noise_type: str):
        if noise_type in ["normal", "Gaussian", "norm"]:
            self._noise_distribution = norm.rvs
        elif noise_type == "uniform":
            self._noise_distribution = uniform(loc=-1,scale=2).rvs
        else:
            raise ValueError(f"Noise distribution \"{noise_type}\" not recognised.")


class AdditiveNoise(NoiseModel): # Should this be named ConstantNoise?
    """
    Noise model for constant noise.

    Parameters:
    * `noise_size` [float, default 1]: The size of the noise. 
    """
    def __init__(self, noise_size: float = 1, **kwargs):
        super().__init__(noise_size=noise_size, **kwargs)

    def get_noise(self,_,Y: float) -> float:
        return self.raw_noise

    
class MultiplicativeNoise(NoiseModel): # Should this be named ProportionalNoise?
    """
    Noise model for noise proportional to the signal.

    Parameters:
    * `noise_size` [float, default 0.01]: The size of the noise realitve to the signal.
    """
    def __init__(self, noise_size : float = 0.01,**kwargs):
        super().__init__(noise_size=noise_size,**kwargs)
    
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
        super().__init__(noise_size=None, **kwargs)
    
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

class SumNoise(NoiseModel):
    """
    Noise model that returns the sum of two or more noise models. Can be used if the
    total noise is a sum of constant and relative noise.

    Args:
    * `noise_model_list` [List[Union[dict,NoiseModel]]]: List of either noise models, or
        dicts containing at least the type of noise model to create.

    """
    def __init__(self,noise_model_list: List[Union[str,dict,NoiseModel]], **kwargs):
        super().__init__(noise_size = None,**kwargs)
        self.noise_model_list: List[NoiseModel] = []
        self.set_noise_model_list(noise_model_list=noise_model_list)

    def set_noise_model_list(self,noise_model_list: List[Union[dict,NoiseModel]]):
        for model_description in noise_model_list:
            self.noise_model_list.append(parse_noise_model(model_description))

    def get_noise(self,X,Y: float) -> float:
        noise_list = [model.get_noise(X,Y) for model in self.noise_model_list]
        return sum(noise_list)


def parse_noise_model(model: Union[str,dict,NoiseModel]):
    if isinstance(model,NoiseModel):
        return model
    elif type(model) == str:
        return noise_model_factory(model_type=model)
    else:
        return noise_model_factory(**model)

def noise_model_factory(model_type: str, **kwargs)-> NoiseModel:
    if model_type == "additive":
        return AdditiveNoise(**kwargs)
    elif model_type == "multiplicative":
        return MultiplicativeNoise(**kwargs)
    elif model_type == "zero":
        return ZeroNoise()
    else:
        raise ValueError(f"Noise model of type '{model_type}' not recognised")