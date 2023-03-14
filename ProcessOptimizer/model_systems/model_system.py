from typing import Callable, Union

import numpy as np
from ProcessOptimizer import expected_minimum
from ProcessOptimizer.model_systems.noise_models import NoiseModel, ZeroNoise, parse_noise_model

class ModelSystem:
    """
    Model System for testing the ProcessOptimizer. Instances of this class 
    will be used in the example notebooks. 

    Parameters:
    * `score` [callable]:
        Function for calculating the noiseless score of the system at a given point in 
        the parameter space.

    * `space` [Space]:
        The parameter space in the form of a Space object. 

    * `true_min` [float]:
        The true minimum value of the score function within the parameter space.

    * `noise_model` [str, dict, or NoiseModel]:
        Noise model to apply to the score.
        If str, it should be the name of the noise model type. In this case, 
            further arguments can be given (e.g. `noise_size`).
        If dict, one key should be `model_type`.
        If NoiseModel, this NoiseModel will be used.
            
        Possible model type strings are:
            "constant": The noise level is constant.
            "proportional": Tne noise level is proportional to the score.
            "zero": No noise is applied.

    """
    def __init__(self, score: Callable[..., float], space, true_min=None, noise_model: NoiseModel = ZeroNoise):
        self.space = space
        self.score = score
        if true_min is None:
            ndims = space.n_dims()
            points = space.lhs(ndims*10)
            scores = [score(point) for point in points]
            true_min = np.min(scores)
        self.true_min = true_min
        self.noise_model = self.set_noise_model(noise_model)
        
    def result_loss(self, result):
        """Calculate the loss of the optimization result. 

        Parameters:
        * `result` [OptimizeResult object of scipy.optimize.optimize module]:
            The result of an optimization. 

        Returns
        * loss [float]:
            The loss of the system, i.e. the difference between the true system 
            value at the location of the model's expected minimum and the best 
            possible system value. 
        """
        # Get the location of the expected minimum
        model_x,_ = expected_minimum(result)
        # Calculate the difference between the score at model_x and the true minimum value
        loss = self.score(model_x) - self.true_min
        return loss
    
    def get_score(self,X) -> float:
        """Returns the noisy score of the system.

        Parameters:
        * `X`: The point in space to evaluate the score at.

        Returns:
        * Noisy score [float].
        """
        Y = self.score(X)
        return self.noise_model.get_noise(X,Y) + Y
    
    def set_noise_model(self, noise_model: Union[str,dict,NoiseModel], **kwargs):
        """Sets the noise model for the model system

        Parameters:
        * `noise_model` [str, dict, or NoiseModel]:
            If str, it should be the name of the noise model type. In this case, 
                further arguments can be given (e.g. `noise_size`).
            If dict, one key should be `model_type`.
            If NoiseModel, this NoiseModel will be used.
            
            Possible model type strings are:
            "constant": The noise level is constant.
            "proportional": Tne noise level is proportional to the score.
            "zero": No noise is applied.
        """
        self.noise_model = parse_noise_model(noise_model, **kwargs)