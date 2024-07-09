from .branin_hoo import create_branin
from .model_system import ModelSystem


def get_model_system(model_system: str) -> ModelSystem:
    """
    Get the model system object for the given model system name.

    Parameters
    ----------
    * `model_system` [str]:
        The name of the model system to get.

    Returns
    -------
    * `model_system` [ModelSystem]:
        The model system object.
    """
    if model_system == "branin_hoo":
        return create_branin(noise=True)
    elif model_system == "branin_no_noise":
        return create_branin(noise=False)
    else:
        raise ValueError(f"Model system {model_system} not found.")
