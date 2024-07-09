from .branin_hoo import create_branin
from .color_pH import create_color_ph
from .gold_map import create_gold_map
from .gold_map_with_wells import create_gold_map_with_wells
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
    elif model_system in ["branin_no_noise", "branin_hoo_no_noise"]:
        return create_branin(noise=False)
    elif model_system in ["color_ph", "color_pH", "colour_ph", "colour_pH"]:
        return create_color_ph()
    elif model_system == "gold_map":
        return create_gold_map()
    elif model_system == "gold_map_with_wells":
        return create_gold_map_with_wells()
    else:
        raise ValueError(f"Model system {model_system} not found.")
