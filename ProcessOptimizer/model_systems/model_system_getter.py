from .branin_hoo import create_branin
from .color_pH import create_color_ph
from .gold_map import create_gold_map
from .gold_map_with_wells import create_gold_map_with_wells
from .hart3 import create_hart3
from .hart6 import create_hart6
from .model_system import ModelSystem
from .peaks import create_peaks
from .poly2 import create_poly2


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
    elif model_system == "hart3":
        return create_hart3()
    elif model_system == "hart3_no_noise":
        return create_hart3(noise=False)
    elif model_system == "hart6":
        return create_hart6()
    elif model_system == "hart6_no_noise":
        return create_hart6(noise=False)
    elif model_system == "poly2":
        return create_poly2()
    elif model_system == "poly2_no_noise":
        return create_poly2(noise=False)
    elif model_system == "peaks":
        return create_peaks()
    elif model_system == "peaks_no_noise":
        return create_peaks(noise=False)
    else:
        raise ValueError(f"Model system {model_system} not found.")
