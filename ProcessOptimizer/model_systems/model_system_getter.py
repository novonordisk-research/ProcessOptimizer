from .branin_hoo import create_branin
from .color_pH import create_color_ph
from .gold_map import create_gold_map, create_distance_map
from .gold_map_with_wells import create_gold_map_with_wells
from .hart3 import create_hart3
from .hart6 import create_hart6
from .model_system import ModelSystem
from .peaks import create_peaks
from .poly2 import create_poly2


def get_model_system(model_system: str, **kwargs) -> ModelSystem:
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
    creator_dict = {
        "branin_hoo": (create_branin,),
        "branin_no_noise": (create_branin, False),
        "color_ph": (create_color_ph,),
        "color_pH": (create_color_ph,),
        "colour_ph": (create_color_ph,),
        "colour_pH": (create_color_ph,),
        "gold_map": (create_gold_map,),
        "distance_map": (create_distance_map,),
        "gold_map_with_wells": (create_gold_map_with_wells,),
        "hart3": (create_hart3,),
        "hart3_no_noise": (create_hart3, False),
        "hart6": (create_hart6,),
        "hart6_no_noise": (create_hart6, False),
        "poly2": (create_poly2,),
        "poly2_no_noise": (create_poly2, False),
        "peaks": (create_peaks,),
        "peaks_no_noise": (create_peaks, False),
    }
    if model_system not in creator_dict:
        raise ValueError(
            f"Model system {model_system} not found. "
            f"Choose from {list(creator_dict.keys())}."
        )
    model_system_tuple = creator_dict[model_system]
    if len(model_system_tuple) == 1:
        return model_system_tuple[0](**kwargs)
    else:
        return model_system_tuple[0](noise=model_system_tuple[1], **kwargs)
