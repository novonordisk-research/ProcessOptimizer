from .constant_suggestor import ConstantSuggestor
import warnings

from .constant_suggestor import ConstantSuggestor
from .default_suggestor import DefaultSuggestor
from .golden_ratio_suggestor import GoldenRatioSuggestor
from .lhs_suggestor import LHSSuggestor
from .po_suggestor import OptimizerSuggestor
from .random_strategizer import RandomStragegizer
from .sequential_strategizer import SequentialStrategizer
from .suggestor import CreatableSuggestor, IncompatibleNumberAsked, Suggestor

try:
    from .multitask_suggestor import MTSuggestor
except ModuleNotFoundError as me:
    warnings.warn(
        "Not all packages necessary for multitask suggestor installed. Will not be available."
    )
from .suggestor_factory import suggestor_factory

__all__ = [
    "ConstantSuggestor",
    "CreatableSuggestor",
    "DefaultSuggestor",
    "GoldenRatioSuggestor",
    "IncompatibleNumberAsked",
    "LHSSuggestor",
    "OptimizerSuggestor",
    "RandomStragegizer",
    "SequentialStrategizer",
    "MTSuggestor",
    "Suggestor",
    "suggestor_factory",
]
