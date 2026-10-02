from abc import ABC, abstractmethod
from typing import List, Any, Dict
from src.common.schemas import SlicerPluginParameterSchema, UniversalSlicedModel

class BaseSlicerPlugin(ABC):
    """
    Abstract Base Class for Stage 1: Slicing Engine plugins.
    """

    @classmethod
    @abstractmethod
    def get_parameter_schema(cls) -> List[SlicerPluginParameterSchema]:
        """
        Exposes the dynamic parameter schema for UI/CLI introspection.
        """
        pass

    @abstractmethod
    def slice(self, geometry: Any, parameters: Dict[str, Any]) -> UniversalSlicedModel:
        """
        Takes input geometry and algorithm parameters to output a Universal Sliced Model.
        """
        pass
