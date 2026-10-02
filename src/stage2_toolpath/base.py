from abc import ABC, abstractmethod
from typing import List
from src.common.schemas import UniversalLayer, MinimalPrintProfile, CLDataTrajectory

class BaseToolpathGenerator(ABC):
    """
    Abstract Base Class for Stage 2: Toolpath Generator.
    """

    @abstractmethod
    def generate_toolpath(self, layers: List[UniversalLayer], profile: MinimalPrintProfile) -> CLDataTrajectory:
        """
        Takes Universal Sliced Layers and a Minimal Print Profile to generate a Universal CL-Data Trajectory.
        """
        pass
