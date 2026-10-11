from abc import ABC, abstractmethod
from src.common.schemas import UniversalSlicedModel, MinimalPrintProfile, CLDataTrajectory

class BaseToolpathGenerator(ABC):
    """
    Abstract Base Class for Stage 2: Toolpath Generator.
    """

    @abstractmethod
    def generate_toolpath(self, model: UniversalSlicedModel, profile: MinimalPrintProfile) -> CLDataTrajectory:
        """
        Takes a UniversalSlicedModel (layers + the Stage 1 metadata describing the slicing
        strategy) and a MinimalPrintProfile and generates a Universal CL-Data Trajectory.
        Implementations must not modify the input model.
        """
        pass
