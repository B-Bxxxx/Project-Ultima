from abc import ABC, abstractmethod
from typing import List
from src.common.schemas import MachineStateVector, MinimalPrintProfile

class BasePostProcessor(ABC):
    """
    Abstract Base Class for Stage 4: Post-Processor.
    """

    @abstractmethod
    def generate_gcode(self, states: List[MachineStateVector], profile: MinimalPrintProfile) -> str:
        """
        Takes machine-specific state vectors and outputs executable G-code string.
        """
        pass
