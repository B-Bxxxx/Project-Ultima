from abc import ABC, abstractmethod
from typing import List
from src.common.schemas import CLDataTrajectory, MachineConfig, MachineStateVector

class BaseKinematicSolver(ABC):
    """
    Abstract Base Class for Stage 3: Kinematic Solver.
    """

    @abstractmethod
    def solve(self, trajectory: CLDataTrajectory, config: MachineConfig) -> List[MachineStateVector]:
        """
        Takes a machine-agnostic CL-Data Trajectory and Machine Config, and outputs a list of Machine State Vectors.
        """
        pass
