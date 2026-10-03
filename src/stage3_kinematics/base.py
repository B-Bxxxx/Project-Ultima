from abc import ABC, abstractmethod
from typing import List
from src.common.schemas import CLDataTrajectory, MachineConfig, MachineTrajectory

class BaseKinematicSolver(ABC):
    """
    Abstract Base Class for Stage 3: Kinematic Solver.
    """

    @abstractmethod
    def solve(self, trajectory: CLDataTrajectory, config: MachineConfig) -> MachineTrajectory:
        """
        Takes a machine-agnostic CL-Data Trajectory and Machine Config, and outputs a Machine Trajectory.
        """
        pass
