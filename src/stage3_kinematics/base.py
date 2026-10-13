from abc import ABC, abstractmethod
import numpy as np
from typing import List
from src.common.schemas import CLDataTrajectory, MachineConfig, MachineTrajectory

def tool_alignment(wp, tool_axis) -> float:
    """cos(angle) between the layer surface normal of a waypoint and the actual nozzle axis (both in the part frame)."""
    n = np.array([wp.i, wp.j, wp.k], float)
    d = np.asarray(tool_axis, float)
    nn, dn = np.linalg.norm(n), np.linalg.norm(d)
    if nn < 1e-9 or dn < 1e-9:
        return 1.0
    return float(np.clip(np.dot(n, d) / (nn * dn), -1.0, 1.0))


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
