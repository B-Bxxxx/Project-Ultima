from typing import Dict, Type
from src.stage3_kinematics.base import BaseKinematicSolver

class KinematicsRegistry:
    """
    Registry for dynamic loading of Stage 3 Kinematic Solvers.
    """
    _solvers: Dict[str, Type[BaseKinematicSolver]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(solver_cls: Type[BaseKinematicSolver]):
            cls._solvers[name] = solver_cls
            return solver_cls
        return wrapper

    @classmethod
    def get_solver(cls, name: str) -> Type[BaseKinematicSolver]:
        if name not in cls._solvers:
            raise ValueError(f"Solver '{name}' not found. Available: {list(cls._solvers.keys())}")
        return cls._solvers[name]

    @classmethod
    def get_all_solvers(cls) -> Dict[str, Type[BaseKinematicSolver]]:
        return cls._solvers
