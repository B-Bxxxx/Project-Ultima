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


C_HOLD_B_DEG = 0.5          # below this tilt C is meaningless (singularity): it keeps its previous value


def resolve_c_angles(raw_c_rad, b_deg, waypoints, c_min: float, c_max: float, hold_b_deg: float = C_HOLD_B_DEG):
    """
    C angles (deg) for the raw atan2 angles of a trajectory.
    - inside one continuous move chain (extrusion after extrusion) C stays continuous: the nearest equivalent
      angle (+-360 deg) to the previous one, so the table never jumps;
    - at the start of every new chain (a travel move: new ring, wall chain, infill line) C is re-chosen as the
      equivalent angle closest to ZERO inside [c_min, c_max]: the winding that a spiralling wall collects is
      undone during the travel instead of piling up to the axis limit;
    - below `hold_b_deg` of tilt C keeps its previous value.
    """
    import logging
    out = []
    prev = 0.0
    clipped = False
    for r, b, wp in zip(raw_c_rad, b_deg, waypoints):
        if out and abs(b) < hold_b_deg:
            c = prev
        else:
            base = float(np.degrees(r))
            if not out or getattr(wp, "is_travel_move", False):
                kmin = int(np.ceil((c_min - base) / 360.0 - 1e-9))
                kmax = int(np.floor((c_max - base) / 360.0 + 1e-9))
                if kmin > kmax:
                    c, clipped = min(max(base, c_min), c_max), True
                else:
                    c = base + 360.0 * min(max(int(round(-base / 360.0)), kmin), kmax)
            else:
                c = base + 360.0 * round((prev - base) / 360.0)
                if c > c_max or c < c_min:
                    c, clipped = min(max(c, c_min), c_max), True
        out.append(c)
        prev = c
    if clipped:
        logging.getLogger(__name__).warning("C axis: a chain needs more rotation than the C limits allow (%g..%g deg); "
                                            "the C angle was clamped, the part will be wrong there", c_min, c_max)
    return out


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
