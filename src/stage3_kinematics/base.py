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


C_HOLD_B_DEG = 0.006        # below this tilt (the 1e-4 rad of the solvers) C is undefined: it keeps its previous value. A larger
                            # threshold makes C jump where B crosses it and the firmware then spins the table inside one segment


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


def resolve_branches(raw_c_rad, b_deg, waypoints, b_lo: float, b_hi: float, c_lo: float, c_hi: float,
                     dead_zone_deg: float = 0.0, hold_b_deg: float = C_HOLD_B_DEG):
    """
    Tilt B and azimuth C (deg) of a trajectory, given the canonical branch (B >= 0, C = azimuth) of every point.

    The same nozzle direction has a second form: (-B, C + 180). Where the tilt direction reverses (the layer normal crosses
    the vertical along a ridge), B can simply pass through zero with C unchanged instead of C turning by 180 deg in a short
    move - which the firmware would draw as a half circle. So inside a move chain each point takes the form whose C is
    nearest to the previous C (tie: the nearer B), as long as -B is inside [b_lo, b_hi]. A new chain (travel move) starts with
    B >= 0 and C closest to zero, as before (see resolve_c_angles).
    Dead zone: a tilt below `dead_zone_deg` is replaced by B = 0 (nozzle vertical, the azimuth is held): the nozzle axis may
    deviate that much from the surface normal (Stage 4 corrects the thickness with tool_alignment).
    Returns (b_list, c_list).
    """
    import logging
    hold = max(hold_b_deg, dead_zone_deg)
    eps = 1e-9
    out_b, out_c = [], []
    prev_c = prev_b = 0.0
    clipped = False

    def near_zero(base):
        kmin = int(np.ceil((c_lo - base) / 360.0 - 1e-9))
        kmax = int(np.floor((c_hi - base) / 360.0 + 1e-9))
        if kmin > kmax:
            return min(max(base, c_lo), c_hi), True
        return base + 360.0 * min(max(int(round(-base / 360.0)), kmin), kmax), False

    for r, b, wp in zip(raw_c_rad, b_deg, waypoints):
        start = not out_c or getattr(wp, "is_travel_move", False)
        if out_c and abs(b) < hold:
            c, bb = prev_c, (0.0 if dead_zone_deg > 0.0 else b)
        else:
            base = float(np.degrees(r))
            ok1 = b_lo - eps <= b <= b_hi + eps               # the canonical form (B, C) / the flipped form (-B, C + 180)
            ok2 = b_lo - eps <= -b <= b_hi + eps
            bb = b
            if start:
                if not ok1 and ok2:
                    bb, (c, cl) = -b, near_zero(base + 180.0)
                else:
                    c, cl = near_zero(base)
                clipped |= cl
            else:
                c = base + 360.0 * round((prev_c - base) / 360.0)
                c2 = base + 180.0 + 360.0 * round((prev_c - base - 180.0) / 360.0)
                d1, d2 = abs(c - prev_c), abs(c2 - prev_c)
                flip = ok2 and (not ok1 or d2 < d1 - eps or (abs(d2 - d1) <= eps and abs(-b - prev_b) < abs(b - prev_b)))
                if flip:
                    c, bb = c2, -b
                if c > c_hi or c < c_lo:
                    c, clipped = min(max(c, c_lo), c_hi), True
        out_b.append(bb)
        out_c.append(c)
        prev_c, prev_b = c, bb
    if clipped:
        logging.getLogger(__name__).warning("C axis: a chain needs more rotation than the C limits allow (%g..%g deg); "
                                            "the C angle was clamped, the part will be wrong there", c_lo, c_hi)
    return out_b, out_c


class BaseKinematicSolver(ABC):
    """
    Abstract Base Class for Stage 3: Kinematic Solver.
    """

    def to_part_batch(self, config: MachineConfig, poses):
        """
        Inverse of solve() for the pose arrays (n, 5) = [x, y, z, b, c] as written to the G-code: where on the PART the
        nozzle tip is. Used to check the straight-line axis interpolation between two points. None = not available
        (the solver is then never refined).
        """
        return None

    @abstractmethod
    def solve(self, trajectory: CLDataTrajectory, config: MachineConfig) -> MachineTrajectory:
        """
        Takes a machine-agnostic CL-Data Trajectory and Machine Config, and outputs a Machine Trajectory.
        """
        pass
