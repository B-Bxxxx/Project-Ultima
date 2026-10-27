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


DEAD_ZONE_HYSTERESIS = 1.5


def resolve_branches(raw_c_rad, b_deg, waypoints, b_lo: float, b_hi: float, c_lo: float, c_hi: float,
                     dead_zone_deg: float = 0.0, hold_b_deg: float = C_HOLD_B_DEG, tolerance_deg: float = 0.0):
    """
    Tilt B and azimuth C (deg) of a trajectory, given the canonical branch (B >= 0, C = azimuth) of every point.

    The same nozzle direction has a second form: (-B, C + 180). Where the tilt direction reverses (the layer normal crosses
    the vertical along a ridge), B can simply pass through zero with C unchanged instead of C turning by 180 deg in a short
    move - which the firmware would draw as a half circle. So inside a move chain each point takes the form whose C is
    nearest to the previous C (tie: the nearer B), as long as -B is inside [b_lo, b_hi]. A new chain (travel move) starts with
    B >= 0 and C closest to zero, as before (see resolve_c_angles).
    Dead zone: a tilt below `dead_zone_deg` is replaced by B = 0 (nozzle vertical, the azimuth is held): the nozzle axis may
    deviate that much from the surface normal (Stage 4 corrects the thickness with tool_alignment).
    The dead zone has hysteresis (`DEAD_ZONE_HYSTERESIS`): it is entered below the dead-zone tilt but left only above 1.5 times
    that, so a tilt that oscillates around the threshold (a rounded ridge with near-vertical layers) does not make the head turn on
    the spot every time it crosses it.
    Orientation smoothing (`tolerance_deg` > 0): after the branches are chosen, B and C of every chain are replaced by the smoothest
    curve that keeps the nozzle axis within `tolerance_deg` of the wanted direction (see `smooth_orientation`).
    Returns (b_list, c_list).
    """
    import logging
    hold = max(hold_b_deg, dead_zone_deg)
    leave = hold * DEAD_ZONE_HYSTERESIS if dead_zone_deg > 0.0 else hold      # the dead zone is left only above this tilt
    in_zone = False
    held = []                                       # True where the point sits in the dead zone (B = 0, C held)
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
        if out_c and abs(b) < (leave if in_zone else hold):
            c, bb = prev_c, (0.0 if dead_zone_deg > 0.0 else b)
            in_zone = True
            held.append(True)
        else:
            in_zone = False
            held.append(False)
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
    if dead_zone_deg > 0.0:
        _ramp_c_in_dead_zone(out_c, held, waypoints, c_lo, c_hi)
    if tolerance_deg > 0.0:
        out_b, out_c = smooth_orientation(out_b, out_c, waypoints, tolerance_deg, b_lo, b_hi, c_lo, c_hi)
    return out_b, out_c


def _taut_string(s, lo, hi):
    """Shortest path x(s) through the corridor lo <= x <= hi (the 'taut string': piecewise linear, bends only on the corridor walls).
    lo == hi at the first and last point pins them. Returns x at every s."""
    n = len(s)
    x = np.empty(n)
    x[0] = lo[0]
    k = 0
    while k < n - 1:
        ak = x[k]
        smax_lo, i_lo = -np.inf, k + 1                  # steepest slope that still clears every lower wall seen so far
        smin_hi, i_hi = np.inf, k + 1                   # shallowest slope that still clears every upper wall
        nxt = None
        for i in range(k + 1, n):
            d = s[i] - s[k]
            if d <= 1e-12:
                continue
            sl, sh = (lo[i] - ak) / d, (hi[i] - ak) / d
            if sl > smin_hi + 1e-12:                    # a lower wall rose above the cone: bend at the upper wall point
                nxt, val = i_hi, hi[i_hi]
                break
            if sh < smax_lo - 1e-12:
                nxt, val = i_lo, lo[i_lo]
                break
            if sl > smax_lo:
                smax_lo, i_lo = sl, i
            if sh < smin_hi:
                smin_hi, i_hi = sh, i
        if nxt is None:                                 # reached the end: straight to the pinned last point
            nxt, val = n - 1, lo[n - 1]
        x[nxt] = val
        for m in range(k + 1, nxt):
            x[m] = ak + (val - ak) * (s[m] - s[k]) / max(s[nxt] - s[k], 1e-12)
        k = nxt
    return x


def smooth_orientation(out_b, out_c, waypoints, tol_deg, b_lo, b_hi, c_lo, c_hi):
    """
    Orientation smoothing: inside every move chain B(s) and C(s) are replaced by the shortest ('taut string') curves that stay within a
    corridor around the exact ones, so B and C turn as little as possible. The corridor keeps the nozzle axis within `tol_deg` of the
    wanted direction: |dB| <= tol/sqrt2 and sin(B)|dC| <= tol/sqrt2 (so the true angle error is <= tol). The first and last point of a
    chain are kept exact. The thickness error of the tilted nozzle is corrected by tool_alignment in Stage 4.
    """
    n = len(out_b)
    b = np.array(out_b, float)
    c = np.array(out_c, float)
    if n < 3:
        return list(b), list(c)
    hd = tol_deg / np.sqrt(2.0)
    P = np.array([[w.x, w.y, w.z] for w in waypoints], float)
    trav = [bool(getattr(w, "is_travel_move", False)) for w in waypoints]
    starts = [0] + [i for i in range(1, n) if trav[i]] + [n]
    for a, z in zip(starts[:-1], starts[1:]):
        if z - a < 3:
            continue
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P[a:z], axis=0), axis=1))]
        if s[-1] < 1e-9:
            continue
        b0, c0 = b[a:z].copy(), c[a:z].copy()
        blo, bhi = np.maximum(b0 - hd, b_lo), np.minimum(b0 + hd, b_hi)
        wide = np.minimum(90.0, hd / np.maximum(np.sin(np.radians(np.minimum(np.abs(b0) + hd, 90.0))), 1e-6))
        clo, chi = np.maximum(c0 - wide, c_lo), np.minimum(c0 + wide, c_hi)
        for arr, v in ((blo, b0), (bhi, b0), (clo, c0), (chi, c0)):   # chain ends stay exact
            arr[0], arr[-1] = v[0], v[-1]
        b[a:z] = _taut_string(s, blo, bhi)
        c[a:z] = _taut_string(s, clo, chi)
    return list(b), list(c)


def _ramp_c_in_dead_zone(out_c, held, waypoints, c_lo, c_hi):
    """
    Inside the dead zone the nozzle is vertical (B = 0), so C does not matter: it is turned there, gradually along the path, from the
    held angle to the angle the next point outside the zone needs. The head then never has to turn on the spot when it leaves the
    zone (the old jump, spread by refine._unwind into stationary steps with almost no extrusion). In place, C only.
    """
    n = len(out_c)
    k = 1
    while k < n:
        if not held[k]:
            k += 1
            continue
        i = k                                           # run of held points i .. j-1, left by point j
        while k < n and held[k]:
            k += 1
        j = k
        if j >= n or any(getattr(waypoints[m], "is_travel_move", False) for m in range(i, j + 1)):
            continue                                    # the run ends the chain / contains a travel: nothing to join
        c0, c1 = out_c[i - 1], out_c[j]
        if abs(c1 - c0) < 1e-9:
            continue
        pts = np.array([[waypoints[m].x, waypoints[m].y, waypoints[m].z] for m in range(i - 1, j + 1)], float)
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        if s[-1] < 1e-9:
            continue
        for m in range(i, j):
            c = c0 + (c1 - c0) * s[m - i + 1] / s[-1]
            out_c[m] = min(max(c, c_lo), c_hi)


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
