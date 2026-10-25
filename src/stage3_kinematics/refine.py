"""
Joint-space interpolation check for rotary machines.

The firmware moves between two G-code points with every axis (X Y Z B C) changing LINEARLY in time. For a straight
segment of the part along which C (or B) changes a lot - e.g. an infill line across a cone, where C follows the azimuth -
the nozzle therefore deposits an arc, not the line. `solve_refined` solves the trajectory, measures that deviation (in
the part frame, through the solver's inverse) at 1/4, 1/2 and 3/4 of every extruding segment, splits the segments that
are off by more than `config.joint_tolerance` (in the part frame, so the extrusion and the layer bookkeeping stay
the same) and solves again, until all are inside the tolerance.
"""
import logging
from typing import List

import numpy as np

from src.common.schemas import CLDataTrajectory, CLDataWaypoint, MachineConfig, MachineTrajectory
from src.stage3_kinematics.base import BaseKinematicSolver

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 14
MIN_SEGMENT = 0.02            # mm: shorter segments are never split
GROWTH_LIMIT = 6              # the refined trajectory may have at most this many times the original number of points
SAMPLES = (0.25, 0.5, 0.75)


def _poses(result: MachineTrajectory, columns: int = 5) -> np.ndarray:
    """[x, y, z, b, c] per state (+ a as 6th column for solvers that have an A axis)."""
    cols = [[s.x, s.y, s.z, s.b, s.c] + ([s.a] if columns > 5 else []) for s in result.states]
    return np.array(cols, float).reshape(-1, columns)


def _bad_segments(solver, config, wps: List[CLDataWaypoint], S: np.ndarray, tol: float):
    """Boolean array: bad[i] = the extruding segment (i-1 -> i) deviates more than `tol`."""
    n = len(wps)
    P = np.array([[w.x, w.y, w.z] for w in wps], float)
    extruding = np.array([not w.is_travel_move for w in wps], bool)
    extruding[0] = False
    length = np.zeros(n)
    length[1:] = np.linalg.norm(np.diff(P, axis=0), axis=1)
    dev = np.zeros(n)
    for t in SAMPLES:
        pose = S[:-1] * (1.0 - t) + S[1:] * t
        dep = solver.to_part_batch(config, pose)
        ref = P[:-1] * (1.0 - t) + P[1:] * t
        dev[1:] = np.maximum(dev[1:], np.linalg.norm(dep - ref, axis=1))
    return extruding & (length > MIN_SEGMENT) & (dev > tol), dev


def _midpoint(a: CLDataWaypoint, b: CLDataWaypoint) -> CLDataWaypoint:
    nrm = np.array([a.i + b.i, a.j + b.j, a.k + b.k], float)
    norm = np.linalg.norm(nrm)
    i, j, k = (nrm / norm) if norm > 1e-9 else (b.i, b.j, b.k)
    return b.model_copy(update=dict(x=0.5 * (a.x + b.x), y=0.5 * (a.y + b.y), z=0.5 * (a.z + b.z),
                                    i=float(i), j=float(j), k=float(k),
                                    extrusion_volume=0.5 * b.extrusion_volume))


def solve_refined(solver: BaseKinematicSolver, trajectory: CLDataTrajectory, config: MachineConfig,
                  tolerance=None) -> MachineTrajectory:
    """solver.solve + the interpolation check / segment splitting described above (a no-op for 3-axis machines)."""
    tol = float(config.joint_tolerance if tolerance is None else tolerance)
    cols = int(getattr(solver, "pose_columns", 5))
    result = solver.solve(trajectory, config)
    if tol <= 0.0 or len(result.states) < 2 or solver.to_part_batch(config, np.zeros((1, 5))) is None:
        return result
    wps = list(trajectory.waypoints)
    original = len(wps)
    for it in range(MAX_ITERATIONS):
        S = _poses(result, cols)
        if len(S) != len(wps):
            logger.warning("Joint tolerance: the solver dropped points; the interpolation check was skipped")
            return result
        bad, dev = _bad_segments(solver, config, wps, S, tol)
        if not bad.any():
            break
        if len(wps) > GROWTH_LIMIT * original + 1000:
            logger.warning("Joint tolerance: stopped at %d points (limit); %d segments still deviate up to %.3f mm",
                           len(wps), int(bad.sum()), float(dev[bad].max()))
            break
        new: List[CLDataWaypoint] = [wps[0]]
        for i in range(1, len(wps)):
            if bad[i]:
                new.append(_midpoint(wps[i - 1], wps[i]))
                new.append(wps[i].model_copy(update=dict(extrusion_volume=0.5 * wps[i].extrusion_volume)))
            else:
                new.append(wps[i])
        wps = new
        result = solver.solve(CLDataTrajectory(waypoints=wps), config)
    else:
        S = _poses(result, cols)
        bad, dev = _bad_segments(solver, config, wps, S, tol) if len(S) == len(wps) else (np.zeros(1, bool), np.zeros(1))
        if bad.any():
            logger.warning("Joint tolerance: %d segments still deviate up to %.3f mm after %d splits",
                           int(bad.sum()), float(dev[bad].max()), MAX_ITERATIONS)
    if len(wps) != original:
        logger.info("Joint tolerance %.3f mm: %d -> %d points", tol, original, len(wps))
    return result
