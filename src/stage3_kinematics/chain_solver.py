"""
`generic_chain`: a kinematic solver driven by a mathematical chain description (see chain.py) instead of hard-coded
formulas. It runs PARALLEL to the hand-written solvers (trunnion_table_xyzbc, swivel_head_xyzbc, planar_3axis_xyz),
which stay untouched; the parity tests prove that a chain description of those machines gives the same G-code.

Algorithm per waypoint (all vectorised):
  1. orientation: closed-form inverse kinematics of up to two rotary joints (0: nozzle fixed, 1: best fit, 2: exact,
     two branches), for ANY axis directions / order / table-or-tool placement;
  2. branch + limits: the branch inside the joint limits wins (tie: positive angle of the limited joint); a `continuous`
     joint is unwound with base.resolve_c_angles (continuous inside a move chain, re-chosen near zero at every travel);
  3. position: the X, Y, Z linear joints are solved from the linear equation  tool(tip) = table(P + part_origin).
"""
import logging

import numpy as np

from src.common.schemas import CLDataTrajectory, MachineConfig, MachineStateVector, MachineTrajectory
from src.stage3_kinematics.base import BaseKinematicSolver, resolve_branches, resolve_c_angles, tool_alignment
from src.stage3_kinematics.chain import Chain, LINEAR_NAMES, angle_about, branch_transform, two_planes
from src.stage3_kinematics.registry import KinematicsRegistry

SINGULAR_TILT_RAD = 1e-4          # same threshold as the hand-written solvers: below it the azimuth is undefined
BIG = 1e6


def _wrap(deg):
    return (np.asarray(deg, float) + 180.0) % 360.0 - 180.0


def _fit_to_limits(deg, lo, hi):
    """Equivalent angle (deg + 360 k) inside [lo, hi] if one exists, else the one closest to the interval. Returns (angle, violation)."""
    deg = _wrap(deg)
    best, best_v = deg.copy(), None
    for k in (-2, -1, 0, 1, 2):
        c = deg + 360.0 * k
        v = np.maximum(lo - c, 0.0) + np.maximum(c - hi, 0.0)
        if best_v is None:
            best, best_v = c, v
        else:
            better = (v < best_v - 1e-9) | ((np.abs(v - best_v) <= 1e-9) & (np.abs(c) < np.abs(best)))
            best, best_v = np.where(better, c, best), np.where(better, v, best_v)
    return best, best_v


@KinematicsRegistry.register("generic_chain")
class GenericChainSolver(BaseKinematicSolver):
    pose_columns = 6                  # refine passes [x, y, z, b, c, a] (the hand-written solvers get the 5-column pose)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _chain(config: MachineConfig) -> Chain:
        if not config.chain:
            raise ValueError('kinematics "generic_chain" needs a chain description (MachineConfig.chain / "chain" in the machine file)')
        return Chain.parse(config.chain, {}, "")

    @staticmethod
    def _candidates(ch: Chain, N: np.ndarray):
        """Candidate rotary solutions: a list of {axis: math angle (rad, (n,))}."""
        n, e = len(N), ch.tool_axis
        E = np.broadcast_to(e, (n, 3))
        rt = [j for j in ch.tool if j.rotary]
        rb = [j for j in ch.table if j.rotary]
        allr = rt + rb
        if not allr:
            return [{}]
        if len(allr) == 1:
            j = allr[0]
            th = angle_about(j.direction, N, E) if rb else angle_about(j.direction, E, N)
            return [{j.axis: th}]
        if rt and rb:                                           # one on each side:  R_a e = R_b N = v
            a, b = rt[0], rb[0]
            sols = two_planes(a.direction, b.direction, np.full(n, a.direction @ e), N @ b.direction)
            return [{a.axis: angle_about(a.direction, E, v), b.axis: angle_about(b.direction, N, v)} for v in sols]
        j1, j2 = allr                                           # j1 outer, j2 inner, same branch
        if rb:                                                  # R1 R2 N = e :  v = R2 N = R1^-1 e
            sols = two_planes(j1.direction, j2.direction, np.full(n, j1.direction @ e), N @ j2.direction)
            return [{j2.axis: angle_about(j2.direction, N, v), j1.axis: angle_about(j1.direction, v, E)} for v in sols]
        sols = two_planes(j1.direction, j2.direction, N @ j1.direction, np.full(n, j2.direction @ e))   # R1 R2 e = N : v = R2 e = R1^-1 N
        return [{j2.axis: angle_about(j2.direction, E, v), j1.axis: angle_about(j1.direction, v, N)} for v in sols]

    dead_zone_deg = 0.0

    def _angles(self, ch: Chain, N: np.ndarray, waypoints):
        """Final MATHEMATICAL rotary angles (deg), {axis: (n,)}: branch choice, continuity, limits."""
        rot = ch.rotaries()
        if not rot:
            return {}
        n = len(N)
        cands = self._candidates(ch, N)
        free = [j for j in rot if not j.continuous]
        lim = {j.axis: j.math_limits() for j in rot}

        def score(c):
            viol, pref = np.zeros(n), np.zeros(n)
            for j in free:
                _, v = _fit_to_limits(np.degrees(c[j.axis]), *lim[j.axis])
                viol += v
            for j in (free or rot[:1]):
                pref += _wrap(np.degrees(c[j.axis]))
            return viol, pref

        sc = [score(c) for c in cands]
        # per-point best candidate: smallest limit violation, then the larger preferred (positive) angle
        viol = np.stack([s[0] for s in sc])
        pref = np.stack([s[1] for s in sc])
        best_v = viol.min(axis=0)
        ok = viol <= best_v + 1e-9
        pick = np.argmax(np.where(ok, pref, -np.inf), axis=0)
        idx = np.arange(n)
        ang = {j.axis: np.degrees(np.stack([c[j.axis] for c in cands])[pick, idx]) for j in rot}

        tilt = None
        if free and any(j.continuous for j in rot):
            tilt = np.abs(_wrap(ang[free[0].axis]))
        e_dir = ch.tool_axis
        out = {}
        az = [j for j in rot if j.continuous and abs(abs(float(j.direction @ e_dir)) - 1.0) < 1e-9]
        if len(rot) == 2 and len(free) == 1 and len(az) == 1:
            # tilt joint + azimuth joint (trunnion / swivel head): branch by continuity of the azimuth, vertical dead zone
            jt, jc = free[0], az[0]
            (tlo, thi), (clo, chi) = lim[jt.axis], lim[jc.axis]
            t, c = _wrap(ang[jt.axis]), ang[jc.axis]
            tpos, ccan = np.abs(t), np.where(t < 0, c + 180.0, c)         # canonical form: tilt >= 0
            raw = np.radians(ccan)
            sing = np.radians(tpos) < SINGULAR_TILT_RAD
            if sing.any():
                last = np.maximum.accumulate(np.where(~sing, np.arange(n), -1))
                raw = np.where(last >= 0, raw[np.maximum(last, 0)], 0.0)
            tb, cb = resolve_branches(raw, tpos, waypoints, tlo, thi, clo if np.isfinite(clo) else -BIG,
                                      chi if np.isfinite(chi) else BIG, self.dead_zone_deg)
            out[jt.axis] = np.clip(np.array(tb), tlo, thi)
            out[jc.axis] = np.clip(np.array(cb), clo, chi)
            return out
        for j in rot:
            lo, hi = lim[j.axis]
            if not j.continuous:
                a, _ = _fit_to_limits(ang[j.axis], lo, hi)
                out[j.axis] = np.clip(a, lo, hi)
                if np.any(np.abs(out[j.axis] - a) > 1e-6):
                    logging.getLogger(__name__).warning("%s axis: the layer normal needs more than the limits allow (%g..%g deg); "
                                                        "the angle was clamped, the nozzle is not along the surface normal there", j.axis, *lim[j.axis])
            else:
                raw = np.radians(ang[j.axis])
                azimuth = tilt is not None and abs(abs(float(j.direction @ e_dir)) - 1.0) < 1e-9     # turns about the nozzle axis
                if azimuth:                                     # azimuth undefined at (almost) zero tilt: keep the previous value
                    sing = np.radians(tilt) < SINGULAR_TILT_RAD
                    if sing.any():
                        last = np.maximum.accumulate(np.where(~sing, np.arange(n), -1))
                        raw = np.where(last >= 0, raw[np.maximum(last, 0)], 0.0)
                hold = tilt if azimuth else np.full(n, 90.0)
                c = np.array(resolve_c_angles(raw, hold, waypoints, lo if np.isfinite(lo) else -BIG, hi if np.isfinite(hi) else BIG))
                out[j.axis] = np.clip(c, lo, hi)
        return out

    @staticmethod
    def _linear_solve(ch: Chain, rot_rad: dict, P: np.ndarray):
        """Math values (mm) of X, Y, Z that put the nozzle tip on the part point. Returns {axis: (n,)} and the rotary-only rotations."""
        n = len(P)
        Pe = P + ch.part_origin[None]

        def f(lin):
            vals = {**rot_rad, **lin}
            Rt, tt = branch_transform(ch.tool, vals, n)
            Rb, tb = branch_transform(ch.table, vals, n)
            return np.einsum("nij,j->ni", Rt, ch.tip) + tt - np.einsum("nij,nj->ni", Rb, Pe) - tb

        zero = {a: np.zeros(n) for a in LINEAR_NAMES}
        f0 = f(zero)
        A = np.stack([f({**zero, a: np.ones(n)}) - f0 for a in LINEAR_NAMES], axis=2)       # (n,3,3)
        s = np.linalg.solve(A, -f0[:, :, None])[:, :, 0]
        return {a: s[:, k] for k, a in enumerate(LINEAR_NAMES)}

    # ------------------------------------------------------------------ solver interface
    def solve(self, trajectory: CLDataTrajectory, config: MachineConfig) -> MachineTrajectory:
        ch = self._chain(config)
        wps = list(trajectory.waypoints)
        n = len(wps)
        if n == 0:
            return MachineTrajectory(states=[])
        N = np.array([[w.i, w.j, w.k] for w in wps], float)
        nn = np.linalg.norm(N, axis=1)
        N = np.where((nn < 1e-6)[:, None], np.array([0.0, 0.0, 1.0]), N / np.maximum(nn, 1e-12)[:, None])
        P = np.array([[w.x, w.y, w.z] for w in wps], float)

        self.dead_zone_deg = float(config.vertical_dead_zone_deg)
        ang_deg = self._angles(ch, N, wps)
        rot_rad = {a: np.radians(v) for a, v in ang_deg.items()}
        lin = self._linear_solve(ch, rot_rad, P)

        # the nozzle axis really reached, in the part frame: R_table^T R_tool tool_axis
        Rt, _ = branch_transform(ch.tool, rot_rad, n)
        Rb, _ = branch_transform(ch.table, rot_rad, n)
        axis_part = np.einsum("nji,nj->ni", Rb, np.einsum("nij,j->ni", Rt, ch.tool_axis))

        sign = {j.axis: j.sign for j in ch.joints}
        states = []
        for k, wp in enumerate(wps):
            def mach(a):
                v = (lin[a][k] if a in lin else np.degrees(rot_rad[a][k]) if a in rot_rad else 0.0)
                return float(v * sign.get(a, 1.0)) + 0.0
            states.append(MachineStateVector(
                x=mach("X"), y=mach("Y"), z=mach("Z"), a=mach("A"), b=mach("B"), c=mach("C"),
                tool_alignment=tool_alignment(wp, axis_part[k]),
                extrusion_volume=wp.extrusion_volume, feedrate=wp.feedrate, is_travel_move=wp.is_travel_move,
                feature_type=wp.feature_type, layer_index=wp.layer_index))
        return MachineTrajectory(states=states)

    def to_part_batch(self, config: MachineConfig, poses):
        """Inverse of solve(): where on the PART the nozzle tip is for the machine poses (n,5) = [x, y, z, b, c]."""
        ch = self._chain(config)
        if not ch.rotaries():
            return None
        S = np.asarray(poses, float)
        S = S.reshape(-1, S.shape[-1])
        n = len(S)
        col = {"X": 0, "Y": 1, "Z": 2, "B": 3, "C": 4, "A": 5}      # a 5-column pose has no A (= 0)
        vals = {}
        for j in ch.joints:
            v = (S[:, col[j.axis]] if col[j.axis] < S.shape[1] else np.zeros(n)) * j.sign
            vals[j.axis] = np.radians(v) if j.rotary else v
        Rt, tt = branch_transform(ch.tool, vals, n)
        Rb, tb = branch_transform(ch.table, vals, n)
        tip = np.einsum("nij,j->ni", Rt, ch.tip) + tt
        return np.einsum("nji,nj->ni", Rb, tip - tb) - ch.part_origin[None]
