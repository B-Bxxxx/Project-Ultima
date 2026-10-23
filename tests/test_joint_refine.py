"""The firmware interpolates every axis linearly between two G-code points: long segments with a changing C must be split."""
import numpy as np
import pytest

from src.common.schemas import CLDataTrajectory, CLDataWaypoint, MachineConfig
from src.stage3_kinematics.head_solver import Cartesian3AxisSolver, SwivelHeadXYZBCSolver
from src.stage3_kinematics.refine import solve_refined
from src.stage3_kinematics.trunnion_solver import TrunnionXYZBCSolver


def _cfg(**kw):
    base = dict(kinematic_chain="x", b_axis_min=-90, b_axis_max=90, c_axis_min=-36000, c_axis_max=36000)
    base.update(kw)
    return MachineConfig(**base)


def _cone_line(n=1):
    """Straight line across a cone (C = -azimuth): the ends are 25 mm apart, C goes 0 -> -180 deg via the centre area."""
    pts = [(-12.0, 3.0), (12.0, 3.0)]
    wps = []
    for k, (x, y) in enumerate(pts):
        r = np.hypot(x, y)
        tilt = np.radians(15)
        nx, ny = -np.sin(tilt) * x / r, -np.sin(tilt) * y / r           # normal tilts toward the centre
        wps.append(CLDataWaypoint(x=x, y=y, z=2.0, i=nx, j=ny, k=np.cos(tilt), feedrate=1500,
                                  is_travel_move=(k == 0), extrusion_volume=0.0 if k == 0 else 0.5))
    return CLDataTrajectory(waypoints=wps)


def _deposit_error(solver, cfg, wps, states):
    P = np.array([[w.x, w.y, w.z] for w in wps])
    S = np.array([[s.x, s.y, s.z, s.b, s.c] for s in states])
    worst = 0.0
    for i in range(1, len(wps)):
        if wps[i].is_travel_move:
            continue
        for t in np.linspace(0.1, 0.9, 9):
            d = solver.to_part_batch(cfg, (S[i - 1] * (1 - t) + S[i] * t)[None, :])[0]
            worst = max(worst, float(np.linalg.norm(d - (P[i - 1] * (1 - t) + P[i] * t))))
    return worst


@pytest.mark.parametrize("solver_cls", [TrunnionXYZBCSolver, SwivelHeadXYZBCSolver])
def test_inverse_matches_solve(solver_cls):
    cfg = _cfg(pivot_offset_x=3, pivot_offset_y=-2, pivot_offset_z=40, c_axis_x=110, c_axis_y=82, c_axis_z=5,
               eccentricity_x=0.3, eccentricity_y=-0.2, c_axis_dir=-1)
    s = solver_cls()
    traj = _cone_line()
    st = s.solve(traj, cfg).states
    for w, state in zip(traj.waypoints, st):
        inv = s.to_part_batch(cfg, [[state.x, state.y, state.z, state.b, state.c]])[0]
        assert inv == pytest.approx([w.x, w.y, w.z], abs=1e-6)


def test_long_segment_with_changing_c_is_split_inside_the_tolerance_and_keeps_the_volume():
    s, cfg, traj = TrunnionXYZBCSolver(), _cfg(joint_tolerance=0.02), _cone_line()
    raw = s.solve(traj, cfg)
    assert _deposit_error(s, cfg, traj.waypoints, raw.states) > 1.0               # the problem: an arc, not a line
    ref = solve_refined(s, traj, cfg)
    assert len(ref.states) > len(raw.states)
    # rebuild the refined waypoints from the states' CL data via the inverse: every sample is inside the tolerance
    P = s.to_part_batch(cfg, [[q.x, q.y, q.z, q.b, q.c] for q in ref.states])
    S = np.array([[q.x, q.y, q.z, q.b, q.c] for q in ref.states])
    line = lambda t: np.array([-12, 3, 2.0]) * (1 - t) + np.array([12, 3, 2.0]) * t
    for i in range(1, len(S)):
        for t in (0.25, 0.5, 0.75):
            d = s.to_part_batch(cfg, (S[i - 1] * (1 - t) + S[i] * t)[None, :])[0]
            a, b = P[i - 1], P[i]
            ref_pt = a * (1 - t) + b * t
            assert np.linalg.norm(d - ref_pt) <= 0.02 + 1e-6
    assert sum(q.extrusion_volume for q in ref.states) == pytest.approx(0.5)
    assert ref.states[-1].layer_index == raw.states[-1].layer_index
    assert [q.is_travel_move for q in ref.states].count(True) == 1               # no travel added


def test_off_and_non_rotary_are_unchanged():
    s, traj = TrunnionXYZBCSolver(), _cone_line()
    assert len(solve_refined(s, traj, _cfg(joint_tolerance=0.0)).states) == 2
    assert len(solve_refined(Cartesian3AxisSolver(), traj, _cfg()).states) == 2


def test_default_tolerance_applies_to_the_machine_config():
    assert MachineConfig(kinematic_chain="x", b_axis_min=0, b_axis_max=1, c_axis_min=0, c_axis_max=1).joint_tolerance == 0.02
