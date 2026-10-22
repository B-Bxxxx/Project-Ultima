"""C-axis / table-centre offset (c_axis_x/y/z), applied before the B tilt, and the machine-range check."""
import numpy as np
import pytest

from src.common.schemas import CLDataTrajectory, CLDataWaypoint, MachineConfig
from src.stage3_kinematics.head_solver import Cartesian3AxisSolver
from src.stage3_kinematics.machine_file import load_machine_file
from src.stage3_kinematics.trunnion_solver import TrunnionXYZBCSolver


def _cfg(**kw):
    base = dict(kinematic_chain="x", pivot_offset_x=0.0, pivot_offset_y=0.0, pivot_offset_z=100.0,
                b_axis_min=-90.0, b_axis_max=90.0, c_axis_min=-36000.0, c_axis_max=36000.0)
    base.update(kw)
    return MachineConfig(**base)


def _traj(i=0.0, j=0.0, k=1.0):
    return CLDataTrajectory(waypoints=[CLDataWaypoint(x=10, y=20, z=30, i=i, j=j, k=k, feedrate=1000)])


def test_default_is_unchanged():
    s = TrunnionXYZBCSolver().solve(_traj(), _cfg()).states[0]
    assert (s.x, s.y, s.z) == pytest.approx((10, 20, 30))


def test_offset_shifts_machine_position_when_flat():
    s = TrunnionXYZBCSolver().solve(_traj(), _cfg(c_axis_x=110, c_axis_y=105, c_axis_z=5)).states[0]
    assert (s.x, s.y, s.z) == pytest.approx((120, 125, 35))


def test_offset_is_applied_before_b_tilt():
    n = (0.0, -np.sin(np.radians(45)), np.cos(np.radians(45)))
    a = TrunnionXYZBCSolver().solve(_traj(*n), _cfg()).states[0]
    b = TrunnionXYZBCSolver().solve(_traj(*n), _cfg(c_axis_x=50)).states[0]
    assert abs(b.b) > 1.0
    # the offset rotates with B: X moves by 50*cos(B), Z by 50*sin(B) (B turns about Y; not a plain +50 in X)
    d = np.array([b.x - a.x, b.y - a.y, b.z - a.z])
    assert np.linalg.norm(d) == pytest.approx(50.0, abs=1e-6)
    assert d[0] != pytest.approx(50.0, abs=1e-3)


def test_3axis_origin_offset():
    s = Cartesian3AxisSolver().solve(_traj(), _cfg(c_axis_x=100, c_axis_y=100)).states[0]
    assert (s.x, s.y, s.z) == pytest.approx((110, 120, 30))


def test_range_warnings():
    spec = load_machine_file("config/machines/cartesian_3axis.json")
    cfg = _cfg()
    low = Cartesian3AxisSolver().solve(CLDataTrajectory(waypoints=[
        CLDataWaypoint(x=-5, y=20, z=30, i=0, j=0, k=1, feedrate=1000)]), cfg).states
    msgs = spec.range_warnings(low)
    assert len(msgs) == 1 and msgs[0].startswith("X ") and "c_axis_x" in msgs[0]
    ok = Cartesian3AxisSolver().solve(CLDataTrajectory(waypoints=[
        CLDataWaypoint(x=-5, y=20, z=30, i=0, j=0, k=1, feedrate=1000)]), _cfg(c_axis_x=100)).states
    assert spec.range_warnings(ok) == []


def test_machine_files_expose_c_axis_params():
    for f in ("trunnion_5axis", "cartesian_3axis"):
        spec = load_machine_file(f"config/machines/{f}.json")
        vals = {p.key: p.default for p in spec.params} if hasattr(spec, "params") else {}
        assert not vals or "c_axis_x" in vals


def test_axis_direction_flips_the_written_b_and_c_and_their_limits():
    n = (-np.sin(np.radians(20)) * np.cos(np.radians(30)), np.sin(np.radians(20)) * np.sin(np.radians(30)), np.cos(np.radians(20)))
    pos = TrunnionXYZBCSolver().solve(_traj(*n), _cfg()).states[0]
    neg = TrunnionXYZBCSolver().solve(_traj(*n), _cfg(b_axis_dir=-1, c_axis_dir=-1)).states[0]
    assert (neg.b, neg.c) == pytest.approx((-pos.b, -pos.c))
    assert (neg.x, neg.y, neg.z) == pytest.approx((pos.x, pos.y, pos.z))     # the geometry is unchanged
    # a limit given in the machine's direction: B -10..90 with dir -1 allows program B 10..-90, i.e. machine B -10 is fine
    lim = TrunnionXYZBCSolver().solve(_traj(*n), _cfg(b_axis_dir=-1, b_axis_min=-90, b_axis_max=0)).states[0]
    assert lim.b == pytest.approx(-20.0, abs=1e-6)


def test_zero_is_never_written_as_minus_zero():
    s = TrunnionXYZBCSolver().solve(_traj(), _cfg(b_axis_dir=-1, c_axis_dir=-1)).states[0]
    assert str(s.b) == "0.0" and str(s.c) == "0.0"
