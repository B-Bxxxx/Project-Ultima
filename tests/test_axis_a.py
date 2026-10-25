"""The A axis (rotation about X) in the generic_chain solver, the G-code letters and the 4/5-axis combinations."""
import numpy as np
import pytest

from src.common.schemas import CLDataTrajectory, MachineConfig, MinimalPrintProfile
from src.stage3_kinematics.chain import Chain, ChainError
from src.stage3_kinematics.refine import solve_refined
from src.stage3_kinematics.registry import KinematicsRegistry
from src.stage4_postproc.klipper_postproc import Klipper5AxisPostProcessor as KlipperPostProcessor
from tests.test_generic_chain import trajectory, wp

LIN = [{"axis": a, "type": "linear", "direction": v} for a, v in (("X", [1, 0, 0]), ("Y", [0, 1, 0]), ("Z", [0, 0, 1]))]


def cfg(tool=(), table=(), **kw):
    chain = {"tool": LIN + list(tool), "table": list(table), "part_origin": kw.pop("part_origin", [0, 0, 0])}
    return MachineConfig(kinematic_chain="generic_chain", b_axis_min=0, b_axis_max=0, c_axis_min=0, c_axis_max=0,
                         chain=Chain.parse(chain).to_dict(), **kw)


def solve(c, tr=None):
    return KinematicsRegistry.get_solver("generic_chain")().solve(tr or trajectory(), c)


def P(tr):
    return np.array([[w.x, w.y, w.z] for w in tr.waypoints])


A_TILT = {"axis": "A", "type": "rotary", "origin": [10, -20], "min": -90, "max": 90}
C_TABLE = {"axis": "C", "type": "rotary", "origin": [100, 80], "min": -1000, "max": 1000}
B_TILT = {"axis": "B", "type": "rotary", "origin": [0, 30], "min": -90, "max": 90}


@pytest.mark.parametrize("name,tool,table", [
    ("table A carries C", [], [A_TILT, C_TABLE]),
    ("table B carries A", [], [B_TILT, {**A_TILT, "min": -180, "max": 180}]),
    ("head A,B (C carries nothing)", [A_TILT, {**B_TILT, "min": -180, "max": 180}], []),
    ("head C carries B", [{**C_TABLE, "origin": [0, 0]}, B_TILT], []),
    ("mixed: A on the head, C on the table", [A_TILT], [C_TABLE]),
])
def test_two_rotary_axes_hit_the_normal_and_the_point(name, tool, table):
    c = cfg(tool, table)
    tr = trajectory()
    res = solve(c, tr)
    ok = [abs(s.tool_alignment - 1) < 1e-7 for s in res.states]
    assert all(ok), name
    reach = np.array(ok)
    S = np.array([[s.x, s.y, s.z, s.b, s.c, s.a] for s in res.states])
    inv = KinematicsRegistry.get_solver("generic_chain")().to_part_batch(c, S)
    np.testing.assert_allclose(inv[reach], P(tr)[reach], atol=1e-6)


def test_a_letter_order_and_values():
    c = cfg(table=[A_TILT, C_TABLE])
    tr = trajectory()
    res = solve(c, tr)
    assert max(abs(s.a) for s in res.states) > 5 and max(abs(s.b) for s in res.states) == 0


def test_gcode_letters_follow_the_machine():
    c = cfg(table=[A_TILT, C_TABLE])
    tr = trajectory()
    res = solve_refined(KinematicsRegistry.get_solver("generic_chain")(), tr, c)
    prof = MinimalPrintProfile(rotary_axes="AC")
    text = KlipperPostProcessor().generate_gcode(res, prof)
    body = [l for l in text.splitlines() if l.startswith("G1 X")]
    assert body and all(" A" in l and " C" in l and " B" not in l for l in body)
    assert "A0 C0" in text
    # unchanged for the classic BC machines
    prof2 = MinimalPrintProfile()
    text2 = KlipperPostProcessor().generate_gcode(res, prof2)
    assert any(" B" in l and " C" in l and " A" not in l for l in text2.splitlines() if l.startswith("G1 X")) and "B0 C0" in text2


def test_refine_works_with_six_pose_columns():
    c = cfg(table=[A_TILT, {**C_TABLE}], joint_tolerance=0.02)
    tr = trajectory()
    plain = solve(c, tr)
    refined = solve_refined(KinematicsRegistry.get_solver("generic_chain")(), tr, c)
    assert len(refined.states) >= len(plain.states)
    S = np.array([[s.x, s.y, s.z, s.b, s.c, s.a] for s in refined.states])
    assert S.shape[1] == 6


# ------------------------------------------------------------- 4 axis (one rotary): extrusion correction
def test_four_axis_nozzle_is_not_along_the_normal_and_extrusion_is_corrected():
    c = cfg(table=[{"axis": "A", "type": "rotary", "origin": [0, 0], "min": -90, "max": 90}])
    tr = CLDataTrajectory(waypoints=[wp(0, 0, 0, [0, 0, 1]), wp(1, 0, 0, [0, np.sin(0.4), np.cos(0.4)]),      # reachable (about X)
                                     wp(2, 0, 0, [np.sin(0.5), 0, np.cos(0.5)])])                                # tilt about Y: not reachable
    res = solve(c, tr)
    al = [s.tool_alignment for s in res.states]
    assert al[0] == pytest.approx(1.0) and al[1] == pytest.approx(1.0, abs=1e-9) and al[2] == pytest.approx(np.cos(0.5), abs=1e-9)
    prof = MinimalPrintProfile(rotary_axes="A", thickness_model="nozzle_axis")
    text = KlipperPostProcessor().generate_gcode(res, prof)
    e = [float(l.split(" E")[1].split()[0]) for l in text.splitlines() if l.startswith("G1 X")]
    assert e[2] / e[1] == pytest.approx(1 / np.cos(0.5), rel=1e-3)               # volume / alignment of the real nozzle axis
    assert all(" A" in l and " B" not in l and " C" not in l for l in text.splitlines() if l.startswith("G1 X"))


# ------------------------------------------------------------- range and winding
def test_continuous_follows_the_range():
    j = lambda lo, hi, **kw: Chain.parse({"tool": LIN, "table": [{"axis": "C", "type": "rotary", "origin": [0, 0], "min": lo, "max": hi, **kw}]}).table[0]
    assert j(-360, 360).continuous and j(0, 360).continuous
    assert not j(-90, 90).continuous and not j(-180, 179).continuous
    assert j(-90, 90, continuous=True).continuous                                    # explicit wins
    assert Chain.parse({"tool": LIN, "table": [{"axis": "C", "type": "rotary", "origin": [0, 0]}]}).table[0].continuous     # no limits


def test_limited_c_does_not_wind_around():
    """A C range of -90..90 cannot follow a spiral: it is clamped (warning), a full range winds and unwinds."""
    tr = trajectory()
    full = solve(cfg(table=[{**B_TILT}, {**C_TABLE, "min": -360, "max": 360}]), tr)
    lim = solve(cfg(table=[{**B_TILT}, {**C_TABLE, "min": -90, "max": 90}]), tr)
    assert max(abs(s.c) for s in full.states) > 150
    assert max(abs(s.c) for s in lim.states) <= 90 + 1e-9


# ------------------------------------------------------------- the rules
@pytest.mark.parametrize("bad", [
    {"axis": "A", "type": "rotary", "direction": [0, 1, 0]},                 # A turns about X
    {"axis": "B", "type": "rotary", "origin": [1, 2, 3]},                    # origin is 2 numbers
    {"axis": "B", "type": "rotary", "origin": [1, 2], "point": [0, 0, 0]},
    {"axis": "X", "type": "rotary"},
])
def test_axis_rules(bad):
    with pytest.raises(ChainError):
        Chain.parse({"tool": LIN, "table": [bad]})


def test_third_rotary_is_refused():
    with pytest.raises(ChainError, match="6 axis"):
        Chain.parse({"tool": LIN, "table": [A_TILT, B_TILT, C_TABLE]})


def test_origin_is_two_coordinates():
    j = Chain.parse({"tool": LIN, "table": [{"axis": "A", "type": "rotary", "origin": [7, 9]}]}).table[0]
    assert list(j.point) == [0, 7, 9]                      # A turns about X: y and z locate the axis
    j = Chain.parse({"tool": LIN, "table": [{"axis": "B", "type": "rotary", "origin": [7, 9]}]}).table[0]
    assert list(j.point) == [7, 0, 9]
    j = Chain.parse({"tool": LIN, "table": [{"axis": "C", "type": "rotary", "origin": [7, 9]}]}).table[0]
    assert list(j.point) == [7, 9, 0]


def test_c_outside_a_cannot_tilt_the_nozzle():
    """C (vertical axis) carrying A keeps the z component of the normal: only normals with N_x = 0 are reachable; no crash, the
    alignment honestly says how far the nozzle is from the normal."""
    c = cfg(table=[C_TABLE, {**A_TILT, "min": -180, "max": 180}])
    res = solve(c, CLDataTrajectory(waypoints=[wp(0, 0, 0, [0, 0.5, 1]), wp(1, 0, 0, [0.5, 0, 1])]))
    assert res.states[0].tool_alignment == pytest.approx(1.0, abs=1e-9)
    assert res.states[1].tool_alignment < 0.999
