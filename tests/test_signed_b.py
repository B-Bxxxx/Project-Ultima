"""Ridge crossings: signed B through zero instead of a 180 deg C flip, the vertical dead zone, and the gradual-C fallback."""
import os

import numpy as np
import pytest

from src.common.gcode_check import parse_gcode, reconstruct
from src.common.schemas import CLDataTrajectory, MinimalPrintProfile
from src.stage3_kinematics.machine_file import load_machine_file
from src.stage3_kinematics.refine import solve_refined
from src.stage3_kinematics.registry import KinematicsRegistry
from src.stage4_postproc.klipper_postproc import Klipper5AxisPostProcessor
from tests.test_generic_chain import EXPERIMENTAL, wp

MACHINES = os.path.join(os.path.dirname(__file__), "..", "config", "machines")
FILES = {"trunnion": os.path.join(MACHINES, "trunnion_5axis.json"),
         "chain": os.path.join(EXPERIMENTAL, "nested_trunnion_5axis.json"),
         "head": os.path.join(EXPERIMENTAL, "nested_swivel_head_5axis.json")}


def setup(kind, **over):
    spec = load_machine_file(FILES[kind])
    over = {"c_axis_x": 60.0, "c_axis_y": 40.0, "pivot_offset_z": 30.0, "vertical_dead_zone_deg": 0.0, **over}
    cfg = spec.to_machine_config({k: v for k, v in over.items() if k in spec.parameters})
    return KinematicsRegistry.get_solver(spec.kinematics)(), cfg


def ridge(tilt_from=-15.0, tilt_to=15.0, n=41, y=3.0):
    """A straight extruding line; the layer normal tilts from one side to the other through the vertical (a ridge crest)."""
    w = [wp(-10, y, 5, [0, 0, 1], travel=True)]
    for i, th in enumerate(np.radians(np.linspace(tilt_from, tilt_to, n))):
        w.append(wp(-10 + 20 * i / (n - 1), y, 5, [np.sin(th), 0, np.cos(th)]))
    return CLDataTrajectory(waypoints=w)


@pytest.mark.parametrize("kind", ["trunnion", "chain", "head"])
def test_ridge_crossing_passes_through_zero_without_c_flip(kind):
    solver, cfg = setup(kind)
    res = solver.solve(ridge(), cfg)
    b = np.array([s.b for s in res.states])
    c = np.array([s.c for s in res.states])
    assert np.abs(np.diff(c)[1:]).max() < 1.0                       # C stays (the old solvers jumped by 180 deg)
    assert b.min() < -5 and b.max() > 5                              # B really goes through zero
    assert all(abs(s.tool_alignment - 1) < 1e-9 for s in res.states)


@pytest.mark.parametrize("kind", ["trunnion", "chain", "head"])
def test_one_sided_b_range_keeps_the_old_flip(kind):
    solver, cfg = setup(kind, b_axis_min=0.0, b_axis_max=90.0)
    res = solver.solve(ridge(), cfg)
    c = np.array([s.c for s in res.states])
    assert np.abs(np.diff(c)).max() > 170                           # no negative B available: C has to turn


@pytest.mark.parametrize("kind", ["trunnion", "chain", "head"])
def test_dead_zone_keeps_the_nozzle_vertical_near_the_crest(kind):
    tr = ridge(-1.5, 1.5)
    solver, cfg = setup(kind, vertical_dead_zone_deg=2.0)
    res = solver.solve(tr, cfg)
    assert all(abs(s.b) < 1e-9 for s in res.states) and np.ptp([s.c for s in res.states]) < 1e-6
    assert min(s.tool_alignment for s in res.states) >= np.cos(np.radians(1.5)) - 1e-9
    solver, cfg = setup(kind)                                        # off: the nozzle follows the normal
    assert max(abs(s.b) for s in solver.solve(tr, cfg).states) > 1.0


@pytest.mark.parametrize("kind", ["trunnion", "chain"])
def test_remaining_flips_are_spread_out_and_the_line_stays_straight(kind):
    """With a one-sided B range the C turn cannot be avoided: it is spread over steps while the tip stays on the line."""
    solver, cfg = setup(kind, b_axis_min=0.0, b_axis_max=90.0)
    tr = ridge()
    res = solve_refined(solver, tr, cfg)
    c = np.array([s.c for s in res.states])
    assert np.abs(np.diff(c)).max() <= 6.0                           # gradual
    text = Klipper5AxisPostProcessor().generate_gcode(res, MinimalPrintProfile())
    rc = reconstruct(parse_gcode(text)[0], solver, cfg)
    feed = np.array([m.kind == "feed" and m.feature != "travel" for m in rc.moves])
    assert rc.deviation[feed].max() < 0.06                           # the firmware draws the straight line
    assert sum(s.extrusion_volume for s in res.states) == pytest.approx(sum(w.extrusion_volume for w in tr.waypoints))


def test_extrusion_is_not_lost_when_a_segment_is_unwound():
    solver, cfg = setup("trunnion", b_axis_min=0.0, b_axis_max=90.0)
    tr = ridge()
    res = solve_refined(solver, tr, cfg)
    assert len(res.states) > len(tr.waypoints)
