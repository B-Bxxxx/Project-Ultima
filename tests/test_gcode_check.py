"""The G-code checker: parser, firmware-style reconstruction, deviation detection, GUI tab."""
import os

import numpy as np
import pytest

from src.common.gcode_check import parse_gcode, reconstruct
from src.common.schemas import MinimalPrintProfile
from src.stage3_kinematics.machine_file import load_machine_file
from src.stage3_kinematics.refine import solve_refined
from src.stage3_kinematics.registry import KinematicsRegistry
from src.stage4_postproc.klipper_postproc import Klipper5AxisPostProcessor
from tests.test_generic_chain import EXPERIMENTAL, trajectory

MACHINES = os.path.join(os.path.dirname(__file__), "..", "config", "machines")


def machine(path):
    spec = load_machine_file(path)
    return KinematicsRegistry.get_solver(spec.kinematics)(), spec.to_machine_config(), spec


TRUNNION = os.path.join(MACHINES, "trunnion_5axis.json")


# ------------------------------------------------------------- parser
def test_parser_markers_relative_mode_and_skips():
    text = """;LAYER:3
; TYPE: outer_wall
G90
G0 X1 Y2 Z3 F3000
G1 X2 Y2 Z3 B10 C20 E0.5 F1200
G1 E-1.0 F2400 ; retract (no motion)
G91
G1 X1 C-30 E0.2
G90
G2 X5 Y5 I1 J1
G1 X9 F600
"""
    moves, warns = parse_gcode(text)
    assert [m.kind for m in moves] == ["rapid", "feed", "feed", "feed"]
    assert moves[1].layer == 3 and moves[1].feature == "outer_wall" and moves[0].feature == "travel"
    np.testing.assert_allclose(moves[2].p1, [3, 2, 3, 0, 10, -10])         # relative move: X 2+1, C 20-30
    assert moves[3].feed == 600 and moves[3].feature == "travel"           # G1 without E is a travel
    assert any("G2" in w for w in warns)


# ------------------------------------------------------------- firmware-style interpolation
def test_half_circle_is_found():
    """C turns 0 -> -180 while B ~ 0 and the nozzle is 10 mm from the C axis: the nozzle draws a half circle."""
    solver, cfg, _ = machine(TRUNNION)
    moves, _ = parse_gcode("G1 X10 Y0 Z0 B0.3 C0 F1500\nG1 X10 Y0 Z0 B0.3 C-180 E0.1 F1500\n")
    rc = reconstruct(moves, solver, cfg)
    assert 9.8 < rc.deviation[1] < 10.2 and rc.c_swing[1] == pytest.approx(180)
    assert rc.bad_moves(0.05)[1]


def test_c_takes_the_shortest_way():
    solver, cfg, _ = machine(TRUNNION)
    moves, _ = parse_gcode("G1 X0 Y0 Z0 B0 C350 F1000\nG1 X0 Y0 Z0 B0 C10 E0.1 F1000\n")
    rc = reconstruct(moves, solver, cfg)
    assert rc.c_swing[1] == pytest.approx(20.0)                           # +20, not -340
    # and the firmware's own position goes on unwrapped: the next move starts at 370
    moves, _ = parse_gcode("G1 C350 F1000\nG1 C10 E0.1\nG1 C30 E0.1\n")
    rc = reconstruct(moves, solver, cfg)
    assert rc.c_swing[2] == pytest.approx(20.0)


def test_time_uses_the_six_axis_length():
    solver, cfg, _ = machine(TRUNNION)
    moves, _ = parse_gcode("G1 X1 F600\nG1 X31 C40 E0.1 F600\n")
    rc = reconstruct(moves, solver, cfg)
    assert rc.time_s[1] == pytest.approx(np.hypot(30, 40) / 10.0)         # 50 "mm" at 10 mm/s
    assert rc.stats["time_min_6d"] > rc.stats["time_min_xyz"]


def test_non_rotary_machine_warns_about_rotary_words():
    solver, cfg, _ = machine(os.path.join(MACHINES, "cartesian_3axis.json"))
    moves, _ = parse_gcode("G1 X1 B10 F1000\nG1 X2 B20 E0.1\n")
    rc = reconstruct(moves, solver, cfg)
    assert any("no rotary axes" in w for w in rc.warnings)


# ------------------------------------------------------------- agreement with the pipeline
@pytest.mark.parametrize("path,rotary", [(TRUNNION, "BC"),
                                         (os.path.join(EXPERIMENTAL, "nested_trunnion_5axis.json"), "BC"),
                                         (os.path.join(EXPERIMENTAL, "nested_swivel_head_5axis.json"), "BC"),
                                         (os.path.join(EXPERIMENTAL, "nested_head_a_table_b.json"), "AB")])
def test_refined_gcode_has_no_deviation_but_unrefined_has(path, rotary):
    spec = load_machine_file(path)
    solver = KinematicsRegistry.get_solver(spec.kinematics)()
    over = {k: v for k, v in dict(c_axis_x=60.0, c_axis_y=40.0, pivot_offset_z=40.0).items() if k in spec.parameters}
    cfg = spec.to_machine_config(over)
    tr = trajectory()
    prof = MinimalPrintProfile(rotary_axes=rotary)
    good = Klipper5AxisPostProcessor().generate_gcode(solve_refined(solver, tr, cfg, tolerance=0.02), prof)
    rc = reconstruct(parse_gcode(good)[0], solver, cfg)
    feed = np.array([m.kind == "feed" and m.feature != "travel" for m in rc.moves])
    assert rc.deviation[feed].max() < 0.06                                # 0.02 tolerance + the 3-decimal rounding of the G-code
    raw = Klipper5AxisPostProcessor().generate_gcode(solver.solve(tr, cfg), prof)
    rc2 = reconstruct(parse_gcode(raw)[0], solver, cfg)
    feed2 = np.array([m.kind == "feed" and m.feature != "travel" for m in rc2.moves])
    assert rc2.deviation[feed2].max() > 0.2                               # the checker sees what refine repairs


# ------------------------------------------------------------- GUI tab
def test_check_tab_loads_and_draws():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from src.gui_check import CheckTab
    solver, cfg, spec = machine(TRUNNION)
    tab = CheckTab(lambda: (solver, cfg, spec), lambda: "G1 X1 F600\nG1 X10 B30 C90 E0.1 F600\nG1 X20 B30 C180 E0.1\n")
    tab.check_generated()
    assert tab.rc is not None and "moves" in tab.txt.toPlainText()
    tab.cmb_frame.setCurrentIndex(1)
    a, b = tab.sample_range(0, tab.layer_total() - 1)
    tab.render(0, tab.layer_total() - 1, a, b - a, {"outer_wall", "infill", "travel"})
    app.processEvents()
    assert tab._items or tab._base[1]


def test_table_map_is_the_inverse_of_the_part_map():
    """machine = M part + t for any pose (trunnion with an offset B, a swivel head chain too)."""
    import os
    from src.common.gcode_check import table_map
    from src.stage3_kinematics.machine_file import load_machine_file
    from src.stage3_kinematics.registry import KinematicsRegistry
    root = os.path.join(os.path.dirname(__file__), "..", "config")
    for f, kin in (("machines/trunnion_5axis.json", "trunnion_table_xyzbc"), ("machines_experimental/chain_trunnion_5axis.json", "generic_chain")):
        spec = load_machine_file(os.path.join(root, f))
        cfg = spec.to_machine_config({"c_axis_x": 175.0, "c_axis_y": 82.0, "b_axis_offset_deg": 90.0, "b_axis_min": 0.0, "b_axis_max": 180.0,
                                      "pivot_offset_z": 12.0})
        sol = KinematicsRegistry.get_solver(kin)()
        pose = np.array([120.0, 70.0, 30.0, 0.0, 130.0, 217.0])
        M, t = table_map(sol, cfg, pose)
        assert abs(np.linalg.det(M) - 1.0) < 1e-9
        part = np.array([[3.0, -4.0, 5.0], [10.0, 20.0, 0.0]])
        mach = part @ M.T + t
        sel = [0, 1, 2, 4, 5]
        back = sol.to_part_batch(cfg, np.column_stack([mach, np.tile(pose[[4, 5]], (2, 1))]))
        np.testing.assert_allclose(back, part, atol=1e-8)
        tip = sol.to_part_batch(cfg, pose[sel].reshape(1, 5))[0]
        np.testing.assert_allclose(M @ tip + t, pose[:3], atol=1e-8)
