"""C axis: continuous inside a move chain, re-chosen closest to zero at every new chain; held at tiny tilt."""
import logging
import numpy as np
import pytest

from src.common.schemas import CLDataTrajectory, CLDataWaypoint, MachineConfig
from src.stage3_kinematics.base import resolve_c_angles
from src.stage3_kinematics.head_solver import SwivelHeadXYZBCSolver
from src.stage3_kinematics.trunnion_solver import TrunnionXYZBCSolver


class W:
    def __init__(self, travel=False):
        self.is_travel_move = travel


def test_chain_is_continuous_and_new_chain_resets_towards_zero():
    ang = np.radians(np.arange(0, 721, 60))                                    # two full turns of the raw angle
    wps = [W(i == 0) for i in range(len(ang))]
    first = resolve_c_angles(np.arctan2(np.sin(ang), np.cos(ang)), [30] * len(ang), wps, -36000, 36000)
    assert first[-1] == pytest.approx(720.0, abs=1e-6)                         # continuous winding inside the chain
    assert np.abs(np.diff(first)).max() == pytest.approx(60.0, abs=1e-6)
    # a second chain starts at an equivalent angle of 0 -> 0, not at 720
    raw = np.arctan2(np.sin(ang), np.cos(ang))
    both = resolve_c_angles(np.concatenate([raw, raw]), [30] * (2 * len(ang)), wps + wps, -36000, 36000)
    assert both[len(ang)] == pytest.approx(0.0, abs=1e-6)


def test_start_of_chain_respects_limits_and_picks_the_nearest_to_zero():
    c = resolve_c_angles([np.radians(170.0)], [20], [W(True)], -360.0, 360.0)
    assert c[0] == pytest.approx(170.0)
    c = resolve_c_angles([np.radians(-170.0)], [20], [W(True)], 0.0, 720.0)    # -170 is not allowed: 190 is
    assert c[0] == pytest.approx(190.0)


def test_tiny_tilt_holds_the_previous_c_and_too_small_a_range_warns(caplog):
    c = resolve_c_angles([np.radians(40.0), np.radians(-100.0)], [20, 0.1], [W(True), W(False)], -36000, 36000)
    assert c == [pytest.approx(40.0), pytest.approx(40.0)]
    with caplog.at_level(logging.WARNING):
        resolve_c_angles([np.radians(0.0), np.radians(90.0), np.radians(180.0), np.radians(-90.0), np.radians(0.0)] + [np.radians(90.0)] * 6,
                         [20] * 11, [W(True)] + [W(False)] * 10, -10.0, 10.0)
    assert any("C axis" in r.getMessage() for r in caplog.records)


def _spiral(n_turns=5, per_turn=24):
    wps = []
    for t in range(n_turns):
        for i in range(per_turn + 1):
            a = 2 * np.pi * i / per_turn
            b = np.radians(25.0)
            n = np.array([-np.sin(b) * np.cos(a), np.sin(b) * np.sin(a), np.cos(b)])    # normal that the trunnion solves to C = a
            wps.append(CLDataWaypoint(x=10 + i, y=5, z=3, i=n[0], j=n[1], k=n[2], feedrate=1000,
                                      is_travel_move=(i == 0), extrusion_volume=0.0 if i == 0 else 0.01))
    return CLDataTrajectory(waypoints=wps)


@pytest.mark.parametrize("solver", [TrunnionXYZBCSolver, SwivelHeadXYZBCSolver])
def test_solvers_do_not_pile_up_winding_over_many_rings(solver):
    cfg = MachineConfig(kinematic_chain="x", b_axis_min=-90, b_axis_max=90, c_axis_min=-36000, c_axis_max=36000)
    st = solver().solve(_spiral(), cfg).states
    c = np.array([s.c for s in st])
    assert np.abs(c).max() < 400.0                                              # 5 rings, but no accumulated 5 * 360
    ext = [(a, b) for a, b in zip(st[:-1], st[1:]) if not b.is_travel_move]
    assert max(abs(b.c - a.c) for a, b in ext) < 90.0                           # no jump while extruding


def test_skirt_and_brim_use_the_nominal_thickness():
    import trimesh
    from src.common.schemas import MinimalPrintProfile
    from src.stage1_slicer.registry import PluginRegistry
    from src.stage2_toolpath.contour_generator import StandardToolpathGenerator
    m = trimesh.creation.cylinder(radius=10, height=8, sections=48)
    m.apply_translation((0, 0, 4))
    p = {"strategy": "custom_expr", "expression": "0.4*np.sqrt(x**2+y**2)", "expression_params": {},
         "layer_height": 0.2, "transition_height": 5.0}                           # compresses the first layers a lot
    model = PluginRegistry.get_plugin("universal_field_slicer")().slice(m, p)
    prof = MinimalPrintProfile(layer_height=0.2, nozzle_diameter=0.4, num_perimeters=2, infill_density=0.15,
                               bottom_solid_layers=2, top_solid_layers=2, skirt_loops=1, brim_width=1.0)
    layers = StandardToolpathGenerator().generate_features(model, prof)
    th = {c.feature_type: c.thicknesses for l in layers for c in l.contours if c.feature_type in ("skirt", "brim", "outer_wall")}
    assert all(t == pytest.approx(0.2, abs=1e-3) for t in th["skirt"] + th["brim"])
