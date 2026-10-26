"""The first layer is printed one layer height above the bed (never at z = 0); layers are cut at mid-height."""
import pytest
import trimesh

from src.common.schemas import MachineConfig, MinimalPrintProfile
from src.stage1_slicer.plugins.field_slicer import UniversalFieldSlicerPlugin
from src.stage2_toolpath.contour_generator import StandardToolpathGenerator
from src.stage3_kinematics.registry import KinematicsRegistry
from src.stage4_postproc.klipper_postproc import Klipper5AxisPostProcessor
from tests.gcode_tools import classify, parse

H = 0.2


def _cube():
    m = trimesh.creation.box(extents=(20, 20, 10))
    m.apply_translation((0, 0, 5))
    return m


def _run(strategy="planar", **extra):
    params = {"strategy": strategy, "layer_height": H}
    params.update(extra)
    model = UniversalFieldSlicerPlugin().slice(_cube(), params)
    prof = MinimalPrintProfile(layer_height=H, nozzle_diameter=0.4, num_perimeters=1, infill_density=0.2,
                               top_solid_layers=1, bottom_solid_layers=1)
    return model, StandardToolpathGenerator().generate_toolpath(model, prof), prof


def test_planar_layers_run_from_one_layer_height_to_the_top():
    model, traj, _ = _run()
    assert len(model.layers) == 50
    z0 = sorted({round(w.z, 6) for w in traj.waypoints if w.layer_index == 0})
    assert z0 == [pytest.approx(H)]
    assert min(w.z for w in traj.waypoints) == pytest.approx(H)
    assert max(w.z for w in traj.waypoints) == pytest.approx(10.0)
    for idx in (0, 7, 49):
        assert {round(w.z, 6) for w in traj.waypoints if w.layer_index == idx} == {round(H * (idx + 1), 6)}


def test_start_z_parameter_is_the_base_of_the_print():
    _, traj, _ = _run(start_z=2.0, end_z=12.0)
    assert min(w.z for w in traj.waypoints) == pytest.approx(2.0 + H)


def test_gcode_never_prints_or_travels_at_z_zero():
    _, traj, prof = _run()
    states = KinematicsRegistry.get_solver("planar_3axis_xyz")().solve(
        traj, MachineConfig(kinematic_chain="planar_3axis_xyz", b_axis_min=0, b_axis_max=0, c_axis_min=0, c_axis_max=0))
    gcode = Klipper5AxisPostProcessor().generate_gcode(states, prof)
    zs = [c.params["Z"] for c in parse(gcode) if c.is_move and "Z" in c.params and c.params["Z"] < 100]
    assert zs and min(zs) >= H - 1e-9
    first_extrusion = next(c for c in parse(gcode) if classify(c) == "extrude")
    assert first_extrusion.params["Z"] == pytest.approx(H)


@pytest.mark.parametrize("strategy,extra", [
    ("conical", {"cone_angle_deg": 10.0, "transition_height": 4.0}),
    ("custom_expr", {"expression": "0.02 * x", "transition_height": 4.0}),
    ("progressive_tilt", {"start_tilt_deg": 0.0, "end_tilt_deg": 5.0}),
])
def test_first_layer_of_every_strategy_is_flat_and_at_layer_height(strategy, extra):
    _, traj, _ = _run(strategy, **extra)
    first = [w for w in traj.waypoints if w.layer_index == 0]
    assert first
    if strategy == "progressive_tilt":
        assert all(w.z >= H - 1e-9 for w in first)          # tilt 0 on layer 0, pivot at the low edge
    assert all(w.z == pytest.approx(H, abs=1e-6) for w in first)
