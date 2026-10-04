import pytest
import trimesh
import numpy as np
from src.stage1_slicer.plugins.field_slicer import UniversalFieldSlicerPlugin
from src.stage2_toolpath.contour_generator import StandardToolpathGenerator
from src.stage3_kinematics.head_solver import Cartesian3AxisSolver
from src.stage4_postproc.klipper_postproc import Klipper5AxisPostProcessor
from src.common.schemas import MinimalPrintProfile, MachineConfig

def test_3axis_progressive_tilt_adaptive_extrusion():
    mesh = trimesh.creation.cylinder(radius=10, height=5)
    mesh.apply_translation([0, 0, 2.5])

    # 1. Slice with progressive tilt
    slicer = UniversalFieldSlicerPlugin()
    params = {
        "strategy": "progressive_tilt",
        "layer_height": 1.0,
        "start_z": 0.0,
        "end_z": 5.0,
        "start_tilt_deg": 0.0,
        "end_tilt_deg": 20.0
    }
    sliced_model = slicer.slice(mesh, params)

    # Check that thicknesses are populated
    c_top = sliced_model.layers[-1].contours[0]
    assert c_top.thicknesses is not None
    assert len(c_top.thicknesses) == len(c_top.points)

    # Assert Z is always positive due to inner-edge pivot
    for layer in sliced_model.layers:
        for contour in layer.contours:
            for pt in contour.points:
                assert pt[2] >= -1e-5

    # Check adaptive thicknesses: the points furthest from pivot_y should have the largest thickness
    # pivot is at y_min (-10)
    outer_pts = [(pt, c_top.thicknesses[i]) for i, pt in enumerate(c_top.points) if pt[1] > -8.0]
    inner_pts = [(pt, c_top.thicknesses[i]) for i, pt in enumerate(c_top.points) if pt[1] < -9.5]

    assert len(outer_pts) > 0
    assert len(inner_pts) > 0
    assert outer_pts[0][1] > inner_pts[0][1]

    # 2. Toolpath Generation
    profile = MinimalPrintProfile(
        layer_height=1.0,
        nozzle_diameter=0.4,
        include_rotary_axes=False
    )
    generator = StandardToolpathGenerator()
    traj = generator.generate_toolpath(sliced_model, profile)

    # Verify extrusion volumes vary
    vols = [wp.extrusion_volume for wp in traj.waypoints if not wp.is_travel_move]
    assert max(vols) > min(vols)

    # 3. Kinematics (3-axis)
    config = MachineConfig(
        kinematic_chain="Cartesian",
        max_3axis_tilt_deg=25.0,
        b_axis_min=0.0, b_axis_max=0.0,
        c_axis_min=0.0, c_axis_max=0.0
    )
    solver = Cartesian3AxisSolver()
    mach_traj = solver.solve(traj, config)

    # 4. G-Code Generation
    postproc = Klipper5AxisPostProcessor()
    gcode = postproc.generate_gcode(mach_traj, profile)

    # Assert B and C coordinates are omitted because include_rotary_axes is False
    # Check that " B0" or similar does not exist in standard G-code lines
    for line in gcode.split('\n'):
        if line.startswith('G0 ') or (line.startswith('G1 ') and ' X' in line):
            assert " B" not in line
            assert " C" not in line
            assert " Z" in line
