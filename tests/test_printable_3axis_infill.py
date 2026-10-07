import pytest
import trimesh
import numpy as np
from src.stage1_slicer.plugins.field_slicer import UniversalFieldSlicerPlugin
from src.stage2_toolpath.contour_generator import StandardToolpathGenerator
from src.stage3_kinematics.head_solver import Cartesian3AxisSolver
from src.stage4_postproc.klipper_postproc import Klipper5AxisPostProcessor
from src.common.schemas import MinimalPrintProfile, MachineConfig

def test_printable_3axis_infill_and_transition():
    # Large enough to test perimeters and infill
    mesh = trimesh.creation.box(extents=(30, 30, 10))
    mesh.apply_translation([0, 0, 5])

    slicer = UniversalFieldSlicerPlugin()
    params = {
        "strategy": "custom_expr",
        "expression": "5.0 * np.sin(0.1*x)",
        "layer_height": 2.0,
        "start_z": 0.0,
        "end_z": 10.0,
        "transition_height": 5.0,  # Layer 0 (z=0) and 1 (z=2) are within transition
        "nozzle_diameter": 0.4,
        "num_perimeters": 2,
        "infill_density": 0.2,
        "max_segment_length": 2.0
    }
    sliced_model = slicer.slice(mesh, params)

    # Assert layer 0 is completely flat (transition weight = 0)
    layer_0 = sliced_model.layers[0]
    for contour in layer_0.contours:
        for pt in contour.points:
            assert pt[2] == pytest.approx(0.0, abs=1e-5)
        for n in contour.normals:
            assert n[0] == pytest.approx(0.0, abs=1e-5)
            assert n[1] == pytest.approx(0.0, abs=1e-5)
            assert n[2] == pytest.approx(1.0, abs=1e-5)

    profile = MinimalPrintProfile(
        layer_height=2.0,
        nozzle_diameter=0.4,
        include_rotary_axes=False,
        retract_distance=1.0,
        z_hop=0.5
    )
    generator = StandardToolpathGenerator()
    traj = generator.generate_toolpath(sliced_model, profile)

    # Assert feature tags are present
    feature_types = set()
    for wp in traj.waypoints:
        feature_types.add(wp.feature_type)
    assert "outer_wall" in feature_types
    assert "infill" in feature_types

    # Assert layers above transition are curved
    layer_top = sliced_model.layers[-1]
    z_coords = []
    for contour in layer_top.contours:
        z_coords.extend([p[2] for p in contour.points])
    assert max(z_coords) - min(z_coords) > 2.0 # Proves it deformed!

    config = MachineConfig(
        kinematic_chain="Cartesian",
        max_3axis_tilt_deg=45.0,
        b_axis_min=0.0, b_axis_max=0.0,
        c_axis_min=0.0, c_axis_max=0.0
    )
    solver = Cartesian3AxisSolver()
    mach_traj = solver.solve(traj, config)

    postproc = Klipper5AxisPostProcessor()
    gcode = postproc.generate_gcode(mach_traj, profile)

    # Validate Output Syntaxes
    assert "TYPE: outer_wall" in gcode
    assert "TYPE: infill" in gcode

    # Retractions
    assert "G1 E-1.000 F2400 ; Retract" in gcode
    assert "; Z-Hop Travel" in gcode
    assert "; Prime" in gcode

    # Strictly 3-axis
    for line in gcode.split('\n'):
        if line.startswith('G0') or line.startswith('G1'):
            assert " B" not in line
            assert " C" not in line
