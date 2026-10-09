import pytest
import trimesh
import numpy as np
from src.stage1_slicer.registry import PluginRegistry

@pytest.fixture
def test_mesh():
    mesh = trimesh.creation.cylinder(radius=10, height=20)
    mesh.apply_translation([0, 0, 10])
    return mesh

@pytest.fixture
def slicer():
    plugin_cls = PluginRegistry.get_plugin("universal_field_slicer")
    return plugin_cls()

def test_planar_mode(slicer, test_mesh):
    params = {
        "mode": "planar",
        "layer_height": 5.0,
        "start_z": 0.0,
        "end_z": 20.0
    }
    model = slicer.slice(test_mesh, params)

    assert len(model.layers) == 4
    for layer in model.layers:
        assert len(layer.contours) > 0
        contour = layer.contours[0]
        # Verify normals are vertical
        n = contour.normals[0]
        assert n[0] == pytest.approx(0.0)
        assert n[1] == pytest.approx(0.0)
        assert n[2] == pytest.approx(1.0)

def test_progressive_tilt_mode(slicer, test_mesh):
    params = {
        "mode": "progressive_tilt",
        "layer_height": 10.0,
        "start_z": 0.0,
        "end_z": 20.0,
        "start_tilt_deg": 0.0,
        "end_tilt_deg": 45.0
    }
    model = slicer.slice(test_mesh, params)

    assert len(model.layers) == 2

    # Layer 0: tilt should be 0
    c0 = model.layers[0].contours[0]
    n0 = c0.normals[0]
    assert n0[0] == pytest.approx(0.0)
    assert n0[1] == pytest.approx(0.0)
    assert n0[2] == pytest.approx(1.0)

    # Layer 1: tilt should be 45 deg around X (so Y and Z are non-zero)
    c1 = model.layers[1].contours[0]
    n1 = c1.normals[0]
    assert n1[0] == pytest.approx(0.0)
    assert n1[1] == pytest.approx(-np.sin(np.radians(45)))
    assert n1[2] == pytest.approx(np.cos(np.radians(45)))

def test_conical_mode(slicer, test_mesh):
    # test_mesh has start_z=0 and end_z=20 (radius 10)
    params = {
        "mode": "conical_or_curved",
        "layer_height": 5.0,
        "start_z": 5.0, # shift up to ensure we hit the mesh during deform
        "end_z": 15.0,
        "cone_angle_deg": 30.0
    }
    model = slicer.slice(test_mesh, params)

    # Due to deform slicing, the number of layers might be less, but at least 1 should exist
    assert len(model.layers) > 0
    contour = model.layers[0].contours[0]

    alpha = np.radians(30.0)
    for pt, norm in zip(contour.points, contour.normals):
        # f(x,y) = r * tan(alpha)
        # Normal gradient = [+x/r tan(alpha), +y/r tan(alpha), 1] normalized
        x, y, z = pt
        r = np.sqrt(x**2 + y**2)
        if r > 1e-6:
            expected_nx = (x / r) * np.tan(alpha)
            expected_ny = (y / r) * np.tan(alpha)
            length = np.sqrt(expected_nx**2 + expected_ny**2 + 1.0)
            expected_nx /= length
            expected_ny /= length
            expected_nz = 1.0 / length

            assert norm[0] == pytest.approx(expected_nx, abs=1e-5)
            assert norm[1] == pytest.approx(expected_ny, abs=1e-5)
            assert norm[2] == pytest.approx(expected_nz, abs=1e-5)
