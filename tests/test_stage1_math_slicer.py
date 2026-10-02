import pytest
from src.stage1_slicer.registry import PluginRegistry

def test_math_slicer_discovery():
    plugins = PluginRegistry.get_all_plugins()
    assert "universal_math_slicer" in plugins

def test_math_slicer_output():
    plugin_cls = PluginRegistry.get_plugin("universal_math_slicer")
    slicer = plugin_cls()

    # Generate a small 2-layer cylinder
    params = {
        "radius": 10.0,
        "height": 0.4,
        "layer_height": 0.2,
        "segments_per_layer": 4,
        "pitch": 0.0
    }

    model = slicer.slice(geometry=None, parameters=params)

    assert len(model.layers) == 2
    assert model.layers[0].layer_index == 0
    assert model.layers[0].z_height == 0.0
    assert len(model.layers[0].contours) == 1

    # 4 segments + closing point = 5 points
    contour = model.layers[0].contours[0]
    assert len(contour.points) == 5
    assert len(contour.normals) == 5

    # Verify first point is on X axis
    pt1 = contour.points[0]
    assert pt1[0] == pytest.approx(10.0)
    assert pt1[1] == pytest.approx(0.0)
    assert pt1[2] == pytest.approx(0.0)

    # Verify normal points outward
    n1 = contour.normals[0]
    assert n1[0] == pytest.approx(1.0)
    assert n1[1] == pytest.approx(0.0)
    assert n1[2] == pytest.approx(0.0)
