"""Stage 1 contour assembly: every segment is used, no loop is dropped or short-cut."""
import os

import numpy as np
import pytest
import trimesh

from src.stage1_slicer.function_file import load_function_file
from src.stage1_slicer.plugins.field_slicer import UniversalFieldSlicerPlugin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _seg(*pts):
    return np.array([[pts[i], pts[(i + 1) % len(pts)]] for i in range(len(pts))], dtype=float)


def _length(c):
    p = np.asarray(c.points)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def test_zero_length_segment_does_not_swallow_the_loop():
    sq = _seg((0, 0, 1), (4, 0, 1), (4, 4, 1), (0, 4, 1))
    degenerate = np.array([[[0, 0, 1], [0.00001, 0, 1]]], dtype=float)         # rounds to the same node
    out = UniversalFieldSlicerPlugin()._process_trimesh_segments_to_contours(np.concatenate([sq, degenerate]))
    assert len(out) == 1 and _length(out[0]) == pytest.approx(16.0, abs=1e-3)


def test_two_loops_touching_in_one_vertex_are_both_traced_completely():
    a = _seg((0, 0, 1), (2, 0, 1), (2, 2, 1), (0, 2, 1))
    b = _seg((2, 2, 1), (4, 2, 1), (4, 4, 1), (2, 4, 1))                       # shares the vertex (2, 2)
    out = UniversalFieldSlicerPlugin()._process_trimesh_segments_to_contours(np.concatenate([a, b]))
    assert sum(_length(c) for c in out) == pytest.approx(16.0, abs=1e-6)
    assert all(np.allclose(c.points[0], c.points[-1]) for c in out)


def test_open_polyline_stays_open():
    seg = np.array([[[0, 0, 0], [1, 0, 0]], [[1, 0, 0], [2, 1, 0]]], dtype=float)
    out = UniversalFieldSlicerPlugin()._process_trimesh_segments_to_contours(seg)
    assert len(out) == 1 and len(out[0].points) == 3


def test_wave_on_a_hollow_cylinder_keeps_both_rings_closed_in_every_wall_layer():
    spec = load_function_file(os.path.join(ROOT, "config", "functions", "examples", "wave.json"))
    mesh = trimesh.creation.annulus(r_min=10, r_max=15, height=20)
    mesh.apply_translation((0, 0, 10))
    model = UniversalFieldSlicerPlugin().slice(mesh, {"layer_height": 0.4, "transition_height": 2.0, **spec.to_slicer_params()})
    expected = sorted([2 * np.pi * 10, 2 * np.pi * 15])
    for layer in model.layers[3:45]:                     # full rings (the first / last layers meet the wavy caps)
        assert len(layer.contours) == 2, layer.layer_index
        lengths = sorted(_length(c) for c in layer.contours)
        for got, want in zip(lengths, expected):
            assert got == pytest.approx(want, rel=0.01), layer.layer_index
        assert all(np.allclose(c.points[0], c.points[-1]) for c in layer.contours)
