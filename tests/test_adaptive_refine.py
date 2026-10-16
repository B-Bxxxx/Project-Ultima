"""Adaptive mesh refinement, batch (vectorized) strategy evaluation and high-resolution meshes."""
import os
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.common.simplify import simplify_ring
from src.stage1_slicer.function_file import load_function_dir
from src.stage1_slicer.math_strategies import build_strategy
from src.stage1_slicer.mesh_refine import refine_adaptive
from src.stage1_slicer.registry import PluginRegistry

FUNCS = load_function_dir(os.path.join(os.path.dirname(__file__), "..", "config", "functions"))
STRESS = {"strategy": "custom_expr", "expression": "1.0*np.sin(x*1.2)*np.cos(y*1.2)", "expression_params": {}}


def _annulus(sections=48):
    m = trimesh.creation.annulus(r_min=7, r_max=12, height=8, sections=sections)
    m.apply_translation((0, 0, 4))
    return m


def _params(base, **extra):
    p = dict(base)
    p.update({"layer_height": 0.4, "transition_height": 6.0})
    p.update(extra)
    return p


# -- refine_adaptive -------------------------------------------------------------------------
def _sin_error(A, B):
    f = lambda x: np.sin(2.0 * x)
    e = np.zeros(len(A))
    for t in (0.25, 0.5, 0.75):
        M = A + t * (B - A)
        e = np.maximum(e, np.abs(f(M[:, 0]) - ((1 - t) * f(A[:, 0]) + t * f(B[:, 0]))))
    return e


def test_refine_is_conforming_and_keeps_the_shape():
    box = trimesh.creation.box(extents=(10, 10, 4))
    v, f = refine_adaptive(box.vertices, box.faces, _sin_error, 0.02, 0.3)
    m = trimesh.Trimesh(v, f, process=False)
    assert len(f) > 12 * 20
    assert m.is_watertight and m.is_winding_consistent          # no T-junctions, no flipped triangles
    assert m.volume == pytest.approx(box.volume, rel=1e-9)
    assert m.area == pytest.approx(box.area, rel=1e-9)


def test_refine_leaves_a_smooth_function_alone_and_respects_limits():
    box = trimesh.creation.box(extents=(10, 10, 4))
    v, f = refine_adaptive(box.vertices, box.faces, lambda A, B: np.zeros(len(A)), 0.02)
    assert len(f) == 12 and len(v) == 8
    # min_edge: nothing shorter than it is split; max_faces stops the loop
    v2, f2 = refine_adaptive(box.vertices, box.faces, _sin_error, 0.0001, min_edge=2.5)
    v1, f1 = refine_adaptive(box.vertices, box.faces, _sin_error, 0.0001, min_edge=0.5)
    assert len(f2) < len(f1) / 4                     # a larger minimum edge stops the refinement earlier
    v3, f3 = refine_adaptive(box.vertices, box.faces, _sin_error, 0.0001, 0.05, max_faces=500)
    assert len(f3) < 500 * 4 + 1


def test_refinement_follows_the_function_not_the_part_size():
    mesh = _annulus()
    flat = build_strategy("custom_expr", mesh, _params({"strategy": "custom_expr", "expression": "0.0"}))
    assert len(flat.deformed.faces) == len(mesh.faces)                      # f = 0: nothing to resolve
    stress = build_strategy("custom_expr", mesh, _params(STRESS))
    assert len(stress.deformed.faces) > 20 * len(mesh.faces)
    coarse = build_strategy("custom_expr", mesh, _params(STRESS, refine_tolerance=0.1))
    assert len(coarse.deformed.faces) < len(stress.deformed.faces)


def test_refined_mesh_follows_the_function_within_the_tolerance():
    tol = 0.02
    s = build_strategy("custom_expr", _annulus(), _params(STRESS, refine_tolerance=tol))
    V = s.deformed.vertices
    F = s.deformed.faces
    # points inside the triangles: the straight (flat) deformed surface vs the true z'
    rng = np.random.default_rng(1)
    pick = rng.integers(0, len(F), 600)
    bary = rng.dirichlet([1, 1, 1], 600)
    P_real = np.einsum("ni,nij->nj", bary, s.mesh.vertices[F[pick]])
    z_flat = np.einsum("ni,ni->n", bary, V[F[pick], 2])
    z_true = s._deformed_z(P_real)
    assert np.abs(z_flat - z_true).max() < 3 * tol


def test_legacy_uniform_refinement_is_kept():
    mesh = _annulus()
    s = build_strategy("custom_expr", mesh, _params(STRESS, feature_scale=12.0, refine_fraction=1 / 4))
    L = s.mesh.edges_unique_length
    assert L.max() <= 3.0 + 1e-9


# -- batch == scalar -------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["Rounded ridge", "Wave", "Conical"])
def test_batch_methods_equal_the_scalar_ones(name):
    s = build_strategy("custom_expr", _annulus(), _params(FUNCS[name].to_slicer_params({})))
    rng = np.random.default_rng(0)
    xy = rng.uniform(-10, 10, (25, 2))
    for layer in (0, 3, s.get_layer_count() - 1):
        zp = s.start_z + layer * s.layer_height - 0.5 * s.layer_height
        pts_d = np.column_stack([xy, np.full(len(xy), zp)])
        und = s.undeform_points(pts_d, layer)
        ref = np.array([s.undeform_point(tuple(p), layer) for p in pts_d])
        assert np.allclose(und, ref, atol=1e-12)
        assert np.allclose(s.project_points(xy, layer), [s.project_to_layer(x, y, layer) for x, y in xy], atol=1e-12)
        assert np.allclose(s.normals_at(und, layer), [s.compute_normal(tuple(p), layer) for p in und], atol=1e-12)
        assert np.allclose(s.thicknesses_at(und, layer), [s.compute_thickness(tuple(p), layer) for p in und], atol=1e-12)
        vg, ng = s.layer_gaps_batch(xy, layer)
        sg = np.array([s.layer_gaps(x, y, layer) for x, y in xy])
        assert np.allclose(vg, sg[:, 0], atol=1e-9) and np.allclose(ng, sg[:, 1], atol=1e-9)


def test_default_batch_fallback_for_other_strategies():
    s = build_strategy("progressive_tilt", _annulus(), {"layer_height": 0.4, "start_tilt_deg": 0.0, "end_tilt_deg": 10.0})
    xy = np.array([[8.0, 1.0], [-9.0, 2.0]])
    assert np.allclose(s.project_points(xy, 2), [s.project_to_layer(x, y, 2) for x, y in xy])
    pts = s.project_points(xy, 2)
    assert np.allclose(s.normals_at(pts, 2), [s.compute_normal(tuple(p), 2) for p in pts])
    assert np.allclose(s.thicknesses_at(pts, 2), [s.compute_thickness(tuple(p), 2) for p in pts])


# -- high-resolution meshes -----------------------------------------------------------------
def _slice(mesh, base, **extra):
    return PluginRegistry.get_plugin("universal_field_slicer")().slice(mesh, _params(base, **extra))


def test_high_resolution_mesh_gives_a_light_stage1_model_of_the_same_shape():
    base = FUNCS["Rounded ridge"].to_slicer_params({})
    lo = _slice(_annulus(96), base, contour_tolerance=0.0)
    hi = _slice(_annulus(720), base)
    hi_raw = _slice(_annulus(720), base, contour_tolerance=0.0)
    n = lambda m: sum(len(c.points) for l in m.layers for c in l.contours)
    assert n(hi) < 0.5 * n(hi_raw)                         # the 720-section rings are thinned out
    assert n(hi) < 2.0 * n(lo)                             # ... to the order of the coarse part
    # same curve: every thinned point lies on the dense contour of its layer (<= tolerance)
    from tests.test_function_origin import dist_to_polyline
    raw = {l.layer_index: l for l in hi_raw.layers}
    for l in hi.layers[:-3]:
        R = [np.asarray(c.points) for c in raw[l.layer_index].contours]
        for c in l.contours:
            P = np.asarray(c.points)
            assert min(dist_to_polyline(P, r, 1e9).max() for r in R) < 0.006


def test_simplify_ring_keeps_rings_closed_and_geometry():
    t = np.linspace(0, 2 * np.pi, 721)
    ring = np.column_stack([12 * np.cos(t), 12 * np.sin(t), np.zeros(721)])
    out = simplify_ring(ring, 0.005)
    assert len(out) < 200 and np.allclose(out[0], out[-1])
    assert np.abs(np.hypot(out[:, 0], out[:, 1]) - 12).max() < 1e-9           # kept points are original points
    open_line = ring[:100]
    assert np.allclose(simplify_ring(open_line, 0.005)[[0, -1]], open_line[[0, -1]])
