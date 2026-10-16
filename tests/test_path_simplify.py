"""Stage 2 path simplification and the one-directional walls."""
import os
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.common.schemas import MinimalPrintProfile, SpatialContour
from src.stage1_slicer.function_file import load_function_dir
from src.stage1_slicer.registry import PluginRegistry
from src.stage2_toolpath.contour_generator import StandardToolpathGenerator, rings_to_polygons
from src.stage2_toolpath.simplify import MERGE_FACTOR, simplify_path
from tests.test_function_origin import dist_to_polyline

FUNCS = load_function_dir(os.path.join(os.path.dirname(__file__), "..", "config", "functions"))


# ------------------------------------------------------------------------------------- unit
def _circle(n=2000, z_amp=0.3):
    t = np.linspace(0, 2 * np.pi, n)
    return np.c_[10 * np.cos(t), 10 * np.sin(t), z_amp * np.sin(5 * t)]


def test_collinear_points_collapse_to_the_end_points():
    P = np.c_[np.linspace(0, 10, 50), np.zeros(50), np.zeros(50)]
    idx, h = simplify_path(P, None, np.full(50, 0.2), 0.01, 0.2)
    assert idx == [0, 49] and len(h) == 1


def test_tolerance_zero_keeps_everything():
    P = _circle(100)
    idx, h = simplify_path(P, None, np.full(100, 0.2), 0.0)
    assert idx == list(range(100)) and len(h) == 99


def test_deviation_is_bounded_and_volume_preserved():
    P = _circle()
    H = np.linspace(0.15, 0.25, len(P))
    idx, h = simplify_path(P, None, H, 0.02, 0.2)
    Q = P[idx]
    assert len(idx) < len(P) / 10
    assert dist_to_polyline(P, Q, max_seg=1e9).max() <= MERGE_FACTOR * 0.02 + 1e-9
    v0 = (np.linalg.norm(np.diff(P, axis=0), axis=1) * H[1:]).sum()
    v1 = (np.linalg.norm(np.diff(Q, axis=0), axis=1) * np.array(h)).sum()
    assert v1 == pytest.approx(v0, rel=1e-9)


def test_without_min_length_the_tolerance_itself_holds():
    P = _circle()
    idx, _ = simplify_path(P, None, np.full(len(P), 0.2), 0.02, 0.0)
    assert dist_to_polyline(P, P[idx], max_seg=1e9).max() <= 0.02 + 1e-9


def test_min_segment_length_merges_short_segments():
    rng = np.random.default_rng(1)
    n = 4000
    P = np.c_[np.linspace(0, 40, n), rng.uniform(-0.0008, 0.0008, n), np.zeros(n)]      # noisy straight line
    H = np.full(n, 0.2)
    idx, _ = simplify_path(P, None, H, 0.0005, 0.0)           # the tolerance alone keeps the jitter
    idx2, _ = simplify_path(P, None, H, 0.0005, 0.3)
    seg = lambda i: np.linalg.norm(np.diff(P[i], axis=0), axis=1)
    assert len(idx2) < 0.2 * len(idx)
    assert np.mean(seg(idx2) < 0.3) < 0.1
    assert dist_to_polyline(P, P[idx2], max_seg=1e9).max() <= MERGE_FACTOR * 0.0005 + 1e-9


def test_a_turning_tool_axis_keeps_points():
    n = 200
    P = np.c_[np.linspace(0, 20, n), np.zeros(n), np.zeros(n)]                    # straight line
    ang = np.radians(40.0 * np.linspace(0, 1, n) ** 2)                            # accelerating turn: the interpolated normal is off
    N = np.c_[np.sin(ang), np.zeros(n), np.cos(ang)]                              # the normal turns by 40 deg in total
    kept_axis = len(simplify_path(P, N, np.full(n, 0.2), 0.02, 0.0, angle_deg=1.0)[0])
    kept_ignore = len(simplify_path(P, N, np.full(n, 0.2), 0.02, 0.0, angle_deg=0.0)[0])
    assert kept_ignore == 2 and kept_axis >= 4          # the axis criterion adds points where the turn is not linear


# ------------------------------------------------------------------------------- integration
def _model(name, mesh, h=0.4):
    p = FUNCS[name].to_slicer_params({})
    p.update({"layer_height": h, "transition_height": 6.0})
    return PluginRegistry.get_plugin("universal_field_slicer")().slice(mesh, p)


def _annulus(sections=96):
    m = trimesh.creation.annulus(r_min=7, r_max=12, height=8, sections=sections)
    m.apply_translation((0, 0, 4))
    return m


def _profile(**kw):
    base = dict(layer_height=0.4, nozzle_diameter=0.4, num_perimeters=2, infill_density=0.2,
                bottom_solid_layers=1, top_solid_layers=1)
    base.update(kw)
    return MinimalPrintProfile(**base)


@pytest.mark.parametrize("name", ["Planar", "Wave", "Rounded ridge"])
def test_simplified_toolpath_is_close_lighter_and_extrudes_the_same(name):
    model = _model(name, _annulus())
    gen = StandardToolpathGenerator()
    off = gen.generate_toolpath(model, _profile(path_tolerance=0.0))
    on = gen.generate_toolpath(model, _profile())
    assert len(on.waypoints) < 0.5 * len(off.waypoints)
    v = lambda t: sum(w.extrusion_volume for w in t.waypoints)
    assert v(on) == pytest.approx(v(off), rel=1e-6)
    # every original contour point lies within the allowed distance of the simplified contours of its layer / feature
    prof = _profile()
    f_off = gen.generate_features(model, _profile(path_tolerance=0.0))
    f_on = gen.generate_features(model, prof)
    for lo, hi in zip(f_off, f_on):
        for feat in ("outer_wall", "inner_wall", "top_solid", "bottom_solid", "infill"):
            tol = MERGE_FACTOR * (prof.infill_path_tolerance if feat == "infill" else prof.path_tolerance) + 1e-6
            A = [np.asarray(c.points) for c in lo.contours if c.feature_type == feat]
            B = [np.asarray(c.points) for c in hi.contours if c.feature_type == feat]
            assert len(A) == len(B)
            if not A:
                continue
            pts = np.vstack(A)
            d = np.min([dist_to_polyline(pts, b, max_seg=1e9) for b in B], axis=0)
            assert d.max() <= tol, (lo.layer_index, feat, d.max())


def test_high_resolution_part_has_few_tiny_segments():
    model = _model("Rounded ridge", _annulus(sections=360))
    prof = _profile()
    traj = StandardToolpathGenerator().generate_toolpath(model, prof)
    P = np.array([[w.x, w.y, w.z] for w in traj.waypoints])
    ex = np.array([not w.is_travel_move for w in traj.waypoints])[1:]
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)[ex]
    assert np.mean(d < 0.5 * prof.min_segment_length) < 0.05


# -------------------------------------------------------------------- one-directional walls
def _signed_area(pts):
    p = np.asarray(pts)[:, :2]
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


@pytest.mark.parametrize("name", sorted(FUNCS))
def test_walls_run_the_same_way_in_every_layer_and_level(name):
    model = _model(name, _annulus(sections=64))
    layers = StandardToolpathGenerator().generate_features(model, _profile(path_tolerance=0.0))
    seen = 0
    for L in layers[:len(layers) - 3]:               # the top layers cut the deformed cap at a grazing angle (PITFALLS #42)
        for feat in ("outer_wall", "inner_wall"):
            rings = sorted([c for c in L.contours if c.feature_type == feat], key=lambda c: -abs(_signed_area(c.points)))
            # per wall level the larger ring is the exterior, the smaller one the hole
            for ext, hole in zip(rings[0::2], rings[1::2]):
                assert _signed_area(ext.points) > 0 and _signed_area(hole.points) < 0
                seen += 1
    assert seen >= len(layers) - 3


def test_input_ring_direction_does_not_matter():
    ccw = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), (0, 0, 0)]
    for ring in (ccw, ccw[::-1]):
        poly = rings_to_polygons([SpatialContour(points=ring, normals=[(0, 0, 1)] * 5, feature_type="boundary")], 0)[0]
        assert _signed_area(list(poly.exterior.coords)) > 0


# --------------------------------------------------------------------------------------- seam
def test_start_at_seam_picks_the_extreme_vertex_and_keeps_the_direction():
    from src.stage2_toolpath.contour_generator import start_at_seam
    sq = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]                       # CCW square
    out = start_at_seam(sq, 90.0)                                           # +Y: the top edge faces it -> its middle, a vertex is inserted
    assert out[0] == out[-1] and out[0] == pytest.approx((5.0, 10.0)) and len(out) == len(sq) + 1
    assert _signed_area(out) > 0                                            # rotation does not change the direction
    assert start_at_seam(sq, 0.0)[0] == pytest.approx((10.0, 5.0))
    assert start_at_seam(sq, None) == [tuple(p) for p in sq]


@pytest.mark.parametrize("name", ["Planar", "Wave", "Rounded ridge"])
def test_every_wall_ring_starts_on_the_seam_side(name):
    model = _model(name, _annulus(sections=96))
    layers = StandardToolpathGenerator().generate_features(model, _profile(path_tolerance=0.0, seam_angle_deg=90.0))
    n = 0
    for L in layers[:len(layers) - 3]:                                      # not the grazing top layers (PITFALLS #42)
        for c in L.contours:
            if c.feature_type in ("outer_wall", "inner_wall"):
                P = np.asarray(c.points)
                assert P[0, 1] >= P[:, 1].max() - 0.1                       # starts at the back of its ring
                assert abs(P[0, 0]) < 1.5                                   # ... on the same vertical line (x ~ 0)
                n += 1
    assert n > 20


def test_seam_of_a_box_is_one_line_across_wall_levels():
    import trimesh as tm
    from src.stage1_slicer.plugins.field_slicer import UniversalFieldSlicerPlugin
    mesh = tm.creation.box(extents=(20, 20, 4))
    mesh.apply_translation([0, 0, 2])
    model = UniversalFieldSlicerPlugin().slice(mesh, {"strategy": "planar", "layer_height": 1.0})
    layers = StandardToolpathGenerator().generate_features(model, _profile(layer_height=1.0, path_tolerance=0.0))
    starts = [c.points[0] for L in layers for c in L.contours if c.feature_type in ("outer_wall", "inner_wall")]
    assert len(starts) >= 8 and all(abs(p[0]) < 0.1 for p in starts)           # all start at x = 0 on the back edge
