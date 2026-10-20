"""Ramp transition mode: planar partial base layers up to the function surface, then exact-f layers.
The standard (linear) mode must stay untouched and remain the default."""
import logging
import os
import sys

import numpy as np
import pytest
import shapely
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.common.schemas import MinimalPrintProfile
from src.stage1_slicer import thickness_check
from src.stage1_slicer.level_region import marching_segments, region_polygon
from src.stage1_slicer.math_strategies import CustomExpressionStrategy, build_strategy
from src.stage1_slicer.ramp_strategy import RampCustomExpressionStrategy
from src.stage1_slicer.registry import PluginRegistry
from src.stage2_toolpath.contour_generator import StandardToolpathGenerator, rings_to_polygons

LINEAR = {"strategy": "custom_expr", "expression": "0.15*(x+12)", "expression_params": {}, "layer_height": 0.4}
RAMP = {**LINEAR, "transition_mode": "ramp", "ramp_layer_height": 0.2, "ramp_min_gap": 0.1}


def _part():
    m = trimesh.creation.annulus(r_min=7, r_max=12, height=8, sections=64)
    m.apply_translation((0, 0, 4))
    return m


def _slice(params, mesh=None):
    return PluginRegistry.get_plugin("universal_field_slicer")().slice(mesh or _part(), params)


# -- the region extraction ---------------------------------------------------------------------
def _grid(n=101):
    X = np.linspace(-10, 10, n)
    Y = np.linspace(-10, 10, n)
    return X, Y, *np.meshgrid(X, Y)


def test_region_of_a_disc_a_half_plane_and_a_hole():
    X, Y, XX, YY = _grid()
    disc = region_polygon(X, Y, 25 - (XX ** 2 + YY ** 2), lambda x, y: 25 - x * x - y * y)
    assert disc.area == pytest.approx(np.pi * 25, rel=2e-3)
    assert region_polygon(X, Y, XX - 3).area == pytest.approx(7 * 20)
    outside = region_polygon(X, Y, XX ** 2 + YY ** 2 - 4, lambda x, y: x * x + y * y - 4)
    assert outside.area == pytest.approx(400 - 4 * np.pi, rel=2e-3)
    assert len(outside.interiors) == 1
    assert region_polygon(X, Y, XX * 0 - 1).is_empty
    assert region_polygon(X, Y, XX * 0 + 1).area == pytest.approx(400)


def test_marching_squares_saddle_is_resolved_with_the_centre_value():
    X = np.array([0.0, 1.0])
    Y = np.array([0.0, 1.0])
    G = np.array([[1.0, -1.0], [-1.0, 1.0]])           # corners a, c inside, centre value 0 (inside)
    assert len(marching_segments(X, Y, G)) == 2


# -- the default stays untouched --------------------------------------------------------------------
def test_standard_mode_is_the_default_and_unchanged():
    p = {**LINEAR, "transition_height": 2.0}
    s = build_strategy("custom_expr", _part(), p)
    assert type(s) is CustomExpressionStrategy
    a = _slice(p)
    b = _slice({**p, "transition_mode": "linear"})
    assert "ramp_layers" not in a.metadata
    assert len(a.layers) == len(b.layers)
    assert all(np.allclose(np.asarray(x.contours[0].points), np.asarray(y.contours[0].points)) for x, y in zip(a.layers, b.layers))


def test_unknown_mode_and_non_function_strategies_are_rejected():
    with pytest.raises(ValueError):
        build_strategy("custom_expr", _part(), {**LINEAR, "transition_mode": "magic"})
    with pytest.raises(ValueError):
        build_strategy("progressive_tilt", _part(), {"layer_height": 0.4, "transition_mode": "ramp"})


# -- the ramp geometry -------------------------------------------------------------------------------
def test_ramp_layers_are_planar_and_full_layers_follow_the_function_with_constant_thickness():
    s = build_strategy("custom_expr", _part(), RAMP)
    assert isinstance(s, RampCustomExpressionStrategy) and s.ramp_layers > 0
    K = s.ramp_layers
    rng = np.random.default_rng(0)
    xy = rng.uniform(-11, 11, (200, 2))
    for k in (0, K // 2, K - 1):                                  # ramp layers: one plane, thickness = ramp layer height
        z = s.project_points(xy, k)[:, 2]
        assert np.ptp(z) == 0.0 and z[0] == pytest.approx((k + 1) * 0.2)
        vg, ng = s.layer_gaps_batch(xy, k)
        assert np.allclose(vg, 0.2) and np.allclose(ng, 0.2)
        assert np.allclose(s.normals_at(s.project_points(xy, k), k), [0, 0, 1])
    for j in (1, 2, 5):                                           # full layers: exact f, spacing h everywhere
        L = K + j
        vg, _ = s.layer_gaps_batch(xy, L)
        assert np.allclose(vg, 0.4, atol=1e-9)
        assert np.allclose(s.thicknesses_at(s.project_points(xy, L), L), 0.4)
        z = s.project_points(xy, L)[:, 2]
        assert np.allclose(z, s.start_z + j * 0.4 - s._f(xy[:, 0], xy[:, 1]))
    # the lowest surface point of the first full layer is one layer height above the bed
    V = s.mesh.vertices
    assert s.project_points(V[:, :2], K)[:, 2].min() == pytest.approx(0.4, abs=1e-6)


def test_first_full_layer_thickness_is_the_gap_to_the_stair_top():
    s = build_strategy("custom_expr", _part(), RAMP)
    K = s.ramp_layers
    xy = np.random.default_rng(1).uniform(-11, 11, (300, 2))
    xy = xy[s._in_footprint(xy[:, 0], xy[:, 1])]                  # under the bed-contact surface (the ring)
    assert len(xy) > 50
    surface = s.project_points(xy, K)[:, 2]
    tops = np.array([s._ramp_top(k) for k in range(K)])
    for (x, y), z, g in zip(xy, surface, s.first_gap(xy[:, 0], xy[:, 1])):
        below = tops[tops + 0.1 <= z + 1e-9]                      # ramp layers that exist at (x, y)
        fill = below.max() if len(below) else 0.0
        assert g == pytest.approx(z - fill, abs=1e-9)
        assert 0.1 - 1e-9 <= g < 0.3 + 1e-9 or g <= 0.4 + 1e-9    # [min gap, min gap + ramp layer) (the lowest point: h)


def test_ramp_layer_contours_are_clipped_to_the_part_under_the_surface():
    s = build_strategy("custom_expr", _part(), RAMP)
    model = _slice(RAMP)
    K = model.metadata["ramp_layers"]
    assert K == s.ramp_layers
    areas = []
    for layer in model.layers[:K]:
        polys = rings_to_polygons(layer.contours, layer.layer_index)
        areas.append(sum(p.area for p in polys))
        top = s._ramp_top(layer.layer_index)
        for c in layer.contours:
            P = np.asarray(c.points)
            assert np.ptp(P[:, 2]) == 0.0 and P[0, 2] == pytest.approx(top - 0.1)       # cut at the mid-plane
            below = s.start_z - s._f(P[:, 0], P[:, 1]) - (top + 0.1)
            assert below.min() > -0.02                                    # everything lies under the function surface
    assert all(b <= a + 1e-6 for a, b in zip(areas, areas[1:]))           # higher ramp layers cover less


def test_the_whole_part_volume_is_printed_exactly_once():
    s = build_strategy("custom_expr", _part(), RAMP)
    model = _slice(RAMP)
    K = s.ramp_layers
    rng = np.random.default_rng(0)
    vol = 0.0
    for layer in model.layers:
        polys = rings_to_polygons(layer.contours, layer.layer_index)
        if layer.layer_index < K:
            vol += sum(p.area for p in polys) * 0.2
        elif layer.layer_index == K:
            for p in polys:
                b = p.bounds
                xs, ys = rng.uniform(b[0], b[2], 30000), rng.uniform(b[1], b[3], 30000)
                inside = shapely.contains_xy(p, xs, ys)
                vol += p.area * s.first_gap(xs[inside], ys[inside]).mean()
        else:
            vol += sum(p.area for p in polys) * 0.4
    assert vol == pytest.approx(_part().volume, rel=0.01)


def test_stage2_rebuilds_the_same_strategy_and_the_pipeline_runs_clean(caplog):
    model = _slice(RAMP)
    meta = model.metadata
    assert meta["transition_mode"] == "ramp" and meta["transition_height"] == 0.0
    s1 = build_strategy("custom_expr", _part(), RAMP)
    s2 = build_strategy("custom_expr", trimesh.Trimesh(), meta)                # what Stage 2 does
    assert s2.ramp_layers == s1.ramp_layers and s2.num_layers == s1.num_layers
    xy = np.random.default_rng(2).uniform(-11, 11, (50, 2))
    for L in (0, s1.ramp_layers - 1, s1.ramp_layers, s1.ramp_layers + 4):
        assert np.allclose(s1.project_points(xy, L), s2.project_points(xy, L))
    prof = MinimalPrintProfile(layer_height=0.4, nozzle_diameter=0.4, num_perimeters=2, infill_density=0.2,
                               bottom_solid_layers=2, top_solid_layers=2)
    with caplog.at_level(logging.WARNING):
        traj = StandardToolpathGenerator().generate_toolpath(model, prof)
    assert not [r for r in caplog.records if "skipping" in r.getMessage()]
    layers = {w.layer_index for w in traj.waypoints if w.layer_index is not None}
    assert layers == set(range(len(model.layers)))
    assert all(w.extrusion_volume >= 0 and np.isfinite([w.x, w.y, w.z]).all() for w in traj.waypoints)
    ramp_z = {round(w.z, 6) for w in traj.waypoints if w.layer_index is not None and w.layer_index < s1.ramp_layers and not w.is_travel_move}
    assert ramp_z == {round(s1._ramp_top(k), 6) for k in range(s1.ramp_layers)}


# -- the point of the mode ---------------------------------------------------------------------------
def test_ramp_reaches_a_large_function_value_where_the_standard_transition_cannot():
    big = {"strategy": "custom_expr", "expression": "0.25*(x+12)", "expression_params": {}, "layer_height": 0.3}   # f up to 6 mm
    std = _slice({**big, "transition_height": 1.5})
    ramp = _slice({**big, "transition_mode": "ramp", "ramp_layer_height": 0.2})
    nozzle = 0.4
    bad = thickness_check.check(thickness_check.measure_model(std), nozzle, False, 0.3)
    good = thickness_check.check(thickness_check.measure_model(ramp), nozzle, False, 0.3)
    assert not bad.ok          # the standard transition leaves the printable band (or folds)
    assert good.ok             # ramp: every measured gap is the ramp layer height or the layer height


def test_ramp_works_with_a_data_file_function(tmp_path):
    from src.stage1_slicer.function_import import GridField
    xs = np.linspace(-15, 15, 31)
    ys = np.linspace(-15, 15, 31)
    vals = [[0.1 * (x + 15) for x in xs] for _ in ys]
    p = {"strategy": "grid_field", "grid_x": xs.tolist(), "grid_y": ys.tolist(), "grid_values": vals,
         "layer_height": 0.4, "transition_mode": "ramp", "ramp_layer_height": 0.2}
    model = _slice(p)
    assert model.metadata["ramp_layers"] > 0 and len(model.layers) > model.metadata["ramp_layers"]


def test_ramp_is_built_only_under_the_bed_contact_surface():
    # a 20x20 foot (z 0..4) with a 40-wide slab on top (z 4..8) that overhangs it on both sides
    foot = trimesh.creation.box(extents=(20, 20, 4)); foot.apply_translation((0, 0, 2))
    slab = trimesh.creation.box(extents=(40, 20, 4)); slab.apply_translation((0, 0, 6))
    part = trimesh.util.concatenate([foot, slab])
    prm = {**RAMP, "expression": "0.3*x", "layer_height": 0.4}
    s = build_strategy("custom_expr", part, prm)
    assert s._footprint is not None and s._footprint.area == pytest.approx(20 * 20, rel=1e-3)
    K = s.ramp_layers
    model = _slice({**prm, "strategy": "custom_expr"}, part)
    for layer in model.layers:
        if layer.layer_index >= K:
            continue
        xs = [p[0] for c in layer.contours for p in c.points]
        assert max(abs(x) for x in xs) <= 10 + 1e-6              # nothing of the ramp over the overhang
    # first full layer: ramp thickness over the foot, nominal thickness over the overhang
    assert s.first_gap([15.0], [0.0])[0] == pytest.approx(0.4)
    assert s.first_gap([0.0], [0.0])[0] >= 0.1
    # Stage 2 rebuild keeps the footprint
    s2 = build_strategy("custom_expr", trimesh.Trimesh(), model.metadata)
    assert s2._footprint is not None and s2.first_gap([15.0], [0.0])[0] == pytest.approx(0.4)
    off = build_strategy("custom_expr", part, {**prm, "ramp_contact_only": False})
    assert off._footprint is None


def _foot_and_slab():
    foot = trimesh.creation.box(extents=(20, 20, 4)); foot.apply_translation((0, 0, 2))
    slab = trimesh.creation.box(extents=(60, 60, 3)); slab.apply_translation((0, 0, 5.5))
    return trimesh.util.concatenate([foot, slab])


def _printed_volume(model):
    from src.stage2_toolpath.contour_generator import rings_to_polygons
    total = 0.0
    for l in model.layers:
        polys = rings_to_polygons(l.contours, l.layer_index)
        th = [t for c in l.contours for t in (c.thicknesses or [])]
        total += sum(p.area for p in polys) * (float(np.mean(th)) if th else 0.2)
    return total


def test_overhang_material_below_the_first_full_layer_is_not_lost_pre_layers():
    part = _foot_and_slab()                                     # slab overhangs the foot by 20 mm all around
    prm = {**RAMP, "strategy": "custom_expr", "expression": "r*tan(radians(45))", "layer_height": 0.2}
    s = build_strategy("custom_expr", part, prm)
    assert s.pre_layers > 0
    assert s.num_layers == s.ramp_layers + s.num_full_layers and s.num_full_layers > s.pre_layers   # pre-layers are counted in the full layers
    model = _slice(prm, part)
    assert model.metadata["ramp_pre_layers"] == s.pre_layers
    vol = _printed_volume(model)
    assert vol == pytest.approx(part.volume, rel=0.03)          # nothing lost
    s2 = build_strategy("custom_expr", trimesh.Trimesh(), model.metadata)
    assert s2.pre_layers == s.pre_layers and s2.start_z == pytest.approx(s.start_z) and s2.first_z == pytest.approx(s.first_z)
    # pre-layers exist only outside the footprint and keep the full-layer thickness
    K = s.ramp_layers
    for layer in model.layers:
        if K <= layer.layer_index < K + s.pre_layers:
            xs = np.abs([p[0] for c in layer.contours for p in c.points]); ys = np.abs([p[1] for c in layer.contours for p in c.points])
            assert (np.maximum(xs, ys) >= 10 - 1e-6).all()
            assert np.allclose([t for c in layer.contours for t in c.thicknesses], 0.2)


def test_a_face_exactly_in_the_slice_plane_still_gives_closed_loops():
    # Part69: its slab top (z = 7.5) lies exactly in the mid-plane of layer 37 (0.1 + 0.2 * 37); without the
    # nudge the section is a closed loop plus two open 3-point fragments ("skipping an open contour")
    mesh = trimesh.load(os.path.join(os.path.dirname(__file__), "data", "part69_coplanar_face.stl"))
    model = PluginRegistry.get_plugin("universal_field_slicer")().slice(mesh, {"strategy": "planar", "layer_height": 0.2})
    layer = next(l for l in model.layers if l.layer_index == 37)
    assert layer.contours
    for l in model.layers:
        for c in l.contours:
            assert c.points[0] == pytest.approx(c.points[-1], abs=1e-6), f"open contour in layer {l.layer_index}"


def _cross():
    def box(w, d, h, z):
        b = trimesh.creation.box(extents=(w, d, h)); b.apply_translation((0, 0, z)); return b
    return trimesh.util.concatenate([box(20, 20, 10, 5), box(40, 20, 20, 20), box(20, 20, 10, 35)])


def test_first_function_layer_touches_the_bed_at_the_farthest_point_of_the_footprint():
    # any geometry: the ramp is only the wedge under the contact surface, everything else is built by the function
    part = _cross()
    prm = {**RAMP, "strategy": "custom_expr", "expression": "r*tan(radians(45))", "layer_height": 0.2}
    s = build_strategy("custom_expr", part, prm)
    r_foot = np.hypot(10, 10)                                     # farthest footprint corner
    assert s.first_z - r_foot == pytest.approx(0.2, abs=0.02)     # first layer: one layer height above the bed there
    assert s.first_z < s.clear_z                                  # the arm tips would dip below the bed -> partial layers
    # the ramp stops at the first function layer: no ramp layer above its surface at the footprint centre
    assert s._ramp_top(s.ramp_layers - 1) <= s.first_z - 0.1 + 1e-6
    # without contact-only the ramp goes everywhere (old behaviour): first layer set by the whole mesh
    off = build_strategy("custom_expr", part, {**prm, "ramp_contact_only": False})
    assert off.first_z == pytest.approx(off.clear_z) and off.first_z > s.first_z + 5
    model = _slice(prm, part)
    K = s.ramp_layers + s.pre_layers
    low = min(p[2] for l in model.layers if l.layer_index >= K for c in l.contours for p in c.points)
    assert low >= 0.1 - 1e-6                                      # no function layer is printed below the bed
    assert any(l.layer_index == K for l in model.layers)          # the first function layer exists
