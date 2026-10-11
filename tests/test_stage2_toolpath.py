"""
Stage 2 tests: solid skin (top/bottom layers), zigzag infill linking, nesting parity of
holes/islands, input immutability, and layer_index propagation to Stage 3.

All geometry assertions use shapely areas/containment or hand-computed values; none of them
re-implements the generator's own formulas.
"""
import inspect
import logging
import math

import numpy as np
import pytest
import shapely.geometry as sg
import trimesh

from src.common.schemas import (
    CLDataTrajectory, CLDataWaypoint, MachineConfig, MinimalPrintProfile, SpatialContour,
    UniversalLayer, UniversalSlicedModel,
)
from src.stage1_slicer.plugins.field_slicer import UniversalFieldSlicerPlugin
from src.stage2_toolpath.base import BaseToolpathGenerator
from src.stage2_toolpath.contour_generator import (
    FEATURE_ORDER, StandardToolpathGenerator, fill_region, rings_to_polygons,
)
from src.stage3_kinematics.registry import KinematicsRegistry

NOZZLE = 0.4


def _slice(mesh, layer_height=1.0):
    return UniversalFieldSlicerPlugin().slice(mesh, {"strategy": "planar", "layer_height": layer_height})


def _profile(**kw):
    base = dict(layer_height=1.0, nozzle_diameter=NOZZLE, num_perimeters=2, infill_density=0.2,
                top_solid_layers=3, bottom_solid_layers=3)
    base.update(kw)
    return MinimalPrintProfile(**base)


def _box(w=30, d=30, h=10):
    m = trimesh.creation.box(extents=(w, d, h))
    m.apply_translation([0, 0, h / 2.0])
    return m


def _xy_length(contour):
    pts = np.array(contour.points)[:, :2]
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _square_ring(size, cx=0.0, cy=0.0, z=0.0):
    h = size / 2.0
    pts = [(cx - h, cy - h, z), (cx + h, cy - h, z), (cx + h, cy + h, z), (cx - h, cy + h, z), (cx - h, cy - h, z)]
    return SpatialContour(points=pts, normals=[(0, 0, 1)] * 5, feature_type="boundary")


def _square_layer(idx, size):
    return UniversalLayer(layer_index=idx, z_height=float(idx), contours=[_square_ring(size, z=float(idx))])


# ---------------------------------------------------------------------------------------------
# 1. solid skin
# ---------------------------------------------------------------------------------------------
def test_cube_skin_layers_are_fully_solid_and_the_core_is_sparse():
    model = _slice(_box())
    assert [l.layer_index for l in model.layers] == list(range(10))
    gen = StandardToolpathGenerator()
    regions = gen.compute_layer_regions(model, _profile())

    # infill region = outline eroded by 2 walls of 0.4 mm -> a 28.4 mm square
    expected_area = (30 - 4 * NOZZLE) ** 2
    for idx in range(10):
        r = regions[idx]
        assert r.region.area == pytest.approx(expected_area, rel=1e-6)
        if idx <= 2:                                   # bottom skin
            assert r.bottom_solid.area == pytest.approx(r.region.area, rel=1e-6)
            assert r.top_solid.is_empty and r.sparse.is_empty
        elif idx >= 7:                                 # top skin
            assert r.top_solid.area == pytest.approx(r.region.area, rel=1e-6)
            assert r.bottom_solid.is_empty and r.sparse.is_empty
        else:                                          # core
            assert r.sparse.area == pytest.approx(r.region.area, rel=1e-6)
            assert r.top_solid.is_empty and r.bottom_solid.is_empty

    layers = {l.layer_index: l for l in gen.generate_features(model, _profile())}
    for idx in range(10):
        types = {c.feature_type for c in layers[idx].contours}
        if idx <= 2:
            assert types == {"inner_wall", "outer_wall", "bottom_solid"}
        elif idx >= 7:
            assert types == {"inner_wall", "outer_wall", "top_solid"}
        else:
            assert types == {"inner_wall", "outer_wall", "infill"}


def test_solid_skin_really_is_dense_and_sparse_infill_is_sparse():
    model = _slice(_box())
    gen = StandardToolpathGenerator()
    layers = {l.layer_index: l for l in gen.generate_features(model, _profile())}
    area = (30 - 4 * NOZZLE) ** 2

    def fill_length(layer, feature):
        return sum(_xy_length(c) for c in layer.contours if c.feature_type == feature)

    # line length * line width ~ covered area: solid ~ 100 %, sparse ~ the configured 20 %
    solid_cover = fill_length(layers[0], "bottom_solid") * NOZZLE / area
    assert 0.95 < solid_cover < 1.10
    sparse_cover = fill_length(layers[5], "infill") * NOZZLE / area
    assert 0.17 < sparse_cover < 0.30
    assert fill_length(layers[9], "top_solid") * NOZZLE / area == pytest.approx(solid_cover, rel=0.1)


def test_solid_skin_scan_direction_alternates_plus_minus_45_degrees():
    model = _slice(_box())
    layers = {l.layer_index: l for l in StandardToolpathGenerator().generate_features(model, _profile())}
    for idx, sign in ((0, +1), (1, -1), (2, +1)):
        c = next(c for c in layers[idx].contours if c.feature_type == "bottom_solid")
        p0, p1 = np.array(c.points[0][:2]), np.array(c.points[1][:2])
        dx, dy = p1 - p0
        assert abs(abs(dx) - abs(dy)) < 1e-6 * max(1.0, abs(dx))      # exactly 45 degrees
        assert np.sign(dx * dy) == sign


def test_top_and_bottom_layers_zero_disable_the_skin():
    model = _slice(_box())
    gen = StandardToolpathGenerator()
    regions = gen.compute_layer_regions(model, _profile(top_solid_layers=0, bottom_solid_layers=0))
    assert all(r.sparse.area > 0 and r.top_solid.is_empty and r.bottom_solid.is_empty
               for r in regions.values())
    types = {c.feature_type for l in gen.generate_features(model, _profile(top_solid_layers=0, bottom_solid_layers=0))
             for c in l.contours}
    assert types == {"inner_wall", "outer_wall", "infill"}


def test_sloped_surface_gets_a_solid_ring_over_a_sparse_core():
    # Cone: every layer is smaller than the one below, so the TOP skin of layer i is the ring
    # that is not covered by layer i+2 (top_n = 2), and the core below the next two layers stays sparse.
    cone = trimesh.creation.cone(radius=10.0, height=10.0, sections=64)
    model = _slice(cone, layer_height=1.0)
    profile = _profile(top_solid_layers=2, bottom_solid_layers=1)
    regions = StandardToolpathGenerator().compute_layer_regions(model, profile)

    def r_region(idx):                       # analytic radius of the eroded infill region
        return 10.0 * (1.0 - idx / 10.0) - 2 * NOZZLE

    for idx in (2, 3, 4, 5):
        r = regions[idx]
        ring = math.pi * (r_region(idx) ** 2 - r_region(idx + 2) ** 2)
        core = math.pi * r_region(idx + 2) ** 2
        assert r.top_solid.area == pytest.approx(ring, rel=0.06)
        assert r.sparse.area == pytest.approx(core, rel=0.06)
        assert r.top_solid.intersection(r.sparse).area < 1e-3          # no overlap
        assert r.top_solid.union(r.sparse).area == pytest.approx(r.region.area, rel=1e-4)
        assert r.bottom_solid.is_empty


def test_gap_in_layer_numbering_is_treated_as_a_surface():
    # Layers 0,1,3,4: layer 2 is missing (e.g. an empty slice), so layer 1 is a top and layer 3 a bottom.
    model = UniversalSlicedModel(layers=[_square_layer(i, 20) for i in (0, 1, 3, 4)])
    regions = StandardToolpathGenerator().compute_layer_regions(
        model, _profile(top_solid_layers=1, bottom_solid_layers=1))
    assert regions[1].top_solid.area == pytest.approx(regions[1].region.area)
    assert regions[3].bottom_solid.area == pytest.approx(regions[3].region.area)
    assert regions[1].sparse.is_empty and regions[3].sparse.is_empty


# ---------------------------------------------------------------------------------------------
# 2. infill linking
# ---------------------------------------------------------------------------------------------
def test_solid_cube_mid_layer_has_exactly_one_infill_contour():
    model = _slice(_box())
    layers = {l.layer_index: l for l in StandardToolpathGenerator().generate_features(model, _profile())}
    for idx in (3, 4, 5, 6):
        assert sum(1 for c in layers[idx].contours if c.feature_type == "infill") == 1


def test_infill_never_crosses_a_through_hole():
    ring = trimesh.creation.annulus(r_min=8.0, r_max=15.0, height=10.0, sections=96)
    ring.apply_translation([0, 0, 5])
    model = _slice(ring)
    layers = StandardToolpathGenerator().generate_features(model, _profile())
    hole = sg.Point(0, 0).buffer(7.9)                  # a bit smaller than the real 8 mm hole
    n_segments = 0
    for layer in layers:
        for c in layer.contours:
            if c.feature_type in ("infill", "bottom_solid", "top_solid"):
                pts = [(p[0], p[1]) for p in c.points]
                for a, b in zip(pts[:-1], pts[1:]):
                    assert sg.LineString([a, b]).intersection(hole).length < 1e-6
                    n_segments += 1
    assert n_segments > 500


def _u_shape():
    # U opening to the right: two arms joined by a base on the left
    return sg.Polygon([(0, 0), (40, 0), (40, 10), (10, 10), (10, 30), (40, 30), (40, 40), (0, 40)])


def test_fill_links_stay_inside_the_region_and_split_where_needed():
    poly = _u_shape()
    spacing = 2.0
    chains = fill_region(poly, spacing, angle_deg=0.0)
    tol = 0.1 * spacing
    allowed = poly.buffer(tol + 1e-9)
    assert len(chains) >= 2                       # the two arms cannot be one zigzag
    for chain in chains:
        for a, b in zip(chain[:-1], chain[1:]):
            assert allowed.covers(sg.LineString([a, b])), (a, b)

    # independent coverage check: every scanline segment must be printed
    x = poly.bounds[0] + spacing / 2.0
    missing = 0
    while x < poly.bounds[2]:
        inter = poly.intersection(sg.LineString([(x, -5), (x, 45)]))
        for seg in getattr(inter, "geoms", [inter]):
            if seg.geom_type != "LineString" or seg.length < 1e-6:
                continue
            y0, y1 = sorted(c[1] for c in seg.coords)[0], sorted(c[1] for c in seg.coords)[-1]
            found = any(abs(a[0] - x) < 1e-9 and abs(b[0] - x) < 1e-9 and
                        sorted([a[1], b[1]]) == pytest.approx([y0, y1], abs=1e-9)
                        for chain in chains for a, b in zip(chain[:-1], chain[1:]))
            missing += 0 if found else 1
        x += spacing
    assert missing == 0


def test_fill_of_convex_region_is_a_single_chain_and_islands_are_filled_separately():
    square = sg.box(0, 0, 20, 20)
    assert len(fill_region(square, 1.0, 45.0)) == 1
    two = sg.MultiPolygon([sg.box(0, 0, 10, 10), sg.box(30, 0, 40, 10)])
    chains = fill_region(two, 1.0, 0.0)
    assert len(chains) == 2
    for chain in chains:                           # no chain jumps between the islands
        xs = [p[0] for p in chain]
        assert max(xs) < 11 or min(xs) > 29


# ---------------------------------------------------------------------------------------------
# 3. outlines, holes, islands, hygiene
# ---------------------------------------------------------------------------------------------
def test_nesting_parity_assigns_holes_and_islands():
    rings = [_square_ring(20), _square_ring(14), _square_ring(8), _square_ring(3)]
    polys = rings_to_polygons(rings)
    assert len(polys) == 2
    polys.sort(key=lambda p: p.area, reverse=True)
    outer, island = polys
    assert len(outer.interiors) == 1 and len(island.interiors) == 1
    assert outer.area == pytest.approx(20 * 20 - 14 * 14)
    assert island.area == pytest.approx(8 * 8 - 3 * 3)


def test_island_inside_a_hole_is_printed_as_an_island():
    ring = trimesh.creation.annulus(r_min=8.0, r_max=15.0, height=10.0, sections=96)
    core = trimesh.creation.cylinder(radius=4.0, height=10.0, sections=64)
    mesh = trimesh.util.concatenate([ring, core])
    mesh.apply_translation([0, 0, 5])
    model = _slice(mesh)

    polys = rings_to_polygons(model.layers[5].contours)
    assert len(polys) == 2
    polys.sort(key=lambda p: p.area, reverse=True)
    assert len(polys[0].interiors) == 1 and len(polys[1].interiors) == 0
    assert polys[0].area == pytest.approx(math.pi * (15 ** 2 - 8 ** 2), rel=0.01)
    assert polys[1].area == pytest.approx(math.pi * 4 ** 2, rel=0.01)

    layer = {l.layer_index: l for l in StandardToolpathGenerator().generate_features(model, _profile())}[5]
    island = [c for c in layer.contours if np.hypot(*np.mean(np.array(c.points)[:, :2], axis=0)) < 1.0
              and max(np.hypot(p[0], p[1]) for p in c.points) < 5.0]
    assert {c.feature_type for c in island} >= {"outer_wall", "inner_wall", "infill"}
    walls = [c for c in island if c.feature_type == "outer_wall"]
    assert np.mean([np.hypot(p[0], p[1]) for p in walls[0].points]) == pytest.approx(4.0, abs=0.05)


def test_feature_order_inside_a_layer_is_inner_outer_solid_sparse():
    model = _slice(_box())
    rank = {name: i for i, name in enumerate(FEATURE_ORDER)}
    for layer in StandardToolpathGenerator().generate_features(model, _profile(bottom_solid_layers=3)):
        ranks = [rank[c.feature_type] for c in layer.contours]
        assert ranks == sorted(ranks), (layer.layer_index, [c.feature_type for c in layer.contours])


def test_generate_toolpath_does_not_modify_the_input_model():
    model = _slice(_box(20, 20, 4), layer_height=1.0)
    before = model.model_dump()
    StandardToolpathGenerator().generate_toolpath(model, _profile())
    assert model.model_dump() == before
    assert all(c.feature_type == "boundary" for l in model.layers for c in l.contours)


def test_invalid_contours_are_skipped_with_a_warning_never_emitted_as_perimeter(caplog):
    bowtie = SpatialContour(points=[(0, 0, 1), (10, 10, 1), (10, 0, 1), (0, 10, 1), (0, 0, 1)],
                            normals=[(0, 0, 1)] * 5, feature_type="boundary")
    open_path = SpatialContour(points=[(0, 0, 1), (5, 0, 1), (9, 3, 1)], normals=[(0, 0, 1)] * 3,
                               feature_type="boundary")
    bad = UniversalLayer(layer_index=1, z_height=1.0, contours=[bowtie, open_path])
    model = UniversalSlicedModel(layers=[_square_layer(0, 20), bad])
    with caplog.at_level(logging.WARNING):
        traj = StandardToolpathGenerator().generate_toolpath(model, _profile())
    assert {w.layer_index for w in traj.waypoints} == {0}                 # layer 1 produced nothing
    assert "boundary" not in {w.feature_type for w in traj.waypoints}
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "self-intersecting" in messages and "open contour" in messages and "Layer 1" in messages


def test_base_signature_matches_the_implementation():
    base = list(inspect.signature(BaseToolpathGenerator.generate_toolpath).parameters)
    impl = list(inspect.signature(StandardToolpathGenerator.generate_toolpath).parameters)
    assert base == impl == ["self", "model", "profile"]


# ---------------------------------------------------------------------------------------------
# 4. travel counts (reported, and guarded against regressions)
# ---------------------------------------------------------------------------------------------
def _travels_per_layer(traj):
    out = {}
    for w in traj.waypoints:
        out.setdefault(w.layer_index, 0)
        out[w.layer_index] += 1 if w.is_travel_move else 0
    return out


def test_travel_counts_per_layer(capsys):
    gen = StandardToolpathGenerator()
    prof = _profile()

    cube = _slice(_box())
    cube_travels = _travels_per_layer(gen.generate_toolpath(cube, prof))
    # inner wall + outer wall + ONE infill/skin chain = 3 contours -> 3 travels (incl. layer entry)
    assert set(cube_travels.values()) == {3}

    ring = trimesh.creation.annulus(r_min=8.0, r_max=15.0, height=10.0, sections=96)
    ring.apply_translation([0, 0, 5])
    ring_travels = _travels_per_layer(gen.generate_toolpath(_slice(ring), prof))
    # 2 rings x 2 walls + the infill chains (2 for the sparse core)
    assert ring_travels[5] == 6
    assert max(ring_travels.values()) <= 8
    with capsys.disabled():
        print(f"\ntravels/layer solid cube: {sorted(set(cube_travels.values()))}, "
              f"hollow cylinder: {ring_travels}")


# ---------------------------------------------------------------------------------------------
# 5. layer_index through Stage 2 -> Stage 3
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("topology", ["planar_3axis_xyz", "trunnion_table_xyzbc", "swivel_head_xyzbc"])
def test_layer_index_reaches_the_machine_states(topology):
    mesh = trimesh.creation.cylinder(radius=10, height=6, sections=48)
    mesh.apply_translation([0, 0, 3])
    model = UniversalFieldSlicerPlugin().slice(
        mesh, {"strategy": "progressive_tilt", "layer_height": 1.0, "start_tilt_deg": 0.0, "end_tilt_deg": 20.0})
    traj = StandardToolpathGenerator().generate_toolpath(model, _profile(top_solid_layers=1, bottom_solid_layers=1))

    indices = [w.layer_index for w in traj.waypoints]
    assert indices == sorted(indices)                                   # monotonic over the print
    assert set(indices) == {l.layer_index for l in model.layers}

    config = MachineConfig(kinematic_chain="x", b_axis_min=-90, b_axis_max=90,
                           c_axis_min=-36000, c_axis_max=36000, pivot_offset_z=100.0)
    states = KinematicsRegistry.get_solver(topology)().solve(traj, config).states
    assert [s.layer_index for s in states] == indices


def test_layer_index_survives_json_and_old_files_still_load(tmp_path):
    mesh = _box(20, 20, 3)
    traj = StandardToolpathGenerator().generate_toolpath(_slice(mesh), _profile())
    path = tmp_path / "cl.json"
    traj.to_json_file(str(path))
    loaded = CLDataTrajectory.from_json_file(str(path))
    assert [w.layer_index for w in loaded.waypoints] == [w.layer_index for w in traj.waypoints]
    assert max(w.layer_index for w in loaded.waypoints) == 2

    legacy = CLDataWaypoint(**{"x": 0, "y": 0, "z": 0, "feedrate": 100})     # file written before layer_index existed
    assert legacy.layer_index == 0
