"""
Stage 2: top/bottom skins, infill linking, hygiene, layer_index propagation.
All geometry checks are done with shapely on the generated polylines, never by
re-implementing the generator's own formulas.
"""
import logging
import math

import numpy as np
import pytest
import shapely.geometry as sg
import trimesh

from src.common.schemas import (
    CLDataTrajectory, CLDataWaypoint, MachineConfig, MinimalPrintProfile,
    SpatialContour, UniversalLayer, UniversalSlicedModel,
)
from src.stage1_slicer.plugins.field_slicer import UniversalFieldSlicerPlugin
from src.stage2_toolpath.contour_generator import FEATURE_ORDER, StandardToolpathGenerator
from src.stage3_kinematics.registry import KinematicsRegistry

SKINS = ("bottom_solid", "top_solid")


def _slice(mesh, layer_height=1.0, start_z=0.5, end_z=10.0):
    return UniversalFieldSlicerPlugin().slice(mesh, {
        "strategy": "planar", "layer_height": layer_height, "start_z": start_z, "end_z": end_z})


def _box(w=30.0, d=30.0, h=10.0):
    m = trimesh.creation.box(extents=(w, d, h))
    m.apply_translation([0, 0, h / 2])
    return m


def _profile(**kw):
    base = dict(layer_height=1.0, nozzle_diameter=0.4, num_perimeters=2, infill_density=0.2,
                top_solid_layers=3, bottom_solid_layers=3)
    base.update(kw)
    return MinimalPrintProfile(**base)


def _square_ring(half, z=0.0, cx=0.0, cy=0.0):
    pts = [(cx - half, cy - half, z), (cx + half, cy - half, z), (cx + half, cy + half, z),
           (cx - half, cy + half, z), (cx - half, cy - half, z)]
    return SpatialContour(points=pts, normals=[(0, 0, 1)] * 5, feature_type="boundary")


def _stack_model(indices, half=10.0):
    return UniversalSlicedModel(layers=[
        UniversalLayer(layer_index=i, z_height=0.5 + i, contours=[_square_ring(half, 0.5 + i)])
        for i in indices])


def _types(contours):
    return [c.feature_type for c in contours]


# ----------------------------------------------------------------------------- skins

def test_cube_skins_are_solid_top_and_bottom_and_sparse_in_between():
    model = _slice(_box())
    assert len(model.layers) == 10
    gen = StandardToolpathGenerator()
    analysis = gen.analyze_layers(model, _profile())

    for idx in range(10):
        info = analysis[idx]
        region = info.region.area
        assert region > 700
        if idx <= 2:        # bottom 3 layers: the whole region is a bottom skin
            assert info.bottom_solid.area == pytest.approx(region, rel=1e-6)
            assert info.sparse.area == pytest.approx(0.0, abs=1e-6)
        elif idx >= 7:      # top 3 layers
            assert info.top_solid.area == pytest.approx(region, rel=1e-6)
            assert info.bottom_solid.area == pytest.approx(0.0, abs=1e-6)
            assert info.sparse.area == pytest.approx(0.0, abs=1e-6)
        else:               # 3..6: nothing exposed, all sparse
            assert info.bottom_solid.area == pytest.approx(0.0, abs=1e-6)
            assert info.top_solid.area == pytest.approx(0.0, abs=1e-6)
            assert info.sparse.area == pytest.approx(region, rel=1e-6)

    features = gen.build_layer_features(model, _profile())
    for idx in range(10):
        t = set(_types(features[idx]))
        if idx <= 2:
            assert {"bottom_solid", "outer_wall", "inner_wall"} <= t and not t & {"infill", "top_solid"}
        elif idx >= 7:
            assert {"top_solid", "outer_wall", "inner_wall"} <= t and not t & {"infill", "bottom_solid"}
        else:
            assert {"infill", "outer_wall", "inner_wall"} <= t and not t & set(SKINS)


def test_zero_skin_layers_disable_the_skins():
    model = _slice(_box())
    features = StandardToolpathGenerator().build_layer_features(
        model, _profile(top_solid_layers=0, bottom_solid_layers=0))
    for contours in features.values():
        t = set(_types(contours))
        assert "infill" in t and not t & set(SKINS)


def test_skin_follows_missing_layers_by_layer_index():
    # layer 3 is missing: layer 2 has nothing above it, layer 4 nothing below it
    model = _stack_model([0, 1, 2, 4, 5])
    gen = StandardToolpathGenerator()
    analysis = gen.analyze_layers(model, _profile(top_solid_layers=1, bottom_solid_layers=1))
    assert analysis[0].bottom_solid.area == pytest.approx(analysis[0].region.area, rel=1e-6)
    assert analysis[1].sparse.area == pytest.approx(analysis[1].region.area, rel=1e-6)
    assert analysis[2].top_solid.area == pytest.approx(analysis[2].region.area, rel=1e-6)
    assert analysis[4].bottom_solid.area == pytest.approx(analysis[4].region.area, rel=1e-6)
    assert analysis[5].top_solid.area == pytest.approx(analysis[5].region.area, rel=1e-6)


def test_internal_overhang_gets_a_bottom_skin_only_where_it_is_unsupported():
    # layers 0-3: small column (10 x 10), layers 4-7: wide plate (24 x 24) that overhangs it
    layers = [UniversalLayer(layer_index=i, z_height=0.5 + i,
                             contours=[_square_ring(5.0 if i < 4 else 12.0, 0.5 + i)]) for i in range(8)]
    analysis = StandardToolpathGenerator().analyze_layers(
        UniversalSlicedModel(layers=layers), _profile(top_solid_layers=1, bottom_solid_layers=1))

    plate, column = analysis[4].region, analysis[3].region
    skin = analysis[4].bottom_solid
    assert skin.area == pytest.approx(plate.area - column.area, rel=1e-3)   # only the overhanging frame
    assert skin.intersection(column).area < 1e-3            # supported part stays sparse (only a numeric sliver)
    assert analysis[5].bottom_solid.area == pytest.approx(0.0, abs=1e-6)   # fully supported -> no skin


# ----------------------------------------------------------------------------- infill linking

def test_solid_cube_has_exactly_one_infill_contour_per_layer():
    model = _slice(_box())
    features = StandardToolpathGenerator().build_layer_features(model, _profile())
    for idx in range(10):
        fill_types = [t for t in _types(features[idx]) if t in ("infill",) + SKINS]
        assert len(fill_types) == 1, f"layer {idx}: {fill_types}"


def test_solid_skin_fill_has_no_gaps():
    # a 100% skin must lay down about region_area / line_width of path length
    model = _slice(_box())
    gen = StandardToolpathGenerator()
    prof = _profile()
    info = gen.analyze_layers(model, prof)[0]
    skin = [c for c in gen.build_layer_features(model, prof)[0] if c.feature_type == "bottom_solid"]
    length = sum(math.dist(a, b) for c in skin for a, b in zip(c.points, c.points[1:]))
    assert length * prof.nozzle_diameter == pytest.approx(info.region.area, rel=0.06)


def _holed_box_model(n_layers=10):
    """30 x 30 plate with a 10 x 10 through-hole, as stacked layers (no mesh backend needed)."""
    layers = [UniversalLayer(layer_index=i, z_height=0.5 + i,
                             contours=[_square_ring(15.0, 0.5 + i), _square_ring(5.0, 0.5 + i)])
              for i in range(n_layers)]
    return UniversalSlicedModel(layers=layers), sg.box(-5, -5, 5, 5)


def test_infill_and_skins_never_cross_a_through_hole():
    model, hole = _holed_box_model()
    features = StandardToolpathGenerator().build_layer_features(model, _profile())
    forbidden = hole.buffer(0.5)     # the walls keep 0.8 mm from the hole; fill may stay >= 0.6 mm away
    checked = 0
    for idx, contours in features.items():
        for c in contours:
            if c.feature_type in ("infill",) + SKINS:
                line = sg.LineString([(p[0], p[1]) for p in c.points])
                assert line.intersection(forbidden).length < 1e-9, f"layer {idx} {c.feature_type} crosses the hole"
                checked += 1
    assert checked >= 10
    # the hole itself is printed as walls
    walls = [c for c in features[5] if c.feature_type == "outer_wall"]
    assert len(walls) == 2             # outer boundary + hole boundary


def test_island_inside_a_hole_is_printed_as_its_own_part():
    layer = UniversalLayer(layer_index=0, z_height=0.5, contours=[
        _square_ring(20.0), _square_ring(15.0), _square_ring(5.0)])   # outer / hole / island
    model = UniversalSlicedModel(layers=[layer])
    gen = StandardToolpathGenerator()
    prof = _profile()
    info = gen.analyze_layers(model, prof)[0]

    areas = sorted(round(p.area) for p in info.parts)
    assert areas == [100, 700]                       # island 10x10; frame 40x40 - 30x30
    assert [len(p.interiors) for p in sorted(info.parts, key=lambda p: -p.area)] == [1, 0]

    features = gen.build_layer_features(model, prof)[0]
    assert _types(features).count("outer_wall") == 3      # frame outside + hole boundary + island
    gap = sg.box(-15, -15, 15, 15).difference(sg.box(-5, -5, 5, 5).buffer(0.5))
    island_fill = 0
    for c in features:
        if c.feature_type in ("infill",) + SKINS:
            line = sg.LineString([(p[0], p[1]) for p in c.points])
            assert line.intersection(gap).length < 1e-9      # nothing printed in the empty gap
            if all(abs(p[0]) < 5 and abs(p[1]) < 5 for p in c.points):
                island_fill += 1
    assert island_fill >= 1


# ----------------------------------------------------------------------------- hygiene

def test_invalid_rings_are_skipped_with_a_warning(caplog):
    bow_tie = SpatialContour(points=[(0, 0, 0), (4, 4, 0), (4, 0, 0), (0, 4, 0), (0, 0, 0)],
                             normals=[(0, 0, 1)] * 5)
    layers = [UniversalLayer(layer_index=0, z_height=0.5, contours=[bow_tie]),
              UniversalLayer(layer_index=1, z_height=1.5, contours=[_square_ring(10.0, 1.5), bow_tie])]
    model = UniversalSlicedModel(layers=layers)
    gen = StandardToolpathGenerator()
    with caplog.at_level(logging.WARNING):
        features = gen.build_layer_features(model, _profile())
    assert features[0] == []                                   # no valid ring: layer skipped, no raw contour
    assert "boundary" not in {c.feature_type for cs in features.values() for c in cs}
    assert {"outer_wall", "inner_wall"} <= set(_types(features[1]))
    assert any("skipp" in r.message for r in caplog.records)
    traj = gen.generate_toolpath(model, _profile())
    assert {w.layer_index for w in traj.waypoints} == {1}


def test_open_paths_are_printed_unchanged_as_single_walls():
    # helix / vase mode / parametric toolpaths are intentionally open: no infill, no re-offsetting
    pts = [(10 * math.cos(t), 10 * math.sin(t), 0.2 + 0.01 * t) for t in np.linspace(0, 2 * math.pi, 40)]
    normals = [(0.0, 0.1, 0.995)] * len(pts)
    path = SpatialContour(points=pts, normals=normals, thicknesses=[0.2] * len(pts), feature_type="boundary")
    model = UniversalSlicedModel(layers=[UniversalLayer(layer_index=0, z_height=0.2, contours=[path])])
    features = StandardToolpathGenerator().build_layer_features(model, _profile())[0]
    assert len(features) == 1 and features[0].feature_type == "outer_wall"
    assert features[0].points == path.points and features[0].normals == path.normals
    assert features[0].thicknesses == path.thicknesses


def test_generate_toolpath_does_not_modify_the_model():
    model = _slice(_box())
    before = model.model_dump()
    StandardToolpathGenerator().generate_toolpath(model, _profile())
    assert model.model_dump() == before


# ----------------------------------------------------------------------------- order and travels

def test_features_follow_the_documented_order_inside_a_layer():
    assert FEATURE_ORDER == ("inner_wall", "outer_wall", "bottom_solid", "top_solid", "infill")
    model = _slice(_box())
    features = StandardToolpathGenerator().build_layer_features(model, _profile())
    for idx, contours in features.items():
        ranks = [FEATURE_ORDER.index(c.feature_type) for c in contours]
        assert ranks == sorted(ranks), f"layer {idx}: {_types(contours)}"


def _layer_stats(mesh):
    model = _slice(mesh)
    gen = StandardToolpathGenerator()
    prof = _profile()
    traj = gen.generate_toolpath(model, prof)
    features = gen.build_layer_features(model, prof)
    travels = {}
    for w in traj.waypoints:
        if w.is_travel_move:
            travels[w.layer_index] = travels.get(w.layer_index, 0) + 1
    return travels, {i: len(c) for i, c in features.items()}


def test_travel_counts_per_layer(capsys):
    cube = _box()
    cyl = trimesh.creation.cylinder(radius=15, height=10, sections=64)
    cyl.apply_translation([0, 0, 5])
    hollow = trimesh.creation.annulus(r_min=8, r_max=15, height=10)
    hollow.apply_translation([0, 0, 5])

    report = {}
    for name, mesh in (("solid cube", cube), ("solid cylinder", cyl), ("hollow cylinder", hollow)):
        travels, contours = _layer_stats(mesh)
        report[name] = (travels, contours)
        for idx in contours:
            # every travel starts a contour, so there can never be more travels than contours
            assert travels.get(idx, 0) <= contours[idx]
    with capsys.disabled():
        for name, (travels, contours) in report.items():
            print(f"\n{name}: travels/layer={[travels.get(i, 0) for i in sorted(contours)]} "
                  f"total={sum(travels.values())}")

    # solid bodies: inner wall + outer wall + ONE fill contour = 3 travels per layer
    for name in ("solid cube", "solid cylinder"):
        travels, contours = report[name]
        assert all(contours[i] == 3 for i in contours)
        assert all(travels[i] == 3 for i in contours if i > 0)
    # hollow cylinder: 4 wall loops + a few fill chains (before: 8 travels/layer, 80 total)
    travels, contours = report["hollow cylinder"]
    assert sum(travels.values()) <= 70
    assert max(travels.values()) <= 7


# ----------------------------------------------------------------------------- layer_index

def test_layer_index_is_set_on_every_waypoint_and_travels_belong_to_the_layer_they_enter():
    model = _slice(_box())
    traj = StandardToolpathGenerator().generate_toolpath(model, _profile())
    idxs = [w.layer_index for w in traj.waypoints]
    assert set(idxs) == {l.layer_index for l in model.layers}
    assert idxs == sorted(idxs)
    layer_z = {l.layer_index: l.z_height for l in model.layers}
    for w in traj.waypoints:
        assert w.z == pytest.approx(layer_z[w.layer_index], abs=1e-6)   # planar: z identifies the layer


@pytest.mark.parametrize("solver_name", ["trunnion_table_xyzbc", "swivel_head_xyzbc", "planar_3axis_xyz"])
def test_solvers_pass_layer_index_through(solver_name):
    import src.stage3_kinematics.trunnion_solver   # noqa: F401  (registers the solvers)
    import src.stage3_kinematics.head_solver       # noqa: F401
    cfg = MachineConfig(kinematic_chain="x", pivot_offset_z=100.0, b_axis_min=-90.0, b_axis_max=90.0,
                        c_axis_min=-36000.0, c_axis_max=36000.0)
    wps = [CLDataWaypoint(x=i, y=2 * i, z=3.0 + i, i=0, j=0, k=1, feedrate=1000, layer_index=n)
           for i, n in enumerate([0, 0, 1, 1, 2, 7])]
    states = KinematicsRegistry.get_solver(solver_name)().solve(CLDataTrajectory(waypoints=wps), cfg).states
    assert [s.layer_index for s in states] == [0, 0, 1, 1, 2, 7]
