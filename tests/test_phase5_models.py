"""Qt-free Phase 5 logic: hollow box, placement, layer range / playback model, view grouping."""
import numpy as np
import pytest
import trimesh

from src.common.feature_styles import FEATURE_STYLES, group_lines_by_feature
from src.common.placement import place_mesh, rotation_matrix
from src.common.primitives import hollow_box
from src.common.schemas import CLDataWaypoint, MinimalPrintProfile
from src.common.view_model import (Playback, effective_layers, head_item, layer_count, move_range,
                                   status_text)
from src.stage1_slicer.plugins.field_slicer import UniversalFieldSlicerPlugin
from src.stage2_toolpath.contour_generator import StandardToolpathGenerator

ALL = [s.key for s in FEATURE_STYLES]


# ---------------------------------------------------------------- hollow box
def test_hollow_box_is_a_valid_solid_with_the_right_volume():
    m = hollow_box(30, 24, 20, 4)
    assert m.is_watertight and m.is_winding_consistent
    assert m.volume == pytest.approx((30 * 24 - 22 * 16) * 20)
    assert np.allclose(m.bounds, [[-15, -12, 0], [15, 12, 20]])


def test_hollow_box_has_a_through_hole():
    m = hollow_box(30, 30, 20, 5)
    for z in (0.1, 10.0, 19.9):                      # every height, including near top and bottom
        section = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        polys = section.to_2D()[0].polygons_full
        assert len(polys) == 1 and len(polys[0].interiors) == 1
    # a ray along Z through the centre must not hit anything
    assert len(m.ray.intersects_location([[0, 0, -5]], [[0, 0, 1]])[0]) == 0


def test_hollow_box_validation():
    with pytest.raises(ValueError):
        hollow_box(10, 10, 10, 5)
    with pytest.raises(ValueError):
        hollow_box(10, 10, 10, 0)


def test_hollow_box_toolpath_has_inner_walls_around_the_hole():
    mesh = hollow_box(30, 30, 6, 5)
    model = UniversalFieldSlicerPlugin().slice(mesh, {"strategy": "planar", "layer_height": 1.0})
    prof = MinimalPrintProfile(layer_height=1.0, nozzle_diameter=0.4, num_perimeters=2, infill_density=0.2,
                               top_solid_layers=1, bottom_solid_layers=1)
    traj = StandardToolpathGenerator().generate_toolpath(model, prof)
    mid = [w for w in traj.waypoints if w.layer_index == 3 and not w.is_travel_move]
    assert not any(abs(w.x) < 4.5 and abs(w.y) < 4.5 for w in mid)     # nothing inside the 20x20 hole
    assert any(abs(w.x) < 10.5 and abs(w.y) < 10.5 for w in mid)       # hole wall exists


# ---------------------------------------------------------------- placement
def test_place_mesh_centres_and_drops_without_touching_the_input():
    src = trimesh.creation.box((10, 20, 30))
    src.apply_translation((50, 5, 100))
    before = src.vertices.copy()
    out = place_mesh(src)
    assert np.array_equal(src.vertices, before)
    assert np.allclose(out.bounds.mean(axis=0)[:2], 0) and out.bounds[0, 2] == pytest.approx(0)
    assert np.allclose(out.extents, [10, 20, 30])


def test_place_mesh_rotation_swaps_extents_and_still_sits_on_the_bed():
    out = place_mesh(trimesh.creation.box((10, 20, 30)), rotation_deg=(90, 0, 0))
    assert np.allclose(out.extents, [10, 30, 20])
    assert out.bounds[0, 2] == pytest.approx(0)
    out = place_mesh(trimesh.creation.box((10, 20, 30)), rotation_deg=(0, 0, 90))
    assert np.allclose(out.extents, [20, 10, 30])


def test_place_mesh_offsets_and_flags():
    box = trimesh.creation.box((10, 10, 10))
    out = place_mesh(box, center_xy=True, drop_to_bed=False, offset_xy=(5, -3))
    assert np.allclose(out.bounds.mean(axis=0), [5, -3, 0])
    out = place_mesh(box, center_xy=False, drop_to_bed=False)
    assert np.allclose(out.vertices, box.vertices)
    assert np.allclose(rotation_matrix(0, 0, 0), np.eye(4))


# ---------------------------------------------------------------- layer range / playback
def _wp(layer, x=0.0, travel=False, feature="outer_wall"):
    return CLDataWaypoint(x=x, y=0.0, z=layer * 0.2, layer_index=layer, is_travel_move=travel,
                          feature_type=feature, extrusion_volume=0.0 if travel else 1.0,
                          feedrate=1000.0, normal=(0, 0, 1), thickness=0.2)


def _items():
    out = []
    for layer in range(4):
        out += [_wp(layer, 0.0, travel=True), _wp(layer, 1.0), _wp(layer, 2.0), _wp(layer, 3.0, feature="infill")]
    return out


def test_layer_range_and_move_range():
    items = _items()
    assert layer_count(items) == 4
    assert move_range(items, 1, 2) == (4, 12)
    assert move_range(items, 3, 3) == (12, 16)
    assert move_range(items, 9, 9) == (0, 0)
    assert effective_layers(2, 1, False, 4) == (1, 1)         # lo is pulled down to hi
    assert effective_layers(0, 99, False, 4) == (0, 3)
    assert effective_layers(0, 2, True, 4) == (2, 2)          # single-layer mode
    assert effective_layers(0, 2, False, 0) == (0, -1)


def test_grouping_by_layer_range_never_bridges_the_gap():
    items = _items()
    g = group_lines_by_feature(items, ALL, layer_limit=3, layer_min=1)
    pts = [p for v in g.values() for p in v]
    zs = {round(p[2], 6) for p in pts}
    assert zs == {0.2, 0.4}                                    # only layers 1 and 2
    # the only travel segment is the one from layer 1 into layer 2; layer 0 -> 1 is outside the range
    assert len(g["travel"]) == 2 and g["travel"][0][2] == pytest.approx(0.2)


def test_playback_move_limit_draws_a_prefix():
    items = _items()
    full = group_lines_by_feature(items, ALL, 99)
    cut = group_lines_by_feature(items, ALL, 99, move_limit=6)
    assert sum(len(v) for v in cut.values()) < sum(len(v) for v in full.values())
    assert sum(len(v) for v in cut.values()) == 2 * 5          # 6 moves -> 5 segments
    assert not any(group_lines_by_feature(items, ALL, 99, move_limit=0).values())


def test_playback_state_machine():
    pb = Playback()
    pb.reset(1000)
    assert pb.shown == 1000 and not pb.playing
    assert pb.toggle() and pb.shown == 0                       # starting at the end restarts
    pb.speed = 1
    pb.advance()
    assert 0 < pb.shown <= pb.base_step
    pb.speed = 5
    before = pb.shown
    pb.advance()
    assert pb.shown - before == pb.base_step * 5
    while pb.advance():
        pass
    assert pb.shown == 1000 and not pb.playing
    pb.reset(10, show_all=False)
    assert pb.shown == 0 and pb.base_step == 1


def test_head_item_and_status_text():
    items = _items()
    assert head_item(items, 4, 0) is None
    h = head_item(items, 4, 3)
    assert h is items[6]
    txt = status_text(h, 4)
    assert "Layer 2/4" in txt and "outer_wall" in txt and " B" not in txt
    assert status_text(None, 4) == "-"
    assert "travel" in status_text(items[0], 4)
