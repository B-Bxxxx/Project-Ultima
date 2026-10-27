"""Qt-free Phase 5 logic: placement, layer range / playback model, view grouping."""
import numpy as np
import pytest
import trimesh

from src.common.feature_styles import FEATURE_STYLES, group_lines_by_feature
from src.common.placement import place_mesh, rotation_matrix
from src.common.schemas import CLDataWaypoint, MinimalPrintProfile
from src.common.view_model import (Playback, effective_layers, head_item, layer_count, move_range,
                                   status_text)

ALL = [s.key for s in FEATURE_STYLES]


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


def test_place_mesh_z_offset_only_applies_without_drop_to_bed():
    box = trimesh.creation.box((10, 10, 10))                  # z from -5 to 5
    out = place_mesh(box, drop_to_bed=False, offset_z=7.0)
    assert out.bounds[:, 2].tolist() == pytest.approx([2.0, 12.0])
    out = place_mesh(box, drop_to_bed=True, offset_z=7.0)     # dropping wins
    assert out.bounds[0, 2] == pytest.approx(0.0)


def test_playback_slow_speeds_accumulate_fractional_steps():
    pb = Playback()
    pb.reset(900, show_all=False)                       # base step 3
    pb.speed = 0.25
    pb.toggle()
    got = []
    for _ in range(8):
        pb.advance()
        got.append(pb.shown)
    assert got[-1] == 6 and got == sorted(got)           # 0.75 / tick: 0, 1, 2, 3 ... carried over, never stuck
    pb.reset(900, show_all=False)
    pb.speed = 0.5
    pb.toggle()
    for _ in range(10):
        pb.advance()
    assert pb.shown == 15
