"""Line-type colours/legend (Qt-free) and the grouping used by the viewer."""
import trimesh

from src.common.feature_styles import FEATURE_STYLES, STYLE_BY_KEY, group_lines_by_feature
from src.common.schemas import MinimalPrintProfile
from src.stage1_slicer.plugins.field_slicer import UniversalFieldSlicerPlugin
from src.stage2_toolpath.contour_generator import FEATURE_ORDER, StandardToolpathGenerator


def _trajectory():
    mesh = trimesh.creation.box(extents=(20, 20, 6))
    mesh.apply_translation((0, 0, 3))
    model = UniversalFieldSlicerPlugin().slice(mesh, {"strategy": "planar", "layer_height": 1.0})
    prof = MinimalPrintProfile(layer_height=1.0, nozzle_diameter=0.4, num_perimeters=2,
                               infill_density=0.2, top_solid_layers=1, bottom_solid_layers=1)
    return StandardToolpathGenerator().generate_toolpath(model, prof)


def test_colours_are_pairwise_distinct():
    colours = [s.color for s in FEATURE_STYLES]
    assert len(set(colours)) == len(colours)
    assert len({s.hex for s in FEATURE_STYLES}) == len(FEATURE_STYLES)


def test_every_stage2_feature_type_has_a_style():
    for f in FEATURE_ORDER:
        assert f in STYLE_BY_KEY
    assert "travel" in STYLE_BY_KEY


def test_skin_differs_from_walls_and_infill():
    skin = {STYLE_BY_KEY[k].color for k in ("bottom_solid", "top_solid")}
    other = {STYLE_BY_KEY[k].color for k in ("outer_wall", "inner_wall", "infill", "travel")}
    assert not skin & other


def test_labels_and_hex():
    for s in FEATURE_STYLES:
        assert s.label and s.description
        assert len(s.hex) == 7 and s.hex.startswith("#")


def test_grouping_respects_layer_limit_and_visibility():
    traj = _trajectory()
    all_keys = [s.key for s in FEATURE_STYLES]
    full = group_lines_by_feature(traj.waypoints, all_keys, 10**6)
    assert full["bottom_solid"] and full["top_solid"] and full["outer_wall"]
    assert all(len(v) % 2 == 0 for v in full.values())

    only_skin = group_lines_by_feature(traj.waypoints, ["top_solid"], 10**6)
    assert only_skin["top_solid"] and not only_skin["outer_wall"] and not only_skin["bottom_solid"]

    first_layer = group_lines_by_feature(traj.waypoints, all_keys, 1)
    assert first_layer["bottom_solid"] and not first_layer["top_solid"]
    assert max(p[2] for v in first_layer.values() for p in v) < 1.0 + 1e-6
    assert not any(group_lines_by_feature(traj.waypoints, all_keys, 0).values())
