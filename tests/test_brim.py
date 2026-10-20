"""Brim around the first layer and the build-volume fit check."""
import os
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.common.schemas import CLDataWaypoint, MinimalPrintProfile
from src.stage1_slicer.registry import PluginRegistry
from src.stage2_toolpath.contour_generator import StandardToolpathGenerator
from src.stage3_kinematics.machine_file import MachineSpec


def _model():
    mesh = trimesh.creation.box(extents=(20, 20, 2))
    mesh.apply_translation((0, 0, 1))
    p = {"layer_height": 0.2, "strategy": "planar"}
    return PluginRegistry.get_plugin("universal_field_slicer")().slice(mesh, p)


def _features(brim):
    prof = MinimalPrintProfile(brim_width=brim, num_perimeters=1)
    return StandardToolpathGenerator().generate_features(_model(), prof)


def test_no_brim_by_default():
    assert MinimalPrintProfile().brim_width == 0.0
    assert all(c.feature_type != "brim" for l in _features(0.0) for c in l.contours)


def test_brim_only_on_the_first_layer_and_outside_the_part():
    layers = _features(2.0)
    brims = [[c for c in l.contours if c.feature_type == "brim"] for l in layers]
    assert len(brims[0]) == 5                       # 2.0 / 0.4 nozzle widths
    assert all(not b for b in brims[1:])
    xs = [p[0] for c in brims[0] for p in c.points]
    assert max(xs) == pytest.approx(10 + 5 * 0.4, abs=0.01)    # outermost loop, 5 nozzle widths outside
    assert min(max(abs(p[0]), abs(p[1])) for c in brims[0] for p in c.points) > 10.3
    assert layers[0].contours[0].feature_type == "brim"        # printed before the walls
    assert brims[0][0].points[0] == pytest.approx(brims[0][0].points[-1], abs=1e-9)     # closed loops


def test_brim_narrower_than_a_nozzle_gives_nothing():
    assert all(c.feature_type != "brim" for l in _features(0.3) for c in l.contours)


def test_build_volume_check_compares_the_size_of_the_part():
    spec = MachineSpec(name="m", kinematics="cartesian", axes=["X", "Y", "Z"],
                       build_volume={"x": [0, 220], "y": [0, 100], "z": [0, 250]})
    def wp(x, y, z):
        return CLDataWaypoint(x=x, y=y, z=z, i=0, j=0, k=1, extrusion_volume=0.1, feedrate=1000,
                              is_travel_move=False, feature_type="outer_wall", layer_index=0)
    ok = [wp(-50, -40, 0), wp(50, 40, 20)]
    assert spec.fit_warnings(ok) == []
    msgs = spec.fit_warnings([wp(-60, -60, 0), wp(60, 60, 20)])      # 120 mm in Y > 100 mm
    assert len(msgs) == 1 and "Y" in msgs[0]
    assert MachineSpec(name="m", kinematics="c", axes=["X"]).fit_warnings(ok) == []


def test_skirt_is_free_standing_and_outside_the_brim():
    prof = MinimalPrintProfile(brim_width=1.2, skirt_loops=2, skirt_distance=3.0, num_perimeters=1)
    layers = StandardToolpathGenerator().generate_features(_model(), prof)
    kinds = [c.feature_type for c in layers[0].contours]
    assert kinds[:2] == ["skirt", "skirt"] and kinds.count("brim") == 3          # skirt first, then brim
    sk = [c for c in layers[0].contours if c.feature_type == "skirt"]
    ext = [max(max(abs(p[0]), abs(p[1])) for p in c.points) for c in sk]
    assert ext[0] > ext[1] > 10 + 1.2 + 3.0 - 1e-6                              # outermost first, beyond brim + gap
    assert all(c.feature_type != "skirt" for l in layers[1:] for c in l.contours)
    assert MinimalPrintProfile().skirt_loops == 0


def test_walls_of_one_boundary_are_printed_in_one_go_outline_and_hole_do_not_alternate():
    ring = trimesh.creation.annulus(r_min=6, r_max=15, height=2, sections=96)
    ring.apply_translation((0, 0, 1))
    model = PluginRegistry.get_plugin("universal_field_slicer")().slice(ring, {"strategy": "planar", "layer_height": 0.2})
    layers = StandardToolpathGenerator().generate_features(model, MinimalPrintProfile(num_perimeters=4))
    walls = [c for c in layers[2].contours if c.feature_type in ("inner_wall", "outer_wall")]
    assert len(walls) == 8
    radii = [float(np.mean([np.hypot(p[0], p[1]) for p in c.points])) for c in walls]
    boundary = ["out" if r > 10.5 else "hole" for r in radii]
    assert sum(a != b for a, b in zip(boundary, boundary[1:])) == 1               # one jump between the two boundaries
    first = radii[:4] if boundary[0] == "out" else radii[:4]
    out_r = [r for r, b in zip(radii, boundary) if b == "out"]
    hole_r = [r for r, b in zip(radii, boundary) if b == "hole"]
    assert out_r == sorted(out_r) and hole_r == sorted(hole_r, reverse=True)     # innermost wall first, the surface wall last
    types = [c.feature_type for c in walls]
    assert types[3] == "outer_wall" and types[7] == "outer_wall" and types.count("outer_wall") == 2
