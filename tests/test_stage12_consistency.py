import copy
import json

import numpy as np
import pytest
import trimesh

from src.common.schemas import MinimalPrintProfile, UniversalSlicedModel
from src.stage1_slicer.math_strategies import build_strategy
from src.stage1_slicer.plugins.field_slicer import UniversalFieldSlicerPlugin
from src.stage2_toolpath.contour_generator import StandardToolpathGenerator


def _offset_cylinder(z0=3.0, height=6.0, radius=10.0):
    """A part that does NOT start at z=0, so mesh-derived defaults matter."""
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=64)
    mesh.apply_translation([0, 0, z0 + height / 2.0])
    return mesh


STRATEGY_PARAMS = {
    "progressive_tilt": {"strategy": "progressive_tilt", "layer_height": 1.0,
                         "start_tilt_deg": 0.0, "end_tilt_deg": 20.0},
    "conical": {"strategy": "conical", "layer_height": 1.0, "cone_angle_deg": 15.0,
                "transition_height": 4.0, "refine_mesh": False},
    "custom_expr": {"strategy": "custom_expr", "layer_height": 1.0,
                    "expression": "0.3 * np.sin(0.2 * x) + 0.1 * y",
                    "transition_height": 4.0},
}


@pytest.mark.parametrize("name", list(STRATEGY_PARAMS))
def test_stage1_metadata_holds_resolved_parameters_and_roundtrips(name, tmp_path):
    mesh = _offset_cylinder()
    params = dict(STRATEGY_PARAMS[name])          # NOTE: no start_z / end_z / pivot_y given
    model = UniversalFieldSlicerPlugin().slice(mesh, params)

    md = model.metadata
    assert md["start_z"] == pytest.approx(3.0)    # mesh-derived default, not 0.0
    assert "end_z" in md and md["end_z"] > md["start_z"]
    if name == "progressive_tilt":
        assert md["pivot_y"] == pytest.approx(-10.0)   # min y of the mesh
    # the caller's dict must not be mutated
    assert "start_z" not in params

    path = tmp_path / "stage1.json"
    model.to_json_file(str(path))
    json.loads(path.read_text())                  # valid JSON, no numpy leftovers
    assert UniversalSlicedModel.from_json_file(str(path)).metadata == json.loads(
        json.dumps(md))


@pytest.mark.parametrize("name", list(STRATEGY_PARAMS))
def test_stage1_and_stage2_strategies_agree_layer_by_layer(name):
    mesh = _offset_cylinder()
    params = dict(STRATEGY_PARAMS[name])
    model = UniversalFieldSlicerPlugin().slice(mesh, params)

    stage1 = build_strategy(params["strategy"], mesh, params)                  # real mesh
    stage2 = build_strategy(model.metadata["strategy"], trimesh.Trimesh(), model.metadata)

    assert stage2.get_layer_count() == stage1.get_layer_count()
    assert len(model.layers) > 3
    probes = [(7.0, 0.0, 0.0), (-3.0, 5.0, 0.0), (0.5, -9.0, 0.0), (0.0, 0.0, 0.0)]
    for layer in model.layers:
        idx = layer.layer_index
        z_nominal = stage1.start_z + idx * stage1.layer_height
        assert stage2.get_blend_weight(z_nominal) == pytest.approx(stage1.get_blend_weight(z_nominal))
        assert layer.z_height == pytest.approx(z_nominal) or name == "progressive_tilt"
        for p in probes:
            assert stage2.compute_normal(p, idx) == pytest.approx(stage1.compute_normal(p, idx), abs=1e-9)
            assert stage2.compute_thickness(p, idx) == pytest.approx(stage1.compute_thickness(p, idx), abs=1e-9)
            assert stage2.project_to_layer(p[0], p[1], idx) == pytest.approx(
                stage1.project_to_layer(p[0], p[1], idx), abs=1e-9)


def test_progressive_tilt_stage2_points_lie_on_the_tilted_plane():
    mesh = _offset_cylinder(z0=2.0, height=6.0)
    params = dict(STRATEGY_PARAMS["progressive_tilt"])
    model = UniversalFieldSlicerPlugin().slice(mesh, params)
    strat = build_strategy("progressive_tilt", trimesh.Trimesh(), model.metadata)

    profile = MinimalPrintProfile(layer_height=1.0, nozzle_diameter=0.4, num_perimeters=2,
                                  infill_density=0.2)
    work = copy.deepcopy(model)
    StandardToolpathGenerator().generate_toolpath(work, profile)   # replaces contours with features

    checked = 0
    types = set()
    for layer in work.layers:
        t = strat._get_tilt_for_layer(layer.layer_index)
        z0 = strat.start_z + layer.layer_index * strat.layer_height
        n = np.array([0.0, -np.sin(t), np.cos(t)])          # plane normal (independent of project_to_layer)
        o = np.array([0.0, strat.pivot_y, z0])
        for c in layer.contours:
            types.add(c.feature_type)
            for p in c.points:
                assert abs(np.dot(n, np.array(p) - o)) < 1e-6
                checked += 1
    assert checked > 200
    assert {"outer_wall", "inner_wall"} <= types
    # and the tilt really is non-trivial: some layer spans several mm in z (not horizontal)
    spans = [max(p[2] for c in l.contours for p in c.points) - min(p[2] for c in l.contours for p in c.points)
             for l in work.layers]
    assert spans[0] < 1e-6 and max(spans) > 2.0   # layer 0 is flat, later layers tilt


def _external_field_model(tmp_path, transition_height=0.0):
    box = trimesh.creation.box(extents=(20, 20, 10))
    box.apply_translation([0, 0, 5])
    vertices, faces = trimesh.remesh.subdivide_to_size(box.vertices, box.faces, max_edge=1.0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    field = 0.05 * mesh.vertices[:, 0]                      # f(x, y) = 0.05 x (numpy array!)
    params = {"strategy": "external_field", "layer_height": 1.0,
              "vertex_deformations": field, "transition_height": transition_height}
    model = UniversalFieldSlicerPlugin().slice(mesh, params)
    return model, params


def test_external_field_end_to_end_stage1_json_stage2(tmp_path):
    model, params = _external_field_model(tmp_path)

    assert "vertex_deformations" not in model.metadata          # not stored (can be huge)
    json.dumps(model.metadata)                                   # JSON-serializable

    path = tmp_path / "ext.json"
    model.to_json_file(str(path))
    loaded = UniversalSlicedModel.from_json_file(str(path))

    profile = MinimalPrintProfile(layer_height=1.0, nozzle_diameter=0.4, num_perimeters=2,
                                  infill_density=0.2)
    work = copy.deepcopy(loaded)
    traj = StandardToolpathGenerator().generate_toolpath(work, profile)
    assert len(traj.waypoints) > 100
    assert all(np.isfinite([w.x, w.y, w.z, w.i, w.j, w.k]).all() for w in traj.waypoints)

    # Geometry check: the layer surface is z' = c - f(x,y) = c - 0.05 x
    for layer in work.layers:
        zp = layer.z_height
        for c in layer.contours:
            if c.feature_type != "outer_wall":
                continue
            for p in c.points:
                assert p[2] == pytest.approx(zp - 0.05 * p[0], abs=0.1)


def test_solve_forward_z_is_exact_inverse_of_undeform():
    mesh = _offset_cylinder()
    strat = build_strategy("conical", mesh, dict(STRATEGY_PARAMS["conical"]))
    rng = np.random.default_rng(0)
    z = rng.uniform(3.0, 9.0, 200)
    f = rng.uniform(0.0, 3.0, 200)
    zp = strat._solve_forward_z(z, f)
    w = np.array([strat.get_blend_weight(v) for v in zp])
    np.testing.assert_allclose(zp - w * f, z, atol=1e-5)


def test_solve_forward_z_raises_clear_error_when_not_converging():
    # Vertices inside the transition zone + a strongly NEGATIVE field make the
    # fixed-point iteration oscillate between two values (no solution found).
    mesh = _offset_cylinder()
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=1.0)
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    params = {"strategy": "conical", "layer_height": 1.0, "cone_angle_deg": -70.0,
              "transition_height": 4.0, "refine_mesh": False}      # |f| up to ~27 >> transition_height
    with pytest.raises(ValueError, match="did not converge"):
        build_strategy("conical", mesh, params)
