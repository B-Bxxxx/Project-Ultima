"""Stage 0 function import (expression file, data file) and the grid_field strategy."""
import json

import numpy as np
import pytest
import trimesh

from src.common.schemas import MinimalPrintProfile
from src.stage1_slicer.function_import import (
    ExpressionField, GridField, load_data_file, load_expression_file, preview_surface,
    resample_scattered,
)
from src.stage1_slicer.math_strategies import build_strategy
from src.stage1_slicer.plugins.field_slicer import UniversalFieldSlicerPlugin
from src.stage2_toolpath.contour_generator import StandardToolpathGenerator


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


# ------------------------------------------------------------------ A. expression file
def test_expression_file_text_comments_and_wrapping(tmp_path):
    p = _write(tmp_path, "f.txt", "# bump\n// another comment\n\n0.5 * np.sin(x / 5)\n  + 0.1 * y\n")
    assert load_expression_file(p) == "0.5 * np.sin(x / 5) + 0.1 * y"


def test_expression_file_json(tmp_path):
    p = _write(tmp_path, "f.json", json.dumps({"expression": " 0.1 * x "}))
    assert load_expression_file(p) == "0.1 * x"
    for bad in ("[1]", '{"expr": "x"}', '{"expression": 3}', "{oops"):
        with pytest.raises(ValueError):
            load_expression_file(_write(tmp_path, "bad.json", bad))


@pytest.mark.parametrize("text", [
    "__import__('os').system('echo hi')",
    "().__class__.__bases__",
    "open('/etc/passwd')",
    "x ** 99",
    "",
    "# only a comment\n",
])
def test_expression_file_rejects_unsafe_or_empty(tmp_path, text):
    with pytest.raises(ValueError):
        load_expression_file(_write(tmp_path, "bad.txt", text))


def test_expression_file_size_limit(tmp_path):
    with pytest.raises(ValueError, match="too large"):
        load_expression_file(_write(tmp_path, "big.txt", "x + " * 5000 + "1"))


def test_expression_field_evaluates_arrays_and_scalars():
    f = ExpressionField("0.1 * x + 2")
    assert f.evaluate(10.0, 3.0) == pytest.approx(3.0)
    assert np.allclose(f.evaluate(np.array([0.0, 10.0]), np.array([0.0, 0.0])), [2.0, 3.0])
    with pytest.raises(ValueError):
        ExpressionField("__import__('os')")


# ------------------------------------------------------------------ GridField
def test_grid_field_is_exact_for_a_plane_and_clamps_outside():
    xs, ys = np.linspace(-10, 10, 5), np.linspace(-5, 5, 3)
    gx, gy = np.meshgrid(xs, ys)
    g = GridField(xs, ys, 0.1 * gx - 0.2 * gy)
    for x, y in [(0.3, 0.7), (-9.9, 4.9), (10, 5), (-10, -5)]:
        assert g.evaluate(x, y) == pytest.approx(0.1 * x - 0.2 * y)
    assert g.evaluate(99, 0) == pytest.approx(g.evaluate(10, 0))           # clamped
    arr = g.evaluate(np.array([[0.0, 1.0]]), np.array([[0.0, 1.0]]))
    assert arr.shape == (1, 2)


def test_grid_field_validation_and_params_roundtrip():
    with pytest.raises(ValueError):
        GridField([0, 1], [0, 1], [[0, 1, 2]])
    with pytest.raises(ValueError):
        GridField([1, 0], [0, 1], [[0, 0], [0, 0]])
    with pytest.raises(ValueError):
        GridField([0, 1], [0, 1], [[0, np.nan], [0, 0]])
    g = GridField([0, 1, 2], [0, 1], [[0, 1, 2], [3, 4, 5]])
    g2 = GridField.from_params(json.loads(json.dumps(g.to_params())))     # JSON-safe
    assert np.array_equal(g2.values, g.values)
    with pytest.raises(ValueError, match="grid_x"):
        GridField.from_params({})
    assert g.value_range == (0.0, 5.0) and g.bounds == (0.0, 2.0, 0.0, 1.0)


# ------------------------------------------------------------------ B. data file
def test_data_file_regular_grid_with_header_and_any_order(tmp_path):
    rows = [(x, y, 2.0 * x + y) for y in (0, 1, 2) for x in (0, 1, 2, 3)]
    rows.reverse()
    p = _write(tmp_path, "g.csv", "x,y,f\n" + "\n".join(",".join(map(str, r)) for r in rows))
    g = load_data_file(p)
    assert g.values.shape == (3, 4)
    assert g.evaluate(1.5, 0.5) == pytest.approx(2 * 1.5 + 0.5)


def test_data_file_scattered_points_are_resampled(tmp_path):
    rng = np.random.default_rng(1)
    pts = rng.uniform(-10, 10, size=(400, 2))
    f = 0.05 * pts[:, 0] + 0.02 * pts[:, 1]
    p = _write(tmp_path, "s.txt", "# scattered\n" + "\n".join(f"{a} {b} {c}" for (a, b), c in zip(pts, f)))
    g = load_data_file(p, resolution=30)
    for x, y in [(0, 0), (5, -5), (-7, 3)]:
        assert g.evaluate(x, y) == pytest.approx(0.05 * x + 0.02 * y, abs=0.03)


def test_resample_reproduces_exact_sample_points():
    pts = np.array([[0, 0, 1.0], [10, 0, 2.0], [0, 10, 3.0], [10, 10, 4.0], [5, 5, 9.0]])
    g = resample_scattered(pts, resolution=3)
    assert g.evaluate(5, 5) == pytest.approx(9.0)
    assert g.evaluate(10, 10) == pytest.approx(4.0)


@pytest.mark.parametrize("text", [
    "1 2\n3 4\n5 6\n7 8\n",                  # only 2 columns
    "0 0 1\n1 1 2\n",                       # too few points
    "0 0 1\n0 1 2\n0 2 3\n0 3 4\n",         # all x equal
    "x y f\n0 0 1\n1 0 2\n0 1 3\n1 1 oops\n",
    "0 0 1\n1 0 2\n0 1 3\n1 1 nan\n",
])
def test_data_file_errors(tmp_path, text):
    with pytest.raises(ValueError):
        load_data_file(_write(tmp_path, "bad.csv", text))


def test_preview_surface_stats_for_both_field_kinds():
    xs, ys, z, (lo, hi) = preview_surface(ExpressionField("0.1 * x"), (-10, 10, -5, 5), n=11)
    assert z.shape == (11, 11) and (lo, hi) == (pytest.approx(-1.0), pytest.approx(1.0))
    _, _, z, (lo, hi) = preview_surface(ExpressionField("0.5"), (-10, 10, -5, 5), n=5)
    assert z.shape == (5, 5) and lo == hi == pytest.approx(0.5)
    g = GridField([0, 1], [0, 1], [[0, 1], [2, 3]])
    _, _, z, (lo, hi) = preview_surface(g, g.bounds, n=4)
    assert (lo, hi) == (0.0, 3.0)


# ------------------------------------------------------------------ strategy end to end
def _plane_grid():
    xs, ys = np.linspace(-20, 20, 9), np.linspace(-20, 20, 9)
    gx, gy = np.meshgrid(xs, ys)
    return GridField(xs, ys, 0.05 * gx)


MESH = lambda: trimesh.creation.box(extents=(20, 20, 6))


def _params(**kw):
    p = {"layer_height": 1.0, "transition_height": 3.0}
    p.update(kw)
    return p


def test_grid_field_strategy_matches_the_equivalent_expression():
    mesh = MESH(); mesh.apply_translation((0, 0, 3))
    grid = build_strategy("grid_field", mesh, _params(**_plane_grid().to_params()))
    expr = build_strategy("custom_expr", mesh, _params(expression="0.05 * x"))
    for x, y, z in [(0, 0, 2.0), (7.3, -2.0, 1.0), (-9.0, 5.0, 3.0)]:
        assert np.allclose(grid.undeform_point((x, y, z), 2), expr.undeform_point((x, y, z), 2))
        assert np.allclose(grid.compute_normal((x, y, z), 2), expr.compute_normal((x, y, z), 2), atol=1e-6)
    assert grid.get_layer_count() == expr.get_layer_count()


def test_grid_field_requires_its_parameters():
    with pytest.raises(ValueError, match="grid_x"):
        build_strategy("grid_field", MESH(), _params())


def test_grid_field_full_pipeline_with_json_safe_metadata():
    mesh = MESH(); mesh.apply_translation((0, 0, 3))
    params = _params(strategy="grid_field", **_plane_grid().to_params())
    model = UniversalFieldSlicerPlugin().slice(mesh, params)
    json.dumps(model.metadata)                                   # must be serialisable
    prof = MinimalPrintProfile(layer_height=1.0, nozzle_diameter=0.4, num_perimeters=1,
                               infill_density=0.2, top_solid_layers=1, bottom_solid_layers=1)
    traj = StandardToolpathGenerator().generate_toolpath(model, prof)
    assert len(traj.waypoints) > 100

    ref = UniversalFieldSlicerPlugin().slice(mesh, _params(strategy="custom_expr", expression="0.05 * x"))
    ref_traj = StandardToolpathGenerator().generate_toolpath(ref, prof)
    assert len(ref_traj.waypoints) == len(traj.waypoints)
    dz = max(abs(a.z - b.z) for a, b in zip(traj.waypoints, ref_traj.waypoints))
    assert dz < 1e-3
    # upper layers (beyond the transition) really are tilted: z falls with x by 0.05 per mm
    top = [w for w in traj.waypoints if w.layer_index == 4 and not w.is_travel_move]
    xs = np.array([w.x for w in top]); zs = np.array([w.z for w in top])
    assert np.polyfit(xs, zs, 1)[0] == pytest.approx(-0.05, abs=0.005)
