import numpy as np
import pytest
import trimesh

from src.common.schemas import MinimalPrintProfile
from src.stage1_slicer.centerline import Centerline
from src.stage1_slicer.centerline_strategy import CenterlineStrategy
from src.stage1_slicer.math_strategies import build_strategy
from src.stage1_slicer.plugins.field_slicer import UniversalFieldSlicerPlugin
from src.stage2_toolpath.contour_generator import StandardToolpathGenerator
from tests.test_centerline import _tube

R_TUBE, R_BEND, H = 4.0, 20.0, 0.5


def _elbow_up():
    """Vertical for 20 mm (bed = the cap at z = 0), then a quarter circle of radius 20 towards +X, then 12 mm horizontal."""
    straight = [[0, 0, z] for z in np.linspace(0, 20, 21)]
    arc = [[R_BEND - R_BEND * np.cos(a), 0, 20 + R_BEND * np.sin(a)] for a in np.linspace(0, np.pi / 2, 60)[1:]]
    end = [[R_BEND + x, 0, 20 + R_BEND] for x in np.linspace(0, 12, 13)[1:]]
    return np.array(straight + arc + end)


@pytest.fixture(scope="module")
def elbow():
    return _tube(_elbow_up(), R_TUBE)


def test_straight_tube_gives_flat_layers_perpendicular_to_the_axis():
    tube = _tube([[0, 0, z] for z in np.linspace(0, 20, 21)], R_TUBE)
    model = UniversalFieldSlicerPlugin().slice(tube, {"strategy": "centerline", "layer_height": 1.0})
    assert len(model.layers) >= 19
    for layer in model.layers:
        for c in layer.contours:
            P = np.array(c.points)
            assert np.ptp(P[:, 2]) < 1e-6                                    # a flat layer ...
            assert np.allclose(np.array(c.normals), [0, 0, 1], atol=1e-6)    # ... with the nozzle along the axis
            assert np.allclose(c.thicknesses, 1.0, atol=1e-3)


def test_layers_of_a_bent_tube_are_perpendicular_to_the_axis_and_planar(elbow):
    strat = build_strategy("centerline", elbow, {"layer_height": H})
    model = UniversalFieldSlicerPlugin().slice(elbow, {"strategy": "centerline", "layer_height": H})
    cl = strat.centerline
    checked = 0
    for layer in model.layers[::6]:
        for c in layer.contours:
            P = np.array(c.points)
            n = np.array(c.normals[0])
            s = (layer.layer_index + 1) * H - H / 2                           # the contour is cut at the mid-plane of the layer
            assert np.ptp((P - P[0]) @ cl.tangent_at(s)) < 0.02               # every contour lies in a plane perpendicular to the axis
            assert np.allclose(n, cl.tangent_at(s + H / 2), atol=0.03)        # and the nozzle follows the axis
            checked += 1
    assert checked > 10


def test_thickness_is_thinner_inside_and_thicker_outside_the_bend(elbow):
    strat = build_strategy("centerline", elbow, {"layer_height": H})
    model = UniversalFieldSlicerPlugin().slice(elbow, {"strategy": "centerline", "layer_height": H})
    mid = next(l for l in model.layers if 28 < l.layer_index * H < 36)       # inside the arc (s = 20 .. 51)
    t = np.concatenate([c.thicknesses for c in mid.contours])
    assert t.min() < H * 0.85 and t.max() > H * 1.15
    # outer wall: thickness ~ h (1 + d / R), inner ~ h (1 - d / R) with d = tube radius
    assert abs(t.max() - H * (1 + R_TUBE / R_BEND)) < 0.06 and abs(t.min() - H * (1 - R_TUBE / R_BEND)) < 0.06


def test_chart_round_trip(elbow):
    strat = build_strategy("centerline", elbow, {"layer_height": H})
    uv = np.array([[3.0, -2.0], [0.0, 0.0], [-3.5, 1.0]])
    for idx in (3, 40, 70, 100):
        P = strat.project_points(uv, idx)
        s = strat._s_top(idx)
        d = P - strat._at(strat.centerline.points, np.full(3, s))
        assert np.allclose(d @ strat.normal_at_s(s), 0.0, atol=1e-6)         # on the layer plane
        back = strat.chart_points(strat.position(np.full(3, s - H / 2), uv[:, 0], uv[:, 1]), idx)
        assert np.allclose(back, uv, atol=1e-6)


def test_max_tilt_smooths_the_normals_but_never_beyond_the_limit(elbow):
    exact = build_strategy("centerline", elbow, {"layer_height": H})
    soft = build_strategy("centerline", elbow, {"layer_height": H, "max_tilt_deg": 8.0})
    T = exact.centerline.tangents
    dev = np.degrees(np.arccos(np.clip((soft.normals * T).sum(1), -1, 1)))
    assert dev.max() <= 8.0 + 0.2 and dev.max() > 3.0                          # uses the freedom, keeps within it
    assert np.allclose(soft.normals[0], T[0], atol=1e-6) and np.allclose(soft.normals[-1], T[-1], atol=1e-6)
    rate = lambda m: np.abs(np.diff(m, axis=0)).sum()
    assert rate(soft.normals) < rate(exact.normals)                           # less rotation


def test_resolved_params_rebuild_the_same_strategy(elbow):
    strat = build_strategy("centerline", elbow, {"layer_height": H, "max_tilt_deg": 5.0})
    rp = strat.get_resolved_params()
    rebuilt = build_strategy("centerline", trimesh.Trimesh(), {"layer_height": H, **rp})
    assert rebuilt.get_layer_count() == strat.get_layer_count()
    uv = np.array([[2.0, 1.0]])
    assert np.allclose(strat.project_points(uv, 50), rebuilt.project_points(uv, 50), atol=0.05)


def test_stage2_plans_in_the_chart_and_lands_on_the_part(elbow):
    model = UniversalFieldSlicerPlugin().slice(elbow, {"strategy": "centerline", "layer_height": H})
    prof = MinimalPrintProfile(layer_height=H, nozzle_diameter=0.4, num_perimeters=2, infill_density=0.2,
                               top_solid_layers=0, bottom_solid_layers=0, thickness_model="nozzle_axis")
    traj = StandardToolpathGenerator().generate_toolpath(model, prof)
    P = np.array([[w.x, w.y, w.z] for w in traj.waypoints if w.feature_type == "outer_wall" and not w.is_travel_move])
    assert len(P) > 500
    d = trimesh.proximity.signed_distance(elbow, P[::7])
    assert abs(np.median(d)) < 0.05 and d.min() > -0.1                         # the outer wall runs on the surface
    assert {w.feature_type for w in traj.waypoints} >= {"outer_wall", "inner_wall"}


def test_function_file_centerline_kind():
    from src.stage1_slicer.function_file import FunctionSpec, FunctionFileError
    d = {"format": "cam-function", "version": 2, "name": "T", "centerline": "auto",
         "parameters": {"max_tilt_deg": {"label": "m", "unit": "deg", "default": 4.0, "min": 0.0, "max": 30.0, "step": 1.0}}}
    spec = FunctionSpec.from_dict(d)
    assert spec.kind == "centerline"
    assert spec.to_slicer_params() == {"strategy": "centerline", "max_tilt_deg": 4.0}
    bad = dict(d, parameters={})
    with pytest.raises(FunctionFileError):
        FunctionSpec.from_dict(bad)
