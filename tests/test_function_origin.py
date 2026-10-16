"""The function origin is the centre of the part's footprint: moving the part on the bed must not change the slice."""
import os
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.common.schemas import MinimalPrintProfile
from src.stage1_slicer.function_file import load_function_dir
from src.stage1_slicer.registry import PluginRegistry
from src.stage2_toolpath.contour_generator import StandardToolpathGenerator

FUNCS = load_function_dir(os.path.join(os.path.dirname(__file__), "..", "config", "functions"))
SHIFT = (150.0, 150.0)



def dist_to_polyline(pts, poly, max_seg=1.5):
    """Distance of every point to the polyline (jumps longer than max_seg between loops are not segments)."""
    a_, b_ = poly[:-1], poly[1:]
    keep = np.linalg.norm(b_ - a_, axis=1) < max_seg
    a_, b_ = a_[keep], b_[keep]
    ab = b_ - a_
    t = np.clip(np.einsum("ijk,jk->ij", pts[:, None, :] - a_[None], ab) /
                np.maximum(np.einsum("jk,jk->j", ab, ab), 1e-12), 0.0, 1.0)
    proj = a_[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - proj, axis=2).min(axis=1)


TOP_SKIP = 3
WALL_TOL = 0.25   # mm: the open wall polylines may leave a gap of up to one subdivision segment at the seam
TOL = 8e-3     # mm: float noise of coordinates around 150 and the resampling of the contours


def _set_distance(P, Q, max_seg=1e9):
    """Largest distance of any point of the contours P to the nearest polyline of Q."""
    pts = np.vstack(P)
    return float(np.min([dist_to_polyline(pts, q, max_seg) for q in Q], axis=0).max())


def _part(shift=(0.0, 0.0)):
    m = trimesh.creation.cylinder(radius=12, height=8, sections=48)
    m.apply_translation((shift[0], shift[1], 4.0))
    return m


def _slice(name, shift):
    p = FUNCS[name].to_slicer_params({})
    p.update({"layer_height": 0.4, "transition_height": 6.0})
    model = PluginRegistry.get_plugin("universal_field_slicer")().slice(_part(shift), p)
    prof = MinimalPrintProfile(layer_height=0.4, nozzle_diameter=0.4, num_perimeters=1, infill_density=0.3,
                               bottom_solid_layers=1, top_solid_layers=1)
    traj = StandardToolpathGenerator().generate_toolpath(model, prof)
    return model, traj


@pytest.mark.parametrize("name", sorted(FUNCS))
def test_slice_is_independent_of_the_position_on_the_bed(name):
    m0, t0 = _slice(name, (0.0, 0.0))
    m1, t1 = _slice(name, SHIFT)
    assert len(m0.layers) == len(m1.layers)
    # Stage 1: every boundary contour is the same curve, only moved (Z unchanged)
    n = len(m0.layers)
    # the top layers are skipped: the plane there grazes the deformed top cap (nearly parallel), the cut is ill-conditioned
    # and float noise alone moves it (docs/PITFALLS.md #42)
    for l0, l1 in list(zip(m0.layers, m1.layers))[:n - TOP_SKIP]:             # contour order / resampling start may differ: compare as sets
        A = [np.asarray(c.points) + [SHIFT[0], SHIFT[1], 0.0] for c in l0.contours if c.feature_type == "boundary"]
        B = [np.asarray(c.points) for c in l1.contours if c.feature_type == "boundary"]
        assert _set_distance(A, B) < TOL and _set_distance(B, A) < TOL
    # Stage 2: the walls (the infill lattice is anchored to the bed grid and may differ) follow the same surface
    walls = ("outer_wall", "inner_wall")

    def paths(t):
        out = {}
        for w in t.waypoints:
            if w.layer_index is not None and w.feature_type in walls and not w.is_travel_move:
                out.setdefault(w.layer_index, []).append([w.x, w.y, w.z])
        return {k: np.array(v) for k, v in out.items()}

    w0, w1 = paths(t0), paths(t1)
    assert w0.keys() == w1.keys() and sum(len(v) for v in w0.values()) > 100
    for k in [k for k in w0 if k < n - TOP_SKIP]:
        a_, b_ = w0[k] + [SHIFT[0], SHIFT[1], 0.0], w1[k]
        assert dist_to_polyline(a_, b_).max() < WALL_TOL and dist_to_polyline(b_, a_).max() < WALL_TOL


def test_origin_is_stored_for_stage2():
    m, _ = _slice("Conical", SHIFT)
    assert m.metadata["origin_x"] == pytest.approx(SHIFT[0], abs=1e-6)
    assert m.metadata["origin_y"] == pytest.approx(SHIFT[1], abs=1e-6)


def test_explicit_origin_overrides_the_part_centre():
    from src.stage1_slicer.math_strategies import build_strategy
    p = FUNCS["Conical"].to_slicer_params({})
    s = p.pop("strategy")
    p.update({"layer_height": 0.4, "transition_height": 6.0, "origin_x": 1.0, "origin_y": 2.0})
    st = build_strategy(s, _part((50, 50)), p)
    assert (st.origin_x, st.origin_y) == (1.0, 2.0)


def test_preview_uses_the_part_centre():
    from src.stage1_slicer.function_import import ExpressionField, preview_surface
    f = ExpressionField("r", {})
    _, _, z, _ = preview_surface(f, (140.0, 160.0, 140.0, 160.0), 21, origin=(150.0, 150.0))
    assert z.min() == pytest.approx(0.0, abs=1e-9) and z.max() == pytest.approx(np.hypot(10, 10), abs=1e-6)
