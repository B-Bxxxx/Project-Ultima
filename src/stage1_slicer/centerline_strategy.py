"""
Centre-line following slicing: every layer is the cross-section plane of the part at one point of a guide curve, so the layers stay
(as far as the allowed tilt permits) perpendicular to the curve. The natural way to print a bent tube or pipe.

The part lives in a straightened "chart": (u, v) = coordinates in the layer plane, z' = arc length s along the guide curve.
The real position of a chart point is  P(s, u, v) = C(s) + u e1(s) + v e2(s)  where e1, e2 span the plane perpendicular to the layer
normal n(s) (rotation-minimising frame). The slicing is then a plain planar slicing of the straightened mesh (z' = const) and
undeform_point maps the contours back. Stage 2 plans walls / infill in the (u, v) chart (`chart_points`) and projects back with
`project_points`; the local layer thickness follows from the map (thinner on the inside of a bend, thicker on the outside).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import trimesh

from src.stage1_slicer.centerline import Centerline
from src.stage1_slicer.math_strategies import BaseSlicingStrategy

logger = logging.getLogger(__name__)

GRID_MM = 0.25          # sampling of the guide curve (normals, frames, inverse map)
END_RAMP_MM = 5.0       # the layer normal is the exact tangent at the two ends and blends into the smoothed one over this length


def _unit(a):
    a = np.asarray(a, float)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


class CenterlineStrategy(BaseSlicingStrategy):
    uses_chart = True          # Stage 2 plans in the (u, v) chart of every layer, not in bed XY

    def __init__(self, mesh: trimesh.Trimesh, params: Dict[str, Any]):
        super().__init__(params)
        self.mesh = mesh.copy()
        self.layer_height = float(params.get("layer_height", 0.2))
        self.max_tilt_deg = float(params.get("max_tilt_deg", 0.0) or 0.0)
        pts = params.get("centerline_points")
        if pts is not None:
            self.centerline = Centerline.from_points(pts, sample_mm=GRID_MM)
        elif len(self.mesh.vertices) > 0:
            start = params.get("centerline_start")
            self.centerline = Centerline.from_tube_mesh(self.mesh, start=start, sample_mm=GRID_MM)
        else:
            raise ValueError("centerline strategy: no centre line (give 'centerline_points' or a tube mesh)")
        self._build_frames()
        L = self.centerline.length
        self.start_z = float(params.get("start_z", 0.0)) + self.layer_height          # z' of layer 0 = s of its top
        self.end_z = L
        self.num_layers = self._count_layers(self.start_z, L, self.layer_height)
        self.deformed = None
        if len(self.mesh.vertices) > 0:
            self.deformed = self._straighten(self.mesh)
            r = float(np.abs(self.deformed.vertices[:, :2]).max())
            rmin = self.centerline.min_bend_radius
            if rmin < r:
                logger.warning("centre line: the smallest bend radius (%.1f mm) is below the distance of the part from the curve "
                               "(%.1f mm): neighbouring layers cross each other inside the part", rmin, r)

    # ------------------------------------------------------------------ guide curve, layer normals, frames
    def _build_frames(self) -> None:
        c = self.centerline
        T = c.tangents
        n = T.copy()
        tol = np.radians(self.max_tilt_deg)
        if tol > 1e-6:
            from scipy.ndimage import gaussian_filter1d

            def smooth(sig):
                m = _unit(np.stack([gaussian_filter1d(T[:, k], sig, mode="nearest") for k in range(3)], 1))
                w = np.clip(np.minimum(c.s, c.length - c.s) / END_RAMP_MM, 0.0, 1.0)[:, None]      # exact tangent at both ends
                return _unit(T + w * (m - T))

            def worst(m):
                return float(np.arccos(np.clip((m * T).sum(1), -1, 1)).max())
            lo, hi = 0.0, 4.0 * c.length / GRID_MM
            if worst(smooth(hi)) <= tol:
                n = smooth(hi)
            else:                                                       # the widest smoothing that keeps the axis error within tol
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if worst(smooth(max(mid, 1e-3))) <= tol:
                        lo = mid
                    else:
                        hi = mid
                n = smooth(lo) if lo > 1e-3 else T.copy()
        self.normals = n
        # rotation-minimising frame along the normal field (double reflection); e1 starts perpendicular to n(0) and to world Z/X
        e1 = np.zeros_like(n)
        ref = np.array([0.0, 0.0, 1.0]) if abs(n[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1[0] = _unit(np.cross(ref, n[0]))
        for i in range(len(n) - 1):
            v1 = c.points[i + 1] - c.points[i]
            c1 = v1 @ v1
            if c1 < 1e-18:
                e1[i + 1] = e1[i]
                continue
            rL = e1[i] - (2.0 / c1) * (v1 @ e1[i]) * v1
            nL = n[i] - (2.0 / c1) * (v1 @ n[i]) * v1
            v2 = n[i + 1] - nL
            c2 = v2 @ v2
            e1[i + 1] = rL - ((2.0 / c2) * (v2 @ rL) * v2 if c2 > 1e-18 else 0.0)
        e1 = _unit(e1 - (e1 * n).sum(1, keepdims=True) * n)
        self.e1 = e1
        self.e2 = np.cross(n, e1)

    def _at(self, arr, s):
        s = np.clip(np.asarray(s, float), 0.0, self.centerline.length)
        c = self.centerline
        return np.stack([np.interp(s, c.s, arr[:, k]) for k in range(3)], axis=-1)

    def position(self, s, u, v) -> np.ndarray:
        """P(s, u, v): chart -> real space (arrays broadcast)."""
        s = np.asarray(s, float)
        return self._at(self.centerline.points, s) + np.asarray(u, float)[..., None] * _unit(self._at(self.e1, s)) \
            + np.asarray(v, float)[..., None] * _unit(self._at(self.e2, s))

    def normal_at_s(self, s) -> np.ndarray:
        return _unit(self._at(self.normals, s))

    def chart_of(self, X) -> np.ndarray:
        """Real points -> (s, u, v): the plane of layer s (through C(s), normal n(s)) holds the point."""
        X = np.asarray(X, float).reshape(-1, 3)
        c = self.centerline
        out = np.empty((len(X), 3))
        P, N = c.points, self.normals
        k = len(P)
        for a in range(0, len(X), 2000):
            B = X[a:a + 2000]
            g = np.einsum("bkj,kj->bk", B[:, None, :] - P[None, :, :], N)
            near = np.argmin(((B[:, None, :] - P[None, :, :]) ** 2).sum(2), axis=1)
            # sign change of g closest to the nearest curve point
            sgn = np.sign(g)
            ch = np.where(sgn[:, :-1] * sgn[:, 1:] <= 0)
            s_idx = np.full(len(B), -1.0)
            for row in range(len(B)):
                cols = ch[1][ch[0] == row]
                if len(cols):
                    j = cols[np.argmin(np.abs(cols - near[row]))]
                    g0, g1 = g[row, j], g[row, j + 1]
                    t = 0.0 if g0 == g1 else g0 / (g0 - g1)
                    s_idx[row] = j + t
                else:
                    s_idx[row] = near[row]
            s = np.interp(s_idx, np.arange(k), c.s)
            out[a:a + 2000, 0] = s
            d = B - self._at(P, s)
            out[a:a + 2000, 1] = (d * _unit(self._at(self.e1, s))).sum(1)
            out[a:a + 2000, 2] = (d * _unit(self._at(self.e2, s))).sum(1)
        return out

    def _straighten(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """The mesh in the chart (u, v, s). It is refined first: a long straight edge becomes a curve in the chart."""
        import trimesh.remesh
        V, F = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=1.5, max_iter=8)
        if len(V) > 600_000:
            raise ValueError(f"centre line: the refined mesh has too many vertices ({len(V)})")
        C = self.chart_of(V)
        return trimesh.Trimesh(vertices=np.stack([C[:, 1], C[:, 2], C[:, 0]], axis=1), faces=F, process=False)

    # ------------------------------------------------------------------ BaseSlicingStrategy interface
    def get_layer_count(self) -> int:
        return self.num_layers

    def get_slice_plane(self, layer_idx: int):
        return [0, 0, self.start_z + layer_idx * self.layer_height - self._mid_offset()], [0, 0, 1]

    def deform_mesh(self) -> trimesh.Trimesh:
        return self.deformed

    def _s_top(self, layer_idx: int) -> float:
        return self.start_z + layer_idx * self.layer_height

    def undeform_point(self, pt: Tuple[float, float, float], layer_idx: int):
        u, v, s = pt
        return tuple(float(a) for a in self.position(s, u, v))

    def undeform_points(self, pts, layer_idx: int) -> np.ndarray:
        p = np.asarray(pts, float).reshape(-1, 3)
        return self.position(p[:, 2], p[:, 0], p[:, 1])

    def project_to_layer(self, x: float, y: float, layer_idx: int):
        return tuple(float(a) for a in self.position(self._s_top(layer_idx), x, y))

    def project_points(self, xy, layer_idx: int) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        return self.position(np.full(len(xy), self._s_top(layer_idx)), xy[:, 0], xy[:, 1])

    def chart_points(self, pts, layer_idx: int) -> np.ndarray:
        """Real points of a layer's contour (cut at the mid-plane) -> (u, v) of the layer chart."""
        p = np.asarray(pts, float).reshape(-1, 3)
        s = np.full(len(p), self._s_top(layer_idx) - self._mid_offset())
        d = p - self._at(self.centerline.points, s)
        return np.stack([(d * _unit(self._at(self.e1, s))).sum(1), (d * _unit(self._at(self.e2, s))).sum(1)], axis=1)

    def compute_normal(self, undeformed_pt, layer_idx: int):
        return tuple(float(a) for a in self.normal_at_s(self._s_top(layer_idx)))

    def normals_at(self, pts, layer_idx: int) -> np.ndarray:
        p = np.asarray(pts, float).reshape(-1, 3)
        return np.tile(self.normal_at_s(self._s_top(layer_idx)), (len(p), 1))

    def _gap(self, uv, s_a: float, s_b: float, normal) -> np.ndarray:
        uv = np.asarray(uv, float).reshape(-1, 2)
        a = self.position(np.full(len(uv), s_a), uv[:, 0], uv[:, 1])
        b = self.position(np.full(len(uv), s_b), uv[:, 0], uv[:, 1])
        return (b - a) @ normal

    def thicknesses_at(self, pts, layer_idx: int) -> np.ndarray:
        """Thickness of layer `layer_idx` at the points: its distance to the layer below, measured along the layer normal."""
        s = self._s_top(layer_idx)
        uv = self.chart_points(pts, layer_idx)
        t = self._gap(uv, s - self.layer_height, s, self.normal_at_s(s))
        return np.maximum(t, 0.01)

    def compute_thickness(self, undeformed_pt, layer_idx: int) -> float:
        return float(self.thicknesses_at([undeformed_pt], layer_idx)[0])

    def layer_gaps_batch(self, xy, layer_idx: int, eps: float = 1e-4):
        """(vertical, normal) gap to the next layer at chart points: here both are the distance along the layer normal."""
        s = self._s_top(layer_idx)
        n = self.normal_at_s(s)
        g = self._gap(xy, s, s + self.layer_height, n)
        return g, g

    def layer_gaps(self, x: float, y: float, layer_idx: int, eps: float = 1e-4):
        g, n = self.layer_gaps_batch([[x, y]], layer_idx)
        return float(g[0]), float(n[0])

    def get_resolved_params(self) -> Dict[str, Any]:
        resolved = super().get_resolved_params()
        resolved["start_z"] = float(self.start_z - self.layer_height)
        resolved["end_z"] = float(self.end_z)
        resolved["max_tilt_deg"] = float(self.max_tilt_deg)
        resolved["centerline_points"] = np.round(self.centerline.to_dict(1.0)["points"], 4).tolist()
        return resolved
