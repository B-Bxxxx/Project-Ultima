"""
Centre line of a tube-like part (and of any guide curve): the input of centre-line following slicing, where every layer is
(as far as the allowed tilt permits) perpendicular to the curve at its point.

`Centerline` is a smooth 3D curve parametrised by arc length: point, unit tangent and curvature at any arc length.
It is built from a polyline (`Centerline.from_points`, JSON file) or extracted from a tube mesh (`Centerline.from_tube_mesh`):
the tube is walked from one end cap to the other; at every step the mesh is cut by the plane perpendicular to the current
direction and the next centre is the centroid of the outer cross-section loop.
"""
from __future__ import annotations

import json
from typing import Optional, Tuple

import numpy as np

FORMAT = "cam-centerline"
VERSION = 1


def _resample(points: np.ndarray, h: float) -> np.ndarray:
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    n = max(2, int(np.ceil(s[-1] / h)) + 1)
    u = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(u, s, points[:, k]) for k in range(3)], axis=1)


class Centerline:
    def __init__(self, points, smoothing_mm: float = 0.0, sample_mm: float = 0.25, exact_ends: int = 0):
        """`points`: (N, 3) polyline. `smoothing_mm`: RMS deviation allowed to a smoothing spline (0 = interpolate).
        `exact_ends`: this many points at each end are kept (almost) exactly: the end tangent stays what they say."""
        P = np.asarray(points, float)
        if P.ndim != 2 or P.shape[1] != 3 or len(P) < 2:
            raise ValueError("a centre line needs at least two 3D points")
        keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9]
        P = P[keep]
        if len(P) < 2:
            raise ValueError("a centre line needs at least two distinct points")
        while len(P) < 5:                                                   # a spline needs a few points: subdivide
            P = np.stack([np.interp(np.linspace(0, len(P) - 1, 2 * len(P) - 1), np.arange(len(P)), P[:, k]) for k in range(3)], 1)
        self.raw = P
        s_raw = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        from scipy.interpolate import splev, splprep
        k = 3
        if smoothing_mm > 0:
            sig = np.full(len(P), smoothing_mm)
            if exact_ends > 0:
                sig[:exact_ends] = sig[-exact_ends:] = 0.002
            tck, _ = splprep(P.T, u=s_raw, w=1.0 / sig, k=k, s=float(len(P)))
        else:
            tck, _ = splprep(P.T, u=s_raw, k=k, s=0.0)
        self._tck, self._splev = tck, splev
        # arc length of the spline itself (differs a little from the polyline)
        dense = np.array(splev(np.linspace(0, s_raw[-1], max(200, int(s_raw[-1] / 0.05))), tck)).T
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
        self.length = float(s[-1])
        n = max(2, int(np.ceil(self.length / sample_mm)) + 1)
        self.s = np.linspace(0.0, self.length, n)
        u = np.interp(self.s, s, np.linspace(0, s_raw[-1], len(dense)))
        self.points = np.array(splev(u, tck)).T
        d1 = np.array(splev(u, tck, der=1)).T
        d2 = np.array(splev(u, tck, der=2)).T
        sp = np.linalg.norm(d1, axis=1)
        self.tangents = d1 / sp[:, None]
        self.curvatures = np.linalg.norm(np.cross(d1, d2), axis=1) / sp ** 3

    # ------------------------------------------------------------------ queries
    def point_at(self, s: float) -> np.ndarray:
        s = np.clip(s, 0.0, self.length)
        return np.array([np.interp(s, self.s, self.points[:, k]) for k in range(3)])

    def tangent_at(self, s: float) -> np.ndarray:
        s = np.clip(s, 0.0, self.length)
        t = np.array([np.interp(s, self.s, self.tangents[:, k]) for k in range(3)])
        return t / np.linalg.norm(t)

    def curvature_at(self, s: float) -> float:
        return float(np.interp(np.clip(s, 0.0, self.length), self.s, self.curvatures))

    @property
    def min_bend_radius(self) -> float:
        kmax = float(self.curvatures.max())
        return float("inf") if kmax < 1e-9 else 1.0 / kmax

    # ------------------------------------------------------------------ files
    def to_dict(self, step_mm: float = 1.0) -> dict:
        pts = _resample(self.points, step_mm)
        return {"format": FORMAT, "version": VERSION, "unit": "mm",
                "length_mm": round(self.length, 3), "points": np.round(pts, 4).tolist()}

    def save(self, path: str, step_mm: float = 1.0) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(step_mm), fh, indent=1)

    @classmethod
    def load(cls, path: str, **kw) -> "Centerline":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("format") != FORMAT:
            raise ValueError(f'not a centre line file (format "{d.get("format")}", expected "{FORMAT}")')
        return cls(d["points"], **kw)

    @classmethod
    def from_points(cls, points, **kw) -> "Centerline":
        return cls(points, **kw)

    # ------------------------------------------------------------------ extraction from a tube mesh
    @staticmethod
    def end_caps(mesh) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """The two planar end caps of a tube: [(centre, outward normal), ...], the two largest flat facets of equal area."""
        fa = np.asarray(mesh.facets_area)
        if len(fa) < 2:
            raise ValueError("no flat end caps found: give the centre line as points")
        o = np.argsort(-fa)[:2]
        if fa[o[1]] < 0.8 * fa[o[0]]:
            raise ValueError("the two largest flat faces differ too much to be the end caps of a tube")
        caps = []
        for i in o:
            f = mesh.facets[i]
            w = mesh.area_faces[f]
            c = (mesh.triangles_center[f] * w[:, None]).sum(0) / w.sum()
            caps.append((c, np.asarray(mesh.facets_normal[i], float)))
        return caps[0], caps[1]

    @classmethod
    def from_tube_mesh(cls, mesh, step_mm: float = 1.0, start: Optional[Tuple[float, float, float]] = None,
                       smoothing_mm: float = 0.02, **kw) -> "Centerline":
        """
        Walks the tube from one end cap to the other. `start`: a point near the end cap the curve should start at (default: the
        cap with the lower Z, then lower Y: the cap on the bed). The tube needs two flat end caps and a constant-ish cross-section.
        """
        cap_a, cap_b = cls.end_caps(mesh)
        if start is not None:
            st = np.asarray(start, float)
            if np.linalg.norm(cap_b[0] - st) < np.linalg.norm(cap_a[0] - st):
                cap_a, cap_b = cap_b, cap_a
        elif tuple(np.round(cap_b[0][[2, 1]], 6)) < tuple(np.round(cap_a[0][[2, 1]], 6)):      # the cap on the bed (lower Z) first
            cap_a, cap_b = cap_b, cap_a
        end_c, end_n = cap_b
        pt = cap_a[0].copy()
        tan = -cap_a[1] / np.linalg.norm(cap_a[1])
        tan0 = tan.copy()
        pts = [pt.copy()]
        max_steps = int(4 * np.sum(mesh.bounds[1] - mesh.bounds[0]) / step_mm) + 10
        for _ in range(max_steps):
            if float((pt - end_c) @ end_n) >= -1.2 * step_mm:              # (almost) at the far cap: the rest is the exact straight lead
                break
            c1 = cls._centre(mesh, pt + tan * step_mm, tan, pt + tan * step_mm)
            t1 = c1 - pt
            t1 /= np.linalg.norm(t1)
            c2 = cls._centre(mesh, pt + t1 * step_mm, t1, c1)               # second pass with the better direction
            d = c2 - pt
            if np.linalg.norm(d) < 0.2 * step_mm:
                raise ValueError("centre line tracking stalled")
            tan = 0.5 * tan + 0.5 * d / np.linalg.norm(d)
            tan /= np.linalg.norm(tan)
            pt = c2
            pts.append(pt.copy())
        else:
            raise ValueError("centre line tracking did not reach the far end cap")
        # the axis is perpendicular to a flat end cap: the last/first mm are exact straight pieces along the cap normal
        lead = 2.0 * step_mm
        pts = [p for p in pts if np.linalg.norm(p - cap_a[0]) > lead and np.linalg.norm(p - end_c) > lead]
        start_pts = [cap_a[0] + tan0 * (k * 0.5 * step_mm) for k in range(0, 5)]
        end_pts = [end_c - end_n * (k * 0.5 * step_mm) for k in range(4, -1, -1)]
        return cls(np.array(start_pts + pts + end_pts), smoothing_mm=smoothing_mm, exact_ends=len(start_pts), **kw)

    @staticmethod
    def _centre(mesh, origin, normal, near) -> np.ndarray:
        """Centroid of the largest (outer) closed cross-section loop of the mesh cut by the plane, close to `near`."""
        sec = mesh.section(plane_origin=origin, plane_normal=normal)
        if sec is None:
            raise ValueError("centre line tracking left the part")
        a = np.cross(normal, [1.0, 0.0, 0.0])
        if np.linalg.norm(a) < 0.5:
            a = np.cross(normal, [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(normal, a)
        best = None
        for loop in sec.discrete:
            Q = np.asarray(loop)
            if len(Q) < 4:
                continue
            x, y = (Q - origin) @ a, (Q - origin) @ b
            x1, y1 = np.roll(x, -1), np.roll(y, -1)
            cr = x * y1 - x1 * y
            area = cr.sum() / 2.0
            if abs(area) < 1e-9:
                continue
            cx, cy = ((x + x1) * cr).sum() / (6 * area), ((y + y1) * cr).sum() / (6 * area)
            c = origin + cx * a + cy * b
            if np.linalg.norm(c - near) > 12.0:
                continue
            if best is None or abs(area) > best[0]:
                best = (abs(area), c)
        if best is None:
            raise ValueError("centre line tracking found no cross-section")
        return best[1]
