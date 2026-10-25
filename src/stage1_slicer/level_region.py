"""
Region {g >= 0} of a scalar field sampled on a regular grid, as a shapely geometry (pure numpy marching
squares + shapely polygonize). Used by the ramp transition mode to find where a planar base layer exists
(the part of the footprint lying below the function surface).
"""
from typing import Callable, Optional

import numpy as np
from shapely.geometry import MultiLineString, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union

# edges of a cell: 0 bottom (a-b), 1 right (b-c), 2 top (d-c), 3 left (a-d); corners a (x0,y0) b (x1,y0) c (x1,y1) d (x0,y1)
# bit i of the case = corner i is inside (g >= 0)
_CASES = {1: [(0, 3)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 6: [(0, 2)], 7: [(2, 3)], 8: [(2, 3)],
          9: [(0, 2)], 11: [(1, 2)], 12: [(1, 3)], 13: [(0, 1)], 14: [(0, 3)]}
_SADDLE_5 = ([(0, 1), (2, 3)], [(0, 3), (1, 2)])        # (centre inside, centre outside)
_SADDLE_10 = ([(0, 3), (1, 2)], [(0, 1), (2, 3)])


def marching_segments(X: np.ndarray, Y: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Iso-line g = 0 of G (shape (len(Y), len(X))) as segments, array (M, 2, 2)."""
    a, b = G[:-1, :-1], G[:-1, 1:]
    d, c = G[1:, :-1], G[1:, 1:]
    case = (a >= 0).astype(np.int8) | ((b >= 0) << 1) | ((c >= 0) << 2) | ((d >= 0) << 3)
    jj, ii = np.nonzero((case != 0) & (case != 15))
    if len(jj) == 0:
        return np.zeros((0, 2, 2))
    x0, x1 = X[ii], X[ii + 1]
    y0, y1 = Y[jj], Y[jj + 1]
    va, vb, vc, vd = a[jj, ii], b[jj, ii], c[jj, ii], d[jj, ii]

    def cross(p, q):                         # position of the zero between values p -> q (only used where signs differ)
        den = p - q
        return p / np.where(den == 0.0, 1.0, den)

    pts = {
        0: np.stack([x0 + cross(va, vb) * (x1 - x0), y0], axis=1),
        1: np.stack([x1, y0 + cross(vb, vc) * (y1 - y0)], axis=1),
        2: np.stack([x0 + cross(vd, vc) * (x1 - x0), y1], axis=1),
        3: np.stack([x0, y0 + cross(va, vd) * (y1 - y0)], axis=1),
    }
    cs = case[jj, ii]
    centre = (va + vb + vc + vd) / 4.0 >= 0
    out = []
    for code in range(1, 15):
        sel = cs == code
        if not sel.any():
            continue
        if code in (5, 10):
            pairs_in, pairs_out = _SADDLE_5 if code == 5 else _SADDLE_10
            for flag, pairs in ((True, pairs_in), (False, pairs_out)):
                m = sel & (centre == flag)
                for e1, e2 in pairs:
                    if m.any():
                        out.append(np.stack([pts[e1][m], pts[e2][m]], axis=1))
        else:
            for e1, e2 in _CASES[code]:
                out.append(np.stack([pts[e1][sel], pts[e2][sel]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2, 2))


def region_polygon(X: np.ndarray, Y: np.ndarray, G: np.ndarray,
                   value: Optional[Callable[[float, float], float]] = None) -> BaseGeometry:
    """
    Geometry of {g >= 0} inside the grid rectangle. `value(x, y)` (the exact field) classifies the faces
    of the iso-line network; by default G is interpolated bilinearly.
    """
    rect = box(float(X[0]), float(Y[0]), float(X[-1]), float(Y[-1]))
    if (G >= 0).all():
        return rect
    if not (G >= 0).any():
        return Polygon()
    segs = marching_segments(X, Y, G)
    if len(segs) == 0:
        return Polygon()
    net = unary_union([MultiLineString(segs.tolist()), rect.exterior])

    def bilinear(x, y):
        i = int(np.clip(np.searchsorted(X, x) - 1, 0, len(X) - 2))
        j = int(np.clip(np.searchsorted(Y, y) - 1, 0, len(Y) - 2))
        tx = (x - X[i]) / (X[i + 1] - X[i])
        ty = (y - Y[j]) / (Y[j + 1] - Y[j])
        return ((1 - tx) * (1 - ty) * G[j, i] + tx * (1 - ty) * G[j, i + 1]
                + tx * ty * G[j + 1, i + 1] + (1 - tx) * ty * G[j + 1, i])

    f = value or bilinear
    keep = []
    for face in polygonize(net):
        p = face.representative_point()
        if f(p.x, p.y) >= 0:
            keep.append(face)
    return unary_union(keep) if keep else Polygon()
