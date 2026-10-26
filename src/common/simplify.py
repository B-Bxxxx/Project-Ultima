"""
Path simplification (pure numpy), used by Stage 1 (raw contours) and Stage 2 (toolpaths).

Why: the contours that come out of Stage 1 and the 1 mm subdivision of Stage 2 are far denser than the
surface needs (flat or smoothly curved stretches carry hundreds of redundant points). Old printer
boards (8 / 32 bit Marlin) cannot plan thousands of very short segments per second - the nozzle then
slows down and speeds up, which shows up as stutter and a "fuzzy skin" look.

`simplify_path` keeps a point only if dropping it would move the path by more than `tol` mm (distance of
every dropped point from the chord that replaces it, in 3D) or tilt the tool axis by more than
`angle_deg` (normal of the dropped point against the normal interpolated along the chord). A second
pass removes points next to segments shorter than `min_len` as long as the deviation stays within
`MERGE_FACTOR * tol`. The first and the last point are always kept.

Extrusion: the thickness of a kept point is chosen so that chord_length * thickness equals the sum of
the replaced segments' length * thickness, i.e. the extruded volume of the path does not change.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np

MERGE_FACTOR = 3.0      # short-segment merging may deviate up to this many tolerances


def _metric(P: np.ndarray, N: Optional[np.ndarray], a: int, b: int, tol: float, angle_deg: float) -> np.ndarray:
    """Normalised deviation (1.0 = at the limit) of the points a+1 .. b-1 from the chord a -> b."""
    seg = P[a + 1:b] - P[a]
    ab = P[b] - P[a]
    l2 = float(ab @ ab)
    if l2 < 1e-18:
        t = np.zeros(len(seg))
        d = np.linalg.norm(seg, axis=1)
    else:
        t = np.clip(seg @ ab / l2, 0.0, 1.0)
        d = np.linalg.norm(seg - t[:, None] * ab, axis=1)
    m = d / tol
    if N is not None and angle_deg > 0.0:
        n_t = (1.0 - t)[:, None] * N[a] + t[:, None] * N[b]
        n_t /= np.maximum(np.linalg.norm(n_t, axis=1, keepdims=True), 1e-12)
        cosang = np.clip(np.einsum("ij,ij->i", n_t, N[a + 1:b]), -1.0, 1.0)
        m = np.maximum(m, np.degrees(np.arccos(cosang)) / angle_deg)
    return m


def simplify_path(points: Sequence[Sequence[float]], normals: Optional[Sequence[Sequence[float]]],
                  thicknesses: Sequence[float], tol: float, min_len: float = 0.0,
                  angle_deg: float = 1.0) -> Tuple[List[int], List[float]]:
    """
    Returns (kept indices, thickness of every kept point except the first) for the polyline `points`.
    thicknesses[i] is the thickness of the segment that ends at point i (the Stage 2 convention).
    """
    P = np.asarray(points, float)
    n = len(P)
    H = np.asarray(thicknesses, float)
    if n < 3 or tol <= 0.0:
        return list(range(n)), [float(h) for h in H[1:]]
    N = None
    if normals is not None:
        N = np.asarray(normals, float)
        if np.ptp(N, axis=0).max() < 1e-9:
            N = None                                    # all normals equal (planar layer): nothing to check

    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        m = _metric(P, N, a, b, tol, angle_deg)
        i = int(np.argmax(m))
        if m[i] > 1.0:
            k = a + 1 + i
            keep[k] = True
            stack.append((a, k))
            stack.append((k, b))
    idx = [int(i) for i in np.flatnonzero(keep)]

    if min_len > 0.0 and len(idx) > 2:
        for _ in range(4):
            changed = False
            j = 1
            while j < len(idx) - 1:
                a, k, b = idx[j - 1], idx[j], idx[j + 1]
                short = min(np.linalg.norm(P[k] - P[a]), np.linalg.norm(P[b] - P[k])) < min_len
                if short and (b - a < 2 or float(_metric(P, N, a, b, tol, angle_deg).max()) <= MERGE_FACTOR):
                    del idx[j]
                    changed = True
                    continue
                j += 1
            if not changed:
                break

    # thickness that keeps the volume: sum(dl_i * h_i) over the replaced segments / chord length
    dl = np.linalg.norm(np.diff(P, axis=0), axis=1)                  # dl[i-1] = |P[i] - P[i-1]|
    w = dl * H[1:]
    cum = np.concatenate([[0.0], np.cumsum(w)])
    out = []
    for a, b in zip(idx[:-1], idx[1:]):
        chord = float(np.linalg.norm(P[b] - P[a]))
        vol = cum[b] - cum[a]
        out.append(float(vol / chord) if chord > 1e-9 else float(H[b]))
    return idx, out


def simplify_ring(points, tol: float) -> np.ndarray:
    """
    Douglas-Peucker for an open polyline or a closed ring (first point == last point) of (N, 3) points;
    returns the kept points (a ring stays closed). Closed rings are split at the point farthest from the
    start so that the start vertex itself is not special.
    """
    P = np.asarray(points, float)
    n = len(P)
    if n < 4 or tol <= 0.0:
        return P
    zeros = np.zeros(n)
    if np.linalg.norm(P[0] - P[-1]) < 1e-9:
        k = int(np.argmax(np.linalg.norm(P - P[0], axis=1)))
        if k <= 0 or k >= n - 1:
            return P
        i1, _ = simplify_path(P[:k + 1], None, zeros[:k + 1], tol)
        i2, _ = simplify_path(P[k:], None, zeros[k:], tol)
        return np.vstack([P[:k + 1][i1], P[k:][i2][1:]])
    idx, _ = simplify_path(P, None, zeros, tol)
    return P[idx]
