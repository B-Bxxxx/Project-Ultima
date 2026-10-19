"""
Adaptive, conforming edge refinement of a triangle mesh (pure numpy).

Why: the surface is deformed by moving every VERTEX (z' = z + w f(x, y)) and the triangles in between stay
flat, so the deformed mesh only follows f where there are vertices. A uniform split (extents / 8) is
too coarse where f changes quickly and wastes faces where it is smooth - and it ignores f altogether.

`refine_adaptive` splits only the edges whose deformed height deviates more than `tol` from the linear
interpolation of their end points (`edge_error(P0, P1)` is supplied by the strategy). The result is
conforming (no T-junctions): a triangle with 1 / 2 / 3 split edges is cut into 2 / 3 / 4 triangles.
Edges shorter than `min_edge` are never split; the loop stops at `max_faces`.
"""
from typing import Callable, Tuple

import numpy as np


def _split_faces(F: np.ndarray, mid: np.ndarray, V: np.ndarray) -> np.ndarray:
    """F (n,3) faces, mid (n,3) midpoint vertex id of edge (0-1, 1-2, 2-0) or -1. Returns the new faces."""
    marked = mid >= 0
    cnt = marked.sum(axis=1)
    out = [F[cnt == 0]]

    # rotate every face so that the split edges come first: edge k is (F[k], F[k+1])
    one = np.flatnonzero(cnt == 1)
    if len(one):
        k = np.argmax(marked[one], axis=1)
        a, b, c = (F[one, (k + j) % 3] for j in range(3))
        m = mid[one, k]
        out += [np.column_stack([a, m, c]), np.column_stack([m, b, c])]

    two = np.flatnonzero(cnt == 2)
    if len(two):
        k0 = np.argmin(marked[two], axis=1)              # the unmarked edge
        # edges k0+1 (b-c) and k0+2 (c-a) are split; rotate so that the unmarked edge is a-b
        a, b, c = (F[two, (k0 + j) % 3] for j in range(3))
        mbc, mca = mid[two, (k0 + 1) % 3], mid[two, (k0 + 2) % 3]
        # the quad a, b, mbc, mca is cut along its SHORTER diagonal (long diagonals make thin sliver triangles,
        # whose plane sections come out as near-duplicate points that break the contour tracing)
        d1 = np.linalg.norm(V[a] - V[mbc], axis=1)
        d2 = np.linalg.norm(V[b] - V[mca], axis=1)
        s1 = (d1 <= d2)[:, None]
        t1 = np.where(s1, np.column_stack([a, b, mbc]), np.column_stack([a, b, mca]))
        t2 = np.where(s1, np.column_stack([a, mbc, mca]), np.column_stack([b, mbc, mca]))
        out += [t1, t2, np.column_stack([mca, mbc, c])]

    three = np.flatnonzero(cnt == 3)
    if len(three):
        a, b, c = F[three, 0], F[three, 1], F[three, 2]
        mab, mbc, mca = mid[three, 0], mid[three, 1], mid[three, 2]
        out += [np.column_stack([a, mab, mca]), np.column_stack([mab, b, mbc]),
                np.column_stack([mca, mbc, c]), np.column_stack([mab, mbc, mca])]
    return np.vstack(out).astype(np.int64)


def refine_adaptive(vertices: np.ndarray, faces: np.ndarray,
                    edge_error: Callable[[np.ndarray, np.ndarray], np.ndarray],
                    tol: float, min_edge: float = 0.2, max_faces: int = 400_000,
                    max_iter: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (vertices, faces) refined until every edge error is <= tol (or min_edge / max_faces is hit)."""
    V = np.asarray(vertices, float).copy()
    F = np.asarray(faces, np.int64).copy()
    for _ in range(max_iter):
        if len(F) == 0:
            break
        e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        uniq, inv = np.unique(e, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        length = np.linalg.norm(V[uniq[:, 0]] - V[uniq[:, 1]], axis=1)
        cand = length > min_edge
        split = np.zeros(len(uniq), bool)
        if cand.any():
            err = np.asarray(edge_error(V[uniq[cand, 0]], V[uniq[cand, 1]]), float)
            split[np.flatnonzero(cand)[err > tol]] = True
        if not split.any():
            break
        # a split edge is cut in the middle: new vertex ids
        ids = np.full(len(uniq), -1, np.int64)
        sel = np.flatnonzero(split)
        ids[sel] = len(V) + np.arange(len(sel))
        V = np.vstack([V, 0.5 * (V[uniq[sel, 0]] + V[uniq[sel, 1]])])
        mid = ids[inv].reshape(3, len(F)).T            # columns: edge 0-1, 1-2, 2-0 (same order as e above)
        F = _split_faces(F, mid, V)
        if len(F) > max_faces:
            break
    return V, F
