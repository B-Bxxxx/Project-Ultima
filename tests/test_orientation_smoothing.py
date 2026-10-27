import numpy as np
from types import SimpleNamespace as NS

from src.stage3_kinematics.base import _taut_string, resolve_branches, smooth_orientation


def _wps(n, step=0.5):
    return [NS(x=i * step, y=0.0, z=0.0, is_travel_move=False) for i in range(n)]


def test_taut_string_is_straight_inside_wide_corridor():
    s = np.linspace(0, 10, 21)
    lo, hi = np.full(21, -1.0), np.full(21, 1.0)
    lo[0] = hi[0] = 0.0
    lo[-1] = hi[-1] = 0.0
    x = _taut_string(s, lo, hi)
    assert np.allclose(x, 0.0)


def test_taut_string_stays_in_corridor_and_is_shorter():
    s = np.linspace(0, 10, 101)
    base = 5 * np.sin(s)
    lo, hi = base - 1.0, base + 1.0
    lo[0] = hi[0] = base[0]
    lo[-1] = hi[-1] = base[-1]
    x = _taut_string(s, lo, hi)
    assert np.all(x >= lo - 1e-9) and np.all(x <= hi + 1e-9)
    assert np.abs(np.diff(x)).sum() < np.abs(np.diff(base)).sum()


def _axis(b, c):
    b, c = np.radians(b), np.radians(c)
    return np.stack([-np.sin(b) * np.cos(c), np.sin(b) * np.sin(c), np.cos(b)], axis=-1)


def test_smoothing_keeps_axis_within_tolerance_and_reduces_c_motion():
    n = 200
    b = np.full(n, 30.0) + 0.0
    c = 40.0 * np.sin(np.linspace(0, 6 * np.pi, n)) + np.linspace(0, 200, n)
    tol = 3.0
    ob, oc = smooth_orientation(b.tolist(), c.tolist(), _wps(n), tol, -90, 90, -1e9, 1e9)
    ob, oc = np.array(ob), np.array(oc)
    err = np.degrees(np.arccos(np.clip((_axis(b, c) * _axis(ob, oc)).sum(axis=1), -1, 1)))
    assert err.max() <= tol + 1e-6
    assert np.abs(np.diff(oc)).sum() < np.abs(np.diff(c)).sum()
    assert ob[0] == b[0] and oc[0] == c[0] and oc[-1] == c[-1]


def test_off_is_identity_and_travel_splits_chains():
    n = 12
    raw = np.radians(np.linspace(0, 90, n))
    b = np.full(n, 20.0)
    wps = _wps(n)
    b0, c0 = resolve_branches(raw, b, wps, -90, 90, -1e9, 1e9, 0.0)
    b1, c1 = resolve_branches(raw, b, wps, -90, 90, -1e9, 1e9, 0.0, tolerance_deg=0.0)
    assert (b0, c0) == (b1, c1)
    wps[6].is_travel_move = True
    ob, oc = smooth_orientation(b0, c0, wps, 3.0, -90, 90, -1e9, 1e9)
    assert oc[5] == c0[5] and oc[6] == c0[6]            # the ends of both chains are exact
