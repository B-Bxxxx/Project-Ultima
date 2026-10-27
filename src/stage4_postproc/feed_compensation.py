"""
Feed compensation (Stage 4): constant deposition speed on a 5-axis machine.

The firmware (Klipper cartesian6) takes the time of a move as L6 / F, L6 = the length over all six axes with degrees counted as mm.
A move that turns B / C a lot while the tip hardly moves over the part is therefore very slow (a few mm/s at F = 25 mm/s). F is raised
for such moves:

    F = F0 * (L6 / tip_path) ** s                    s = profile.feed_compensation (0 = off, 1 = the tip keeps F0)

limited by max_feed (mm/min), rotary_max_speed (deg/s of the fastest rotary axis) and xyz_max_speed (mm/s of the X Y Z vector), never
below F0, and smoothed so that neighbouring extruding moves differ by at most `feed_ramp` (fraction). Only F changes: the G-code format and
E are untouched, and with s = 0 the output is identical to before.
"""
import math
from typing import List

import numpy as np

MIN_TIP_PATH = 1e-3                  # mm: a shorter tip path counts as this (a move that only turns the head)


def _wrap(d):
    return (np.asarray(d, float) + 180.0) % 360.0 - 180.0


def compensated_feeds(states, profile) -> List[float]:
    """F (mm/min) of every state: its own feedrate, raised where the move is dominated by rotation."""
    F0 = np.array([s.feedrate for s in states], float)
    s_exp = float(getattr(profile, "feed_compensation", 0.0))
    n = len(states)
    if s_exp <= 0.0 or n < 2:
        return [float(f) for f in F0]
    P = np.array([[s.x, s.y, s.z, s.a, s.b, s.c] for s in states], float)
    F = F0.copy()
    active = np.zeros(n, bool)
    for i in range(1, n):
        st = states[i]
        step = getattr(st, "part_step", None)
        if st.is_travel_move or step is None:
            continue
        d = P[i] - P[i - 1]
        d[3:] = _wrap(d[3:])                                     # the firmware turns rotary axes the short way round
        l6 = float(np.sqrt((d ** 2).sum()))
        if l6 < 1e-9:
            continue
        ratio = l6 / max(step, MIN_TIP_PATH)
        if ratio <= 1.0:
            continue
        lxyz = float(np.linalg.norm(d[:3]))
        rot = float(np.abs(d[3:]).max())
        f = F0[i] * ratio ** s_exp
        cap = float(profile.max_feed)
        if rot > 1e-9:
            cap = min(cap, float(profile.rotary_max_speed) * 60.0 * l6 / rot)
        if lxyz > 1e-9:
            cap = min(cap, float(profile.xyz_max_speed) * 60.0 * l6 / lxyz)
        F[i] = max(F0[i], min(f, cap))
        active[i] = F[i] > F0[i]
    g = 1.0 + float(profile.feed_ramp)
    for i in range(1, n):                                        # F may rise by the factor g per move ...
        if not states[i].is_travel_move and not states[i - 1].is_travel_move:
            F[i] = min(F[i], F[i - 1] * g)
    for i in range(n - 2, -1, -1):                               # ... and fall by it: it starts rising before the slow part
        if not states[i].is_travel_move and not states[i + 1].is_travel_move:
            F[i] = min(F[i], F[i + 1] * g)
    return [float(max(f0, f)) for f0, f in zip(F0, F)]
