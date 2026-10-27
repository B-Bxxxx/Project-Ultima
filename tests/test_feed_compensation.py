"""Feed compensation (Stage 4): F is raised on moves dominated by rotation, everything else stays."""
import numpy as np
import pytest

from src.common.schemas import MachineStateVector, MachineTrajectory, MinimalPrintProfile
from src.stage4_postproc.feed_compensation import compensated_feeds
from src.stage4_postproc.klipper_postproc import Klipper5AxisPostProcessor


def st(x, c, step, travel=False, f=1500.0):
    return MachineStateVector(x=x, y=0, z=5, b=10, c=c, extrusion_volume=0.0 if travel else 0.1, feedrate=f, is_travel_move=travel,
                              part_step=step)


def chain():
    """straight moves (tip path = machine path), then a rotation-dominated stretch (tip hardly moves while C turns 10 deg per move)."""
    out = [st(0, 0, 0.0)]
    for k in range(1, 6):
        out.append(st(1.0 * k, 0, 1.0))
    c = 0.0
    for k in range(10):
        c += 10.0
        out.append(st(6.0 + 0.01 * k, c, 0.01))
    for k in range(1, 6):
        out.append(st(6.1 + k, c, 1.0))
    return out


def prof(**kw):
    return MinimalPrintProfile(**kw)


def test_off_by_default_leaves_every_feedrate():
    s = chain()
    assert compensated_feeds(s, prof()) == [x.feedrate for x in s]


def test_rotation_dominated_moves_get_a_higher_f_within_the_limits():
    s = chain()
    F = np.array(compensated_feeds(s, prof(feed_compensation=1.0, max_feed=6000, rotary_max_speed=60, feed_ramp=10.0)))
    assert np.all(F >= 1500.0 - 1e-9) and np.all(F[1:6] == 1500.0)             # straight moves untouched
    slow = F[6:16]
    assert slow.max() > 1500.0 and slow.max() <= 6000.0 + 1e-9
    # the fastest rotary axis never exceeds 60 deg/s in a raised move: dC / t with t = L6 / F
    for i in range(6, 16):
        d = np.array([s[i].x - s[i - 1].x, 0, 0, 0, 0, s[i].c - s[i - 1].c])
        l6 = np.linalg.norm(d)
        assert abs(d[5]) * (F[i] / 60.0) / l6 <= 60.0 + 1e-6


def test_exponent_blends_between_off_and_full():
    s = chain()
    kw = dict(max_feed=100000, rotary_max_speed=10000, xyz_max_speed=10000, feed_ramp=100.0)
    f0 = max(compensated_feeds(s, prof(feed_compensation=0.0, **kw)))
    f5 = max(compensated_feeds(s, prof(feed_compensation=0.5, **kw)))
    f1 = max(compensated_feeds(s, prof(feed_compensation=1.0, **kw)))
    assert f0 < f5 < f1


def test_ramp_limits_the_change_between_neighbours_and_travel_is_untouched():
    s = chain()
    s[3] = st(3, 0, 1.0, travel=True, f=3000.0)
    F = compensated_feeds(s, prof(feed_compensation=1.0, max_feed=60000, rotary_max_speed=5000, xyz_max_speed=5000, feed_ramp=0.5))
    assert F[3] == 3000.0
    for i in range(4, len(s)):
        if not s[i].is_travel_move and not s[i - 1].is_travel_move:
            assert F[i] <= F[i - 1] * 1.5 + 1e-6 and F[i - 1] <= F[i] * 1.5 + 1e-6


def test_the_gcode_only_changes_in_f_and_without_part_steps_nothing_happens():
    s = chain()
    tr = MachineTrajectory(states=s)
    off = Klipper5AxisPostProcessor().generate_gcode(tr, prof()).splitlines()
    on = Klipper5AxisPostProcessor().generate_gcode(tr, prof(feed_compensation=1.0, max_feed=6000, rotary_max_speed=60)).splitlines()
    assert len(off) == len(on) and off != on
    strip = lambda l: " ".join(w for w in l.split() if not w.startswith("F"))
    assert [strip(a) for a in off] == [strip(b) for b in on]                       # same G-code except the F words
    for x in s:
        x.part_step = None
    assert Klipper5AxisPostProcessor().generate_gcode(MachineTrajectory(states=s), prof(feed_compensation=1.0)).splitlines() == off
