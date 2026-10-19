"""
Layer-gap check against the nozzle's printable window - the same for EVERY function (no function specific maths).

For each point of a layer the strategy reports two gaps to the next layer (BaseSlicingStrategy.layer_gaps):
  * vertical : Z distance          - what a VERTICAL nozzle (3-axis machine) works in
  * normal   : along the surface normal (vertical * cos slope) - what a nozzle along the normal (5-axis) works in
normal <= vertical, so the normal gap decides the minimum and the vertical gap the maximum when the
mode is not known; the machine file tells the mode (rotary axes -> normal, otherwise vertical).

The printable window is 0.25 .. 0.80 of the nozzle diameter (0.4 nozzle: 0.10 .. 0.32 mm).

The numbers come either from a finished Stage 1 model (`measure_model`, the real slice) or from a probe
that evaluates the strategy on sample points of the placed part without slicing (`probe`, fast enough for a
live label). `suggest_transition` bisects the smallest transition height that passes; it reports that the
transition has no effect when the gaps do not depend on it (e.g. tilting layers).
"""
import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

from src.stage1_slicer.math_strategies import build_strategy

MIN_FRACTION = 0.25
MAX_FRACTION = 0.80


def printable_window(nozzle: float) -> Tuple[float, float]:
    return MIN_FRACTION * nozzle, MAX_FRACTION * nozzle


@dataclass
class GapStats:
    v_min: float
    v_max: float
    n_min: float
    n_max: float
    count: int

    def as_tuple(self):
        return (self.v_min, self.v_max, self.n_min, self.n_max)


@dataclass
class ThicknessCheck:
    direction: str                         # "vertical" | "normal": the one that decides (machine mode)
    low: float                             # smallest / largest gap in that direction
    high: float
    other_low: float                       # the same in the other direction (information)
    other_high: float
    window: Tuple[float, float]
    ok: bool                               # the deciding direction is inside the window
    both_ok: bool                          # inside the window in both directions (any machine)
    messages: List[str] = field(default_factory=list)


def measure_model(model) -> Optional[GapStats]:
    """Min / max gaps over every point of a finished Stage 1 model (None if it carries no gaps)."""
    v, n = [], []
    for layer in model.layers:
        for c in layer.contours:
            if c.vertical_gaps and c.normal_gaps:
                v.extend(c.vertical_gaps)
                n.extend(c.normal_gaps)
    if not v:
        return None
    return GapStats(min(v), max(v), min(n), max(n), len(v))


@contextmanager
def _quiet():
    lg = logging.getLogger("src.stage1_slicer.math_strategies")
    old = lg.level
    lg.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        lg.setLevel(old)


def probe(mesh: trimesh.Trimesh, strategy_name: str, params: Dict[str, Any], transition: Optional[float] = None,
          n_points: int = 400, n_layers: int = 20, seed: int = 0) -> GapStats:
    """Gaps on sample points of the part (surface samples, XY only) over evenly spread layers - no slicing."""
    p = dict(params)
    p["refine_mesh"] = False                       # sample the strategy, a refined mesh is not needed
    if transition is not None:
        p["transition_height"] = float(transition)
    with _quiet():
        strat = build_strategy(strategy_name, mesh, p)
        n = strat.get_layer_count()
        layers = sorted({int(round(v)) for v in np.linspace(0, max(n - 2, 0), min(n_layers, max(n - 1, 1)))})
        try:
            pts = np.asarray(mesh.sample(n_points, seed=seed))[:, :2]
        except TypeError:
            np.random.seed(seed)
            pts = np.asarray(mesh.sample(n_points))[:, :2]
        v, nn = [], []
        for i in layers:
            for x, y in pts:
                g, gn = strat.layer_gaps(float(x), float(y), i)
                v.append(g)
                nn.append(gn)
    return GapStats(min(v), max(v), min(nn), max(nn), len(v))


def check(stats: GapStats, nozzle: float, rotary: bool, layer_height: Optional[float] = None) -> ThicknessCheck:
    lo, hi = printable_window(nozzle)
    eps = 1e-9
    direction = "normal" if rotary else "vertical"
    a = (stats.n_min, stats.n_max) if rotary else (stats.v_min, stats.v_max)
    b = (stats.v_min, stats.v_max) if rotary else (stats.n_min, stats.n_max)
    msgs: List[str] = []

    def inside(r):
        return r[0] >= lo - eps and r[1] <= hi + eps

    def describe(r, label):
        out = []
        if r[0] < lo - eps:
            out.append(f"{label}: gap as small as {max(r[0], 0.0):.3f} mm (nozzle {nozzle:g}: at least {lo:g})"
                       + (" - that is the layer height itself, raise it" if layer_height and r[0] >= layer_height - eps else "")
                       + (" - the layers fold over" if r[0] <= 0 else ""))
        if r[1] > hi + eps:
            out.append(f"{label}: gap as large as {r[1]:.3f} mm (nozzle {nozzle:g}: at most {hi:g})"
                       + (" - that is the layer height itself, lower it" if layer_height and r[1] <= layer_height + eps else ""))
        return out

    msgs += describe(a, "this machine")
    if inside(a) and not inside(b):
        msgs += describe(b, "other nozzle direction")
    return ThicknessCheck(direction, a[0], a[1], b[0], b[1], (lo, hi), inside(a), inside(a) and inside(b), msgs)


@dataclass
class TransitionSuggestion:
    relevant: bool                         # the gaps depend on the transition height at all
    possible: bool                         # some transition height passes
    min_transition: Optional[float]        # smallest passing value (rounded up to 0.1), None if not relevant / impossible


def suggest_transition(mesh: trimesh.Trimesh, strategy_name: str, params: Dict[str, Any], nozzle: float, rotary: bool,
                       current: float, layer_height: Optional[float] = None, t_max: float = 2000.0) -> TransitionSuggestion:
    def ok_at(T):
        return check(probe(mesh, strategy_name, params, T), nozzle, rotary, layer_height).ok

    a = probe(mesh, strategy_name, params, max(current, 0.1))
    b = probe(mesh, strategy_name, params, max(current, 0.1) * 2.0 + 1.0)
    if all(math.isclose(x, y, abs_tol=1e-9) for x, y in zip(a.as_tuple(), b.as_tuple())):
        return TransitionSuggestion(False, check(a, nozzle, rotary, layer_height).ok, None)
    if not ok_at(t_max):
        return TransitionSuggestion(True, False, None)
    lo, hi = 0.05, t_max
    if ok_at(lo):
        return TransitionSuggestion(True, True, 0.1)
    for _ in range(22):                              # geometric bisection: pass(T) is monotone in T
        mid = math.sqrt(lo * hi)
        if ok_at(mid):
            hi = mid
        else:
            lo = mid
    return TransitionSuggestion(True, True, float(np.ceil(hi * 10.0) / 10.0))
