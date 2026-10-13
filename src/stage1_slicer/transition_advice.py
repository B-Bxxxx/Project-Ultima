"""
Qt-free advice for the transition height of a field function.

Inside the transition zone the real layer spacing of a point is   t = h * (1 - f / T)   (h layer
height, f the function value there, T the transition height; see _compute_custom_field_thickness).
So a positive f squeezes the layers (t -> 0 at f = T: the layers fold over) and a negative f
stretches them. A nozzle can only lay down layers inside a window, taken as

    MIN_FRACTION * nozzle  <=  t  <=  MAX_FRACTION * nozzle         (0.25 .. 0.80 of the diameter)

which gives the smallest transition height that keeps every point of the part printable:

    T >= f_max / (1 - t_min / h)      for f_max > 0
    T >= |f_min| / (t_max / h - 1)    for f_min < 0
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.stage1_slicer.function_file import FunctionSpec
from src.stage1_slicer.function_import import ExpressionField, GridField

MIN_FRACTION = 0.25
MAX_FRACTION = 0.80


@dataclass
class TransitionAdvice:
    f_min: float
    f_max: float
    min_transition: float                 # smallest T that keeps the layers printable (0 = any)
    possible: bool                        # False: no T can fix it (layer height outside the nozzle window)
    thickness_min: Optional[float]        # real layer spacing range at the CURRENT transition height
    thickness_max: Optional[float]
    window: Tuple[float, float]
    ok: bool
    messages: List[str] = field(default_factory=list)


def printable_window(nozzle: float) -> Tuple[float, float]:
    return MIN_FRACTION * nozzle, MAX_FRACTION * nozzle


def field_range(spec: FunctionSpec, values: Dict[str, Any], bounds, n: int = 40) -> Optional[Tuple[float, float]]:
    """(min f, max f) over the XY bounds (x0, x1, y0, y1) of the part; None for tilting functions."""
    if spec.kind == "tilt":
        return None
    x0, x1, y0, y1 = bounds
    gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    if spec.kind == "expression":
        f = ExpressionField(spec.expression, values).evaluate(gx, gy)
    else:
        f = GridField.from_params(spec.to_slicer_params(values)).evaluate(gx, gy)
    f = np.broadcast_to(np.asarray(f, float), gx.shape)
    return float(f.min()), float(f.max())


def advise(f_min: float, f_max: float, layer_height: float, nozzle: float, transition: float) -> TransitionAdvice:
    t_lo, t_hi = printable_window(nozzle)
    h = layer_height
    msgs: List[str] = []
    possible = True
    need = 0.0
    if f_max > 1e-9:
        if h <= t_lo + 1e-12:
            possible = False
            msgs.append(f"layer height {h:g} is already at the lower limit {t_lo:g} of the {nozzle:g} nozzle: "
                        f"a positive function would squeeze the layers below it")
        else:
            need = max(need, f_max / (1.0 - t_lo / h))
    if f_min < -1e-9:
        if h >= t_hi - 1e-12:
            possible = False
            msgs.append(f"layer height {h:g} is already at the upper limit {t_hi:g} of the {nozzle:g} nozzle: "
                        f"a negative function would stretch the layers above it")
        else:
            need = max(need, -f_min / (t_hi / h - 1.0))
    tmin = tmax = None
    ok = possible
    if transition > 1e-9:
        r_lo, r_hi = 1.0 - f_max / transition, 1.0 - f_min / transition
        tmin, tmax = h * r_lo, h * r_hi
        if tmin < t_lo - 1e-9:
            ok = False
            msgs.append(f"layers get as thin as {max(tmin, 0.0):.3f} mm (nozzle {nozzle:g}: at least {t_lo:g})"
                        + (" - that is the layer height itself, raise it" if tmin >= h - 1e-9 else ""))
        if tmax > t_hi + 1e-9:
            ok = False
            msgs.append(f"layers get as thick as {tmax:.3f} mm (nozzle {nozzle:g}: at most {t_hi:g})"
                        + (" - that is the layer height itself, lower it" if tmax <= h + 1e-9 else ""))
    elif max(abs(f_min), abs(f_max)) > 1e-9:
        ok = False
        msgs.append("transition height 0: the very first layer is already deformed and cannot sit flat on the bed")
    if need > 0 and transition + 1e-9 < need:
        ok = False
    return TransitionAdvice(f_min, f_max, need, possible, tmin, tmax, (t_lo, t_hi), ok and possible, msgs)
