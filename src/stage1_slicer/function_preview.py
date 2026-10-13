"""
Qt-free preview of a function file: the shape the layers take over a rectangular area.

`height_map` returns the Z shift of the LAST layer relative to its flat position (z = z0 + shift):
  * expression / grid : -f(x, y)
  * conical           : -r * tan(cone angle)
  * progressive_tilt  : (y - pivot_y) * tan(end tilt)
  * planar            : 0
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from src.stage1_slicer.function_file import FunctionFileError, FunctionSpec
from src.stage1_slicer.function_import import ExpressionField, GridField


@dataclass
class HeightMap:
    xs: np.ndarray
    ys: np.ndarray
    z: np.ndarray                 # shape (len(ys), len(xs))
    zmin: float
    zmax: float
    label: str


def height_map(spec: FunctionSpec, overrides: Optional[Dict[str, Any]] = None,
               bounds: Sequence[float] = (-20.0, 20.0, -20.0, 20.0), n: int = 60) -> HeightMap:
    x0, x1, y0, y1 = bounds
    if not (x1 > x0 and y1 > y0):
        raise FunctionFileError("preview area must have x1 > x0 and y1 > y0")
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    gx, gy = np.meshgrid(xs, ys)
    v = spec.values(overrides)

    if spec.type == "planar":
        z, label = np.zeros_like(gx), "Flat layers - no deformation"
    elif spec.type == "progressive_tilt":
        pivot = float(v.get("pivot_y", y0))
        z = (gy - pivot) * np.tan(np.radians(v["end_tilt_deg"]))
        label = (f"Last layer tilted {v['end_tilt_deg']:g} deg about X (pivot y = {pivot:g}); "
                 f"first layer {v['start_tilt_deg']:g} deg")
    elif spec.type == "conical":
        z = -np.sqrt(gx ** 2 + gy ** 2) * np.tan(np.radians(v["cone_angle_deg"]))
        label = f"Cone {v['cone_angle_deg']:g} deg: layer height falls with the radius (z = z0 - r*tan a)"
    elif spec.type == "expression":
        f = np.asarray(ExpressionField(spec.expression, v).evaluate(gx, gy), float)
        z = -(np.full(gx.shape, float(f)) if f.shape != gx.shape else f)
        label = f"z = z0 - ({spec.expression})"
    else:  # grid
        params = spec.to_slicer_params(overrides)
        z = -np.asarray(GridField.from_params(params).evaluate(gx, gy), float)
        label = f"z = z0 - f(x, y) from {spec.data_file}"
    return HeightMap(xs=xs, ys=ys, z=z, zmin=float(z.min()), zmax=float(z.max()), label=label)
