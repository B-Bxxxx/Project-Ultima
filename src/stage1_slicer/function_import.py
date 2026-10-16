"""
Stage 0 function import: user-supplied deformation functions f(x, y).

Coordinate reference (decision D4, revised): x, y are relative to the CENTRE of the part's footprint
(bounding-box centre of the placed mesh; the strategies subtract origin_x / origin_y), z is not an argument. The sliced layers follow z = z0 - w(z)*f(x, y)
(w = transition blend), i.e. a positive f lowers the layer surface.

A. Expression file   - text (one expression, '#'/'//' comment lines ignored, lines are joined) or
                       JSON {"expression": "..."}; validated by the AST whitelist (safe_expr).
B. Data file         - CSV/whitespace text with columns x, y, f (header optional). A complete
                       regular grid is used as is; scattered points are resampled to a regular grid
                       with inverse-distance weighting. Pure numpy (no scipy, AGENTS.md rule 3).
"""
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from src.stage1_slicer.safe_expr import compile_expression, evaluate_compiled

MAX_EXPRESSION_FILE_BYTES = 10_000
MAX_DATA_POINTS = 500_000
DEFAULT_GRID_RESOLUTION = 64


# ------------------------------------------------------------------------------------ fields
@dataclass
class GridField:
    """Regular grid f(x_i, y_j) with bilinear interpolation and clamping outside the grid."""
    xs: np.ndarray            # (nx,) ascending
    ys: np.ndarray            # (ny,) ascending
    values: np.ndarray        # (ny, nx)

    def __post_init__(self):
        self.xs = np.asarray(self.xs, float)
        self.ys = np.asarray(self.ys, float)
        self.values = np.asarray(self.values, float)
        if self.xs.ndim != 1 or self.ys.ndim != 1 or len(self.xs) < 2 or len(self.ys) < 2:
            raise ValueError("grid needs at least 2 x and 2 y positions")
        if self.values.shape != (len(self.ys), len(self.xs)):
            raise ValueError(f"grid_values must have shape ({len(self.ys)}, {len(self.xs)}), "
                             f"got {self.values.shape}")
        if np.any(np.diff(self.xs) <= 0) or np.any(np.diff(self.ys) <= 0):
            raise ValueError("grid_x and grid_y must be strictly ascending")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("grid values must be finite")

    def evaluate(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        xc = np.clip(x, self.xs[0], self.xs[-1])
        yc = np.clip(y, self.ys[0], self.ys[-1])
        ix = np.clip(np.searchsorted(self.xs, xc, side="right") - 1, 0, len(self.xs) - 2)
        iy = np.clip(np.searchsorted(self.ys, yc, side="right") - 1, 0, len(self.ys) - 2)
        tx = (xc - self.xs[ix]) / (self.xs[ix + 1] - self.xs[ix])
        ty = (yc - self.ys[iy]) / (self.ys[iy + 1] - self.ys[iy])
        v = self.values
        out = (v[iy, ix] * (1 - tx) * (1 - ty) + v[iy, ix + 1] * tx * (1 - ty)
               + v[iy + 1, ix] * (1 - tx) * ty + v[iy + 1, ix + 1] * tx * ty)
        return out[()] if out.ndim == 0 else out

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return float(self.xs[0]), float(self.xs[-1]), float(self.ys[0]), float(self.ys[-1])

    @property
    def value_range(self) -> Tuple[float, float]:
        return float(self.values.min()), float(self.values.max())

    def to_params(self) -> Dict[str, Any]:
        return {"grid_x": self.xs.tolist(), "grid_y": self.ys.tolist(), "grid_values": self.values.tolist()}

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "GridField":
        for k in ("grid_x", "grid_y", "grid_values"):
            if k not in params:
                raise ValueError(f"grid_field strategy needs the parameter '{k}'")
        return cls(params["grid_x"], params["grid_y"], params["grid_values"])


class ExpressionField:
    """Validated, compiled expression f(x, y) with the same evaluate() interface as GridField."""

    def __init__(self, expression: str, parameters=None):
        self.expression = expression
        self.parameters = {k: float(v) for k, v in (parameters or {}).items()}
        self._code = compile_expression(expression, self.parameters.keys())   # raises ValueError on unsafe/invalid input

    def evaluate(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return evaluate_compiled(self._code, x, y, np.sqrt(x ** 2 + y ** 2), self.parameters)


# --------------------------------------------------------------------------------- A. expression
def load_expression_file(path: str) -> str:
    """Reads and validates a single expression from a text or JSON file."""
    if os.path.getsize(path) > MAX_EXPRESSION_FILE_BYTES:
        raise ValueError(f"expression file too large (max {MAX_EXPRESSION_FILE_BYTES} bytes)")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.lower().endswith(".json"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON: {e}")
        if not isinstance(data, dict) or not isinstance(data.get("expression"), str):
            raise ValueError('JSON expression file must be an object with a string field "expression"')
        expr = data["expression"].strip()
    else:
        lines = [ln.strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln and not ln.startswith("#") and not ln.startswith("//")]
        expr = " ".join(lines)
    if not expr:
        raise ValueError("expression file contains no expression")
    compile_expression(expr)            # validate now so the GUI can report the error at load time
    return expr


# ------------------------------------------------------------------------------------ B. data
_NUM = re.compile(r"^[\s+\-0-9.eE,;]+$")


def _read_columns(path: str) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.split("#", 1)[0].strip()
            if not ln:
                continue
            parts = [p for p in re.split(r"[,;\t ]+", ln) if p != ""]
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                if not rows:
                    continue                    # header line
                raise ValueError(f"non-numeric data row: {ln!r}")
            if len(vals) < 3:
                raise ValueError(f"data rows need 3 columns x, y, f; got: {ln!r}")
            rows.append(vals[:3])
            if len(rows) > MAX_DATA_POINTS:
                raise ValueError(f"too many data points (max {MAX_DATA_POINTS})")
    if len(rows) < 4:
        raise ValueError("need at least 4 data points")
    arr = np.asarray(rows, float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("data contains non-finite values")
    return arr


def _regular_grid(arr: np.ndarray) -> Optional[GridField]:
    xs = np.unique(arr[:, 0])
    ys = np.unique(arr[:, 1])
    if len(xs) < 2 or len(ys) < 2 or len(xs) * len(ys) != len(arr):
        return None
    ix = np.searchsorted(xs, arr[:, 0])
    iy = np.searchsorted(ys, arr[:, 1])
    vals = np.full((len(ys), len(xs)), np.nan)
    vals[iy, ix] = arr[:, 2]
    if np.isnan(vals).any():
        return None
    return GridField(xs, ys, vals)


def resample_scattered(points: np.ndarray, resolution: int = DEFAULT_GRID_RESOLUTION, k: int = 8) -> GridField:
    """Inverse-distance-weighted resampling of scattered (x, y, f) points onto a regular grid."""
    x0, x1 = points[:, 0].min(), points[:, 0].max()
    y0, y1 = points[:, 1].min(), points[:, 1].max()
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        raise ValueError("data points must span a 2D area (all x or all y are equal)")
    span = max(x1 - x0, y1 - y0)
    nx = max(2, int(round(resolution * (x1 - x0) / span)))
    ny = max(2, int(round(resolution * (y1 - y0) / span)))
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    gx, gy = np.meshgrid(xs, ys)
    q = np.column_stack([gx.ravel(), gy.ravel()])
    k = min(k, len(points))
    out = np.empty(len(q))
    for s in range(0, len(q), 256):                       # chunked: bounded memory
        chunk = q[s:s + 256]
        d = np.linalg.norm(chunk[:, None, :] - points[None, :, :2], axis=2)
        idx = np.argpartition(d, k - 1, axis=1)[:, :k]
        dk = np.take_along_axis(d, idx, axis=1)
        fk = points[idx, 2]
        exact = dk.min(axis=1) < 1e-12
        w = 1.0 / np.maximum(dk, 1e-12) ** 2
        res = (w * fk).sum(axis=1) / w.sum(axis=1)
        if exact.any():
            res[exact] = fk[np.arange(len(chunk)), dk.argmin(axis=1)][exact]
        out[s:s + 256] = res
    return GridField(xs, ys, out.reshape(ny, nx))


def load_data_file(path: str, resolution: int = DEFAULT_GRID_RESOLUTION) -> GridField:
    """Loads x, y, f samples (grid or scattered) and returns a GridField."""
    arr = _read_columns(path)
    return _regular_grid(arr) or resample_scattered(arr, resolution)


# ---------------------------------------------------------------------------------- preview
def preview_surface(field, bounds_xy: Sequence[float], n: int = 40, origin: Sequence[float] = (0.0, 0.0)):
    """
    Samples `field` (anything with evaluate(x, y)) on an n x n grid over (x0, x1, y0, y1).
    Returns xs, ys, Z[(ny, nx)], (zmin, zmax) - used by the Stage 0 preview. `origin` is the function origin
    (centre of the part): the field is evaluated at (x - origin_x, y - origin_y).
    """
    x0, x1, y0, y1 = bounds_xy
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    gx, gy = np.meshgrid(xs, ys)
    z = np.asarray(field.evaluate(gx - origin[0], gy - origin[1]), float)
    if z.shape != gx.shape:                              # constant expression like "0.5"
        z = np.full(gx.shape, float(z))
    return xs, ys, z, (float(z.min()), float(z.max()))
