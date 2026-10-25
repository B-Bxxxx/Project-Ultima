"""
Mathematical description of a machine's axis chain (used by the parallel `generic_chain` solver).

A chain is two lists of joints, each listed OUTERMOST FIRST (= "who carries whom"):

    "tool":  joints between the machine base and the nozzle      (the nozzle sits at the end of this list)
    "table": joints between the machine base and the printed part (the part sits at the end of this list)

A joint is given in the machine's ZERO POSE (all joints at 0), in the machine base frame:

    {"axis": "B", "type": "rotary", "direction": [0,1,0], "point": [px,py,pz],
     "sign": 1, "min": -90, "max": 90, "continuous": false}
    {"axis": "X", "type": "linear", "direction": [1,0,0], "sign": 1}

`direction` is the right-handed rotation axis / the translation direction, `point` a point on the rotation axis,
`sign` (+1/-1) the direction of the machine's axis value relative to that mathematical direction (machine value =
sign * mathematical value), `min`/`max` the limits of the MACHINE value, `continuous` a rotary axis that may wind
around freely (its angle is re-chosen at every travel move, see base.resolve_c_angles).

Product of exponentials: T_branch = T_1(q_1) T_2(q_2) ... T_n(q_n), every T_j built from its zero-pose axis.
Position of the nozzle tip in the base frame:  T_tool(tip).  Position of a part point P:  T_table(P + part_origin).
The nozzle axis (zero pose `tool_axis`) must lie along the layer normal N (part frame):  R_tool tool_axis = R_table N.

Every number may be an expression of the machine file's parameters ("-pivot_offset_x", "c_axis_x + eccentricity_x").
"""
import ast
import operator
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.common.param_spec import SpecError

AXIS_NAMES = ("X", "Y", "Z", "B", "C")
LINEAR_NAMES = ("X", "Y", "Z")
ROTARY_NAMES = ("B", "C")
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}


class ChainError(SpecError):
    pass


def eval_expr(value: Any, params: Dict[str, Any], where: str = "") -> float:
    """A number, or a string expression (+ - * / ( ) numbers and parameter names only)."""
    if isinstance(value, bool):
        raise ChainError(f"{where}a boolean is not a number")
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ChainError(f"{where}expected a number or an expression string, got {value!r}")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in params:
                raise ChainError(f"{where}unknown parameter '{node.id}' in expression {value!r}")
            return float(params[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return _BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
            return _UN[type(node.op)](ev(node.operand))
        raise ChainError(f"{where}unsupported expression {value!r} (only numbers, parameters and + - * / are allowed)")

    try:
        return float(ev(ast.parse(value.strip(), mode="eval")))
    except ChainError:
        raise
    except (SyntaxError, ZeroDivisionError) as e:
        raise ChainError(f"{where}bad expression {value!r}: {e}") from e


def _vec(raw, params, where, default=None, unit=False):
    if raw is None:
        if default is None:
            raise ChainError(f"{where}is missing")
        raw = default
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ChainError(f"{where}must be a list of 3 numbers")
    v = np.array([eval_expr(x, params, where) for x in raw], float)
    if unit:
        n = np.linalg.norm(v)
        if n < 1e-9:
            raise ChainError(f"{where}must not be the zero vector")
        v = v / n
    return v


@dataclass
class Joint:
    axis: str                          # machine axis name: X Y Z B C
    rotary: bool
    direction: np.ndarray              # unit vector, zero pose
    point: np.ndarray                  # a point on the rotation axis (rotary only)
    sign: float = 1.0
    lo: float = -np.inf                # limits of the MACHINE value (deg for rotary)
    hi: float = np.inf
    continuous: bool = False

    def math_limits(self):
        """Limits of the mathematical angle (deg): machine value = sign * math value."""
        if self.sign < 0:
            return -self.hi, -self.lo
        return self.lo, self.hi


@dataclass
class Chain:
    tool: List[Joint]
    table: List[Joint]
    part_origin: np.ndarray
    tip: np.ndarray
    tool_axis: np.ndarray

    @property
    def joints(self):
        return self.tool + self.table

    def rotaries(self):
        return [j for j in self.joints if j.rotary]

    # ---------------------------------------------------------------- parsing / serialising
    @classmethod
    def parse(cls, raw: Any, params: Optional[Dict[str, Any]] = None, where: str = "") -> "Chain":
        params = params or {}
        if not isinstance(raw, dict):
            raise ChainError(f'{where}"chain" must be an object')
        extra = set(raw) - {"tool", "table", "part_origin", "tip", "tool_axis"}
        if extra:
            raise ChainError(f'{where}unknown "chain" field(s) {sorted(extra)}')
        branches = []
        for key in ("tool", "table"):
            lst = raw.get(key, [])
            if not isinstance(lst, list):
                raise ChainError(f'{where}"chain.{key}" must be a list of joints')
            branches.append([cls._joint(j, params, f"{where}chain.{key}[{i}]: ") for i, j in enumerate(lst)])
        ch = cls(tool=branches[0], table=branches[1],
                 part_origin=_vec(raw.get("part_origin"), params, f"{where}chain.part_origin ", default=[0, 0, 0]),
                 tip=_vec(raw.get("tip"), params, f"{where}chain.tip ", default=[0, 0, 0]),
                 tool_axis=_vec(raw.get("tool_axis"), params, f"{where}chain.tool_axis ", default=[0, 0, 1], unit=True))
        ch.validate(where)
        return ch

    @staticmethod
    def _joint(raw, params, where) -> Joint:
        if not isinstance(raw, dict):
            raise ChainError(f"{where}a joint must be an object")
        extra = set(raw) - {"axis", "type", "direction", "point", "sign", "min", "max", "continuous"}
        if extra:
            raise ChainError(f"{where}unknown joint field(s) {sorted(extra)}")
        axis = raw.get("axis")
        if axis not in AXIS_NAMES:
            raise ChainError(f'{where}"axis" must be one of {list(AXIS_NAMES)} (A is not supported yet)')
        typ = raw.get("type")
        if typ not in ("linear", "rotary"):
            raise ChainError(f'{where}"type" must be "linear" or "rotary"')
        if (typ == "linear") != (axis in LINEAR_NAMES):
            raise ChainError(f"{where}axis {axis} cannot be {typ}")
        sign = eval_expr(raw.get("sign", 1), params, f"{where}sign: ")
        if abs(abs(sign) - 1.0) > 1e-9:
            raise ChainError(f"{where}\"sign\" must be +1 or -1")
        j = Joint(axis=axis, rotary=typ == "rotary", direction=_vec(raw.get("direction"), params, f"{where}direction ", unit=True),
                  point=_vec(raw.get("point"), params, f"{where}point ", default=[0, 0, 0]), sign=1.0 if sign > 0 else -1.0,
                  continuous=bool(raw.get("continuous", False)))
        if j.rotary:
            for key, attr in (("min", "lo"), ("max", "hi")):
                if key in raw:
                    setattr(j, attr, eval_expr(raw[key], params, f"{where}{key}: "))
            if j.lo > j.hi:
                raise ChainError(f'{where}"min" is greater than "max"')
        elif j.continuous or "min" in raw or "max" in raw:
            raise ChainError(f"{where}limits / continuous only apply to rotary joints (the build volume limits the linear axes)")
        return j

    def validate(self, where: str = ""):
        names = [j.axis for j in self.joints]
        if len(set(names)) != len(names):
            raise ChainError(f"{where}every axis may appear only once in the chain")
        rot = self.rotaries()
        if len(rot) > 2:
            raise ChainError(f"{where}at most two rotary joints are supported for now (the third one - 6 axis - is in progress)")
        if len(rot) == 2 and np.linalg.norm(np.cross(rot[0].direction, rot[1].direction)) < 1e-6:
            raise ChainError(f"{where}the two rotary axes must not be parallel")
        lin = [j for j in self.joints if not j.rotary]
        if len(lin) != 3 or {j.axis for j in lin} != set(LINEAR_NAMES):
            raise ChainError(f"{where}the chain needs exactly one X, one Y and one Z linear joint")
        A = np.array([j.direction for j in lin])
        # the directions must span 3D (otherwise the position cannot be reached)
        if abs(np.linalg.det(A)) < 1e-6:
            raise ChainError(f"{where}the X, Y, Z directions must be linearly independent")

    def to_dict(self) -> Dict[str, Any]:
        """Purely numeric form (no expressions), accepted again by `parse`."""
        def jd(j: Joint):
            d = {"axis": j.axis, "type": "rotary" if j.rotary else "linear", "direction": [float(x) for x in j.direction], "sign": j.sign}
            if j.rotary:
                d["point"] = [float(x) for x in j.point]
                if np.isfinite(j.lo):
                    d["min"] = float(j.lo)
                if np.isfinite(j.hi):
                    d["max"] = float(j.hi)
                d["continuous"] = j.continuous
            return d
        return {"tool": [jd(j) for j in self.tool], "table": [jd(j) for j in self.table],
                "part_origin": [float(x) for x in self.part_origin], "tip": [float(x) for x in self.tip],
                "tool_axis": [float(x) for x in self.tool_axis]}


# ------------------------------------------------------------------------------------ geometry (vectorised)
def rodrigues(axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """(n,3,3) rotations about the unit `axis` by the angles `theta` (rad)."""
    th = np.asarray(theta, float)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    c, s = np.cos(th)[:, None, None], np.sin(th)[:, None, None]
    return np.eye(3)[None] + s * K[None] + (1.0 - c) * (K @ K)[None]


def branch_transform(joints: List[Joint], values: Dict[str, np.ndarray], n: int):
    """
    Composite rigid transform (R (n,3,3), t (n,3)) of a branch. `values` maps the axis name to the MATHEMATICAL value
    (rad for rotary, mm for linear); an axis that is not in `values` is at 0.
    """
    R = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    t = np.zeros((n, 3))
    for j in joints:
        q = values.get(j.axis)
        q = np.zeros(n) if q is None else np.asarray(q, float)
        if j.rotary:
            Rj = rodrigues(j.direction, q)
            tj = j.point[None] - Rj @ j.point
        else:
            Rj = np.broadcast_to(np.eye(3), (n, 3, 3))
            tj = q[:, None] * j.direction[None]
        t = t + np.einsum("nij,nj->ni", R, tj)
        R = R @ Rj
    return R, t


def angle_about(axis: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Signed angle (rad) of the rotation about `axis` that takes the vectors u (n,3) towards w (n,3) (components along the axis ignored)."""
    up = u - (u @ axis)[:, None] * axis[None]
    wp = w - (w @ axis)[:, None] * axis[None]
    return np.arctan2(np.einsum("ni,i->n", np.cross(up, wp), axis), np.einsum("ni,ni->n", up, wp))


def two_planes(a: np.ndarray, b: np.ndarray, alpha: np.ndarray, beta: np.ndarray):
    """Both unit vectors v (n,3) each with a.v = alpha and b.v = beta (the nearest one twice when unreachable)."""
    g = float(a @ b)
    cr = np.cross(a, b)
    det = 1.0 - g * g
    u = (alpha - g * beta) / det
    w = (beta - g * alpha) / det
    v0 = u[:, None] * a[None] + w[:, None] * b[None]
    rest = (1.0 - np.einsum("ni,ni->n", v0, v0)) / (cr @ cr)
    s = np.sqrt(np.clip(rest, 0.0, None))
    reach = np.linalg.norm(v0, axis=1)
    v0 = np.where((reach > 1.0)[:, None], v0 / np.maximum(reach, 1e-12)[:, None], v0)
    return v0 + s[:, None] * cr[None], v0 - s[:, None] * cr[None]
