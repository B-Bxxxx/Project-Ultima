"""
G-code checker (Qt-free): reads a G-code file back and shows what the FIRMWARE will do with it.

    parse_gcode(text)                    G0/G1 moves with X Y Z A B C, F, E, ;LAYER and ; TYPE markers
    reconstruct(moves, solver, config)   the way the nozzle really goes over the PART

The firmware moves all axes LINEARLY between two G-code points (Klipper cartesian6) and a continuous C axis takes the
SHORTEST way to the target in G90. Every move is therefore cut into small steps in axis space (X Y Z A B C interpolated
linearly, C by the shortest angle) and each step is sent through the solver's inverse (`to_part_batch`) to the part
frame. A straight segment of the part that is deposited as an arc (C swinging by 180 deg while B ~ 0 ...) shows up as a
large `deviation`: the distance of the interpolated path from the straight chord between the two G-code points.
"""
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

WORD = re.compile(r"([A-Z])\s*([-+]?(?:\d+\.?\d*|\.\d+))")
LAYER_RE = re.compile(r";\s*LAYER\s*:\s*(-?\d+)", re.I)
TYPE_RE = re.compile(r";\s*TYPE\s*:\s*(\S+)", re.I)
AXES = "XYZABC"                       # pose column order of a Move: x y z a b c
SHORTEST = "C"                        # axes that take the shortest way in G90 (continuous_axis + shortest_path)


@dataclass
class Move:
    line: int                          # 1-based line number in the file
    kind: str                          # "rapid" (G0) | "feed" (G1)
    layer: int
    feature: str                       # "travel" for moves without extrusion
    p0: np.ndarray                     # pose before (x y z a b c)
    p1: np.ndarray                     # pose commanded
    feed: float                        # mm/min of the line (0 if unknown)
    extrusion: float                   # E of the line (0 = no extrusion)


def parse_gcode(text: str) -> Tuple[List[Move], List[str]]:
    pos = np.zeros(6)
    absolute, layer, feature, feed = True, 0, "extrusion", 0.0
    moves: List[Move] = []
    warnings: List[str] = []

    def warn(ln, msg):
        if len(warnings) < 30:
            warnings.append(f"line {ln}: {msg}")

    for i, raw in enumerate(text.splitlines()):
        ln = i + 1
        m = LAYER_RE.search(raw)
        if m:
            layer = int(m.group(1))
        m = TYPE_RE.search(raw)
        if m:
            feature = m.group(1)
        code = raw.split(";", 1)[0].upper().strip()
        if not code:
            continue
        words = [(l, float(v)) for l, v in WORD.findall(code)]
        gs = {int(v) for l, v in words if l == "G" and abs(v - round(v)) < 1e-9}
        if 90 in gs:
            absolute = True
        if 91 in gs:
            absolute = False
        if gs & {2, 3}:
            warn(ln, "arcs (G2/G3) are not supported, skipped")
            continue
        if gs & {28, 92}:
            warn(ln, f"G{sorted(gs & {28, 92})[0]} is not simulated")
            continue
        if not gs & {0, 1}:
            continue
        w = dict(words)
        if "F" in w:
            feed = w["F"]
        tgt = pos.copy()
        for k, a in enumerate(AXES):
            if a in w:
                tgt[k] = w[a] if absolute else pos[k] + w[a]
        e = w.get("E", 0.0)
        if np.allclose(tgt, pos, atol=1e-9):
            continue                                   # retract / prime / feed-rate only
        is_g0 = 0 in gs
        moves.append(Move(ln, "rapid" if is_g0 else "feed", layer, "travel" if (is_g0 or e <= 0) else feature,
                          pos.copy(), tgt.copy(), feed, e))
        pos = tgt
    return moves, warnings


@dataclass
class ViewPoint:
    """One sample of the reconstructed path (what the GUI draws; same fields as a machine state)."""
    x: float
    y: float
    z: float
    layer_index: int
    feature_type: str
    is_travel_move: bool
    bad: bool = False


@dataclass
class Reconstruction:
    moves: List[Move]
    move_of_point: np.ndarray          # (m,) index into `moves` for every sample (-1 = start point)
    machine: np.ndarray                # (m, 3) machine X Y Z of the samples
    part: np.ndarray                   # (m, 3) the same samples on the part
    poses: np.ndarray                  # (m, 6) x y z a b c with C unwrapped like the firmware does
    deviation: np.ndarray              # (n_moves,) max distance from the straight chord on the part (mm)
    c_swing: np.ndarray                # (n_moves,) |delta C| the firmware really turns (deg)
    time_s: np.ndarray                 # (n_moves,) L6 / F with the firmware's 6D length
    stats: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def bad_moves(self, tol: float) -> np.ndarray:
        feed = np.array([m.kind == "feed" and m.feature != "travel" for m in self.moves], bool)
        return feed & (self.deviation > tol)

    def points(self, frame: str = "part", tol: float = 0.05) -> List[ViewPoint]:
        """Samples as drawable items; `frame` is "part" or "machine". Samples of a bad move are flagged."""
        P = self.part if frame == "part" else self.machine
        bad = self.bad_moves(tol)
        out = []
        for k in range(len(P)):
            mi = int(self.move_of_point[k])
            mv = self.moves[mi] if mi >= 0 else self.moves[0]
            out.append(ViewPoint(float(P[k, 0]), float(P[k, 1]), float(P[k, 2]), max(mv.layer, 0), mv.feature,
                                 mv.feature == "travel", bool(mi >= 0 and bad[mi])))
        return out


def _wrap(d):
    return (np.asarray(d, float) + 180.0) % 360.0 - 180.0


def reconstruct(moves: Sequence[Move], solver, config, step_deg: float = 2.0, max_sub: int = 48,
                shortest: str = SHORTEST) -> Reconstruction:
    """See the module docstring. `solver` has to_part_batch (otherwise the part frame = the machine frame)."""
    if not moves:
        raise ValueError("no moves in the G-code")
    n = len(moves)
    cols = int(getattr(solver, "pose_columns", 5))

    # the firmware's own position: a shortest-path axis ends at the equivalent angle nearest to where it stands
    eff0 = np.zeros((n, 6))
    eff1 = np.zeros((n, 6))
    cur = moves[0].p0.copy()
    sidx = [AXES.index(a) for a in shortest]
    for i, mv in enumerate(moves):
        tgt = mv.p1.copy()
        for k in sidx:
            tgt[k] = tgt[k] + 360.0 * np.round((cur[k] - tgt[k]) / 360.0)
        eff0[i], eff1[i] = cur, tgt
        cur = tgt
    delta = eff1 - eff0
    rot = np.abs(delta[:, 3:]).max(axis=1)
    k = np.clip(np.ceil(rot / step_deg), 1, max_sub).astype(int)
    k = np.where((rot > 0.2) & (k < 4), 4, k)

    total = int(k.sum()) + 1
    mid = np.concatenate([[-1], np.repeat(np.arange(n), k)])
    start = np.cumsum(k) - k
    j = np.arange(total - 1) - np.repeat(start, k) + 1
    u = j / np.repeat(k, k)
    poses = np.empty((total, 6))
    poses[0] = eff0[0]
    poses[1:] = eff0[mid[1:]] + delta[mid[1:]] * u[:, None]

    def to_part(P6):
        sel = [0, 1, 2, 4, 5] + ([3] if cols > 5 else [])         # solver pose order: x y z b c [a]
        got = solver.to_part_batch(config, P6[:, sel]) if hasattr(solver, "to_part_batch") else None
        return None if got is None else np.asarray(got, float)

    part = to_part(poses)
    warnings = []
    if part is None:
        part = poses[:, :3].copy()
        if np.abs(poses[:, 3:]).max() > 1e-9:
            warnings.append("the selected machine has no rotary axes (no inverse kinematics), but the G-code turns A/B/C: "
                            "select the machine the G-code was made for; the part view equals the machine view")

    # deviation from the chord (part frame), per move
    ends = np.cumsum(k)                                         # index (in poses) of the END sample of move i: ends[i]
    begins = ends - k
    dev = np.zeros(n)
    for i in range(n):
        a, b = part[begins[i]], part[ends[i]]
        seg = part[begins[i]:ends[i] + 1]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            dev[i] = float(np.linalg.norm(seg - a, axis=1).max())
        else:
            t = np.clip(((seg - a) @ d) / (L * L), 0.0, 1.0)
            dev[i] = float(np.linalg.norm(seg - (a + t[:, None] * d), axis=1).max())

    feed = np.array([m.feed for m in moves])
    L6 = np.sqrt((delta ** 2).sum(axis=1))
    Lxyz = np.linalg.norm(delta[:, :3], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t6 = np.where(feed > 0, L6 / (feed / 60.0), 0.0)
        txyz = np.where(feed > 0, Lxyz / (feed / 60.0), 0.0)
    is_feed = np.array([m.kind == "feed" and m.feature != "travel" for m in moves], bool)
    c_swing = np.abs(delta[:, 5])
    stats = {
        "moves": float(n), "extruding": float(is_feed.sum()),
        "time_min_6d": float(t6.sum() / 60.0), "time_min_xyz": float(txyz.sum() / 60.0),
        "max_deviation": float(dev[is_feed].max()) if is_feed.any() else 0.0,
        "c_swing_max": float(c_swing[is_feed].max()) if is_feed.any() else 0.0,
        "c_over_90": float((is_feed & (c_swing > 90)).sum()),
    }
    amb = is_feed & (np.abs(np.abs(_wrap(np.array([m.p1[5] - m.p0[5] for m in moves]))) - 180.0) < 1e-6)
    if amb.any():
        warnings.append(f"{int(amb.sum())} extruding moves turn C by exactly 180 deg: the direction is the firmware's rounding choice")
    return Reconstruction(list(moves), mid, poses[:, :3].copy(), part, poses, dev, c_swing, t6, stats, warnings)


def table_map(solver, config, pose, tool_len: float = 0.0):
    """Rigid map of the PART into the MACHINE frame for one machine pose (x y z a b c): machine = M @ part + t.

    The solver's inverse (`to_part_batch`) is affine in X Y Z for fixed rotary values, so it is probed at the origin and the
    three unit vectors. Returns (M, t); (identity, 0) if the solver has no inverse (no rotary axes)."""
    pose = np.asarray(pose, float)
    cols = int(getattr(solver, "pose_columns", 5))
    sel = [0, 1, 2, 4, 5] + ([3] if cols > 5 else [])
    probe = np.tile(pose, (4, 1))
    probe[0, :3] = 0.0
    for k in range(3):
        probe[k + 1, :3] = 0.0
        probe[k + 1, k] = 1.0
    got = solver.to_part_batch(config, probe[:, sel]) if hasattr(solver, "to_part_batch") else None
    if got is None:
        return np.eye(3), np.zeros(3)
    got = np.asarray(got, float)
    d = got[0]
    A = (got[1:] - d).T                    # part = A @ xyz + d
    Ai = np.linalg.inv(A)
    return Ai, -Ai @ d
