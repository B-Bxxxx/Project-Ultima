"""
Minimal G-code parser used by the Stage 4 tests. Tests assert on parsed semantics
(order of moves, axes, E values), never on the template text.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_WORD = re.compile(r"([A-Za-z])\s*(-?\d+(?:\.\d+)?)")


@dataclass
class Cmd:
    line_no: int
    code: str                                   # "G0", "G1", "M83", ... ("" for comment-only lines)
    params: Dict[str, float] = field(default_factory=dict)
    comment: str = ""
    raw: str = ""

    @property
    def has_xy(self) -> bool:
        return "X" in self.params or "Y" in self.params

    @property
    def is_move(self) -> bool:
        return self.code in ("G0", "G1")


def parse(gcode: str) -> List[Cmd]:
    cmds: List[Cmd] = []
    for n, raw in enumerate(gcode.splitlines(), start=1):
        code_part, _, comment = raw.partition(";")
        tokens = code_part.split()
        if not tokens:
            cmds.append(Cmd(n, "", {}, comment.strip(), raw))
            continue
        code = tokens[0].upper()
        params = {m.group(1).upper(): float(m.group(2)) for m in _WORD.finditer(" ".join(tokens[1:]))}
        cmds.append(Cmd(n, code, params, comment.strip(), raw))
    return cmds


def moves(cmds: List[Cmd]) -> List[Cmd]:
    return [c for c in cmds if c.is_move]


def classify(c: Cmd) -> str:
    """retract | prime | extrude | zmove | travel | other"""
    if not c.is_move:
        return "other"
    e = c.params.get("E")
    if e is not None and e < 0 and not c.has_xy and "Z" not in c.params:
        return "retract"
    if e is not None and e > 0 and not c.has_xy and "Z" not in c.params:
        return "prime"
    if e is not None and e > 0:
        return "extrude"
    if c.has_xy:
        return "travel"
    if "Z" in c.params:
        return "zmove"
    return "other"


def travel_runs(cmds: List[Cmd]) -> List[List[Cmd]]:
    """Maximal runs of consecutive non-extruding moves (what happens between two extrusions)."""
    runs: List[List[Cmd]] = []
    current: List[Cmd] = []
    for c in moves(cmds):
        if classify(c) == "extrude":
            if current:
                runs.append(current)
            current = []
        else:
            current.append(c)
    if current:
        runs.append(current)
    return runs


def layer_markers(cmds: List[Cmd]) -> List[int]:
    return [int(m.group(1)) for c in cmds if c.code == "" and (m := re.match(r"^LAYER:(\d+)$", c.comment))]


def type_markers(cmds: List[Cmd]) -> List[str]:
    return [m.group(1) for c in cmds if c.code == "" and (m := re.match(r"^TYPE:\s*(\S+)$", c.comment))]
