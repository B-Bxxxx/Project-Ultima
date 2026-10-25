import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

from src.common.schemas import MachineStateVector, MachineTrajectory, MinimalPrintProfile
from src.stage4_postproc.base import BasePostProcessor

LAYER_MARKER_RE = re.compile(r"^;LAYER:(\d+)\s*$")
FOOTER_MARKER = "; End G-Code"
MIN_TOOL_ALIGNMENT = 0.2          # cos(78 deg): limits the thickness correction of nozzle_axis


@dataclass
class GcodeLayers:
    """A G-code file split at its ';LAYER:n' markers (header + layers + footer)."""
    header: str = ""
    layers: Dict[int, str] = field(default_factory=dict)
    footer: str = ""

    def join(self) -> str:
        """Reassembles the original G-code (split_gcode_by_layer(g).join() == g)."""
        return self.header + "".join(self.layers[i] for i in sorted(self.layers)) + self.footer


def split_gcode_by_layer(gcode: str) -> GcodeLayers:
    """
    Splits G-code produced by the post-processor into header, per-layer blocks and footer.
    A layer block starts at its ';LAYER:n' line and runs to the next marker or to the end block
    ('; -------------------' + '; End G-Code').
    """
    lines = gcode.splitlines(keepends=True)
    footer_at = len(lines)
    for i, line in enumerate(lines):
        if line.strip() == FOOTER_MARKER:
            footer_at = i - 1 if i > 0 and lines[i - 1].startswith("; ---") else i
            break

    out = GcodeLayers()
    current: Optional[int] = None
    header, footer = [], lines[footer_at:]
    blocks: Dict[int, List[str]] = {}
    for line in lines[:footer_at]:
        m = LAYER_MARKER_RE.match(line)
        if m:
            current = int(m.group(1))
            blocks.setdefault(current, [])
        if current is None:
            header.append(line)
        else:
            blocks[current].append(line)
    out.header = "".join(header)
    out.layers = {i: "".join(b) for i, b in blocks.items()}
    out.footer = "".join(footer)
    return out


class Klipper5AxisPostProcessor(BasePostProcessor):
    """
    Stage 4 Post-Processor for Klipper 5-axis.
    Converts extrusion volume to linear E delta, plans the travel sequences and templates G-code.

    Output conventions
      * relative extrusion (M83), absolute positioning (G90)
      * ';LAYER:n' before the first move of every layer (a travel belongs to the layer it travels
        into), ';LAYER_COUNT:N' in the header
      * '; TYPE: <feature>' before the first extrusion of every layer and whenever the feature changes
      * every travel of at least `min_retract_travel` is: retract -> Z-hop up (highest Z of the layer + z_hop) -> travel -> Z-hop
        down -> prime (retract and unretract distance / speed are separate parameters); shorter travels are made directly. The travel and hop moves use
        `profile.travel_command` (G0 or G1).
      * thickness_model 'nozzle_axis': E is scaled by 1 / cos(angle between nozzle axis and layer normal)
      * the end block lifts the nozzle `park_lift` mm above the highest point of the part (no fixed Z)
      * no heater commands are written at the start (temperatures are handled by the printer
        configuration / start macro)
    """

    def generate_gcode(self, trajectory: MachineTrajectory, profile: MinimalPrintProfile) -> str:
        moves = self._plan_moves(trajectory.states, profile)
        park_z = max((s.z for s in trajectory.states), default=0.0) + profile.park_lift
        layer_count = len({s.layer_index for s in trajectory.states})

        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        env = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True,
                          keep_trailing_newline=True)
        template = env.get_template("klipper_5axis.gcode.j2")
        return template.render(moves=moves, profile=profile, cmd=profile.travel_command,
                               layer_count=layer_count, park_z=park_z,
                               rot_home=" ".join(f"{l}0" for l in profile.rotary_axes if l in "ABC"))

    def generate_gcode_layers(self, trajectory: MachineTrajectory, profile: MinimalPrintProfile) -> GcodeLayers:
        """The same G-code, split into header / per-layer blocks / footer."""
        return split_gcode_by_layer(self.generate_gcode(trajectory, profile))

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _rot_words(s: MachineStateVector, profile: MinimalPrintProfile) -> str:
        """' B12.345 C-90.000' - the rotary words of a move (empty on a 3-axis machine); letters/order from profile.rotary_axes."""
        if not profile.include_rotary_axes:
            return ""
        val = {"A": s.a, "B": s.b, "C": s.c}
        return "".join(f" {l}{val[l]:.3f}" for l in profile.rotary_axes if l in val)

    def _plan_moves(self, states: List[MachineStateVector], profile: MinimalPrintProfile) -> List[Dict[str, Any]]:
        """Turns machine states into template-ready moves (all decisions are made here, not in Jinja)."""
        r = profile.filament_diameter / 2.0
        area = math.pi * (r ** 2)

        # highest Z of every layer: a travel hops above the highest point of the layer it runs in
        layer_top: Dict[int, float] = {}
        for st in states:
            layer_top[st.layer_index] = max(layer_top.get(st.layer_index, st.z), st.z)

        moves: List[Dict[str, Any]] = []
        prev: Optional[MachineStateVector] = None
        last_layer: Optional[int] = None
        last_type: Optional[str] = None

        for s in states:
            move: Dict[str, Any] = {
                "x": s.x, "y": s.y, "z": s.z, "b": s.b, "c": s.c, "a": s.a,
                "rot": self._rot_words(s, profile),
                "feedrate": s.feedrate,
                "layer_marker": s.layer_index if s.layer_index != last_layer else None,
                "type_marker": None,
                "kind": "travel" if s.is_travel_move else "extrude",
            }
            if move["layer_marker"] is not None:
                last_layer = s.layer_index
                last_type = None                      # re-announce the feature in every layer

            if s.is_travel_move:
                move.update(self._plan_travel(prev, s, profile, layer_top))
            else:
                volume = s.extrusion_volume
                if profile.thickness_model == "nozzle_axis":
                    # thickness along the real nozzle axis (guarded against grazing angles)
                    volume /= max(s.tool_alignment, MIN_TOOL_ALIGNMENT)
                move["e"] = volume / area
                if s.feature_type != last_type:
                    move["type_marker"] = s.feature_type
                    last_type = s.feature_type

            moves.append(move)
            prev = s
        return moves

    @staticmethod
    def _plan_travel(prev: Optional[MachineStateVector], s: MachineStateVector,
                     profile: MinimalPrintProfile, layer_top: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
        hop = profile.z_hop > 0.0
        layer_top = layer_top or {}

        def lift(*states):
            """Z-hop height: the highest point of the layer(s) involved plus the hop (never lower than the end points)."""
            top = max(max(layer_top.get(st.layer_index, st.z), st.z) for st in states)
            return top + profile.z_hop
        if prev is None:
            # First move of the print: nothing to retract; lift first so the nozzle never
            # drags across the bed on the way to the start point.
            return {"retract": False, "prime": False, "hop": hop, "lift_z": lift(s)}

        distance = math.dist((prev.x, prev.y, prev.z), (s.x, s.y, s.z))
        if distance < profile.min_retract_travel:
            # short: never retract; hop only if requested
            return {"retract": False, "prime": False, "hop": hop and profile.hop_short_travels,
                    "lift_z": lift(prev, s)}

        retract = profile.retract_distance > 0.0
        return {
            "retract": retract,
            # the filament pulled back before the travel is pushed back after it, right before printing
            "prime": retract and profile.unretract_distance > 0.0,
            "hop": hop,
            # lift above the HIGHEST point of the layer (both layers at a layer change), not just the end points
            "lift_z": lift(prev, s),
        }
