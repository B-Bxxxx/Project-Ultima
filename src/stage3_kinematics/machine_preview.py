"""Qt-free preview helpers for machine files: outline for drawing, a text summary and a solver probe."""
import math
from typing import Any, Dict, List, Optional

from src.common.schemas import CLDataTrajectory, CLDataWaypoint
from src.stage3_kinematics.machine_file import PLANAR, MachineSpec
from src.stage3_kinematics.registry import KinematicsRegistry

# direction of the nozzle axis relative to the part for a pure B move (C = 0), as the solvers define it
_TOOL_SIGN = {"trunnion_table_xyzbc": -1.0, "swivel_head_xyzbc": 1.0}


def outline(spec: MachineSpec, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    {"volume": {"x": [..], "y": [..], "z": [..]} | None,
     "fan": [(b_deg, dx, dz), ...]      # nozzle-axis direction at B min / 0 / B max (XZ plane, C = 0)
     "summary": [str, ...]}
    """
    cfg = spec.to_machine_config(overrides)
    lines: List[str] = [f"Kinematics: {spec.kinematics}", f"Axes in the G-code: {' '.join(spec.axes)}"]
    fan = []
    if spec.has_rotary_axes:
        sign = _TOOL_SIGN.get(spec.kinematics, 1.0)
        for b in sorted({cfg.b_axis_min, 0.0, cfg.b_axis_max}):
            fan.append((b, sign * math.sin(math.radians(b)), math.cos(math.radians(b))))
        lines.append(f"B range {cfg.b_axis_min:g} .. {cfg.b_axis_max:g} deg "
                     f"(max. tilt from vertical {max(abs(cfg.b_axis_min), abs(cfg.b_axis_max)):g} deg)")
        span = cfg.c_axis_max - cfg.c_axis_min
        lines.append(f"C range {cfg.c_axis_min:g} .. {cfg.c_axis_max:g} deg" + (" (endless rotation)" if span >= 720 else ""))
    else:
        lines.append(f"Vertical nozzle only; surfaces tilted more than {cfg.max_3axis_tilt_deg:g} deg raise a warning")
    if spec.build_volume:
        lines.append("Build volume: " + ", ".join(f"{k.upper()} {v[0]:g}..{v[1]:g}" for k, v in spec.build_volume.items()))
    return {"volume": spec.build_volume, "fan": fan, "summary": lines}


def probe(spec: MachineSpec, overrides: Optional[Dict[str, Any]] = None,
          tilt_deg: float = 0.0, azimuth_deg: float = 0.0) -> Dict[str, Any]:
    """Runs the real solver on one waypoint whose surface normal is tilted `tilt_deg` towards `azimuth_deg`."""
    cfg = spec.to_machine_config(overrides)
    t, a = math.radians(tilt_deg), math.radians(azimuth_deg)
    n = (math.sin(t) * math.cos(a), math.sin(t) * math.sin(a), math.cos(t))
    wp = CLDataWaypoint(x=0.0, y=0.0, z=1.0, i=n[0], j=n[1], k=n[2], extrusion_volume=1.0, feedrate=1000.0,
                        feature_type="outer_wall", is_travel_move=False, layer_index=0)
    state = KinematicsRegistry.get_solver(spec.kinematics)().solve(CLDataTrajectory(waypoints=[wp]), cfg).states[0]
    if spec.kinematics == PLANAR:
        reached = tilt_deg <= cfg.max_3axis_tilt_deg + 1e-9
    else:
        reached = state.tool_alignment > 1.0 - 1e-6
    return {"b": state.b, "c": state.c, "alignment": state.tool_alignment,
            "angle_error_deg": math.degrees(math.acos(max(-1.0, min(1.0, state.tool_alignment)))),
            "reached": bool(reached)}
