"""
Machine files: the kinematic description of a machine as an editable JSON file.

    {
      "format": "cam-machine", "version": 1,
      "name": "Generic 3-axis Cartesian", "description": "...",
      "kinematics": "planar_3axis_xyz",        # a registered Stage 3 solver
      "axes": ["X", "Y", "Z"],                  # only these axes appear in the G-code
      "build_volume": {"x": [0, 220], "y": [0, 220], "z": [0, 250]},     # optional
      "parameters": { "max_3axis_tilt_deg": {"label": "...", "unit": "deg", "default": 25} }
    }

Parameters are MachineConfig fields; the GUI shows one named input field per declared parameter.
A 5-axis machine (axes X Y Z B C) must declare the B and C limits; a 3-axis machine must not use
a rotary kinematic. Anything missing or inconsistent is an error.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.common.param_spec import ParamSpec, SpecError, parse_parameters, resolve_values
from src.common.schemas import MachineConfig

# importing the solver modules registers them
from src.stage3_kinematics import head_solver, trunnion_solver  # noqa: F401
from src.stage3_kinematics.registry import KinematicsRegistry

FORMAT = "cam-machine"
VERSION = 1
PLANAR = "planar_3axis_xyz"
VALID_AXES = ("X", "Y", "Z", "B", "C")
CONFIG_FIELDS = ("b_axis_dir", "c_axis_dir", "c_axis_x", "c_axis_y", "c_axis_z", "pivot_offset_x", "pivot_offset_y", "pivot_offset_z", "eccentricity_x", "eccentricity_y",
                 "b_axis_min", "b_axis_max", "c_axis_min", "c_axis_max", "max_3axis_tilt_deg")
ROTARY_REQUIRED = ("b_axis_min", "b_axis_max", "c_axis_min", "c_axis_max")


class MachineFileError(SpecError):
    pass


@dataclass
class MachineSpec:
    name: str
    kinematics: str
    axes: list
    description: str = ""
    build_volume: Optional[Dict[str, list]] = None
    parameters: Dict[str, ParamSpec] = field(default_factory=dict)
    source: str = ""

    @property
    def has_rotary_axes(self) -> bool:
        return "B" in self.axes or "C" in self.axes

    def fit_warnings(self, waypoints) -> list:
        """
        Non-blocking check: does the printed part (brim included) fit the machine's build volume?
        Compares the SIZE of the toolpath's bounding box per axis with the size of the build volume
        (positions are not compared: where the part sits on the bed is the user's placement).
        Returns a list of messages; empty = fits, or the machine file gives no build volume.
        """
        if not self.build_volume:
            return []
        pts = [(w.x, w.y, w.z) for w in waypoints if not w.is_travel_move]
        if not pts:
            return []
        msgs = []
        for i, axis in enumerate("xyz"):
            lim = self.build_volume.get(axis)
            if not lim:
                continue
            vals = [p[i] for p in pts]
            size, room = max(vals) - min(vals), float(lim[1]) - float(lim[0])
            if size > room + 1e-6:
                msgs.append(f"the part is {size:.1f} mm in {axis.upper()}, the build volume only {room:.1f} mm")
        return msgs

    def range_warnings(self, states) -> list:
        """
        Non-blocking check of the MACHINE coordinates (Stage 3 output, travel moves included) against the
        linear axis range given in `build_volume` (e.g. X 0..220: the end stop is at 0, Klipper refuses to
        move below it). Returns one message per axis that leaves its range; empty = all inside.
        """
        if not self.build_volume or not states:
            return []
        msgs = []
        for axis in "xyz":
            lim = self.build_volume.get(axis)
            if not lim:
                continue
            lo, hi = float(lim[0]), float(lim[1])
            vals = [getattr(s, axis) for s in states]
            vmin, vmax = min(vals), max(vals)
            if vmin < lo - 1e-6 or vmax > hi + 1e-6:
                bad = next(s for s in states if getattr(s, axis) < lo - 1e-6 or getattr(s, axis) > hi + 1e-6)
                msgs.append(f"{axis.upper()} runs {vmin:.1f}..{vmax:.1f} mm, the machine range is {lo:g}..{hi:g} mm "
                            f"(first at layer {bad.layer_index}): move the part origin / table centre (c_axis_{axis}) or shrink the part")
        return msgs

    @classmethod
    def from_dict(cls, d: Any, source: str = "") -> "MachineSpec":
        where = f"{os.path.basename(source)}: " if source else ""
        try:
            return cls._from_dict(d, source, where)
        except MachineFileError:
            raise
        except SpecError as e:
            raise MachineFileError(str(e)) from e

    @classmethod
    def _from_dict(cls, d, source, where) -> "MachineSpec":
        if not isinstance(d, dict):
            raise MachineFileError(f"{where}the file must contain a JSON object")
        if d.get("format") != FORMAT:
            raise MachineFileError(f'{where}"format" must be "{FORMAT}"')
        if d.get("version") != VERSION:
            raise MachineFileError(f'{where}unsupported "version" {d.get("version")!r} (expected {VERSION})')
        extra = set(d) - {"format", "version", "name", "description", "kinematics", "axes", "build_volume", "parameters"}
        if extra:
            raise MachineFileError(f"{where}unknown field(s) {sorted(extra)}")
        name = d.get("name")
        if not isinstance(name, str) or not name.strip():
            raise MachineFileError(f'{where}"name" is missing')
        kin = d.get("kinematics")
        try:
            KinematicsRegistry.get_solver(kin)
        except Exception:
            raise MachineFileError(f'{where}"kinematics" {kin!r} is not a registered solver')
        axes = d.get("axes")
        if (not isinstance(axes, list) or any(a not in VALID_AXES for a in axes) or len(set(axes)) != len(axes)):
            raise MachineFileError(f'{where}"axes" must be a list of distinct axes from {list(VALID_AXES)}')
        if not {"X", "Y", "Z"} <= set(axes):
            raise MachineFileError(f'{where}"axes" must contain X, Y and Z')
        rotary = {"B", "C"} & set(axes)
        if rotary and rotary != {"B", "C"}:
            raise MachineFileError(f'{where}a rotary machine needs both B and C in "axes"')
        if rotary and kin == PLANAR:
            raise MachineFileError(f'{where}kinematics "{PLANAR}" is a 3-axis solver but "axes" lists B and C')
        if not rotary and kin != PLANAR:
            raise MachineFileError(f'{where}kinematics "{kin}" needs "B" and "C" in "axes"')
        params = parse_parameters(d.get("parameters"), where)
        unknown = set(params) - set(CONFIG_FIELDS)
        if unknown:
            raise MachineFileError(f"{where}unknown machine parameter(s) {sorted(unknown)}; allowed: {list(CONFIG_FIELDS)}")
        if rotary:
            for n in ROTARY_REQUIRED:
                if n not in params:
                    raise MachineFileError(f"{where}the rotary axes need the parameter '{n}' in \"parameters\"")
        bv = d.get("build_volume")
        if bv is not None:
            ok = (isinstance(bv, dict) and set(bv) <= {"x", "y", "z"} and all(
                isinstance(v, list) and len(v) == 2 and all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in v)
                and v[0] < v[1] for v in bv.values()))
            if not ok:
                raise MachineFileError(f'{where}"build_volume" must look like {{"x": [0, 220], "y": [0, 220], "z": [0, 250]}}')
        return cls(name=name.strip(), kinematics=kin, axes=list(axes), description=str(d.get("description", "")),
                   build_volume=bv, parameters=params, source=source)

    def defaults(self) -> Dict[str, Any]:
        return {n: p.default for n, p in self.parameters.items()}

    def to_machine_config(self, overrides: Optional[Dict[str, Any]] = None) -> MachineConfig:
        try:
            v = resolve_values(self.parameters, overrides, f"machine '{self.name}': ")
        except SpecError as e:
            raise MachineFileError(str(e)) from e
        for n in ROTARY_REQUIRED:                   # a 3-axis machine has no rotary limits
            v.setdefault(n, 0.0)
        return MachineConfig(kinematic_chain=self.kinematics, **v)

    def to_dict(self, values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            vals = resolve_values(self.parameters, values, "") if values is not None else self.defaults()
        except SpecError as e:
            raise MachineFileError(str(e)) from e
        d: Dict[str, Any] = {"format": FORMAT, "version": VERSION, "name": self.name}
        if self.description:
            d["description"] = self.description
        d["kinematics"] = self.kinematics
        d["axes"] = list(self.axes)
        if self.build_volume:
            d["build_volume"] = self.build_volume
        d["parameters"] = {n: ParamSpec(**{**p.__dict__, "default": vals[n]}).to_dict() for n, p in self.parameters.items()}
        return d


def load_machine_file(path: str) -> MachineSpec:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as e:
        raise MachineFileError(f"{os.path.basename(path)}: invalid JSON: {e}") from e
    except OSError as e:
        raise MachineFileError(f"cannot read {path}: {e}") from e
    return MachineSpec.from_dict(data, source=path)


def save_machine_file(spec: MachineSpec, path: str, values: Optional[Dict[str, Any]] = None) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec.to_dict(values), fh, indent=2)
        fh.write("\n")


def load_machine_dir(directory: str) -> Dict[str, MachineSpec]:
    specs, errors = {}, {}
    if os.path.isdir(directory):
        for fn in sorted(os.listdir(directory)):
            if fn.lower().endswith(".json"):
                try:
                    s = load_machine_file(os.path.join(directory, fn))
                    specs[s.name] = s
                except MachineFileError as e:
                    errors[fn] = str(e)
    load_machine_dir.errors = errors
    return specs
