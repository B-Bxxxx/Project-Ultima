"""
Function files: the mathematical description of how the layers are formed, as an editable JSON file.
The file CONTAINS the maths - there is no hidden "type" that selects code.

    {
      "format": "cam-function", "version": 2,
      "name": "Wave", "description": "...",
      "expression": "a * np.sin(x / b)",     # layers follow z = z0 - f(x, y)   (x, y, r + parameters)
      "parameters": { "a": {"label": "Amplitude", "unit": "mm", "default": 0.5, "min": 0, "max": 5} }
    }

Exactly ONE of these defines the layer surfaces:
  * "expression"  : f(x, y) in mm, layers follow z = z0 - f(x, y). Flat layers: "0".
                    Cone: "r * tan(radians(cone_angle_deg))".
  * "data_file"   : f(x, y) from a regular grid (csv/txt/npy), path relative to this file.
  * "tilt_x_deg"  : every layer is a PLANE tilted about the X axis by this angle (degrees); the
                    expression may use "t" = layer index / (layers - 1), 0..1, e.g.
                    "start_tilt_deg + (end_tilt_deg - start_tilt_deg) * t".  The pivot line is
                    "pivot_y" (default "y_min" = lowest Y of the part; "y_min" is the only name
                    the program supplies besides "t").

Nothing is predefined: every parameter a function uses must be declared in "parameters" (the GUI
shows one named input field per declared parameter) and a missing one is an error. Expressions are
validated by the AST whitelist in safe_expr.py.
Coordinates: x, y are relative to the centre of the part's footprint (bounding-box centre), so the slice does not depend on where the part sits on the bed.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.common.param_spec import ParamSpec, SpecError, parse_parameters, resolve_values
from src.stage1_slicer.function_import import load_data_file
from src.stage1_slicer.safe_expr import compile_expression, expression_names

FORMAT = "cam-function"
VERSION = 2
TILT_VAR = "t"           # layer position 0..1 inside "tilt_x_deg"
PIVOT_VAR = "y_min"      # lowest Y of the part inside "pivot_y"
DEFAULT_PIVOT = PIVOT_VAR


class FunctionFileError(SpecError):
    pass


@dataclass
class FunctionSpec:
    name: str
    description: str = ""
    expression: Optional[str] = None        # f(x, y), layers follow z = z0 - f
    data_file: Optional[str] = None         # f(x, y) from a grid file
    tilt_x_deg: Optional[str] = None        # planar layers tilted about X, expression of t
    pivot_y: Optional[str] = None           # pivot line of the tilt (default "y_min")
    parameters: Dict[str, ParamSpec] = field(default_factory=dict)
    base_dir: str = ""                      # where a relative data_file is looked up
    source: str = ""                        # path the spec was loaded from

    @property
    def kind(self) -> str:
        return "tilt" if self.tilt_x_deg is not None else ("grid" if self.data_file is not None else "expression")

    # ----------------------------------------------------------------------------- build / check
    @classmethod
    def from_dict(cls, d: Any, base_dir: str = "", source: str = "") -> "FunctionSpec":
        where = f"{os.path.basename(source)}: " if source else ""
        try:
            return cls._from_dict(d, base_dir, source, where)
        except FunctionFileError:
            raise
        except SpecError as e:
            raise FunctionFileError(str(e)) from e

    @classmethod
    def _from_dict(cls, d, base_dir, source, where) -> "FunctionSpec":
        if not isinstance(d, dict):
            raise FunctionFileError(f"{where}the file must contain a JSON object")
        if d.get("format") != FORMAT:
            raise FunctionFileError(f'{where}"format" must be "{FORMAT}"')
        if d.get("version") != VERSION:
            raise FunctionFileError(f'{where}unsupported "version" {d.get("version")!r} (expected {VERSION})')
        if "type" in d:
            raise FunctionFileError(f'{where}"type" does not exist in version {VERSION}: write the maths '
                                    f'("expression", "data_file" or "tilt_x_deg") into the file')
        known = {"format", "version", "name", "description", "expression", "data_file", "tilt_x_deg", "pivot_y",
                 "parameters"}
        extra = set(d) - known
        if extra:
            raise FunctionFileError(f"{where}unknown field(s) {sorted(extra)}")
        name = d.get("name")
        if not isinstance(name, str) or not name.strip():
            raise FunctionFileError(f'{where}"name" is missing')
        spec = cls(name=name.strip(), description=str(d.get("description", "")),
                   expression=d.get("expression"), data_file=d.get("data_file"),
                   tilt_x_deg=d.get("tilt_x_deg"), pivot_y=d.get("pivot_y"),
                   parameters=parse_parameters(d.get("parameters"), where), base_dir=base_dir, source=source)
        spec._check(where)
        return spec

    def _check_expr(self, label: str, expr: Any, allowed_extra, where: str, scalar: bool = False) -> None:
        if not isinstance(expr, str) or not expr.strip():
            raise FunctionFileError(f'{where}"{label}" must be a non-empty expression')
        declared = set(self.parameters)
        try:
            if scalar:
                import ast
                bad = {n.id for n in ast.walk(ast.parse(expr.strip(), mode="eval")) if isinstance(n, ast.Name)} & {"x", "y", "r"}
                if bad:
                    raise FunctionFileError(f'{where}"{label}" is one number per layer: x, y, r are not available')
            missing = expression_names(expr) - declared - set(allowed_extra)
            if missing:
                raise FunctionFileError(
                    f"{where}\"{label}\" uses {sorted(missing)} but they are not declared in \"parameters\"")
            compile_expression(expr, declared | set(allowed_extra))
        except FunctionFileError:
            raise
        except (ValueError, SyntaxError) as e:
            raise FunctionFileError(f"{where}\"{label}\": {e}") from e

    def _check(self, where: str) -> None:
        given = [k for k in ("expression", "data_file", "tilt_x_deg") if getattr(self, k) is not None]
        if len(given) != 1:
            raise FunctionFileError(
                f'{where}exactly one of "expression", "data_file", "tilt_x_deg" is needed (found {given or "none"})')
        for n, p in self.parameters.items():
            if n in (TILT_VAR, PIVOT_VAR):
                raise FunctionFileError(f"{where}parameter name '{n}' is reserved")
            if self.kind != "grid" and p.type not in ("number", "int"):
                raise FunctionFileError(f"{where}parameter '{n}' must be numeric")
        if self.pivot_y is not None and self.kind != "tilt":
            raise FunctionFileError(f'{where}"pivot_y" belongs to "tilt_x_deg"')
        if self.kind == "expression":
            self._check_expr("expression", self.expression, (), where)
        elif self.kind == "tilt":
            self._check_expr("tilt_x_deg", self.tilt_x_deg, (TILT_VAR,), where, scalar=True)
            self._check_expr("pivot_y", self.pivot_y if self.pivot_y is not None else DEFAULT_PIVOT, (PIVOT_VAR,), where, scalar=True)
        else:
            if not isinstance(self.data_file, str) or not self.data_file.strip():
                raise FunctionFileError(f'{where}"data_file" must be a file name')
            if self.parameters:
                raise FunctionFileError(f'{where}a "data_file" function has no parameters')

    # ----------------------------------------------------------------------------------- use
    def defaults(self) -> Dict[str, Any]:
        return {n: p.default for n, p in self.parameters.items()}

    def values(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            return resolve_values(self.parameters, overrides, f"function '{self.name}': ")
        except SpecError as e:
            raise FunctionFileError(str(e)) from e

    def to_slicer_params(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Parameters for UniversalFieldSlicerPlugin.slice() (without layer_height etc.)."""
        v = self.values(overrides)
        if self.kind == "tilt":
            out = {"strategy": "progressive_tilt", "tilt_expression": self.tilt_x_deg, "expression_params": v}
            if self.pivot_y is not None:
                out["pivot_expression"] = self.pivot_y
            return out
        if self.kind == "expression":
            return {"strategy": "custom_expr", "expression": self.expression, "expression_params": v}
        path = self.data_file if os.path.isabs(self.data_file) else os.path.join(self.base_dir or ".", self.data_file)
        try:
            grid = load_data_file(path)
        except (ValueError, OSError) as e:
            raise FunctionFileError(f"function '{self.name}': data file {self.data_file!r}: {e}") from e
        return {"strategy": "grid_field", **grid.to_params()}

    def to_dict(self, values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """JSON-able description; `values` (optional) become the new defaults."""
        vals = self.values(values) if values is not None else self.defaults()
        params = {}
        for n, p in self.parameters.items():
            q = ParamSpec(**{**p.__dict__, "default": vals[n]})
            params[n] = q.to_dict()
        d: Dict[str, Any] = {"format": FORMAT, "version": VERSION, "name": self.name}
        if self.description:
            d["description"] = self.description
        for k in ("expression", "data_file", "tilt_x_deg", "pivot_y"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        d["parameters"] = params
        return d


def load_function_file(path: str) -> FunctionSpec:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as e:
        raise FunctionFileError(f"{os.path.basename(path)}: invalid JSON: {e}") from e
    except OSError as e:
        raise FunctionFileError(f"cannot read {path}: {e}") from e
    return FunctionSpec.from_dict(data, base_dir=os.path.dirname(os.path.abspath(path)), source=path)


def save_function_file(spec: FunctionSpec, path: str, values: Optional[Dict[str, Any]] = None) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec.to_dict(values), fh, indent=2)
        fh.write("\n")


def load_function_dir(directory: str) -> Dict[str, FunctionSpec]:
    """All valid function files of a folder (name -> spec). Invalid files are skipped and returned in `errors`."""
    specs, errors = {}, {}
    if os.path.isdir(directory):
        for fn in sorted(os.listdir(directory)):
            if fn.lower().endswith(".json"):
                try:
                    s = load_function_file(os.path.join(directory, fn))
                    specs[s.name] = s
                except FunctionFileError as e:
                    errors[fn] = str(e)
    load_function_dir.errors = errors
    return specs
