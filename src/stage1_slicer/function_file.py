"""
Function files: the mathematical description of a slicing deformation as an editable JSON file.

    {
      "format": "cam-function", "version": 1,
      "name": "Progressive tilt", "description": "...",
      "type": "progressive_tilt",            # planar | progressive_tilt | conical | expression | grid
      "expression": "a * np.sin(x / b)",     # type "expression" only: f(x, y)
      "data_file": "field.csv",              # type "grid" only (path relative to this file)
      "parameters": { "a": {"label": "Amplitude", "unit": "mm", "default": 0.5, "min": 0, "max": 5} }
    }

Nothing is predefined: every parameter a function needs must be declared in "parameters" (the GUI
shows one named input field per declared parameter) and a missing one is an error. Expressions use
the declared names like variables (validated by the AST whitelist in safe_expr.py).
Coordinates: x, y are those of the part as placed on the bed; layers follow z = z0 - f(x, y).
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.common.param_spec import ParamSpec, SpecError, parse_parameters, resolve_values
from src.stage1_slicer.function_import import load_data_file
from src.stage1_slicer.safe_expr import compile_expression, expression_names

FORMAT = "cam-function"
VERSION = 1

# type -> (strategy name, parameters the type NEEDS, optional parameters it understands)
BUILTIN_TYPES = {
    "planar": ("planar", (), ()),
    "progressive_tilt": ("progressive_tilt", ("start_tilt_deg", "end_tilt_deg"), ("pivot_y",)),
    "conical": ("conical", ("cone_angle_deg",), ()),
}
TYPES = tuple(BUILTIN_TYPES) + ("expression", "grid")


class FunctionFileError(SpecError):
    pass


@dataclass
class FunctionSpec:
    name: str
    type: str
    description: str = ""
    expression: Optional[str] = None
    data_file: Optional[str] = None
    parameters: Dict[str, ParamSpec] = field(default_factory=dict)
    base_dir: str = ""                      # where a relative data_file is looked up
    source: str = ""                        # path the spec was loaded from

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
        known = {"format", "version", "name", "description", "type", "expression", "data_file", "parameters"}
        extra = set(d) - known
        if extra:
            raise FunctionFileError(f"{where}unknown field(s) {sorted(extra)}")
        name = d.get("name")
        if not isinstance(name, str) or not name.strip():
            raise FunctionFileError(f'{where}"name" is missing')
        typ = d.get("type")
        if typ not in TYPES:
            raise FunctionFileError(f'{where}"type" must be one of {list(TYPES)}, got {typ!r}')
        spec = cls(name=name.strip(), type=typ, description=str(d.get("description", "")),
                   expression=d.get("expression"), data_file=d.get("data_file"),
                   parameters=parse_parameters(d.get("parameters"), where), base_dir=base_dir, source=source)
        spec._check(where)
        return spec

    def _check(self, where: str) -> None:
        declared = set(self.parameters)
        if self.type in BUILTIN_TYPES:
            _, needed, optional = BUILTIN_TYPES[self.type]
            for n in needed:
                if n not in declared:
                    raise FunctionFileError(f"{where}type '{self.type}' needs the parameter '{n}' - it is not declared in \"parameters\"")
            unknown = declared - set(needed) - set(optional)
            if unknown:
                raise FunctionFileError(f"{where}type '{self.type}' does not use parameter(s) {sorted(unknown)}")
            if self.expression is not None or self.data_file is not None:
                raise FunctionFileError(f"{where}type '{self.type}' takes no \"expression\" / \"data_file\"")
            for n in needed + optional:
                if n in declared and self.parameters[n].type not in ("number", "int"):
                    raise FunctionFileError(f"{where}parameter '{n}' must be numeric")
        elif self.type == "expression":
            if not isinstance(self.expression, str) or not self.expression.strip():
                raise FunctionFileError(f'{where}type "expression" needs an "expression"')
            if self.data_file is not None:
                raise FunctionFileError(f'{where}type "expression" takes no "data_file"')
            for n, p in self.parameters.items():
                if p.type not in ("number", "int"):
                    raise FunctionFileError(f"{where}parameter '{n}' of an expression must be numeric")
            try:
                used = expression_names(self.expression)
                missing = used - declared
                if missing:
                    raise FunctionFileError(
                        f"{where}the expression uses {sorted(missing)} but they are not declared in \"parameters\"")
                compile_expression(self.expression, declared)
            except FunctionFileError:
                raise
            except (ValueError, SyntaxError) as e:
                raise FunctionFileError(f"{where}{e}") from e
        else:  # grid
            if not isinstance(self.data_file, str) or not self.data_file.strip():
                raise FunctionFileError(f'{where}type "grid" needs a "data_file"')
            if self.expression is not None:
                raise FunctionFileError(f'{where}type "grid" takes no "expression"')
            if self.parameters:
                raise FunctionFileError(f'{where}type "grid" has no parameters')

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
        if self.type in BUILTIN_TYPES:
            return {"strategy": BUILTIN_TYPES[self.type][0], **v}
        if self.type == "expression":
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
        d["type"] = self.type
        if self.expression is not None:
            d["expression"] = self.expression
        if self.data_file is not None:
            d["data_file"] = self.data_file
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
