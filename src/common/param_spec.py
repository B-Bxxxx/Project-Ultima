"""
Named, editable parameters of function files and machine files.

A parameter is declared in JSON as
    "name": {"label": "Start tilt", "unit": "deg", "default": 0, "min": -45, "max": 45, "step": 1}
`type` is "number" (default), "int", "bool" or "choice" (then "choices": [...] is required).
`default` is mandatory. The GUI builds one input field per declared parameter.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class SpecError(ValueError):
    """A function/machine file is invalid (message names the file element)."""


@dataclass
class ParamSpec:
    name: str
    default: Any
    label: str = ""
    unit: str = ""
    type: str = "number"
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    choices: List[Any] = field(default_factory=list)
    description: str = ""

    TYPES = ("number", "int", "bool", "choice")

    @classmethod
    def from_dict(cls, name: str, d: Any, where: str = "") -> "ParamSpec":
        w = f"{where}parameter '{name}'"
        if not isinstance(d, dict):
            raise SpecError(f"{w}: must be an object like {{\"default\": 0}}")
        known = {"label", "unit", "type", "default", "min", "max", "step", "choices", "description"}
        extra = set(d) - known
        if extra:
            raise SpecError(f"{w}: unknown field(s) {sorted(extra)}")
        if "default" not in d:
            raise SpecError(f"{w}: 'default' is missing")
        typ = d.get("type", "number")
        if typ not in cls.TYPES:
            raise SpecError(f"{w}: type must be one of {cls.TYPES}")
        spec = cls(name=name, default=d["default"], label=str(d.get("label", name)), unit=str(d.get("unit", "")),
                   type=typ, min=d.get("min"), max=d.get("max"), step=d.get("step"),
                   choices=list(d.get("choices", [])), description=str(d.get("description", "")))
        for k in ("min", "max", "step"):
            v = getattr(spec, k)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
                raise SpecError(f"{w}: '{k}' must be a number")
        if spec.min is not None and spec.max is not None and spec.min > spec.max:
            raise SpecError(f"{w}: min is greater than max")
        if spec.step is not None and spec.step <= 0:
            raise SpecError(f"{w}: step must be positive")
        if typ == "choice" and not spec.choices:
            raise SpecError(f"{w}: a 'choice' parameter needs a non-empty 'choices' list")
        spec.default = spec.validate(spec.default, where=where)
        return spec

    def validate(self, value: Any, where: str = "") -> Any:
        w = f"{where}parameter '{self.name}'"
        if self.type == "bool":
            if not isinstance(value, bool):
                raise SpecError(f"{w}: expected true/false, got {value!r}")
            return value
        if self.type == "choice":
            if value not in self.choices:
                raise SpecError(f"{w}: {value!r} is not one of {self.choices}")
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SpecError(f"{w}: expected a number, got {value!r}")
        if self.type == "int":
            if float(value) != int(value):
                raise SpecError(f"{w}: expected an integer, got {value!r}")
            value = int(value)
        else:
            value = float(value)
        if value != value or value in (float("inf"), float("-inf")):
            raise SpecError(f"{w}: value must be finite")
        if self.min is not None and value < self.min:
            raise SpecError(f"{w}: {value} is below the minimum {self.min}")
        if self.max is not None and value > self.max:
            raise SpecError(f"{w}: {value} is above the maximum {self.max}")
        return value

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"label": self.label}
        if self.unit:
            d["unit"] = self.unit
        if self.type != "number":
            d["type"] = self.type
        d["default"] = self.default
        for k in ("min", "max", "step"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        if self.choices:
            d["choices"] = self.choices
        if self.description:
            d["description"] = self.description
        return d


def parse_parameters(raw: Any, where: str = "") -> Dict[str, ParamSpec]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SpecError(f"{where}'parameters' must be an object")
    return {name: ParamSpec.from_dict(name, d, where) for name, d in raw.items()}


def resolve_values(specs: Dict[str, ParamSpec], overrides: Optional[Dict[str, Any]] = None,
                   where: str = "") -> Dict[str, Any]:
    """Defaults + overrides, every value validated; an override for an undeclared parameter is an error."""
    overrides = overrides or {}
    unknown = set(overrides) - set(specs)
    if unknown:
        raise SpecError(f"{where}unknown parameter(s) {sorted(unknown)}")
    return {n: s.validate(overrides.get(n, s.default), where) for n, s in specs.items()}
