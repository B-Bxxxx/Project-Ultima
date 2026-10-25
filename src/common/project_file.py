"""
Project (settings) file: everything the GUI needs to restore a session, as one editable JSON file.

    {
      "format": "cam-project", "version": 1,
      "geometry":  {"type": "Hollow Cylinder", "radius": 15, "height": 20, "wall": 5, "width": 30, "depth": 30,
                    "cube_height": 20, "stl_path": "C:/parts/vase.stl"},
      "placement": {"rot_x": 0, ..., "center_xy": true, "drop_to_bed": true},
      "function":  {"name": "Wave", "source": "functions/wave.json", "values": {"amplitude": 1.5}},
      "machine":   {"name": "Generic 3-axis", "source": "machines/cartesian_3axis.json", "values": {}},
      "slicer":    {"layer_height": 0.4, "transition_height": 2.0, "transition_mode": "linear", ...},
      "profile":   {"nozzle_diameter": 0.4, "num_perimeters": 2, ...}
    }

The function and machine are stored by name (+ the path of their file, relative to the project file when it
lies below it) and their current parameter values; the STL by path only. Unknown fields are not an error
(a newer file opened by an older version): they are reported back so the GUI can tell the user.
"""
import json
import os
from typing import Any, Dict, List, Tuple

FORMAT = "cam-project"
VERSION = 1
SECTIONS = ("geometry", "placement", "function", "machine", "slicer", "profile")


class ProjectFileError(ValueError):
    pass


def validate(d: Any, where: str = "") -> Dict[str, Any]:
    prefix = f"{where}: " if where else ""
    if not isinstance(d, dict):
        raise ProjectFileError(f"{prefix}the file must contain a JSON object")
    if d.get("format") != FORMAT:
        raise ProjectFileError(f'{prefix}"format" must be "{FORMAT}" (this is not a project file)')
    if d.get("version") != VERSION:
        raise ProjectFileError(f'{prefix}unsupported "version" {d.get("version")!r} (this program reads version {VERSION})')
    for name in SECTIONS:
        if name in d and not isinstance(d[name], dict):
            raise ProjectFileError(f'{prefix}"{name}" must be an object')
    for name in ("function", "machine"):
        sec = d.get(name, {})
        if "values" in sec and not isinstance(sec["values"], dict):
            raise ProjectFileError(f'{prefix}"{name}.values" must be an object')
    return d


def save_project(d: Dict[str, Any], path: str) -> None:
    out = {"format": FORMAT, "version": VERSION}
    out.update({k: v for k, v in d.items() if k not in ("format", "version")})
    for name in ("function", "machine"):                  # file references relative to the project file when below it
        sec = out.get(name)
        if isinstance(sec, dict) and sec.get("source"):
            sec["source"] = _relative(sec["source"], os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")


def load_project(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except json.JSONDecodeError as e:
        raise ProjectFileError(f"{os.path.basename(path)}: invalid JSON: {e}") from e
    except OSError as e:
        raise ProjectFileError(f"cannot read {path}: {e}") from e
    validate(d, os.path.basename(path))
    base = os.path.dirname(os.path.abspath(path))
    for name in ("function", "machine"):
        sec = d.get(name)
        if isinstance(sec, dict) and sec.get("source") and not os.path.isabs(sec["source"]):
            sec["source"] = os.path.normpath(os.path.join(base, sec["source"]))
    return d


def _relative(source: str, base: str) -> str:
    try:
        rel = os.path.relpath(source, base)
    except ValueError:                                    # another drive (Windows)
        return source
    return source if rel.startswith("..") else rel.replace("\\", "/")


def unknown_keys(section: Dict[str, Any], known) -> List[str]:
    return sorted(set(section) - set(known))


def split_known(section: Dict[str, Any], known) -> Tuple[Dict[str, Any], List[str]]:
    return {k: v for k, v in section.items() if k in known}, unknown_keys(section, known)
