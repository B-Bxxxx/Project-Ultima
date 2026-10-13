"""JSON function files and machine files: parsing, strictness, round trips, shipped templates."""
import copy
import json
import math
import os

import numpy as np
import pytest
import trimesh

from src.common.param_spec import ParamSpec, SpecError, resolve_values
from src.common.schemas import CLDataWaypoint, CLDataTrajectory, MachineConfig, MinimalPrintProfile
from src.stage1_slicer.function_file import (
    FunctionFileError, FunctionSpec, load_function_dir, load_function_file, save_function_file,
)
from src.stage1_slicer.plugins.field_slicer import UniversalFieldSlicerPlugin
from src.stage1_slicer.safe_expr import compile_expression
from src.stage2_toolpath.contour_generator import StandardToolpathGenerator
from src.stage3_kinematics.machine_file import (
    MachineFileError, MachineSpec, load_machine_dir, load_machine_file, save_machine_file,
)
from src.stage3_kinematics.registry import KinematicsRegistry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUNCS = os.path.join(ROOT, "config", "functions")
MACHINES = os.path.join(ROOT, "config", "machines")


def _fn(**kw):
    d = {"format": "cam-function", "version": 1, "name": "T", "type": "expression",
         "expression": "a * x", "parameters": {"a": {"default": 0.1}}}
    d.update(kw)
    return d


# ------------------------------------------------------------------ ParamSpec
def test_param_spec_requires_default_and_validates_range_and_type():
    with pytest.raises(SpecError, match="'default' is missing"):
        ParamSpec.from_dict("a", {"label": "A"})
    p = ParamSpec.from_dict("a", {"default": 1, "min": 0, "max": 5})
    assert p.validate(5) == 5.0
    for bad in (-1, 6, "x", True, float("nan")):
        with pytest.raises(SpecError):
            p.validate(bad)
    with pytest.raises(SpecError, match="above the maximum"):
        ParamSpec.from_dict("a", {"default": 9, "max": 5})
    i = ParamSpec.from_dict("n", {"type": "int", "default": 3})
    assert i.validate(4.0) == 4
    with pytest.raises(SpecError):
        i.validate(4.5)
    c = ParamSpec.from_dict("c", {"type": "choice", "choices": ["a", "b"], "default": "a"})
    with pytest.raises(SpecError):
        c.validate("z")
    with pytest.raises(SpecError):
        ParamSpec.from_dict("c", {"type": "choice", "default": "a"})
    with pytest.raises(SpecError, match="unknown field"):
        ParamSpec.from_dict("a", {"default": 1, "colour": "red"})


def test_resolve_values_applies_defaults_and_rejects_unknown_names():
    specs = {"a": ParamSpec.from_dict("a", {"default": 1}), "b": ParamSpec.from_dict("b", {"default": 2})}
    assert resolve_values(specs, {"b": 5}) == {"a": 1.0, "b": 5.0}
    with pytest.raises(SpecError, match="unknown parameter"):
        resolve_values(specs, {"c": 1})


# ------------------------------------------------------------------ function files
def test_shipped_function_templates_load_and_planar_plus_tilt_are_the_defaults():
    specs = load_function_dir(FUNCS)
    assert set(specs) == {"Planar", "Progressive tilt"} and not load_function_dir.errors
    tilt = specs["Progressive tilt"]
    assert set(tilt.parameters) == {"start_tilt_deg", "end_tilt_deg"}
    assert tilt.to_slicer_params({"end_tilt_deg": 12})["end_tilt_deg"] == 12.0
    example = load_function_file(os.path.join(FUNCS, "examples", "wave.json"))
    assert example.type == "expression"


def test_expression_function_uses_its_named_parameters_in_the_slicer():
    spec = FunctionSpec.from_dict(_fn())
    params = spec.to_slicer_params({"a": 0.05})
    assert params == {"strategy": "custom_expr", "expression": "a * x", "expression_params": {"a": 0.05}}
    mesh = trimesh.creation.box(extents=(20, 20, 6))
    mesh.apply_translation((0, 0, 3))
    p = {"layer_height": 1.0, "transition_height": 3.0, **params}
    model = UniversalFieldSlicerPlugin().slice(mesh, p)
    json.dumps(model.metadata)                                    # parameters travel in the metadata
    prof = MinimalPrintProfile(layer_height=1.0, nozzle_diameter=0.4, num_perimeters=1, infill_density=0.2,
                               top_solid_layers=1, bottom_solid_layers=1)
    traj = StandardToolpathGenerator().generate_toolpath(model, prof)
    top = [w for w in traj.waypoints if w.layer_index == 4 and not w.is_travel_move]
    slope = np.polyfit([w.x for w in top], [w.z for w in top], 1)[0]
    assert slope == pytest.approx(-0.05, abs=0.005)
    # a different value changes the geometry
    p2 = {"layer_height": 1.0, "transition_height": 3.0, **spec.to_slicer_params({"a": 0.0})}
    flat = StandardToolpathGenerator().generate_toolpath(UniversalFieldSlicerPlugin().slice(mesh, p2), prof)
    assert max(w.z for w in flat.waypoints if w.layer_index == 4) == pytest.approx(5.0)


@pytest.mark.parametrize("mutation,message", [
    (lambda d: d.update(format="nope"), "format"),
    (lambda d: d.update(version=2), "version"),
    (lambda d: d.pop("name"), "name"),
    (lambda d: d.update(type="spiral"), "type"),
    (lambda d: d.update(colour="red"), "unknown field"),
    (lambda d: d.update(expression="a * x + b"), "'b'"),                    # undeclared parameter used
    (lambda d: d.update(expression="__import__('os')"), "unknown name|not allowed|Unsafe"),
    (lambda d: d.update(expression=None), "needs an \"expression\""),
    (lambda d: d.update(data_file="x.csv"), "takes no"),
    (lambda d: d["parameters"].update(x={"default": 1}), "reserved"),
    (lambda d: d["parameters"].update(a={"label": "A"}), "'default' is missing"),
])
def test_invalid_function_files_are_rejected_with_a_clear_message(mutation, message):
    d = copy.deepcopy(_fn())
    mutation(d)
    with pytest.raises(FunctionFileError, match=message):
        FunctionSpec.from_dict(d)


def test_builtin_types_need_their_parameters_and_reject_foreign_ones():
    base = {"format": "cam-function", "version": 1, "name": "T", "type": "progressive_tilt"}
    with pytest.raises(FunctionFileError, match="end_tilt_deg"):
        FunctionSpec.from_dict({**base, "parameters": {"start_tilt_deg": {"default": 0}}})
    with pytest.raises(FunctionFileError, match="does not use"):
        FunctionSpec.from_dict({**base, "parameters": {"start_tilt_deg": {"default": 0}, "end_tilt_deg": {"default": 5},
                                                       "zoom": {"default": 1}}})
    ok = FunctionSpec.from_dict({**base, "parameters": {"start_tilt_deg": {"default": 0}, "end_tilt_deg": {"default": 5},
                                                        "pivot_y": {"default": -10}}})
    assert ok.to_slicer_params()["pivot_y"] == -10.0
    with pytest.raises(FunctionFileError, match="not declared|needs"):
        FunctionSpec.from_dict({**base, "type": "conical", "parameters": {}})
    with pytest.raises(FunctionFileError):
        FunctionSpec.from_dict({**base, "type": "planar", "parameters": {"a": {"default": 1}}})


def test_values_outside_the_declared_range_are_an_error():
    spec = FunctionSpec.from_dict(_fn(parameters={"a": {"default": 0.1, "min": 0, "max": 1}}))
    with pytest.raises(FunctionFileError, match="above the maximum"):
        spec.to_slicer_params({"a": 2})
    with pytest.raises(FunctionFileError, match="unknown parameter"):
        spec.to_slicer_params({"b": 2})


def test_function_file_round_trip_and_saving_new_defaults(tmp_path):
    spec = load_function_file(os.path.join(FUNCS, "examples", "wave.json"))
    out = tmp_path / "mine.json"
    save_function_file(spec, str(out), {"a": 1.5, "b": 7.0})
    again = load_function_file(str(out))
    assert again.defaults() == {"a": 1.5, "b": 7.0}
    assert again.expression == spec.expression and again.parameters["a"].max == spec.parameters["a"].max
    with pytest.raises(FunctionFileError):
        save_function_file(spec, str(out), {"a": 999})


def test_grid_function_reads_its_data_file_relative_to_the_function_file(tmp_path):
    (tmp_path / "f.csv").write_text("x,y,f\n" + "\n".join(f"{x},{y},{0.05 * x}" for x in range(-20, 21, 10) for y in range(-20, 21, 10)))
    (tmp_path / "g.json").write_text(json.dumps({"format": "cam-function", "version": 1, "name": "G", "type": "grid",
                                                 "data_file": "f.csv", "parameters": {}}))
    spec = load_function_file(str(tmp_path / "g.json"))
    params = spec.to_slicer_params()
    assert params["strategy"] == "grid_field" and len(params["grid_x"]) == 5
    (tmp_path / "bad.json").write_text(json.dumps({"format": "cam-function", "version": 1, "name": "B", "type": "grid",
                                                   "data_file": "missing.csv"}))
    with pytest.raises(FunctionFileError, match="data file"):
        load_function_file(str(tmp_path / "bad.json")).to_slicer_params()


def test_invalid_json_and_missing_file_are_reported(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{oops")
    with pytest.raises(FunctionFileError, match="invalid JSON"):
        load_function_file(str(p))
    with pytest.raises(FunctionFileError, match="cannot read"):
        load_function_file(str(tmp_path / "nothing.json"))
    (tmp_path / "ok.json").write_text(json.dumps(_fn()))
    specs = load_function_dir(str(tmp_path))                     # bad files are skipped and reported
    assert list(specs) == ["T"] and "x.json" in load_function_dir.errors


# ------------------------------------------------------------------ machine files
def _m(**kw):
    d = {"format": "cam-machine", "version": 1, "name": "M", "kinematics": "planar_3axis_xyz",
         "axes": ["X", "Y", "Z"], "parameters": {}}
    d.update(kw)
    return d


def test_shipped_machines_load_and_describe_their_axes():
    specs = load_machine_dir(MACHINES)
    assert len(specs) == 2 and not load_machine_dir.errors
    m3 = specs["Generic 3-axis Cartesian"]
    m5 = specs["Generic 5-axis trunnion table (XYZBC)"]
    assert m3.axes == ["X", "Y", "Z"] and not m3.has_rotary_axes
    assert m5.has_rotary_axes and m5.kinematics == "trunnion_table_xyzbc"
    cfg = m5.to_machine_config({"b_axis_max": 60})
    assert isinstance(cfg, MachineConfig) and cfg.b_axis_max == 60 and cfg.b_axis_min == -90
    assert m3.to_machine_config().kinematic_chain == "planar_3axis_xyz"
    for spec in specs.values():                                  # kinematics are real solvers
        KinematicsRegistry.get_solver(spec.kinematics)


@pytest.mark.parametrize("mutation,message", [
    (lambda d: d.update(kinematics="warp_drive"), "not a registered solver"),
    (lambda d: d.update(axes=["X", "Y"]), "X, Y and Z"),
    (lambda d: d.update(axes=["X", "Y", "Z", "Q"]), "axes"),
    (lambda d: d.update(axes=["X", "Y", "Z", "B"]), "both B and C"),
    (lambda d: d.update(axes=["X", "Y", "Z", "B", "C"]), "3-axis solver"),
    (lambda d: d.update(kinematics="trunnion_table_xyzbc"), "needs \"B\" and \"C\""),
    (lambda d: d.update(parameters={"warp": {"default": 1}}), "unknown machine parameter"),
    (lambda d: d.update(build_volume={"x": [10, 0]}), "build_volume"),
    (lambda d: d.update(extra=1), "unknown field"),
])
def test_invalid_machine_files_are_rejected(mutation, message):
    d = _m()
    mutation(d)
    with pytest.raises(MachineFileError, match=message):
        MachineSpec.from_dict(d)


def test_a_rotary_machine_must_declare_its_axis_limits():
    d = _m(kinematics="trunnion_table_xyzbc", axes=["X", "Y", "Z", "B", "C"],
           parameters={"b_axis_min": {"default": -90}, "b_axis_max": {"default": 90}, "c_axis_min": {"default": -360}})
    with pytest.raises(MachineFileError, match="c_axis_max"):
        MachineSpec.from_dict(d)


def test_machine_round_trip(tmp_path):
    spec = load_machine_file(os.path.join(MACHINES, "trunnion_5axis.json"))
    out = tmp_path / "m.json"
    save_machine_file(spec, str(out), {"b_axis_max": 45})
    again = load_machine_file(str(out))
    assert again.defaults()["b_axis_max"] == 45 and again.axes == spec.axes and again.name == spec.name
    with pytest.raises(MachineFileError, match="above the maximum"):
        save_machine_file(spec, str(out), {"b_axis_max": 999})


# ------------------------------------------------------------------ tool alignment (Stage 3)
def _traj(i, j, k):
    wp = CLDataWaypoint(x=0, y=0, z=1, i=i, j=j, k=k, extrusion_volume=1.0, feedrate=1000.0, feature_type="outer_wall",
                        is_travel_move=False, layer_index=0)
    return CLDataTrajectory(waypoints=[wp])


def test_tool_alignment_of_the_three_solvers():
    n = (math.sin(math.radians(30)), 0.0, math.cos(math.radians(30)))      # surface tilted 30 degrees
    cfg3 = load_machine_file(os.path.join(MACHINES, "cartesian_3axis.json")).to_machine_config()
    cfg5 = load_machine_file(os.path.join(MACHINES, "trunnion_5axis.json")).to_machine_config()
    s3 = KinematicsRegistry.get_solver("planar_3axis_xyz")().solve(_traj(*n), cfg3).states[0]
    assert s3.tool_alignment == pytest.approx(math.cos(math.radians(30)))     # vertical nozzle on a tilted layer
    s5 = KinematicsRegistry.get_solver("trunnion_table_xyzbc")().solve(_traj(*n), cfg5).states[0]
    assert s5.tool_alignment == pytest.approx(1.0, abs=1e-6)                  # nozzle follows the normal
    limited = cfg5.model_copy(update={"b_axis_max": 10.0})
    s5l = KinematicsRegistry.get_solver("trunnion_table_xyzbc")().solve(_traj(*n), limited).states[0]
    assert s5l.b == pytest.approx(10.0)
    assert s5l.tool_alignment == pytest.approx(math.cos(math.radians(20)), abs=1e-6)   # 30 wanted, 10 reached
    sh = KinematicsRegistry.get_solver("swivel_head_xyzbc")().solve(_traj(*n), cfg5).states[0]
    assert sh.tool_alignment == pytest.approx(1.0, abs=1e-6)
