"""Qt-free previews of function files and machine files."""
import math
import os

import numpy as np
import pytest

from src.stage1_slicer.function_file import FunctionFileError, FunctionSpec, load_function_file
from src.stage1_slicer.function_preview import height_map
from src.stage3_kinematics.machine_file import load_machine_file
from src.stage3_kinematics.machine_preview import outline, probe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUNCS = os.path.join(ROOT, "config", "functions")
MACHINES = os.path.join(ROOT, "config", "machines")
AREA = (-10.0, 10.0, -10.0, 10.0)


def test_planar_and_tilt_height_maps():
    flat = height_map(load_function_file(os.path.join(FUNCS, "planar.json")), bounds=AREA, n=11)
    assert flat.zmin == flat.zmax == 0.0
    tilt = load_function_file(os.path.join(FUNCS, "progressive_tilt.json"))
    hm = height_map(tilt, {"end_tilt_deg": 20.0}, AREA, n=11)
    assert hm.z.shape == (11, 11)
    assert hm.zmin == pytest.approx(0.0, abs=1e-9)                      # pivot at the lowest-y edge
    assert hm.zmax == pytest.approx(20.0 * math.tan(math.radians(20.0)))
    assert np.allclose(hm.z[:, 0], hm.z[:, -1])                         # constant along x
    assert "20" in hm.label


def test_expression_height_map_uses_the_parameters_and_flips_the_sign():
    spec = load_function_file(os.path.join(FUNCS, "examples", "wave.json"))
    a = height_map(spec, {"a": 1.0, "b": 5.0}, AREA, n=41)
    b = height_map(spec, {"a": 2.0, "b": 5.0}, AREA, n=41)
    assert b.zmax == pytest.approx(2 * a.zmax) and a.zmin == pytest.approx(-a.zmax, abs=0.05)
    gx = np.meshgrid(a.xs, a.ys)[0]
    assert np.allclose(a.z, -1.0 * np.sin(gx / 5.0))


def test_conical_and_grid_height_maps(tmp_path):
    import json
    cone = FunctionSpec.from_dict({"format": "cam-function", "version": 1, "name": "C", "type": "conical",
                                   "parameters": {"cone_angle_deg": {"default": 10}}})
    hm = height_map(cone, None, AREA, n=21)
    assert hm.zmax == pytest.approx(0.0) and hm.zmin == pytest.approx(-math.hypot(10, 10) * math.tan(math.radians(10)))
    (tmp_path / "f.csv").write_text("x,y,f\n" + "\n".join(f"{x},{y},{0.1 * x}" for x in (-10, 0, 10) for y in (-10, 0, 10)))
    (tmp_path / "g.json").write_text(json.dumps({"format": "cam-function", "version": 1, "name": "G", "type": "grid",
                                                 "data_file": "f.csv", "parameters": {}}))
    g = height_map(load_function_file(str(tmp_path / "g.json")), None, AREA, n=5)
    assert g.zmin == pytest.approx(-1.0) and g.zmax == pytest.approx(1.0)


def test_height_map_errors():
    spec = load_function_file(os.path.join(FUNCS, "progressive_tilt.json"))
    with pytest.raises(FunctionFileError):
        height_map(spec, None, (5, 5, 0, 1))
    with pytest.raises(FunctionFileError, match="above the maximum"):
        height_map(spec, {"end_tilt_deg": 99})


def test_machine_outline_for_both_shipped_machines():
    m3 = load_machine_file(os.path.join(MACHINES, "cartesian_3axis.json"))
    o3 = outline(m3)
    assert o3["fan"] == [] and o3["volume"]["z"] == [0, 250]
    assert any("Vertical nozzle" in s for s in o3["summary"])
    m5 = load_machine_file(os.path.join(MACHINES, "trunnion_5axis.json"))
    o5 = outline(m5, {"b_axis_min": -45, "b_axis_max": 60})
    assert [round(b) for b, _, _ in o5["fan"]] == [-45, 0, 60]
    b, dx, dz = o5["fan"][-1]
    assert dx**2 + dz**2 == pytest.approx(1.0)
    assert any("endless" in s for s in o5["summary"])


def test_probe_reports_solved_angles_and_reachability():
    m3 = load_machine_file(os.path.join(MACHINES, "cartesian_3axis.json"))
    p = probe(m3, None, tilt_deg=30)
    assert p["b"] == 0 and p["alignment"] == pytest.approx(math.cos(math.radians(30)))
    assert not p["reached"] and probe(m3, None, tilt_deg=10)["reached"]
    m5 = load_machine_file(os.path.join(MACHINES, "trunnion_5axis.json"))
    p = probe(m5, None, tilt_deg=30, azimuth_deg=90)
    assert p["b"] == pytest.approx(30.0, abs=1e-6) and p["reached"] and p["angle_error_deg"] == pytest.approx(0, abs=1e-3)
    p = probe(m5, {"b_axis_max": 10.0}, tilt_deg=30, azimuth_deg=0)
    assert not p["reached"] and p["angle_error_deg"] == pytest.approx(20.0, abs=1e-3)
