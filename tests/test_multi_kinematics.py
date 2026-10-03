import pytest
import os
import json
from src.pipeline_cli import run_stage1, run_stage2, run_stage3
from src.common.schemas import MachineTrajectory

@pytest.fixture(scope="module")
def setup_multi_kinematics():
    os.makedirs("tests/output", exist_ok=True)

    # 1. Configs
    slicer_cfg = {
        "strategy": "progressive_tilt",
        "layer_height": 5.0,
        "start_z": 0.0,
        "end_z": 5.0,
        "start_tilt_deg": 45.0,
        "end_tilt_deg": 45.0
    }
    with open("tests/output/test_mk_slicer.json", "w") as f:
        json.dump(slicer_cfg, f)

    run_stage1("universal_field_slicer", "tests/output/test_mk_slicer.json", "tests/output/test_mk_s1.json")
    run_stage2("tests/output/test_mk_s1.json", "config/print_profile.json", "tests/output/test_mk_s2.json")
    yield

def test_multi_kinematics_trunnion(setup_multi_kinematics):
    cfg = {
        "kinematic_topology": "trunnion_table_xyzbc",
        "kinematic_chain": "Generic",
        "pivot_offset_x": 0.0,
        "pivot_offset_y": 0.0,
        "pivot_offset_z": 100.0,
        "eccentricity_x": 0.0,
        "eccentricity_y": 0.0,
        "b_axis_min": -90.0,
        "b_axis_max": 90.0,
        "c_axis_min": -36000.0,
        "c_axis_max": 36000.0
    }
    with open("tests/output/test_mk_cfg_trun.json", "w") as f:
        json.dump(cfg, f)

    run_stage3("tests/output/test_mk_s2.json", "tests/output/test_mk_cfg_trun.json", "tests/output/test_mk_s3_trun.json")
    traj = MachineTrajectory.from_json_file("tests/output/test_mk_s3_trun.json")

    # Just verify it solved cleanly
    assert len(traj.states) > 0
    state = traj.states[-1]

    # It should have tilted 45 deg
    assert state.b == pytest.approx(45.0, abs=0.1)

def test_multi_kinematics_head(setup_multi_kinematics):
    cfg = {
        "kinematic_topology": "swivel_head_xyzbc",
        "kinematic_chain": "Generic",
        "pivot_offset_x": 0.0,
        "pivot_offset_y": 0.0,
        "pivot_offset_z": 50.0,
        "eccentricity_x": 0.0,
        "eccentricity_y": 0.0,
        "b_axis_min": -90.0,
        "b_axis_max": 90.0,
        "c_axis_min": -36000.0,
        "c_axis_max": 36000.0
    }
    with open("tests/output/test_mk_cfg_head.json", "w") as f:
        json.dump(cfg, f)

    run_stage3("tests/output/test_mk_s2.json", "tests/output/test_mk_cfg_head.json", "tests/output/test_mk_s3_head.json")
    traj = MachineTrajectory.from_json_file("tests/output/test_mk_s3_head.json")

    assert len(traj.states) > 0
    state = traj.states[-1]
    # Tool normal is defined negatively usually for head?
    # Regardless it should not be zero tilt
    assert state.b > 0.0
