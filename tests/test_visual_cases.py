import pytest
import os
import json
from src.pipeline_cli import run_all, visualize

@pytest.fixture(scope="module")
def setup_dirs():
    os.makedirs("tests/output", exist_ok=True)
    yield
    # Cleanup is optional, but we can leave the files for visual inspection

def test_visual_case_1_planar_cylinder(setup_dirs):
    config = {
        "radius": 15.0,
        "height": 2.0,
        "layer_height": 0.2,
        "segments_per_layer": 50,
        "pitch": 0.0,
        "tool_tilt_deg": 0.0
    }
    with open("tests/output/cfg_planar.json", "w") as f:
        json.dump(config, f)

    out_prefix = "tests/output/case1_planar"
    run_all("universal_math_slicer", "tests/output/cfg_planar.json", "config/machine_config.json", "config/print_profile.json", out_prefix)

    # Generate visualization
    visualize(f"{out_prefix}_s2.json", f"{out_prefix}_vis.html")

    assert os.path.exists(f"{out_prefix}_s1.json")
    assert os.path.exists(f"{out_prefix}_s2.json")
    assert os.path.exists(f"{out_prefix}_s3.json")
    assert os.path.exists(f"{out_prefix}.gcode")
    assert os.path.exists(f"{out_prefix}_vis.html")


def test_visual_case_2_helical_vase(setup_dirs):
    config = {
        "radius": 15.0,
        "height": 2.0,
        "layer_height": 0.2,
        "segments_per_layer": 50,
        "pitch": 0.2,
        "tool_tilt_deg": 0.0
    }
    with open("tests/output/cfg_helical.json", "w") as f:
        json.dump(config, f)

    # We need to temporarily set continuous_spiral in profile to True for this to be a true vase mode without travel moves
    with open("config/print_profile.json", "r") as f:
        prof = json.load(f)
    prof["continuous_spiral"] = True
    with open("tests/output/prof_vase.json", "w") as f:
        json.dump(prof, f)

    out_prefix = "tests/output/case2_helical"
    run_all("universal_math_slicer", "tests/output/cfg_helical.json", "config/machine_config.json", "tests/output/prof_vase.json", out_prefix)

    visualize(f"{out_prefix}_s2.json", f"{out_prefix}_vis.html")
    assert os.path.exists(f"{out_prefix}_vis.html")

def test_visual_case_3_conical_tilted(setup_dirs):
    config = {
        "radius": 15.0,
        "height": 2.0,
        "layer_height": 0.2,
        "segments_per_layer": 50,
        "pitch": 0.2,
        "tool_tilt_deg": 45.0
    }
    with open("tests/output/cfg_conical.json", "w") as f:
        json.dump(config, f)

    out_prefix = "tests/output/case3_conical"
    run_all("universal_math_slicer", "tests/output/cfg_conical.json", "config/machine_config.json", "tests/output/prof_vase.json", out_prefix)

    visualize(f"{out_prefix}_s2.json", f"{out_prefix}_vis.html")
    assert os.path.exists(f"{out_prefix}_vis.html")
