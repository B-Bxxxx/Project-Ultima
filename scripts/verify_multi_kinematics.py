import os
import json
import trimesh
from src.pipeline_cli import run_stage1, run_stage2, run_stage3
from src.common.visualizer import generate_stage3_comparison_dashboard
from src.common.schemas import MachineTrajectory

def main():
    print("Generating Test Toolpath for Multi-Kinematics Comparison...")

    os.makedirs("tests/output", exist_ok=True)

    # 1. We create a complex conical toolpath so all axes (XYZBC) are engaged
    # using the conical field slicer on a slightly taller cylinder for multiple layers
    mesh = trimesh.creation.cylinder(radius=15, height=10)
    mesh.apply_translation([0, 0, 5])

    slicer_cfg = {
        "strategy": "conical",
        "layer_height": 2.0,
        "cone_angle_deg": 30.0
    }
    with open("tests/output/mk_slicer_cfg.json", "w") as f:
        json.dump(slicer_cfg, f)

    run_stage1("universal_field_slicer", "tests/output/mk_slicer_cfg.json", "tests/output/mk_s1.json")
    run_stage2("tests/output/mk_s1.json", "config/print_profile.json", "tests/output/mk_s2.json")

    # 2. Define Machine Configs
    base_machine = {
        "kinematic_chain": "Generic",
        "pivot_offset_x": 0.0,
        "pivot_offset_y": -50.0, # 50mm pivot offset
        "pivot_offset_z": 100.0, # 100mm pivot offset
        "eccentricity_x": 0.0,
        "eccentricity_y": 0.0,
        "b_axis_min": -90.0,
        "b_axis_max": 90.0,
        "c_axis_min": -36000.0,
        "c_axis_max": 36000.0
    }

    kinematics = [
        "trunnion_table_xyzbc",
        "swivel_head_xyzbc",
        "planar_3axis_xyz"
    ]

    trajectories = {}

    for kin in kinematics:
        cfg = base_machine.copy()
        cfg["kinematic_topology"] = kin
        cfg_path = f"tests/output/mk_cfg_{kin}.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        out_path = f"tests/output/mk_s3_{kin}.json"
        run_stage3("tests/output/mk_s2.json", cfg_path, out_path)

        trajectories[kin] = MachineTrajectory.from_json_file(out_path)

    out_html = "kinematics_comparison.html"
    print(f"Generating Kinematics Comparison Dashboard -> {out_html}")
    generate_stage3_comparison_dashboard(trajectories, out_html)
    print("Done! Open kinematics_comparison.html in a web browser.")

if __name__ == "__main__":
    main()
