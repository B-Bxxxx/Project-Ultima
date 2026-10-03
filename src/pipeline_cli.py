import argparse
import sys
import json
from src.common.schemas import MachineConfig, MinimalPrintProfile, UniversalSlicedModel, CLDataTrajectory, MachineTrajectory

# Stage 1
from src.stage1_slicer.registry import PluginRegistry
# Stage 3
from src.stage3_kinematics.trunnion_solver import TrunnionXYZBCSolver
# Stage 4
from src.stage4_postproc.klipper_postproc import Klipper5AxisPostProcessor


def run_stage1(plugin_name: str, config_path: str, output_path: str):
    print(f"Running Stage 1 using plugin: {plugin_name}")
    try:
        plugin_cls = PluginRegistry.get_plugin(plugin_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        params = json.load(f)

    slicer = plugin_cls()

    geometry = None
    if plugin_name == "universal_field_slicer":
        # Create a dummy mesh for CLI if not provided programmatically
        import trimesh
        geometry = trimesh.creation.cylinder(radius=10, height=10)

    sliced_model = slicer.slice(geometry=geometry, parameters=params)
    sliced_model.to_json_file(output_path)
    print(f"Stage 1 complete. Output saved to {output_path}")

def run_stage2(input_path: str, config_path: str, output_path: str):
    print("Running Stage 2 (Toolpath Generation)")
    model = UniversalSlicedModel.from_json_file(input_path)

    with open(config_path, 'r') as f:
        profile_data = json.load(f)
    profile = MinimalPrintProfile(**profile_data)

    from src.stage2_toolpath.contour_generator import StandardToolpathGenerator
    generator = StandardToolpathGenerator()
    traj = generator.generate_toolpath(model, profile)

    traj.to_json_file(output_path)
    print(f"Stage 2 complete. Output saved to {output_path}")

def run_stage3(input_path: str, config_path: str, output_path: str):
    print("Running Stage 3 (Kinematics)")
    trajectory = CLDataTrajectory.from_json_file(input_path)

    with open(config_path, 'r') as f:
        config_data = json.load(f)
    config = MachineConfig(**config_data)

    # Dynamic loader via registry
    from src.stage3_kinematics.registry import KinematicsRegistry
    import src.stage3_kinematics.trunnion_solver
    import src.stage3_kinematics.head_solver
    # Default to trunnion for compatibility if not specified
    solver_name = config_data.get("kinematic_topology", "trunnion_table_xyzbc")

    try:
        solver_cls = KinematicsRegistry.get_solver(solver_name)
    except ValueError as e:
        print(f"Error: {e}")
        import sys
        sys.exit(1)

    solver = solver_cls()
    machine_traj = solver.solve(trajectory, config)
    machine_traj.to_json_file(output_path)
    print(f"Stage 3 complete. Output saved to {output_path}")

def run_stage4(input_path: str, config_path: str, output_path: str):
    print("Running Stage 4 (Post-Processor)")
    trajectory = MachineTrajectory.from_json_file(input_path)

    with open(config_path, 'r') as f:
        profile_data = json.load(f)
    profile = MinimalPrintProfile(**profile_data)

    postproc = Klipper5AxisPostProcessor()
    gcode = postproc.generate_gcode(trajectory, profile)

    with open(output_path, 'w') as f:
        f.write(gcode)
    print(f"Stage 4 complete. G-code saved to {output_path}")


def run_all(plugin_name: str, slicer_config: str, machine_config: str, print_profile: str, out_prefix: str):
    print(f"Running Full Pipeline with prefix '{out_prefix}'")
    stage1_out = f"{out_prefix}_s1.json"
    stage2_out = f"{out_prefix}_s2.json"
    stage3_out = f"{out_prefix}_s3.json"
    stage4_out = f"{out_prefix}.gcode"

    run_stage1(plugin_name, slicer_config, stage1_out)
    run_stage2(stage1_out, print_profile, stage2_out)
    run_stage3(stage2_out, machine_config, stage3_out)
    run_stage4(stage3_out, print_profile, stage4_out)
    print(f"Full pipeline complete. Final G-code: {stage4_out}")


def visualize(input_path: str, output_path: str):
    from src.common.visualizer import plot_universal_model, plot_cldata_trajectory
    print(f"Visualizing {input_path} -> {output_path}")

    # Try to guess the schema
    with open(input_path, 'r') as f:
        data = json.load(f)

    if "layers" in data:
        model = UniversalSlicedModel(**data)
        plot_universal_model(model, output_path)
    elif "waypoints" in data:
        traj = CLDataTrajectory(**data)
        plot_cldata_trajectory(traj, output_path)
    else:
        print("Unknown JSON format. Must be UniversalSlicedModel or CLDataTrajectory.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Modular 5-Axis CAM Pipeline Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run-stage1
    p1 = subparsers.add_parser("run-stage1")
    p1.add_argument("--plugin", required=True, help="Slicer plugin name")
    p1.add_argument("--config", required=True, help="Path to plugin parameters JSON")
    p1.add_argument("--output", required=True, help="Path to output UniversalSlicedModel JSON")

    # run-stage2
    p2 = subparsers.add_parser("run-stage2")
    p2.add_argument("--input", required=True, help="Path to input UniversalSlicedModel JSON")
    p2.add_argument("--config", required=True, help="Path to print profile JSON")
    p2.add_argument("--output", required=True, help="Path to output CLDataTrajectory JSON")

    # run-stage3
    p3 = subparsers.add_parser("run-stage3")
    p3.add_argument("--input", required=True, help="Path to input CLDataTrajectory JSON")
    p3.add_argument("--config", required=True, help="Path to machine config JSON")
    p3.add_argument("--output", required=True, help="Path to output MachineTrajectory JSON")

    # run-stage4
    p4 = subparsers.add_parser("run-stage4")
    p4.add_argument("--input", required=True, help="Path to input MachineTrajectory JSON")
    p4.add_argument("--config", required=True, help="Path to print profile JSON")
    p4.add_argument("--output", required=True, help="Path to output G-Code file")

    # run-all
    pall = subparsers.add_parser("run-all")
    pall.add_argument("--plugin", required=True, help="Slicer plugin name")
    pall.add_argument("--slicer-config", required=True, help="Path to plugin parameters JSON")
    pall.add_argument("--machine-config", required=True, help="Path to machine config JSON")
    pall.add_argument("--print-profile", required=True, help="Path to print profile JSON")
    pall.add_argument("--out-prefix", required=True, help="Prefix for output files")

    # visualize
    pvis = subparsers.add_parser("visualize")
    pvis.add_argument("--input", required=True, help="Path to intermediate JSON file")
    pvis.add_argument("--output", required=True, help="Path to output HTML plot")

    args = parser.parse_args()

    if args.command == "run-stage1":
        run_stage1(args.plugin, args.config, args.output)
    elif args.command == "run-stage2":
        run_stage2(args.input, args.config, args.output)
    elif args.command == "run-stage3":
        run_stage3(args.input, args.config, args.output)
    elif args.command == "run-stage4":
        run_stage4(args.input, args.config, args.output)
    elif args.command == "run-all":
        run_all(args.plugin, args.slicer_config, args.machine_config, args.print_profile, args.out_prefix)
    elif args.command == "visualize":
        visualize(args.input, args.output)

if __name__ == "__main__":
    main()
