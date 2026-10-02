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

    # Read slicer params (for math model, we just pass params)
    with open(config_path, 'r') as f:
        params = json.load(f)

    slicer = plugin_cls()
    sliced_model = slicer.slice(geometry=None, parameters=params)
    sliced_model.to_json_file(output_path)
    print(f"Stage 1 complete. Output saved to {output_path}")

def run_stage2(input_path: str, config_path: str, output_path: str):
    print("Stage 2 toolpath generator is not yet fully implemented. Mocking bypass...")
    # Normally we load UniversalSlicedModel, but here we just convert it to a dummy trajectory for testing
    model = UniversalSlicedModel.from_json_file(input_path)
    # mock conversion
    from src.common.schemas import CLDataWaypoint
    waypoints = []
    for layer in model.layers:
        for contour in layer.contours:
            for pt, norm in zip(contour.points, contour.normals):
                waypoints.append(CLDataWaypoint(
                    x=pt[0], y=pt[1], z=pt[2],
                    i=norm[0], j=norm[1], k=norm[2],
                    feedrate=1500.0,
                    is_travel_move=False,
                    extrusion_volume=0.1
                ))
    traj = CLDataTrajectory(waypoints=waypoints)
    traj.to_json_file(output_path)
    print(f"Stage 2 complete. Output saved to {output_path}")

def run_stage3(input_path: str, config_path: str, output_path: str):
    print("Running Stage 3 (Kinematics)")
    trajectory = CLDataTrajectory.from_json_file(input_path)

    with open(config_path, 'r') as f:
        config_data = json.load(f)
    config = MachineConfig(**config_data)

    solver = TrunnionXYZBCSolver()
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

    args = parser.parse_args()

    if args.command == "run-stage1":
        run_stage1(args.plugin, args.config, args.output)
    elif args.command == "run-stage2":
        run_stage2(args.input, args.config, args.output)
    elif args.command == "run-stage3":
        run_stage3(args.input, args.config, args.output)
    elif args.command == "run-stage4":
        run_stage4(args.input, args.config, args.output)

if __name__ == "__main__":
    main()
