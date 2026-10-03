import os
import trimesh
from src.stage1_slicer.registry import PluginRegistry
from src.common.visualizer import generate_stage1_comparison_dashboard

def main():
    print("Generating Test Mesh (Cylinder)...")
    # A simple cylinder: radius 20, height 40
    mesh = trimesh.creation.cylinder(radius=20, height=40)
    # trimesh centers it at origin, let's move it to rest on Z=0
    mesh.apply_translation([0, 0, 20])

    print("Initializing Universal Field Slicer...")
    plugin_cls = PluginRegistry.get_plugin("universal_field_slicer")
    slicer = plugin_cls()

    models = {}

    # 1. Planar Slicing
    print("Slicing Mode 1: Planar")
    params_planar = {
        "mode": "planar",
        "layer_height": 2.0,
        "start_z": 0.0,
        "end_z": 40.0
    }
    models["Planar"] = slicer.slice(mesh, params_planar)

    # 2. Progressive Tilt Slicing
    print("Slicing Mode 2: Progressive Tilt (0 to 35 deg)")
    params_tilt = {
        "mode": "progressive_tilt",
        "layer_height": 2.0,
        "start_z": 0.0,
        "end_z": 40.0,
        "start_tilt_deg": 0.0,
        "end_tilt_deg": 35.0
    }
    models["Progressive Tilt"] = slicer.slice(mesh, params_tilt)

    # 3. Conical/Curved Slicing
    print("Slicing Mode 3: Conical/Curved Field (15 deg)")
    params_conical = {
        "mode": "conical_or_curved",
        "layer_height": 2.0,
        "start_z": 0.0,
        "end_z": 40.0,
        "cone_angle_deg": 15.0
    }
    models["Conical Field"] = slicer.slice(mesh, params_conical)

    # 4. Generate Dashboard
    out_path = "stage1_comparison.html"
    print(f"Generating Comparison Dashboard -> {out_path}")
    generate_stage1_comparison_dashboard(models, out_path)
    print("Done! Open stage1_comparison.html in a web browser to inspect.")

if __name__ == "__main__":
    main()
