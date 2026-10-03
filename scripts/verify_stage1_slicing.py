import os
import trimesh
from src.stage1_slicer.registry import PluginRegistry
from src.common.visualizer import generate_stage1_comparison_dashboard

def main():
    print("Generating Test Mesh (Hollow Cylinder / Frustum)...")
    outer = trimesh.creation.cylinder(radius=20, height=40)
    inner = trimesh.creation.cylinder(radius=15, height=40)

    # We do a boolean difference. Note: trimesh boolean can be brittle without engines,
    # but for simple primitives, or we just merge them if we only care about slicing lines.
    # Alternatively, create an annulus extrusion manually. Let's create an annulus to be safe.
    annulus = trimesh.creation.annulus(r_min=15, r_max=20, height=40)
    annulus.apply_translation([0, 0, 20])
    mesh = annulus

    print("Initializing Universal Field Slicer...")
    plugin_cls = PluginRegistry.get_plugin("universal_field_slicer")
    slicer = plugin_cls()

    models = {}

    # 1. Planar
    print("Slicing: Planar")
    models["Planar"] = slicer.slice(mesh, {
        "strategy": "planar", "layer_height": 2.0, "start_z": 0.0, "end_z": 40.0
    })

    # 2. Progressive Tilt
    print("Slicing: Progressive Tilt")
    models["Progressive Tilt"] = slicer.slice(mesh, {
        "strategy": "progressive_tilt", "layer_height": 2.0, "start_z": 0.0, "end_z": 40.0,
        "start_tilt_deg": 0.0, "end_tilt_deg": 35.0
    })

    # 3. Conical
    print("Slicing: Conical Field (15 deg)")
    models["Conical"] = slicer.slice(mesh, {
        "strategy": "conical", "layer_height": 2.0, "start_z": 0.0, "end_z": 40.0,
        "cone_angle_deg": 15.0
    })

    # 4. Custom Expression
    print("Slicing: Custom Expression (Sine Wave)")
    models["Custom Expr (Sine)"] = slicer.slice(mesh, {
        "strategy": "custom_expr", "layer_height": 2.0, "start_z": 0.0, "end_z": 40.0,
        "expression": "5.0 * np.sin(x / 5.0)"
    })

    # 5. External Scalar Field (Mocked)
    print("Slicing: External Scalar Field (Mock Paraboloid)")
    vertex_z_deformations = 0.01 * (mesh.vertices[:, 0]**2 + mesh.vertices[:, 1]**2)
    models["External Field"] = slicer.slice(mesh, {
        "strategy": "external_field", "layer_height": 2.0, "start_z": 0.0, "end_z": 40.0,
        "vertex_deformations": vertex_z_deformations
    })

    out_path = "stage1_comparison.html"
    print(f"Generating Comparison Dashboard -> {out_path}")
    generate_stage1_comparison_dashboard(models, out_path)

if __name__ == "__main__":
    main()
