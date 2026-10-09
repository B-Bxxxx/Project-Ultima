import streamlit as st
import trimesh
import json
import os
from src.common.schemas import MachineConfig, MinimalPrintProfile, MachineTrajectory, UniversalSlicedModel, CLDataTrajectory
from src.stage1_slicer.registry import PluginRegistry
from src.stage2_toolpath.contour_generator import StandardToolpathGenerator
from src.stage3_kinematics.registry import KinematicsRegistry
from src.stage4_postproc.klipper_postproc import Klipper5AxisPostProcessor
from src.common.visualizer import plot_universal_model, plot_cldata_trajectory
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="Modular 5-Axis CAM GUI", layout="wide")

st.title("Modular 5-Axis CAM GUI")

# Initialize session state for meshes and paths
if "mesh" not in st.session_state:
    st.session_state.mesh = None
if "stage1_model" not in st.session_state:
    st.session_state.stage1_model = None
if "stage2_traj" not in st.session_state:
    st.session_state.stage2_traj = None
if "stage3_traj" not in st.session_state:
    st.session_state.stage3_traj = None
if "stage4_gcode" not in st.session_state:
    st.session_state.stage4_gcode = None

with st.sidebar:
    st.header("1. Input Geometry")
    geom_type = st.selectbox("Geometry", ["Cylinder", "Hollow Cylinder", "Upload STL"])
    if geom_type == "Cylinder":
        r = st.number_input("Radius", value=15.0)
        h = st.number_input("Height", value=20.0)
        if st.button("Generate Mesh"):
            st.session_state.mesh = trimesh.creation.cylinder(radius=r, height=h)
            st.session_state.mesh.apply_translation([0, 0, h/2])
            st.success("Cylinder generated!")
    elif geom_type == "Hollow Cylinder":
        r_max = st.number_input("Outer Radius", value=20.0)
        r_min = st.number_input("Inner Radius", value=15.0)
        h = st.number_input("Height", value=20.0)
        if st.button("Generate Mesh"):
            st.session_state.mesh = trimesh.creation.annulus(r_min=r_min, r_max=r_max, height=h)
            st.session_state.mesh.apply_translation([0, 0, h/2])
            st.success("Hollow Cylinder generated!")
    elif geom_type == "Upload STL":
        uploaded_file = st.file_uploader("Upload STL", type=["stl"])
        if uploaded_file is not None:
            # We can load trimesh directly from file-like object
            st.session_state.mesh = trimesh.load(uploaded_file, file_type='stl')
            st.success("STL loaded!")

    st.header("2. Stage 1: Math Strategy")
    strategy = st.selectbox("Strategy", ["planar", "progressive_tilt", "conical", "custom_expr"])

    stage1_params = {
        "strategy": strategy,
        "layer_height": st.number_input("Layer Height", value=0.4, step=0.1),
        "transition_height": st.number_input("Flat Transition Height", value=2.0, step=0.5)
    }

    if strategy == "progressive_tilt":
        stage1_params["start_tilt_deg"] = st.number_input("Start Tilt (deg)", value=0.0)
        stage1_params["end_tilt_deg"] = st.number_input("End Tilt (deg)", value=25.0)
    elif strategy == "conical":
        stage1_params["cone_angle_deg"] = st.number_input("Cone Angle (deg)", value=15.0)
    elif strategy == "custom_expr":
        stage1_params["expression"] = st.text_input("Expression f(x, y)", value="0.05 * (x**2 + y**2)")

    st.header("3. Stage 2: Print Profile")
    nozzle_dia = st.number_input("Nozzle Dia", value=0.4)
    num_perims = st.number_input("Num Perimeters", value=2)
    infill_density = st.number_input("Infill Density", value=0.2, step=0.1)
    continuous_spiral = st.checkbox("Continuous Spiral", value=False)
    feedrate = st.number_input("Feedrate", value=1500)
    include_rotary = st.checkbox("Include Rotary Axes (B/C)", value=True)

    # Inject walls and infill to stage 1 slice params because they happen before mapping
    stage1_params["nozzle_diameter"] = nozzle_dia
    stage1_params["num_perimeters"] = int(num_perims)
    stage1_params["infill_density"] = infill_density

    st.header("4. Stage 3: Kinematics")
    topology = st.selectbox("Machine Topology", [
        "planar_3axis_xyz",
        "trunnion_table_xyzbc",
        "swivel_head_xyzbc"
    ])
    max_tilt = st.number_input("Max 3-Axis Tilt (deg)", value=25.0)

    if st.button("Run Full Pipeline", type="primary"):
        if st.session_state.mesh is None:
            st.error("Please generate or upload a mesh first.")
        else:
            try:
                # Stage 1
                plugin_cls = PluginRegistry.get_plugin("universal_field_slicer")
                slicer = plugin_cls()
                st.session_state.stage1_model = slicer.slice(st.session_state.mesh, stage1_params)

                # Stage 2
                profile = MinimalPrintProfile(
                    nozzle_diameter=nozzle_dia,
                    layer_height=stage1_params["layer_height"],
                    continuous_spiral=continuous_spiral,
                    include_rotary_axes=include_rotary
                )
                generator = StandardToolpathGenerator()
                # Hack feedrate into the generated waypoints (Stage 2 currently hardcodes 1500, we override for GUI)
                traj = generator.generate_toolpath(st.session_state.stage1_model, profile)
                for wp in traj.waypoints:
                    if not wp.is_travel_move:
                        wp.feedrate = feedrate
                st.session_state.stage2_traj = traj

                # Stage 3
                config = MachineConfig(
                    kinematic_chain="UI Config",
                    kinematic_topology=topology,
                    pivot_offset_z=50.0,
                    b_axis_min=-90.0, b_axis_max=90.0,
                    c_axis_min=-36000.0, c_axis_max=36000.0,
                    max_3axis_tilt_deg=max_tilt
                )
                solver_cls = KinematicsRegistry.get_solver(topology)
                solver = solver_cls()
                st.session_state.stage3_traj = solver.solve(st.session_state.stage2_traj, config)

                # Stage 4
                postproc = Klipper5AxisPostProcessor()
                st.session_state.stage4_gcode = postproc.generate_gcode(st.session_state.stage3_traj, profile)

                st.success("Pipeline executed successfully!")
            except Exception as e:
                st.error(f"Pipeline failed: {str(e)}")

# Main Panel
tab1, tab2, tab3 = st.tabs(["Stage 1: Layer Inspector", "Stage 2/3: Toolpath & Kinematics", "Stage 4: G-Code"])

with tab1:
    if st.session_state.stage1_model:
        fig = plot_universal_model(st.session_state.stage1_model, output_html=None)
        st.plotly_chart(fig, use_container_width=True, height=600)
    else:
        st.info("Run the pipeline to inspect layers.")

with tab2:
    if st.session_state.stage2_traj and st.session_state.stage3_traj:
        fig = plot_cldata_trajectory(st.session_state.stage2_traj, output_html=None)
        st.plotly_chart(fig, use_container_width=True, height=600)

        # Simple metrics
        b_angles = [s.b for s in st.session_state.stage3_traj.states if getattr(s, "is_travel_move", False) == False]
        z_coords = [s.z for s in st.session_state.stage3_traj.states if not s.is_travel_move]
        if b_angles:
            st.metric("Max Tilt (B axis)", f"{max(b_angles):.2f}°")
            st.metric("Max Z Height", f"{max(z_coords):.2f} mm")
    else:
        st.info("Run the pipeline to inspect toolpaths.")

with tab3:
    if st.session_state.stage4_gcode:
        st.download_button(
            label="Download .gcode",
            data=st.session_state.stage4_gcode,
            file_name="output.gcode",
            mime="text/plain"
        )
        st.text_area("G-Code Preview", st.session_state.stage4_gcode, height=600)
    else:
        st.info("Run the pipeline to generate G-code.")
