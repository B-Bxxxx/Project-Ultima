import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from typing import Dict
from src.common.schemas import UniversalSlicedModel, CLDataTrajectory, MachineTrajectory

def plot_universal_model(model: UniversalSlicedModel, output_html: str, fig: go.Figure = None, row=None, col=None, show=True):
    if fig is None:
        fig = go.Figure()

    num_layers = len(model.layers)
    colors = px.colors.sample_colorscale("viridis", [n/(max(1, num_layers-1)) for n in range(num_layers)])

    for layer in model.layers:
        color = colors[layer.layer_index % len(colors)]

        layer_x, layer_y, layer_z = [], [], []
        vec_x, vec_y, vec_z = [], [], []

        for contour in layer.contours:
            pts = np.array(contour.points)
            if len(pts) == 0:
                continue

            layer_x.extend(pts[:, 0].tolist() + [None])
            layer_y.extend(pts[:, 1].tolist() + [None])
            layer_z.extend(pts[:, 2].tolist() + [None])

            step = max(1, len(pts) // 20)
            norms = np.array(contour.normals)
            scale = 2.0

            for i in range(0, len(pts), step):
                px_val, py_val, pz_val = pts[i]
                nx, ny, nz = norms[i]
                vec_x.extend([px_val, px_val + nx * scale, None])
                vec_y.extend([py_val, py_val + ny * scale, None])
                vec_z.extend([pz_val, pz_val + nz * scale, None])

        if layer_x:
            trace = go.Scatter3d(
                x=layer_x, y=layer_y, z=layer_z,
                mode='lines',
                line=dict(color=color, width=4),
                name=f'Layer {layer.layer_index}',
                legendgroup=f'layer_{layer.layer_index}'
            )
            if row is not None and col is not None:
                fig.add_trace(trace, row=row, col=col)
            else:
                fig.add_trace(trace)

        if vec_x:
            quiver_trace = go.Scatter3d(
                x=vec_x, y=vec_y, z=vec_z,
                mode='lines',
                line=dict(color='gray', width=2),
                name=f'Normals L{layer.layer_index}',
                legendgroup=f'layer_{layer.layer_index}',
                showlegend=False
            )
            if row is not None and col is not None:
                fig.add_trace(quiver_trace, row=row, col=col)
            else:
                fig.add_trace(quiver_trace)

    fig.update_layout(title="Stage 1: Universal Sliced Model")
    fig.update_scenes(aspectmode='data')
    if show and output_html:
        fig.write_html(output_html)
    return fig


def generate_stage1_comparison_dashboard(models: Dict[str, UniversalSlicedModel], output_html: str):
    titles = list(models.keys())

    fig = make_subplots(
        rows=1, cols=len(models),
        specs=[[{'type': 'scene'} for _ in range(len(models))]],
        subplot_titles=titles
    )

    for idx, (title, model) in enumerate(models.items()):
        plot_universal_model(model, output_html, fig=fig, row=1, col=idx+1, show=False)

    fig.update_layout(title="Stage 1: Multi-Mode Comparison Dashboard", height=800)
    fig.update_scenes(aspectmode='data')
    fig.write_html(output_html)


def generate_stage3_comparison_dashboard(trajectories: Dict[str, MachineTrajectory], output_html: str):
    titles = list(trajectories.keys())
    fig = make_subplots(
        rows=1, cols=len(trajectories),
        specs=[[{'type': 'scene'} for _ in range(len(trajectories))]],
        subplot_titles=titles
    )

    for idx, (title, traj) in enumerate(trajectories.items()):
        extrude_x, extrude_y, extrude_z = [], [], []
        travel_x, travel_y, travel_z = [], [], []
        last_pt = None

        for wp in traj.states:
            if last_pt is not None:
                if wp.is_travel_move:
                    travel_x.extend([last_pt.x, wp.x, None])
                    travel_y.extend([last_pt.y, wp.y, None])
                    travel_z.extend([last_pt.z, wp.z, None])
                else:
                    extrude_x.extend([last_pt.x, wp.x, None])
                    extrude_y.extend([last_pt.y, wp.y, None])
                    extrude_z.extend([last_pt.z, wp.z, None])
            last_pt = wp

        fig.add_trace(go.Scatter3d(
            x=extrude_x, y=extrude_y, z=extrude_z,
            mode='lines', line=dict(color='blue', width=4),
            name=f'Extrude {title}'
        ), row=1, col=idx+1)

        fig.add_trace(go.Scatter3d(
            x=travel_x, y=travel_y, z=travel_z,
            mode='lines', line=dict(color='red', width=2, dash='dash'),
            name=f'Travel {title}'
        ), row=1, col=idx+1)

    fig.update_layout(title="Stage 3: Kinematics Machine Path Comparison Dashboard", height=800)
    fig.update_scenes(aspectmode='data')
    fig.write_html(output_html)


def plot_cldata_trajectory(traj: CLDataTrajectory, output_html: str):
    fig = go.Figure()

    extrude_x, extrude_y, extrude_z = [], [], []
    travel_x, travel_y, travel_z = [], [], []
    vec_x, vec_y, vec_z = [], [], []

    last_pt = None
    for i, wp in enumerate(traj.waypoints):
        if last_pt is not None:
            if wp.is_travel_move:
                travel_x.extend([last_pt.x, wp.x, None])
                travel_y.extend([last_pt.y, wp.y, None])
                travel_z.extend([last_pt.z, wp.z, None])
            else:
                extrude_x.extend([last_pt.x, wp.x, None])
                extrude_y.extend([last_pt.y, wp.y, None])
                extrude_z.extend([last_pt.z, wp.z, None])

        last_pt = wp

        if not wp.is_travel_move and i % 10 == 0:
            scale = 3.0
            vec_x.extend([wp.x, wp.x + wp.i * scale, None])
            vec_y.extend([wp.y, wp.y + wp.j * scale, None])
            vec_z.extend([wp.z, wp.z + wp.k * scale, None])

    fig.add_trace(go.Scatter3d(
        x=extrude_x, y=extrude_y, z=extrude_z,
        mode='lines', line=dict(color='blue', width=4),
        name='Extrusion'
    ))

    fig.add_trace(go.Scatter3d(
        x=travel_x, y=travel_y, z=travel_z,
        mode='lines', line=dict(color='red', width=2, dash='dash'),
        name='Travel'
    ))

    if vec_x:
        fig.add_trace(go.Scatter3d(
            x=vec_x, y=vec_y, z=vec_z,
            mode='lines', line=dict(color='orange', width=2),
            name='Tool Vector'
        ))

    fig.update_layout(title="Stage 2: CL-Data Trajectory (Toolpath)")
    fig.update_scenes(aspectmode='data')
    if output_html:
        fig.write_html(output_html)
    return fig
