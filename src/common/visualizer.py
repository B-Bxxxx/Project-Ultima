import plotly.graph_objects as go
import numpy as np
from src.common.schemas import UniversalSlicedModel, CLDataTrajectory

def plot_universal_model(model: UniversalSlicedModel, output_html: str):
    fig = go.Figure()

    for layer in model.layers:
        for contour in layer.contours:
            pts = np.array(contour.points)
            if len(pts) == 0:
                continue

            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

            # Plot line
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='blue', width=4),
                name=f'Layer {layer.layer_index}'
            ))

            # Subsample normals for visualization (e.g. max 20 arrows per contour)
            step = max(1, len(pts) // 20)
            norms = np.array(contour.normals)

            # Plot quivers (normals)
            # Scaling normals for visibility
            scale = 2.0
            for i in range(0, len(pts), step):
                px, py, pz = pts[i]
                nx, ny, nz = norms[i]
                fig.add_trace(go.Scatter3d(
                    x=[px, px + nx * scale],
                    y=[py, py + ny * scale],
                    z=[pz, pz + nz * scale],
                    mode='lines',
                    line=dict(color='green', width=2),
                    showlegend=False
                ))

    fig.update_layout(title="Stage 1: Universal Sliced Model", scene=dict(aspectmode='data'))
    fig.write_html(output_html)


def plot_cldata_trajectory(traj: CLDataTrajectory, output_html: str):
    fig = go.Figure()

    # Separate travel and extrusion segments
    extrude_x, extrude_y, extrude_z = [], [], []
    travel_x, travel_y, travel_z = [], [], []

    # For tool vectors
    vec_x, vec_y, vec_z = [], [], []
    vec_i, vec_j, vec_k = [], [], []

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

        # Subsample vectors (e.g., every 10th extruding waypoint)
        if not wp.is_travel_move and i % 10 == 0:
            vec_x.append(wp.x)
            vec_y.append(wp.y)
            vec_z.append(wp.z)
            vec_i.append(wp.i)
            vec_j.append(wp.j)
            vec_k.append(wp.k)

    # Plot Extrusions
    fig.add_trace(go.Scatter3d(
        x=extrude_x, y=extrude_y, z=extrude_z,
        mode='lines',
        line=dict(color='blue', width=4),
        name='Extrusion'
    ))

    # Plot Travels
    fig.add_trace(go.Scatter3d(
        x=travel_x, y=travel_y, z=travel_z,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Travel'
    ))

    # Plot Tool Vectors
    scale = 3.0
    for x, y, z, i, j, k in zip(vec_x, vec_y, vec_z, vec_i, vec_j, vec_k):
        fig.add_trace(go.Scatter3d(
            x=[x, x + i * scale],
            y=[y, y + j * scale],
            z=[z, z + k * scale],
            mode='lines',
            line=dict(color='orange', width=2),
            showlegend=False
        ))

    fig.update_layout(title="Stage 2: CL-Data Trajectory (Toolpath)", scene=dict(aspectmode='data'))
    fig.write_html(output_html)
