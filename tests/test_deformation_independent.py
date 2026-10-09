import pytest
import numpy as np
import trimesh
from src.stage1_slicer.math_strategies import ConicalStrategy, CustomExpressionStrategy

def test_cone_normal_outward():
    # Create a box to represent the mesh
    mesh = trimesh.creation.box(extents=(20, 20, 20))

    alpha_deg = 30
    alpha = np.radians(alpha_deg)

    strat = ConicalStrategy(mesh, {"cone_angle_deg": alpha_deg, "layer_height": 1.0, "start_z": 0.0, "end_z": 10.0, "transition_height": 0.0})

    x, y, z = 5.0, 5.0, 0.0

    nx, ny, nz = strat.compute_normal((x, y, z), layer_idx=5)

    r = np.sqrt(x**2 + y**2)
    expected_nx = (x/r) * np.tan(alpha)
    expected_ny = (y/r) * np.tan(alpha)

    length = np.sqrt(expected_nx**2 + expected_ny**2 + 1.0)
    expected_nx /= length
    expected_ny /= length
    expected_nz = 1.0 / length

    assert nx == pytest.approx(expected_nx, abs=1e-5)
    assert ny == pytest.approx(expected_ny, abs=1e-5)
    assert nz == pytest.approx(expected_nz, abs=1e-5)

    assert nx > 0
    assert ny > 0


def test_round_trip_deformation():
    mesh = trimesh.creation.box(extents=(20, 20, 20))
    mesh.apply_translation([0, 0, 10])

    expr = "5 * np.sin(0.1 * x)"

    strat = CustomExpressionStrategy(mesh, {
        "expression": expr,
        "layer_height": 1.0,
        "start_z": 0.0,
        "end_z": 20.0,
        "transition_height": 10.0,
        "refine_mesh": False
    })

    orig_x, orig_y, orig_z = 5.0, 5.0, 5.0
    f_val = 5 * np.sin(0.1 * orig_x)

    z_prime = strat._solve_forward_z(orig_z, f_val)

    w = strat.get_blend_weight(z_prime)
    z_undeformed = z_prime - w * f_val

    assert z_undeformed == pytest.approx(orig_z, abs=1e-6)

    und_x, und_y, und_z = strat.undeform_point((orig_x, orig_y, z_prime), layer_idx=5)
    assert und_x == pytest.approx(orig_x, abs=1e-6)
    assert und_y == pytest.approx(orig_y, abs=1e-6)
    assert und_z == pytest.approx(orig_z, abs=1e-6)


def test_mesh_refinement_analytic_surface():
    mesh = trimesh.creation.box(extents=(30, 30, 10))
    mesh.apply_translation([0, 0, 5])

    expr = "5 * np.sin(0.1 * x)"

    strat = CustomExpressionStrategy(mesh.copy(), {
        "expression": expr,
        "layer_height": 1.0,
        "start_z": 0.0,
        "end_z": 20.0,
        "transition_height": 0.0,
        "refine_mesh": True,
        "refine_max_edge": 2.0
    })

    deformed = strat.deform_mesh()

    x_target = 15.0
    tol = 0.5

    mask = np.abs(deformed.vertices[:, 0] - x_target) < tol
    verts_near_x = deformed.vertices[mask]

    max_z = np.max(verts_near_x[:, 2])
    expected_max_z = 10.0 + 5 * np.sin(0.1 * x_target)

    assert max_z == pytest.approx(expected_max_z, abs=0.2)
