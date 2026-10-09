import pytest
import numpy as np
import trimesh
import math
from src.stage1_slicer.math_strategies import CustomExpressionStrategy
from src.stage3_kinematics.trunnion_solver import TrunnionXYZBCSolver
from src.stage3_kinematics.head_solver import SwivelHeadXYZBCSolver
from src.common.schemas import MachineConfig, CLDataTrajectory, CLDataWaypoint

def test_math_validation_custom_expressions():
    # We will test 3 functions:
    # 1. Paraboloid: z = 0.05*(x^2 + y^2)
    # 2. Saddle: z = 0.04*(x^2 - y^2)
    # 3. Trig Wave: z = 2.5*sin(0.2*x)*cos(0.2*y)

    mesh = trimesh.creation.box(extents=(20, 20, 20))
    # We just need the strategy, we'll probe it manually without slicing

    cases = [
        {
            "expr": "0.05 * (x**2 + y**2)",
            "dzdx_analytical": lambda x, y: 0.1 * x,
            "dzdy_analytical": lambda x, y: 0.1 * y
        },
        {
            "expr": "0.04 * (x**2 - y**2)",
            "dzdx_analytical": lambda x, y: 0.08 * x,
            "dzdy_analytical": lambda x, y: -0.08 * y
        },
        {
            "expr": "2.5 * np.sin(0.2 * x) * np.cos(0.2 * y)",
            "dzdx_analytical": lambda x, y: 2.5 * 0.2 * np.cos(0.2 * x) * np.cos(0.2 * y),
            "dzdy_analytical": lambda x, y: 2.5 * 0.2 * np.sin(0.2 * x) * -np.sin(0.2 * y)
        }
    ]

    test_points = [
        (0.0, 0.0),
        (5.0, 0.0),
        (0.0, -5.0),
        (4.2, -7.8),
        (-9.1, 8.4)
    ]

    for case in cases:
        strat = CustomExpressionStrategy(mesh, {"expression": case["expr"], "layer_height": 2.0})

        for tx, ty in test_points:
            # Emulate an intersected point on the Z=10 slicing plane
            z_plane = 10.0

            # 1. Undeform point
            ux, uy, uz = strat.undeform_point((tx, ty, z_plane), layer_idx=0)

            # Assert Z reconstruction is mathematically exact
            f_val = strat._evaluate(tx, ty)
            assert uz == pytest.approx(z_plane - f_val, abs=1e-5)

            # 2. Normal validation
            norm = strat.compute_normal((ux, uy, uz), layer_idx=0)

            dzdx = case["dzdx_analytical"](tx, ty)
            dzdy = case["dzdy_analytical"](tx, ty)

            # Analytical normal = [-dz/dx, -dz/dy, 1] normalized
            a_norm = np.array([dzdx, dzdy, 1.0])
            a_norm /= np.linalg.norm(a_norm)

            assert norm[0] == pytest.approx(a_norm[0], abs=1e-4)
            assert norm[1] == pytest.approx(a_norm[1], abs=1e-4)
            assert norm[2] == pytest.approx(a_norm[2], abs=1e-4)

def test_math_validation_kinematics_roundtrip():
    # Forward Kinematics verifier for Trunnion (Bed tilts B and rotates C)
    def trunnion_fk(x_m, y_m, z_m, b_deg, c_deg, pivot):
        b_rad = np.radians(b_deg)
        c_rad = np.radians(c_deg)

        cos_b, sin_b = np.cos(b_rad), np.sin(b_rad)
        R_B = np.array([
            [ cos_b, 0, sin_b],
            [ 0,     1, 0    ],
            [-sin_b, 0, cos_b]
        ])

        cos_c, sin_c = np.cos(c_rad), np.sin(c_rad)
        R_C = np.array([
            [cos_c, -sin_c, 0],
            [sin_c,  cos_c, 0],
            [0,      0,     1]
        ])

        P_mach = np.array([x_m, y_m, z_m])
        P_pivot = np.array(pivot)

        # The Inverse kinematics calculated:
        # P_c = R_C @ P_part
        # V_pivot = P_c - P_pivot
        # V_rot = R_B @ V_pivot
        # P_mach = P_pivot + V_rot

        # Reverse:
        # V_rot = P_mach - P_pivot
        # V_pivot = R_B_inv @ V_rot
        # P_c = V_pivot + P_pivot
        # P_part = R_C_inv @ P_c

        R_B_inv = R_B.T
        R_C_inv = R_C.T

        V_rot = P_mach - P_pivot
        V_pivot = R_B_inv @ V_rot
        P_c = V_pivot + P_pivot
        P_part = R_C_inv @ P_c

        # Tool normal N_m is always [0, 0, 1] for the machine tool
        # N_part = R_C_inv * R_B_inv * [0, 0, 1]
        N_part = R_C_inv @ (R_B_inv @ np.array([0, 0, 1]))

        return P_part, N_part

    config = MachineConfig(
        kinematic_chain="Trunnion",
        pivot_offset_x=10.0, pivot_offset_y=-20.0, pivot_offset_z=50.0,
        b_axis_min=-90, b_axis_max=90, c_axis_min=-36000, c_axis_max=36000
    )

    solver = TrunnionXYZBCSolver()

    # Test arbitrary tool vectors and positions
    test_wps = [
        CLDataWaypoint(x=5, y=5, z=5, i=0, j=0, k=1, feedrate=1000), # flat
        CLDataWaypoint(x=-15, y=30, z=0, i=0.707, j=0, k=0.707, feedrate=1000), # 45 tilt
        CLDataWaypoint(x=42, y=-11, z=99, i=-0.5, j=0.5, k=0.707, feedrate=1000) # compound
    ]

    traj = CLDataTrajectory(waypoints=test_wps)
    machine_traj = solver.solve(traj, config)

    pivot = [config.pivot_offset_x, config.pivot_offset_y, config.pivot_offset_z]

    for i, state in enumerate(machine_traj.states):
        orig_wp = test_wps[i]

        p_part, n_part = trunnion_fk(state.x, state.y, state.z, state.b, state.c, pivot)

        assert p_part[0] == pytest.approx(orig_wp.x, abs=1e-5)
        assert p_part[1] == pytest.approx(orig_wp.y, abs=1e-5)
        assert p_part[2] == pytest.approx(orig_wp.z, abs=1e-5)

        # Verify tool normal
        norm = np.linalg.norm([orig_wp.i, orig_wp.j, orig_wp.k])
        orig_i = orig_wp.i / norm
        orig_j = orig_wp.j / norm
        orig_k = orig_wp.k / norm

        assert n_part[0] == pytest.approx(orig_i, abs=1e-5)
        assert n_part[1] == pytest.approx(orig_j, abs=1e-5)
        assert n_part[2] == pytest.approx(orig_k, abs=1e-5)


def test_math_validation_swivel_head_kinematics_roundtrip():
    # Forward Kinematics verifier for Swivel Head (Tool tilts B and rotates C)
    def swivel_fk(x_m, y_m, z_m, b_deg, c_deg, pivot):
        b_rad = np.radians(b_deg)
        c_rad = np.radians(c_deg)

        cos_b, sin_b = np.cos(b_rad), np.sin(b_rad)
        R_B = np.array([
            [ cos_b, 0, sin_b],
            [ 0,     1, 0    ],
            [-sin_b, 0, cos_b]
        ])

        cos_c, sin_c = np.cos(c_rad), np.sin(c_rad)
        R_C = np.array([
            [cos_c, -sin_c, 0],
            [sin_c,  cos_c, 0],
            [0,      0,     1]
        ])

        P_mach = np.array([x_m, y_m, z_m])
        P_pivot = np.array(pivot)

        # Inverse mapping from head_solver:
        # offset = P_pivot - R_C @ (R_B @ P_pivot)
        # P_mach = P_part + offset

        offset = P_pivot - R_C @ (R_B @ P_pivot)
        P_part = P_mach - offset

        # Tool normal
        N_part = R_C @ (R_B @ np.array([0, 0, 1]))

        return P_part, N_part

    config = MachineConfig(
        kinematic_chain="SwivelHead",
        pivot_offset_x=0.0, pivot_offset_y=0.0, pivot_offset_z=75.0,
        b_axis_min=-90, b_axis_max=90, c_axis_min=-36000, c_axis_max=36000
    )

    solver = SwivelHeadXYZBCSolver()

    test_wps = [
        CLDataWaypoint(x=10, y=-5, z=20, i=0, j=0, k=1, feedrate=1000),
        CLDataWaypoint(x=50, y=50, z=50, i=0.5, j=0.5, k=0.707, feedrate=1000),
        CLDataWaypoint(x=-100, y=0, z=10, i=1.0, j=0.0, k=0.0, feedrate=1000)
    ]

    traj = CLDataTrajectory(waypoints=test_wps)
    machine_traj = solver.solve(traj, config)

    pivot = [config.pivot_offset_x, config.pivot_offset_y, config.pivot_offset_z]

    for i, state in enumerate(machine_traj.states):
        orig_wp = test_wps[i]

        p_part, n_part = swivel_fk(state.x, state.y, state.z, state.b, state.c, pivot)

        assert p_part[0] == pytest.approx(orig_wp.x, abs=1e-5)
        assert p_part[1] == pytest.approx(orig_wp.y, abs=1e-5)
        assert p_part[2] == pytest.approx(orig_wp.z, abs=1e-5)

        norm = np.linalg.norm([orig_wp.i, orig_wp.j, orig_wp.k])
        orig_i = orig_wp.i / norm
        orig_j = orig_wp.j / norm
        orig_k = orig_wp.k / norm

        assert n_part[0] == pytest.approx(orig_i, abs=1e-5)
        assert n_part[1] == pytest.approx(orig_j, abs=1e-5)
        assert n_part[2] == pytest.approx(orig_k, abs=1e-5)
