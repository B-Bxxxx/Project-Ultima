import numpy as np
from typing import List
from src.common.schemas import CLDataTrajectory, MachineConfig, MachineStateVector, MachineTrajectory
from src.stage3_kinematics.base import BaseKinematicSolver

class TrunnionXYZBCSolver(BaseKinematicSolver):
    """
    Kinematic solver for a 5-axis trunnion/gantry (XYZBC) setup.
    B-axis: Rotation around Y (tilt)
    C-axis: Rotation around Z (rotary table)
    """

    def solve(self, trajectory: CLDataTrajectory, config: MachineConfig) -> MachineTrajectory:
        states = []

        # Unpack configurations
        px, py, pz = config.pivot_offset_x, config.pivot_offset_y, config.pivot_offset_z
        ex, ey = config.eccentricity_x, config.eccentricity_y
        b_min, b_max = config.b_axis_min, config.b_axis_max
        c_min, c_max = config.c_axis_min, config.c_axis_max

        # We need to compute C angles and unwrap them to prevent 360-degree flips
        raw_c_angles = []
        b_angles = []
        valid_waypoints = []

        # First pass: Calculate ideal B and C angles for the tool vectors (I, J, K)
        # We want to align the surface normal (I, J, K) to point straight up (0, 0, 1) in machine space
        # Machine Tool Vector T_m = R_C(C) * R_B(B) * [0, 0, 1]^T
        # Therefore, surface normal N = [I, J, K]^T relative to bed must be oriented
        # N = R_C(-C) * R_B(-B) * [0, 0, 1]^T (Wait, if the table rotates, the part rotates).
        # Actually, for a Trunnion table (table tilts B and rotates C),
        # Part orientation in machine coordinates: v_m = R_B(B) * R_C(C) * v_part
        # To align normal N=[I,J,K] to tool [0,0,1], N_m = R_B(B) * R_C(C) * N = [0,0,1]
        # This implies: N = R_C(-C) * R_B(-B) * [0,0,1]
        # N = [ sin(B)*cos(C), -sin(B)*sin(C), cos(B) ]
        # From this:
        # I = sin(B)cos(C)
        # J = -sin(B)sin(C)
        # K = cos(B)

        for wp in trajectory.waypoints:
            norm = np.sqrt(wp.i**2 + wp.j**2 + wp.k**2)
            if norm < 1e-6:
                i, j, k = 0.0, 0.0, 1.0
            else:
                i, j, k = wp.i / norm, wp.j / norm, wp.k / norm

            # B is tilt. From K = cos(B), B = acos(K)
            b_rad = np.arccos(np.clip(k, -1.0, 1.0))
            b_deg = np.degrees(b_rad)

            # C is rotation.
            # I = sin(B)cos(C) => cos(C) = I / sin(B)
            # J = -sin(B)sin(C) => sin(C) = -J / sin(B)
            # C = atan2(-J, I)

            # Singularity handling: if B is very small, C is arbitrary. We keep it as previous C.
            if abs(b_rad) < 1e-4:
                c_rad = 0.0 if not raw_c_angles else raw_c_angles[-1]
            else:
                c_rad = np.arctan2(-j, i)

            raw_c_angles.append(c_rad)
            b_angles.append(b_deg)
            valid_waypoints.append(wp)

        # Unwrap C angles
        unwrapped_c_rad = np.unwrap(raw_c_angles)
        c_angles = np.degrees(unwrapped_c_rad)

        # Second pass: Compute Machine XYZ taking into account pivot offsets and eccentricity
        for idx, wp in enumerate(valid_waypoints):
            b_deg = b_angles[idx]
            c_deg = c_angles[idx]

            # Clamp limits
            b_deg = max(b_min, min(b_max, b_deg))
            c_deg = max(c_min, min(c_max, c_deg))

            b_rad = np.radians(b_deg)
            c_rad = np.radians(c_deg)

            # Point on part: P = [x, y, z]
            P = np.array([wp.x, wp.y, wp.z])

            # Apply eccentricity
            P_e = P + np.array([ex, ey, 0.0])

            # Pivot offset: relative to the rotation center
            P_pivot = np.array([px, py, pz])

            # Vector from pivot to point
            V = P_e - P_pivot

            # Rotate by C (around Z) then B (around Y)
            cos_c, sin_c = np.cos(c_rad), np.sin(c_rad)
            R_C = np.array([
                [cos_c, -sin_c, 0],
                [sin_c,  cos_c, 0],
                [0,      0,     1]
            ])

            cos_b, sin_b = np.cos(b_rad), np.sin(b_rad)
            R_B = np.array([
                [ cos_b, 0, sin_b],
                [ 0,     1, 0    ],
                [-sin_b, 0, cos_b]
            ])

            # The part moves relative to the tool.
            # In a trunnion, the bed rotates C, then tilts B.
            V_rot = R_B @ (R_C @ V)

            # Machine coordinates: the rotated vector plus the pivot point
            # (assuming tool is stationary and table moves, or tool moves in XYZ and table rotates BC)
            # Typically, G-code XYZ commands the tool relative to the machine origin.
            P_mach = P_pivot + V_rot

            state = MachineStateVector(
                x=float(P_mach[0]),
                y=float(P_mach[1]),
                z=float(P_mach[2]),
                b=float(b_deg),
                c=float(c_deg),
                extrusion_volume=wp.extrusion_volume,
                feedrate=wp.feedrate,
                is_travel_move=wp.is_travel_move
            )
            states.append(state)

        return MachineTrajectory(states=states)
