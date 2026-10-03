import numpy as np
from src.common.schemas import CLDataTrajectory, MachineConfig, MachineStateVector, MachineTrajectory
from src.stage3_kinematics.base import BaseKinematicSolver
from src.stage3_kinematics.registry import KinematicsRegistry

@KinematicsRegistry.register("swivel_head_xyzbc")
class SwivelHeadXYZBCSolver(BaseKinematicSolver):
    """
    Kinematic solver for a 5-axis swivel head setup.
    Tool head tilts and rotates instead of the bed.
    """
    def solve(self, trajectory: CLDataTrajectory, config: MachineConfig) -> MachineTrajectory:
        states = []

        px, py, pz = config.pivot_offset_x, config.pivot_offset_y, config.pivot_offset_z
        ex, ey = config.eccentricity_x, config.eccentricity_y
        b_min, b_max = config.b_axis_min, config.b_axis_max
        c_min, c_max = config.c_axis_min, config.c_axis_max

        raw_c_angles = []
        b_angles = []
        valid_waypoints = []

        # Tool normal T_m = [I, J, K]
        # For a swivel head, the tool orientation is controlled by B (tilt) and C (rot)
        for wp in trajectory.waypoints:
            norm = np.sqrt(wp.i**2 + wp.j**2 + wp.k**2)
            if norm < 1e-6:
                i, j, k = 0.0, 0.0, 1.0
            else:
                i, j, k = wp.i / norm, wp.j / norm, wp.k / norm

            # For swivel head, the tool points downward. Assuming neutral is K=1.
            # Normal N = [ sin(B)cos(C), sin(B)sin(C), cos(B) ] (sign depends on kinematics)
            # Let's say tilt B is around Y, rot C around Z.
            # T = R_C(C) * R_B(B) * [0, 0, 1]^T
            # I = sin(B)cos(C)
            # J = sin(B)sin(C)
            # K = cos(B)

            b_rad = np.arccos(np.clip(k, -1.0, 1.0))
            b_deg = np.degrees(b_rad)

            if abs(b_rad) < 1e-4:
                c_rad = 0.0 if not raw_c_angles else raw_c_angles[-1]
            else:
                c_rad = np.arctan2(j, i)

            raw_c_angles.append(c_rad)
            b_angles.append(b_deg)
            valid_waypoints.append(wp)

        unwrapped_c_rad = np.unwrap(raw_c_angles)
        c_angles = np.degrees(unwrapped_c_rad)

        for idx, wp in enumerate(valid_waypoints):
            b_deg = max(b_min, min(b_max, b_angles[idx]))
            c_deg = max(c_min, min(c_max, c_angles[idx]))

            b_rad = np.radians(b_deg)
            c_rad = np.radians(c_deg)

            P = np.array([wp.x, wp.y, wp.z])

            # Swivel head tool center point offset calculation
            # Tool pivot offset vector
            P_pivot = np.array([px, py, pz])

            # Rotation matrices
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

            # TCP offset = R_C * R_B * -P_pivot + P_pivot
            # (Where P_pivot is distance from pivot to tool tip)
            offset = P_pivot - R_C @ (R_B @ P_pivot)

            P_mach = P + offset

            state = MachineStateVector(
                x=float(P_mach[0]), y=float(P_mach[1]), z=float(P_mach[2]),
                b=float(b_deg), c=float(c_deg),
                extrusion_volume=wp.extrusion_volume, feedrate=wp.feedrate, is_travel_move=wp.is_travel_move
            )
            states.append(state)

        return MachineTrajectory(states=states)

@KinematicsRegistry.register("planar_3axis_xyz")
class Cartesian3AxisSolver(BaseKinematicSolver):
    """
    Standard 3-axis Cartesian machine solver.
    Obeys standard XYZ moves, but logs warning if surface tilt exceeds machine limits.
    """
    def solve(self, trajectory: CLDataTrajectory, config: MachineConfig) -> MachineTrajectory:
        states = []
        import logging
        logger = logging.getLogger(__name__)

        warned = False
        max_tilt = config.max_3axis_tilt_deg

        for wp in trajectory.waypoints:
            # Check tilt warning
            norm = np.sqrt(wp.i**2 + wp.j**2 + wp.k**2)
            if norm > 1e-6:
                k = wp.k / norm
                b_rad = np.arccos(np.clip(k, -1.0, 1.0))
                b_deg = np.degrees(b_rad)
                if b_deg > max_tilt and not warned:
                    logger.warning(f"Toolpath surface normal tilt ({b_deg:.1f} deg) exceeds max_3axis_tilt_deg ({max_tilt} deg). Collision risk!")
                    warned = True

            state = MachineStateVector(
                x=wp.x, y=wp.y, z=wp.z,
                b=0.0, c=0.0,
                extrusion_volume=wp.extrusion_volume, feedrate=wp.feedrate, is_travel_move=wp.is_travel_move
            )
            states.append(state)

        return MachineTrajectory(states=states)
