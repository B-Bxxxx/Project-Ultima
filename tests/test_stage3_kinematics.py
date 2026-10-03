import pytest
import numpy as np
from src.common.schemas import CLDataTrajectory, CLDataWaypoint, MachineConfig
from src.stage3_kinematics.trunnion_solver import TrunnionXYZBCSolver

@pytest.fixture
def base_config():
    return MachineConfig(
        kinematic_chain="Tool -> Z -> Y -> X -> B -> C -> Bed",
        pivot_offset_x=0.0,
        pivot_offset_y=0.0,
        pivot_offset_z=100.0,
        eccentricity_x=0.0,
        eccentricity_y=0.0,
        b_axis_min=-90.0,
        b_axis_max=90.0,
        c_axis_min=-36000.0,
        c_axis_max=36000.0
    )

def test_flat_layer(base_config):
    # Test a simple vertical tool (normal = 0, 0, 1)
    # B and C should be 0
    solver = TrunnionXYZBCSolver()

    traj = CLDataTrajectory(waypoints=[
        CLDataWaypoint(x=10, y=20, z=30, i=0, j=0, k=1, feedrate=1000)
    ])

    result = solver.solve(traj, base_config)
    state = result.states[0]

    assert state.b == pytest.approx(0.0)
    assert state.c == pytest.approx(0.0)

    # Machine XYZ should match input XYZ because part doesn't rotate and there's no eccentricity
    # wait, if B=0, C=0, then V = (P - Pivot). R_B(0) R_C(0) V = V.
    # P_mach = Pivot + V = Pivot + P - Pivot = P.
    assert state.x == pytest.approx(10.0)
    assert state.y == pytest.approx(20.0)
    assert state.z == pytest.approx(30.0)

def test_tilted_surface(base_config):
    solver = TrunnionXYZBCSolver()

    # Normal is tilted 45 deg around X.
    # Normal vector N = (0, -sin(45), cos(45))
    i = 0.0
    j = -np.sin(np.radians(45))
    k = np.cos(np.radians(45))

    traj = CLDataTrajectory(waypoints=[
        CLDataWaypoint(x=0, y=0, z=100, i=i, j=j, k=k, feedrate=1000)
    ])

    result = solver.solve(traj, base_config)
    state = result.states[0]

    # Since normal tilts down Y, B axis (tilt around Y) must be used.
    # With the corrected trunnion derivation (I=-sinBcosC, J=sinBsinC, K=cosB)
    # J = sin(45)sin(C) -> -sin(45) = sin(45)sin(C) -> sin(C) = -1 -> C = -90 deg.
    assert state.b == pytest.approx(45.0)
    assert state.c == pytest.approx(-90.0)
