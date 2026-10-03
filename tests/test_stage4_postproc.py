import pytest
from src.common.schemas import MachineTrajectory, MachineStateVector, MinimalPrintProfile
from src.stage4_postproc.klipper_postproc import Klipper5AxisPostProcessor
import math

def test_klipper_postproc():
    profile = MinimalPrintProfile(
        filament_diameter=1.75
    )

    trajectory = MachineTrajectory(states=[
        MachineStateVector(x=10, y=10, z=5, b=0, c=0, feedrate=3000, is_travel_move=True),
        MachineStateVector(x=20, y=10, z=5, b=45, c=90, extrusion_volume=1.0, feedrate=1500)
    ])

    postproc = Klipper5AxisPostProcessor()
    gcode = postproc.generate_gcode(trajectory, profile)

    # Assert boilerplate
    assert "M83 ; Relative extrusion mode" in gcode
    assert "G28 ; Home all axes" in gcode

    # Assert travel move (no E)
    assert "G0 X10.000 Y10.000 Z5.000 B0.000 C0.000 F3000" in gcode

    # Assert extrude move (with E)
    area = math.pi * ((1.75/2.0)**2)
    expected_e = 1.0 / area
    assert f"G1 X20.000 Y10.000 Z5.000 B45.000 C90.000 E{expected_e:.5f} F1500" in gcode
