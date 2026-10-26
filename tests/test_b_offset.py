"""B-axis offset: machine B = b_axis_dir * program B + b_axis_offset_deg (a B homed at its -90 deg end stop: offset 90, limits 0..180)."""
import os

import numpy as np
import pytest

from src.stage3_kinematics.machine_file import load_machine_file
from src.stage3_kinematics.registry import KinematicsRegistry
from tests.test_generic_chain import EXPERIMENTAL, MACHINES, PAIRS, trajectory

OFF = dict(b_axis_offset_deg=90.0, b_axis_min=0.0, b_axis_max=180.0)
BASE = dict(c_axis_x=175.0, c_axis_y=82.0, pivot_offset_z=12.0, vertical_dead_zone_deg=0.0)


def cfg(path, **over):
    spec = load_machine_file(path)
    return spec.to_machine_config({k: v for k, v in over.items() if k in spec.parameters})


def states(kin, c):
    return KinematicsRegistry.get_solver(kin)().solve(trajectory(), c).states


@pytest.mark.parametrize("kin", ["trunnion_table_xyzbc", "swivel_head_xyzbc", "generic_chain"])
@pytest.mark.parametrize("dir_", [1.0, -1.0])
def test_offset_shifts_only_b(kin, dir_):
    f = os.path.join(EXPERIMENTAL, "chain_swivel_head_5axis.json") if kin == "swivel_head_xyzbc" else (
        os.path.join(EXPERIMENTAL, "chain_trunnion_5axis.json") if kin == "generic_chain" else os.path.join(MACHINES, "trunnion_5axis.json"))
    ref = cfg(f, b_axis_dir=dir_, **BASE)
    sh = cfg(f, b_axis_dir=dir_, **BASE, **{**OFF, "b_axis_min": -90.0 + 90.0, "b_axis_max": 180.0})
    if kin == "swivel_head_xyzbc":
        ref, sh = ref.model_copy(update={"kinematic_chain": kin}), sh.model_copy(update={"kinematic_chain": kin})
    a, b = states(kin, ref), states(kin, sh)
    for p, q in zip(a, b):
        assert q.b == pytest.approx(p.b + 90.0, abs=1e-6)
        assert (q.x, q.y, q.z, q.c) == pytest.approx((p.x, p.y, p.z, p.c), abs=1e-6)
        assert 0.0 - 1e-9 <= q.b <= 180.0 + 1e-9


def test_limits_are_machine_values_and_inverse_round_trips():
    f = os.path.join(MACHINES, "trunnion_5axis.json")
    c = cfg(f, **BASE, **OFF)
    sol = KinematicsRegistry.get_solver("trunnion_table_xyzbc")()
    st = sol.solve(trajectory(), c).states
    assert min(s.b for s in st) >= -1e-9 and max(s.b for s in st) <= 180.0 + 1e-9
    assert min(s.b for s in st) < 89.0                     # the negative program B (crest crossing) is reachable
    P = np.array([[s.x, s.y, s.z, s.b, s.c] for s in st])
    ref = sol.to_part_batch(cfg(f, **BASE), P - np.array([0, 0, 0, 90.0, 0]))
    np.testing.assert_allclose(sol.to_part_batch(c, P), ref, atol=1e-9)


def test_chain_inverse_and_end_home():
    f = os.path.join(EXPERIMENTAL, "chain_trunnion_5axis.json")
    c = cfg(f, **BASE, **OFF)
    sol = KinematicsRegistry.get_solver("generic_chain")()
    st = sol.solve(trajectory(), c).states
    P = np.array([[s.x, s.y, s.z, s.b, s.c] for s in st])
    old = cfg(os.path.join(MACHINES, "trunnion_5axis.json"), **BASE, **OFF)
    ref = KinematicsRegistry.get_solver("trunnion_table_xyzbc")().to_part_batch(old, P)
    np.testing.assert_allclose(sol.to_part_batch(c, P), ref, atol=1e-6)
    spec = load_machine_file(f)
    assert spec.rotary_home(c) == {"B": 90.0}
    assert load_machine_file(os.path.join(MACHINES, "trunnion_5axis.json")).rotary_home(old) == {"B": 90.0}


def test_end_block_goes_to_the_level_value():
    from src.common.schemas import MinimalPrintProfile
    from src.stage4_postproc.klipper_postproc import Klipper5AxisPostProcessor
    from src.common.schemas import MachineTrajectory
    c = cfg(os.path.join(MACHINES, "trunnion_5axis.json"), **BASE, **OFF)
    tr = KinematicsRegistry.get_solver("trunnion_table_xyzbc")().solve(trajectory(), c)
    prof = MinimalPrintProfile(rotary_home={"B": 90.0})
    g = Klipper5AxisPostProcessor().generate_gcode(tr, prof)
    assert "B90 C0" in g
    g0 = Klipper5AxisPostProcessor().generate_gcode(tr, MinimalPrintProfile())
    assert "B0 C0" in g0
