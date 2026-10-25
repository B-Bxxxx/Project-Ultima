"""
Parity tests of the parallel `generic_chain` solver: a mathematical chain description of a machine must give the same
G-code poses as the hand-written solver of that machine (which stay untouched), and its inverse must match.
"""
import json
import os

import numpy as np
import pytest

from src.common.schemas import CLDataTrajectory, CLDataWaypoint
from src.stage3_kinematics.chain import Chain, ChainError
from src.stage3_kinematics.machine_file import MachineFileError, MachineSpec, load_machine_file
from src.stage3_kinematics.refine import solve_refined
from src.stage3_kinematics.registry import KinematicsRegistry

MACHINES = os.path.join(os.path.dirname(__file__), "..", "config", "machines")
EXPERIMENTAL = os.path.join(os.path.dirname(__file__), "..", "config", "machines_experimental")
PAIRS = {"trunnion_table_xyzbc": "chain_trunnion_5axis.json", "swivel_head_xyzbc": "chain_swivel_head_5axis.json",
         "planar_3axis_xyz": "chain_cartesian_3axis.json"}
CLASSIC = {"trunnion_table_xyzbc": "trunnion_5axis.json", "planar_3axis_xyz": "cartesian_3axis.json", "swivel_head_xyzbc": "trunnion_5axis.json"}

VARIANTS = [
    {},
    dict(c_axis_x=175.0, c_axis_y=82.0, c_axis_z=12.0, pivot_offset_z=33.0, pivot_offset_x=4.0, pivot_offset_y=-2.0),
    dict(c_axis_x=60.0, c_axis_y=40.0, eccentricity_x=3.0, eccentricity_y=-1.5, pivot_offset_z=25.0, c_axis_dir=-1.0),
    dict(b_axis_dir=-1.0, b_axis_min=-90.0, b_axis_max=90.0, c_axis_min=-360.0, c_axis_max=360.0, c_axis_dir=-1.0, pivot_offset_z=10.0),
]


def wp(x, y, z, n, travel=False, layer=0, e=0.1):
    n = np.array(n, float)
    n = n / np.linalg.norm(n)
    return CLDataWaypoint(x=x, y=y, z=z, i=n[0], j=n[1], k=n[2], extrusion_volume=0.0 if travel else e, feedrate=1000.0,
                          is_travel_move=travel, feature_type="wall", layer_index=layer)


def trajectory():
    """Vertical, tilted in all directions, a cone-like spiral with travels, nearly vertical, downward."""
    rng = np.random.default_rng(7)
    w = [wp(0, 0, 5, [0, 0, 1], travel=True)]
    for t in np.linspace(0, 4 * np.pi, 60):                                   # extruding spiral: C winds a lot
        w.append(wp(15 * np.cos(t), 15 * np.sin(t), 5 + 0.4 * t, [0.5 * np.cos(t), 0.5 * np.sin(t), 0.8]))
    w.append(wp(1, 1, 30, [0.001, 0.0005, 1], travel=True))
    for k in range(40):                                                       # random tilts up to 55 deg, travel every 5th
        a, b = rng.uniform(0, 2 * np.pi), np.radians(rng.uniform(0.2, 55))
        w.append(wp(*rng.uniform(-20, 20, 2), 6 + k * 0.1, [np.sin(b) * np.cos(a), np.sin(b) * np.sin(a), np.cos(b)], travel=(k % 5 == 0)))
    w.append(wp(5, 5, 3, [0, 0, 1], travel=True))
    return CLDataTrajectory(waypoints=w)


def configs(kin, variant):
    spec_c = load_machine_file(os.path.join(EXPERIMENTAL, PAIRS[kin]))
    classic = load_machine_file(os.path.join(MACHINES, CLASSIC[kin]))
    over = {k: v for k, v in variant.items() if k in spec_c.parameters}
    cfg_chain = spec_c.to_machine_config(over)
    over2 = {k: v for k, v in variant.items() if k in classic.parameters}
    cfg_old = classic.to_machine_config(over2)
    if kin == "swivel_head_xyzbc":                       # the classic trunnion file's limits, solver swapped
        cfg_old = cfg_old.model_copy(update={"kinematic_chain": kin})
        for k in ("eccentricity_x", "eccentricity_y"):
            assert getattr(cfg_old, k) == 0.0 or k not in over2
    return cfg_old, cfg_chain


def poses(res):
    return np.array([[s.x, s.y, s.z, s.b, s.c] for s in res.states])


@pytest.mark.parametrize("kin", list(PAIRS))
@pytest.mark.parametrize("variant", VARIANTS)
def test_chain_matches_hand_written_solver(kin, variant):
    if kin == "swivel_head_xyzbc":
        variant = {k: v for k, v in variant.items() if not k.startswith("eccentricity")}
    old, new = configs(kin, variant)
    tr = trajectory()
    a = KinematicsRegistry.get_solver(kin)().solve(tr, old)
    b = KinematicsRegistry.get_solver("generic_chain")().solve(tr, new)
    assert len(a.states) == len(b.states)
    np.testing.assert_allclose(poses(b), poses(a), atol=2e-6, rtol=0)
    np.testing.assert_allclose([s.tool_alignment for s in b.states], [s.tool_alignment for s in a.states], atol=1e-6)
    assert [s.is_travel_move for s in b.states] == [s.is_travel_move for s in a.states]


@pytest.mark.parametrize("kin", ["trunnion_table_xyzbc", "swivel_head_xyzbc"])
def test_inverse_matches(kin):
    old, new = configs(kin, VARIANTS[1] if kin != "swivel_head_xyzbc" else {k: v for k, v in VARIANTS[1].items()})
    tr = trajectory()
    S = poses(KinematicsRegistry.get_solver("generic_chain")().solve(tr, new))
    P = np.array([[w.x, w.y, w.z] for w in tr.waypoints])
    inv = KinematicsRegistry.get_solver("generic_chain")().to_part_batch(new, S)
    inv_old = KinematicsRegistry.get_solver(kin)().to_part_batch(old, S)
    np.testing.assert_allclose(inv, P, atol=1e-6)               # solve -> inverse returns the part point
    np.testing.assert_allclose(inv, inv_old, atol=1e-6)         # and agrees with the hand-written inverse for any pose


def test_refine_gives_same_as_classic():
    old, new = configs("trunnion_table_xyzbc", VARIANTS[1])
    tr = trajectory()
    a = solve_refined(KinematicsRegistry.get_solver("trunnion_table_xyzbc")(), tr, old)
    b = solve_refined(KinematicsRegistry.get_solver("generic_chain")(), tr, new)
    assert len(a.states) == len(b.states)
    np.testing.assert_allclose(poses(b), poses(a), atol=2e-6)


def test_three_axis_has_no_inverse_so_never_refined():
    _, new = configs("planar_3axis_xyz", {})
    assert KinematicsRegistry.get_solver("generic_chain")().to_part_batch(new, np.zeros((1, 5))) is None


# ------------------------------------------------------------- what the hand-written solvers cannot do
def test_mixed_head_and_table_rotaries_roundtrip():
    """Tilt on the tool, rotation on the table (a different machine type): every reachable normal is hit exactly."""
    chain = {"tool": [{"axis": "X", "type": "linear", "direction": [1, 0, 0]}, {"axis": "Y", "type": "linear", "direction": [0, 1, 0]},
                      {"axis": "Z", "type": "linear", "direction": [0, 0, 1]},
                      {"axis": "B", "type": "rotary", "direction": [0, 1, 0], "point": [0, 0, -30], "min": -90, "max": 90}],
             "table": [{"axis": "C", "type": "rotary", "direction": [0, 0, 1], "point": [100, 100, 0], "continuous": True}],
             "part_origin": [100, 100, 0]}
    from src.common.schemas import MachineConfig
    cfg = MachineConfig(kinematic_chain="generic_chain", b_axis_min=-90, b_axis_max=90, c_axis_min=-1e6, c_axis_max=1e6,
                        chain=Chain.parse(chain).to_dict())
    solver = KinematicsRegistry.get_solver("generic_chain")()
    tr = trajectory()
    res = solver.solve(tr, cfg)
    assert all(abs(s.tool_alignment - 1.0) < 1e-9 for s in res.states)          # nozzle axis lies along every layer normal
    P = np.array([[w.x, w.y, w.z] for w in tr.waypoints])
    np.testing.assert_allclose(solver.to_part_batch(cfg, poses(res)), P, atol=1e-6)


def test_table_carries_linear_axis():
    """Y moves the bed (which carries B and C) instead of the nozzle. With sign -1 (bed +Y = nozzle -Y relative to the part)
    the machine's Y value - hence the whole G-code - is the same as for a nozzle that moves in Y."""
    from src.common.param_spec import resolve_values
    spec = load_machine_file(os.path.join(EXPERIMENTAL, "chain_trunnion_5axis.json"))
    over = {"c_axis_y": 50.0, "pivot_offset_z": 20.0}
    vals = resolve_values(spec.parameters, over)
    d = json.loads(json.dumps(spec.chain))
    d["tool"] = [j for j in d["tool"] if j["axis"] != "Y"]
    d["table"].insert(0, {"axis": "Y", "type": "linear", "direction": [0, 1, 0], "sign": -1})
    cfg = spec.to_machine_config(over)
    cfg2 = cfg.model_copy(update={"chain": Chain.parse(d, vals).to_dict()})
    tr = trajectory()
    solver = KinematicsRegistry.get_solver("generic_chain")()
    a, b = poses(solver.solve(tr, cfg)), poses(solver.solve(tr, cfg2))
    np.testing.assert_allclose(b, a, atol=1e-6)
    np.testing.assert_allclose(solver.to_part_batch(cfg2, b), np.array([[w.x, w.y, w.z] for w in tr.waypoints]), atol=1e-6)


def test_single_rotary_best_fit():
    d = {"tool": [{"axis": a, "type": "linear", "direction": v} for a, v in (("X", [1, 0, 0]), ("Y", [0, 1, 0]), ("Z", [0, 0, 1]))],
         "table": [{"axis": "B", "type": "rotary", "direction": [0, 1, 0], "min": -45, "max": 45}]}
    from src.common.schemas import MachineConfig
    cfg = MachineConfig(kinematic_chain="generic_chain", b_axis_min=-45, b_axis_max=45, c_axis_min=0, c_axis_max=0, chain=Chain.parse(d).to_dict())
    res = KinematicsRegistry.get_solver("generic_chain")().solve(CLDataTrajectory(waypoints=[wp(0, 0, 0, [0, 0, 1]), wp(1, 0, 0, [-np.sin(0.3), 0, np.cos(0.3)]),
                                                                                          wp(1, 0, 0, [-np.sin(1.2), 0, np.cos(1.2)])]), cfg)
    assert [round(s.b, 6) for s in res.states[:2]] == [0.0, round(np.degrees(0.3), 6)]
    assert res.states[2].b == pytest.approx(45.0)                                   # beyond the limit: clamped
    assert all(s.c == 0.0 for s in res.states)


# ------------------------------------------------------------- validation
def _chain(**kw):
    lin = [{"axis": a, "type": "linear", "direction": v} for a, v in (("X", [1, 0, 0]), ("Y", [0, 1, 0]), ("Z", [0, 0, 1]))]
    base = {"tool": lin, "table": []}
    base.update(kw)
    return base


@pytest.mark.parametrize("bad", [
    _chain(table=[{"axis": "A", "type": "rotary", "direction": [0, 1, 0]}]),                                      # A turns about X
    _chain(table=[{"axis": "B", "type": "rotary", "direction": [0, 0, 1]}]),                                      # B turns about Y
    _chain(table=[{"axis": "B", "type": "rotary", "direction": [0, 0, 0]}]),                                      # zero vector
    _chain(part_origin=["nope", 0, 0]),                                                                           # unknown parameter
    _chain(part_origin=["__import__('os')", 0, 0]),                                                               # no code
    _chain(table=[{"axis": "B", "type": "rotary", "direction": [0, 1, 0], "min": 5, "max": 1}]),
    {"tool": [{"axis": "X", "type": "linear", "direction": [1, 0, 0]}], "table": []},                             # Y, Z missing
    _chain(bogus=1),
])
def test_bad_chain_is_rejected(bad):
    with pytest.raises(ChainError):
        Chain.parse(bad)


def test_machine_file_rules():
    spec = load_machine_file(os.path.join(EXPERIMENTAL, "chain_trunnion_5axis.json"))
    d = spec.to_dict()
    d2 = json.loads(json.dumps(d))
    d2["axes"] = ["X", "Y", "Z"]                                                  # axes must match the chain
    with pytest.raises(MachineFileError):
        MachineSpec.from_dict(d2)
    d3 = json.loads(json.dumps(d))
    d3["kinematics"] = "trunnion_table_xyzbc"                                     # a chain belongs to generic_chain only
    with pytest.raises(MachineFileError):
        MachineSpec.from_dict(d3)
    d4 = json.loads(json.dumps(d))
    del d4["chain"]
    with pytest.raises(MachineFileError):
        MachineSpec.from_dict(d4)
    assert MachineSpec.from_dict(d).to_dict() == d                                 # round trip
