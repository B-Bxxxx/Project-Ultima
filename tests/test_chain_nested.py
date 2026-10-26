"""The nested ("head"/"table" + "carries") chain description: same maths as the list form and as the hand-written solvers."""
import os

import numpy as np
import pytest

from src.common.param_spec import resolve_values
from src.stage3_kinematics.chain import Chain, ChainError
from src.stage3_kinematics.machine_file import load_machine_file
from src.stage3_kinematics.registry import KinematicsRegistry
from tests.test_generic_chain import CLASSIC, EXPERIMENTAL, VARIANTS, poses, trajectory

NESTED = {"trunnion_table_xyzbc": "nested_trunnion_5axis.json", "swivel_head_xyzbc": "nested_swivel_head_5axis.json",
          "planar_3axis_xyz": "nested_cartesian_3axis.json"}
LISTED = {"trunnion_table_xyzbc": "chain_trunnion_5axis.json", "swivel_head_xyzbc": "chain_swivel_head_5axis.json",
          "planar_3axis_xyz": "chain_cartesian_3axis.json"}
MACHINES = os.path.join(os.path.dirname(__file__), "..", "config", "machines")


def cfgs(kin, variant):
    spec = load_machine_file(os.path.join(EXPERIMENTAL, NESTED[kin]))
    classic = load_machine_file(os.path.join(MACHINES, CLASSIC[kin]))
    over = {k: v for k, v in variant.items() if k in spec.parameters}
    old = classic.to_machine_config({k: v for k, v in variant.items() if k in classic.parameters})
    if kin == "swivel_head_xyzbc":
        old = old.model_copy(update={"kinematic_chain": kin})
    return old, spec.to_machine_config(over)


@pytest.mark.parametrize("kin", list(NESTED))
@pytest.mark.parametrize("variant", VARIANTS)
def test_nested_matches_hand_written_solver(kin, variant):
    if kin == "swivel_head_xyzbc":
        variant = {k: v for k, v in variant.items() if not k.startswith("eccentricity")}
    old, new = cfgs(kin, variant)
    tr = trajectory()
    a = KinematicsRegistry.get_solver(kin)().solve(tr, old)
    b = KinematicsRegistry.get_solver("generic_chain")().solve(tr, new)
    np.testing.assert_allclose(poses(b), poses(a), atol=2e-6, rtol=0)
    np.testing.assert_allclose([s.tool_alignment for s in b.states], [s.tool_alignment for s in a.states], atol=1e-6)


@pytest.mark.parametrize("kin", list(NESTED))
def test_nested_and_list_form_give_the_same_chain(kin):
    n = load_machine_file(os.path.join(EXPERIMENTAL, NESTED[kin]))
    l = load_machine_file(os.path.join(EXPERIMENTAL, LISTED[kin]))
    vals = resolve_values(n.parameters, {k: v for k, v in VARIANTS[2].items() if k in n.parameters})
    a = Chain.parse(n.chain, vals).to_dict()
    b = Chain.parse(l.chain, resolve_values(l.parameters, {k: v for k, v in VARIANTS[2].items() if k in l.parameters})).to_dict()
    assert a == b


@pytest.mark.parametrize("fn", ["nested_head_a_table_b.json", "nested_corexy_table_ab.json"])
def test_new_machines_hit_the_normal_and_invert(fn):
    spec = load_machine_file(os.path.join(EXPERIMENTAL, fn))
    cfg = spec.to_machine_config()
    solver = KinematicsRegistry.get_solver("generic_chain")()
    tr = trajectory()
    res = solver.solve(tr, cfg)
    S = np.array([[s.x, s.y, s.z, s.b, s.c, s.a] for s in res.states])
    ok = np.array([abs(s.tool_alignment - 1) < 1e-7 for s in res.states])
    assert ok.mean() > 0.9                                                 # +-90 deg limits: nearly all normals reachable
    P = np.array([[w.x, w.y, w.z] for w in tr.waypoints])
    np.testing.assert_allclose(solver.to_part_batch(cfg, S)[ok], P[ok], atol=1e-6)
    assert np.abs(S[:, 5]).max() > 1 and (np.abs(S[:, 3]).max() > 1)       # both rotary axes are used


def test_a_is_the_outer_axis_on_the_corexy_table():
    spec = load_machine_file(os.path.join(EXPERIMENTAL, "nested_corexy_table_ab.json"))
    ch = Chain.parse(spec.chain, resolve_values(spec.parameters))
    assert [j.axis for j in ch.table] == ["A", "B"] and [j.axis for j in ch.tool] == ["X", "Y", "Z"]


# ------------------------------------------------------------- the form itself
def test_missing_or_empty_sections_are_left_out():
    for raw in ({}, {"head": {}}, {"head": None, "table": {}}, {"table": None}):
        ch = Chain.parse(raw)
        assert not ch.rotaries() and [j.axis for j in ch.tool] == ["X", "Y", "Z"]


def test_linear_signs_and_limits_alias():
    ch = Chain.parse({"linear_signs": [1, -1, 1], "table": {"B": {"origin": [0, 0], "limits": [-30, 40]}}})
    assert [j.sign for j in ch.tool] == [1, -1, 1]
    assert (ch.table[0].lo, ch.table[0].hi) == (-30, 40)


def test_linear_axes_can_be_written_out():
    ch = Chain.parse({"table": {"Y": {"sign": -1, "carries": {"B": {"origin": [0, 0]}}}},
                      "head": {"X": {"carries": {"Z": {}}}}})
    assert [j.axis for j in ch.table] == ["Y", "B"]


@pytest.mark.parametrize("bad", [
    {"head": {"A": {}, "B": {}}},                                                    # siblings: not supported yet
    {"head": {"A": {"origin": [0, 0], "fixed": True}}},                              # unknown joint field
    {"head": {"A": {"limits": [0, 1], "min": 0}}},                                   # limits + min
    {"head": {"Q": {}}},                                                             # not an axis
    {"tool": [], "head": {}},                                                        # forms mixed
    {"head": {"X": {}}},                                                             # X written out without Y, Z
    {"table": {"A": {"carries": {"B": {"carries": {"C": {}}}}}}},                   # three rotaries
    {"linear_signs": [1, 1]},
])
def test_bad_nested_chain_is_rejected(bad):
    with pytest.raises(ChainError):
        Chain.parse(bad)
