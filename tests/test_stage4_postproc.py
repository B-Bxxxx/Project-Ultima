"""
Stage 4 tests. Every assertion works on PARSED G-code (see gcode_tools.py): order of moves,
axes and E values; nothing compares against the template text.
"""
import json
import math

import pytest
import trimesh
from pydantic import ValidationError

from src.common.schemas import MachineConfig, MachineStateVector, MachineTrajectory, MinimalPrintProfile
from src.stage1_slicer.plugins.field_slicer import UniversalFieldSlicerPlugin
from src.stage2_toolpath.contour_generator import StandardToolpathGenerator
from src.stage3_kinematics.registry import KinematicsRegistry
from src.stage4_postproc.klipper_postproc import Klipper5AxisPostProcessor, split_gcode_by_layer
from tests.gcode_tools import (
    classify, layer_markers, moves, parse, travel_runs, type_markers,
)

AREA = math.pi * (1.75 / 2.0) ** 2


def _state(x, y, z, layer, feature="outer_wall", travel=False, vol=1.0, feed=None, b=0.0, c=0.0):
    return MachineStateVector(x=x, y=y, z=z, b=b, c=c, layer_index=layer, feature_type=feature,
                              is_travel_move=travel, extrusion_volume=0.0 if travel else vol,
                              feedrate=feed if feed is not None else (3000.0 if travel else 1500.0))


def _synthetic():
    return MachineTrajectory(states=[
        _state(0, 0, 0.2, 0, travel=True),                      # first move of the print
        _state(10, 0, 0.2, 0),
        _state(10, 10, 0.2, 0),
        _state(10.5, 10, 0.2, 0, feature="infill", travel=True),   # 0.5 mm: short travel
        _state(10.5, 20, 0.2, 0, feature="infill"),
        _state(0, 0, 0.4, 1, travel=True),                      # long travel + layer change
        _state(10, 0, 0.4, 1),
    ])


def _gcode(trajectory=None, **profile_kw):
    profile = MinimalPrintProfile(include_rotary_axes=False, **profile_kw)
    return Klipper5AxisPostProcessor().generate_gcode(trajectory or _synthetic(), profile), profile


def _body(cmds):
    """Commands between the start block and the end block."""
    start = next(i for i, c in enumerate(cmds) if c.code == "" and c.comment.startswith("LAYER:"))
    end = next(i for i, c in enumerate(cmds) if c.code == "" and c.comment == "End G-Code")
    return cmds[start:end]


# ---------------------------------------------------------------------------------------------
# boilerplate
# ---------------------------------------------------------------------------------------------
def test_boilerplate_and_no_heater_commands_before_the_print():
    gcode, _ = _gcode()
    cmds = parse(gcode)
    codes = [c.code for c in cmds]
    assert "M83" in codes and "G90" in codes and "G28" in codes
    first_layer = next(i for i, c in enumerate(cmds) if c.comment.startswith("LAYER:0"))
    before = {c.code for c in cmds[:first_layer]}
    assert not before & {"M104", "M109", "M140", "M190"}          # temperatures come from the printer config
    assert [c.code for c in cmds[-3:]] == ["M104", "M140", "M84"]  # heaters off at the end
    assert layer_markers(cmds) == [0, 1]
    assert any(c.comment == "LAYER_COUNT:2" for c in cmds)


def test_extrusion_value_is_volume_over_filament_area_and_uses_the_state_feedrate():
    gcode, _ = _gcode()
    extr = [c for c in moves(parse(gcode)) if classify(c) == "extrude"]
    assert len(extr) == 4
    for c in extr:
        assert c.params["E"] == pytest.approx(1.0 / AREA, abs=1e-5)
        assert c.params["F"] == 1500


# ---------------------------------------------------------------------------------------------
# travel sequence
# ---------------------------------------------------------------------------------------------
def test_long_travel_is_retract_lift_travel_lower_prime():
    gcode, p = _gcode(retract_distance=0.8, z_hop=0.4, retract_speed=2000, z_speed=900)
    cmds = _body(parse(gcode))
    runs = travel_runs(cmds)
    long_run = runs[-1]                                    # (10.5,20,0.2) -> (0,0,0.4)
    assert [classify(c) for c in long_run] == ["retract", "zmove", "travel", "zmove", "prime"]
    retract, up, travel, down, prime = long_run
    assert retract.params["E"] == pytest.approx(-0.8) and retract.params["F"] == 2000
    # lift above the HIGHER of both ends: max(0.2, 0.4) + 0.4
    assert up.params["Z"] == pytest.approx(0.8) and up.params["F"] == 900
    assert "Z" not in travel.params and travel.params["X"] == 0 and travel.params["Y"] == 0
    assert travel.params["F"] == 3000                     # the waypoint feedrate
    assert down.params["Z"] == pytest.approx(0.4) and down.params["F"] == 900
    assert prime.params["E"] == pytest.approx(0.8) and prime.params["F"] == 2000


def test_short_travel_is_made_directly():
    gcode, _ = _gcode(min_retract_travel=1.5)
    short = travel_runs(_body(parse(gcode)))[1]            # (10,10,0.2) -> (10.5,10,0.2)
    assert [classify(c) for c in short] == ["travel"]
    assert short[0].params["Z"] == pytest.approx(0.2)      # no hop, so Z is part of the move

    gcode, _ = _gcode(min_retract_travel=1.5, hop_short_travels=True)   # hop, but still no retract
    short = travel_runs(_body(parse(gcode)))[1]
    assert [classify(c) for c in short] == ["zmove", "travel", "zmove"]
    assert short[0].params["Z"] == pytest.approx(0.2 + 0.4)

    gcode, _ = _gcode(min_retract_travel=0.1)              # now the same travel is long enough
    assert [classify(c) for c in travel_runs(_body(parse(gcode)))[1]] == ["retract", "zmove", "travel", "zmove", "prime"]


def test_first_move_lifts_without_retracting():
    gcode, _ = _gcode(z_hop=0.5)
    first = travel_runs(_body(parse(gcode)))[0]
    assert [classify(c) for c in first] == ["zmove", "travel", "zmove"]
    assert first[0].params["Z"] == pytest.approx(0.2 + 0.5)
    assert first[2].params["Z"] == pytest.approx(0.2)


def test_travel_never_moves_xy_while_extruding_or_retracted_state_is_unbalanced():
    """Retract and prime always come in pairs and no E motion happens while hopped."""
    gcode, _ = _gcode()
    z = 0.0
    hopped = False
    retracted = 0.0
    for c in moves(parse(gcode)):
        kind = classify(c)
        if "Z" in c.params:
            new_z = c.params["Z"]
            hopped = kind == "zmove" and new_z > z
            z = new_z
        if hopped:
            assert "E" not in c.params                       # no E motion while Z is hopped
        if kind == "retract":
            retracted += c.params["E"]
        elif kind == "prime":
            retracted += c.params["E"]
        elif kind == "extrude":
            assert retracted == pytest.approx(0.0, abs=1e-9)  # fully primed before extruding
    assert retracted == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("command", ["G0", "G1"])
def test_travel_and_hop_use_the_configured_command(command):
    gcode, _ = _gcode(travel_command=command)
    cmds = parse(gcode)
    for c in moves(cmds):
        kind = classify(c)
        if kind in ("travel", "zmove"):
            assert c.code == command, c.raw
        elif kind in ("retract", "prime", "extrude"):
            assert c.code == "G1", c.raw                      # E moves are always G1
    assert command in {c.code for c in moves(cmds) if classify(c) == "travel"}


def test_zero_hop_and_zero_retraction_disable_those_moves():
    gcode, _ = _gcode(z_hop=0.0, retract_distance=0.0)
    body = _body(parse(gcode))
    assert {classify(c) for c in moves(body)} <= {"travel", "extrude"}
    gcode, _ = _gcode(z_hop=0.4, retract_distance=0.0)
    assert {classify(c) for c in moves(_body(parse(gcode)))} == {"zmove", "travel", "extrude"}


def test_rotary_axes_only_when_enabled():
    traj = MachineTrajectory(states=[
        _state(0, 0, 0.2, 0, travel=True, b=10, c=20), _state(5, 0, 0.2, 0, b=12, c=25)])
    five = Klipper5AxisPostProcessor().generate_gcode(traj, MinimalPrintProfile(include_rotary_axes=True))
    three = Klipper5AxisPostProcessor().generate_gcode(traj, MinimalPrintProfile(include_rotary_axes=False))
    ext5 = [c for c in moves(parse(five)) if classify(c) == "extrude"][0]
    assert ext5.params["B"] == pytest.approx(12) and ext5.params["C"] == pytest.approx(25)
    assert all("B" not in c.params and "C" not in c.params for c in moves(parse(three)))


# ---------------------------------------------------------------------------------------------
# markers
# ---------------------------------------------------------------------------------------------
def test_layer_and_type_markers_on_a_synthetic_print():
    gcode, _ = _gcode()
    cmds = parse(gcode)
    assert layer_markers(cmds) == [0, 1]
    # outer_wall, then infill (layer 0); layer 1 announces its feature again although it is
    # the same as the previous layer's last... (infill -> outer_wall is a change anyway)
    assert type_markers(cmds) == ["outer_wall", "infill", "outer_wall"]

    # the layer marker comes BEFORE the travel into the layer
    body = _body(cmds)
    idx = next(i for i, c in enumerate(body) if c.comment == "LAYER:1")
    nxt = next(c for c in body[idx + 1:] if c.is_move)
    assert classify(nxt) == "retract"


def test_type_is_re_announced_in_every_layer_even_if_unchanged():
    traj = MachineTrajectory(states=[
        _state(0, 0, 0.2, 0, travel=True), _state(5, 0, 0.2, 0),
        _state(0, 0, 0.4, 1, travel=True), _state(5, 0, 0.4, 1),
        _state(5, 5, 0.4, 1)])
    gcode = Klipper5AxisPostProcessor().generate_gcode(traj, MinimalPrintProfile(include_rotary_axes=False))
    assert type_markers(parse(gcode)) == ["outer_wall", "outer_wall"]      # once per layer, not per line


# ---------------------------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pipeline_print():
    mesh = trimesh.creation.box(extents=(20, 20, 4))
    mesh.apply_translation([0, 0, 2])
    model = UniversalFieldSlicerPlugin().slice(mesh, {"strategy": "planar", "layer_height": 1.0})
    profile = MinimalPrintProfile(layer_height=1.0, nozzle_diameter=0.4, include_rotary_axes=False,
                                  top_solid_layers=1, bottom_solid_layers=1,
                                  retract_distance=0.8, z_hop=0.4, min_retract_travel=1.5)
    traj = StandardToolpathGenerator().generate_toolpath(model, profile)
    config = MachineConfig(kinematic_chain="x", b_axis_min=0, b_axis_max=0, c_axis_min=0, c_axis_max=0)
    states = KinematicsRegistry.get_solver("planar_3axis_xyz")().solve(traj, config)
    gcode = Klipper5AxisPostProcessor().generate_gcode(states, profile)
    return {"profile": profile, "states": states, "gcode": gcode, "cmds": parse(gcode)}


def test_every_travel_in_a_real_print_follows_the_pattern(pipeline_print):
    cmds = _body(pipeline_print["cmds"])
    runs = travel_runs(cmds)
    assert len(runs) > 8
    long_pattern = ["retract", "zmove", "travel", "zmove", "prime"]
    z = 0.0
    n_long = n_short = 0
    pos = {"x": 0.0, "y": 0.0, "z": 0.0}
    for c in moves(cmds):
        pass
    for run in runs:
        kinds = [classify(c) for c in run]
        assert kinds in (long_pattern, ["travel"], ["zmove", "travel", "zmove"]), kinds
        if kinds == long_pattern:
            n_long += 1
            up, down = run[1], run[3]
            assert up.params["Z"] > down.params["Z"] + 0.4 - 1e-9 or up.params["Z"] >= down.params["Z"] + 0.4 - 1e-9
            assert up.params["Z"] - down.params["Z"] == pytest.approx(0.4) or up.params["Z"] > down.params["Z"]
        elif kinds == ["travel"]:
            n_short += 1
    assert n_long > 0 and n_short > 0                      # both cases occur in a real print


def test_layer_markers_match_the_layers_of_the_extrusions(pipeline_print):
    cmds = pipeline_print["cmds"]
    states = pipeline_print["states"].states
    markers = layer_markers(cmds)
    assert markers == sorted(set(s.layer_index for s in states))
    assert any(c.comment == f"LAYER_COUNT:{len(markers)}" for c in cmds)

    current = None
    parsed_layers = []
    for c in cmds:
        if c.code == "" and c.comment.startswith("LAYER:") and not c.comment.startswith("LAYER_COUNT"):
            current = int(c.comment.split(":")[1])
        elif c.is_move and classify(c) == "extrude":
            parsed_layers.append(current)
    assert parsed_layers == [s.layer_index for s in states if not s.is_travel_move]


def test_type_markers_appear_only_when_layer_or_feature_changes(pipeline_print):
    states = [s for s in pipeline_print["states"].states if not s.is_travel_move]
    expected = []
    last = (None, None)
    for s in states:
        key = (s.layer_index, s.feature_type)
        if key != last:
            expected.append(s.feature_type)
            last = key
    assert type_markers(pipeline_print["cmds"]) == expected
    n_extrusions = len(states)
    assert len(expected) < n_extrusions / 10               # NOT once per extruding line


def test_total_extruded_filament_equals_total_volume(pipeline_print):
    cmds = pipeline_print["cmds"]
    e_total = sum(c.params["E"] for c in moves(cmds) if classify(c) == "extrude")
    volume = sum(s.extrusion_volume for s in pipeline_print["states"].states)
    assert e_total == pytest.approx(volume / AREA, rel=1e-4)      # 5-decimal rounding per line


def test_split_by_layer_keeps_every_line(pipeline_print):
    gcode = pipeline_print["gcode"]
    parts = split_gcode_by_layer(gcode)
    assert parts.join() == gcode
    assert sorted(parts.layers) == layer_markers(pipeline_print["cmds"])
    for idx, text in parts.layers.items():
        assert text.splitlines()[0] == f";LAYER:{idx}"
        assert all(f";LAYER:{other}" not in text for other in parts.layers if other != idx)
    assert "End G-Code" in parts.footer and "LAYER" not in parts.footer
    assert "G28" in parts.header

    via_postproc = Klipper5AxisPostProcessor().generate_gcode_layers(pipeline_print["states"], pipeline_print["profile"])
    assert via_postproc.layers == parts.layers


# ---------------------------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------------------------
def test_profile_density_is_a_validated_fraction():
    assert MinimalPrintProfile(infill_density=1.0).infill_density == 1.0
    for bad in (15.0, 1.5, -0.1):
        with pytest.raises(ValidationError):
            MinimalPrintProfile(infill_density=bad)


def test_wall_count_is_an_alias_of_num_perimeters():
    assert MinimalPrintProfile(wall_count=4).num_perimeters == 4
    assert MinimalPrintProfile(num_perimeters=3).wall_count == 3
    assert MinimalPrintProfile().num_perimeters == 2


def test_shipped_print_profile_loads_and_is_consistent():
    with open("config/print_profile.json") as f:
        data = json.load(f)
    profile = MinimalPrintProfile(**data)
    assert 0.0 <= profile.infill_density <= 1.0
    assert profile.travel_command in ("G0", "G1")
    # every key of the shipped file is a real profile field (no silently ignored settings)
    known = set(MinimalPrintProfile.model_fields) | {"wall_count"}
    assert set(data) <= known


def test_invalid_travel_command_and_speeds_are_rejected():
    with pytest.raises(ValidationError):
        MinimalPrintProfile(travel_command="G2")
    with pytest.raises(ValidationError):
        MinimalPrintProfile(travel_speed=0)


# ---------------------------------------------------------------------------------------------
# retract and un-retract are separate parameters
# ---------------------------------------------------------------------------------------------
def test_unretract_defaults_to_the_retract_values():
    p = MinimalPrintProfile(retract_distance=1.2, retract_speed=1800)
    assert p.unretract_distance == pytest.approx(1.2) and p.unretract_speed == pytest.approx(1800)
    p = MinimalPrintProfile(retract_distance=1.2, unretract_distance=0.0, unretract_speed=900)
    assert p.unretract_distance == 0.0 and p.unretract_speed == 900


def test_unretract_distance_and_speed_are_independent_of_retract():
    gcode, _ = _gcode(retract_distance=0.8, retract_speed=2000, unretract_distance=1.1, unretract_speed=700)
    long_run = travel_runs(_body(parse(gcode)))[-1]
    kinds = [classify(c) for c in long_run]
    assert kinds == ["retract", "zmove", "travel", "zmove", "prime"]
    retract, prime = long_run[0], long_run[-1]
    assert retract.params["E"] == pytest.approx(-0.8) and retract.params["F"] == 2000
    assert prime.params["E"] == pytest.approx(1.1) and prime.params["F"] == 700
    # the prime is the very last move before printing resumes
    cmds = [c for c in parse(gcode) if c.is_move]
    i = cmds.index(prime)
    assert classify(cmds[i + 1]) == "extrude"


def test_zero_unretract_distance_skips_only_the_prime_move():
    gcode, _ = _gcode(retract_distance=0.8, unretract_distance=0.0)
    assert [classify(c) for c in travel_runs(_body(parse(gcode)))[-1]] == ["retract", "zmove", "travel", "zmove"]


def test_negative_unretract_is_rejected():
    with pytest.raises(Exception):
        MinimalPrintProfile(unretract_distance=-0.1)
    with pytest.raises(Exception):
        MinimalPrintProfile(unretract_speed=0)
