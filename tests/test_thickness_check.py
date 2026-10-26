"""Function-independent layer-gap check: vertical / normal gaps vs the nozzle's printable window."""
import os
import sys

import numpy as np
import pytest
import trimesh

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stage1_slicer import thickness_check as tc
from src.stage1_slicer.function_file import load_function_dir
from src.stage1_slicer.registry import PluginRegistry

FUNCS = load_function_dir(os.path.join(os.path.dirname(__file__), "..", "config", "functions"))
_QAPP = None


def _cyl():
    m = trimesh.creation.cylinder(radius=15, height=20, sections=64)
    m.apply_translation((0, 0, 10))
    return m


def _params(name, h=0.2, T=6.0, **ov):
    p = FUNCS[name].to_slicer_params(ov)
    s = p.pop("strategy")
    p["layer_height"] = h
    p["transition_height"] = T
    return s, p


def test_window_is_fraction_of_nozzle():
    assert tc.printable_window(0.4) == pytest.approx((0.1, 0.32))


def test_planar_gaps_equal_layer_height():
    s, p = _params("Planar", h=0.2)
    st = tc.probe(_cyl(), s, p)
    assert st.as_tuple() == pytest.approx((0.2, 0.2, 0.2, 0.2), abs=1e-6)


@pytest.mark.parametrize("name", ["Wave", "Dome (button)", "Conical", "Progressive tilt"])
def test_normal_never_exceeds_vertical(name):
    s, p = _params(name)
    st = tc.probe(_cyl(), s, p)
    assert st.n_min <= st.v_min + 1e-9 and st.n_max <= st.v_max + 1e-9


def test_model_stats_agree_with_probe():
    s, p = _params("Wave", a=1.0)
    mesh = _cyl()
    model = PluginRegistry.get_plugin("universal_field_slicer")().slice(mesh, {"strategy": s, **p})
    meas, est = tc.measure_model(model), tc.probe(mesh, s, p)
    assert meas is not None and meas.count > 0
    assert meas.v_min == pytest.approx(est.v_min, abs=0.03) and meas.v_max == pytest.approx(est.v_max, abs=0.03)


def test_direction_follows_machine_and_both_values_reported():
    st = tc.GapStats(0.12, 0.30, 0.08, 0.28, 10)        # vertical passes, normal too small
    c3, c5 = tc.check(st, 0.4, rotary=False), tc.check(st, 0.4, rotary=True)
    assert c3.direction == "vertical" and c3.ok and not c3.both_ok
    assert c5.direction == "normal" and not c5.ok
    assert (c3.other_low, c3.other_high) == (0.08, 0.28)
    assert any("other nozzle direction" in m for m in c3.messages)


def test_layer_height_itself_hint():
    st = tc.GapStats(0.4, 0.4, 0.4, 0.4, 1)
    c = tc.check(st, 0.4, False, layer_height=0.4)
    assert not c.ok and "layer height itself" in " ".join(c.messages)


def test_transition_suggestion_makes_it_pass():
    s, p = _params("Dome (button)", h=0.2, T=1.0)
    mesh = _cyl()
    assert not tc.check(tc.probe(mesh, s, p, 1.0), 0.4, False).ok
    sug = tc.suggest_transition(mesh, s, p, 0.4, False, 1.0, 0.2)
    assert sug.relevant and sug.possible and sug.min_transition > 1.0
    assert tc.check(tc.probe(mesh, s, p, sug.min_transition), 0.4, False).ok


def test_dome_t6_no_false_alarm():
    s, p = _params("Dome (button)", h=0.2, T=6.0)
    assert tc.check(tc.probe(_cyl(), s, p), 0.4, False).ok


def test_tilt_transition_has_no_effect():
    s, p = _params("Progressive tilt", h=0.2)
    sug = tc.suggest_transition(_cyl(), s, p, 0.4, False, 6.0, 0.2)
    assert not sug.relevant and sug.min_transition is None


def test_impossible_when_layer_height_outside_window():
    s, p = _params("Conical", h=0.4)
    sug = tc.suggest_transition(_cyl(), s, p, 0.4, False, 6.0, 0.4)
    assert sug.relevant and not sug.possible


def _window():
    from PyQt6.QtWidgets import QApplication
    global _QAPP
    _QAPP = QApplication.instance() or QApplication([])
    from src.gui_app import MainWindow
    w = MainWindow()
    w.layer_height_spin.setValue(0.2)
    return w


def test_gui_label_shows_both_directions():
    w = _window()
    w.function_combo.setCurrentText("Dome (button)")
    w.update_transition_advice()
    txt = w.lbl_transition.text()
    assert "vertical" in txt and "normal" in txt and ("OK" in txt or "NOT printable" in txt)


def test_gui_note_is_not_repeated_and_button_hidden_without_effect():
    w = _window()
    w.function_combo.setCurrentText("Progressive tilt")
    w.update_transition_advice()
    for _ in range(3):
        w.use_suggested_transition()
    assert w.lbl_transition.text().count("does not change") <= 1
    assert not w.btn_suggested.isVisibleTo(w)             # the transition cannot help tilting layers


def test_gui_button_visible_when_transition_helps():
    w = _window()
    w.function_combo.setCurrentText("Conical")
    w.update_transition_advice()
    assert w.btn_suggested.isVisibleTo(w)
    w.use_suggested_transition()
    assert w.trans_height_spin.value() > 2.0


def test_tilt_of_the_layer_surfaces_is_measured_and_warned_about_for_3_axis_machines():
    import math
    tc = __import__("src.stage1_slicer.thickness_check", fromlist=["x"])
    v = [0.2, 0.2, 0.2]
    n = [0.2, 0.2 * math.cos(math.radians(30)), 0.2 * math.cos(math.radians(10))]
    assert tc.tilt_deg(v, n) == pytest.approx(30.0)
    st = tc.GapStats(0.2, 0.2, min(n), 0.2, 3, tc.tilt_deg(v, n))
    assert tc.check(st, 0.4, False, 0.2, max_tilt_deg=25.0).tilt_warning is not None     # 30 deg > 25 deg
    assert tc.check(st, 0.4, False, 0.2, max_tilt_deg=35.0).tilt_warning is None
    assert tc.check(st, 0.4, True, 0.2, max_tilt_deg=25.0).tilt_warning is None          # rotary machines follow the surface
    assert tc.check(st, 0.4, False, 0.2).tilt_warning is None                           # no limit given
    assert tc.tilt_deg([0.2], [0.2]) == 0.0


def test_probe_and_measure_report_the_slope_of_a_tilted_function():
    import trimesh
    from src.stage1_slicer import thickness_check as tc
    mesh = trimesh.creation.box(extents=(20, 20, 10))
    mesh.apply_translation((0, 0, 5))
    p = {"layer_height": 0.2, "transition_height": 2.0, "expression": "0.5*x", "expression_params": {}}   # slope atan(0.5) = 26.6 deg
    st = tc.probe(mesh, "custom_expr", p)
    assert st.tilt_max_deg == pytest.approx(26.565, abs=0.3)
    flat = tc.probe(mesh, "custom_expr", {**p, "expression": "0.0"})
    assert flat.tilt_max_deg == pytest.approx(0.0, abs=1e-6)
