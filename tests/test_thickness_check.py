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


def test_gui_label_follows_machine_mode():
    from PyQt6.QtWidgets import QApplication
    global _QAPP
    _QAPP = QApplication.instance() or QApplication([])
    from src.gui_app import MainWindow
    w = MainWindow()
    w.layer_height_spin.setValue(0.2)
    w.function_combo.setCurrentText("Dome (button)")
    w.update_transition_advice()
    txt = w.lbl_transition.text()
    assert "Vertical gap" in txt and "normal gap" in txt and "direction" in txt
