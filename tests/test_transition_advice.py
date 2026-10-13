"""Transition height advice: the thickness window of the nozzle and the shipped function templates."""
import os

import numpy as np
import pytest
import trimesh

from src.stage1_slicer.function_file import load_function_dir, load_function_file
from src.stage1_slicer.plugins.field_slicer import UniversalFieldSlicerPlugin
from src.stage1_slicer.transition_advice import advise, field_range, printable_window

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUNCS = os.path.join(ROOT, "config", "functions")


def test_window_is_a_fraction_of_the_nozzle():
    assert printable_window(0.4) == pytest.approx((0.1, 0.32))


def test_positive_function_needs_a_transition_that_keeps_layers_thick_enough():
    a = advise(0.0, 4.0, 0.2, 0.4, transition=2.0)               # t = 0.2 * (1 - 4/2) < 0: folded
    assert not a.ok and a.min_transition == pytest.approx(4.0 / (1 - 0.1 / 0.2))      # 8 mm
    ok = advise(0.0, 4.0, 0.2, 0.4, transition=a.min_transition)
    assert ok.ok and ok.thickness_min == pytest.approx(0.1)


def test_negative_function_needs_a_transition_that_keeps_layers_thin_enough():
    a = advise(-7.5, 0.0, 0.2, 0.4, transition=5.0)              # t = 0.2 * (1 + 7.5/5) = 0.5 > 0.32
    assert not a.ok and a.min_transition == pytest.approx(7.5 / (0.32 / 0.2 - 1.0))
    assert advise(-7.5, 0.0, 0.2, 0.4, transition=a.min_transition).ok


def test_spacing_formula_matches_the_strategy_thickness():
    spec = load_function_file(os.path.join(FUNCS, "ripple.json"))
    mesh = trimesh.creation.cylinder(radius=10, height=10, sections=48)
    mesh.apply_translation((0, 0, 5))
    model = UniversalFieldSlicerPlugin().slice(mesh, {"layer_height": 0.4, "transition_height": 2.0, **spec.to_slicer_params()})
    th = np.concatenate([np.asarray(c.thicknesses) for c in model.layers[1].contours])
    rng = field_range(spec, spec.defaults(), (-10, 10, -10, 10))
    a = advise(rng[0], rng[1], 0.4, 0.4, 2.0)
    assert th.min() >= a.thickness_min - 1e-3 and th.max() <= a.thickness_max + 1e-3


def test_layer_height_outside_the_nozzle_window_cannot_be_fixed_by_the_transition():
    a = advise(0.0, 1.0, 0.1, 0.4, transition=50.0)              # h is already at the 0.25 * nozzle limit
    assert not a.possible and not a.ok and a.messages
    assert not advise(-1.0, 0.0, 0.32, 0.4, transition=50.0).possible


def test_zero_transition_with_a_deformation_is_flagged():
    a = advise(0.0, 1.0, 0.2, 0.4, transition=0.0)
    assert not a.ok and "bed" in " ".join(a.messages)


def test_tilt_functions_have_no_field_range():
    spec = load_function_dir(FUNCS)["Progressive tilt"]
    assert field_range(spec, spec.defaults(), (-10, 10, -10, 10)) is None


@pytest.mark.parametrize("name", ["Rounded ridge", "Ripple", "Dome (button)", "Wave", "Conical"])
def test_shipped_templates_slice_a_cylinder(name):
    spec = load_function_dir(FUNCS)[name]
    mesh = trimesh.creation.cylinder(radius=15, height=20, sections=48)
    mesh.apply_translation((0, 0, 10))
    rng = field_range(spec, spec.defaults(), (-15, 15, -15, 15))
    T = max(advise(*rng, 0.4, 0.4, 2.0).min_transition, 2.0) + 0.5
    model = UniversalFieldSlicerPlugin().slice(mesh, {"layer_height": 0.4, "transition_height": T, **spec.to_slicer_params()})
    assert len(model.layers) > 40
    for layer in model.layers[2:-3]:
        assert layer.contours and all(np.allclose(c.points[0], c.points[-1]) for c in layer.contours)


def test_rounded_ridge_is_round_in_the_middle_and_linear_on_the_flanks():
    spec = load_function_dir(FUNCS)["Rounded ridge"]
    k, R = 0.4, 6.0
    v = spec.values({"k": k, "R": R})
    x = np.array([-30.0, -3.0, 0.0, 3.0, 30.0])
    from src.stage1_slicer.function_import import ExpressionField
    f = ExpressionField(spec.expression, v).evaluate(x, np.zeros_like(x))
    assert f[2] == pytest.approx(0.0) and f[0] == pytest.approx(f[4])           # symmetric, zero at the apex
    assert (f[4] - f[3]) / 27.0 == pytest.approx(k, rel=0.05)                   # slope k far from the apex
    assert (f[3] - f[2]) / 3.0 < 0.6 * k                                          # still flat near the apex


def test_gui_shows_the_advice_and_the_button_applies_it(monkeypatch):
    pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
    pytest.importorskip("pyqtgraph.opengl", exc_type=ImportError)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from src import gui_app
    _app = QApplication.instance() or QApplication([])
    globals().setdefault("_KEEP", []).append(_app)
    w = gui_app.MainWindow()
    w.layer_height_spin.setValue(0.2)
    w.trans_height_spin.setValue(1.0)
    assert w.lbl_transition.text() == ""                        # planar: nothing to advise
    w.function_combo.setCurrentText("Dome (button)")
    w.function_form._widgets["Rs"].setValue(20.0)                # steep dome: f up to ~5.9 over a 15 mm radius
    w.update_transition_advice()
    assert "Suggested minimum" in w.lbl_transition.text()
    assert not w.btn_suggested.isHidden()
    w.btn_suggested.click()
    adv = w.transition_advice()
    assert adv.ok and w.trans_height_spin.value() >= adv.min_transition - 1e-9
    w.run_pipeline()
    assert "Warning" not in w.lbl_metrics.text()
    w.trans_height_spin.setValue(1.0)
    w.run_pipeline()
    assert "Warning" in w.lbl_metrics.text()


def test_a_layer_height_above_the_nozzle_window_is_named_as_the_cause():
    a = advise(0.0, 1.0, 0.4, 0.4, transition=5.0)               # 0.4 on a 0.4 nozzle is above 0.8 * nozzle
    assert not a.ok and any("layer height itself" in m for m in a.messages)
