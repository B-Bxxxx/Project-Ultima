"""Offscreen tests of the JSON editors (skipped without PyQt6)."""
import json
import os

import pytest

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
pytest.importorskip("pyqtgraph.opengl", exc_type=ImportError)

_APP = []


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    return _APP[0]


def _set(ed, d):
    ed.text.setPlainText(d if isinstance(d, str) else json.dumps(d))


def test_function_editor_shows_the_file_validates_and_blocks_apply_on_errors(app):
    from src.gui_editors import FunctionEditor
    from src.stage1_slicer.function_file import load_function_file
    ed = FunctionEditor()
    ed.open_spec(load_function_file(os.path.join("config", "functions", "progressive_tilt.json")))
    d = json.loads(ed.text.toPlainText())
    assert "tilt_x_deg" in d and "type" not in d and "valid" in ed.status.text() and ed.btn_apply.isEnabled()
    d["tilt_x_deg"] = "start_tilt_deg + zoom * t"                 # undeclared parameter
    _set(ed, d)
    assert ed.status.text().startswith("Error") and "zoom" in ed.status.text() and not ed.btn_apply.isEnabled()
    d["tilt_x_deg"] = "__import__('os')"
    _set(ed, d)
    assert ed.status.text().startswith("Error")
    _set(ed, "{ not json")
    assert "invalid JSON" in ed.status.text() and not ed.btn_save.isEnabled()


def test_function_editor_apply_and_save(app, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    from src.gui_editors import FunctionEditor
    from src.stage1_slicer.function_file import load_function_file
    ed = FunctionEditor()
    ed.open_spec(None)
    d = json.loads(ed.text.toPlainText())
    d["name"] = "mine"
    _set(ed, d)
    got = []
    ed.applied.connect(got.append)
    ed.apply()
    assert got and got[0].name == "mine"
    out = str(tmp_path / "mine.json")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (out, ""))
    ed.save_as()
    assert load_function_file(out).name == "mine" and len(got) == 2


def test_machine_editor_validates_with_the_real_loader(app):
    from src.gui_editors import MachineEditor
    from src.stage3_kinematics.machine_file import load_machine_file
    ed = MachineEditor()
    ed.open_spec(load_machine_file(os.path.join("config", "machines", "trunnion_5axis.json")))
    assert "valid" in ed.status.text()
    d = json.loads(ed.text.toPlainText())
    d["axes"] = ["X", "Y", "Z", "B"]                               # B without C
    _set(ed, d)
    assert ed.status.text().startswith("Error") and not ed.btn_apply.isEnabled()
    ed.open_spec(None)
    got = []
    ed.applied.connect(got.append)
    ed.apply()
    assert got[0].name == "new_machine" and not got[0].has_rotary_axes


def test_main_window_edit_buttons_add_to_combos(app):
    from src import gui_app
    w = gui_app.MainWindow()
    ed = w.open_function_editor(False)
    d = json.loads(ed.text.toPlainText())
    d["name"] = "edited_fn"
    _set(ed, d)
    ed.apply()
    assert w.function_combo.currentText() == "edited_fn"
    me = w.open_machine_editor(True)
    me.apply()
    assert w.machine_combo.currentText() == "new_machine"
