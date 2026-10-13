"""Offscreen tests of the function / machine editors (skipped without PyQt6)."""
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


def test_function_editor_form_json_roundtrip_and_preview(app):
    from src.gui_editors import FunctionEditor
    from src.stage1_slicer.function_file import load_function_file
    spec = load_function_file(os.path.join("config", "functions", "progressive_tilt.json"))
    ed = FunctionEditor()
    ed.open_spec(spec)
    assert "valid" in ed.status.text() and ed.btn_apply.isEnabled()
    d = json.loads(ed.json_edit.toPlainText())
    assert d["type"] == "progressive_tilt" and "end_tilt_deg" in d["parameters"]
    assert ed.lbl_range.text().strip() != ""
    # change a parameter default in the table -> JSON follows
    for r in range(ed.table.rowCount()):
        if ed.table.item(r, 0).text() == "end_tilt_deg":
            ed.table.item(r, 4).setText("20")
    assert json.loads(ed.json_edit.toPlainText())["parameters"]["end_tilt_deg"]["default"] == 20
    assert ed.spec.defaults()["end_tilt_deg"] == 20


def test_function_editor_reports_errors_and_blocks_apply(app):
    from src.gui_editors import FunctionEditor
    ed = FunctionEditor()
    ed.open_spec(None)
    ed.cb_type.setCurrentText("expression")
    ed.ed_expr.setText("a * x")                      # parameter a not declared
    assert ed.status.text().startswith("Error") and not ed.btn_apply.isEnabled()
    ed.table.add_row("a", {"default": 2.0})
    ed.table.edited.emit()
    assert "valid" in ed.status.text() and ed.btn_apply.isEnabled()
    ed.ed_expr.setText("__import__('os')")
    ed.ed_expr.textEdited.emit(ed.ed_expr.text())
    assert ed.status.text().startswith("Error")


def test_function_editor_json_tab_edit_syncs_to_form(app):
    from src.gui_editors import FunctionEditor
    ed = FunctionEditor()
    ed.open_spec(None)
    ed.tabs.setCurrentIndex(1)
    d = json.loads(ed.json_edit.toPlainText())
    d["name"] = "renamed"
    ed.json_edit.setPlainText(json.dumps(d))
    assert "valid" in ed.status.text()
    ed.tabs.setCurrentIndex(0)
    assert ed.ed_name.text() == "renamed"
    ed.tabs.setCurrentIndex(1)
    ed.json_edit.setPlainText("{ not json")
    assert ed.status.text().startswith("Error")


def test_function_editor_apply_and_save(app, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    from src.gui_editors import FunctionEditor
    from src.stage1_slicer.function_file import load_function_file
    ed = FunctionEditor()
    ed.open_spec(None)
    ed.ed_name.setText("mine")
    ed.ed_name.textEdited.emit("mine")
    got = []
    ed.applied.connect(got.append)
    ed.apply()
    assert got and got[0].name == "mine"
    out = str(tmp_path / "mine.json")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (out, ""))
    ed.save_as()
    assert load_function_file(out).name == "mine"


def test_machine_editor_preview_probe_and_validation(app):
    from src.gui_editors import MachineEditor
    from src.stage3_kinematics.machine_file import load_machine_file
    ed = MachineEditor()
    ed.open_spec(load_machine_file(os.path.join("config", "machines", "trunnion_5axis.json")))
    assert "valid" in ed.status.text()
    assert "B range" in ed.summary.text() and "reachable" in ed.lbl_probe.text()
    ed.axis_boxes["B"].setChecked(False)             # B and C must come together
    assert ed.status.text().startswith("Error") and not ed.btn_save.isEnabled()
    ed.axis_boxes["B"].setChecked(True)
    assert "valid" in ed.status.text()
    ed.sp_tilt.setValue(80)                           # beyond B limit of the file? still computed
    assert ed.lbl_probe.text()


def test_machine_editor_new_and_apply(app):
    from src.gui_editors import MachineEditor
    ed = MachineEditor()
    ed.open_spec(None)
    assert "valid" in ed.status.text()
    got = []
    ed.applied.connect(got.append)
    ed.apply()
    assert got[0].name == "new_machine" and not got[0].has_rotary_axes


def test_main_window_edit_buttons_add_to_combos(app, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from src import gui_app
    w = gui_app.MainWindow()
    ed = w.open_function_editor(False)
    ed.ed_name.setText("edited_fn")
    ed.ed_name.textEdited.emit("edited_fn")
    ed.apply()
    assert w.function_combo.currentText() == "edited_fn"
    me = w.open_machine_editor(True)
    me.apply()
    assert w.machine_combo.currentText() == "new_machine"
    assert w.current_machine().axes == ["X", "Y", "Z"]
