"""Project (settings) files: pure file handling + the GUI round trip + the ramp mode selector."""
import json

import pytest

from src.common import project_file
from src.common.project_file import ProjectFileError

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
pytest.importorskip("pyqtgraph.opengl", exc_type=ImportError)

_APP = []


def _window(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from src import gui_app
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    return gui_app.MainWindow()


@pytest.fixture()
def win(monkeypatch):
    return _window(monkeypatch)


# -- the file format ---------------------------------------------------------------------------
@pytest.mark.parametrize("data,message", [
    ([], "JSON object"),
    ({"format": "x", "version": 1}, "not a project file"),
    ({"format": "cam-project", "version": 99}, "unsupported"),
    ({"format": "cam-project", "version": 1, "slicer": 3}, "must be an object"),
    ({"format": "cam-project", "version": 1, "function": {"values": []}}, "values"),
])
def test_invalid_project_files_are_rejected(data, message):
    with pytest.raises(ProjectFileError, match=message):
        project_file.validate(data)


def test_save_and_load_make_file_references_relative_to_the_project(tmp_path):
    src = tmp_path / "funcs" / "w.json"
    src.parent.mkdir()
    d = {"slicer": {"layer_height": 0.3}, "function": {"name": "W", "source": str(src), "values": {"a": 1}},
         "machine": {"name": "M", "source": "D:/elsewhere/m.json", "values": {}}}
    path = tmp_path / "p.json"
    project_file.save_project(d, str(path))
    raw = json.loads(path.read_text())
    assert raw["format"] == "cam-project" and raw["version"] == 1
    assert raw["function"]["source"] == "funcs/w.json"                      # below the project -> relative
    loaded = project_file.load_project(str(path))
    assert loaded["function"]["source"] == str(src)                         # ... and absolute again after loading
    assert loaded["slicer"] == {"layer_height": 0.3}


def test_unreadable_or_broken_files_are_reported(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    with pytest.raises(ProjectFileError, match="invalid JSON"):
        project_file.load_project(str(bad))
    with pytest.raises(ProjectFileError, match="cannot read"):
        project_file.load_project(str(tmp_path / "missing.json"))


# -- GUI round trip -----------------------------------------------------------------------------------
def _change_everything(w):
    w.geom_combo.setCurrentText("Hollow Cylinder")
    w.cyl_r.setValue(18.0)
    w.cyl_h.setValue(12.0)
    w.cyl_wall.setValue(4.0)
    w.rot_z.setValue(30.0)
    w.off_x.setValue(5.0)
    w.chk_drop.setChecked(False)
    w.off_z.setValue(2.0)
    w.layer_height_spin.setValue(0.3)
    w.trans_height_spin.setValue(4.5)
    w.mode_combo.setCurrentText("ramp")
    w.ramp_h_spin.setValue(0.25)
    w.ramp_gap_spin.setValue(0.15)
    w.nozzle_spin.setValue(0.6)
    w.perims_spin.setValue(3)
    w.infill_spin.setValue(0.35)
    w.travel_cmd_combo.setCurrentText("G1")
    w.retract_spin.setValue(1.2)
    w.zhop_spin.setValue(0.6)
    w.hop_short_chk.setChecked(True)
    w.path_tol_spin.setValue(0.05)
    w.seam_spin.setValue(-1)                                    # off
    w.thickness_combo.setCurrentText("nozzle_axis")
    w.function_combo.setCurrentText("Wave")
    w.machine_combo.setCurrentText("Generic 5-axis trunnion table (XYZBC)")


def test_everything_survives_a_save_and_load(win, monkeypatch, tmp_path):
    _change_everything(win)
    name = next(iter(win.function_form.names()))
    win.function_form._widgets[name].setValue(win.function_form._widgets[name].value() * 0.5 + 0.01)
    before = win.collect_project()
    path = tmp_path / "session.json"
    project_file.save_project(before, str(path))

    fresh = _window(monkeypatch)                                # a new session with the defaults
    assert fresh.collect_project() != before
    assert fresh.load_project_path(str(path))
    after = fresh.collect_project()
    for section in ("geometry", "placement", "slicer", "profile", "function", "machine"):
        assert after[section] == before[section], section
    assert "Note:" not in fresh.lbl_project.text(), fresh.lbl_project.text()
    # the loaded session really uses the values: mesh follows geometry + placement, mode is ramp, trans. height disabled
    assert fresh.mesh is not None and fresh.mesh.extents[2] == pytest.approx(12.0)
    assert fresh.mode_combo.currentText() == "ramp" and not fresh.trans_height_spin.isEnabled()
    assert fresh.seam_spin.value() < 0


def test_a_function_without_a_file_travels_inside_the_project(win, monkeypatch, tmp_path):
    from src.stage1_slicer.function_file import FunctionSpec
    spec = FunctionSpec.from_dict({"format": "cam-function", "version": 2, "name": "Only here", "expression": "a * x",
                                   "parameters": {"a": {"label": "a", "default": 0.07, "min": 0, "max": 1}}})
    win.add_function(spec)
    win.function_form._widgets["a"].setValue(0.2)
    path = tmp_path / "p.json"
    project_file.save_project(win.collect_project(), str(path))
    fresh = _window(monkeypatch)
    assert "Only here" not in fresh.functions
    assert fresh.load_project_path(str(path))
    assert fresh.function_combo.currentText() == "Only here" and fresh.function_form.values() == {"a": 0.2}


def test_problems_are_reported_but_the_rest_is_applied(win, tmp_path):
    d = {"format": "cam-project", "version": 1,
         "slicer": {"layer_height": 0.35, "future_option": 1},
         "profile": {"travel_command": "G9", "nozzle_diameter": 99.0},
         "function": {"name": "No such function", "source": "", "values": {}},
         "geometry": {"type": "Upload STL...", "stl_path": str(tmp_path / "missing.stl")}}
    path = tmp_path / "p.json"
    path.write_text(json.dumps(d))
    assert win.load_project_path(str(path))
    text = win.lbl_project.text()
    assert win.layer_height_spin.value() == pytest.approx(0.35)
    for part in ("slicer.future_option", "travel_command", "nozzle_diameter", "No such function", "missing.stl"):
        assert part in text, (part, text)
    assert win.nozzle_spin.value() == win.nozzle_spin.maximum()


def test_loading_a_project_with_an_stl_restores_the_mesh(win, tmp_path):
    import trimesh
    stl = tmp_path / "part.stl"
    trimesh.creation.box(extents=(10, 12, 7)).export(str(stl))
    d = {"format": "cam-project", "version": 1, "geometry": {"type": "Upload STL...", "stl_path": str(stl)}}
    path = tmp_path / "p.json"
    path.write_text(json.dumps(d))
    assert win.load_project_path(str(path))
    assert win.mesh.extents == pytest.approx([10, 12, 7])
    assert win.collect_project()["geometry"]["stl_path"] == str(stl)
    assert "Note:" not in win.lbl_project.text()


def test_not_a_project_file_is_refused_with_a_message(win, tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"format": "cam-function", "version": 2}))
    assert not win.load_project_path(str(p))
    assert "not a project file" in win.lbl_project.text()


# -- the transition mode in the GUI -----------------------------------------------------------------
def test_ramp_mode_is_off_by_default_and_runs_the_whole_pipeline_when_selected(win):
    assert win.mode_combo.currentText() == "standard"
    assert win._mode_params() == {} and not win.ramp_h_spin.isVisibleTo(win)
    win.layer_height_spin.setValue(1.0)
    win.function_combo.setCurrentText("Rounded ridge")
    win.run_pipeline()
    assert win.lbl_metrics.text().startswith("Done"), win.lbl_metrics.text()
    assert "transition_mode" not in win.stage1_model.metadata
    std_layers = len(win.stage1_model.layers)

    win.mode_combo.setCurrentText("ramp")
    assert win.ramp_h_spin.isVisibleTo(win) and not win.trans_height_spin.isEnabled()
    win.ramp_h_spin.setValue(0.5)
    win.run_pipeline()
    assert win.lbl_metrics.text().startswith("Done"), win.lbl_metrics.text()
    meta = win.stage1_model.metadata
    assert meta["transition_mode"] == "ramp" and meta["ramp_layers"] > 0 and meta["ramp_layer_height"] == 0.5
    assert len(win.stage1_model.layers) > std_layers - 1
    assert win.gcode.count("\n") > 100
    win.update_transition_advice()
    assert "Ramp mode" in win.lbl_transition.text()

    win.mode_combo.setCurrentText("standard")                       # and back: nothing of the ramp remains
    win.run_pipeline()
    assert "transition_mode" not in win.stage1_model.metadata and win.trans_height_spin.isEnabled()


def test_ramp_with_a_tilt_function_reports_a_clear_error(win):
    win.function_combo.setCurrentText("Progressive tilt")
    win.mode_combo.setCurrentText("ramp")
    win.run_pipeline()
    assert win.lbl_metrics.text().startswith("Error") and "ramp" in win.lbl_metrics.text()
