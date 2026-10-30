"""Offscreen GUI smoke tests (skipped when PyQt6 / pyqtgraph are not installed)."""
import numpy as np
import pytest

pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
pytest.importorskip("pyqtgraph.opengl", exc_type=ImportError)


_APP = []          # keeps the QApplication alive for the whole session (a dropped reference aborts Qt)


@pytest.fixture()
def win(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from src import gui_app
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    w = gui_app.MainWindow()
    w.layer_height_spin.setValue(1.0)
    w.trans_height_spin.setValue(3.0)
    return w


def test_placement_rotates_drops_and_invalidates_results(win):
    win.run_pipeline()
    assert win.stage2_traj is not None
    win.rot_x.setValue(90)
    assert win.stage2_traj is None and win.stage1_model is None
    assert "run the pipeline again" in win.lbl_metrics.text()
    assert win.mesh.bounds[0, 2] == pytest.approx(0)
    assert win.mesh.extents[2] == pytest.approx(2 * win.cyl_r.value())      # cylinder lying on its side
    win.reset_placement()
    assert win.mesh.extents[2] == pytest.approx(win.cyl_h.value())


def test_layer_range_playback_and_status(win):
    win.run_pipeline()
    n = win._layer_total()
    assert n > 10 and win.layer_hi.value() == n
    full_total = win.playback.total

    win.layer_lo.setValue(5)
    win.layer_hi.setValue(8)
    assert win._layers() == (4, 7)
    assert 0 < win.playback.total < full_total
    assert all(4 <= it.layer_index <= 7 for it in win.stage2_traj.waypoints[slice(*win._ranges["stage2"])])

    win.layer_lo.setValue(9)                                  # pushes the other handle: lo <= hi
    assert win.layer_lo.value() <= win.layer_hi.value()

    win.chk_single.setChecked(True)
    lo, hi = win._layers()
    assert lo == hi

    win.chk_single.setChecked(False)
    win.play_slider.setValue(0)
    assert win.lbl_status.text() == "-"
    win.toggle_play()                                         # starts (shown == 0)
    assert win.playback.playing
    shown = win.playback.shown
    win._on_tick()
    assert win.playback.shown > shown
    assert "Stage 3" in win.lbl_status.text() and " B" in win.lbl_status.text()
    for _ in range(1000):
        win._on_tick()
        if not win.playback.playing:
            break
    assert not win.playback.playing and win.playback.shown == win.playback.total


def test_shipped_functions_and_machines_are_listed(win):
    assert [win.function_combo.itemText(i) for i in range(win.function_combo.count())] == ["Conical", "Delayed tilt (2 stages)", "Dome (button)", "Planar", "Progressive tilt", "Ripple", "Rounded ridge", "Wave"]
    assert win.function_combo.currentText() == "Planar"
    assert win.machine_combo.count() == 2
    assert win.function_form.names() == []                      # planar has no parameters
    win.function_combo.setCurrentText("Progressive tilt")
    assert win.function_form.names() == ["start_tilt_deg", "end_tilt_deg"]


def test_function_parameters_become_named_fields_and_drive_the_slice(win, tmp_path):
    import json
    spec = {"format": "cam-function", "version": 2, "name": "Slope", "expression": "a * x",
            "parameters": {"a": {"label": "Slope", "unit": "mm/mm", "default": 0.05, "min": 0, "max": 1, "step": 0.01}}}
    p = tmp_path / "slope.json"
    p.write_text(json.dumps(spec))
    assert win.load_function_path(str(p))
    assert win.function_combo.currentText() == "Slope" and win.function_form.names() == ["a"]
    assert win.function_form.values() == {"a": 0.05}
    win.chk_field_preview.setChecked(True)
    win.run_pipeline()
    assert win.lbl_metrics.text().startswith("Done"), win.lbl_metrics.text()
    top = [w for w in win.stage2_traj.waypoints if w.layer_index == win._layer_total() - 1 and not w.is_travel_move]
    import numpy as np
    assert np.polyfit([w.x for w in top], [w.z for w in top], 1)[0] == pytest.approx(-0.05, abs=0.005)
    # a changed field value changes the result
    win.function_form._widgets["a"].setValue(0.0)
    assert win.function_form.values() == {"a": 0.0}
    win.run_pipeline()
    top = [w for w in win.stage2_traj.waypoints if w.layer_index == win._layer_total() - 1 and not w.is_travel_move]
    assert np.ptp([w.z for w in top]) < 1e-6


def test_invalid_function_and_machine_files_report_errors(win, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "cam-function", "version": 2, "name": "B", "expression": "a*x"}')
    assert not win.load_function_path(str(bad))
    assert win.lbl_function.text().startswith("Error") and "'a'" in win.lbl_function.text()
    assert not win.load_machine_path(str(bad))
    assert win.lbl_machine.text().startswith("Error")


def test_grid_function_from_a_data_file(win, tmp_path):
    import json
    (tmp_path / "f.csv").write_text("x,y,f\n" + "\n".join(f"{x},{y},{0.05 * x}" for x in range(-20, 21, 5) for y in range(-20, 21, 5)))
    (tmp_path / "g.json").write_text(json.dumps({"format": "cam-function", "version": 2, "name": "Grid",
                                                 "data_file": "f.csv", "parameters": {}}))
    assert win.load_function_path(str(tmp_path / "g.json"))
    win.chk_field_preview.setChecked(True)
    win.run_pipeline()
    assert win.lbl_metrics.text().startswith("Done"), win.lbl_metrics.text()


def test_function_and_machine_sections_can_be_saved_with_new_defaults(win, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    from src.stage1_slicer.function_file import load_function_file
    win.function_combo.setCurrentText("Progressive tilt")
    win.function_form._widgets["end_tilt_deg"].setValue(15.0)
    out = tmp_path / "t.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    win.save_function_to_file()
    assert load_function_file(str(out)).defaults()["end_tilt_deg"] == 15.0


def test_legend_sections_and_controls_exist(win):
    from src.common.feature_styles import FEATURE_STYLES
    assert set(win.feature_checks) == {s.key for s in FEATURE_STYLES}
    assert win.bottom_spin.value() == 3 and win.top_spin.value() == 3
    assert [win.travel_cmd_combo.itemText(i) for i in range(2)] == ["G0", "G1"]
    assert win.sections_layout.count() >= 7                   # collapsible sections (+ stretch)


def test_hollow_cube_is_gone_from_the_geometry_list(win):
    items = [win.geom_combo.itemText(i) for i in range(win.geom_combo.count())]
    assert not any("Hollow Cube" in t for t in items) and "Hollow Cylinder" in items


def test_z_offset_is_shown_only_without_drop_to_bed(win):
    assert win.off_z.isHidden() and win.off_z_label.isHidden()
    win.chk_drop.setChecked(False)
    assert not win.off_z.isHidden() and not win.off_z_label.isHidden()
    win.off_z.setValue(12.0)
    assert win.mesh.bounds[0, 2] == pytest.approx(win.source_mesh.bounds[0, 2] + 12.0 - 0.0 - (
        0.0))                                                  # keeps its own height + offset
    win.chk_drop.setChecked(True)
    assert win.off_z.isHidden() and win.mesh.bounds[0, 2] == pytest.approx(0.0)


def test_camera_follows_the_part_in_xy_but_the_grid_stays_on_the_bed(win):
    win.off_x.setValue(120.0)
    win.off_y.setValue(80.0)
    c = win.gl_stage0.opts["center"]
    assert (c.x(), c.y()) == pytest.approx((120.0, 80.0), abs=1e-6)
    assert c.z() == pytest.approx(win.cyl_h.value() / 2.0)          # real height above the bed, not shifted
    assert win._grid_origin == pytest.approx((120.0, 80.0))
    for w in (win.gl_stage1, win.gl_stage2, win.gl_kine):
        assert w.opts["center"].x() == pytest.approx(120.0)
    win.chk_drop.setChecked(False)
    win.off_z.setValue(30.0)
    assert win.gl_stage0.opts["center"].z() == pytest.approx(win.mesh.bounds.mean(axis=0)[2])


def test_wheel_never_changes_a_parameter(win):
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent
    widgets = [win.layer_height_spin, win.perims_spin, win.function_combo, win.machine_combo, win.layer_hi, win.travel_cmd_combo,
               win.play_slider, win.retract_spin]
    for w in widgets:
        before = w.value() if hasattr(w, "value") else w.currentIndex()
        ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120),
                         Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                         Qt.ScrollPhase.NoScrollPhase, False)
        ev.accept()
        w.wheelEvent(ev)
        assert not ev.isAccepted(), type(w).__name__               # left for the scroll area
        assert (w.value() if hasattr(w, "value") else w.currentIndex()) == before


def test_planar_gcode_has_only_xyz_but_a_trunnion_machine_gets_b_and_c(win):
    win.machine_combo.setCurrentText("Generic 3-axis Cartesian")
    win.run_pipeline()
    assert win.lbl_metrics.text().startswith("Done"), win.lbl_metrics.text()
    moves = [ln for ln in win.gcode.splitlines() if ln.startswith(("G0 ", "G1 "))]
    assert moves and not any(" B" in ln or " C" in ln for ln in moves)
    win.machine_combo.setCurrentText("Generic 5-axis trunnion table (XYZBC)")
    win.run_pipeline()
    assert win.lbl_metrics.text().startswith("Done"), win.lbl_metrics.text()
    assert any(" B" in ln and " C" in ln for ln in win.gcode.splitlines() if ln.startswith("G1 X"))


def test_retract_and_unretract_come_from_separate_fields(win):
    win.retract_spin.setValue(0.6)
    win.unretract_spin.setValue(0.9)
    win.retract_speed_spin.setValue(1800)
    win.unretract_speed_spin.setValue(900)
    win.run_pipeline()
    assert win.lbl_metrics.text().startswith("Done"), win.lbl_metrics.text()
    assert "G1 E-0.600 F1800 ; Retract" in win.gcode
    assert "G1 E0.900 F900 ; Prime" in win.gcode


def test_end_block_uses_the_lift_and_thickness_model_follows_the_machine(win):
    win.machine_combo.setCurrentText("Generic 3-axis Cartesian")
    win.park_spin.setValue(12.0)
    win.run_pipeline()
    assert win.lbl_metrics.text().startswith("Done"), win.lbl_metrics.text()
    assert win.profile.thickness_model == "stage1" and win.profile.park_lift == 12.0
    assert "Z150" not in win.gcode
    top = max(s.z for s in win.stage3_traj.states)
    end = win.gcode.split("; End G-Code")[1]
    assert f"Z{top + 12.0:.3f}" in end
    win.machine_combo.setCurrentText("Generic 5-axis trunnion table (XYZBC)")
    win.run_pipeline()
    assert win.profile.thickness_model == "nozzle_axis"
    win.thickness_combo.setCurrentText("stage1")
    win.run_pipeline()
    assert win.profile.thickness_model == "stage1"


def _gcode3():
    return "\n".join(
        [f";LAYER:{L}\n; TYPE: outer_wall\n" + "\n".join(f"G1 X{10 + k} Y{L} Z{L * 0.2 + 0.2} E0.1 F1500" for k in range(5)) for L in range(3)])


def test_one_panel_drives_the_gcode_check_viewer(win):
    """Layer range, playback and line stepping of the G-code check tab come from the main panel."""
    tab = win.check_tab
    tab.load_text(_gcode3(), "t")
    win.tabs.setCurrentWidget(tab)
    assert win._check_active() and win.layer_hi.maximum() == 3 and win.btn_step_fwd.isEnabled()
    n = len(tab._layer_arr) - 1
    assert win.playback.total == n and win.play_slider.maximum() == n          # the whole selected range
    win.layer_lo.setValue(2)                                                  # layers 2..3 only
    assert win.playback.total == int((tab._layer_arr >= 1).sum())
    a, b = win._check_range
    win.play_slider.setValue(0)
    win.step_gcode(1)
    assert win.playback.shown >= 1 and tab.rc.move_of_point[a + win.playback.shown - 1] == tab.rc.move_of_point[a]
    first = win.playback.shown
    win.step_gcode(1)
    assert tab.rc.move_of_point[a + win.playback.shown - 1] == tab.rc.move_of_point[a] + 1
    win.step_gcode(-1)
    assert win.playback.shown == first
    win.step_gcode(-1)
    assert win.playback.shown == 0
    win.tabs.setCurrentIndex(0)                                               # back to the pipeline viewers
    assert not win._check_active() and not win.btn_step_fwd.isEnabled()


def test_check_tab_machine_motion_view(win):
    """Fixed-camera machine view: it draws, and the table moves (its circle changes) while the camera does not."""
    mi = win.machine_combo.findText("Generic 5-axis trunnion table (XYZBC)")
    win.machine_combo.setCurrentIndex(mi)
    gcode = ";LAYER:0\n; TYPE: outer_wall\nG1 X100 Y100 Z1 B90 C0 E0.1 F1500\n" + "\n".join(
        f"G1 X{100 + k} Y{100 - k} Z1 B{90 + 5 * k} C{20 * k} E0.1 F1500" for k in range(1, 12))
    tab = win.check_tab
    tab.load_text(gcode, "t")
    win.tabs.setCurrentWidget(tab)
    tab.cmb_frame.setCurrentIndex(2)
    cam = (tab.view.opts["distance"], tuple(tab.view.opts["center"]))
    win.play_slider.setValue(win.play_slider.maximum() // 3)
    first = [np.array(i.pos).copy() for i in tab._items if getattr(i, "pos", None) is not None]
    win.play_slider.setValue(win.play_slider.maximum())
    last = [np.array(i.pos).copy() for i in tab._items if getattr(i, "pos", None) is not None]
    assert first and last and (tab.view.opts["distance"], tuple(tab.view.opts["center"])) == cam
    assert any(a.shape == b.shape and not np.allclose(a, b) for a, b in zip(first[-4:], last[-4:]))


def test_check_tab_shows_the_full_gcode_line_and_e(win):
    tab = win.check_tab
    tab.load_text(_gcode3(), "t")
    win.tabs.setCurrentWidget(tab)
    win.play_slider.setValue(3)
    txt = tab.lbl_pos.text()
    assert "G1 X" in txt and "E0.1" in txt and "E 0.10000" in txt and "path on the part" in txt and "tip speed / F" in txt
