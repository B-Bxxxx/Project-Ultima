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


def test_function_import_into_the_gui(win, tmp_path):
    expr = tmp_path / "f.txt"
    expr.write_text("# slope\n0.05 * x\n")
    from src.stage1_slicer.function_import import load_expression_file
    win.expr_input.setText(load_expression_file(str(expr)))
    win.strategy_combo.setCurrentText("custom_expr")
    win.chk_field_preview.setChecked(True)
    win.run_pipeline()
    assert win.lbl_metrics.text().startswith("Done"), win.lbl_metrics.text()

    data = tmp_path / "g.csv"
    data.write_text("x,y,f\n" + "\n".join(f"{x},{y},{0.05 * x}" for x in range(-20, 21, 5) for y in range(-20, 21, 5)))
    assert win.load_data_file_path(str(data))
    assert win.strategy_combo.currentText() == "grid_field"
    win.run_pipeline()
    assert win.lbl_metrics.text().startswith("Done"), win.lbl_metrics.text()
    assert "preview" not in win.lbl_metrics.text()

    bad = tmp_path / "bad.csv"
    bad.write_text("1 2\n")
    assert not win.load_data_file_path(str(bad))
    assert win.lbl_field.text().startswith("Error")


def test_grid_strategy_without_data_reports_an_error(win):
    win.strategy_combo.setCurrentText("grid_field")
    win.run_pipeline()
    assert win.lbl_metrics.text().startswith("Error") and "data file" in win.lbl_metrics.text()


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
    widgets = [win.layer_height_spin, win.perims_spin, win.strategy_combo, win.layer_hi, win.travel_cmd_combo,
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
    win.run_pipeline()
    assert win.lbl_metrics.text().startswith("Done"), win.lbl_metrics.text()
    moves = [ln for ln in win.gcode.splitlines() if ln.startswith(("G0 ", "G1 "))]
    assert moves and not any(" B" in ln or " C" in ln for ln in moves)
    win.topo_combo.setCurrentText("trunnion_table_xyzbc")
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
