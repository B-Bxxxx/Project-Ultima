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


def test_hollow_cube_has_a_through_hole_and_pipeline_runs(win):
    win.geom_combo.setCurrentText("Hollow Cube")
    win.load_geometry()
    assert win.mesh.is_watertight and win.mesh.bounds[0, 2] == pytest.approx(0)
    win.run_pipeline()
    assert win.lbl_metrics.text().startswith("Done"), win.lbl_metrics.text()
    mid = [w for w in win.stage2_traj.waypoints if w.layer_index == 8 and not w.is_travel_move]
    assert mid and not any(abs(w.x) < 4.5 and abs(w.y) < 4.5 for w in mid)


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
