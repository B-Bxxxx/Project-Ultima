"""
"5. G-Code Check" tab: reads G-code back and draws the path the firmware really takes over the PART (or the machine),
re-centred, playable, with the moves that deviate from their straight line marked red. Logic: src/common/gcode_check.py.
"""
import os
from typing import Callable, Optional

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph import Vector
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QPlainTextEdit,
                             QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget)

from src import gui_theme
from src.common.feature_styles import FEATURE_STYLES
from src.common.gcode_check import Reconstruction, parse_gcode, reconstruct, table_map

BAD_COLOR = (1.0, 0.15, 0.15, 1.0)


class CheckTab(QWidget):
    """`machine_provider()` returns (solver, MachineConfig, MachineSpec) of the selected machine; `gcode_provider()` the last G-code."""

    def __init__(self, machine_provider: Callable, gcode_provider: Callable[[], Optional[str]], parent=None):
        super().__init__(parent)
        self.machine_provider, self.gcode_provider = machine_provider, gcode_provider
        self.rc: Optional[Reconstruction] = None
        self.title = ""
        self._items = []
        self._base = None                    # (key, item): the complete layers below the playing layer, rebuilt only when the key changes
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        self.btn_check = QPushButton("Check generated G-code")
        self.btn_check.clicked.connect(self.check_generated)
        self.btn_open = QPushButton("Open G-code file...")
        self.btn_open.clicked.connect(self.open_file)
        self.cmb_frame = QComboBox()
        self.cmb_frame.addItems(["Part (path on the part)", "Machine (axis positions)", "Machine motion (fixed view, table moves)"])
        self.cmb_frame.currentIndexChanged.connect(self._frame_changed)
        self.chk_travel = QCheckBox("Travel")
        self.chk_travel.setChecked(True)
        self.chk_travel.toggled.connect(self.redraw)
        self.chk_bad = QCheckBox("Mark deviating moves")
        self.chk_bad.setChecked(True)
        self.chk_bad.toggled.connect(self.redraw)
        self.tol = QDoubleSpinBox()
        self.tol.setRange(0.001, 10.0)
        self.tol.setDecimals(3)
        self.tol.setSingleStep(0.01)
        self.tol.setValue(0.05)
        self.tol.setSuffix(" mm")
        self.tol.valueChanged.connect(self._tolerance_changed)
        self.layer = QSpinBox()
        self.layer.setSpecialValueText("all layers")
        self.layer.setRange(-1, 0)
        self.layer.setValue(-1)
        self.layer.valueChanged.connect(self._layer_changed)
        for w in (self.btn_check, self.btn_open, self.cmb_frame, self.chk_travel, self.chk_bad, QLabel("tol."), self.tol,
                  QLabel("layer"), self.layer):
            top.addWidget(w)
        top.addStretch()
        lay.addLayout(top)

        self.view = gl.GLViewWidget()
        self.view.opts["distance"] = 80
        self.view.setBackgroundColor(gui_theme.GL_BACKGROUND)
        lay.addWidget(self.view, 1)

        play = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.redraw)
        self.lbl_pos = QLabel("")
        play.addWidget(self.btn_play)
        play.addWidget(self.slider, 1)
        play.addWidget(self.lbl_pos)
        lay.addLayout(play)

        self.txt = QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMaximumHeight(150)
        self.txt.setStyleSheet("font-family: monospace;")
        lay.addWidget(self.txt)
        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------ loading
    def check_generated(self):
        text = self.gcode_provider()
        if not text:
            self.txt.setPlainText("No generated G-code yet: run the pipeline first.")
            return
        self.load_text(text, "generated G-code")

    def open_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open G-code", "", "G-code (*.gcode *.nc *.gc *.g *.txt);;All files (*)")
        if fname:
            with open(fname, encoding="utf-8", errors="replace") as fh:
                self.load_text(fh.read(), os.path.basename(fname))

    def load_text(self, text: str, title: str):
        try:
            solver, cfg, _spec = self.machine_provider()
            moves, warns = parse_gcode(text)
            self.rc = reconstruct(moves, solver, cfg)
        except Exception as e:                      # a bad file must never break the GUI
            self.rc = None
            self.txt.setPlainText(f"Error: {e}")
            return
        self.title = title
        self._solver, self._cfg = solver, cfg
        self._warns = warns
        layers = [max(m.layer, 0) for m in self.rc.moves]
        self.layer.blockSignals(True)
        self.layer.setRange(-1, max(layers))
        self.layer.setValue(-1)
        self.layer.blockSignals(False)
        self._items_frame = None
        self._rebuild_points()
        self._play_range()
        self._summary()
        self.frame_view()
        self.redraw()

    def _play_range(self):
        """The slider (and Play) runs through the samples of ONE layer: the chosen one, else the last; layers below are complete."""
        la = self._layer_arr
        target = self.layer.value() if self.layer.value() >= 0 else int(la.max())
        idx = np.nonzero(la == target)[0]
        idx = idx[idx >= 1]
        self._play = (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (1, 1)
        self.slider.blockSignals(True)
        self.slider.setRange(0, self._play[1] - self._play[0])
        self.slider.setValue(self._play[1] - self._play[0])
        self.slider.blockSignals(False)

    def _frame_changed(self, *_):
        if self.rc is not None:
            self._clear_base()
            self.frame_view()
        self.redraw()

    def _layer_changed(self, *_):
        if self.rc is not None:
            self._play_range()
        self.redraw()

    def _tolerance_changed(self, *_):
        if self.rc is not None:
            self._rebuild_points()
            self._summary()
            self.redraw()

    # ------------------------------------------------------------------ data for drawing
    def _rebuild_points(self):
        rc = self.rc
        mi = rc.move_of_point
        mv = np.maximum(mi, 0)
        self._layer_arr = np.array([max(m.layer, 0) for m in rc.moves])[mv]
        self._feat_arr = np.array([m.feature for m in rc.moves])[mv]
        self._bad_arr = np.where(mi >= 0, rc.bad_moves(self.tol.value())[mv], False)

    def _summary(self):
        rc, s = self.rc, self.rc.stats
        nb = int(rc.bad_moves(self.tol.value()).sum())
        lines = [f"{self.title}: {int(s['moves'])} moves ({int(s['extruding'])} extruding)",
                 f"Time (firmware 6D length / F): {s['time_min_6d']:.1f} min   |  XYZ length only: {s['time_min_xyz']:.1f} min"
                 + (f"   -> rotation slows the print x{s['time_min_6d'] / s['time_min_xyz']:.2f}" if s['time_min_xyz'] > 0 else ""),
                 f"Largest C turn in one extruding move: {s['c_swing_max']:.0f} deg   (> 90 deg: {int(s['c_over_90'])} moves)",
                 f"Deviation from the straight line: max {s['max_deviation']:.2f} mm;  moves over {self.tol.value():g} mm: {nb}"]
        if nb:
            worst = np.argsort(-np.where(rc.bad_moves(self.tol.value()), rc.deviation, 0))[:5]
            lines.append("Worst moves (line: deviation, C turn): " + "; ".join(
                f"{rc.moves[i].line}: {rc.deviation[i]:.1f} mm, {rc.c_swing[i]:.0f} deg" for i in worst))
        lines += ["! " + w for w in rc.warnings + list(getattr(self, "_warns", []))[:5]]
        self.txt.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------ drawing
    def _clear_base(self):
        if self._base is not None:
            for it in self._base[1]:
                self.view.removeItem(it)
        self._base = None

    def frame_view(self):
        if self.rc is None:
            return
        P = self.rc.part if self.cmb_frame.currentIndex() == 0 else self.rc.machine
        sel = ~np.array([self.rc.moves[max(i, 0)].feature == "travel" for i in self.rc.move_of_point])
        Q = P[sel] if sel.any() else P
        if self.cmb_frame.currentIndex() == 2:                   # fixed view: the whole machine-frame range of the print + the table
            r = self._table_radius()
            Q = np.vstack([Q + [r, r, 0], Q - [r, r, 0]])
        lo, hi = Q.min(axis=0), Q.max(axis=0)
        self.view.opts["center"] = Vector(*((lo + hi) / 2))
        self.view.opts["distance"] = float(max(np.ptp(Q, axis=0).max(), 5.0)) * 1.8

    def _table_radius(self) -> float:
        part = self.rc.part
        return float(max(np.ptp(part[:, 0]), np.ptp(part[:, 1]), 20.0)) * 0.75 + 10.0

    def _redraw_motion(self, stop: int):
        """Fixed camera, machine frame: the nozzle stays on its axis line, the table (and the part printed so far) tilts and turns."""
        rc = self.rc
        k = max(stop - 1, 0)
        pose = rc.poses[k]
        M, t = table_map(self._solver, self._cfg, pose)
        p0 = self._play[0]
        # deposited samples: everything up to the head (complete layers below + the playing layer); thinned to keep drawing fast
        idx = np.arange(1, max(stop, 1))
        feat = self._feat_arr[idx] if len(idx) else idx
        keep = (feat != "travel") if len(idx) else np.zeros(0, bool)
        if self.layer.value() >= 0:
            keep &= self._layer_arr[idx] == self.layer.value()
        idx = idx[keep]
        stride = max(1, len(idx) // 25000)
        idx = idx[::stride]
        if len(idx):
            a, b = rc.part[idx - 1] @ M.T + t, rc.part[idx] @ M.T + t
            for st in FEATURE_STYLES:
                sel = self._feat_arr[idx] == st.key
                if sel.any():
                    pts = np.empty((int(sel.sum()) * 2, 3), np.float32)
                    pts[0::2], pts[1::2] = a[sel], b[sel]
                    it = gl.GLLinePlotItem(pos=pts, color=st.color, mode="lines")
                    self.view.addItem(it)
                    self._items.append(it)
        # the table: a circle and a cross in the part's bottom plane, moved by the same pose
        r = self._table_radius()
        c = rc.part[rc.part[:, 2] >= 0][:, :2].mean(axis=0) if len(rc.part) else np.zeros(2)
        z0 = float(rc.part[:, 2].min())
        ang = np.linspace(0, 2 * np.pi, 73)
        circ = np.column_stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang), np.full_like(ang, z0)])
        cross = np.array([[c[0] - r, c[1], z0], [c[0] + r, c[1], z0], [c[0], c[1] - r, z0], [c[0], c[1] + r, z0]])
        for pts, mode, col in ((circ, "line_strip", (0.55, 0.65, 0.8, 1.0)), (cross, "lines", (0.35, 0.4, 0.5, 1.0))):
            it = gl.GLLinePlotItem(pos=(pts @ M.T + t).astype(np.float32), color=col, mode=mode)
            self.view.addItem(it)
            self._items.append(it)
        tip = pose[:3]
        it = gl.GLLinePlotItem(pos=np.array([tip, tip + [0, 0, 25.0]], np.float32), color=(1, 1, 1, 1), mode="lines", width=2)
        self.view.addItem(it)                                          # the Z carriage / nozzle axis line (machine frame)
        self._items.append(it)
        head = gl.GLScatterPlotItem(pos=np.array([tip], np.float32), size=10, color=(1, 1, 1, 1))
        self.view.addItem(head)
        self._items.append(head)
        mi = int(rc.move_of_point[k])
        if mi >= 0:
            mv = rc.moves[mi]
            self.lbl_pos.setText(f"line {mv.line} | layer {mv.layer} | {mv.feature} | machine X{pose[0]:.2f} Y{pose[1]:.2f} Z{pose[2]:.2f} "
                                 f"A{pose[3]:.1f} B{pose[4]:.1f} C{pose[5]:.1f}")

    def redraw(self, *_):
        if self.rc is None:
            return
        for it in self._items:
            self.view.removeItem(it)
        self._items = []
        rc = self.rc
        if self.cmb_frame.currentIndex() == 2:
            self._clear_base()
            self._redraw_motion(min(self._play[0] + self.slider.value(), len(rc.part)))
            return
        P = rc.part if self.cmb_frame.currentIndex() == 0 else rc.machine
        m = len(P)
        p0, p1 = self._play
        stop = min(p0 + self.slider.value(), m)
        layer = self.layer.value()

        def add(idx, sel, color):
            ii = idx[sel]
            if not len(ii):
                return None
            pts = np.empty((len(ii) * 2, 3), np.float32)
            pts[0::2], pts[1::2] = P[ii - 1], P[ii]
            it = gl.GLLinePlotItem(pos=pts, color=color, mode="lines")
            self.view.addItem(it)
            return it

        def draw(idx):
            """Items of the samples `idx` (feature colours, deviating moves red); returns the created items."""
            if not len(idx):
                return []
            ok = np.ones(len(idx), bool)
            if layer >= 0:
                ok &= self._layer_arr[idx] == layer
            trav = self._feat_arr[idx] == "travel"
            if not self.chk_travel.isChecked():
                ok &= ~trav
            bad = self._bad_arr[idx] & self.chk_bad.isChecked()
            out = []
            for st in FEATURE_STYLES:
                out.append(add(idx, ok & ~bad & (self._feat_arr[idx] == st.key), st.color))
            known = {s.key for s in FEATURE_STYLES}
            out.append(add(idx, ok & ~bad & ~np.isin(self._feat_arr[idx], list(known)), (0.8, 0.8, 0.8, 1.0)))
            out.append(add(idx, ok & bad, BAD_COLOR))
            return [o for o in out if o is not None]

        # complete layers below the playing layer: cached, rebuilt only when the view settings change
        key = (id(rc), self.cmb_frame.currentIndex(), self.chk_travel.isChecked(), self.chk_bad.isChecked(),
               self.tol.value(), layer, p0)
        if self._base is None or self._base[0] != key:
            self._clear_base()
            self._base = (key, draw(np.arange(1, p0)))
        self._items = draw(np.arange(p0, stop))
        if stop > 0:
            head = gl.GLScatterPlotItem(pos=P[stop - 1:stop].astype(np.float32), size=10, color=(1, 1, 1, 1))
            self.view.addItem(head)
            self._items.append(head)
            mi = int(rc.move_of_point[stop - 1])
            if mi >= 0:
                mv = rc.moves[mi]
                self.lbl_pos.setText(f"line {mv.line} | layer {mv.layer} | {mv.feature} | "
                                     f"X{mv.p1[0]:.2f} Y{mv.p1[1]:.2f} Z{mv.p1[2]:.2f} A{mv.p1[3]:.1f} B{mv.p1[4]:.1f} C{mv.p1[5]:.1f}")

    # ------------------------------------------------------------------ playback
    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("Play")
            return
        if self.slider.value() >= self.slider.maximum():
            self.slider.setValue(0)
        self.timer.start()
        self.btn_play.setText("Pause")

    def _tick(self):
        step = max(1, self.slider.maximum() // 300)
        self.slider.setValue(min(self.slider.maximum(), self.slider.value() + step))
        if self.slider.value() >= self.slider.maximum():
            self.timer.stop()
            self.btn_play.setText("Play")
