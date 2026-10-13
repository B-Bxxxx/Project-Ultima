"""
Editors with live preview for function files (cam-function) and machine files (cam-machine).

Both dialogs share one base: a Form tab and a raw JSON tab that always describe the same document,
a status line that shows the first validation error (the real file validators are used, so what the
editor accepts is exactly what the loader accepts), a preview on the right, and Apply / Save as...
buttons. `applied` is emitted with the validated spec; the main window decides what to do with it.
"""
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)

from src.common.param_spec import SpecError
from src.stage1_slicer.function_file import (
    FORMAT as FUNCTION_FORMAT, TYPES as FUNCTION_TYPES, VERSION as FUNCTION_VERSION, FunctionFileError,
    FunctionSpec, save_function_file,
)
from src.stage1_slicer.function_preview import height_map
from src.stage3_kinematics.machine_file import (
    FORMAT as MACHINE_FORMAT, VALID_AXES, VERSION as MACHINE_VERSION, MachineFileError, MachineSpec,
    save_machine_file,
)
from src.stage3_kinematics.machine_preview import outline, probe
from src.stage3_kinematics.registry import KinematicsRegistry

PARAM_COLUMNS = ["name", "label", "unit", "type", "default", "min", "max", "step", "choices"]
PARAM_TYPES = ["number", "int", "bool", "choice"]


# ----------------------------------------------------------------------------- parameter table
def _lit(text: str) -> Any:
    """A table cell: JSON literal if it parses (12, 0.5, true, "a"), otherwise the plain string."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


class ParamTable(QTableWidget):
    """One row per declared parameter; cells hold JSON literals."""
    edited = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(0, len(PARAM_COLUMNS), parent)
        self.setHorizontalHeaderLabels(PARAM_COLUMNS)
        self.horizontalHeader().setStretchLastSection(True)
        self.setMinimumHeight(120)
        self._loading = False
        self.itemChanged.connect(lambda *_: None if self._loading else self.edited.emit())

    def add_row(self, name="", d: Optional[Dict[str, Any]] = None):
        d = d or {}
        self._loading = True
        r = self.rowCount()
        self.insertRow(r)
        vals = [name, d.get("label", name), d.get("unit", ""), d.get("type", "number"),
                json.dumps(d["default"]) if "default" in d else "0",
                json.dumps(d["min"]) if "min" in d else "", json.dumps(d["max"]) if "max" in d else "",
                json.dumps(d["step"]) if "step" in d else "",
                ", ".join(json.dumps(c) for c in d.get("choices", []))]
        for c, v in enumerate(vals):
            self.setItem(r, c, QTableWidgetItem(str(v)))
        self._loading = False

    def remove_selected(self):
        rows = sorted({i.row() for i in self.selectedIndexes()}, reverse=True) or ([self.rowCount() - 1] if self.rowCount() else [])
        for r in rows:
            self.removeRow(r)
        self.edited.emit()

    def set_parameters(self, params: Dict[str, Dict[str, Any]]):
        self._loading = True
        self.setRowCount(0)
        self._loading = False
        for n, d in params.items():
            self.add_row(n, d)

    def to_parameters(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for r in range(self.rowCount()):
            cell = lambda c: (self.item(r, c).text() if self.item(r, c) else "").strip()  # noqa: E731
            name = cell(0)
            if not name:
                raise SpecError(f"parameter row {r + 1}: the name is empty")
            if name in out:
                raise SpecError(f"parameter '{name}' is declared twice")
            d: Dict[str, Any] = {"label": cell(1) or name}
            if cell(2):
                d["unit"] = cell(2)
            if cell(3) and cell(3) != "number":
                d["type"] = cell(3)
            d["default"] = _lit(cell(4))
            for c, key in ((5, "min"), (6, "max"), (7, "step")):
                if cell(c):
                    v = _lit(cell(c))
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        raise SpecError(f"parameter '{name}': {key} must be a number")
                    d[key] = v
            if cell(8):
                d["choices"] = [_lit(p) for p in cell(8).split(",") if p.strip()]
            out[name] = d
        return out


# ----------------------------------------------------------------------------- shared dialog base
class _SpecEditor(QDialog):
    applied = pyqtSignal(object)
    KIND = "file"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{self.KIND} editor")
        self.resize(1100, 640)
        self._syncing = False
        self.spec = None
        self.source = ""

        self.tabs = QTabWidget()
        form_page = QWidget()
        self.form = QFormLayout(form_page)
        self.tabs.addTab(form_page, "Form")
        self.json_edit = QPlainTextEdit()
        self.json_edit.setStyleSheet("font-family: monospace;")
        self.tabs.addTab(self.json_edit, "JSON")
        self.tabs.currentChanged.connect(self._on_tab)
        self.json_edit.textChanged.connect(self._on_json_text)

        self.table = ParamTable()
        self.table.edited.connect(self._from_form)
        row = QHBoxLayout()
        add = QPushButton("Add parameter")
        add.clicked.connect(lambda: (self.table.add_row("param"), self._from_form()))
        rem = QPushButton("Remove selected")
        rem.clicked.connect(self.table.remove_selected)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch()

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.clicked.connect(self.apply)
        self.btn_save = QPushButton("Save as...")
        self.btn_save.clicked.connect(self.save_as)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        buttons = QHBoxLayout()
        for b in (self.btn_apply, self.btn_save):
            buttons.addWidget(b)
        buttons.addStretch()
        buttons.addWidget(btn_close)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(self.tabs, 1)
        lv.addWidget(QLabel("Parameters (cells are JSON values; choices comma separated):"))
        lv.addWidget(self.table, 1)
        lv.addLayout(row)
        lv.addWidget(self.status)
        lv.addLayout(buttons)

        self.preview_box = QWidget()
        self.preview_layout = QVBoxLayout(self.preview_box)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.preview_box)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        outer = QVBoxLayout(self)
        outer.addWidget(split)

    # --- to be provided by the subclasses
    def _form_to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _dict_to_form(self, d: Dict[str, Any]):
        raise NotImplementedError

    def _parse(self, d: Dict[str, Any]):
        raise NotImplementedError

    def _show_preview(self, spec):
        raise NotImplementedError

    # --- shared flow
    def load_dict(self, d: Dict[str, Any], source: str = ""):
        self.source = source
        self._syncing = True
        self._dict_to_form(d)
        self._syncing = False
        self._refresh(d)

    def _from_form(self, *_):
        if self._syncing:
            return
        try:
            d = self._form_to_dict()
        except (SpecError, ValueError) as e:
            self._fail(str(e))
            return
        self._refresh(d)

    def _refresh(self, d: Dict[str, Any]):
        self._syncing = True
        self.json_edit.setPlainText(json.dumps(d, indent=2, ensure_ascii=False))
        self._syncing = False
        self._validate(d)

    def _validate(self, d) -> bool:
        try:
            spec = self._parse(d)
            self._show_preview(spec)
        except (SpecError, ValueError, KeyError) as e:
            self._fail(str(e))
            return False
        self.spec = spec
        self.status.setText("OK - the file is valid.")
        self.status.setStyleSheet("color: #2e7d32;")
        self.btn_apply.setEnabled(True)
        self.btn_save.setEnabled(True)
        return True

    def _fail(self, msg: str):
        self.spec = None
        self.status.setText("Error: " + msg)
        self.status.setStyleSheet("color: #c62828;")
        self.btn_apply.setEnabled(False)
        self.btn_save.setEnabled(False)

    def _on_json_text(self):
        if self._syncing or self.tabs.currentIndex() != 1:
            return
        try:
            d = json.loads(self.json_edit.toPlainText())
        except ValueError as e:
            self._fail(f"invalid JSON: {e}")
            return
        self._validate(d)

    def _on_tab(self, idx: int):
        if idx == 0 and not self._syncing:       # JSON -> form
            try:
                d = json.loads(self.json_edit.toPlainText())
                if not isinstance(d, dict):
                    raise ValueError("the document must be a JSON object")
                self._syncing = True
                self._dict_to_form(d)
                self._syncing = False
            except (ValueError, SpecError) as e:
                self._syncing = False
                self.tabs.blockSignals(True)
                self.tabs.setCurrentIndex(1)
                self.tabs.blockSignals(False)
                self._fail(f"the JSON cannot be shown in the form: {e}")

    def apply(self):
        if self.spec is not None:
            self.applied.emit(self.spec)

    def _save(self, spec, path: str):
        raise NotImplementedError

    def save_as(self):
        if self.spec is None:
            return
        start = self.source or f"{self.spec.name}.json"
        fname, _ = QFileDialog.getSaveFileName(self, f"Save {self.KIND} file", start, "JSON files (*.json)")
        if not fname:
            return
        try:
            self._save(self.spec, fname)
        except (SpecError, OSError) as e:
            self._fail(str(e))
            return
        self.source = fname
        self.spec.source = fname
        self.status.setText(f"Saved {os.path.basename(fname)}.")
        self.status.setStyleSheet("color: #2e7d32;")
        self.applied.emit(self.spec)

    @staticmethod
    def _clear(layout):
        while layout.count():
            w = layout.takeAt(0).widget()
            if w is not None:
                w.deleteLater()


# ----------------------------------------------------------------------------- function editor
class FunctionEditor(_SpecEditor):
    KIND = "Function"

    def __init__(self, parent=None, bounds=(-20.0, 20.0, -20.0, 20.0)):
        super().__init__(parent)
        self.setWindowTitle("Function editor")
        self.base_dir = ""
        self.ed_name = QLineEdit()
        self.ed_desc = QLineEdit()
        self.cb_type = QComboBox()
        self.cb_type.addItems(list(FUNCTION_TYPES))
        self.ed_expr = QLineEdit()
        self.ed_expr.setPlaceholderText("e.g. a * np.sin(x / b)   (x, y, r, np.* and declared parameters)")
        self.ed_data = QLineEdit()
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse)
        data_row = QWidget()
        dl = QHBoxLayout(data_row)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.addWidget(self.ed_data)
        dl.addWidget(self.btn_browse)
        self.form.addRow("Name:", self.ed_name)
        self.form.addRow("Description:", self.ed_desc)
        self.form.addRow("Type:", self.cb_type)
        self.lbl_expr = QLabel("Expression:")
        self.form.addRow(self.lbl_expr, self.ed_expr)
        self.lbl_data = QLabel("Data file:")
        self.form.addRow(self.lbl_data, data_row)
        for w in (self.ed_name, self.ed_desc, self.ed_expr, self.ed_data):
            w.textEdited.connect(self._from_form)
        self.cb_type.currentTextChanged.connect(self._on_type)

        # preview: height map of the LAST layer + the area it is drawn over
        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.setLabel("bottom", "x [mm]")
        self.plot.setLabel("left", "y [mm]")
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        try:
            self.image.setColorMap(pg.colormap.get("viridis"))
        except Exception:
            pass
        self.lbl_range = QLabel("")
        self.lbl_range.setWordWrap(True)
        self.area = []
        area_row = QHBoxLayout()
        for lab, val in zip(("x0", "x1", "y0", "y1"), bounds):
            area_row.addWidget(QLabel(lab))
            sp = QDoubleSpinBox()
            sp.setRange(-5000, 5000)
            sp.setDecimals(1)
            sp.setValue(float(val))
            sp.valueChanged.connect(self._from_form)
            area_row.addWidget(sp)
            self.area.append(sp)
        self.preview_layout.addWidget(QLabel("Preview: shape of the last layer (shift of z in mm; layers follow z = z0 - f)"))
        self.preview_layout.addLayout(area_row)
        self.preview_layout.addWidget(self.plot, 1)
        self.preview_layout.addWidget(self.lbl_range)
        self._on_type(self.cb_type.currentText())

    def set_bounds(self, bounds):
        for sp, v in zip(self.area, bounds):
            sp.blockSignals(True)
            sp.setValue(float(v))
            sp.blockSignals(False)

    def _on_type(self, typ: str):
        for w in (self.lbl_expr, self.ed_expr):
            w.setVisible(typ == "expression")
        for w in (self.lbl_data, self.ed_data, self.btn_browse):
            w.setVisible(typ == "grid")
        self._from_form()

    def _browse(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open data file", self.base_dir, "Data files (*.csv *.txt *.npy);;All files (*)")
        if fname:
            self.ed_data.setText(fname)
            self._from_form()

    def _form_to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"format": FUNCTION_FORMAT, "version": FUNCTION_VERSION, "name": self.ed_name.text().strip()}
        if self.ed_desc.text().strip():
            d["description"] = self.ed_desc.text().strip()
        typ = self.cb_type.currentText()
        d["type"] = typ
        if typ == "expression":
            d["expression"] = self.ed_expr.text().strip()
        if typ == "grid":
            d["data_file"] = self.ed_data.text().strip()
        d["parameters"] = self.table.to_parameters()
        return d

    def _dict_to_form(self, d: Dict[str, Any]):
        self.ed_name.setText(str(d.get("name", "")))
        self.ed_desc.setText(str(d.get("description", "")))
        typ = d.get("type", "planar")
        if self.cb_type.findText(typ) >= 0:
            self.cb_type.setCurrentText(typ)
        self.ed_expr.setText(str(d.get("expression") or ""))
        self.ed_data.setText(str(d.get("data_file") or ""))
        self.table.set_parameters(d.get("parameters") or {})
        self._on_type_visibility(self.cb_type.currentText())

    def _on_type_visibility(self, typ: str):
        for w in (self.lbl_expr, self.ed_expr):
            w.setVisible(typ == "expression")
        for w in (self.lbl_data, self.ed_data, self.btn_browse):
            w.setVisible(typ == "grid")

    def _parse(self, d) -> FunctionSpec:
        return FunctionSpec.from_dict(d, base_dir=self.base_dir, source=self.source)

    def _show_preview(self, spec: FunctionSpec):
        x0, x1, y0, y1 = (s.value() for s in self.area)
        hm = height_map(spec, None, (x0, x1, y0, y1))
        self.image.setImage(hm.z.T, autoLevels=False, levels=(hm.zmin, hm.zmax if hm.zmax > hm.zmin else hm.zmin + 1.0))
        self.image.setRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        self.plot.setXRange(x0, x1)
        self.plot.setYRange(y0, y1)
        self.lbl_range.setText(f"{hm.label}\nshift of the last layer: {hm.zmin:.2f} .. {hm.zmax:.2f} mm "
                               f"(range {hm.zmax - hm.zmin:.2f} mm)")

    def _save(self, spec, path):
        save_function_file(spec, path)

    def open_spec(self, spec: Optional[FunctionSpec], template: Optional[dict] = None):
        """spec=None starts a new function from `template` (or a planar one)."""
        if spec is not None:
            self.base_dir = spec.base_dir or (os.path.dirname(spec.source) if spec.source else "")
            self.load_dict(spec.to_dict(), spec.source)
        else:
            self.base_dir = ""
            self.load_dict(template or {"format": FUNCTION_FORMAT, "version": FUNCTION_VERSION, "name": "new_function",
                                        "description": "", "type": "planar", "parameters": {}}, "")


# ----------------------------------------------------------------------------- machine editor
class MachineEditor(_SpecEditor):
    KIND = "Machine"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Machine editor")
        self.ed_name = QLineEdit()
        self.ed_desc = QLineEdit()
        self.cb_kin = QComboBox()
        self.cb_kin.addItems(sorted(KinematicsRegistry.get_all_solvers()))
        self.axis_boxes = {}
        axes_row = QWidget()
        al = QHBoxLayout(axes_row)
        al.setContentsMargins(0, 0, 0, 0)
        for a in VALID_AXES:
            cb = QCheckBox(a)
            cb.setChecked(a in "XYZ")
            cb.stateChanged.connect(self._from_form)
            self.axis_boxes[a] = cb
            al.addWidget(cb)
        self.chk_volume = QCheckBox("Build volume")
        self.chk_volume.stateChanged.connect(self._from_form)
        vol_row = QWidget()
        vl = QHBoxLayout(vol_row)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(self.chk_volume)
        self.vol = {}
        for ax in "xyz":
            for k, lab in enumerate(("min", "max")):
                sp = QDoubleSpinBox()
                sp.setRange(-5000, 5000)
                sp.setDecimals(1)
                sp.setValue(0.0 if k == 0 else 200.0)
                sp.valueChanged.connect(self._from_form)
                self.vol[(ax, k)] = sp
                vl.addWidget(QLabel(f"{ax}{lab}"))
                vl.addWidget(sp)
        self.form.addRow("Name:", self.ed_name)
        self.form.addRow("Description:", self.ed_desc)
        self.form.addRow("Kinematics:", self.cb_kin)
        self.form.addRow("Axes in G-code:", axes_row)
        self.form.addRow(vol_row)
        for w in (self.ed_name, self.ed_desc):
            w.textEdited.connect(self._from_form)
        self.cb_kin.currentTextChanged.connect(self._from_form)

        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.setLabel("bottom", "X [mm]")
        self.plot.setLabel("left", "Z [mm]")
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.sp_tilt = QDoubleSpinBox()
        self.sp_tilt.setRange(0, 89)
        self.sp_tilt.setValue(20.0)
        self.sp_az = QDoubleSpinBox()
        self.sp_az.setRange(-360, 360)
        self.sp_az.setValue(0.0)
        for s in (self.sp_tilt, self.sp_az):
            s.valueChanged.connect(self._from_form)
        probe_row = QHBoxLayout()
        probe_row.addWidget(QLabel("Probe: surface tilt [deg]"))
        probe_row.addWidget(self.sp_tilt)
        probe_row.addWidget(QLabel("direction [deg]"))
        probe_row.addWidget(self.sp_az)
        self.lbl_probe = QLabel("")
        self.lbl_probe.setWordWrap(True)
        self.preview_layout.addWidget(QLabel("Preview: side view (XZ) with the nozzle direction at the B limits"))
        self.preview_layout.addWidget(self.plot, 1)
        self.preview_layout.addWidget(self.summary)
        self.preview_layout.addLayout(probe_row)
        self.preview_layout.addWidget(self.lbl_probe)

    def _form_to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"format": MACHINE_FORMAT, "version": MACHINE_VERSION, "name": self.ed_name.text().strip()}
        if self.ed_desc.text().strip():
            d["description"] = self.ed_desc.text().strip()
        d["kinematics"] = self.cb_kin.currentText()
        d["axes"] = [a for a in VALID_AXES if self.axis_boxes[a].isChecked()]
        if self.chk_volume.isChecked():
            d["build_volume"] = {ax: [self.vol[(ax, 0)].value(), self.vol[(ax, 1)].value()] for ax in "xyz"}
        d["parameters"] = self.table.to_parameters()
        return d

    def _dict_to_form(self, d: Dict[str, Any]):
        self.ed_name.setText(str(d.get("name", "")))
        self.ed_desc.setText(str(d.get("description", "")))
        if self.cb_kin.findText(str(d.get("kinematics", ""))) >= 0:
            self.cb_kin.setCurrentText(d["kinematics"])
        axes = [str(a).upper() for a in d.get("axes", ["X", "Y", "Z"])]
        for a, cb in self.axis_boxes.items():
            cb.setChecked(a in axes)
        bv = d.get("build_volume")
        self.chk_volume.setChecked(bool(bv))
        if isinstance(bv, dict):
            for ax in "xyz":
                if isinstance(bv.get(ax), list) and len(bv[ax]) == 2:
                    for k in (0, 1):
                        self.vol[(ax, k)].setValue(float(bv[ax][k]))
        self.table.set_parameters(d.get("parameters") or {})

    def _parse(self, d) -> MachineSpec:
        return MachineSpec.from_dict(d, source=self.source)

    def _show_preview(self, spec: MachineSpec):
        o = outline(spec)
        self.plot.clear()
        vol = o["volume"]
        xs = vol["x"] if vol else [-100.0, 100.0]
        zs = vol["z"] if vol else [0.0, 200.0]
        self.plot.plot([xs[0], xs[1], xs[1], xs[0], xs[0]], [zs[0], zs[0], zs[1], zs[1], zs[0]],
                       pen=pg.mkPen("#888", width=2))
        cx, cz = (xs[0] + xs[1]) / 2.0, zs[0] + 0.3 * (zs[1] - zs[0])
        length = 0.25 * max(zs[1] - zs[0], xs[1] - xs[0])
        for b, dx, dz in o["fan"]:
            col = "#2e7d32" if abs(b) < 1e-9 else "#d84315"
            self.plot.plot([cx, cx + dx * length], [cz, cz + dz * length], pen=pg.mkPen(col, width=3))
        if not o["fan"]:
            self.plot.plot([cx, cx], [cz, cz + length], pen=pg.mkPen("#2e7d32", width=3))
        self.summary.setText("\n".join(o["summary"]))
        r = probe(spec, None, self.sp_tilt.value(), self.sp_az.value())
        self.lbl_probe.setText(
            f"B = {r['b']:.1f}, C = {r['c']:.1f} deg; nozzle misalignment {r['angle_error_deg']:.1f} deg; "
            + ("reachable" if r["reached"] else "NOT reachable (limit hit)"))

    def _save(self, spec, path):
        save_machine_file(spec, path)

    def open_spec(self, spec: Optional[MachineSpec], template: Optional[dict] = None):
        if spec is not None:
            self.load_dict(spec.to_dict(), spec.source)
        else:
            self.load_dict(template or {"format": MACHINE_FORMAT, "version": MACHINE_VERSION, "name": "new_machine",
                                        "description": "", "kinematics": "planar_3axis_xyz", "axes": ["X", "Y", "Z"],
                                        "parameters": {"max_3axis_tilt_deg": {"label": "Max. tilt", "unit": "deg", "default": 25}}}, "")
