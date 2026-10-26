"""
JSON editors for function files (cam-function) and machine files (cam-machine).

One window, one text field: the file IS the description, so everything is edited as JSON. A status
line validates the text with the real loaders on every keystroke (what the editor accepts is exactly
what the loader accepts). Apply adds the validated spec to the session, Save as... writes the file.
"""
import json
import os
from typing import Any, Dict, Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDialog, QFileDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout

from src import gui_theme
from src.common.param_spec import SpecError
from src.stage1_slicer.function_file import FORMAT as FUNCTION_FORMAT, VERSION as FUNCTION_VERSION, FunctionSpec, save_function_file
from src.stage3_kinematics.machine_file import FORMAT as MACHINE_FORMAT, VERSION as MACHINE_VERSION, MachineSpec, save_machine_file


class _JsonEditor(QDialog):
    applied = pyqtSignal(object)
    KIND = "file"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{self.KIND} editor")
        self.resize(760, 640)
        self.spec = None
        self.source = ""
        self.base_dir = ""
        self.text = QPlainTextEdit()
        self.text.setStyleSheet("font-family: monospace;")
        self.text.setTabChangesFocus(False)
        self.text.textChanged.connect(self._on_text)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.clicked.connect(self.apply)
        self.btn_save = QPushButton("Save as...")
        self.btn_save.clicked.connect(self.save_as)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        row = QHBoxLayout()
        row.addWidget(self.btn_apply)
        row.addWidget(self.btn_save)
        row.addStretch()
        row.addWidget(btn_close)
        lay = QVBoxLayout(self)
        lay.addWidget(self.text, 1)
        lay.addWidget(self.status)
        lay.addLayout(row)

    # provided by the subclasses
    def _parse(self, d: Dict[str, Any]):
        raise NotImplementedError

    def _save(self, spec, path: str):
        raise NotImplementedError

    def _template(self) -> Dict[str, Any]:
        raise NotImplementedError

    # --------------------------------------------------------------------------------------
    def load_dict(self, d: Dict[str, Any], source: str = ""):
        self.source = source
        self.text.setPlainText(json.dumps(d, indent=2, ensure_ascii=False))     # triggers validation

    def open_spec(self, spec=None):
        """spec=None starts a new file from the template."""
        if spec is None:
            self.base_dir = ""
            self.load_dict(self._template(), "")
        else:
            self.base_dir = getattr(spec, "base_dir", "") or (os.path.dirname(spec.source) if spec.source else "")
            self.load_dict(spec.to_dict(), spec.source)

    def _on_text(self):
        try:
            d = json.loads(self.text.toPlainText())
        except ValueError as e:
            self._fail(f"invalid JSON: {e}")
            return
        try:
            spec = self._parse(d)
        except (SpecError, ValueError, KeyError, TypeError) as e:
            self._fail(str(e))
            return
        self.spec = spec
        self.status.setText("OK - the file is valid.")
        self.status.setStyleSheet(f"color: {gui_theme.OK};")
        self.btn_apply.setEnabled(True)
        self.btn_save.setEnabled(True)

    def _fail(self, msg: str):
        self.spec = None
        self.status.setText("Error: " + msg)
        self.status.setStyleSheet(f"color: {gui_theme.ERR};")
        self.btn_apply.setEnabled(False)
        self.btn_save.setEnabled(False)

    def apply(self):
        if self.spec is not None:
            self.applied.emit(self.spec)

    def save_as(self):
        if self.spec is None:
            return
        fname, _ = QFileDialog.getSaveFileName(self, f"Save {self.KIND} file", self.source or f"{self.spec.name}.json",
                                               "JSON files (*.json)")
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
        self.applied.emit(self.spec)


class FunctionEditor(_JsonEditor):
    KIND = "Function"

    def _parse(self, d):
        return FunctionSpec.from_dict(d, base_dir=self.base_dir, source=self.source)

    def _save(self, spec, path):
        save_function_file(spec, path)

    def _template(self):
        return {"format": FUNCTION_FORMAT, "version": FUNCTION_VERSION, "name": "new_function",
                "description": "Layers follow z = z0 - f(x, y); x, y, r and the declared parameters can be used.",
                "expression": "a * np.sin(x / 10)",
                "parameters": {"a": {"label": "Amplitude", "unit": "mm", "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}}}


class MachineEditor(_JsonEditor):
    KIND = "Machine"

    def _parse(self, d):
        return MachineSpec.from_dict(d, source=self.source)

    def _save(self, spec, path):
        save_machine_file(spec, path)

    def _template(self):
        return {"format": MACHINE_FORMAT, "version": MACHINE_VERSION, "name": "new_machine", "description": "",
                "kinematics": "planar_3axis_xyz", "axes": ["X", "Y", "Z"],
                "parameters": {"max_3axis_tilt_deg": {"label": "Max. tilt", "unit": "deg", "default": 25}}}
