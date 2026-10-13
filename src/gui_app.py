import os
import sys
import time

import numpy as np
import trimesh
from PyQt6.QtCore import QLocale, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, QLabel,
    QCheckBox, QTabWidget, QPlainTextEdit, QFileDialog, QLineEdit, QStackedWidget, QScrollArea,
    QToolButton, QFrame, QSizePolicy,
)
from PyQt6.QtWidgets import QComboBox as _QComboBox
from PyQt6.QtWidgets import QDoubleSpinBox as _QDoubleSpinBox
from PyQt6.QtWidgets import QSlider as _QSlider
from PyQt6.QtWidgets import QSpinBox as _QSpinBox
from pyqtgraph import Vector
import pyqtgraph.opengl as gl

from src.common.feature_styles import FEATURE_STYLES, group_lines_by_feature
from src.common.placement import place_mesh
from src.common.schemas import MinimalPrintProfile
from src.common.view_model import (Playback, effective_layers, head_item, layer_count, move_range,
                                   status_text)
from src.common.param_spec import ParamSpec
from src.gui_editors import FunctionEditor, MachineEditor
from src import gui_theme
from src.stage1_slicer.function_file import (
    FunctionFileError, FunctionSpec, load_function_dir, load_function_file, save_function_file,
)
from src.stage1_slicer.function_import import ExpressionField, GridField, preview_surface
from src.stage1_slicer.registry import PluginRegistry
from src.stage1_slicer.transition_advice import advise, field_range
from src.stage2_toolpath.contour_generator import StandardToolpathGenerator
from src.stage3_kinematics.machine_file import (
    MachineFileError, MachineSpec, load_machine_dir, load_machine_file, save_machine_file,
)
from src.stage3_kinematics.registry import KinematicsRegistry
from src.stage4_postproc.klipper_postproc import Klipper5AxisPostProcessor


# The settings column scrolls with the mouse wheel. Spin boxes, combo boxes and sliders must NOT
# react to the wheel (a value changing while the user scrolls the menu is a nasty surprise): they
# ignore the event, which then reaches the scroll area. Keyboard and mouse editing are unchanged.
class _NoWheel:
    def wheelEvent(self, event):
        event.ignore()


class QDoubleSpinBox(_NoWheel, _QDoubleSpinBox):
    """
    Decimal point AND decimal comma are accepted whatever the system language is. (With a Hungarian
    locale Qt treats "." as a group separator and silently turns the typed "0.5" into 5.)
    The value is always shown with a point, like in the JSON files.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setLocale(QLocale.c())
        self.setGroupSeparatorShown(False)
        self.setKeyboardTracking(False)     # commit on Enter / Tab / arrows, not on every typed digit

    def validate(self, text, pos):
        return super().validate(text.replace(",", "."), pos)

    def valueFromText(self, text):
        return super().valueFromText(text.replace(",", "."))


class QSpinBox(_NoWheel, _QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setKeyboardTracking(False)


class QComboBox(_NoWheel, _QComboBox):
    pass


class QSlider(_NoWheel, _QSlider):
    pass


CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")


class ParameterForm(QWidget):
    """Builds one named input field per declared parameter (function files, machine files)."""
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._form = QFormLayout(self)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._widgets = {}
        self._specs = {}

    def set_specs(self, specs, values=None):
        while self._form.rowCount():
            self._form.removeRow(0)
        self._widgets, self._specs = {}, dict(specs)
        for name, sp in specs.items():
            w = self._make(sp, (values or {}).get(name, sp.default))
            label = sp.label or name
            if sp.unit:
                label += f" ({sp.unit})"
            w.setToolTip(f"{name}" + (f" - {sp.description}" if sp.description else ""))
            self._form.addRow(label + ":", w)
            self._widgets[name] = w

    def _make(self, sp: ParamSpec, value):
        if sp.type == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            w.stateChanged.connect(lambda *_: self.changed.emit())
        elif sp.type == "choice":
            w = QComboBox()
            w.addItems([str(c) for c in sp.choices])
            w.setCurrentIndex(sp.choices.index(value))
            w.currentIndexChanged.connect(lambda *_: self.changed.emit())
        elif sp.type == "int":
            w = QSpinBox()
            w.setRange(int(sp.min if sp.min is not None else -10**9), int(sp.max if sp.max is not None else 10**9))
            w.setSingleStep(int(sp.step or 1))
            w.setValue(int(value))
            w.valueChanged.connect(lambda *_: self.changed.emit())
        else:
            w = QDoubleSpinBox()
            w.setDecimals(4)
            w.setRange(sp.min if sp.min is not None else -1e9, sp.max if sp.max is not None else 1e9)
            w.setSingleStep(sp.step or 0.1)
            w.setValue(float(value))
            w.valueChanged.connect(lambda *_: self.changed.emit())
        return w

    def values(self):
        out = {}
        for name, w in self._widgets.items():
            sp = self._specs[name]
            if sp.type == "bool":
                out[name] = w.isChecked()
            elif sp.type == "choice":
                out[name] = sp.choices[w.currentIndex()]
            else:
                out[name] = w.value()
        return out

    def names(self):
        return list(self._widgets)


class CollapsibleSection(QWidget):
    """A header button that folds its content away, so the long settings column stays readable."""

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self.button = QToolButton()
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.button.toggled.connect(self._toggle)

        self.content = QWidget()
        self.form = QFormLayout(self.content)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.content.setVisible(expanded)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.button)
        lay.addWidget(self.content)
        lay.addWidget(line)

    def _toggle(self, on: bool):
        self.button.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.content.setVisible(on)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(gui_theme.NAME)
        self.setStyleSheet(gui_theme.stylesheet())
        self.setGeometry(100, 100, 1500, 900)

        self.source_mesh = None      # geometry as generated / loaded
        self.mesh = None             # source_mesh after placement (what is sliced)
        self.stage1_model = None
        self.stage2_traj = None
        self.stage3_traj = None
        self.profile = None
        self.gcode = ""

        self._grid_origin = (0.0, 0.0)
        self.playback = Playback()
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._on_tick)
        self._dynamic = []           # (widget, item) pairs redrawn on every layer/playback change
        self._ranges = {}            # stage name -> (start, stop) item indices of the visible layers

        self.init_ui()
        self.load_geometry()
        for w in (self.trans_height_spin, self.layer_height_spin, self.nozzle_spin):
            w.valueChanged.connect(self.update_transition_advice)
        self.function_form.changed.connect(self.update_transition_advice)
        self.function_combo.currentTextChanged.connect(self.update_transition_advice)
        self.update_transition_advice()

    # ------------------------------------------------------------------------------- UI
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- LEFT PANEL: scrollable, collapsible sections + pinned action buttons ---
        left_panel = QWidget()
        left_panel.setFixedWidth(430)
        left_outer = QVBoxLayout(left_panel)
        left_outer.setContentsMargins(0, 0, 0, 0)

        brand = QLabel(f'<span style="color:{gui_theme.TEXT}">ULTIMA</span> <span style="color:{gui_theme.CORAL}">slicer</span>')
        brand.setObjectName("brand")
        sub = QLabel("non-planar CAM")
        sub.setObjectName("brandSub")
        left_outer.addWidget(brand)
        left_outer.addWidget(sub)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        self.sections_layout = QVBoxLayout(inner)
        self.sections_layout.setContentsMargins(4, 4, 8, 4)
        scroll.setWidget(inner)
        left_outer.addWidget(scroll, 1)

        self._build_geometry_section()
        self._build_placement_section()
        self._build_function_section()
        self._build_slicer_section()
        self._build_profile_section()
        self._build_kinematics_section()
        self._build_view_section()
        self.sections_layout.addStretch()

        # pinned actions
        self.btn_run = QPushButton("Run CAM Pipeline")
        self.btn_run.setObjectName("runButton")
        self.btn_run.clicked.connect(self.run_pipeline)
        left_outer.addWidget(self.btn_run)
        self.btn_export = QPushButton("Export G-Code (.gcode)")
        self.btn_export.clicked.connect(self.export_gcode)
        left_outer.addWidget(self.btn_export)
        self.lbl_metrics = QLabel("Ready")
        self.lbl_metrics.setWordWrap(True)
        left_outer.addWidget(self.lbl_metrics)
        layout.addWidget(left_panel)

        # --- RIGHT PANEL ---
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()

        def gl_tab(title):
            w = gl.GLViewWidget()
            w.opts['distance'] = 80
            w.setBackgroundColor(gui_theme.GL_BACKGROUND)
            self.tabs.addTab(w, title)
            return w

        self.gl_stage0 = gl_tab("0. Stage 0 (Mesh)")
        self.gl_stage1 = gl_tab("1. Stage 1 (Raw Slices)")
        self.gl_stage2 = gl_tab("2. Stage 2 (Toolpaths)")
        self.gl_kine = gl_tab("3. Kinematics")

        self.txt_gcode = QPlainTextEdit()
        self.txt_gcode.setReadOnly(True)
        self.txt_gcode.setStyleSheet("font-family: monospace;")
        self.tabs.addTab(self.txt_gcode, "4. G-Code Preview")
        right_lay.addWidget(self.tabs, 1)

        self.lbl_status = QLabel("-")
        self.lbl_status.setStyleSheet("font-family: monospace;")
        right_lay.addWidget(self.lbl_status)
        layout.addWidget(right, 1)

    def _section(self, title, expanded=True) -> CollapsibleSection:
        sec = CollapsibleSection(title, expanded)
        self.sections_layout.addWidget(sec)
        return sec

    @staticmethod
    def _dspin(value, lo, hi, step, decimals=2):
        s = QDoubleSpinBox()
        s.setDecimals(decimals)
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(value)
        return s

    @staticmethod
    def _ispin(value, lo, hi):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(value)
        return s

    def _build_geometry_section(self):
        sec = self._section("1. Input Geometry")
        f = sec.form
        self.geom_combo = QComboBox()
        self.geom_combo.addItems(["Cylinder", "Hollow Cylinder", "Cube", "Upload STL..."])
        self.geom_combo.currentIndexChanged.connect(self.on_geom_changed)
        f.addRow("Geometry:", self.geom_combo)

        self.geom_stack = QStackedWidget()
        self.cyl_widget = QWidget()
        cl = QFormLayout(self.cyl_widget)
        self.cyl_r = self._dspin(15.0, 0.5, 500, 1)
        self.cyl_h = self._dspin(20.0, 0.5, 500, 1)
        self.cyl_wall = self._dspin(5.0, 0.2, 500, 0.5)
        cl.addRow("Radius:", self.cyl_r)
        cl.addRow("Height:", self.cyl_h)
        cl.addRow("Wall Thickness:", self.cyl_wall)
        self.geom_stack.addWidget(self.cyl_widget)

        self.cube_widget = QWidget()
        bl = QFormLayout(self.cube_widget)
        self.cube_w = self._dspin(30.0, 1, 500, 1)
        self.cube_d = self._dspin(30.0, 1, 500, 1)
        self.cube_h = self._dspin(20.0, 0.5, 500, 1)
        bl.addRow("Width:", self.cube_w)
        bl.addRow("Depth:", self.cube_d)
        bl.addRow("Height:", self.cube_h)
        self.geom_stack.addWidget(self.cube_widget)

        self.stl_widget = QWidget()
        sl = QVBoxLayout(self.stl_widget)
        self.lbl_stl = QLabel("No STL loaded.")
        sl.addWidget(self.lbl_stl)
        self.geom_stack.addWidget(self.stl_widget)

        f.addRow(self.geom_stack)
        self.btn_geom = QPushButton("Generate/Load Mesh")
        self.btn_geom.clicked.connect(self.load_geometry)
        f.addRow(self.btn_geom)

    def _build_placement_section(self):
        sec = self._section("1b. Placement", expanded=False)
        f = sec.form
        self.rot_x = self._dspin(0, -360, 360, 5, 1)
        self.rot_y = self._dspin(0, -360, 360, 5, 1)
        self.rot_z = self._dspin(0, -360, 360, 5, 1)
        self.off_x = self._dspin(0, -1000, 1000, 1, 1)
        self.off_y = self._dspin(0, -1000, 1000, 1, 1)
        self.off_z = self._dspin(0, -1000, 1000, 1, 1)
        self.off_z_label = QLabel("Offset Z (mm):")
        self.chk_center = QCheckBox("Center on bed (XY)")
        self.chk_center.setChecked(True)
        self.chk_drop = QCheckBox("Drop to bed (Z = 0)")
        self.chk_drop.setChecked(True)
        f.addRow("Rotate X (deg):", self.rot_x)
        f.addRow("Rotate Y (deg):", self.rot_y)
        f.addRow("Rotate Z (deg):", self.rot_z)
        f.addRow("Offset X (mm):", self.off_x)
        f.addRow("Offset Y (mm):", self.off_y)
        f.addRow(self.off_z_label, self.off_z)
        f.addRow(self.chk_center)
        f.addRow(self.chk_drop)
        btn = QPushButton("Reset placement")
        btn.clicked.connect(self.reset_placement)
        f.addRow(btn)
        self.lbl_mesh_info = QLabel("")
        f.addRow(self.lbl_mesh_info)
        for w in (self.rot_x, self.rot_y, self.rot_z, self.off_x, self.off_y, self.off_z):
            w.valueChanged.connect(self.apply_placement)
        for c in (self.chk_center, self.chk_drop):
            c.stateChanged.connect(self.apply_placement)
        self._update_z_offset_visibility()

    def _build_function_section(self):
        sec = self._section("2. Function (deformation)", expanded=True)
        f = sec.form
        self.function_combo = QComboBox()
        f.addRow("Function:", self.function_combo)
        self.lbl_function = QLabel("")
        self.lbl_function.setWordWrap(True)
        f.addRow(self.lbl_function)
        self.function_form = ParameterForm()
        f.addRow(self.function_form)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        b1 = QPushButton("Load...")
        b1.clicked.connect(self.load_function_from_file)
        b2 = QPushButton("Save as...")
        b2.clicked.connect(self.save_function_to_file)
        b3 = QPushButton("Reload")
        b3.clicked.connect(self.reload_functions)
        b4 = QPushButton("Edit...")
        b4.clicked.connect(lambda: self.open_function_editor(False))
        b5 = QPushButton("New...")
        b5.clicked.connect(lambda: self.open_function_editor(True))
        for b in (b1, b2, b3):
            rl.addWidget(b)
        f.addRow(row)
        row2 = QWidget()
        r2 = QHBoxLayout(row2)
        r2.setContentsMargins(0, 0, 0, 0)
        r2.addWidget(b4)
        r2.addWidget(b5)
        f.addRow(row2)
        self.chk_field_preview = QCheckBox("Preview layer surface in Stage 0")
        self.chk_field_preview.stateChanged.connect(self.update_viewport)
        f.addRow(self.chk_field_preview)
        note = QLabel("x, y = coordinates of the PLACED part on the bed; layers follow z = z0 - f(x, y).")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {gui_theme.MUTED};")
        f.addRow(note)
        self.functions = {}
        self.function_combo.currentTextChanged.connect(self.on_function_changed)
        self.function_form.changed.connect(self._on_function_values_changed)
        self.reload_functions()

    def _build_slicer_section(self):
        sec = self._section("3. Stage 1 Slicer")
        f = sec.form
        self.layer_height_spin = self._dspin(0.4, 0.01, 5, 0.1)
        f.addRow("Layer Height:", self.layer_height_spin)
        self.trans_height_spin = self._dspin(2.0, 0.0, 100, 0.5)
        self.trans_height_spin.valueChanged.connect(self.update_viewport)
        f.addRow("Transition H:", self.trans_height_spin)
        self.lbl_transition = QLabel("")
        self.lbl_transition.setWordWrap(True)
        f.addRow(self.lbl_transition)
        self.btn_suggested = QPushButton("Use suggested transition")
        self.btn_suggested.clicked.connect(self.use_suggested_transition)
        self.btn_suggested.setVisible(False)
        f.addRow(self.btn_suggested)

    def _build_profile_section(self):
        sec = self._section("4. Stage 2/4 Profile", expanded=False)
        f = sec.form
        self.nozzle_spin = self._dspin(0.4, 0.1, 2.0, 0.05)
        f.addRow("Nozzle Dia:", self.nozzle_spin)
        self.perims_spin = self._ispin(2, 0, 20)
        f.addRow("Perimeters:", self.perims_spin)
        self.infill_spin = self._dspin(0.2, 0.0, 1.0, 0.05)
        f.addRow("Infill Density (0-1):", self.infill_spin)
        self.bottom_spin = self._ispin(3, 0, 50)
        f.addRow("Bottom skin layers:", self.bottom_spin)
        self.top_spin = self._ispin(3, 0, 50)
        f.addRow("Top skin layers:", self.top_spin)
        self.travel_cmd_combo = QComboBox()
        self.travel_cmd_combo.addItems(["G0", "G1"])
        f.addRow("Travel command:", self.travel_cmd_combo)
        self.retract_spin = self._dspin(0.8, 0.0, 20.0, 0.1)
        f.addRow("Retract (mm):", self.retract_spin)
        self.unretract_spin = self._dspin(0.8, 0.0, 20.0, 0.1)
        f.addRow("Unretract / prime (mm):", self.unretract_spin)
        self.retract_speed_spin = self._dspin(2400, 60, 30000, 100, 0)
        f.addRow("Retract speed (mm/min):", self.retract_speed_spin)
        self.unretract_speed_spin = self._dspin(2400, 60, 30000, 100, 0)
        f.addRow("Unretract speed (mm/min):", self.unretract_speed_spin)
        self.zhop_spin = self._dspin(0.4, 0.0, 10.0, 0.1)
        f.addRow("Z-hop (mm):", self.zhop_spin)
        self.hop_short_chk = QCheckBox("Z-hop on short travels too")
        f.addRow(self.hop_short_chk)
        self.park_spin = self._dspin(30.0, 0.0, 500.0, 5.0, 1)
        f.addRow("End lift above part (mm):", self.park_spin)
        self.overlap_spin = self._dspin(0.15, 0.0, 0.5, 0.05)
        f.addRow("Infill overlap (x nozzle):", self.overlap_spin)
        self.thickness_combo = QComboBox()
        self.thickness_combo.addItems(["auto", "stage1", "nozzle_axis"])
        self.thickness_combo.setToolTip("auto: nozzle_axis on machines with rotary axes, stage1 on 3-axis machines")
        f.addRow("Thickness model:", self.thickness_combo)

    def _build_kinematics_section(self):
        sec = self._section("5. Machine", expanded=False)
        f = sec.form
        self.machine_combo = QComboBox()
        f.addRow("Machine:", self.machine_combo)
        self.lbl_machine = QLabel("")
        self.lbl_machine.setWordWrap(True)
        f.addRow(self.lbl_machine)
        self.machine_form = ParameterForm()
        f.addRow(self.machine_form)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        b1 = QPushButton("Load...")
        b1.clicked.connect(self.load_machine_from_file)
        b2 = QPushButton("Save as...")
        b2.clicked.connect(self.save_machine_to_file)
        b3 = QPushButton("Reload")
        b3.clicked.connect(self.reload_machines)
        b4 = QPushButton("Edit...")
        b4.clicked.connect(lambda: self.open_machine_editor(False))
        b5 = QPushButton("New...")
        b5.clicked.connect(lambda: self.open_machine_editor(True))
        for b in (b1, b2, b3):
            rl.addWidget(b)
        f.addRow(row)
        row2 = QWidget()
        r2 = QHBoxLayout(row2)
        r2.setContentsMargins(0, 0, 0, 0)
        r2.addWidget(b4)
        r2.addWidget(b5)
        f.addRow(row2)
        self.machines = {}
        self.machine_combo.currentTextChanged.connect(self.on_machine_changed)
        self.reload_machines()

    def _build_view_section(self):
        sec = self._section("6. View", expanded=True)
        f = sec.form

        # one checkbox per line type; the coloured label is the legend, the description sits below it
        self.feature_checks = {}
        for st in FEATURE_STYLES:
            chk = QCheckBox(f"■ {st.label}")
            chk.setStyleSheet(f"QCheckBox {{ color: {st.hex}; font-weight: bold; }}")
            chk.setToolTip(st.description)
            chk.setChecked(st.key != "travel")
            chk.stateChanged.connect(self._refresh_paths)
            desc = QLabel(st.description)
            desc.setWordWrap(True)
            desc.setStyleSheet(f"color: {gui_theme.MUTED}; margin-left: 22px;")
            f.addRow(chk)
            f.addRow(desc)
            self.feature_checks[st.key] = chk

        self.layer_lo = QSlider(Qt.Orientation.Horizontal)
        self.layer_hi = QSlider(Qt.Orientation.Horizontal)
        for s in (self.layer_lo, self.layer_hi):
            s.setRange(1, 1)
        self.lbl_layers = QLabel("Layers: -")
        f.addRow(self.lbl_layers)
        f.addRow("First layer:", self.layer_lo)
        f.addRow("Last layer:", self.layer_hi)
        self.chk_single = QCheckBox("Show only the last layer")
        f.addRow(self.chk_single)
        self.layer_lo.valueChanged.connect(self.on_layer_range_changed)
        self.layer_hi.valueChanged.connect(self.on_layer_range_changed)
        self.chk_single.stateChanged.connect(self.on_layer_range_changed)

        self.play_slider = QSlider(Qt.Orientation.Horizontal)
        self.play_slider.setRange(0, 0)
        self.play_slider.valueChanged.connect(self.on_play_slider)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self.toggle_play)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["1x", "2x", "5x", "10x", "25x"])
        self.speed_combo.currentIndexChanged.connect(self.on_speed_changed)
        rl.addWidget(self.btn_play)
        rl.addWidget(self.speed_combo)
        f.addRow("Playback:", row)
        f.addRow(self.play_slider)

    # ------------------------------------------------------------------ small slots
    def on_geom_changed(self):
        txt = self.geom_combo.currentText()
        if "Cylinder" in txt:
            self.geom_stack.setCurrentWidget(self.cyl_widget)
        elif "Cube" in txt:
            self.geom_stack.setCurrentWidget(self.cube_widget)
        elif "Upload" in txt:
            self.geom_stack.setCurrentWidget(self.stl_widget)

    # ------------------------------------------------------------------ geometry / placement
    def load_geometry(self):
        gtype = self.geom_combo.currentText()
        src = self.source_mesh
        if gtype == "Cylinder":
            src = trimesh.creation.cylinder(radius=self.cyl_r.value(), height=self.cyl_h.value())
        elif gtype == "Hollow Cylinder":
            r_out = self.cyl_r.value()
            r_in = max(0.1, r_out - self.cyl_wall.value())
            src = trimesh.creation.annulus(r_min=r_in, r_max=r_out, height=self.cyl_h.value())
        elif gtype == "Cube":
            src = trimesh.creation.box(extents=(self.cube_w.value(), self.cube_d.value(), self.cube_h.value()))
        elif gtype == "Upload STL...":
            fname, _ = QFileDialog.getOpenFileName(self, "Open STL", "", "STL Files (*.stl)")
            if fname:
                loaded = trimesh.load(fname, file_type='stl', force='mesh')
                src = loaded
                self.lbl_stl.setText(fname.replace("\\", "/").split('/')[-1])
        self.source_mesh = src
        self.apply_placement()

    def _update_z_offset_visibility(self):
        """The Z offset only makes sense when the part is NOT dropped to the bed."""
        show = not self.chk_drop.isChecked()
        self.off_z_label.setVisible(show)
        self.off_z.setVisible(show)

    def reset_placement(self):
        widgets = (self.rot_x, self.rot_y, self.rot_z, self.off_x, self.off_y, self.off_z,
                   self.chk_center, self.chk_drop)
        for w in widgets:
            w.blockSignals(True)
        for s in (self.rot_x, self.rot_y, self.rot_z, self.off_x, self.off_y, self.off_z):
            s.setValue(0)
        self.chk_center.setChecked(True)
        self.chk_drop.setChecked(True)
        for w in widgets:
            w.blockSignals(False)
        self._update_z_offset_visibility()
        self.apply_placement()

    def apply_placement(self):
        """Re-places the source mesh. Results of earlier pipeline runs no longer match and are dropped."""
        self._update_z_offset_visibility()
        if self.source_mesh is None:
            return
        had_results = self.stage1_model is not None
        self.mesh = place_mesh(
            self.source_mesh,
            rotation_deg=(self.rot_x.value(), self.rot_y.value(), self.rot_z.value()),
            center_xy=self.chk_center.isChecked(), drop_to_bed=self.chk_drop.isChecked(),
            offset_xy=(self.off_x.value(), self.off_y.value()), offset_z=self.off_z.value())
        self.stage1_model = self.stage2_traj = self.stage3_traj = None
        self.gcode = ""
        self.txt_gcode.clear()
        self.playback.reset(0)
        self._stop_timer()
        e = self.mesh.extents
        self.lbl_mesh_info.setText(f"Size: {e[0]:.1f} x {e[1]:.1f} x {e[2]:.1f} mm | "
                                   f"Z {self.mesh.bounds[0, 2]:.1f}..{self.mesh.bounds[1, 2]:.1f}")
        if had_results:
            self.lbl_metrics.setText("Placement changed - run the pipeline again.")
        self._setup_view_controls()
        self.frame_views()
        self.update_viewport()

    def frame_views(self):
        """
        Points all 3D views at the part. X and Y of the camera (and of the floor grid) follow the part
        wherever it was moved; Z is never faked - the grid stays on the bed (z = 0) so the part's
        height above the bed is always what it really is.
        """
        if self.mesh is None or len(self.mesh.vertices) == 0:
            return
        b = self.mesh.bounds
        centre = (b[0] + b[1]) / 2.0
        dist = float(max(self.mesh.extents.max(), 10.0)) * 2.4
        self._grid_origin = (round(float(centre[0]) / 5.0) * 5.0, round(float(centre[1]) / 5.0) * 5.0)
        for w in (self.gl_stage0, self.gl_stage1, self.gl_stage2, self.gl_kine):
            w.setCameraPosition(pos=Vector(float(centre[0]), float(centre[1]), float(centre[2])), distance=dist)

    # ------------------------------------------------------------------ Stage 0 function import
    # ------------------------------------------------------------------ function / machine files
    def reload_functions(self):
        """Reads config/functions/*.json (and keeps functions loaded from elsewhere)."""
        keep = {n: sp for n, sp in getattr(self, "functions", {}).items()
                if sp.source and os.path.dirname(os.path.abspath(sp.source)) != os.path.join(CONFIG_DIR, "functions")}
        self.functions = dict(load_function_dir(os.path.join(CONFIG_DIR, "functions")))
        self.functions.update(keep)
        current = self.function_combo.currentText()
        self.function_combo.blockSignals(True)
        self.function_combo.clear()
        self.function_combo.addItems(list(self.functions))
        self.function_combo.blockSignals(False)
        if current in self.functions:
            self.function_combo.setCurrentText(current)
        elif "Planar" in self.functions:
            self.function_combo.setCurrentText("Planar")          # the default start
        self.on_function_changed(self.function_combo.currentText())
        errors = getattr(load_function_dir, "errors", {})
        if errors:
            self.lbl_function.setText("Skipped: " + "; ".join(f"{k}: {v}" for k, v in errors.items()))

    def transition_advice(self):
        """Advice for the transition height of the current function over the placed part (None: not applicable)."""
        spec, mesh = self.current_function(), self.mesh
        if spec is None or mesh is None or len(mesh.vertices) == 0:
            return None
        b = mesh.bounds
        try:
            rng = field_range(spec, self.function_form.values(), (b[0, 0], b[1, 0], b[0, 1], b[1, 1]))
        except (FunctionFileError, ValueError):
            return None
        if rng is None or (abs(rng[0]) < 1e-9 and abs(rng[1]) < 1e-9):
            return None                                    # tilting or flat layers: the transition does not stretch anything
        return advise(rng[0], rng[1], self.layer_height_spin.value(), self.nozzle_spin.value(), self.trans_height_spin.value())

    def update_transition_advice(self, *_):
        if not hasattr(self, "lbl_transition"):
            return
        adv = self.transition_advice()
        self.btn_suggested.setVisible(False)
        if adv is None:
            self.lbl_transition.setText("")
            return
        lo, hi = adv.window
        lines = [f"Function over the part: {adv.f_min:.2f} .. {adv.f_max:.2f} mm. "
                 f"Printable layer spacing with this nozzle: {lo:.2f} .. {hi:.2f} mm."]
        if adv.thickness_min is not None:
            lines.append(f"Layer spacing at this transition: {max(adv.thickness_min, 0.0):.2f} .. {adv.thickness_max:.2f} mm.")
        if adv.min_transition > 0:
            lines.append(f"Suggested minimum transition: {adv.min_transition:.1f} mm.")
        lines.extend(adv.messages)
        self.lbl_transition.setText("\n".join(lines))
        colour = gui_theme.OK if adv.ok else gui_theme.ERR
        self.lbl_transition.setStyleSheet(f"color: {colour};")
        self.btn_suggested.setVisible(adv.possible and adv.min_transition > self.trans_height_spin.value() + 1e-9)

    def use_suggested_transition(self):
        adv = self.transition_advice()
        if adv is not None and adv.possible and adv.min_transition > 0:
            self.trans_height_spin.setValue(float(np.ceil(adv.min_transition * 10.0) / 10.0))

    def current_function(self):
        return self.functions.get(self.function_combo.currentText())

    def on_function_changed(self, name):
        spec = self.functions.get(name)
        if spec is None:
            self.function_form.set_specs({})
            self.lbl_function.setText("No function loaded.")
            return
        self.function_form.set_specs(spec.parameters)
        self.lbl_function.setText(spec.description)
        self.update_viewport()

    def _on_function_values_changed(self):
        if self.chk_field_preview.isChecked():
            self.update_viewport()

    def add_function(self, spec: FunctionSpec):
        self.functions[spec.name] = spec
        self.function_combo.blockSignals(True)
        if self.function_combo.findText(spec.name) < 0:
            self.function_combo.addItem(spec.name)
        self.function_combo.blockSignals(False)
        self.function_combo.setCurrentText(spec.name)
        self.on_function_changed(spec.name)

    def load_function_path(self, fname: str) -> bool:
        try:
            spec = load_function_file(fname)
        except FunctionFileError as e:
            self.lbl_function.setText(f"Error: {e}")
            return False
        self.add_function(spec)
        return True

    def load_function_from_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open function file", "", "Function files (*.json);;All files (*)")
        if fname and self.load_function_path(fname):
            self.chk_field_preview.setChecked(True)

    def save_function_to_file(self):
        spec = self.current_function()
        if spec is None:
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save function file", f"{spec.name}.json", "Function files (*.json)")
        if fname:
            try:
                save_function_file(spec, fname, self.function_form.values())
                self.lbl_function.setText(f"Saved {os.path.basename(fname)} (the current values are its defaults).")
            except (FunctionFileError, OSError) as e:
                self.lbl_function.setText(f"Error: {e}")

    def open_function_editor(self, new: bool = False):
        ed = FunctionEditor(self)
        ed.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        spec = None if new else self.current_function()
        if spec is not None:
            # edit the values currently shown as the new defaults
            try:
                d = spec.to_dict(self.function_form.values())
            except FunctionFileError:
                d = spec.to_dict()
            ed.base_dir = spec.base_dir
            ed.load_dict(d, spec.source)
        else:
            ed.open_spec(None)
        ed.applied.connect(self.add_function)
        self.function_editor = ed
        ed.show()
        return ed

    def open_machine_editor(self, new: bool = False):
        ed = MachineEditor(self)
        ed.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        spec = None if new else self.current_machine()
        if spec is not None:
            try:
                ed.load_dict(spec.to_dict(self.machine_form.values()), spec.source)
            except MachineFileError:
                ed.open_spec(spec)
        else:
            ed.open_spec(None)
        ed.applied.connect(self.add_machine)
        self.machine_editor = ed
        ed.show()
        return ed

    def add_machine(self, spec: MachineSpec):
        self.machines[spec.name] = spec
        self.machine_combo.blockSignals(True)
        if self.machine_combo.findText(spec.name) < 0:
            self.machine_combo.addItem(spec.name)
        self.machine_combo.blockSignals(False)
        self.machine_combo.setCurrentText(spec.name)
        self.on_machine_changed(spec.name)

    def reload_machines(self):
        keep = {n: sp for n, sp in getattr(self, "machines", {}).items()
                if sp.source and os.path.dirname(os.path.abspath(sp.source)) != os.path.join(CONFIG_DIR, "machines")}
        self.machines = dict(load_machine_dir(os.path.join(CONFIG_DIR, "machines")))
        self.machines.update(keep)
        current = self.machine_combo.currentText()
        self.machine_combo.blockSignals(True)
        self.machine_combo.clear()
        self.machine_combo.addItems(list(self.machines))
        self.machine_combo.blockSignals(False)
        if current in self.machines:
            self.machine_combo.setCurrentText(current)
        self.on_machine_changed(self.machine_combo.currentText())
        errors = getattr(load_machine_dir, "errors", {})
        if errors:
            self.lbl_machine.setText("Skipped: " + "; ".join(f"{k}: {v}" for k, v in errors.items()))

    def current_machine(self):
        return self.machines.get(self.machine_combo.currentText())

    def on_machine_changed(self, name):
        spec = self.machines.get(name)
        if spec is None:
            self.machine_form.set_specs({})
            self.lbl_machine.setText("No machine loaded.")
            return
        self.machine_form.set_specs(spec.parameters)
        self.lbl_machine.setText(f"Axes in the G-code: {' '.join(spec.axes)}. {spec.description}")

    def load_machine_path(self, fname: str) -> bool:
        try:
            spec = load_machine_file(fname)
        except MachineFileError as e:
            self.lbl_machine.setText(f"Error: {e}")
            return False
        self.add_machine(spec)
        return True

    def load_machine_from_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open machine file", "", "Machine files (*.json);;All files (*)")
        if fname:
            self.load_machine_path(fname)

    def save_machine_to_file(self):
        spec = self.current_machine()
        if spec is None:
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save machine file", f"{spec.name}.json", "Machine files (*.json)")
        if fname:
            try:
                save_machine_file(spec, fname, self.machine_form.values())
                self.lbl_machine.setText(f"Saved {os.path.basename(fname)} (the current values are its defaults).")
            except (MachineFileError, OSError) as e:
                self.lbl_machine.setText(f"Error: {e}")

    def _active_field(self):
        """The field object of the current function (for the preview), or None (planar / tilt)."""
        spec = self.current_function()
        if spec is None or spec.kind == "tilt":
            return None
        try:
            params = spec.to_slicer_params(self.function_form.values())
            if spec.kind == "expression":
                return ExpressionField(spec.expression, params["expression_params"])
            return GridField.from_params(params)
        except (FunctionFileError, ValueError) as e:
            self.lbl_function.setText(f"Error: {e}")
            return None

    def _field_preview_item(self):
        field = self._active_field()
        if field is None or self.mesh is None:
            return None
        b = self.mesh.bounds
        xs, ys, z, (zlo, zhi) = preview_surface(field, (b[0, 0], b[1, 0], b[0, 1], b[1, 1]), 40)
        base = float(b[0, 2]) + self.trans_height_spin.value()
        t = (z - zlo) / (zhi - zlo) if zhi > zlo else np.zeros_like(z)
        colors = np.zeros(z.shape + (4,))
        colors[..., 0] = t
        colors[..., 2] = 1.0 - t
        colors[..., 1] = 0.3
        colors[..., 3] = 0.6
        # surface z = z0 + T - f(x, y); pyqtgraph wants z indexed [x, y]
        item = gl.GLSurfacePlotItem(x=xs, y=ys, z=(base - z).T, colors=colors.transpose(1, 0, 2),
                                    shader=None, smooth=False, computeNormals=False)
        self.lbl_status.setText(f"Function preview: f = {zlo:.3g} .. {zhi:.3g}")
        return item

    # ------------------------------------------------------------------ view state
    def _setup_view_controls(self):
        """Adapts layer sliders / playback to the layers that exist."""
        n = self._layer_total()
        for s in (self.layer_lo, self.layer_hi):
            s.blockSignals(True)
            s.setRange(1, max(1, n))
        self.layer_lo.setValue(1)
        self.layer_hi.setValue(max(1, n))
        for s in (self.layer_lo, self.layer_hi):
            s.blockSignals(False)
        self._recompute_ranges(reset_playback=True)

    def _layer_total(self) -> int:
        if self.stage2_traj is not None:
            return layer_count(self.stage2_traj.waypoints)
        if self.stage1_model is not None and self.stage1_model.layers:
            return max(l.layer_index for l in self.stage1_model.layers) + 1
        return 0

    def _layers(self):
        """(first, last) layer index shown, 0-based inclusive."""
        n = self._layer_total()
        return effective_layers(self.layer_lo.value() - 1, self.layer_hi.value() - 1,
                                self.chk_single.isChecked(), n)

    def _stage_items(self):
        out = {}
        if self.stage2_traj is not None:
            out["stage2"] = self.stage2_traj.waypoints
        if self.stage3_traj is not None:
            out["stage3"] = self.stage3_traj.states
        return out

    def _recompute_ranges(self, reset_playback: bool):
        lo, hi = self._layers()
        n = self._layer_total()
        self.lbl_layers.setText(f"Layers {lo + 1}..{hi + 1} of {n}" if n else "Layers: -")
        self._ranges = {k: move_range(items, lo, hi) for k, items in self._stage_items().items()}
        total = 0
        if "stage2" in self._ranges:
            a, b = self._ranges["stage2"]
            total = b - a
        if reset_playback:
            self._stop_timer()
            self.playback.reset(total)
            self.play_slider.blockSignals(True)
            self.play_slider.setRange(0, total)
            self.play_slider.setValue(total)
            self.play_slider.blockSignals(False)

    def on_layer_range_changed(self, *_):
        # keep first <= last by pushing the other handle
        if self.layer_lo.value() > self.layer_hi.value():
            sender = self.sender()
            target = self.layer_hi if sender is self.layer_lo else self.layer_lo
            target.blockSignals(True)
            target.setValue(sender.value() if sender in (self.layer_lo, self.layer_hi) else target.value())
            target.blockSignals(False)
        self._recompute_ranges(reset_playback=True)
        self.update_viewport()

    def on_play_slider(self, value):
        self.playback.shown = value
        self._refresh_paths()

    def on_speed_changed(self, _):
        self.playback.speed = int(self.speed_combo.currentText().rstrip("x"))

    def toggle_play(self):
        if self.playback.total <= 0:
            return
        if self.playback.toggle():
            self.btn_play.setText("⏸ Pause")
            self.timer.start()
        else:
            self._stop_timer()
        self._sync_play_slider()
        self._refresh_paths()

    def _stop_timer(self):
        self.timer.stop()
        self.playback.playing = False
        self.btn_play.setText("▶ Play")

    def _sync_play_slider(self):
        self.play_slider.blockSignals(True)
        self.play_slider.setValue(self.playback.shown)
        self.play_slider.blockSignals(False)

    def _on_tick(self):
        running = self.playback.advance()
        self._sync_play_slider()
        self._refresh_paths()
        if not running:
            self._stop_timer()

    # ------------------------------------------------------------------ pipeline
    def _thickness_model(self, machine: MachineSpec) -> str:
        choice = self.thickness_combo.currentText()
        if choice == "auto":
            return "nozzle_axis" if machine.has_rotary_axes else "stage1"
        return choice

    def run_pipeline(self):
        if self.mesh is None:
            return

        self.lbl_metrics.setText("Running...")
        QApplication.processEvents()

        t0 = time.time()
        try:
            fspec = self.current_function()
            mspec = self.current_machine()
            if fspec is None:
                raise ValueError("no function selected (config/functions is empty)")
            if mspec is None:
                raise ValueError("no machine selected (config/machines is empty)")
            s1_params = {
                "layer_height": self.layer_height_spin.value(),
                "transition_height": self.trans_height_spin.value(),
            }
            s1_params.update(fspec.to_slicer_params(self.function_form.values()))
            machine_cfg = mspec.to_machine_config(self.machine_form.values())
            slicer_cls = PluginRegistry.get_plugin("universal_field_slicer")
            self.stage1_model = slicer_cls().slice(self.mesh, s1_params)

            prof = MinimalPrintProfile(
                layer_height=self.layer_height_spin.value(),
                nozzle_diameter=self.nozzle_spin.value(),
                num_perimeters=self.perims_spin.value(),
                infill_density=self.infill_spin.value(),
                bottom_solid_layers=self.bottom_spin.value(),
                top_solid_layers=self.top_spin.value(),
                travel_command=self.travel_cmd_combo.currentText(),
                retract_distance=self.retract_spin.value(),
                unretract_distance=self.unretract_spin.value(),
                retract_speed=self.retract_speed_spin.value(),
                unretract_speed=self.unretract_speed_spin.value(),
                # only the axes the machine really has appear in the G-code
                include_rotary_axes=mspec.has_rotary_axes,
                park_lift=self.park_spin.value(),
                infill_overlap=self.overlap_spin.value(),
                thickness_model=self._thickness_model(mspec),
                z_hop=self.zhop_spin.value(),
                hop_short_travels=self.hop_short_chk.isChecked(),
            )
            self.profile = prof
            self.stage2_traj = StandardToolpathGenerator().generate_toolpath(self.stage1_model, prof)

            solver_cls = KinematicsRegistry.get_solver(mspec.kinematics)
            self.stage3_traj = solver_cls().solve(self.stage2_traj, machine_cfg)

            self.gcode = Klipper5AxisPostProcessor().generate_gcode(self.stage3_traj, prof)
            self.txt_gcode.setPlainText(self.gcode)

            dt = time.time() - t0
            pts = len(self.stage2_traj.waypoints)
            self._setup_view_controls()
            note = ""
            adv = self.transition_advice()
            if adv is not None and not adv.ok:
                note = "\nWarning: " + ("; ".join(adv.messages) if adv.messages else
                                         f"transition height is below the suggested minimum {adv.min_transition:.1f} mm")
            self.lbl_metrics.setText(f"Done in {dt:.2f}s | Layers: {self._layer_total()} | Pts: {pts}{note}")
            self.update_viewport()

        except Exception as e:
            self.lbl_metrics.setText(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------ drawing
    def _batch_lines_s1(self, model, layer_min, layer_limit):
        lines = []
        for l in model.layers:
            if l.layer_index >= layer_limit:
                break
            if l.layer_index < layer_min:
                continue
            for c in l.contours:
                if c.feature_type == "boundary":
                    for i in range(len(c.points) - 1):
                        lines.extend([c.points[i], c.points[i + 1]])
        return lines

    def update_viewport(self, *_):
        """Full redraw: grids, mesh, field preview (static) and then the toolpaths (dynamic)."""
        if not hasattr(self, "gl_kine") or not hasattr(self, "lbl_status"):
            return                                   # still building the UI
        self._dynamic = []
        for w in (self.gl_stage0, self.gl_stage1, self.gl_stage2, self.gl_kine):
            w.clear()
            g = gl.GLGridItem()
            g.scale(5, 5, 1)
            g.setColor(gui_theme.GL_GRID)
            g.translate(self._grid_origin[0], self._grid_origin[1], 0.0)
            w.addItem(g)

        if self.mesh is not None:
            self.gl_stage0.addItem(gl.GLMeshItem(vertexes=self.mesh.vertices, faces=self.mesh.faces,
                                                 color=(0.36, 0.34, 0.40, 0.55), smooth=True, drawEdges=True,
                                                 edgeColor=(1.0, 0.69, 0.13, 0.40)))
            if self.chk_field_preview.isChecked():
                try:
                    item = self._field_preview_item()
                    if item is not None:
                        self.gl_stage0.addItem(item)
                except Exception as e:            # a preview problem must never break the view
                    self.lbl_function.setText(f"Preview error: {e}")
        self._refresh_paths()

    def _refresh_paths(self, *_):
        for w, item in self._dynamic:
            w.removeItem(item)
        self._dynamic = []

        lo, hi = self._layers()
        lines_added = []

        def add(widget, item):
            widget.addItem(item)
            self._dynamic.append((widget, item))

        if self.stage1_model is not None and hi >= lo:
            b = self._batch_lines_s1(self.stage1_model, lo, hi + 1)
            if b:
                add(self.gl_stage1, gl.GLLinePlotItem(pos=np.array(b, dtype=np.float32),
                                                      color=(1, 1, 1, 1), mode='lines'))

        status = []
        if hi >= lo:
            vis = [k for k, chk in self.feature_checks.items() if chk.isChecked()]
            for key, widget, label in (("stage2", self.gl_stage2, "Stage 2"), ("stage3", self.gl_kine, "Stage 3")):
                items = self._stage_items().get(key)
                if items is None:
                    continue
                start, stop = self._ranges.get(key, (0, 0))
                shown = min(self.playback.shown, stop - start)
                groups = group_lines_by_feature(items, vis, hi + 1, layer_min=lo, move_limit=start + shown)
                for st in FEATURE_STYLES:
                    pts = groups[st.key]
                    if pts:
                        add(widget, gl.GLLinePlotItem(pos=np.array(pts, dtype=np.float32),
                                                      color=st.color, mode='lines'))
                head = head_item(items, start, shown)
                if head is not None:
                    add(widget, gl.GLScatterPlotItem(pos=np.array([[head.x, head.y, head.z]], dtype=np.float32),
                                                     size=10, color=(1, 1, 1, 1)))
                    status.append(f"{label}: {status_text(head, self._layer_total())}")
        self.lbl_status.setText("\n".join(status) if status else "-")

    def export_gcode(self):
        if not self.gcode:
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Export G-Code", "", "G-Code Files (*.gcode)")
        if fname:
            with open(fname, "w") as f:
                f.write(self.gcode)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
