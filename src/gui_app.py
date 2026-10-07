import sys
import time
import numpy as np
import trimesh
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QPushButton, QLabel, QComboBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QTabWidget, QPlainTextEdit, QFileDialog, QSlider
)
from PyQt6.QtCore import Qt
import pyqtgraph.opengl as gl

from src.common.schemas import MachineConfig, MinimalPrintProfile
from src.stage1_slicer.registry import PluginRegistry
from src.stage2_toolpath.contour_generator import StandardToolpathGenerator
from src.stage3_kinematics.registry import KinematicsRegistry
from src.stage4_postproc.klipper_postproc import Klipper5AxisPostProcessor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modular 5-Axis CAM Native GUI")
        self.setGeometry(100, 100, 1400, 900)

        self.mesh = None
        self.stage1_model = None
        self.stage2_traj = None
        self.stage3_traj = None
        self.gcode = ""

        self.init_ui()
        self.load_geometry()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- LEFT PANEL ---
        left_panel = QWidget()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)

        # Geometry
        geom_group = QWidget()
        geom_layout = QFormLayout(geom_group)
        self.geom_combo = QComboBox()
        self.geom_combo.addItems(["Hollow Cylinder", "Overhanging Frustum", "Curved Tube"])
        self.geom_combo.currentIndexChanged.connect(self.load_geometry)
        geom_layout.addRow("Geometry:", self.geom_combo)
        left_layout.addWidget(QLabel("<b>1. Input Geometry</b>"))
        left_layout.addWidget(geom_group)

        # Stage 1
        s1_group = QWidget()
        s1_layout = QFormLayout(s1_group)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["planar", "progressive_tilt", "conical", "custom_expr"])
        s1_layout.addRow("Strategy:", self.strategy_combo)

        self.layer_height_spin = QDoubleSpinBox()
        self.layer_height_spin.setValue(0.4)
        self.layer_height_spin.setSingleStep(0.1)
        s1_layout.addRow("Layer Height:", self.layer_height_spin)

        self.trans_height_spin = QDoubleSpinBox()
        self.trans_height_spin.setValue(2.0)
        self.trans_height_spin.setSingleStep(0.5)
        s1_layout.addRow("Transition H:", self.trans_height_spin)

        left_layout.addWidget(QLabel("<b>2. Stage 1 Slicer</b>"))
        left_layout.addWidget(s1_group)

        # Stage 2
        s2_group = QWidget()
        s2_layout = QFormLayout(s2_group)
        self.nozzle_spin = QDoubleSpinBox()
        self.nozzle_spin.setValue(0.4)
        s2_layout.addRow("Nozzle Dia:", self.nozzle_spin)

        self.perims_spin = QSpinBox()
        self.perims_spin.setValue(2)
        s2_layout.addRow("Perimeters:", self.perims_spin)

        self.infill_spin = QDoubleSpinBox()
        self.infill_spin.setValue(0.2)
        self.infill_spin.setSingleStep(0.1)
        s2_layout.addRow("Infill Density:", self.infill_spin)

        left_layout.addWidget(QLabel("<b>3. Stage 2 Profile</b>"))
        left_layout.addWidget(s2_group)

        # Stage 3
        s3_group = QWidget()
        s3_layout = QFormLayout(s3_group)
        self.topo_combo = QComboBox()
        self.topo_combo.addItems(["planar_3axis_xyz", "trunnion_table_xyzbc", "swivel_head_xyzbc"])
        s3_layout.addRow("Topology:", self.topo_combo)
        left_layout.addWidget(QLabel("<b>4. Kinematics</b>"))
        left_layout.addWidget(s3_group)

        # View Settings
        view_group = QWidget()
        view_layout = QVBoxLayout(view_group)
        self.chk_outer = QCheckBox("Outer Walls")
        self.chk_outer.setChecked(True)
        self.chk_inner = QCheckBox("Inner Walls")
        self.chk_inner.setChecked(True)
        self.chk_infill = QCheckBox("Infill")
        self.chk_infill.setChecked(True)
        self.chk_travel = QCheckBox("Travel Moves")
        self.chk_travel.setChecked(False)

        for chk in (self.chk_outer, self.chk_inner, self.chk_infill, self.chk_travel):
            chk.stateChanged.connect(self.update_viewport)
            view_layout.addWidget(chk)

        left_layout.addWidget(QLabel("<b>5. View Settings</b>"))
        left_layout.addWidget(view_group)

        # Layer Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(100)
        self.slider.valueChanged.connect(self.update_viewport)
        left_layout.addWidget(QLabel("Layer Z-Scrub:"))
        left_layout.addWidget(self.slider)

        # Actions
        self.btn_run = QPushButton("Run CAM Pipeline")
        self.btn_run.setStyleSheet("background-color: #2e8b57; color: white; padding: 10px; font-weight: bold;")
        self.btn_run.clicked.connect(self.run_pipeline)
        left_layout.addWidget(self.btn_run)

        self.btn_export = QPushButton("Export G-Code (.gcode)")
        self.btn_export.clicked.connect(self.export_gcode)
        left_layout.addWidget(self.btn_export)

        self.lbl_metrics = QLabel("Ready")
        left_layout.addWidget(self.lbl_metrics)
        left_layout.addStretch()

        layout.addWidget(left_panel)

        # --- RIGHT PANEL ---
        self.tabs = QTabWidget()

        # Toolpath GL
        self.gl_toolpath = gl.GLViewWidget()
        self.gl_toolpath.opts['distance'] = 80
        gx = gl.GLGridItem()
        gx.scale(5, 5, 1)
        self.gl_toolpath.addItem(gx)
        self.tabs.addTab(self.gl_toolpath, "3D Toolpath Viewport")

        # Kinematics GL
        self.gl_kine = gl.GLViewWidget()
        self.gl_kine.opts['distance'] = 80
        gy = gl.GLGridItem()
        gy.scale(5, 5, 1)
        self.gl_kine.addItem(gy)
        self.tabs.addTab(self.gl_kine, "Kinematics Viewport")

        # GCode
        self.txt_gcode = QPlainTextEdit()
        self.txt_gcode.setReadOnly(True)
        self.txt_gcode.setStyleSheet("font-family: monospace;")
        self.tabs.addTab(self.txt_gcode, "G-Code Preview")

        layout.addWidget(self.tabs)

    def load_geometry(self):
        gtype = self.geom_combo.currentText()
        if gtype == "Hollow Cylinder":
            self.mesh = trimesh.creation.annulus(r_min=10, r_max=15, height=20)
            self.mesh.apply_translation([0, 0, 10])
        elif gtype == "Overhanging Frustum":
            self.mesh = trimesh.creation.cylinder(radius=15, height=20)
            self.mesh.apply_translation([0, 0, 10])
        elif gtype == "Curved Tube":
            self.mesh = trimesh.creation.annulus(r_min=10, r_max=15, height=20)
            self.mesh.apply_translation([0, 0, 10])

        self.update_viewport()

    def run_pipeline(self):
        if not self.mesh:
            return

        self.lbl_metrics.setText("Running...")
        QApplication.processEvents()

        t0 = time.time()
        try:
            # Stage 1
            s1_params = {
                "strategy": self.strategy_combo.currentText(),
                "layer_height": self.layer_height_spin.value(),
                "transition_height": self.trans_height_spin.value(),
                "start_tilt_deg": 0.0,
                "end_tilt_deg": 30.0,
                "cone_angle_deg": 15.0,
                "expression": "0.1 * x"
            }
            slicer_cls = PluginRegistry.get_plugin("universal_field_slicer")
            self.stage1_model = slicer_cls().slice(self.mesh, s1_params)

            # Stage 2
            prof = MinimalPrintProfile(
                layer_height=self.layer_height_spin.value(),
                nozzle_diameter=self.nozzle_spin.value(),
                num_perimeters=self.perims_spin.value(),
                infill_density=self.infill_spin.value()
            )
            gen = StandardToolpathGenerator()
            self.stage2_traj = gen.generate_toolpath(self.stage1_model, prof)

            # Stage 3
            cfg = MachineConfig(
                kinematic_chain=self.topo_combo.currentText(),
                b_axis_min=-90, b_axis_max=90,
                c_axis_min=-36000, c_axis_max=36000
            )
            solver_cls = KinematicsRegistry.get_solver(self.topo_combo.currentText())
            self.stage3_traj = solver_cls().solve(self.stage2_traj, cfg)

            # Stage 4
            post = Klipper5AxisPostProcessor()
            self.gcode = post.generate_gcode(self.stage3_traj, prof)
            self.txt_gcode.setPlainText(self.gcode)

            dt = time.time() - t0
            pts = len(self.stage2_traj.waypoints)
            self.slider.setMaximum(pts)
            self.slider.setValue(pts)

            self.lbl_metrics.setText(f"Done in {dt:.2f}s | Pts: {pts}")
            self.update_viewport()

        except Exception as e:
            self.lbl_metrics.setText(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def _batch_lines(self, waypoints, visible_types, limit):
        outer, inner, infill, travel = [], [], [], []

        last_wp = None
        for i, wp in enumerate(waypoints):
            if i > limit:
                break

            if last_wp is not None:
                p1 = [last_wp.x, last_wp.y, last_wp.z]
                p2 = [wp.x, wp.y, wp.z]

                if wp.is_travel_move and "travel" in visible_types:
                    travel.extend([p1, p2])
                elif wp.feature_type == "outer_wall" and "outer_wall" in visible_types:
                    outer.extend([p1, p2])
                elif wp.feature_type == "inner_wall" and "inner_wall" in visible_types:
                    inner.extend([p1, p2])
                elif wp.feature_type == "infill" and "infill" in visible_types:
                    infill.extend([p1, p2])

            last_wp = wp

        return outer, inner, infill, travel

    def update_viewport(self):
        self.gl_toolpath.clear()
        self.gl_kine.clear()

        gx = gl.GLGridItem()
        gx.scale(5, 5, 1)
        self.gl_toolpath.addItem(gx)

        gy = gl.GLGridItem()
        gy.scale(5, 5, 1)
        self.gl_kine.addItem(gy)

        if self.mesh:
            v = self.mesh.vertices
            f = self.mesh.faces
            mesh_item = gl.GLMeshItem(vertexes=v, faces=f, color=(0.4, 0.4, 0.4, 0.3), smooth=True)
            self.gl_toolpath.addItem(mesh_item)

        if not self.stage2_traj:
            return

        limit = self.slider.value()
        vis = []
        if self.chk_outer.isChecked(): vis.append("outer_wall")
        if self.chk_inner.isChecked(): vis.append("inner_wall")
        if self.chk_infill.isChecked(): vis.append("infill")
        if self.chk_travel.isChecked(): vis.append("travel")

        outer, inner, infill, travel = self._batch_lines(self.stage2_traj.waypoints, vis, limit)

        if outer:
            self.gl_toolpath.addItem(gl.GLLinePlotItem(pos=np.array(outer, dtype=np.float32), color=(0, 0.8, 0, 1), mode='lines'))
        if inner:
            self.gl_toolpath.addItem(gl.GLLinePlotItem(pos=np.array(inner, dtype=np.float32), color=(0.8, 0.8, 0, 1), mode='lines'))
        if infill:
            self.gl_toolpath.addItem(gl.GLLinePlotItem(pos=np.array(infill, dtype=np.float32), color=(0, 0.5, 0.8, 1), mode='lines'))
        if travel:
            self.gl_toolpath.addItem(gl.GLLinePlotItem(pos=np.array(travel, dtype=np.float32), color=(0.8, 0, 0, 0.5), mode='lines'))

        # Kinematics drawing (Stage 3)
        if self.stage3_traj:
            k_outer, k_inner, k_infill, k_travel = self._batch_lines(self.stage3_traj.states, vis, limit)
            if k_outer:
                self.gl_kine.addItem(gl.GLLinePlotItem(pos=np.array(k_outer, dtype=np.float32), color=(0, 0.8, 0, 1), mode='lines'))
            if k_inner:
                self.gl_kine.addItem(gl.GLLinePlotItem(pos=np.array(k_inner, dtype=np.float32), color=(0.8, 0.8, 0, 1), mode='lines'))
            if k_infill:
                self.gl_kine.addItem(gl.GLLinePlotItem(pos=np.array(k_infill, dtype=np.float32), color=(0, 0.5, 0.8, 1), mode='lines'))
            if k_travel:
                self.gl_kine.addItem(gl.GLLinePlotItem(pos=np.array(k_travel, dtype=np.float32), color=(0.8, 0, 0, 0.5), mode='lines'))

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
