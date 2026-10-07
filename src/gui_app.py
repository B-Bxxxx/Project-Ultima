import sys
import os
import trimesh
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QDoubleSpinBox, QCheckBox,
                             QFileDialog, QTabWidget, QTextEdit, QSplitter)
from PyQt6.QtCore import Qt
import pyqtgraph.opengl as gl
import numpy as np

from src.common.schemas import MachineConfig, MinimalPrintProfile
from src.stage1_slicer.registry import PluginRegistry
from src.stage2_toolpath.contour_generator import StandardToolpathGenerator
from src.stage3_kinematics.registry import KinematicsRegistry
from src.stage4_postproc.klipper_postproc import Klipper5AxisPostProcessor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modular 5-Axis CAM Native GUI")
        self.setGeometry(100, 100, 1200, 800)

        self.mesh = None
        self.stage1_model = None
        self.stage2_traj = None
        self.stage3_traj = None
        self.stage4_gcode = ""

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(300)
        sidebar_layout = QVBoxLayout(sidebar)

        # Geometry
        sidebar_layout.addWidget(QLabel("1. Input Geometry"))
        self.geom_combo = QComboBox()
        self.geom_combo.addItems(["Cylinder", "Hollow Cylinder", "Upload STL"])
        sidebar_layout.addWidget(self.geom_combo)

        self.btn_geom = QPushButton("Generate/Upload Mesh")
        self.btn_geom.clicked.connect(self.load_geometry)
        sidebar_layout.addWidget(self.btn_geom)

        # Strategy
        sidebar_layout.addWidget(QLabel("\n2. Math Strategy"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["planar", "progressive_tilt", "conical", "custom_expr"])
        sidebar_layout.addWidget(self.strategy_combo)

        # Print Profile
        sidebar_layout.addWidget(QLabel("\n3. Print Profile"))
        self.layer_height_spin = QDoubleSpinBox()
        self.layer_height_spin.setPrefix("Layer Height: ")
        self.layer_height_spin.setValue(0.4)
        sidebar_layout.addWidget(self.layer_height_spin)

        self.nozzle_spin = QDoubleSpinBox()
        self.nozzle_spin.setPrefix("Nozzle: ")
        self.nozzle_spin.setValue(0.4)
        sidebar_layout.addWidget(self.nozzle_spin)

        # Kinematics
        sidebar_layout.addWidget(QLabel("\n4. Kinematics"))
        self.kine_combo = QComboBox()
        self.kine_combo.addItems(["planar_3axis_xyz", "trunnion_table_xyzbc", "swivel_head_xyzbc"])
        sidebar_layout.addWidget(self.kine_combo)

        # Run Button
        self.btn_run = QPushButton("RUN PIPELINE")
        self.btn_run.setStyleSheet("background-color: #2e8b57; color: white; font-weight: bold; padding: 10px;")
        self.btn_run.clicked.connect(self.run_pipeline)
        sidebar_layout.addWidget(self.btn_run)

        sidebar_layout.addStretch()
        main_layout.addWidget(sidebar)

        # Main Tabs
        self.tabs = QTabWidget()

        # 3D Viewport
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.opts['distance'] = 50

        # Grid
        gx = gl.GLGridItem()
        gx.scale(5, 5, 1)
        self.gl_widget.addItem(gx)

        self.tabs.addTab(self.gl_widget, "3D Viewport")

        # GCode Text
        self.gcode_text = QTextEdit()
        self.gcode_text.setReadOnly(True)
        self.tabs.addTab(self.gcode_text, "G-Code")

        main_layout.addWidget(self.tabs)

    def load_geometry(self):
        geom_type = self.geom_combo.currentText()
        if geom_type == "Cylinder":
            self.mesh = trimesh.creation.cylinder(radius=15, height=20)
            self.mesh.apply_translation([0, 0, 10])
        elif geom_type == "Upload STL":
            fname, _ = QFileDialog.getOpenFileName(self, "Open STL", "", "STL Files (*.stl)")
            if fname:
                self.mesh = trimesh.load(fname, file_type='stl')
        self.update_viewport()

    def update_viewport(self):
        self.gl_widget.clear()

        gx = gl.GLGridItem()
        gx.scale(5, 5, 1)
        self.gl_widget.addItem(gx)

        if self.mesh:
            verts = self.mesh.vertices
            faces = self.mesh.faces
            mesh_item = gl.GLMeshItem(vertexes=verts, faces=faces, color=(0.5, 0.5, 0.5, 0.5), smooth=True)
            self.gl_widget.addItem(mesh_item)

        if self.stage2_traj:
            # Vectorized line batching for fast rendering
            extrude_pts = []
            travel_pts = []

            last_pt = None
            for wp in self.stage2_traj.waypoints:
                pt = [wp.x, wp.y, wp.z]
                if last_pt is not None:
                    if wp.is_travel_move:
                        travel_pts.extend([last_pt, pt])
                    else:
                        extrude_pts.extend([last_pt, pt])
                last_pt = pt

            if extrude_pts:
                lines = gl.GLLinePlotItem(pos=np.array(extrude_pts), color=(0, 1, 0, 1), mode='lines')
                self.gl_widget.addItem(lines)
            if travel_pts:
                lines = gl.GLLinePlotItem(pos=np.array(travel_pts), color=(1, 0, 0, 0.5), mode='lines')
                self.gl_widget.addItem(lines)

    def run_pipeline(self):
        if not self.mesh:
            return

        try:
            # 1. Slice
            plugin = PluginRegistry.get_plugin("universal_field_slicer")()
            params = {
                "strategy": self.strategy_combo.currentText(),
                "layer_height": self.layer_height_spin.value(),
                "nozzle_diameter": self.nozzle_spin.value(),
                "start_tilt_deg": 0.0,
                "end_tilt_deg": 25.0,
                "cone_angle_deg": 15.0,
                "expression": "0.05 * (x**2 + y**2)"
            }
            self.stage1_model = plugin.slice(self.mesh, params)

            # 2. Toolpath
            profile = MinimalPrintProfile(
                nozzle_diameter=self.nozzle_spin.value(),
                layer_height=self.layer_height_spin.value()
            )
            gen = StandardToolpathGenerator()
            self.stage2_traj = gen.generate_toolpath(self.stage1_model, profile)

            # 3. Kinematics
            kine = self.kine_combo.currentText()
            cfg = MachineConfig(
                kinematic_chain=kine,
                b_axis_min=-90, b_axis_max=90,
                c_axis_min=-36000, c_axis_max=36000
            )
            solver = KinematicsRegistry.get_solver(kine)()
            self.stage3_traj = solver.solve(self.stage2_traj, cfg)

            # 4. G-Code
            post = Klipper5AxisPostProcessor()
            self.stage4_gcode = post.generate_gcode(self.stage3_traj, profile)

            self.gcode_text.setPlainText(self.stage4_gcode)
            self.update_viewport()

        except Exception as e:
            self.gcode_text.setPlainText(f"Error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
