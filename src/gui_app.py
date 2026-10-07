import sys
import time
import numpy as np
import trimesh
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QPushButton, QLabel, QComboBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QTabWidget, QPlainTextEdit, QFileDialog, QSlider,
    QLineEdit, QStackedWidget
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
        self.geom_combo.addItems(["Cylinder", "Hollow Cylinder", "Cube", "Hollow Cube", "Upload STL..."])
        self.geom_combo.currentIndexChanged.connect(self.on_geom_changed)
        geom_layout.addRow("Geometry:", self.geom_combo)

        # Dynamic geometry inputs
        self.geom_stack = QStackedWidget()

        # 1. Cylinders
        self.cyl_widget = QWidget()
        cyl_lyt = QFormLayout(self.cyl_widget)
        self.cyl_r = QDoubleSpinBox(); self.cyl_r.setValue(15.0)
        self.cyl_h = QDoubleSpinBox(); self.cyl_h.setValue(20.0)
        self.cyl_wall = QDoubleSpinBox(); self.cyl_wall.setValue(5.0)
        cyl_lyt.addRow("Radius:", self.cyl_r)
        cyl_lyt.addRow("Height:", self.cyl_h)
        cyl_lyt.addRow("Wall Thickness:", self.cyl_wall)
        self.geom_stack.addWidget(self.cyl_widget)

        # 2. Cubes
        self.cube_widget = QWidget()
        cube_lyt = QFormLayout(self.cube_widget)
        self.cube_w = QDoubleSpinBox(); self.cube_w.setValue(30.0)
        self.cube_d = QDoubleSpinBox(); self.cube_d.setValue(30.0)
        self.cube_h = QDoubleSpinBox(); self.cube_h.setValue(20.0)
        self.cube_wall = QDoubleSpinBox(); self.cube_wall.setValue(5.0)
        cube_lyt.addRow("Width:", self.cube_w)
        cube_lyt.addRow("Depth:", self.cube_d)
        cube_lyt.addRow("Height:", self.cube_h)
        cube_lyt.addRow("Wall Thickness:", self.cube_wall)
        self.geom_stack.addWidget(self.cube_widget)

        # 3. STL
        self.stl_widget = QWidget()
        stl_lyt = QVBoxLayout(self.stl_widget)
        self.lbl_stl = QLabel("No STL loaded.")
        stl_lyt.addWidget(self.lbl_stl)
        self.geom_stack.addWidget(self.stl_widget)

        geom_layout.addRow("", self.geom_stack)
        self.btn_geom = QPushButton("Generate/Load Mesh")
        self.btn_geom.clicked.connect(self.load_geometry)
        geom_layout.addRow("", self.btn_geom)

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

        self.expr_label = QLabel("Expression:")
        self.expr_input = QLineEdit("0.1 * x")
        self.expr_label.hide()
        self.expr_input.hide()
        s1_layout.addRow(self.expr_label, self.expr_input)

        self.strategy_combo.currentTextChanged.connect(self.on_strategy_changed)

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

        # Stage 0 GL
        self.gl_stage0 = gl.GLViewWidget()
        self.gl_stage0.opts['distance'] = 80
        gx0 = gl.GLGridItem(); gx0.scale(5, 5, 1)
        self.gl_stage0.addItem(gx0)
        self.tabs.addTab(self.gl_stage0, "0. Stage 0 (Mesh)")

        # Stage 1 GL
        self.gl_stage1 = gl.GLViewWidget()
        self.gl_stage1.opts['distance'] = 80
        gx1 = gl.GLGridItem(); gx1.scale(5, 5, 1)
        self.gl_stage1.addItem(gx1)
        self.tabs.addTab(self.gl_stage1, "1. Stage 1 (Raw Slices)")

        # Stage 2 GL
        self.gl_stage2 = gl.GLViewWidget()
        self.gl_stage2.opts['distance'] = 80
        gx2 = gl.GLGridItem(); gx2.scale(5, 5, 1)
        self.gl_stage2.addItem(gx2)
        self.tabs.addTab(self.gl_stage2, "2. Stage 2 (Toolpaths)")

        # Stage 3 GL
        self.gl_kine = gl.GLViewWidget()
        self.gl_kine.opts['distance'] = 80
        gy = gl.GLGridItem(); gy.scale(5, 5, 1)
        self.gl_kine.addItem(gy)
        self.tabs.addTab(self.gl_kine, "3. Kinematics")

        # GCode
        self.txt_gcode = QPlainTextEdit()
        self.txt_gcode.setReadOnly(True)
        self.txt_gcode.setStyleSheet("font-family: monospace;")
        self.tabs.addTab(self.txt_gcode, "4. G-Code Preview")

        layout.addWidget(self.tabs)

    def on_strategy_changed(self, text):
        if text == "custom_expr":
            self.expr_label.show()
            self.expr_input.show()
        else:
            self.expr_label.hide()
            self.expr_input.hide()

    def on_geom_changed(self):
        txt = self.geom_combo.currentText()
        if "Cylinder" in txt:
            self.geom_stack.setCurrentWidget(self.cyl_widget)
        elif "Cube" in txt:
            self.geom_stack.setCurrentWidget(self.cube_widget)
        elif "Upload" in txt:
            self.geom_stack.setCurrentWidget(self.stl_widget)

    def load_geometry(self):
        gtype = self.geom_combo.currentText()
        if gtype == "Cylinder":
            self.mesh = trimesh.creation.cylinder(radius=self.cyl_r.value(), height=self.cyl_h.value())
            self.mesh.apply_translation([0, 0, self.cyl_h.value()/2])
        elif gtype == "Hollow Cylinder":
            r_out = self.cyl_r.value()
            r_in = max(0.1, r_out - self.cyl_wall.value())
            self.mesh = trimesh.creation.annulus(r_min=r_in, r_max=r_out, height=self.cyl_h.value())
            self.mesh.apply_translation([0, 0, self.cyl_h.value()/2])
        elif gtype == "Cube":
            self.mesh = trimesh.creation.box(extents=(self.cube_w.value(), self.cube_d.value(), self.cube_h.value()))
            self.mesh.apply_translation([0, 0, self.cube_h.value()/2])
        elif gtype == "Hollow Cube":
            outer = trimesh.creation.box(extents=(self.cube_w.value(), self.cube_d.value(), self.cube_h.value()))
            inner_w = max(0.1, self.cube_w.value() - self.cube_wall.value()*2)
            inner_d = max(0.1, self.cube_d.value() - self.cube_wall.value()*2)
            inner_h = max(0.1, self.cube_h.value() - self.cube_wall.value()*2)
            inner = trimesh.creation.box(extents=(inner_w, inner_d, inner_h))
            self.mesh = trimesh.boolean.difference([outer, inner], engine='blender') if trimesh.interfaces.blender.exists else outer
            # Fallback for simple testing
            if not trimesh.interfaces.blender.exists:
                print("Blender missing, fallback to cube")
                self.mesh = outer
            self.mesh.apply_translation([0, 0, self.cube_h.value()/2])
        elif gtype == "Upload STL...":
            fname, _ = QFileDialog.getOpenFileName(self, "Open STL", "", "STL Files (*.stl)")
            if fname:
                self.mesh = trimesh.load(fname, file_type='stl')
                self.lbl_stl.setText(fname.split('/')[-1])

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
                "expression": self.expr_input.text()
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
            # Deepcopy model before passing to Stage 2 because generate_toolpath mutates contours
            import copy
            model_for_s2 = copy.deepcopy(self.stage1_model)
            gen = StandardToolpathGenerator()
            self.stage2_traj = gen.generate_toolpath(model_for_s2, prof)

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

            # Use number of layers for slider instead of points for better UX
            num_layers = len(self.stage1_model.layers)
            self.slider.setMaximum(num_layers)
            self.slider.setValue(num_layers)

            self.lbl_metrics.setText(f"Done in {dt:.2f}s | Layers: {num_layers} | Pts: {pts}")
            self.update_viewport()

        except Exception as e:
            self.lbl_metrics.setText(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def _batch_lines_s2(self, waypoints, visible_types, layer_limit, profile):
        outer, inner, infill, travel = [], [], [], []

        last_wp = None
        for wp in waypoints:
            idx = int(wp.z / profile.layer_height)
            if idx > layer_limit:
                continue

            if last_wp is not None:
                # To prevent drawing lines across travels for the same feature type
                # we don't connect if the current point is the start of a travel
                p1 = [last_wp.x, last_wp.y, last_wp.z]
                p2 = [wp.x, wp.y, wp.z]

                if wp.is_travel_move:
                    if "travel" in visible_types:
                        travel.extend([p1, p2])
                else:
                    if wp.feature_type == "outer_wall" and "outer_wall" in visible_types:
                        outer.extend([p1, p2])
                    elif wp.feature_type == "inner_wall" and "inner_wall" in visible_types:
                        inner.extend([p1, p2])
                    elif wp.feature_type == "infill" and "infill" in visible_types:
                        infill.extend([p1, p2])

            last_wp = wp

        return outer, inner, infill, travel

    def _batch_lines_s1(self, model, layer_limit):
        lines = []
        for l in model.layers:
            if l.layer_index > layer_limit:
                break
            for c in l.contours:
                if c.feature_type == "boundary":
                    # Connect points in pairs for 'lines' mode
                    for i in range(len(c.points)-1):
                        lines.extend([c.points[i], c.points[i+1]])
        return lines

    def update_viewport(self):
        self.gl_stage0.clear()
        self.gl_stage1.clear()
        self.gl_stage2.clear()
        self.gl_kine.clear()

        gx0 = gl.GLGridItem(); gx0.scale(5, 5, 1); self.gl_stage0.addItem(gx0)
        gx1 = gl.GLGridItem(); gx1.scale(5, 5, 1); self.gl_stage1.addItem(gx1)
        gx2 = gl.GLGridItem(); gx2.scale(5, 5, 1); self.gl_stage2.addItem(gx2)
        gy = gl.GLGridItem(); gy.scale(5, 5, 1); self.gl_kine.addItem(gy)

        if self.mesh:
            v = self.mesh.vertices
            f = self.mesh.faces
            mesh_item_solid = gl.GLMeshItem(vertexes=v, faces=f, color=(0.4, 0.4, 0.4, 0.6), smooth=True, drawEdges=True, edgeColor=(1.0, 1.0, 1.0, 0.5))
            self.gl_stage0.addItem(mesh_item_solid)

        limit = self.slider.value()

        if self.stage1_model:
            b_lines = self._batch_lines_s1(self.stage1_model, limit)
            if b_lines:
                self.gl_stage1.addItem(gl.GLLinePlotItem(pos=np.array(b_lines, dtype=np.float32), color=(1, 1, 1, 1), mode='lines'))

        if self.stage2_traj:
            vis = []
            if self.chk_outer.isChecked(): vis.append("outer_wall")
            if self.chk_inner.isChecked(): vis.append("inner_wall")
            if self.chk_infill.isChecked(): vis.append("infill")
            if self.chk_travel.isChecked(): vis.append("travel")

            prof = MinimalPrintProfile(layer_height=self.layer_height_spin.value(), nozzle_diameter=0.4)
            outer, inner, infill, travel = self._batch_lines_s2(self.stage2_traj.waypoints, vis, limit, prof)

            if outer:
                self.gl_stage2.addItem(gl.GLLinePlotItem(pos=np.array(outer, dtype=np.float32), color=(0, 0.8, 0, 1), mode='lines'))
            if inner:
                self.gl_stage2.addItem(gl.GLLinePlotItem(pos=np.array(inner, dtype=np.float32), color=(0.8, 0.8, 0, 1), mode='lines'))
            if infill:
                self.gl_stage2.addItem(gl.GLLinePlotItem(pos=np.array(infill, dtype=np.float32), color=(0, 0.5, 0.8, 1), mode='lines'))
            if travel:
                self.gl_stage2.addItem(gl.GLLinePlotItem(pos=np.array(travel, dtype=np.float32), color=(0.8, 0, 0, 0.5), mode='lines'))

            # Stage 3
            if self.stage3_traj:
                k_outer, k_inner, k_infill, k_travel = self._batch_lines_s2(self.stage3_traj.states, vis, limit, prof)
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
