from src.gui_app import MainWindow
from PyQt6.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
win = MainWindow()

win.strategy_combo.setCurrentText("custom_expr")
win.geom_combo.setCurrentText("Cube")
win.run_pipeline()

z_vals = [wp.z for wp in win.stage2_traj.waypoints]
print(f"GUI Traj Z min: {min(z_vals)}, max: {max(z_vals)}")
