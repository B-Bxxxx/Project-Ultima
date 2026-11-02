"""Renders src/assets/logo.svg into packaging/ultima.ico (needs PyQt6 + Pillow)."""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from PyQt6.QtGui import QGuiApplication
from PIL import Image

app = QGuiApplication([])
from src import gui_theme

pm = gui_theme.logo_pixmap(256)
png = os.path.join(os.path.dirname(__file__), "ultima_256.png")
pm.save(png)
Image.open(png).save(os.path.join(os.path.dirname(__file__), "ultima.ico"), sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
os.remove(png)
print("wrote packaging/ultima.ico")
