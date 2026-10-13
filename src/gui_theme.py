"""ULTIMA slicer look: dark background with neon accents (Qt-free constants + one stylesheet)."""

NAME = "ULTIMA slicer"

BG = "#070b14"
PANEL = "#0d1424"
FIELD = "#101a2e"
TEXT = "#d6f6ff"
MUTED = "#6f87a6"
CYAN = "#00e5ff"
MAGENTA = "#ff2bd6"
GREEN = "#39ff14"
YELLOW = "#ffe600"
ORANGE = "#ff7a00"
RED = "#ff3860"

GL_BACKGROUND = (7, 11, 20, 255)
GL_GRID = (0, 229, 255, 60)

_TEMPLATE = """
QWidget {{ background-color: {BG}; color: {TEXT}; font-size: 12px; }}
QMainWindow, QDialog {{ background-color: {BG}; }}
QLabel {{ background: transparent; }}
QLabel#brand {{ font-size: 26px; font-weight: 800; letter-spacing: 4px; color: {CYAN}; padding: 6px 4px 0 4px; }}
QLabel#brandSub {{ color: {MAGENTA}; letter-spacing: 3px; font-size: 11px; padding: 0 6px 6px 6px; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background-color: {BG}; }}
QToolButton {{ background: transparent; color: {CYAN}; border: none; font-weight: bold; padding: 4px;
              border-bottom: 1px solid {PANEL}; }}
QToolButton:hover {{ color: {MAGENTA}; }}
QFrame[frameShape="4"] {{ color: {PANEL}; }}
QPushButton {{ background-color: {PANEL}; color: {CYAN}; border: 1px solid {CYAN}; border-radius: 4px; padding: 5px 10px; }}
QPushButton:hover {{ background-color: #0f2a3a; color: {TEXT}; border-color: {MAGENTA}; }}
QPushButton:pressed {{ background-color: {CYAN}; color: {BG}; }}
QPushButton:disabled {{ color: {MUTED}; border-color: #26364f; }}
QPushButton#runButton {{ background-color: {BG}; color: {GREEN}; border: 2px solid {GREEN}; padding: 10px;
                         font-weight: bold; font-size: 14px; letter-spacing: 2px; }}
QPushButton#runButton:hover {{ background-color: #0c2a12; color: {TEXT}; }}
QPushButton#runButton:pressed {{ background-color: {GREEN}; color: {BG}; }}
QLineEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
    background-color: {FIELD}; color: {TEXT}; border: 1px solid #1f3a5a; border-radius: 3px; padding: 2px 4px;
    selection-background-color: {MAGENTA}; selection-color: {BG}; }}
QLineEdit:focus, QPlainTextEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{ border: 1px solid {CYAN}; }}
QComboBox QAbstractItemView {{ background-color: {PANEL}; color: {TEXT}; border: 1px solid {CYAN};
                               selection-background-color: {MAGENTA}; selection-color: {BG}; }}
QDoubleSpinBox, QSpinBox {{ padding-right: 18px; }}
QDoubleSpinBox::up-button, QSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 16px;
    border-left: 1px solid #1f3a5a; background: {PANEL}; border-top-right-radius: 3px; }}
QDoubleSpinBox::down-button, QSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 16px;
    border-left: 1px solid #1f3a5a; background: {PANEL}; border-bottom-right-radius: 3px; }}
QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{ background: #0f2a3a; }}
QDoubleSpinBox::up-button:pressed, QSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed, QSpinBox::down-button:pressed {{ background: {CYAN}; }}
QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{ image: url({UP}); width: 9px; height: 6px; }}
QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{ image: url({DOWN}); width: 9px; height: 6px; }}
QComboBox {{ padding-right: 18px; }}
QComboBox::drop-down {{ subcontrol-origin: border; subcontrol-position: top right; width: 18px; border-left: 1px solid #1f3a5a; }}
QComboBox::down-arrow {{ image: url({DOWN}); width: 9px; height: 6px; }}
QCheckBox {{ spacing: 6px; background: transparent; }}
QCheckBox::indicator {{ width: 13px; height: 13px; border: 1px solid {CYAN}; border-radius: 2px; background: {FIELD}; }}
QCheckBox::indicator:checked {{ background-color: {CYAN}; image: none; }}
QTabWidget::pane {{ border: 1px solid #1f3a5a; }}
QTabBar::tab {{ background: {PANEL}; color: {MUTED}; padding: 6px 12px; border: 1px solid #1f3a5a; border-bottom: none; }}
QTabBar::tab:selected {{ color: {CYAN}; border-bottom: 2px solid {MAGENTA}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QSlider::groove:horizontal {{ height: 4px; background: #1f3a5a; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {CYAN}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {MAGENTA}; width: 12px; margin: -5px 0; border-radius: 6px; }}
QScrollBar:vertical {{ background: {BG}; width: 10px; }}
QScrollBar::handle:vertical {{ background: #1f3a5a; border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {CYAN}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QHeaderView::section {{ background: {PANEL}; color: {CYAN}; border: 1px solid #1f3a5a; padding: 3px; }}
QToolTip {{ background-color: {PANEL}; color: {TEXT}; border: 1px solid {CYAN}; }}
"""


_arrow_cache = {}


def _arrow_paths():
    """Draws the two spin/combo arrows once into a temp folder (QSS can only use image files)."""
    if _arrow_cache:
        return _arrow_cache
    import os
    import tempfile
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtGui import QColor, QPainter, QPixmap, QPolygonF
    folder = os.path.join(tempfile.gettempdir(), "ultima_slicer_ui")
    os.makedirs(folder, exist_ok=True)
    for name, pts in (("up", [(0.5, 5.5), (4.5, 0.5), (8.5, 5.5)]), ("down", [(0.5, 0.5), (4.5, 5.5), (8.5, 0.5)])):
        pm = QPixmap(9, 6)
        pm.fill(Qt.GlobalColor.transparent)
        pt = QPainter(pm)
        pt.setRenderHint(QPainter.RenderHint.Antialiasing)
        pt.setBrush(QColor(CYAN))
        pt.setPen(Qt.PenStyle.NoPen)
        pt.drawPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))
        pt.end()
        path = os.path.join(folder, f"arrow_{name}.png")
        pm.save(path)
        _arrow_cache[name] = path.replace("\\", "/")
    return _arrow_cache


def stylesheet() -> str:
    """The complete stylesheet (needs a running QGuiApplication for the arrow images)."""
    arrows = _arrow_paths()
    values = dict(BG=BG, PANEL=PANEL, FIELD=FIELD, TEXT=TEXT, MUTED=MUTED, CYAN=CYAN, MAGENTA=MAGENTA, GREEN=GREEN,
                  YELLOW=YELLOW, ORANGE=ORANGE, RED=RED, UP=arrows["up"], DOWN=arrows["down"])
    return _TEMPLATE.format(**values)
