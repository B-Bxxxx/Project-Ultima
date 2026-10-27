"""ULTIMA slicer look: charcoal panels with coral and amber accents (Qt-free constants + one stylesheet)."""

NAME = "ULTIMA slicer"

BG = "#141318"          # window
PANEL = "#1d1c23"       # cards / buttons
FIELD = "#25242c"       # input fields
LINE = "#2f2e38"        # thin borders
TEXT = "#ebe9f0"
MUTED = "#8b8896"
CORAL = "#ff3b5c"       # main accent
AMBER = "#ffb020"       # second accent
OK = "#6ee7a0"
ERR = "#ff5c7a"

GL_BACKGROUND = (18, 17, 22, 255)
GL_GRID = (255, 255, 255, 38)

_TEMPLATE = """
QWidget {{ background-color: {BG}; color: {TEXT}; font-size: 12px; }}
QMainWindow, QDialog {{ background-color: {BG}; }}
QLabel {{ background: transparent; }}
QLabel#brand {{ font-size: 24px; font-weight: 800; letter-spacing: 2px; padding: 8px 6px 0 6px; }}
QLabel#brandSub {{ color: {MUTED}; letter-spacing: 2px; font-size: 10px; padding: 0 8px 8px 8px; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background-color: {BG}; }}
QToolButton {{ background: {PANEL}; color: {TEXT}; border: none; border-radius: 6px; font-weight: bold; padding: 6px; margin: 2px 0; }}
QToolButton:hover {{ background: {FIELD}; color: {AMBER}; }}
QFrame[frameShape="4"] {{ color: {BG}; }}
QPushButton {{ background-color: {FIELD}; color: {TEXT}; border: none; border-radius: 6px; padding: 6px 12px; }}
QPushButton:hover {{ background-color: {LINE}; color: {AMBER}; }}
QPushButton:pressed {{ background-color: {CORAL}; color: {BG}; }}
QPushButton:disabled {{ color: {MUTED}; background-color: {PANEL}; }}
QPushButton#runButton {{ background-color: {AMBER}; color: {BG}; padding: 11px; font-weight: bold; font-size: 14px; }}
QPushButton#runButton:hover {{ background-color: #ffc450; color: {BG}; }}
QPushButton#runButton:pressed {{ background-color: {CORAL}; }}
QLineEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
    background-color: {FIELD}; color: {TEXT}; border: 1px solid {LINE}; border-radius: 5px; padding: 3px 6px;
    selection-background-color: {CORAL}; selection-color: {TEXT}; }}
QLineEdit:focus, QPlainTextEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{ border: 1px solid {AMBER}; }}
QComboBox QAbstractItemView {{ background-color: {PANEL}; color: {TEXT}; border: 1px solid {LINE};
                               selection-background-color: {CORAL}; selection-color: {TEXT}; }}
QDoubleSpinBox, QSpinBox {{ padding-right: 18px; }}
QDoubleSpinBox::up-button, QSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 16px;
    border-left: 1px solid {LINE}; background: {PANEL}; border-top-right-radius: 5px; }}
QDoubleSpinBox::down-button, QSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 16px;
    border-left: 1px solid {LINE}; background: {PANEL}; border-bottom-right-radius: 5px; }}
QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{ background: {LINE}; }}
QDoubleSpinBox::up-button:pressed, QSpinBox::up-button:pressed, QDoubleSpinBox::down-button:pressed, QSpinBox::down-button:pressed {{ background: {CORAL}; }}
QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{ image: url({UP}); width: 9px; height: 6px; }}
QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{ image: url({DOWN}); width: 9px; height: 6px; }}
QComboBox {{ padding-right: 18px; }}
QComboBox::drop-down {{ subcontrol-origin: border; subcontrol-position: top right; width: 18px; border-left: 1px solid {LINE}; }}
QComboBox::down-arrow {{ image: url({DOWN}); width: 9px; height: 6px; }}
QCheckBox {{ spacing: 6px; background: transparent; }}
QCheckBox::indicator {{ width: 13px; height: 13px; border: 1px solid {MUTED}; border-radius: 3px; background: {FIELD}; }}
QCheckBox::indicator:checked {{ background-color: {AMBER}; border-color: {AMBER}; }}
QTabWidget::pane {{ border: 1px solid {LINE}; border-radius: 6px; }}
QTabBar::tab {{ background: {PANEL}; color: {MUTED}; padding: 7px 14px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; }}
QTabBar::tab:selected {{ color: {TEXT}; background: {FIELD}; border-bottom: 2px solid {CORAL}; }}
QTabBar::tab:hover {{ color: {AMBER}; }}
QSlider::groove:horizontal {{ height: 4px; background: {LINE}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {CORAL}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {AMBER}; width: 12px; margin: -5px 0; border-radius: 6px; }}
QScrollBar:vertical {{ background: {BG}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {MUTED}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QHeaderView::section {{ background: {PANEL}; color: {AMBER}; border: 1px solid {LINE}; padding: 3px; }}
QToolTip {{ background-color: {PANEL}; color: {TEXT}; border: 1px solid {LINE}; }}
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
        pt.setBrush(QColor(AMBER))
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
    values = dict(BG=BG, PANEL=PANEL, FIELD=FIELD, LINE=LINE, TEXT=TEXT, MUTED=MUTED, CORAL=CORAL, AMBER=AMBER,
                  UP=arrows["up"], DOWN=arrows["down"])
    return _TEMPLATE.format(**values)


def logo_path() -> str:
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.svg")


def logo_pixmap(size: int):
    """The ULTIMA logo rendered at size x size px on a transparent background (None if it cannot be loaded)."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QImage, QPainter, QPixmap
    from PyQt6.QtSvg import QSvgRenderer
    r = QSvgRenderer(logo_path())
    if not r.isValid():
        return None
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    r.render(p)
    p.end()
    return QPixmap.fromImage(img)
