# PyInstaller spec: one-file, windowed ULTIMA slicer.   Build:  pyinstaller packaging/ultima.spec --noconfirm
import os, sys

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ICON = os.path.join(SPECPATH, "ultima.ico")

datas = [
    (os.path.join(ROOT, "config"), "config"),                                          # functions, machines, defaults
    (os.path.join(ROOT, "src", "assets"), os.path.join("src", "assets")),              # logo
    (os.path.join(ROOT, "src", "stage4_postproc", "templates"), os.path.join("src", "stage4_postproc", "templates")),
]

# not used by the GUI / pipeline (only by the command line plots): left out to keep the program smaller
excludes = ["plotly", "matplotlib", "tkinter", "IPython", "pytest", "PyQt6.QtWebEngineCore", "PyQt6.QtWebEngineWidgets",
            "PyQt6.QtWebEngineQuick", "PyQt6.QtQml", "PyQt6.QtQuick", "PyQt6.QtQuickWidgets", "PyQt6.QtMultimedia",
            "PyQt6.QtMultimediaWidgets", "PyQt6.QtPdf", "PyQt6.QtPdfWidgets", "PyQt6.QtDesigner", "PyQt6.QtBluetooth",
            "PyQt6.QtNfc", "PyQt6.QtSensors", "PyQt6.QtSerialPort", "PyQt6.QtSql", "PyQt6.QtTest", "PyQt6.QtHelp",
            "PyQt6.QtRemoteObjects", "PyQt6.QtPositioning", "PyQt6.QtDBus", "PyQt6.QtNetwork"]

a = Analysis([os.path.join(ROOT, "src", "gui_app.py")], pathex=[ROOT], datas=datas, excludes=excludes,
             hiddenimports=["OpenGL.platform.win32", "OpenGL.arrays.numpymodule", "PyQt6.QtSvg", "PyQt6.QtOpenGLWidgets"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="ULTIMA-slicer", console=False, upx=False,
          icon=ICON if os.path.exists(ICON) else None)
