"""STL sanity check on import (common/mesh_check.py) and its GUI hook."""
import os
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.common.mesh_check import check_mesh


def test_a_clean_part_has_no_issues():
    assert check_mesh(trimesh.creation.box(extents=(20, 20, 10))) == []


def test_open_mesh_two_bodies_inverted_and_units_are_reported():
    box = trimesh.creation.box(extents=(20, 20, 10))
    holed = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-2], process=False)
    assert any("not watertight" in m for m in check_mesh(holed))

    b2 = trimesh.creation.box(extents=(10, 10, 10)); b2.apply_translation((50, 0, 0))
    assert any("2 separate bodies" in m for m in check_mesh(trimesh.util.concatenate([box, b2])))

    inv = box.copy(); inv.invert()
    assert any("inwards" in m for m in check_mesh(inv))

    assert any("metres or inches" in m for m in check_mesh(trimesh.creation.box(extents=(0.02, 0.02, 0.01))))
    assert any("micrometres" in m for m in check_mesh(trimesh.creation.box(extents=(2000, 20, 10))))
    assert check_mesh(trimesh.Trimesh()) == ["The file contains no faces."]


def test_gui_reports_stl_problems_on_load(tmp_path, monkeypatch):
    pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
    pytest.importorskip("pyqtgraph.opengl", exc_type=ImportError)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from src import gui_app
    global _APP
    _APP = QApplication.instance() or QApplication([])
    box = trimesh.creation.box(extents=(20, 20, 10))
    b2 = trimesh.creation.box(extents=(10, 10, 10)); b2.apply_translation((50, 0, 0))
    path = str(tmp_path / "two.stl")
    trimesh.util.concatenate([box, b2]).export(path)
    w = gui_app.MainWindow()
    w.geom_combo.setCurrentText("Upload STL...")
    w.source_mesh = w._read_stl(path)
    w.apply_placement()
    assert any("separate bodies" in m for m in w._mesh_issues)
    assert "Warning" in w.lbl_mesh_info.text() and "separate bodies" in w.lbl_stl.toolTip()
