import math

import numpy as np
import pytest
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QApplication

import pyqtgraph.opengl as gl

from src.gui_viewcube import ViewCube, angles_for_direction, attach_view_cube, camera_basis, SIZE


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def cube(app):
    view = gl.GLViewWidget()
    view.resize(500, 400)
    c = attach_view_cube(view)
    return c


def test_camera_basis_matches_pyqtgraph(app):
    view = gl.GLViewWidget()
    for az, el in ((0, 0), (-90, 0), (30, 40), (120, -25)):
        view.setCameraPosition(azimuth=az, elevation=el, distance=10)
        R = np.array(view.viewMatrix().copyDataTo()).reshape(4, 4)[:3, :3]
        right, up, toward = camera_basis(az, el)
        assert np.allclose(R[0], right, atol=1e-6) and np.allclose(R[1], up, atol=1e-6) and np.allclose(R[2], toward, atol=1e-6)


def test_directions_to_angles():
    assert angles_for_direction((1, 0, 0)) == (0.0, 0.0)
    assert angles_for_direction((0, -1, 0)) == (-90.0, 0.0)
    az, el = angles_for_direction((1, -1, 1))
    assert abs(az + 45) < 1e-9 and abs(el - math.degrees(math.asin(1 / math.sqrt(3)))) < 1e-9
    assert angles_for_direction((0, 0, 1), current_az=-80.0) == (-90.0, 90.0)     # from above: heading snapped to 90 deg
    assert angles_for_direction((0, 0, -1), current_az=10.0)[1] == -90.0


def test_click_cells_turn_the_camera(cube):
    cube.go((0, -1, 0), animate=False)                       # front view
    assert cube.angles() == (-90.0, 0.0)
    hit = cube.hit_test(cube._center())                      # the middle of the cube is the front face
    assert hit[0] == "cell" and hit[4] == (0.0, -1.0, 0.0)
    cube.go((0, 0, 1), animate=False)
    assert cube.angles()[1] == 90.0
    cube.go((1, 0, 0), animate=False)
    assert cube.angles() == (0.0, 0.0)
    cube.go((-1, 1, 1), animate=False)                       # a corner
    az, el = cube.angles()
    assert abs(_wrap(az - 135)) < 1e-6 and abs(el - 35.264) < 0.01


def _wrap(a):
    return (a + 180) % 360 - 180


def test_corner_and_edge_cells_have_diagonal_directions(cube):
    cube.view.setCameraPosition(azimuth=-45, elevation=35.264)        # looking at the front / right / top corner
    basis = camera_basis(*cube.angles())
    corner = cube.project((1, -1, 1), basis)
    inside = cube._center() * 0.12 + corner * 0.88                    # a little towards the middle of the cube
    hit = cube.hit_test(QPointF(inside.x(), inside.y()))
    assert hit[0] == "cell" and tuple(np.sign(hit[4])) == (1.0, -1.0, 1.0) and tuple(np.abs(hit[4])) == (1.0, 1.0, 1.0)
    edge = cube.project((1, -1, 0), basis)                             # the vertical edge between Front and Right
    inside = cube._center() * 0.1 + edge * 0.9
    hit = cube.hit_test(QPointF(inside.x(), inside.y()))
    assert hit[0] == "cell" and tuple(hit[4])[2] == 0.0 and tuple(np.abs(hit[4])[:2]) == (1.0, 1.0)


def test_arrows_rotate_in_90_degree_steps(cube):
    cube.go((0, -1, 0), animate=False)
    cube.rotate_view(90, 0, animate=False)
    assert abs(_wrap(cube.angles()[0] - 0)) < 1e-6
    cube.rotate_view(0, 90, animate=False)
    assert cube.angles()[1] == 90.0
    cube.rotate_view(0, 90, animate=False)                              # clamped at the pole
    assert cube.angles()[1] == 90.0


def test_animation_ends_at_the_target(cube, app):
    cube.go_angles(-90.0, 0.0, animate=False)
    cube.go_angles(0.0, 60.0, animate=True)
    for _ in range(40):
        app.processEvents()
        cube._anim_step() if cube._anim else None
    az, el = cube.angles()
    assert abs(_wrap(az)) < 1e-6 and abs(el - 60.0) < 1e-6


def test_follows_the_camera_and_sits_top_right(cube, app):
    cube.view.setCameraPosition(azimuth=33, elevation=12)
    cube._poll_camera()
    assert cube._last == (33.0, 12.0)
    assert cube.pos().x() == 500 - SIZE - 8 and cube.pos().y() == 8
    cube.view.resize(700, 400)
    cube._poll_camera()
    assert cube.pos().x() == 700 - SIZE - 8


def test_paints_without_error(cube):
    for az, el in ((-90, 0), (-45, 35), (30, 70), (0, 90), (180, -60)):
        cube.view.setCameraPosition(azimuth=az, elevation=el)
        img = cube.grab().toImage()
        assert not img.isNull() and img.width() == SIZE


def test_main_window_has_a_cube_in_every_3d_view(app):
    from src.gui_app import MainWindow
    win = MainWindow()
    views = [win.gl_stage0, win.gl_stage1, win.gl_stage2, win.gl_kine, win.check_tab.view]
    for v in views:
        assert isinstance(getattr(v, "view_cube", None), ViewCube) and v.view_cube.view is v
