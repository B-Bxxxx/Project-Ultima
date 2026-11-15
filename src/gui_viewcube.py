"""
ViewCube: a small clickable orientation cube in the corner of a 3D view (pyqtgraph GLViewWidget).

* the cube always shows the way the camera looks (it follows mouse orbiting and camera changes made in code),
* a click on a face / edge / corner turns the camera to look at the part from that side (smoothly),
* the four arrows turn the view by 90 degrees,
* a small X / Y / Z triad (red / green / blue) sits in the lower left corner of the widget.

World convention (printer): +Z up, the front of the machine is -Y. Camera: azimuth / elevation of pyqtgraph, the camera sits at
centre + distance * (cos(el) cos(az), cos(el) sin(az), sin(el)); screen right = (-sin az, cos az, 0).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QCursor, QFont, QPainter, QPen, QPolygonF, QTransform
from PyQt6.QtWidgets import QWidget

from src import gui_theme

SIZE = 140
SCALE = 27.0            # px per cube half edge
EDGE = 0.6              # a face is split 3 x 3: centre = face, strips = edges, squares = corners (|s| > EDGE)
MARGIN = 8
ANIM_STEPS = 10
ANIM_MS = 15

# face: (name, outward normal, up vector on the face)
FACES = [
    ("Right", (1, 0, 0), (0, 0, 1)),
    ("Left", (-1, 0, 0), (0, 0, 1)),
    ("Back", (0, 1, 0), (0, 0, 1)),
    ("Front", (0, -1, 0), (0, 0, 1)),
    ("Top", (0, 0, 1), (0, 1, 0)),
    ("Bottom", (0, 0, -1), (0, -1, 0)),
]


def camera_basis(az_deg: float, el_deg: float):
    """(right, up, towards-the-camera) unit vectors of the pyqtgraph camera."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    right = np.array([-math.sin(a), math.cos(a), 0.0])
    up = np.array([-math.sin(e) * math.cos(a), -math.sin(e) * math.sin(a), math.cos(e)])
    toward = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return right, up, toward


def angles_for_direction(d, current_az: float = -90.0) -> Tuple[float, float]:
    """Camera azimuth / elevation that look at the part from direction `d` (the camera sits on that side)."""
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    el = math.degrees(math.asin(max(-1.0, min(1.0, d[2]))))
    if abs(d[0]) < 1e-9 and abs(d[1]) < 1e-9:                      # straight from above / below: keep the heading (snapped to 90 deg)
        az = round(current_az / 90.0) * 90.0
    else:
        az = math.degrees(math.atan2(d[1], d[0]))
    return az, el


def _norm_az(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class ViewCube(QWidget):
    def __init__(self, view):
        super().__init__(view)
        self.view = view
        self.setFixedSize(SIZE, SIZE)
        self.setMouseTracking(True)
        self._hover = None                       # ("cell", face_index, i, j) | ("arrow", k) | None
        self._last = None
        self._anim: List[Tuple[float, float]] = []
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(ANIM_MS)
        self._anim_timer.timeout.connect(self._anim_step)
        self._poll = QTimer(self)                # follows the camera when it is moved by the mouse or by code
        self._poll.setInterval(40)
        self._poll.timeout.connect(self._poll_camera)
        self._poll.start()
        view.installEventFilter(self)
        self._reposition()
        self.show()
        self.raise_()

    # ------------------------------------------------------------------ camera
    def angles(self) -> Tuple[float, float]:
        return float(self.view.opts["azimuth"]), float(self.view.opts["elevation"])

    def _set_angles(self, az: float, el: float) -> None:
        self.view.setCameraPosition(azimuth=az, elevation=max(-90.0, min(90.0, el)))

    def go(self, direction, animate: bool = True) -> None:
        """Look at the part from the side `direction` (e.g. (0, -1, 0) = from the front)."""
        az0, el0 = self.angles()
        az1, el1 = angles_for_direction(direction, az0)
        self.go_angles(az1, el1, animate)

    def go_angles(self, az1: float, el1: float, animate: bool = True) -> None:
        az0, el0 = self.angles()
        daz = _norm_az(az1 - az0)
        if not animate:
            self._anim_timer.stop()
            self._anim = []
            self._set_angles(az0 + daz, el1)
            return
        self._anim = []
        for k in range(1, ANIM_STEPS + 1):
            t = k / ANIM_STEPS
            t = t * t * (3 - 2 * t)                                  # ease in / out
            self._anim.append((az0 + daz * t, el0 + (el1 - el0) * t))
        self._anim_timer.start()

    def rotate_view(self, d_az: float, d_el: float, animate: bool = True) -> None:
        az, el = self.angles()
        self.go_angles(az + d_az, max(-90.0, min(90.0, el + d_el)), animate)

    def _anim_step(self) -> None:
        if not self._anim:
            self._anim_timer.stop()
            return
        az, el = self._anim.pop(0)
        self._set_angles(az, el)
        if not self._anim:
            self._anim_timer.stop()

    def _poll_camera(self) -> None:
        cur = self.angles()
        if cur != self._last:
            self._last = cur
            self.update()
        self._reposition()

    # ------------------------------------------------------------------ geometry
    def _center(self) -> QPointF:
        return QPointF(SIZE / 2.0, SIZE / 2.0 + 2.0)

    def project(self, p3, basis=None) -> QPointF:
        """3D point (cube units) -> widget pixel."""
        r, u, _ = basis or camera_basis(*self.angles())
        p = np.asarray(p3, float)
        c = self._center()
        return QPointF(c.x() + SCALE * float(p @ r), c.y() - SCALE * float(p @ u))

    def _cells(self, basis):
        """All visible cells: (face_index, i, j, polygon, direction)."""
        _, _, d = basis
        out = []
        edges = (-1.0, -EDGE, EDGE, 1.0)
        for fi, (name, n, v) in enumerate(FACES):
            n = np.array(n, float)
            v = np.array(v, float)
            if float(n @ d) <= 1e-6:
                continue
            u = np.cross(v, n)
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    s0, s1 = (edges[0], edges[1]) if i == -1 else ((edges[1], edges[2]) if i == 0 else (edges[2], edges[3]))
                    t0, t1 = (edges[0], edges[1]) if j == -1 else ((edges[1], edges[2]) if j == 0 else (edges[2], edges[3]))
                    quad = [n + s * u + t * v for s, t in ((s0, t0), (s1, t0), (s1, t1), (s0, t1))]
                    poly = QPolygonF([self.project(q, basis) for q in quad])
                    out.append((fi, i, j, poly, tuple(n + i * u + j * v)))
        return out

    def _arrows(self):
        """Four triangles around the cube: (kind, polygon, (d_az, d_el))."""
        c = self._center()
        R, w, h = 62.0, 7.0, 9.0
        out = []
        # up / down / left / right (pointing away from the cube)
        out.append((QPolygonF([QPointF(c.x(), c.y() - R - h), QPointF(c.x() - w, c.y() - R), QPointF(c.x() + w, c.y() - R)]), (0.0, 90.0)))
        out.append((QPolygonF([QPointF(c.x(), c.y() + R + h), QPointF(c.x() - w, c.y() + R), QPointF(c.x() + w, c.y() + R)]), (0.0, -90.0)))
        out.append((QPolygonF([QPointF(c.x() - R - h, c.y()), QPointF(c.x() - R, c.y() - w), QPointF(c.x() - R, c.y() + w)]), (90.0, 0.0)))
        out.append((QPolygonF([QPointF(c.x() + R + h, c.y()), QPointF(c.x() + R, c.y() - w), QPointF(c.x() + R, c.y() + w)]), (-90.0, 0.0)))
        return out

    def hit_test(self, pos: QPointF):
        """What is under the pixel: ("cell", face, i, j, direction) | ("arrow", k, (d_az, d_el)) | None."""
        basis = camera_basis(*self.angles())
        for k, (poly, delta) in enumerate(self._arrows()):
            if poly.containsPoint(pos, Qt.FillRule.OddEvenFill):
                return ("arrow", k, delta)
        for fi, i, j, poly, direction in self._cells(basis):
            if poly.containsPoint(pos, Qt.FillRule.OddEvenFill):
                return ("cell", fi, i, j, direction)
        return None

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        basis = camera_basis(*self.angles())
        r, u_cam, d = basis
        hov = self._hover
        face_fill = QColor(gui_theme.PANEL)
        face_fill.setAlpha(235)
        strip_fill = QColor(gui_theme.FIELD)
        strip_fill.setAlpha(235)
        line = QColor(gui_theme.MUTED)
        line.setAlpha(150)
        amber = QColor(gui_theme.AMBER)
        # arrows
        for k, (poly, _delta) in enumerate(self._arrows()):
            on = hov is not None and hov[0] == "arrow" and hov[1] == k
            p.setPen(Qt.PenStyle.NoPen)
            col = QColor(gui_theme.AMBER if on else gui_theme.MUTED)
            col.setAlpha(255 if on else 170)
            p.setBrush(col)
            p.drawPolygon(poly)
        # cube cells, faces facing the camera
        p.setFont(QFont("Sans Serif", 1))
        for fi, i, j, poly, _dir in self._cells(basis):
            on = hov is not None and hov[0] == "cell" and hov[1] == fi and hov[2] == i and hov[3] == j
            p.setPen(QPen(line, 1.0))
            p.setBrush(amber if on else (face_fill if (i == 0 and j == 0) else strip_fill))
            p.drawPolygon(poly)
        # face labels (drawn in the plane of the face, so they foreshorten with it)
        font = QFont("Sans Serif")
        font.setPixelSize(int(SCALE * 0.62))
        font.setBold(True)
        p.setFont(font)
        for fi, (name, n, v) in enumerate(FACES):
            n3, v3 = np.array(n, float), np.array(v, float)
            facing = float(n3 @ d)
            if facing < 0.3:
                continue
            u3 = np.cross(v3, n3)
            on = hov is not None and hov[0] == "cell" and hov[1] == fi and hov[2] == 0 and hov[3] == 0
            base = self.project(n3, basis)
            tf = QTransform(float(u3 @ r), -float(u3 @ u_cam), -float(v3 @ r), float(v3 @ u_cam), base.x(), base.y())
            p.save()
            p.setTransform(tf, True)
            p.setPen(QColor(gui_theme.BG if on else gui_theme.TEXT))
            half = SCALE * 0.9
            p.drawText(QRectF(-half, -half, 2 * half, 2 * half), int(Qt.AlignmentFlag.AlignCenter), name)
            p.restore()
        # axis triad in the lower left corner
        o = QPointF(16.0, SIZE - 16.0)
        for axis, col, label in (((1, 0, 0), "#ff5c5c", "X"), ((0, 1, 0), "#6ee7a0", "Y"), ((0, 0, 1), "#5aa2ff", "Z")):
            a = np.array(axis, float)
            end = QPointF(o.x() + 20.0 * float(a @ r), o.y() - 20.0 * float(a @ u_cam))
            p.setPen(QPen(QColor(col), 2.0))
            p.drawLine(o, end)
            f2 = QFont("Sans Serif")
            f2.setPixelSize(10)
            f2.setBold(True)
            p.setFont(f2)
            p.drawText(QRectF(end.x() - 6 + 5 * float(a @ r), end.y() - 6 - 5 * float(a @ u_cam), 12, 12),
                       int(Qt.AlignmentFlag.AlignCenter), label)
        p.end()

    # ------------------------------------------------------------------ mouse
    def mouseMoveEvent(self, ev) -> None:
        hit = self.hit_test(ev.position())
        new = None if hit is None else (("cell", hit[1], hit[2], hit[3]) if hit[0] == "cell" else ("arrow", hit[1]))
        if new != self._hover:
            self._hover = new
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor if new else Qt.CursorShape.ArrowCursor))
            self.update()

    def leaveEvent(self, _ev) -> None:
        self._hover = None
        self.update()

    def mousePressEvent(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        hit = self.hit_test(ev.position())
        if hit is None:
            ev.ignore()                              # not on the cube: the 3D view gets the click
            return
        if hit[0] == "cell":
            self.go(hit[4])
        else:
            self.rotate_view(*hit[2])
        ev.accept()

    def wheelEvent(self, ev) -> None:
        ev.ignore()                                  # zoom the 3D view

    # ------------------------------------------------------------------ placement
    def eventFilter(self, obj, ev) -> bool:
        if obj is self.view and ev.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self._reposition()
        return False

    def _reposition(self) -> None:
        x = max(0, self.view.width() - SIZE - MARGIN)
        if self.pos().x() != x or self.pos().y() != MARGIN:
            self.move(x, MARGIN)
        self.raise_()


def attach_view_cube(view) -> ViewCube:
    """Puts a ViewCube into the top right corner of a GLViewWidget (and keeps it there)."""
    cube = ViewCube(view)
    view.view_cube = cube
    return cube
