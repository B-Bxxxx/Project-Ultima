import json

import numpy as np
import pytest
import trimesh

from src.stage1_slicer.centerline import Centerline


def _tube(path, radius, nring=48):
    """Closed tube mesh swept along a polyline (rotation-minimising frames) with flat end caps."""
    P = np.asarray(path, float)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]
    u = np.cross(T[0], [0, 0, 1.0])
    if np.linalg.norm(u) < 0.5:
        u = np.cross(T[0], [1.0, 0, 0])
    u /= np.linalg.norm(u)
    ring = []
    for p, t in zip(P, T):
        u = u - (u @ t) * t
        u /= np.linalg.norm(u)
        v = np.cross(t, u)
        a = np.linspace(0, 2 * np.pi, nring, endpoint=False)
        ring.append(p + radius * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * v))
    V = np.concatenate(ring)
    faces = []
    n = len(P)
    for i in range(n - 1):
        for j in range(nring):
            a, b = i * nring + j, i * nring + (j + 1) % nring
            faces += [[a, b, b + nring], [a, b + nring, a + nring]]
    c0, c1 = len(V), len(V) + 1
    V = np.vstack([V, P[0], P[-1]])
    for j in range(nring):
        faces.append([c0, (j + 1) % nring, j])
        o = (n - 1) * nring
        faces.append([c1, o + j, o + (j + 1) % nring])
    m = trimesh.Trimesh(V, faces, process=True)
    trimesh.repair.fix_normals(m)
    return m


def _elbow(R=20.0, L=30.0):
    """Straight along +Y for L mm, then a quarter circle of radius R turning towards -X, then 15 mm straight."""
    straight = [[0, y, 0] for y in np.linspace(0, L, 31)]
    arc = [[-R + R * np.cos(a), L + R * np.sin(a), 0] for a in np.linspace(0, np.pi / 2, 60)[1:]]
    end = [[-R - x, L + R, 0] for x in np.linspace(0, 15, 16)[1:]]
    return np.array(straight + arc + end)


def test_extracts_the_axis_of_a_bent_tube():
    path = _elbow()
    mesh = _tube(path, 4.0)
    c = Centerline.from_tube_mesh(mesh)
    expected = 30 + 20 * np.pi / 2 + 15
    assert abs(c.length - expected) < 0.6
    dense = np.stack([np.interp(np.linspace(0, len(path) - 1, 4000), np.arange(len(path)), path[:, k]) for k in range(3)], 1)
    d = np.array([np.min(np.linalg.norm(dense - p, axis=1)) for p in c.points[::5]])
    assert d.max() < 0.25                                          # the points lie on the true axis
    assert np.allclose(c.tangents[0], [0, 1, 0], atol=0.02)
    assert np.allclose(c.tangents[-1], [-1, 0, 0], atol=0.03)
    assert abs(c.min_bend_radius - 20.0) < 2.0


def test_curvature_of_a_circle_and_a_line():
    a = np.linspace(0, np.pi, 80)
    arc = Centerline.from_points(np.stack([10 * np.cos(a), 10 * np.sin(a), np.zeros_like(a)], 1))
    assert abs(arc.length - 10 * np.pi) < 0.05
    assert abs(arc.curvature_at(arc.length / 2) - 0.1) < 0.003
    line = Centerline.from_points([[0, 0, 0], [0, 0, 10]])
    assert line.min_bend_radius == float("inf") and np.allclose(line.tangent_at(5), [0, 0, 1])


def test_file_round_trip(tmp_path):
    c = Centerline.from_points(_elbow())
    p = tmp_path / "c.json"
    c.save(str(p))
    assert json.loads(p.read_text())["format"] == "cam-centerline"
    c2 = Centerline.load(str(p))
    assert abs(c2.length - c.length) < 0.2
    (tmp_path / "bad.json").write_text(json.dumps({"format": "other"}))
    with pytest.raises(ValueError):
        Centerline.load(str(tmp_path / "bad.json"))


def test_rejects_a_mesh_without_two_caps():
    with pytest.raises(ValueError):
        Centerline.from_tube_mesh(trimesh.creation.icosphere(radius=5))


def test_needs_two_points():
    with pytest.raises(ValueError):
        Centerline.from_points([[0, 0, 0]])
