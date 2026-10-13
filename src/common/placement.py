"""Part placement on the bed: rotate, centre in XY, drop to z = 0 (pure numpy/trimesh, Qt-free)."""
from typing import Sequence, Tuple

import numpy as np
import trimesh


def rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """4x4 rotation: first about X, then about Y, then about Z (all in degrees, right-handed)."""
    m = np.eye(4)
    for angle, axis in ((rx_deg, (1, 0, 0)), (ry_deg, (0, 1, 0)), (rz_deg, (0, 0, 1))):
        if angle:
            m = trimesh.transformations.rotation_matrix(np.radians(angle), axis) @ m
    return m


def place_mesh(mesh: trimesh.Trimesh,
               rotation_deg: Sequence[float] = (0.0, 0.0, 0.0),
               center_xy: bool = True,
               drop_to_bed: bool = True,
               offset_xy: Tuple[float, float] = (0.0, 0.0),
               offset_z: float = 0.0) -> trimesh.Trimesh:
    """
    Returns a placed COPY of `mesh` (the input is never modified).
      1. rotate about the centre of the bounding box (X, then Y, then Z)
      2. centre the bounding box in XY (if `center_xy`), then add `offset_xy`
      3. translate in Z so that the lowest point is at z = 0 (if `drop_to_bed`); otherwise the
         part keeps its height and is moved by `offset_z` (ignored when dropping to the bed)
    """
    out = mesh.copy()
    if len(out.vertices) == 0:
        return out
    centre = out.bounds.mean(axis=0)
    rot = rotation_matrix(*rotation_deg)
    to_origin = np.eye(4); to_origin[:3, 3] = -centre
    back = np.eye(4); back[:3, 3] = centre
    out.apply_transform(back @ rot @ to_origin)

    shift = np.zeros(3)
    if center_xy:
        shift[:2] = -out.bounds.mean(axis=0)[:2]
    shift[0] += offset_xy[0]
    shift[1] += offset_xy[1]
    if drop_to_bed:
        shift[2] = -out.bounds[0, 2]
    else:
        shift[2] = offset_z
    out.apply_translation(shift)
    return out
