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
               offset_z: float = 0.0,
               center_ref: str = "bbox") -> trimesh.Trimesh:
    """
    Returns a placed COPY of `mesh` (the input is never modified).
      1. rotate about the centre of the bounding box (X, then Y, then Z)
      2. centre the part in XY (if `center_xy`), then add `offset_xy`. `center_ref` says what is centred: "bbox" = the bounding
         box, "contact" = the area that touches the table (the centre of the lowest flat faces, see `contact_center`)
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
        ref = contact_center(out) if center_ref == "contact" else out.bounds.mean(axis=0)[:2]
        shift[:2] = -ref
    shift[0] += offset_xy[0]
    shift[1] += offset_xy[1]
    if drop_to_bed:
        shift[2] = -out.bounds[0, 2]
    else:
        shift[2] = offset_z
    out.apply_translation(shift)
    return out


CONTACT_TOLERANCE = 0.1      # mm: faces whose three vertices are this close to the lowest point touch the table


def contact_center(mesh: trimesh.Trimesh, tol: float = CONTACT_TOLERANCE) -> np.ndarray:
    """
    XY centre (area-weighted centroid) of the part's contact area with the table: the triangles lying flat on the lowest point.
    A part without a flat bottom (a sphere, a point) has no contact area: the centroid of its lowest vertices is used.
    """
    V = mesh.vertices
    zmin = V[:, 2].min()
    tri = mesh.triangles
    flat = (tri[:, :, 2] <= zmin + tol).all(axis=1)
    if flat.any():
        area = mesh.area_faces[flat]
        if area.sum() > 1e-12:
            return (tri[flat].mean(axis=1)[:, :2] * area[:, None]).sum(axis=0) / area.sum()
    return V[V[:, 2] <= zmin + tol][:, :2].mean(axis=0)
