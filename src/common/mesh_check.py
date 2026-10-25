"""
Sanity check of an imported mesh (Qt-free): reports problems that make the slice fail or look wrong.
It never blocks anything; the messages are shown in the GUI and logged to the terminal.
"""
import logging
from typing import List

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

TINY_MM = 1.0           # the largest dimension below this: probably metres / inches instead of mm
HUGE_MM = 1000.0        # above this: probably micrometres, or a scene instead of a part


def check_mesh(mesh: trimesh.Trimesh) -> List[str]:
    """Returns a list of human readable problems (empty = the mesh looks fine)."""
    issues: List[str] = []
    if mesh is None or len(mesh.faces) == 0:
        return ["The file contains no faces."]

    edges = mesh.edges_sorted
    _, counts = np.unique(edges, axis=0, return_counts=True)
    open_edges = int((counts == 1).sum())
    bad_edges = int((counts > 2).sum())
    if open_edges:
        issues.append(f"The mesh is not watertight ({open_edges} open edges): the slice may have open or missing contours.")
    if bad_edges:
        issues.append(f"{bad_edges} edges are shared by more than two faces (non-manifold): overlapping or touching bodies.")

    try:
        bodies = int(mesh.body_count)
    except Exception:                                   # body counting needs networkx; not worth failing for
        bodies = 1
    if bodies > 1:
        issues.append(f"The file holds {bodies} separate bodies; they are sliced together (check their placement).")

    if mesh.is_watertight and mesh.volume < 0:
        issues.append("The face normals point inwards (negative volume): the part may be sliced inside out.")

    degenerate = int((mesh.area_faces < 1e-12).sum())
    if degenerate:
        issues.append(f"{degenerate} degenerate (zero area) faces.")

    biggest = float(mesh.extents.max())
    if biggest < TINY_MM:
        issues.append(f"The part is only {biggest:.3g} mm large: is the STL in metres or inches? (units are taken as mm)")
    elif biggest > HUGE_MM:
        issues.append(f"The part is {biggest:.0f} mm large: is the STL in micrometres, or does it hold more than the part?")
    return issues
