"""Test/demo solids that need no boolean or triangulation engine (pure numpy + trimesh)."""
import numpy as np
import trimesh


def hollow_box(width: float, depth: float, height: float, wall: float) -> trimesh.Trimesh:
    """
    Rectangular tube with a rectangular THROUGH-hole along Z (open at the top and the bottom),
    centred on the XY origin, standing on z = 0. `wall` is the wall thickness on every side.
    """
    if min(width, depth, height) <= 0 or wall <= 0:
        raise ValueError("width, depth, height and wall must be positive")
    if 2 * wall >= min(width, depth):
        raise ValueError("wall too thick: 2*wall must be smaller than width and depth")
    ow, od = width / 2.0, depth / 2.0
    iw, idp = ow - wall, od - wall

    def ring(hx, hy, z):                      # counter-clockwise rectangle
        return [(-hx, -hy, z), (hx, -hy, z), (hx, hy, z), (-hx, hy, z)]

    verts = np.array(ring(ow, od, 0) + ring(ow, od, height) + ring(iw, idp, 0) + ring(iw, idp, height), float)
    O0, O1, I0, I1 = 0, 4, 8, 12
    faces = []
    for k in range(4):
        a, b = k, (k + 1) % 4
        # outer side (normal outwards)
        faces += [(O0 + a, O0 + b, O1 + b), (O0 + a, O1 + b, O1 + a)]
        # inner side (normal towards the hole axis): reversed winding
        faces += [(I0 + a, I1 + b, I0 + b), (I0 + a, I1 + a, I1 + b)]
        # top ring (+z)
        faces += [(O1 + a, O1 + b, I1 + b), (O1 + a, I1 + b, I1 + a)]
        # bottom ring (-z)
        faces += [(O0 + a, I0 + b, O0 + b), (O0 + a, I0 + a, I0 + b)]
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    mesh.fix_normals()
    return mesh
