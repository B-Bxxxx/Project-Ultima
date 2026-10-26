"""
Stage 2: toolpath generation.

UniversalSlicedModel (raw Stage 1 layer outlines)  ->  CLDataTrajectory (walls, solid shells,
sparse infill, travel moves).

Pipeline per model:
  1. Outline pass   - turn the closed contours of every layer into shapely polygons
                      (nesting-depth parity decides what is a hole and what is an island),
                      generate the wall rings and the infill region
                      (outline eroded by the wall offsets).
  2. Shell pass     - compare every layer's infill region with its neighbours
                      (by layer_index, never by z) to find the top / bottom solid skin;
                      everything else is sparse infill.
  3. Fill pass      - scanline fill of every region, per connected polygon, joined into
                      zigzag chains only where the straight link stays inside the region.
  4. Trajectory     - project every 2D point back to 3D with the Stage 1 strategy and emit
                      waypoints (travel moves between non-adjacent contours).

The input model is never modified.
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import shapely.geometry as sg
from shapely import affinity
from shapely.errors import GEOSException
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.prepared import prep

from src.common.schemas import (
    CLDataTrajectory, CLDataWaypoint, MinimalPrintProfile, SpatialContour,
    UniversalLayer, UniversalSlicedModel,
)
from src.stage2_toolpath.base import BaseToolpathGenerator
from src.stage2_toolpath.simplify import simplify_path

logger = logging.getLogger(__name__)

# Print order of the features of one layer. Walls are printed per outline from the innermost
# wall to the outer wall, then the solid skin, then the sparse infill.
FEATURE_ORDER = ("skirt", "brim", "inner_wall", "outer_wall", "bottom_solid", "top_solid", "infill")

# Scanline angle (deg) of solid skin layers; alternates sign with the layer index.
SOLID_INFILL_ANGLE_DEG = 45.0

# Geometry tolerances, relative to the nozzle diameter.
SLIVER_TOLERANCE_FACTOR = 0.02    # slivers thinner than 2x this are numerical noise
MIN_REGION_AREA_FACTOR = 0.25     # regions smaller than factor * nozzle^2 are not printable
# A zigzag link may leave the region by at most this fraction of the line spacing.
LINK_TOLERANCE_FACTOR = 0.1

# Wall direction: every outer ring runs counter-clockwise and every hole clockwise (shapely orient sign=+1),
# so the material is always on the left of the nozzle - in every layer and for every wall level.
WALL_ORIENTATION = 1.0

_EMPTY = sg.Polygon()


def start_at_seam(coords: Sequence[Tuple[float, float]], angle_deg: Optional[float]) -> List[Tuple[float, float]]:
    """
    Rotates a closed ring so that it starts (and ends) at its seam: the point farthest in the direction
    `angle_deg` (0 = +X, 90 = +Y, counter-clockwise). Every ring of every layer and wall level then starts
    on the same side of the part, so the seam is one clean line. When a straight edge faces that direction
    (several vertices tie) the seam is the middle of that edge - a vertex is inserted there. angle_deg=None
    keeps the ring as it is.
    """
    pts = [tuple(c) for c in coords]
    if angle_deg is None or len(pts) < 4:
        return pts
    ring = np.asarray(pts[:-1] if pts[0] == pts[-1] else pts, float)
    d = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))])
    perp = np.array([-d[1], d[0]])
    s = ring @ d
    cand = np.flatnonzero(s >= s.max() - 1e-6)
    if len(cand) == 1:
        k = int(cand[0])
    else:
        t = ring[cand] @ perp
        q = d * s.max() + perp * 0.5 * (t.min() + t.max())               # middle of the edge facing the seam
        nxt = np.roll(ring, -1, axis=0)
        ab = nxt - ring
        u = np.clip(np.einsum("ij,ij->i", q - ring, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0.0, 1.0)
        dist = np.linalg.norm(ring + u[:, None] * ab - q, axis=1)
        e = int(np.argmin(dist))                                          # the segment e -> e+1 holds q
        vert = np.flatnonzero(np.linalg.norm(ring - q, axis=1) < 1e-9)
        if len(vert):
            k = int(vert[0])
        else:
            ring = np.insert(ring, e + 1, q, axis=0)
            k = e + 1
    out = np.roll(ring, -k, axis=0)
    return [tuple(map(float, p)) for p in out] + [tuple(map(float, out[0]))]


# ----------------------------------------------------------------------------------------------
# Shapely helpers
# ----------------------------------------------------------------------------------------------
def _polygons_of(geom: Optional[BaseGeometry]) -> List[sg.Polygon]:
    """Flatten any geometry into its Polygon parts (lines and points are dropped)."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [orient(geom, WALL_ORIENTATION)]
    if hasattr(geom, "geoms"):
        parts: List[sg.Polygon] = []
        for g in geom.geoms:
            parts.extend(_polygons_of(g))
        return parts
    return []


def _lines_of(geom: Optional[BaseGeometry]) -> List[sg.LineString]:
    """Flatten any geometry into its LineString parts (points and polygons are dropped)."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    if hasattr(geom, "geoms"):
        parts: List[sg.LineString] = []
        for g in geom.geoms:
            parts.extend(_lines_of(g))
        return parts
    return []


def _union(parts: Sequence[BaseGeometry]) -> BaseGeometry:
    parts = [p for p in parts if p is not None and not p.is_empty]
    return unary_union(parts) if parts else _EMPTY


def _clean(geom: BaseGeometry, tol: float, min_area: float) -> BaseGeometry:
    """Remove numerical slivers (thinner than 2*tol) and unprintably small pieces."""
    if geom is None or geom.is_empty:
        return _EMPTY
    try:
        opened = geom.buffer(-tol).buffer(tol)
    except GEOSException as exc:
        logger.warning("Region cleanup failed (%s); keeping the region unmodified.", exc)
        opened = geom
    return _union([p for p in _polygons_of(opened) if p.area >= min_area])


def _exposed(region: BaseGeometry, neighbours: Sequence[BaseGeometry],
             tol: float, min_area: float) -> BaseGeometry:
    """
    The part of `region` that is NOT covered by ALL the neighbour regions.
    A missing neighbour (empty region: beyond the first / last layer, or a gap in the layer
    numbering) means the whole region is exposed.
    """
    if region.is_empty:
        return _EMPTY
    covered = None
    for nb in neighbours:
        if nb is None or nb.is_empty:
            return region
        covered = nb if covered is None else covered.intersection(nb)
        if covered.is_empty:
            return region
    if covered is None:
        return _EMPTY
    return _clean(region.difference(covered), tol, min_area)


def rings_to_polygons(contours: Sequence[SpatialContour], layer_index: Optional[int] = None) -> List[sg.Polygon]:
    """
    Builds the printable outlines of one layer from its closed contours.

    Nesting depth decides the role of a ring: even depth = outer boundary (a part or an island),
    odd depth = hole of its smallest enclosing ring. An island inside a hole is therefore an
    outline of its own. Open or self-intersecting contours are skipped with a warning.
    """
    tag = f"Layer {layer_index}" if layer_index is not None else "Layer"
    rings: List[sg.Polygon] = []
    for c in contours:
        pts = c.points
        if len(pts) < 4 or math.dist(pts[0], pts[-1]) > 1e-9:
            logger.warning("%s: skipping an open contour (%d points).", tag, len(pts))
            continue
        try:
            poly = sg.Polygon([(p[0], p[1]) for p in pts])
        except (ValueError, TypeError) as exc:
            logger.warning("%s: skipping a contour that is not a ring (%s).", tag, exc)
            continue
        if not poly.is_valid:
            logger.warning("%s: skipping a self-intersecting contour.", tag)
            continue
        if poly.is_empty or poly.area < 1e-12:
            logger.warning("%s: skipping a degenerate (zero-area) contour.", tag)
            continue
        if any(poly.equals(r) for r in rings):
            continue  # duplicate ring
        rings.append(poly)

    if not rings:
        return []

    order = sorted(range(len(rings)), key=lambda i: rings[i].area, reverse=True)
    parent: Dict[int, Optional[int]] = {}
    depth: Dict[int, int] = {}
    for pos, i in enumerate(order):
        container = None
        for j in order[:pos]:                      # only larger rings can contain ring i
            if rings[j].contains(rings[i]):
                container = j                      # the last match is the smallest container
        parent[i] = container
        depth[i] = 0 if container is None else depth[container] + 1

    polygons: List[sg.Polygon] = []
    for i in order:
        if depth[i] % 2 == 0:
            holes = [rings[k].exterior.coords for k in order if parent[k] == i and depth[k] % 2 == 1]
            polygons.append(orient(sg.Polygon(rings[i].exterior.coords, holes), WALL_ORIENTATION))
    return polygons


# ----------------------------------------------------------------------------------------------
# Fill
# ----------------------------------------------------------------------------------------------
def fill_region(region: BaseGeometry, spacing: float, angle_deg: float,
                link_tol: Optional[float] = None) -> List[List[Tuple[float, float]]]:
    """
    Parallel-line fill of `region` as zigzag chains.

    Every connected polygon of the region is filled on its own. Consecutive scanlines are joined
    into one chain only if the straight link between them lies inside
    region.buffer(link_tol) (default 0.1 * spacing); otherwise a new chain starts, i.e. a real
    travel move. Returns a list of chains (lists of (x, y) points).
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if link_tol is None:
        link_tol = LINK_TOLERANCE_FACTOR * spacing
    chains: List[List[Tuple[float, float]]] = []
    for poly in _polygons_of(region):
        chains.extend(_fill_polygon(poly, spacing, angle_deg, link_tol))
    return chains


def _fill_polygon(poly: sg.Polygon, spacing: float, angle_deg: float,
                  link_tol: float) -> List[List[Tuple[float, float]]]:
    # Work in a frame where the scanlines are vertical (x = const).
    rot = affinity.rotate(poly, angle_deg, origin=(0.0, 0.0))
    minx, miny, maxx, maxy = rot.bounds
    zone = prep(rot.buffer(link_tol))

    lines: List[List[Tuple[float, float, float]]] = []     # per scanline: (x, y_low, y_high)
    for x in np.arange(minx + spacing / 2.0, maxx, spacing):
        probe = sg.LineString([(x, miny - 1.0), (x, maxy + 1.0)])
        try:
            inter = rot.intersection(probe)
        except GEOSException as exc:
            logger.warning("Scanline at x=%.4f skipped (%s).", x, exc)
            lines.append([])
            continue
        segs = []
        for ls in _lines_of(inter):
            ys = [c[1] for c in ls.coords]
            lo, hi = min(ys), max(ys)
            if hi - lo > 1e-6:
                segs.append((float(x), lo, hi))
        segs.sort(key=lambda s: s[1])
        lines.append(segs)

    used = [[False] * len(segs) for segs in lines]
    rot_chains: List[List[Tuple[float, float]]] = []
    for li, segs in enumerate(lines):
        for si, (x, lo, hi) in enumerate(segs):
            if used[li][si]:
                continue
            used[li][si] = True
            chain = [(x, lo), (x, hi)]
            lj = li
            while lj + 1 < len(lines):
                cur = chain[-1]
                best = None
                for sj, (cx, clo, chi) in enumerate(lines[lj + 1]):
                    if used[lj + 1][sj]:
                        continue
                    d_lo = math.hypot(cur[0] - cx, cur[1] - clo)
                    d_hi = math.hypot(cur[0] - cx, cur[1] - chi)
                    if d_lo <= d_hi:
                        start, end, dist = (cx, clo), (cx, chi), d_lo
                    else:
                        start, end, dist = (cx, chi), (cx, clo), d_hi
                    if best is not None and dist >= best[0]:
                        continue
                    if not zone.covers(sg.LineString([cur, start])):
                        continue                      # the link would leave the region
                    best = (dist, sj, start, end)
                if best is None:
                    break
                _, sj, start, end = best
                used[lj + 1][sj] = True
                chain.extend([start, end])
                lj += 1
            rot_chains.append(chain)

    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    return [[(px * cos_a + py * sin_a, -px * sin_a + py * cos_a) for px, py in chain]
            for chain in rot_chains]


# ----------------------------------------------------------------------------------------------
# Per-layer plan
# ----------------------------------------------------------------------------------------------
@dataclass
class LayerRegions:
    """Shapely description of one layer (XY plane)."""
    layer_index: int
    polygons: List[sg.Polygon] = field(default_factory=list)   # printable outlines (holes included)
    # wall rings per outline: [(level, ring coords, chain)], level 0 = outer wall; a chain is the series of
    # nested rings of one boundary (the outer outline, or one hole), from the innermost wall to the outer wall
    wall_groups: List[List[Tuple[int, List[Tuple[float, float]], int]]] = field(default_factory=list)
    region: BaseGeometry = _EMPTY          # infill region = outlines eroded by all walls
    bottom_solid: BaseGeometry = _EMPTY
    top_solid: BaseGeometry = _EMPTY
    sparse: BaseGeometry = _EMPTY


def _outside_rings(polygons, offsets, seam_angle_deg=None):
    """Closed rings running `offsets` (mm, in the given order) outside the outlines; holes are not surrounded."""
    if not polygons:
        return []
    solid = _union([sg.Polygon(p.exterior) for p in polygons])
    rings = []
    for d in offsets:
        try:
            grown = solid.buffer(d, join_style=1)
        except GEOSException:
            continue
        for poly in _polygons_of(grown):
            ring = orient(sg.Polygon(poly.exterior), sign=WALL_ORIENTATION).exterior.coords
            rings.append(start_at_seam(ring, seam_angle_deg))
    return rings


def brim_loops(width: float, nozzle: float) -> int:
    """Number of nozzle-wide brim loops that fit in `width`."""
    return int(math.floor(width / nozzle + 1e-9)) if nozzle > 0 and width > 0 else 0


def brim_rings(polygons, width: float, nozzle: float, seam_angle_deg=None):
    """
    Brim loops around the OUTSIDE of the given outlines: as many nozzle-wide loops as fit in `width`,
    the outermost first. Loop k runs k*nozzle outside the outline (the outer wall is centred on it).
    """
    n = brim_loops(width, nozzle)
    return _outside_rings(polygons, [k * nozzle for k in range(n, 0, -1)], seam_angle_deg)


def skirt_rings(polygons, loops: int, distance: float, nozzle: float, brim_width: float = 0.0, seam_angle_deg=None):
    """
    `loops` free-standing loops around the outlines (and the brim), the innermost `distance` away from the
    outermost brim loop (or from the outline without a brim); the outermost loop first.
    """
    if loops <= 0:
        return []
    start = brim_loops(brim_width, nozzle) * nozzle + distance + 0.5 * nozzle
    return _outside_rings(polygons, [start + j * nozzle for j in range(loops - 1, -1, -1)], seam_angle_deg)


class StandardToolpathGenerator(BaseToolpathGenerator):
    """
    Stage 2 Toolpath Generator.
    Converts UniversalSlicedModel to CLDataTrajectory by generating walls, solid skin and sparse
    infill, inserting travel moves and calculating extrusion volumes.
    """

    # -- helpers ---------------------------------------------------------------------------
    def _subdivide_segment(self, p1, p2, max_len):
        dist = math.dist(p1, p2)
        if dist <= max_len:
            return [p2]

        num_segments = int(math.ceil(dist / max_len))
        pts = []
        for i in range(1, num_segments + 1):
            t = i / num_segments
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            pts.append((x, y))
        return pts

    @staticmethod
    def _build_strategy(model: UniversalSlicedModel):
        """Rebuilds the Stage 1 strategy from the RESOLVED parameters stored in the metadata."""
        if "strategy" not in model.metadata:
            return None
        import trimesh
        from src.stage1_slicer.math_strategies import build_strategy
        strat_name = model.metadata["strategy"]
        # The strategies only need their math here, so an empty mesh is enough.
        strategy = build_strategy(strat_name, trimesh.Trimesh(), model.metadata)

        if strat_name == "external_field":
            # Rebuild the IDW field from ALL layer boundaries
            pts_2d = []
            vertex_deformations = []
            for lyr in model.layers:
                z_prime = lyr.z_prime if lyr.z_prime is not None else lyr.z_height
                w = strategy.get_blend_weight(z_prime)
                if w <= 1e-6:
                    continue
                for contour in lyr.contours:
                    for pt in contour.points:
                        pts_2d.append([pt[0], pt[1]])
                        vertex_deformations.append((z_prime - pt[2]) / w)
            if pts_2d:
                strategy.pts_2d = np.array(pts_2d)
                strategy.vertex_deformations = np.array(vertex_deformations)
        return strategy

    # -- outline + shell passes --------------------------------------------------------------
    def compute_layer_regions(self, model: UniversalSlicedModel,
                              profile: MinimalPrintProfile) -> Dict[int, LayerRegions]:
        """
        Outlines, walls and the infill split (bottom solid / top solid / sparse) of every layer,
        keyed by layer_index. Pure shapely, no 3D projection.
        """
        if profile.infill_pattern != "lines":
            logger.warning("infill_pattern '%s' is not implemented; using 'lines'.", profile.infill_pattern)
        offset = profile.nozzle_diameter
        tol = SLIVER_TOLERANCE_FACTOR * offset
        min_area = MIN_REGION_AREA_FACTOR * offset ** 2

        plans: Dict[int, LayerRegions] = {}
        for layer in model.layers:
            plan = LayerRegions(layer_index=layer.layer_index)
            plan.polygons = rings_to_polygons(layer.contours, layer.layer_index)
            if not plan.polygons:
                logger.warning("Layer %d: no closed, valid contour; the layer is skipped.", layer.layer_index)
                plans[layer.layer_index] = plan
                continue

            region_parts = []
            for outline in plan.polygons:
                group: List[Tuple[int, List[Tuple[float, float]], int]] = []
                prev: List[Tuple[int, sg.LinearRing]] = []          # (chain, ring) of the previous wall level
                n_chains = 0
                current: BaseGeometry = outline
                for level in range(profile.num_perimeters):
                    if current.is_empty:
                        break
                    this_level: List[Tuple[int, sg.LinearRing]] = []
                    ring_list = []
                    for cp in _polygons_of(current):
                        ring_list.append(cp.exterior)
                        ring_list.extend(cp.interiors)
                    for lr in ring_list:
                        # the ring continues the nearest ring of the previous level (the same boundary, one wall in)
                        if prev:
                            chain = min(prev, key=lambda pc: pc[1].distance(lr))[0]
                        else:
                            chain = n_chains
                            n_chains += 1
                        this_level.append((chain, lr))
                        group.append((level, start_at_seam(lr.coords, profile.seam_angle_deg), chain))
                    prev = this_level
                    try:
                        current = current.buffer(-offset)
                    except GEOSException as exc:
                        logger.warning("Layer %d: wall offset failed (%s); no further walls.", layer.layer_index, exc)
                        current = _EMPTY
                        break
                plan.wall_groups.append(group)
                region_parts.append(current)
            plan.region = _union([p for p in _polygons_of(_union(region_parts)) if p.area >= min_area])
            plans[layer.layer_index] = plan

        top_n = max(0, int(profile.top_solid_layers))
        bottom_n = max(0, int(profile.bottom_solid_layers))
        regions = {i: p.region for i, p in plans.items()}
        for idx, plan in plans.items():
            region = plan.region
            if region.is_empty:
                continue
            below = [regions.get(idx - k, _EMPTY) for k in range(1, bottom_n + 1)]
            above = [regions.get(idx + k, _EMPTY) for k in range(1, top_n + 1)]
            bottom = _exposed(region, below, tol, min_area) if bottom_n > 0 else _EMPTY
            top = _exposed(region, above, tol, min_area) if top_n > 0 else _EMPTY
            if not top.is_empty and not bottom.is_empty:
                top = _clean(top.difference(bottom), tol, min_area)
            solid = _union([bottom, top])
            plan.bottom_solid = bottom
            plan.top_solid = top
            plan.sparse = region if solid.is_empty else _clean(region.difference(solid), tol, min_area)
        return plans

    # -- feature generation (2D -> 3D contours) ---------------------------------------------
    def generate_features(self, model: UniversalSlicedModel,
                          profile: MinimalPrintProfile) -> List[UniversalLayer]:
        """
        Returns NEW layers whose contours are the printable features (outer_wall, inner_wall,
        bottom_solid, top_solid, infill) in print order. The input model is not modified.
        """
        strategy = self._build_strategy(model)
        plans = self.compute_layer_regions(model, profile)
        out_layers: List[UniversalLayer] = []
        ordered_layers = sorted(model.layers, key=lambda l: l.layer_index)
        # the first layer that has any printable outline carries the brim
        first_layer = next((l for l in ordered_layers if plans[l.layer_index].polygons), None)

        for layer in ordered_layers:
            plan = plans[layer.layer_index]
            layer_idx = layer.layer_index

            contours: List[SpatialContour] = []

            def add_contour(coords, f_type, _layer=layer, _idx=layer_idx):
                if len(coords) < 2:
                    return
                subdivided = [coords[0]]
                for i in range(1, len(coords)):
                    subdivided.extend(self._subdivide_segment(subdivided[-1], coords[i], profile.max_segment_length))

                if strategy:
                    arr = strategy.project_points(subdivided, _idx)
                    pts = [tuple(map(float, p)) for p in arr]
                    normals = [tuple(map(float, n)) for n in strategy.normals_at(arr, _idx)]
                    thicknesses = [float(t) for t in strategy.thicknesses_at(arr, _idx)]
                else:
                    pts = [(x, y, _layer.z_height) for x, y in subdivided]
                    normals = [(0.0, 0.0, 1.0)] * len(pts)
                    thicknesses = [profile.layer_height] * len(pts)
                if f_type in ("skirt", "brim"):
                    thicknesses = [float(profile.layer_height)] * len(pts)     # on the bed: the nominal bead, not the function's local gap
                tol = profile.infill_path_tolerance if f_type == "infill" else profile.path_tolerance
                if tol > 0.0 and len(pts) > 2:
                    keep, h_eff = simplify_path(pts, normals, thicknesses, tol, profile.min_segment_length,
                                                profile.path_angle_tolerance_deg)
                    pts = [pts[i] for i in keep]
                    normals = [normals[i] for i in keep]
                    thicknesses = [thicknesses[0]] + h_eff
                contours.append(SpatialContour(points=pts, normals=normals,
                                               thicknesses=thicknesses, feature_type=f_type))

            # 0. skirt and brim: only around the very first layer
            if profile.skirt_loops > 0 and layer is first_layer:
                for ring in skirt_rings(plan.polygons, profile.skirt_loops, profile.skirt_distance,
                                        profile.nozzle_diameter, profile.brim_width, profile.seam_angle_deg):
                    add_contour(ring, "skirt")
            if profile.brim_width > 0 and layer is first_layer:
                for ring in brim_rings(plan.polygons, profile.brim_width, profile.nozzle_diameter, profile.seam_angle_deg):
                    add_contour(ring, "brim")

            # 1. walls: per outline from the innermost wall to the outer wall
            # (one boundary - the outline or a hole - is printed in one go, wall after wall; the next boundary is
            # the nearest one: no jumping between the outline and its holes level by level)
            for group in plan.wall_groups:
                chains: Dict[int, List[Tuple[int, List[Tuple[float, float]]]]] = {}
                for lv, ring, chain in group:
                    chains.setdefault(chain, []).append((lv, ring))
                remaining = list(chains)
                while remaining:
                    pos = contours[-1].points[-1][:2] if contours else None

                    def start_dist(c):
                        if pos is None:
                            return 0.0
                        first = max(chains[c], key=lambda t: t[0])[1][0]       # innermost ring of the chain
                        return math.dist(pos, first)
                    nxt = min(remaining, key=start_dist)
                    remaining.remove(nxt)
                    for lv, ring in sorted(chains[nxt], key=lambda t: -t[0]):
                        add_contour(ring, "outer_wall" if lv == 0 else "inner_wall")

            # 2. solid skin (100 % density, +-45 deg alternating per layer)
            solid_angle = SOLID_INFILL_ANGLE_DEG if layer_idx % 2 == 0 else -SOLID_INFILL_ANGLE_DEG
            solid_spacing = profile.nozzle_diameter
            grow = profile.infill_overlap * profile.nozzle_diameter      # small overlap into the wall / between skin and core
            for f_type, geom in (("bottom_solid", plan.bottom_solid), ("top_solid", plan.top_solid)):
                if grow > 0 and not geom.is_empty:
                    geom = geom.buffer(grow)
                for chain in fill_region(geom, solid_spacing, solid_angle):
                    add_contour(chain, f_type)

            # 3. sparse infill
            density = profile.infill_density
            if density > 0 and not plan.sparse.is_empty:
                spacing = profile.nozzle_diameter / min(density, 1.0)
                angle = profile.infill_angle_deg if layer_idx % 2 == 0 else -profile.infill_angle_deg
                sparse = plan.sparse.buffer(grow) if grow > 0 else plan.sparse
                for chain in fill_region(sparse, spacing, angle):
                    add_contour(chain, "infill")

            out_layers.append(UniversalLayer(layer_index=layer_idx, z_height=layer.z_height,
                                             z_prime=layer.z_prime, contours=contours))
        return out_layers

    # -- trajectory ------------------------------------------------------------------------
    def generate_toolpath(self, model: UniversalSlicedModel, profile: MinimalPrintProfile) -> CLDataTrajectory:
        waypoints: List[CLDataWaypoint] = []

        # We will keep track of the last point to calculate distance (dL)
        last_pt = None

        for layer in self.generate_features(model, profile):
            layer_idx = layer.layer_index
            for contour in layer.contours:
                if not contour.points:
                    continue

                # First point of the contour: is it a travel move?
                first_pt = contour.points[0]
                first_norm = contour.normals[0]

                # If we are jumping from a previous contour/layer and not continuous spiral, insert a travel move
                if last_pt is not None and not profile.continuous_spiral:
                    dist_to_start = math.dist(last_pt, first_pt)
                    if dist_to_start > 1e-6:
                        # Insert a travel move to the first point of this contour
                        waypoints.append(CLDataWaypoint(
                            x=first_pt[0], y=first_pt[1], z=first_pt[2],
                            i=first_norm[0], j=first_norm[1], k=first_norm[2],
                            extrusion_volume=0.0,
                            feedrate=profile.travel_speed,
                            is_travel_move=True,
                            feature_type=contour.feature_type,
                            layer_index=layer_idx
                        ))

                # For continuous spiral or normal extrusion, add the first point of the contour
                # If it's the very first point of the print, it's a travel move to get there
                if not waypoints:
                    waypoints.append(CLDataWaypoint(
                        x=first_pt[0], y=first_pt[1], z=first_pt[2],
                        i=first_norm[0], j=first_norm[1], k=first_norm[2],
                        extrusion_volume=0.0,
                        feedrate=profile.travel_speed,
                        is_travel_move=True,
                        feature_type=contour.feature_type,
                        layer_index=layer_idx
                    ))
                elif profile.continuous_spiral and last_pt is not None:
                    # In continuous spiral mode, extrude directly to the first point of the new contour
                    dl = math.dist(last_pt, first_pt)
                    if dl > 1e-6:
                        local_h = profile.layer_height
                        if contour.thicknesses and len(contour.thicknesses) > 0:
                            local_h = contour.thicknesses[0]
                        vol = dl * local_h * profile.nozzle_diameter
                        waypoints.append(CLDataWaypoint(
                            x=first_pt[0], y=first_pt[1], z=first_pt[2],
                            i=first_norm[0], j=first_norm[1], k=first_norm[2],
                            extrusion_volume=vol,
                            feedrate=profile.print_speed,
                            is_travel_move=False,
                            feature_type=contour.feature_type,
                            layer_index=layer_idx
                        ))

                last_pt = first_pt

                # Now iterate through the rest of the contour points and extrude
                for idx, (pt, norm) in enumerate(zip(contour.points[1:], contour.normals[1:])):
                    # offset by 1 since we are iterating from 1:
                    real_idx = idx + 1

                    dl = math.dist(last_pt, pt)

                    if dl > 1e-6:
                        # Use local thickness if available, else nominal profile layer_height
                        local_h = profile.layer_height
                        if contour.thicknesses and len(contour.thicknesses) > real_idx:
                            local_h = contour.thicknesses[real_idx]

                        # V_ext = dL * local_layer_height * nozzle_diameter
                        vol = dl * local_h * profile.nozzle_diameter

                        waypoints.append(CLDataWaypoint(
                            x=pt[0], y=pt[1], z=pt[2],
                            i=norm[0], j=norm[1], k=norm[2],
                            extrusion_volume=vol,
                            feedrate=profile.print_speed,
                            is_travel_move=False,
                            feature_type=contour.feature_type,
                            layer_index=layer_idx
                        ))
                        last_pt = pt

        return CLDataTrajectory(waypoints=waypoints)
