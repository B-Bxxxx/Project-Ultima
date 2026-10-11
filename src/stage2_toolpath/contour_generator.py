import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import shapely.affinity as sa
import shapely.geometry as sg
from shapely.errors import GEOSException
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep

from src.common.schemas import (
    UniversalSlicedModel, UniversalLayer, MinimalPrintProfile, CLDataTrajectory,
    CLDataWaypoint, SpatialContour,
)
from src.stage2_toolpath.base import BaseToolpathGenerator

logger = logging.getLogger(__name__)

# Shapely operations can raise ValueError (bad input) or GEOSException (topology).
SHAPELY_ERRORS = (ValueError, GEOSException)

# Print order of the features inside one layer, for every connected part.
# Inner walls first, the visible outer wall last, then solid skins, then sparse infill.
FEATURE_ORDER = ("inner_wall", "outer_wall", "bottom_solid", "top_solid", "infill")

SOLID_ANGLE_DEG = 45.0          # solid skins alternate +45 / -45 deg per layer
LINK_TOLERANCE_FACTOR = 0.1     # an infill link may leave the region by 0.1 * line spacing
MIN_SEGMENT_FACTOR = 0.25       # scanline pieces shorter than 0.25 * spacing are noise
MIN_FILL_AREA_FACTOR = 1.0      # fill regions smaller than (nozzle diameter)^2 are skipped
MIN_RING_AREA = 1e-9


@dataclass
class LayerAnalysis:
    """2D analysis of one layer (shapely geometry, in the XY plane)."""
    layer_index: int
    parts: List[sg.Polygon] = field(default_factory=list)          # printable parts (with holes)
    open_paths: List[SpatialContour] = field(default_factory=list)  # intentional open paths (spiral / vase / parametric)
    part_regions: List[BaseGeometry] = field(default_factory=list)  # parts eroded by the walls
    region: BaseGeometry = field(default_factory=sg.Polygon)        # union of part_regions
    bottom_solid: BaseGeometry = field(default_factory=sg.Polygon)
    top_solid: BaseGeometry = field(default_factory=sg.Polygon)
    sparse: BaseGeometry = field(default_factory=sg.Polygon)


def _polygons(geom: BaseGeometry) -> List[sg.Polygon]:
    """All Polygon members of a (Multi)Polygon / GeometryCollection."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if hasattr(geom, "geoms"):
        out = []
        for g in geom.geoms:
            out.extend(_polygons(g))
        return out
    return []


def _erode(poly: BaseGeometry, distance: float) -> BaseGeometry:
    if distance <= 0:
        return poly
    try:
        return poly.buffer(-distance)
    except SHAPELY_ERRORS as exc:
        logger.warning("Offsetting a polygon by -%.3f failed (%s); skipping it.", distance, exc)
        return sg.Polygon()


def _density_fraction(profile: MinimalPrintProfile) -> float:
    # Accepts a fraction (<= 1.0) or a percentage (> 1.0). Unit cleanup is tracked for Stage 4.
    value = profile.infill_density
    frac = value if value <= 1.0 else value / 100.0
    return min(max(frac, 0.0), 1.0)


class StandardToolpathGenerator(BaseToolpathGenerator):
    """
    Stage 2 Toolpath Generator.
    Converts a UniversalSlicedModel into a CLDataTrajectory: walls, top/bottom solid skins and
    sparse infill per layer, travel moves and extrusion volumes.

    The input model is never modified.
    """

    # ------------------------------------------------------------------ geometry helpers

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

    def _classify_rings(self, layer: UniversalLayer) -> Tuple[List[sg.Polygon], List[SpatialContour]]:
        """
        Turns the closed boundary rings of a layer into printable polygons with holes.
        Open contours are intentional paths (helix / vase mode / parametric toolpaths) and are
        returned separately; they are printed as they are. Closed but invalid rings are skipped.

        Nesting depth decides what a ring is: even depth = outer boundary of a part,
        odd depth = a hole of its parent. An island inside a hole has depth 2 again and
        is therefore a part of its own.
        """
        rings: List[sg.Polygon] = []
        open_paths: List[SpatialContour] = []
        for contour in layer.contours:
            pts = contour.points
            if len(pts) >= 2 and math.dist(pts[0], pts[-1]) > 1e-9:
                open_paths.append(contour)
                continue
            if len(pts) < 4:
                logger.warning("Layer %d: skipping a degenerate contour (%d points).", layer.layer_index, len(pts))
                continue
            try:
                poly = sg.Polygon([(p[0], p[1]) for p in pts])
            except SHAPELY_ERRORS as exc:
                logger.warning("Layer %d: invalid ring skipped (%s).", layer.layer_index, exc)
                continue
            if not poly.is_valid or poly.is_empty or poly.area < MIN_RING_AREA:
                logger.warning("Layer %d: skipping an invalid or empty ring.", layer.layer_index)
                continue
            rings.append(poly)

        rings.sort(key=lambda r: r.area, reverse=True)

        # drop exact duplicates
        unique: List[sg.Polygon] = []
        for r in rings:
            if not any(abs(u.area - r.area) <= 1e-9 * max(u.area, 1.0) and u.contains(r.representative_point())
                       for u in unique):
                unique.append(r)
        rings = unique

        n = len(rings)
        depth = [0] * n
        parent: List[Optional[int]] = [None] * n
        reps = [r.representative_point() for r in rings]
        for i in range(n):
            best = None
            for j in range(i):  # only larger rings can contain ring i
                if rings[j].contains(reps[i]) and (best is None or depth[j] > depth[best]):
                    best = j
            if best is not None:
                parent[i] = best
                depth[i] = depth[best] + 1

        parts: List[sg.Polygon] = []
        for i in range(n):
            if depth[i] % 2 != 0:
                continue
            holes = [rings[h].exterior.coords for h in range(n) if parent[h] == i and depth[h] % 2 == 1]
            try:
                part = sg.Polygon(rings[i].exterior.coords, holes)
            except SHAPELY_ERRORS as exc:
                logger.warning("Layer %d: could not build a part with holes (%s).", layer.layer_index, exc)
                continue
            if part.is_valid and not part.is_empty:
                parts.append(part)
            else:
                logger.warning("Layer %d: skipping an invalid part.", layer.layer_index)
        return parts, open_paths

    @staticmethod
    def _exposed(idx: int, direction: int, count: int, regions: Dict[int, BaseGeometry]) -> BaseGeometry:
        """Area of layer `idx` that is not covered by the next `count` layers in `direction`."""
        region = regions[idx]
        if count <= 0 or region.is_empty:
            return sg.Polygon()
        covered: Optional[BaseGeometry] = None
        for k in range(1, count + 1):
            neighbour = regions.get(idx + direction * k)
            if neighbour is None or neighbour.is_empty:
                return region  # nothing above/below: everything is exposed
            covered = neighbour if covered is None else covered.intersection(neighbour)
        try:
            return region.difference(covered)
        except SHAPELY_ERRORS as exc:
            logger.warning("Layer %d: skin computation failed (%s); treating the whole region as solid.", idx, exc)
            return region

    @staticmethod
    def _clean_skin(skin: BaseGeometry, region: BaseGeometry, line_width: float) -> BaseGeometry:
        """Removes slivers narrower than one extrusion line from a skin region."""
        if skin.is_empty:
            return sg.Polygon()
        if skin.area >= 0.999 * region.area:
            return region
        try:
            opened = skin.buffer(-line_width / 2.0).buffer(line_width / 2.0).intersection(region)
        except SHAPELY_ERRORS:
            return sg.Polygon()
        keep = [p for p in _polygons(opened) if p.area >= line_width * line_width * MIN_FILL_AREA_FACTOR]
        return unary_union(keep) if keep else sg.Polygon()

    def analyze_layers(self, model: UniversalSlicedModel, profile: MinimalPrintProfile) -> Dict[int, LayerAnalysis]:
        """
        Pre-pass over the whole model: parts, wall-eroded infill regions and, from the
        neighbouring layers, the bottom / top solid skins and the remaining sparse area.
        """
        d = profile.nozzle_diameter
        n_walls = profile.num_perimeters

        analysis: Dict[int, LayerAnalysis] = {}
        regions: Dict[int, BaseGeometry] = {}
        for layer in model.layers:
            parts, open_paths = self._classify_rings(layer)
            part_regions = [_erode(p, n_walls * d) for p in parts]
            nonempty = [r for r in part_regions if not r.is_empty]
            region = unary_union(nonempty) if nonempty else sg.Polygon()
            analysis[layer.layer_index] = LayerAnalysis(layer.layer_index, parts, open_paths=open_paths, part_regions=part_regions, region=region)
            regions[layer.layer_index] = region

        for idx, info in analysis.items():
            if info.region.is_empty:
                continue
            bottom = self._clean_skin(self._exposed(idx, -1, profile.bottom_solid_layers, regions), info.region, d)
            top = self._clean_skin(self._exposed(idx, +1, profile.top_solid_layers, regions), info.region, d)
            if not bottom.is_empty and not top.is_empty:
                try:
                    top = top.difference(bottom)
                except SHAPELY_ERRORS:
                    pass
            info.bottom_solid, info.top_solid = bottom, top
            try:
                info.sparse = info.region.difference(unary_union([bottom, top]))
            except SHAPELY_ERRORS as exc:
                logger.warning("Layer %d: sparse region computation failed (%s).", idx, exc)
                info.sparse = sg.Polygon()
        return analysis

    # ------------------------------------------------------------------ infill

    def _fill_polygon(self, poly: sg.Polygon, spacing: float, angle_deg: float,
                      cursor: Optional[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        """
        Rectilinear fill of ONE connected polygon.

        Scanlines are computed in a frame rotated by -angle. Pieces on adjacent scanlines are
        joined into one zig-zag only when the straight link lies inside the polygon
        (buffered by LINK_TOLERANCE_FACTOR * spacing); otherwise a new chain (= a real
        travel) is started. Returns a list of polylines in the original frame.
        """
        rot = sa.rotate(poly, -angle_deg, origin=(0, 0))
        minx, miny, maxx, maxy = rot.bounds
        k_first = math.ceil(miny / spacing - 0.5)
        k_last = math.floor(maxy / spacing - 0.5)

        # pieces[k] = list of (x0, x1) on scanline y = (k + 0.5) * spacing
        pieces: Dict[int, List[Tuple[float, float]]] = {}
        for k in range(k_first, k_last + 1):
            y = (k + 0.5) * spacing
            try:
                inter = rot.intersection(sg.LineString([(minx - 1.0, y), (maxx + 1.0, y)]))
            except SHAPELY_ERRORS as exc:
                logger.warning("Scanline intersection failed (%s); line skipped.", exc)
                continue
            lines = [inter] if inter.geom_type == "LineString" else (
                [g for g in getattr(inter, "geoms", []) if g.geom_type == "LineString"])
            segs = []
            for ln in lines:
                xs = [c[0] for c in ln.coords]
                x0, x1 = min(xs), max(xs)
                if x1 - x0 >= MIN_SEGMENT_FACTOR * spacing:
                    segs.append((x0, x1))
            if segs:
                pieces[k] = sorted(segs)

        if not pieces:
            return []

        unvisited = {(k, j) for k, segs in pieces.items() for j in range(len(segs))}
        allowed = prep(rot.buffer(LINK_TOLERANCE_FACTOR * spacing))

        def pt(k, x):
            return (x, (k + 0.5) * spacing)

        cur_rot = None
        if cursor is not None:
            cur_rot = sa.rotate(sg.Point(cursor), -angle_deg, origin=(0, 0))
            cur_rot = (cur_rot.x, cur_rot.y)

        chains_rot: List[List[Tuple[float, float]]] = []
        while unvisited:
            # --- choose where the next chain starts
            if cur_rot is None:
                k, j = min(unvisited)
                a, b = pt(k, pieces[k][j][0]), pt(k, pieces[k][j][1])
            else:
                best = None
                for (kk, jj) in unvisited:
                    for near, far in ((pieces[kk][jj][0], pieces[kk][jj][1]), (pieces[kk][jj][1], pieces[kk][jj][0])):
                        dist = math.dist(cur_rot, pt(kk, near))
                        key = (dist, kk, jj)
                        if best is None or key < best[0]:
                            best = (key, kk, jj, near, far)
                _, k, j, near, far = best
                a, b = pt(k, near), pt(k, far)
            unvisited.discard((k, j))
            chain = [a, b]
            head_k = tail_k = k          # scanline index at both ends of the chain
            reversed_once = False

            # --- extend over adjacent scanlines while the straight link stays inside the polygon.
            # The chain may have been started in the middle: when it gets stuck at one end it is
            # reversed once and continued at the other end.
            while True:
                candidates = []
                for kk in (tail_k - 1, tail_k + 1):
                    for jj, (x0, x1) in enumerate(pieces.get(kk, [])):
                        if (kk, jj) not in unvisited:
                            continue
                        for near, far in ((x0, x1), (x1, x0)):
                            candidates.append((math.dist(chain[-1], pt(kk, near)), kk, jj, near, far))
                candidates.sort()
                chosen = None
                for dist, kk, jj, near, far in candidates:
                    if allowed.covers(sg.LineString([chain[-1], pt(kk, near)])):
                        chosen = (kk, jj, near, far)
                        break
                if chosen is None:
                    if reversed_once:
                        break
                    chain.reverse()
                    head_k, tail_k = tail_k, head_k
                    reversed_once = True
                    continue
                kk, jj, near, far = chosen
                unvisited.discard((kk, jj))
                chain.extend([pt(kk, near), pt(kk, far)])
                tail_k = kk

            # print the chain starting at the end that is nearer to the current position
            if cur_rot is not None and math.dist(cur_rot, chain[-1]) < math.dist(cur_rot, chain[0]):
                chain.reverse()

            chains_rot.append(chain)
            cur_rot = chain[-1]

        chains = []
        for chain in chains_rot:
            line = sa.rotate(sg.LineString(chain), angle_deg, origin=(0, 0))
            chains.append([(float(x), float(y)) for x, y in line.coords])
        return chains

    # ------------------------------------------------------------------ features of one layer

    @staticmethod
    def _start_near(coords: List[Tuple[float, float]], cursor) -> List[Tuple[float, float]]:
        """Rotates a closed ring so that it starts at the vertex nearest to `cursor`."""
        if cursor is None or len(coords) < 4:
            return coords
        ring = coords[:-1]
        i = min(range(len(ring)), key=lambda n: math.dist(ring[n], cursor))
        ring = ring[i:] + ring[:i]
        return ring + [ring[0]]

    def _build_layer_contours(self, layer: UniversalLayer, info: LayerAnalysis,
                              profile: MinimalPrintProfile, strategy, cursor):
        d = profile.nozzle_diameter
        contours: List[SpatialContour] = []
        state = {"cursor": cursor}

        def add_contour(coords, f_type):
            if len(coords) < 2:
                return
            dense = [coords[0]]
            for i in range(1, len(coords)):
                dense.extend(self._subdivide_segment(dense[-1], coords[i], profile.max_segment_length))

            pts, normals, thicknesses = [], [], []
            for x, y in dense:
                if strategy:
                    pt_3d = strategy.project_to_layer(x, y, layer.layer_index)
                    norm = strategy.compute_normal(pt_3d, layer.layer_index)
                    thick = strategy.compute_thickness(pt_3d, layer.layer_index)
                else:
                    pt_3d = (x, y, layer.z_height)
                    norm = (0.0, 0.0, 1.0)
                    thick = profile.layer_height
                pts.append(pt_3d)
                normals.append(norm)
                thicknesses.append(thick)
            contours.append(SpatialContour(points=pts, normals=normals, thicknesses=thicknesses, feature_type=f_type))
            state["cursor"] = (dense[-1][0], dense[-1][1])

        density = _density_fraction(profile)
        parity_sign = 1.0 if layer.layer_index % 2 == 0 else -1.0
        n_walls = profile.num_perimeters

        for part, part_region in zip(info.parts, info.part_regions):
            # 1. walls: innermost first, the outer wall last
            for k in range(n_walls - 1, -1, -1):
                offset_poly = part if k == 0 else _erode(part, k * d)
                f_type = "outer_wall" if k == 0 else "inner_wall"
                for poly in _polygons(offset_poly):
                    for ring in [poly.exterior] + list(poly.interiors):
                        coords = self._start_near(list(ring.coords), state["cursor"])
                        add_contour(coords, f_type)

            # 2. skins and sparse infill, clipped to this part
            if part_region.is_empty:
                continue
            fills = (
                ("bottom_solid", info.bottom_solid, d, parity_sign * SOLID_ANGLE_DEG),
                ("top_solid", info.top_solid, d, parity_sign * SOLID_ANGLE_DEG),
                ("infill", info.sparse, (d / density) if density > 0 else None,
                 parity_sign * profile.infill_angle_deg),
            )
            for f_type, fill_region, spacing, angle in fills:
                if fill_region.is_empty or spacing is None:
                    continue
                try:
                    clipped = fill_region.intersection(part_region)
                except SHAPELY_ERRORS as exc:
                    logger.warning("Layer %d: clipping %s failed (%s).", layer.layer_index, f_type, exc)
                    continue
                for poly in _polygons(clipped):
                    if poly.area < d * d * MIN_FILL_AREA_FACTOR:
                        continue
                    for chain in self._fill_polygon(poly, spacing, angle, state["cursor"]):
                        add_contour(chain, f_type)

        return contours, state["cursor"]

    def build_layer_features(self, model: UniversalSlicedModel, profile: MinimalPrintProfile,
                             strategy=None) -> Dict[int, List[SpatialContour]]:
        """Generates the printable features (walls, skins, infill) for every layer."""
        analysis = self.analyze_layers(model, profile)
        features: Dict[int, List[SpatialContour]] = {}
        cursor = None
        for layer in sorted(model.layers, key=lambda l: l.layer_index):
            info = analysis[layer.layer_index]
            if not info.parts and not info.open_paths:
                logger.warning("Layer %d: no valid closed contour; the layer is skipped.", layer.layer_index)
                features[layer.layer_index] = []
                continue
            contours: List[SpatialContour] = []
            if info.parts:
                contours, cursor = self._build_layer_contours(layer, info, profile, strategy, cursor)
            # Open paths (spiral / vase / parametric) are printed exactly as Stage 1 produced them.
            for path in info.open_paths:
                contours.append(path.model_copy(update={"feature_type": "outer_wall"}))
                cursor = (path.points[-1][0], path.points[-1][1])
            features[layer.layer_index] = contours
        return features

    # ------------------------------------------------------------------ strategy from metadata

    @staticmethod
    def _strategy_from_metadata(model: UniversalSlicedModel):
        if "strategy" not in model.metadata:
            return None
        import trimesh
        from src.stage1_slicer.math_strategies import build_strategy

        strat_name = model.metadata["strategy"]
        # The metadata holds the RESOLVED parameters written by Stage 1, so an empty
        # mesh is enough: the strategies only need their math here.
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

    # ------------------------------------------------------------------ trajectory

    def generate_toolpath(self, model: UniversalSlicedModel, profile: MinimalPrintProfile) -> CLDataTrajectory:
        waypoints: List[CLDataWaypoint] = []
        last_pt = None

        strategy = self._strategy_from_metadata(model)
        features = self.build_layer_features(model, profile, strategy)

        for layer in sorted(model.layers, key=lambda l: l.layer_index):
            layer_idx = layer.layer_index
            for contour in features.get(layer_idx, []):
                if not contour.points:
                    continue

                first_pt = contour.points[0]
                first_norm = contour.normals[0]

                # Jumping from a previous contour/layer (and not in continuous spiral): travel move
                if last_pt is not None and not profile.continuous_spiral:
                    dist_to_start = math.dist(last_pt, first_pt)
                    if dist_to_start > 1e-6:
                        waypoints.append(CLDataWaypoint(
                            x=first_pt[0], y=first_pt[1], z=first_pt[2],
                            i=first_norm[0], j=first_norm[1], k=first_norm[2],
                            extrusion_volume=0.0,
                            feedrate=3000.0,  # default travel feedrate
                            is_travel_move=True,
                            feature_type=contour.feature_type,
                            layer_index=layer_idx
                        ))

                # Very first point of the print: a travel move to get there
                if not waypoints:
                    waypoints.append(CLDataWaypoint(
                        x=first_pt[0], y=first_pt[1], z=first_pt[2],
                        i=first_norm[0], j=first_norm[1], k=first_norm[2],
                        extrusion_volume=0.0,
                        feedrate=3000.0,
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
                            feedrate=1500.0,
                            is_travel_move=False,
                            feature_type=contour.feature_type,
                            layer_index=layer_idx
                        ))

                last_pt = first_pt

                for idx, (pt, norm) in enumerate(zip(contour.points[1:], contour.normals[1:])):
                    real_idx = idx + 1
                    dl = math.dist(last_pt, pt)

                    if dl > 1e-6:
                        # Local thickness if available, else nominal profile layer_height
                        local_h = profile.layer_height
                        if contour.thicknesses and len(contour.thicknesses) > real_idx:
                            local_h = contour.thicknesses[real_idx]

                        # V_ext = dL * local_layer_height * nozzle_diameter
                        vol = dl * local_h * profile.nozzle_diameter

                        waypoints.append(CLDataWaypoint(
                            x=pt[0], y=pt[1], z=pt[2],
                            i=norm[0], j=norm[1], k=norm[2],
                            extrusion_volume=vol,
                            feedrate=1500.0,  # default extrusion feedrate
                            is_travel_move=False,
                            feature_type=contour.feature_type,
                            layer_index=layer_idx
                        ))
                        last_pt = pt

        return CLDataTrajectory(waypoints=waypoints)
