import numpy as np
import trimesh
from typing import List, Any, Dict, Optional
from src.common.schemas import SlicerPluginParameterSchema, UniversalLayer, SpatialContour, UniversalSlicedModel
from src.stage1_slicer.base import BaseSlicerPlugin
from src.stage1_slicer.registry import PluginRegistry
from src.stage1_slicer.math_strategies import (
    PlanarStrategy, ProgressiveTiltStrategy, ConicalStrategy,
    CustomExpressionStrategy, ExternalScalarFieldStrategy
)

@PluginRegistry.register("universal_field_slicer")
class UniversalFieldSlicerPlugin(BaseSlicerPlugin):
    """
    Universal slicing engine plugin acting as a geometric execution engine.
    Utilizes decoupled strategies from math_strategies.py.
    """

    @classmethod
    def get_parameter_schema(cls) -> List[SlicerPluginParameterSchema]:
        return [
            SlicerPluginParameterSchema(
                name="strategy", type="str", description="Strategy: 'planar', 'progressive_tilt', 'conical', 'custom_expr', 'external_field'", default="planar"
            ),
            SlicerPluginParameterSchema(
                name="layer_height", type="float", description="Nominal Z layer height", default=0.2
            ),
            SlicerPluginParameterSchema(
                name="start_z", type="float", description="Starting Z coordinate for slicing", default=0.0
            ),
            SlicerPluginParameterSchema(
                name="end_z", type="float", description="Ending Z coordinate for slicing", default=10.0
            ),
            # progressive tilt parameters
            SlicerPluginParameterSchema(
                name="start_tilt_deg", type="float", description="Starting tilt angle", default=0.0
            ),
            SlicerPluginParameterSchema(
                name="end_tilt_deg", type="float", description="Ending tilt angle", default=0.0
            ),
            # conical parameters
            SlicerPluginParameterSchema(
                name="cone_angle_deg", type="float", description="Angle of the conical field", default=15.0
            ),
            # custom expression
            SlicerPluginParameterSchema(
                name="expression", type="str", description="Expression f(x, y) for custom strategy", default="0.0"
            ),
        ]

    def slice(self, geometry: Any, parameters: Dict[str, Any]) -> UniversalSlicedModel:
        """
        geometry is expected to be a trimesh.Trimesh object.
        """
        if not isinstance(geometry, trimesh.Trimesh):
            raise ValueError("geometry must be a trimesh.Trimesh object")

        strategy_name = parameters.get("strategy", parameters.get("mode", "planar"))

        # Load the selected mathematical strategy
        if strategy_name == "planar":
            strategy = PlanarStrategy(geometry, parameters)
        elif strategy_name == "progressive_tilt":
            strategy = ProgressiveTiltStrategy(geometry, parameters)
        elif strategy_name == "conical" or strategy_name == "conical_or_curved":
            strategy = ConicalStrategy(geometry, parameters)
        elif strategy_name == "custom_expr":
            strategy = CustomExpressionStrategy(geometry, parameters)
        elif strategy_name == "external_field":
            strategy = ExternalScalarFieldStrategy(geometry, parameters)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        layers = []

        # 1. Forward Mapping (Deform Mesh)
        deformed_mesh = strategy.deform_mesh()

        # 2. Slice Mesh
        num_layers = strategy.get_layer_count()
        for layer_idx in range(num_layers):
            plane_origin, plane_normal = strategy.get_slice_plane(layer_idx)

            # Slice the deformed mesh
            path_2d = trimesh.intersections.mesh_plane(deformed_mesh, plane_normal, plane_origin)

            if path_2d is None or len(path_2d) == 0:
                continue

            # Extract raw contours from intersection segments
            contours = self._process_trimesh_segments_to_contours(path_2d)

            # 3. Generate Infill & Perimeters in 2D using shapely, then undeform
            # Subdivide segments before undeforming to ensure they hug the curved 3D surface
            nozzle_dia = parameters.get("nozzle_diameter", 0.4)
            num_perims = parameters.get("num_perimeters", 2)
            infill_density = parameters.get("infill_density", 0.2)
            infill_angle = parameters.get("infill_angle_deg", 45.0)
            max_segment_len = parameters.get("max_segment_length", 1.0)

            enriched_contours = self._generate_infill_and_perimeters(
                contours, nozzle_dia, num_perims, infill_density, infill_angle, layer_idx, max_segment_len
            )

            # 4. Inverse Mapping (Undeform points & evaluate normals)
            undeformed_contours = []
            for contour in enriched_contours:
                undeformed_points = []
                undeformed_normals = []
                undeformed_thicknesses = []

                for pt in contour.points:
                    # pt is (x, y, z_prime)
                    u_pt = strategy.undeform_point(pt, layer_idx)
                    norm = strategy.compute_normal(u_pt, layer_idx)
                    t = strategy.compute_thickness(u_pt, layer_idx)

                    undeformed_points.append(u_pt)
                    undeformed_normals.append(norm)
                    undeformed_thicknesses.append(t)

                undeformed_contours.append(SpatialContour(
                    points=undeformed_points,
                    normals=undeformed_normals,
                    thicknesses=undeformed_thicknesses,
                    feature_type=contour.feature_type
                ))

            if undeformed_contours:
                # The z_height of the layer conceptually remains the slicing plane's nominal Z
                z_nominal = plane_origin[2]
                layers.append(UniversalLayer(layer_index=layer_idx, z_height=z_nominal, contours=undeformed_contours))

        return UniversalSlicedModel(layers=layers)


    def _subdivide_segment(self, p1: tuple, p2: tuple, max_len: float) -> List[tuple]:
        dist = np.linalg.norm(np.array(p1) - np.array(p2))
        if dist <= max_len or dist == 0:
            return [p2]

        steps = int(np.ceil(dist / max_len))
        points = []
        for i in range(1, steps + 1):
            t = i / steps
            pt = (
                p1[0] + t * (p2[0] - p1[0]),
                p1[1] + t * (p2[1] - p1[1]),
                p1[2] + t * (p2[2] - p1[2])
            )
            points.append(pt)
        return points

    def _generate_infill_and_perimeters(self, base_contours: List[SpatialContour], nozzle_dia: float, num_perims: int, infill_density: float, infill_angle: float, layer_idx: int, max_seg: float) -> List[SpatialContour]:
        from shapely.geometry import Polygon, LineString, Point
        from shapely.ops import unary_union
        import math

        if not base_contours:
            return []

        z_val = base_contours[0].points[0][2]
        out_contours = []

        # 1. Build Polygons from base_contours
        polygons = []
        for c in base_contours:
            if len(c.points) >= 3:
                # Remove z for shapely
                pts_2d = [(p[0], p[1]) for p in c.points]
                poly = Polygon(pts_2d)
                if poly.is_valid and not poly.is_empty:
                    polygons.append(poly)

        if not polygons:
            return base_contours

        # Merge overlapping/touching polygons
        merged_poly = unary_union(polygons)
        poly_list = [merged_poly] if isinstance(merged_poly, Polygon) else list(merged_poly.geoms)

        for poly in poly_list:
            # Perimeters
            current_poly = poly
            for i in range(num_perims):
                # Offset inward
                offset_dist = -(i * nozzle_dia + nozzle_dia / 2.0)
                perim_poly = poly.buffer(offset_dist)

                if perim_poly.is_empty:
                    break

                p_list = [perim_poly] if isinstance(perim_poly, Polygon) else list(perim_poly.geoms)
                for p in p_list:
                    # Exterior wall
                    pts_3d = []
                    coords = list(p.exterior.coords)
                    # Subdivide
                    for k in range(len(coords)-1):
                        p1 = (coords[k][0], coords[k][1], z_val)
                        if len(pts_3d) == 0:
                            pts_3d.append(p1)
                        p2 = (coords[k+1][0], coords[k+1][1], z_val)
                        sub_pts = self._subdivide_segment(pts_3d[-1], p2, max_seg)
                        pts_3d.extend(sub_pts)

                    ftype = "outer_wall" if i == 0 else "inner_wall"
                    out_contours.append(SpatialContour(points=pts_3d, normals=[(0,0,1)]*len(pts_3d), feature_type=ftype))

                    # Intersect holes if any
                    for interior in p.interiors:
                        pts_3d = []
                        coords = list(interior.coords)
                        for k in range(len(coords)-1):
                            p1 = (coords[k][0], coords[k][1], z_val)
                            if len(pts_3d) == 0:
                                pts_3d.append(p1)
                            p2 = (coords[k+1][0], coords[k+1][1], z_val)
                            sub_pts = self._subdivide_segment(pts_3d[-1], p2, max_seg)
                            pts_3d.extend(sub_pts)
                        out_contours.append(SpatialContour(points=pts_3d, normals=[(0,0,1)]*len(pts_3d), feature_type=ftype))

                current_poly = perim_poly

            # Infill
            if infill_density > 0.01 and not current_poly.is_empty:
                # Buffer inward slightly more for infill boundary
                infill_poly = current_poly.buffer(-nozzle_dia / 2.0)
                if not infill_poly.is_empty:
                    spacing = nozzle_dia / infill_density
                    angle = infill_angle if layer_idx % 2 == 0 else -infill_angle
                    rad = math.radians(angle)

                    bounds = infill_poly.bounds # minx, miny, maxx, maxy
                    cx = (bounds[0] + bounds[2]) / 2.0
                    cy = (bounds[1] + bounds[3]) / 2.0
                    diag = math.hypot(bounds[2]-bounds[0], bounds[3]-bounds[1])

                    lines = []
                    num_lines = int(diag / spacing) + 2

                    for i in range(-num_lines, num_lines):
                        d = i * spacing
                        # Line equation: x*cos(rad) + y*sin(rad) = d
                        # Or parameterized from center
                        px = cx + d * math.cos(rad + math.pi/2)
                        py = cy + d * math.sin(rad + math.pi/2)

                        dx = (diag/2) * math.cos(rad)
                        dy = (diag/2) * math.sin(rad)

                        p1 = (px - dx, py - dy)
                        p2 = (px + dx, py + dy)
                        lines.append(LineString([p1, p2]))

                    for line in lines:
                        intersection = infill_poly.intersection(line)
                        if intersection.is_empty:
                            continue

                        segments = []
                        if isinstance(intersection, LineString):
                            segments.append(intersection)
                        elif hasattr(intersection, 'geoms'):
                            segments.extend([g for g in intersection.geoms if isinstance(g, LineString)])

                        for seg in segments:
                            pts_3d = []
                            coords = list(seg.coords)
                            for k in range(len(coords)-1):
                                p1 = (coords[k][0], coords[k][1], z_val)
                                if len(pts_3d) == 0:
                                    pts_3d.append(p1)
                                p2 = (coords[k+1][0], coords[k+1][1], z_val)
                                sub_pts = self._subdivide_segment(pts_3d[-1], p2, max_seg)
                                pts_3d.extend(sub_pts)

                            if pts_3d:
                                out_contours.append(SpatialContour(points=pts_3d, normals=[(0,0,1)]*len(pts_3d), feature_type="infill"))

        return out_contours

    def _process_trimesh_segments_to_contours(self, segments: np.ndarray, default_normal: List[float] = [0, 0, 1]) -> List[SpatialContour]:
        """
        Converts unordered line segments (N, 2, 3) into ordered SpatialContours.
        """
        import networkx as nx
        G = nx.Graph()
        precision = 4

        for p1, p2 in segments:
            p1_t = (round(p1[0], precision), round(p1[1], precision), round(p1[2], precision))
            p2_t = (round(p2[0], precision), round(p2[1], precision), round(p2[2], precision))
            G.add_node(p1_t, exact=p1)
            G.add_node(p2_t, exact=p2)
            G.add_edge(p1_t, p2_t)

        contours = []
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            try:
                cycle = nx.find_cycle(subgraph)
                points = []
                for u, v in cycle:
                    points.append(tuple(float(x) for x in G.nodes[u]['exact']))
                if points:
                    points.append(points[0])
                normals = [tuple(float(x) for x in default_normal)] * len(points)
                contours.append(SpatialContour(points=points, normals=normals))
            except nx.NetworkXNoCycle:
                path_nodes = list(nx.dfs_preorder_nodes(subgraph))
                points = [tuple(float(x) for x in G.nodes[n]['exact']) for n in path_nodes]
                normals = [tuple(float(x) for x in default_normal)] * len(points)
                contours.append(SpatialContour(points=points, normals=normals))

        return contours
