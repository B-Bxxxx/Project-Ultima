import numpy as np
import trimesh
from typing import List, Any, Dict, Optional
from src.common.schemas import SlicerPluginParameterSchema, UniversalLayer, SpatialContour, UniversalSlicedModel
from src.stage1_slicer.base import BaseSlicerPlugin
from src.stage1_slicer.registry import PluginRegistry
from src.stage1_slicer.math_strategies import build_strategy
from src.common.simplify import simplify_ring

MIN_CONTOUR_LENGTH = 0.05      # mm: shorter traced contours are numerical debris


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
        strategy = build_strategy(strategy_name, geometry, parameters)

        # Metadata = input parameters + the RESOLVED values (mesh-derived start_z,
        # end_z, pivot_y, ...), so Stage 2 rebuilds the strategy identically.
        # Large numpy fields (vertex_deformations) are NOT stored: Stage 2 rebuilds
        # them from the layer contours, and numpy arrays are not JSON-serializable.
        metadata = {k: v for k, v in parameters.items() if k != "vertex_deformations"}
        metadata["strategy"] = strategy_name
        metadata.update(strategy.get_resolved_params())

        # raw contours follow the (possibly very fine) mesh; drop points that move the outline < tol (mm)
        # (external_field: Stage 2 rebuilds f from these contour points, so they must stay dense -> default off)
        tol = float(parameters.get("contour_tolerance", 0.0 if strategy_name == "external_field" else 0.005))
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
            # a plane that grazes a vertex leaves micro loops (a few microns long) - no material in them
            contours = [c for c in contours if self._path_length(c.points) >= MIN_CONTOUR_LENGTH]

            if contours:
                # Store default normals on the raw boundary contour so Stage 1 tests continue passing
                # (Stage 2 will evaluate the true normals for perimeter/infill generation anyway)
                for contour in contours:
                    pts_d = np.asarray(contour.points, float)
                    undeformed = strategy.undeform_points(pts_d, layer_idx)
                    if tol > 0.0:                                 # simplify the REAL 3D outline
                        undeformed = simplify_ring(undeformed, tol)
                    xy = undeformed[:, :2]
                    contour.points = [tuple(map(float, p)) for p in undeformed]
                    contour.normals = [tuple(map(float, n)) for n in strategy.normals_at(undeformed, layer_idx)]
                    contour.thicknesses = [float(t) for t in strategy.thicknesses_at(undeformed, layer_idx)]
                    vg, ng = strategy.layer_gaps_batch(xy, layer_idx)
                    contour.vertical_gaps = [float(v) for v in vg]
                    contour.normal_gaps = [float(v) for v in ng]
                    contour.feature_type = "boundary"

                z_nominal = plane_origin[2]
                layers.append(UniversalLayer(layer_index=layer_idx, z_height=z_nominal, contours=contours))

        return UniversalSlicedModel(layers=layers, metadata=metadata)

    @staticmethod
    def _path_length(points) -> float:
        P = np.asarray(points, float)
        return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0

    def _process_trimesh_segments_to_contours(self, segments: np.ndarray, default_normal: List[float] = [0, 0, 1]) -> List[SpatialContour]:
        """
        Converts unordered line segments (N, 2, 3) into ordered SpatialContours.
        """
        # Segments -> polylines by TRACING (every segment is used exactly once).
        # Why not nx.find_cycle: a plane that passes through a mesh vertex, or two vertices that
        # round to the same key, gives nodes of degree > 2 or zero-length (self-loop) edges; the
        # first cycle found then skips part of the loop (jagged contour) or is a 1-edge loop that
        # silently throws the whole ring away.
        precision = 4
        exact = {}
        adj = {}
        for p1, p2 in segments:
            k1 = (round(float(p1[0]), precision), round(float(p1[1]), precision), round(float(p1[2]), precision))
            k2 = (round(float(p2[0]), precision), round(float(p2[1]), precision), round(float(p2[2]), precision))
            if k1 == k2:
                continue                                   # zero-length segment
            exact.setdefault(k1, p1)
            exact.setdefault(k2, p2)
            adj.setdefault(k1, []).append(k2)
            adj.setdefault(k2, []).append(k1)

        used = set()

        def take(u):
            for v in adj[u]:
                e = (u, v) if u <= v else (v, u)
                if e not in used:
                    used.add(e)
                    return v
            return None

        def trace(start):
            path = [start]
            cur = start
            while True:
                nxt = take(cur)
                if nxt is None:
                    return path
                path.append(nxt)
                cur = nxt

        # open paths first (they start at nodes of odd degree), then the closed loops
        order = [k for k in adj if len(adj[k]) % 2 == 1] + list(adj)
        contours = []
        for start in order:
            while True:
                path = trace(start)
                if len(path) < 2:
                    break
                points = [tuple(float(x) for x in exact[k]) for k in path]
                normals = [tuple(float(x) for x in default_normal)] * len(points)
                contours.append(SpatialContour(points=points, normals=normals))

        return contours
