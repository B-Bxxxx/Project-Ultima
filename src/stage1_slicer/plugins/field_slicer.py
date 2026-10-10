import numpy as np
import trimesh
from typing import List, Any, Dict, Optional
from src.common.schemas import SlicerPluginParameterSchema, UniversalLayer, SpatialContour, UniversalSlicedModel
from src.stage1_slicer.base import BaseSlicerPlugin
from src.stage1_slicer.registry import PluginRegistry
from src.stage1_slicer.math_strategies import build_strategy

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

            if contours:
                # Store default normals on the raw boundary contour so Stage 1 tests continue passing
                # (Stage 2 will evaluate the true normals for perimeter/infill generation anyway)
                for contour in contours:
                    undeformed_normals = []
                    undeformed_pts = []
                    thicknesses = []
                    for pt in contour.points:
                        u_pt = strategy.undeform_point(pt, layer_idx)
                        norm = strategy.compute_normal(u_pt, layer_idx)
                        thick = strategy.compute_thickness(u_pt, layer_idx)
                        undeformed_pts.append(u_pt)
                        undeformed_normals.append(norm)
                        thicknesses.append(thick)
                    contour.points = undeformed_pts
                    contour.normals = undeformed_normals
                    contour.thicknesses = thicknesses
                    contour.feature_type = "boundary"

                z_nominal = plane_origin[2]
                layers.append(UniversalLayer(layer_index=layer_idx, z_height=z_nominal, contours=contours))

        return UniversalSlicedModel(layers=layers, metadata=metadata)

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
