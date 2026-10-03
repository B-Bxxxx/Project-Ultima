import numpy as np
import trimesh
from typing import List, Any, Dict, Optional, Tuple
from src.common.schemas import SlicerPluginParameterSchema, UniversalLayer, SpatialContour, UniversalSlicedModel
from src.stage1_slicer.base import BaseSlicerPlugin
from src.stage1_slicer.registry import PluginRegistry

@PluginRegistry.register("universal_field_slicer")
class UniversalFieldSlicerPlugin(BaseSlicerPlugin):
    """
    Universal slicing engine plugin using trimesh.
    Supports Planar, Progressive Tilt, and Non-Planar (Curved/Conical) modes.
    """

    @classmethod
    def get_parameter_schema(cls) -> List[SlicerPluginParameterSchema]:
        return [
            SlicerPluginParameterSchema(
                name="mode", type="str", description="Slicing mode: 'planar', 'progressive_tilt', or 'conical_or_curved'", default="planar"
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
            SlicerPluginParameterSchema(
                name="start_tilt_deg", type="float", description="Starting tilt angle (for progressive_tilt)", default=0.0
            ),
            SlicerPluginParameterSchema(
                name="end_tilt_deg", type="float", description="Ending tilt angle (for progressive_tilt)", default=0.0
            ),
            SlicerPluginParameterSchema(
                name="cone_angle_deg", type="float", description="Angle of the conical field (for conical_or_curved)", default=15.0
            ),
        ]

    def slice(self, geometry: Any, parameters: Dict[str, Any]) -> UniversalSlicedModel:
        """
        geometry is expected to be a trimesh.Trimesh object.
        """
        if not isinstance(geometry, trimesh.Trimesh):
            raise ValueError("geometry must be a trimesh.Trimesh object")

        mode = parameters.get("mode", "planar")
        layer_height = parameters.get("layer_height", 0.2)
        start_z = parameters.get("start_z", 0.0)
        end_z = parameters.get("end_z", 10.0)

        num_layers = max(1, int(np.ceil((end_z - start_z) / layer_height)))
        layers = []

        mesh = geometry.copy()

        if mode == "planar":
            for layer_idx in range(num_layers):
                z = start_z + layer_idx * layer_height
                plane_origin = [0, 0, z]
                plane_normal = [0, 0, 1]

                layer = self._slice_mesh_with_plane(mesh, plane_origin, plane_normal, layer_idx, z)
                if layer and layer.contours:
                    layers.append(layer)

        elif mode == "progressive_tilt":
            start_tilt = np.radians(parameters.get("start_tilt_deg", 0.0))
            end_tilt = np.radians(parameters.get("end_tilt_deg", 0.0))

            for layer_idx in range(num_layers):
                t = layer_idx / max(1, (num_layers - 1))
                tilt = start_tilt + t * (end_tilt - start_tilt)

                # Z progresses normally along the vertical axis
                z = start_z + layer_idx * layer_height

                # Plane tilts around X axis (Y moves, Z moves)
                # Normal vector of the tilted plane:
                plane_normal = [0, -np.sin(tilt), np.cos(tilt)]

                # The plane must pass through the Z height at the center (X=0, Y=0)
                plane_origin = [0, 0, z]

                layer = self._slice_mesh_with_plane(mesh, plane_origin, plane_normal, layer_idx, z)
                if layer and layer.contours:
                    layers.append(layer)

        elif mode == "conical_or_curved":
            # Deform -> Slice -> Undeform
            cone_angle = np.radians(parameters.get("cone_angle_deg", 15.0))
            tan_alpha = np.tan(cone_angle)

            # Deform function: z' = z + r * tan(alpha)
            # Inverse (Undeform): z = z' - r * tan(alpha)
            # So to slice at flat z'_plane, the real z was lower.
            # We deform the mesh vertices UPWARD by r*tan(alpha)

            deformed_mesh = mesh.copy()
            r = np.linalg.norm(deformed_mesh.vertices[:, :2], axis=1)
            deformed_mesh.vertices[:, 2] += r * tan_alpha

            for layer_idx in range(num_layers):
                z_prime = start_z + layer_idx * layer_height
                plane_origin = [0, 0, z_prime]
                plane_normal = [0, 0, 1]

                # Slice the deformed mesh with a flat plane
                path_2d = trimesh.intersections.mesh_plane(deformed_mesh, plane_normal, plane_origin)

                if path_2d is None or len(path_2d) == 0:
                    continue

                # path_2d is a (n, 2, 3) array of line segments
                contours = self._process_trimesh_segments_to_contours(path_2d)

                undeformed_contours = []
                for contour in contours:
                    undeformed_points = []
                    undeformed_normals = []

                    for pt in contour.points:
                        x, y, zp = pt
                        rad = np.sqrt(x**2 + y**2)

                        # Undeform Z
                        real_z = zp - rad * tan_alpha
                        undeformed_points.append((float(x), float(y), float(real_z)))

                        # Analytical normal gradient [ -df/dx, -df/dy, 1 ] normalized
                        # f(x,y) = sqrt(x^2+y^2)*tan(alpha)
                        # df/dx = (x/r)*tan(alpha)
                        if rad > 1e-6:
                            df_dx = (x / rad) * tan_alpha
                            df_dy = (y / rad) * tan_alpha
                        else:
                            df_dx = 0.0
                            df_dy = 0.0

                        # The surface normal is orthogonal to the slicing surface
                        norm_vec = np.array([-df_dx, -df_dy, 1.0])
                        norm_vec /= np.linalg.norm(norm_vec)

                        undeformed_normals.append((float(norm_vec[0]), float(norm_vec[1]), float(norm_vec[2])))

                    undeformed_contours.append(SpatialContour(points=undeformed_points, normals=undeformed_normals))

                if undeformed_contours:
                    layers.append(UniversalLayer(layer_index=layer_idx, z_height=z_prime, contours=undeformed_contours))

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return UniversalSlicedModel(layers=layers)


    def _slice_mesh_with_plane(self, mesh: trimesh.Trimesh, plane_origin: List[float], plane_normal: List[float], layer_idx: int, z_height: float) -> Optional[UniversalLayer]:
        """
        Helper to intersect a mesh with a plane and extract contours.
        Returns a UniversalLayer or None.
        """
        segments = trimesh.intersections.mesh_plane(mesh, plane_normal, plane_origin)
        if segments is None or len(segments) == 0:
            return None

        contours = self._process_trimesh_segments_to_contours(segments, default_normal=plane_normal)
        if not contours:
            return None

        return UniversalLayer(layer_index=layer_idx, z_height=z_height, contours=contours)


    def _process_trimesh_segments_to_contours(self, segments: np.ndarray, default_normal: List[float] = [0, 0, 1]) -> List[SpatialContour]:
        """
        Converts unordered line segments (N, 2, 3) into ordered SpatialContours.
        """
        # Create a graph to order the segments
        import networkx as nx

        G = nx.Graph()

        # Round coordinates slightly to handle floating point errors when connecting segments
        precision = 4

        for p1, p2 in segments:
            p1_t = (round(p1[0], precision), round(p1[1], precision), round(p1[2], precision))
            p2_t = (round(p2[0], precision), round(p2[1], precision), round(p2[2], precision))

            # Map back to exact floats
            G.add_node(p1_t, exact=p1)
            G.add_node(p2_t, exact=p2)
            G.add_edge(p1_t, p2_t)

        contours = []

        # Extract connected components (cycles)
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)

            # Find an Eulerian path/circuit or just traverse simple cycles
            # Since these are contours, they should ideally form 2-regular graphs (cycles)
            # Find a cycle in the subgraph
            try:
                cycle = nx.find_cycle(subgraph)

                points = []
                for u, v in cycle:
                    points.append(tuple(float(x) for x in G.nodes[u]['exact']))

                # Close loop
                if points:
                    points.append(points[0])

                normals = [tuple(float(x) for x in default_normal)] * len(points)

                contours.append(SpatialContour(points=points, normals=normals))
            except nx.NetworkXNoCycle:
                # If it's not a cycle, maybe an open path (e.g. slicing a plane). Just traverse the edges in order
                path_nodes = list(nx.dfs_preorder_nodes(subgraph))
                points = [tuple(float(x) for x in G.nodes[n]['exact']) for n in path_nodes]
                normals = [tuple(float(x) for x in default_normal)] * len(points)
                contours.append(SpatialContour(points=points, normals=normals))

        return contours
