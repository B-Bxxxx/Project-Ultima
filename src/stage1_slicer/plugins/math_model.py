import numpy as np
from typing import List, Any, Dict
from src.common.schemas import SlicerPluginParameterSchema, UniversalLayer, SpatialContour, UniversalSlicedModel
from src.stage1_slicer.base import BaseSlicerPlugin
from src.stage1_slicer.registry import PluginRegistry

@PluginRegistry.register("universal_math_slicer")
class UniversalMathSlicerPlugin(BaseSlicerPlugin):
    """
    Universal mathematical slicing engine plugin.
    Generates layers based on mathematical guide curves / parametric surface equations.
    """

    @classmethod
    def get_parameter_schema(cls) -> List[SlicerPluginParameterSchema]:
        return [
            SlicerPluginParameterSchema(
                name="radius", type="float", description="Radius of the cylinder (mm)", default=20.0
            ),
            SlicerPluginParameterSchema(
                name="height", type="float", description="Total height of the cylinder (mm)", default=50.0
            ),
            SlicerPluginParameterSchema(
                name="layer_height", type="float", description="Height per layer (mm)", default=0.2
            ),
            SlicerPluginParameterSchema(
                name="segments_per_layer", type="int", description="Number of line segments per circular layer", default=100
            ),
            SlicerPluginParameterSchema(
                name="pitch", type="float", description="Pitch of the helix (mm/rev). 0 for planar layers.", default=0.0
            ),
        ]

    def slice(self, geometry: Any, parameters: Dict[str, Any]) -> UniversalSlicedModel:
        """
        geometry is unused for math slicer; it relies entirely on mathematical generation.
        Here we implement a parametric helical cylinder as a baseline example.
        """
        radius = parameters.get("radius", 20.0)
        height = parameters.get("height", 50.0)
        layer_height = parameters.get("layer_height", 0.2)
        segments = parameters.get("segments_per_layer", 100)
        pitch = parameters.get("pitch", 0.0)

        num_layers = int(height / layer_height)
        layers = []

        # Helical continuous or standard planar
        for layer_idx in range(num_layers):
            base_z = layer_idx * layer_height

            points = []
            normals = []
            for i in range(segments):
                # t goes from 0 to 2*pi for one revolution
                t = (i / segments) * 2 * np.pi

                x = radius * np.cos(t)
                y = radius * np.sin(t)

                # Z increases throughout the layer if it's a helix (vase mode)
                # If pitch=0, it's a flat layer
                z_offset = (i / segments) * pitch
                z = base_z + z_offset

                points.append((float(x), float(y), float(z)))

                # For a simple cylinder, the normal points purely outwards horizontally
                nx, ny, nz = x / radius, y / radius, 0.0
                normals.append((float(nx), float(ny), float(nz)))

            # Close the loop
            points.append(points[0])
            normals.append(normals[0])

            contour = SpatialContour(points=points, normals=normals)

            layer = UniversalLayer(
                layer_index=layer_idx,
                z_height=base_z,
                contours=[contour]
            )
            layers.append(layer)

        return UniversalSlicedModel(layers=layers)
