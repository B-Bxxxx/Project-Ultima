import trimesh
import numpy as np
from src.stage1_slicer.plugins.field_slicer import UniversalFieldSlicerPlugin
from src.stage2_toolpath.contour_generator import StandardToolpathGenerator
from src.common.schemas import MinimalPrintProfile

mesh = trimesh.creation.box(extents=(30, 30, 10))
mesh.apply_translation([0, 0, 5])
slicer = UniversalFieldSlicerPlugin()
params = {
    "strategy": "progressive_tilt",
    "layer_height": 2.0,
    "start_z": 0.0,
    "end_z": 10.0,
    "start_tilt_deg": 0.0,
    "end_tilt_deg": 30.0,
    "transition_height": 0.0,
    "nozzle_diameter": 0.4,
    "num_perimeters": 2,
    "infill_density": 0.2,
    "max_segment_length": 2.0
}
sliced_model = slicer.slice(mesh, params)

profile = MinimalPrintProfile(layer_height=2.0, nozzle_diameter=0.4)
generator = StandardToolpathGenerator()
# Let's inspect the `generate_2d_features` calls:
strategy = generator._generate_2d_features.__defaults__
print("Test")
