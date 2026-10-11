from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from typing import List, Optional, Tuple, Literal, Any, Dict

class SlicerPluginParameterSchema(BaseModel):
    """
    Base model for dynamic UI/CLI parameter discovery for Slicer Plugins.
    """
    name: str = Field(..., description="Name of the parameter")
    type: str = Field(..., description="Data type of the parameter (e.g. float, int, bool, str)")
    description: str = Field(..., description="Description for UI/CLI prompt")
    default: Any = Field(None, description="Default value")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Optional constraints (e.g., min, max, choices)")

class MinimalPrintProfile(BaseModel):
    """
    Minimal standard print profile for essential FDM parameters.
    Speeds are feedrates in mm/min, distances in mm.
    """
    model_config = ConfigDict(populate_by_name=True)

    # --- geometry / shells ---
    num_perimeters: int = Field(
        2, ge=0, validation_alias=AliasChoices("num_perimeters", "wall_count"),
        description="Number of perimeters (walls). Profile files may also call it 'wall_count'.")
    bottom_solid_layers: int = Field(3, ge=0, description="Number of solid bottom skin layers (0 disables the bottom skin)")
    top_solid_layers: int = Field(3, ge=0, description="Number of solid top skin layers (0 disables the top skin)")
    infill_density: float = Field(0.2, ge=0.0, le=1.0, description="Sparse infill density as a FRACTION 0..1 (0.2 = 20 %)")
    infill_pattern: str = Field("lines", description="Infill pattern. Only 'lines' (parallel zigzag) is implemented; other values fall back to it with a warning.")
    infill_angle_deg: float = Field(45.0, description="Angle of infill in degrees")
    nozzle_diameter: float = Field(0.4, gt=0.0, description="Nozzle diameter in mm (also the line width)")
    layer_height: float = Field(0.2, gt=0.0, description="Layer height in mm")
    continuous_spiral: bool = Field(False, description="True if continuous single-wall spiral/vase mode")
    filament_diameter: float = Field(1.75, gt=0.0, description="Filament diameter in mm")
    max_segment_length: float = Field(1.0, gt=0.0, description="Maximum length of a segment for surface conformity")
    include_rotary_axes: bool = Field(True, description="True to output 5-axis G-code (with B and C). False for strictly 3-axis G-code.")

    # --- speeds (mm/min) ---
    print_speed: float = Field(1500.0, gt=0.0, description="Feedrate of extruding moves in mm/min")
    travel_speed: float = Field(3000.0, gt=0.0, description="Feedrate of travel moves in mm/min")
    z_speed: float = Field(1200.0, gt=0.0, description="Feedrate of the Z-hop moves in mm/min")
    retract_speed: float = Field(2400.0, gt=0.0, description="Feedrate of retraction and priming in mm/min")

    # --- travel behaviour ---
    retract_distance: float = Field(0.8, ge=0.0, description="Retraction distance in mm for travel moves (0 disables retraction)")
    z_hop: float = Field(0.4, ge=0.0, description="Z-hop height in mm during travel moves (0 disables the hop)")
    min_retract_travel: float = Field(1.5, ge=0.0, description="Travels shorter than this (mm) are made directly, without retraction and Z-hop")
    hop_short_travels: bool = Field(False, description="Z-hop also on travels shorter than min_retract_travel (they are still made without retraction).")
    travel_command: Literal["G0", "G1"] = Field("G0", description="G-code command used for non-extruding moves (G0 rapid or G1 linear)")

    @property
    def wall_count(self) -> int:
        """Alias of num_perimeters (kept for older profiles and code)."""
        return self.num_perimeters

class SpatialContour(BaseModel):
    """
    Represents an ordered list of 3D points and normal vectors for a spatial contour.
    """
    points: List[Tuple[float, float, float]] = Field(..., description="Ordered list of (X, Y, Z) coordinates")
    normals: List[Tuple[float, float, float]] = Field(..., description="Ordered list of (I, J, K) surface normals corresponding to the points")
    thicknesses: Optional[List[float]] = Field(None, description="Ordered list of local layer thicknesses at each point")
    feature_type: str = Field("boundary", description="Type of feature: boundary, outer_wall, inner_wall, bottom_solid, top_solid or infill (sparse)")

class UniversalLayer(BaseModel):
    """
    A single layer consisting of multiple spatial contours.
    Stage 1 -> Stage 2 contract.
    """
    layer_index: int = Field(..., description="Index of this layer")
    z_height: float = Field(..., description="Nominal Z height of the layer")
    z_prime: Optional[float] = Field(None, description="Deformed Z space height (if applicable)")
    contours: List[SpatialContour] = Field(..., description="List of spatial contours forming this layer")

class UniversalSlicedModel(BaseModel):
    """
    A container holding multiple universal layers.
    Stage 1 -> Stage 2 contract.
    """
    layers: List[UniversalLayer] = Field(..., description="List of universal layers")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata containing parameters used during slicing")

    def to_json_file(self, path: str) -> None:
        with open(path, 'w') as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def from_json_file(cls, path: str) -> "UniversalSlicedModel":
        with open(path, 'r') as f:
            return cls.model_validate_json(f.read())

class CLDataWaypoint(BaseModel):
    """
    A single machine-agnostic cutter location waypoint.
    """
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    z: float = Field(..., description="Z coordinate")
    i: float = Field(0.0, description="I component of tool axis vector")
    j: float = Field(0.0, description="J component of tool axis vector")
    k: float = Field(1.0, description="K component of tool axis vector")
    extrusion_volume: float = Field(0.0, description="Extrusion volume for this move")
    feedrate: float = Field(..., description="Feedrate in mm/min")
    is_travel_move: bool = Field(False, description="True if this is a travel move (non-extruding)")
    feature_type: str = Field("outer_wall", description="Type of feature: outer_wall, inner_wall, bottom_solid, top_solid or infill (sparse)")
    layer_index: int = Field(0, description="Index of the layer this waypoint belongs to (same numbering as UniversalLayer.layer_index; a travel move belongs to the layer it travels INTO)")

class CLDataTrajectory(BaseModel):
    """
    A collection of toolpath waypoints.
    Stage 2 -> Stage 3 contract.
    """
    waypoints: List[CLDataWaypoint] = Field(..., description="Ordered list of cutter location waypoints")

    def to_json_file(self, path: str) -> None:
        with open(path, 'w') as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def from_json_file(cls, path: str) -> "CLDataTrajectory":
        with open(path, 'r') as f:
            return cls.model_validate_json(f.read())

class MachineConfig(BaseModel):
    """
    Kinematic configuration for Stage 3.
    """
    kinematic_chain: str = Field(..., description="String description of the kinematic chain (e.g. 'Tool -> Z -> Y -> X -> B -> C -> Bed')")
    pivot_offset_x: float = Field(0.0, description="Pivot offset X in mm")
    pivot_offset_y: float = Field(0.0, description="Pivot offset Y in mm")
    pivot_offset_z: float = Field(0.0, description="Pivot offset Z in mm")
    eccentricity_x: float = Field(0.0, description="C-axis eccentricity X in mm")
    eccentricity_y: float = Field(0.0, description="C-axis eccentricity Y in mm")
    b_axis_min: float = Field(..., description="Minimum B-axis angle in degrees")
    b_axis_max: float = Field(..., description="Maximum B-axis angle in degrees")
    c_axis_min: float = Field(..., description="Minimum C-axis angle in degrees")
    c_axis_max: float = Field(..., description="Maximum C-axis angle in degrees")
    b_axis_vector: Tuple[float, float, float] = Field((0.0, 1.0, 0.0), description="B-axis rotation vector")
    c_axis_vector: Tuple[float, float, float] = Field((0.0, 0.0, 1.0), description="C-axis rotation vector")
    max_3axis_tilt_deg: float = Field(25.0, description="Maximum safe surface tilt for a 3-axis active-Z machine before collision risk.")

class MachineStateVector(BaseModel):
    """
    A machine-specific state vector resulting from Inverse Kinematics.
    Stage 3 -> Stage 4 contract.
    """
    x: float = Field(..., description="Machine X position")
    y: float = Field(..., description="Machine Y position")
    z: float = Field(..., description="Machine Z position")
    b: float = Field(..., description="Machine B-axis angle in degrees")
    c: float = Field(..., description="Machine C-axis angle in degrees")
    extrusion_volume: float = Field(0.0, description="Extrusion volume for this move")
    feedrate: float = Field(..., description="Feedrate in mm/min")
    is_travel_move: bool = Field(False, description="True if this is a travel move")
    feature_type: str = Field("outer_wall", description="Type of feature")
    layer_index: int = Field(0, description="Index of the layer this state belongs to (copied unchanged from the CL waypoint)")


class MachineTrajectory(BaseModel):
    """
    A container holding multiple machine state vectors.
    Stage 3 -> Stage 4 contract.
    """
    states: List[MachineStateVector] = Field(..., description="List of machine state vectors")

    def to_json_file(self, path: str) -> None:
        with open(path, 'w') as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def from_json_file(cls, path: str) -> "MachineTrajectory":
        with open(path, 'r') as f:
            return cls.model_validate_json(f.read())
