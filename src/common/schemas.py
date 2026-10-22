from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
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
    brim_width: float = Field(0.0, ge=0.0, description="Width in mm of the brim: concentric loops around the outside of the first layer for bed adhesion (0 = no brim). Rounded down to whole nozzle widths.")
    skirt_loops: int = Field(0, ge=0, description="Number of skirt loops: free-standing loops around the first layer (and its brim), `skirt_distance` away from it, to prime the nozzle (0 = no skirt).")
    skirt_distance: float = Field(3.0, ge=0.0, description="Gap in mm between the part (or its brim) and the innermost skirt loop.")
    infill_density: float = Field(0.2, ge=0.0, le=1.0, description="Sparse infill density as a FRACTION 0..1 (0.2 = 20 %)")
    infill_overlap: float = Field(0.15, ge=0.0, le=0.5, description="Infill (and skin) lines reach this fraction of the nozzle diameter into the innermost wall / into each other, for a good bond. 0 = lines only touch.")
    infill_pattern: str = Field("lines", description="Infill pattern. Only 'lines' (parallel zigzag) is implemented; other values fall back to it with a warning.")
    infill_angle_deg: float = Field(45.0, description="Angle of infill in degrees")
    nozzle_diameter: float = Field(0.4, gt=0.0, description="Nozzle diameter in mm (also the line width)")
    layer_height: float = Field(0.2, gt=0.0, description="Layer height in mm")
    continuous_spiral: bool = Field(False, description="True if continuous single-wall spiral/vase mode")
    filament_diameter: float = Field(1.75, gt=0.0, description="Filament diameter in mm")
    max_segment_length: float = Field(1.0, gt=0.0, description="Maximum length of a segment for surface conformity")
    path_tolerance: float = Field(0.02, ge=0.0, description="Walls and solid skin: a path point is dropped when the path moves by less than this (mm). 0 switches the simplification off.")
    infill_path_tolerance: float = Field(0.1, ge=0.0, description="The same for sparse infill (inside the part, so it may be coarser).")
    min_segment_length: float = Field(0.2, ge=0.0, description="Points next to segments shorter than this (mm) are merged while the path stays within 3x the tolerance (old 8/32-bit boards cannot plan very short segments). 0 = off.")
    path_angle_tolerance_deg: float = Field(1.0, ge=0.0, description="A point is also kept when the tool axis (surface normal) would turn by more than this (deg) against the line that replaces it. 0 = ignore the normals.")
    seam_angle_deg: Optional[float] = Field(90.0, description="Direction of the wall seam, seen from the part: every wall ring starts at its vertex farthest in this direction (0 = +X, 90 = +Y = back, counter-clockwise), so the seam is one line. None = keep the start point of the Stage 1 ring.")
    include_rotary_axes: bool = Field(True, description="True to output 5-axis G-code (with B and C). False for strictly 3-axis G-code.")

    # --- speeds (mm/min) ---
    print_speed: float = Field(1500.0, gt=0.0, description="Feedrate of extruding moves in mm/min")
    travel_speed: float = Field(3000.0, gt=0.0, description="Feedrate of travel moves in mm/min")
    z_speed: float = Field(1200.0, gt=0.0, description="Feedrate of the Z-hop moves in mm/min")
    retract_speed: float = Field(2400.0, gt=0.0, description="Feedrate of the retraction (before a travel) in mm/min")
    unretract_speed: Optional[float] = Field(None, gt=0.0, description="Feedrate of the un-retraction / priming (right before printing resumes) in mm/min; defaults to retract_speed")

    # --- travel behaviour ---
    retract_distance: float = Field(0.8, ge=0.0, description="Retraction distance in mm before a travel (0 disables retraction)")
    unretract_distance: Optional[float] = Field(None, ge=0.0, description="Filament pushed back in mm right before printing resumes (0 = no prime move); defaults to retract_distance")
    z_hop: float = Field(0.4, ge=0.0, description="Z-hop height in mm during travel moves (0 disables the hop)")
    min_retract_travel: float = Field(1.5, ge=0.0, description="Travels shorter than this (mm) are made directly, without retraction and Z-hop")
    park_lift: float = Field(30.0, ge=0.0, description="At the end of the print the nozzle travels this far (mm) above the highest point of the part")
    thickness_model: Literal["stage1", "nozzle_axis"] = Field(
        "stage1", description="'stage1': extrusion from the Stage 1/2 layer thickness (normal to the layer). "
                              "'nozzle_axis': thickness measured along the ACTUAL nozzle axis, t / cos(angle between nozzle axis and layer normal) "
                              "- use it for 5-axis machines (and vertical-nozzle tilted layers).")
    hop_short_travels: bool = Field(False, description="Z-hop also on travels shorter than min_retract_travel (they are still made without retraction).")
    travel_command: Literal["G0", "G1"] = Field("G0", description="G-code command used for non-extruding moves (G0 rapid or G1 linear)")

    @model_validator(mode="after")
    def _default_unretract(self):
        if self.unretract_distance is None:
            self.unretract_distance = self.retract_distance
        if self.unretract_speed is None:
            self.unretract_speed = self.retract_speed
        return self

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
    vertical_gaps: Optional[List[float]] = Field(None, description="Stage 1: vertical (Z) distance from this layer's surface to the next layer at each point - the gap a vertical (3-axis) nozzle works in")
    normal_gaps: Optional[List[float]] = Field(None, description="Stage 1: distance between this and the next layer measured along the layer's surface normal at each point - the gap a nozzle aligned with the normal (5-axis) works in")
    feature_type: str = Field("boundary", description="Type of feature: boundary, skirt, brim, outer_wall, inner_wall, bottom_solid, top_solid or infill (sparse)")

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
    c_axis_x: float = Field(0.0, description="Machine X of the C axis (table centre) at B = 0; for a 3-axis / swivel-head machine the machine X of the part origin (0, 0)")
    c_axis_y: float = Field(0.0, description="Machine Y of the C axis (table centre) at B = 0 / of the part origin")
    c_axis_z: float = Field(0.0, description="Machine Z of the table surface at the C axis at B = 0 / of the part origin")
    eccentricity_x: float = Field(0.0, description="C-axis eccentricity X in mm")
    eccentricity_y: float = Field(0.0, description="C-axis eccentricity Y in mm")
    b_axis_dir: float = Field(1.0, description="+1: the machine's positive B turns like the program's (seen from the front, clockwise); -1: opposite. Applied to the B value written to the G-code")
    c_axis_dir: float = Field(1.0, description="+1: the machine's positive C turns like the program's (seen from above, counter-clockwise); -1: opposite")
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
    tool_alignment: float = Field(1.0, description="cos of the angle between the ACTUAL nozzle axis (after axis limits / for a 3-axis machine: vertical) and the layer surface normal, in the part frame. 1.0 = nozzle along the normal.")


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
