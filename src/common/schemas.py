from pydantic import BaseModel, Field
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
    """
    wall_count: int = Field(2, description="Number of perimeters")
    bottom_solid_layers: int = Field(3, description="Number of solid bottom layers")
    top_solid_layers: int = Field(3, description="Number of solid top layers")
    infill_density: float = Field(15.0, description="Infill percentage (0-100)")
    infill_pattern: str = Field("grid", description="Infill pattern (e.g., grid, lines, none)")
    nozzle_diameter: float = Field(0.4, description="Nozzle diameter in mm")
    layer_height: float = Field(0.2, description="Layer height in mm")
    continuous_spiral: bool = Field(False, description="True if continuous single-wall spiral/vase mode")

class SpatialContour(BaseModel):
    """
    Represents an ordered list of 3D points and normal vectors for a spatial contour.
    """
    points: List[Tuple[float, float, float]] = Field(..., description="Ordered list of (X, Y, Z) coordinates")
    normals: List[Tuple[float, float, float]] = Field(..., description="Ordered list of (I, J, K) surface normals corresponding to the points")

class UniversalLayer(BaseModel):
    """
    A single layer consisting of multiple spatial contours.
    Stage 1 -> Stage 2 contract.
    """
    layer_index: int = Field(..., description="Index of this layer")
    z_height: float = Field(..., description="Nominal Z height of the layer")
    contours: List[SpatialContour] = Field(..., description="List of spatial contours forming this layer")

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

class CLDataTrajectory(BaseModel):
    """
    A collection of toolpath waypoints.
    Stage 2 -> Stage 3 contract.
    """
    waypoints: List[CLDataWaypoint] = Field(..., description="Ordered list of cutter location waypoints")

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
