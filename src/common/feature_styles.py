"""
Qt-free description of how the toolpath line types are shown in the viewer.

Single source of truth for "which colour means which line type": the GUI draws its lines and its
legend from FEATURE_STYLES, and the tests check it against the feature types Stage 2 produces.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

RGBA = Tuple[float, float, float, float]


@dataclass(frozen=True)
class FeatureStyle:
    key: str                  # == waypoint.feature_type ("travel" for non-extruding moves)
    label: str
    description: str
    color: RGBA

    @property
    def hex(self) -> str:
        r, g, b, _ = self.color
        return "#{:02x}{:02x}{:02x}".format(*(int(round(max(0.0, min(1.0, v)) * 255)) for v in (r, g, b)))


FEATURE_STYLES: Tuple[FeatureStyle, ...] = (
    FeatureStyle("outer_wall", "Outer wall", "Outermost perimeter (visible surface)", (0.22, 1.0, 0.08, 1.0)),
    FeatureStyle("inner_wall", "Inner wall", "Inner perimeters", (1.0, 0.9, 0.0, 1.0)),
    FeatureStyle("bottom_solid", "Bottom skin", "Solid layers at the bottom (and under overhangs)", (1.0, 0.48, 0.0, 1.0)),
    FeatureStyle("top_solid", "Top skin", "Solid layers at the top (and over cavities)", (1.0, 0.17, 0.84, 1.0)),
    FeatureStyle("infill", "Sparse infill", "Internal sparse infill", (0.0, 0.81, 1.0, 1.0)),
    FeatureStyle("travel", "Travel", "Non-extruding moves", (1.0, 0.22, 0.38, 0.45)),
)

STYLE_BY_KEY: Dict[str, FeatureStyle] = {s.key: s for s in FEATURE_STYLES}


def group_lines_by_feature(items: Iterable, visible: Sequence[str], layer_limit: int,
                           layer_min: int = 0, move_limit: Optional[int] = None
                           ) -> Dict[str, List[List[float]]]:
    """
    Builds line-segment vertex lists (pairs of points, for GL 'lines' mode) per feature key from
    waypoints / machine states.
      * only layers with layer_min <= layer_index < layer_limit are drawn
      * `move_limit` (exclusive index into `items`) cuts the drawing short - used by the playback
      * features not in `visible` are skipped; a segment is classified by its END point
    No segment connects a point outside the layer range to one inside it.
    """
    out: Dict[str, List[List[float]]] = {s.key: [] for s in FEATURE_STYLES}
    last = None
    for i, wp in enumerate(items):
        if move_limit is not None and i >= move_limit:
            break
        if wp.layer_index >= layer_limit or wp.layer_index < layer_min:
            last = None
            continue
        if last is not None:
            key = "travel" if wp.is_travel_move else wp.feature_type
            if key in visible and key in out:
                out[key].extend([[last.x, last.y, last.z], [wp.x, wp.y, wp.z]])
        last = wp
    return out
