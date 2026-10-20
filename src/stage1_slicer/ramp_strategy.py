"""
Ramp transition mode ("transition_mode": "ramp") for expression / data-file functions.

Standard mode blends the function in over `transition_height` (w = 0 -> 1). That is impossible when the
function has to be reached within a few millimetres: the layer thickness h * (1 - w' f) leaves the printable
band. The ramp mode does not blend at all:

  * the part is printed PLANAR from the bed up to the function surface, with partial layers of
    `ramp_layer_height` (ramp layers 0 .. K-1): ramp layer k exists only where the function surface of
    the first full layer lies at least `ramp_min_gap` above its top;
  * from layer K on every layer follows the function exactly (w = 1, constant thickness h). The lowest
    point of the first full layer sits one layer height above the bed.

The first full layer rests on the stair-stepped top of the ramp layers, so its thickness varies between
`ramp_min_gap` and `ramp_min_gap + ramp_layer_height` (closed form, `first_gap`). Layer indices are global:
0 .. K-1 ramp layers, K .. K+n-1 full layers. The standard mode is untouched.

By default (`ramp_contact_only`) the ramp is built only under the BED-CONTACT surface of the part (its
cross-section just above the bed). Where the function surface runs over a part of the model that does not
touch the bed (an overhang), there is no ramp: the first full layer follows the function directly (nominal
thickness, nothing below it). The contact footprint is stored in the metadata (`ramp_footprint`, WKT).

Material outside the footprint that lies BELOW the first full layer (e.g. a slab overhanging the foot) would
be lost, so the function layers also exist before the first full layer: the PRE-LAYERS (layer indices
K .. K+P-1, constant thickness, full w = 1). A pre-layer is a partial function layer: it exists only outside
the footprint and only where its surface is at least one layer height above the bed, so it grows outward from
the foot, each layer resting on the previous one. Global indices: 0 .. K-1 ramp, K .. K+P-1 pre-layers,
K+P first full layer, then the rest. `first_z` is the surface level of the first full layer; the base
class's `start_z` is lowered by P layers so the common maths stays unchanged.

The first full layer touches the bed (one layer height above it) at the FARTHEST point of the footprint, not
of the whole mesh: with `ramp_contact_only` the ramp is the small wedge under that layer. Every function layer
whose surface still dips below the bed somewhere outside the footprint (z' < `clear_z`) is clipped to the
region where it is at least one layer height above the bed.
"""
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import shapely
import shapely.wkt
import trimesh
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from src.stage1_slicer.level_region import region_polygon
from src.stage1_slicer.math_strategies import CustomExpressionStrategy, GridFieldStrategy

logger = logging.getLogger(__name__)

GRID_CELL = 0.25            # mm: resolution of the ramp region grid
MAX_GRID_CELLS = 4_000_000
FOOTPRINT_EPS = 0.01        # mm above the bed: the part's cross-section there is its bed-contact surface


def contact_footprint(mesh: trimesh.Trimesh, base_z: float) -> Optional[BaseGeometry]:
    """
    Cross-section of the mesh just above the bed as a shapely (multi)polygon (holes by even-odd); None if it
    cannot be found. The reason is logged (terminal), because without the footprint the ramp is built under the
    whole part.
    """
    try:
        sec = mesh.section(plane_origin=[0, 0, base_z + FOOTPRINT_EPS], plane_normal=[0, 0, 1])
        if sec is None:
            logger.warning("Ramp: the part has no cross-section just above the bed (it touches the bed in a point or "
                           "a line only); the ramp is built under the WHOLE part.")
            return None
        geom = Polygon()
        for d in sec.discrete:
            if len(d) < 4:
                continue
            ring = Polygon(np.asarray(d)[:, :2]).buffer(0)
            geom = geom.symmetric_difference(ring)
        geom = geom.buffer(0)
    except Exception as exc:
        logger.warning("Ramp: the bed-contact surface could not be computed (%s: %s); the ramp is built under the "
                       "WHOLE part.", type(exc).__name__, exc)
        return None
    if geom.is_empty:
        logger.warning("Ramp: the bed-contact surface is empty; the ramp is built under the WHOLE part.")
        return None
    return geom


class RampMixin:
    def __init__(self, mesh: trimesh.Trimesh, params: Dict[str, Any]):
        p = dict(params)
        p["transition_height"] = 0.0                                    # w = 1: the full layers follow f exactly
        self.ramp_layer_height = float(p.get("ramp_layer_height", p.get("layer_height", 0.2)))
        self.ramp_min_gap = float(p.get("ramp_min_gap", 0.1))
        if self.ramp_layer_height <= 0.0:
            raise ValueError("ramp_layer_height must be > 0")
        if self.ramp_min_gap < 0.0:
            raise ValueError("ramp_min_gap must be >= 0")
        self.ramp_contact_only = bool(p.get("ramp_contact_only", True))
        self._footprint: Optional[BaseGeometry] = None
        self.pre_layers = 0
        self.clear_z = 0.0
        self._source = mesh.copy()
        self._regions: Dict[int, Any] = {}
        self._grid = None
        super().__init__(mesh, p)
        h = self.layer_height
        if p.get("ramp_layers") is not None:                            # Stage 2 rebuild: resolved values
            self.ramp_layers = int(p["ramp_layers"])
            self.ramp_base_z = float(p["ramp_base_z"])
            if p.get("ramp_footprint"):
                self._footprint = shapely.wkt.loads(p["ramp_footprint"])
            self.pre_layers = int(p.get("ramp_pre_layers", 0))
            self.first_z = float(p.get("ramp_first_z", self.start_z + self.pre_layers * h))
            self.start_z = self.first_z - self.pre_layers * h
        elif len(self.mesh.vertices) > 0:
            self.ramp_base_z = float(self._source.bounds[0, 2])
            f = self._f(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1])
            # lowest point of the first full layer = one layer above the bed - over the CONTACT SURFACE only
            # (the whole mesh when the ramp is built everywhere): the first function layer touches the bed at the
            # farthest point of the footprint; everything else lies outside it and is built by the function.
            f_min, f_max = float(f.min()), float(f.max())
            self.clear_z = self.ramp_base_z + h + f_max                  # from this z' on every layer is above the bed everywhere
            if self.ramp_contact_only:
                self._footprint = contact_footprint(self._source, self.ramp_base_z)
                if self._footprint is not None:
                    f_foot = self._f_over(self._footprint)
                    f_min, f_max = float(f_foot.min()), float(f_foot.max())
            self.start_z = self.ramp_base_z + h + f_max
            top = self.start_z - f_min
            self.ramp_layers = max(0, int(np.floor((top - self.ramp_min_gap - self.ramp_base_z) / self.ramp_layer_height + 1e-9)))
            self.first_z = self.start_z
            self.pre_layers = self._count_pre_layers()
            self.start_z = self.first_z - self.pre_layers * h
        else:
            self.ramp_layers, self.ramp_base_z = 0, 0.0
            self.first_z = getattr(self, "start_z", 0.0)
        self.num_full_layers = self._count_layers(self.start_z, self.end_z, h)
        self.num_layers = self.ramp_layers + self.num_full_layers

    # -- helpers ------------------------------------------------------------------------------
    def _f(self, x, y) -> np.ndarray:
        x = np.asarray(x, float)
        return np.broadcast_to(self._evaluate(x, y), x.shape).astype(float)

    def _f_over(self, geom: BaseGeometry) -> np.ndarray:
        """f sampled over a polygon: its boundary vertices and an interior grid."""
        b = geom.bounds
        step = max(GRID_CELL, 0.5)
        gx, gy = np.meshgrid(np.arange(b[0], b[2] + step, step), np.arange(b[1], b[3] + step, step))
        gx, gy = gx.ravel(), gy.ravel()
        inside = shapely.contains_xy(geom, gx, gy)
        xs, ys = [gx[inside]], [gy[inside]]
        for poly in getattr(geom, "geoms", [geom]):
            for ring in [poly.exterior, *poly.interiors]:
                c = np.asarray(ring.coords)
                xs.append(c[:, 0]); ys.append(c[:, 1])
        return self._f(np.concatenate(xs), np.concatenate(ys))

    def _count_pre_layers(self) -> int:
        """Function layers needed BEFORE the first full layer: the lowest deformed height z' = z + f of the
        material outside the contact footprint, counted down from the first full layer."""
        if self._footprint is None:
            return 0
        pts = np.vstack([self._source.vertices, trimesh.sample.sample_surface(self._source, 20000, seed=0)[0]])
        out = ~self._in_footprint(pts[:, 0], pts[:, 1])
        if not out.any():
            return 0
        zp_min = float((pts[out, 2] + self._f(pts[out, 0], pts[out, 1])).min())
        return max(0, int(np.ceil((self.first_z - zp_min) / self.layer_height - 1e-9)))

    def _in_footprint(self, x, y) -> np.ndarray:
        x = np.asarray(x, float)
        if self._footprint is None:
            return np.ones(x.shape, bool)
        return shapely.contains_xy(self._footprint.buffer(1e-3), x, np.broadcast_to(np.asarray(y, float), x.shape))

    def _ramp_top(self, k: int) -> float:
        return self.ramp_base_z + (k + 1) * self.ramp_layer_height

    def _is_ramp(self, layer_idx: int) -> bool:
        return layer_idx < self.ramp_layers

    def first_gap(self, x, y) -> np.ndarray:
        """Thickness of the first full layer at (x, y): its surface minus the top of the highest ramp layer there."""
        surface = self.first_z - self._f(x, y)
        count = np.floor((surface - self.ramp_min_gap - self.ramp_base_z) / self.ramp_layer_height + 1e-9)
        count = np.clip(count, 0, self.ramp_layers)
        gap = surface - (self.ramp_base_z + count * self.ramp_layer_height)
        # off the contact surface there is no ramp: the layer is printed at its nominal thickness
        return np.where(self._in_footprint(x, y), gap, self.layer_height)

    def get_resolved_params(self) -> Dict[str, Any]:
        r = super().get_resolved_params()
        r.update({"transition_mode": "ramp", "ramp_layers": int(self.ramp_layers), "ramp_base_z": float(self.ramp_base_z),
                  "ramp_layer_height": float(self.ramp_layer_height), "ramp_min_gap": float(self.ramp_min_gap),
                  "ramp_contact_only": bool(self.ramp_contact_only),
                  "ramp_pre_layers": int(self.pre_layers), "ramp_first_z": float(self.first_z)})
        if self._footprint is not None:
            r["ramp_footprint"] = self._footprint.simplify(0.002).wkt
        return r

    # -- Stage 1 slicing hooks -----------------------------------------------------------------
    def get_slice_plane(self, layer_idx: int):
        if self._is_ramp(layer_idx):
            return [0, 0, self._ramp_top(layer_idx) - 0.5 * self.ramp_layer_height], [0, 0, 1]
        return super().get_slice_plane(layer_idx - self.ramp_layers)

    def slice_source(self, layer_idx: int) -> trimesh.Trimesh:
        return self._source if self._is_ramp(layer_idx) else self.deform_mesh()

    def _region_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._grid is None:
            b = self._source.bounds
            cell = GRID_CELL
            span = (b[1, :2] - b[0, :2]) + 4 * cell
            if span[0] * span[1] / cell ** 2 > MAX_GRID_CELLS:
                cell = float(np.sqrt(span[0] * span[1] / MAX_GRID_CELLS))
            X = np.arange(b[0, 0] - 2 * cell, b[1, 0] + 2 * cell + cell, cell)
            Y = np.arange(b[0, 1] - 2 * cell, b[1, 1] + 2 * cell + cell, cell)
            XX, YY = np.meshgrid(X, Y)
            self._grid = (X, Y, self._f(XX, YY))                        # the function on the grid
        return self._grid

    def _is_pre(self, layer_idx: int) -> bool:
        return self.ramp_layers <= layer_idx < self.ramp_layers + self.pre_layers

    def layer_region(self, layer_idx: int):
        if layer_idx in self._regions:
            return self._regions[layer_idx]
        if self._is_ramp(layer_idx):
            X, Y, F = self._region_grid()
            level = self._ramp_top(layer_idx) + self.ramp_min_gap          # the first full surface must be above this
            region = region_polygon(
                X, Y, self.first_z - F - level, lambda x, y: float(self.first_z - self._f(np.array(x), np.array(y)) - level))
            if self._footprint is not None and region is not None and not region.is_empty:
                region = region.intersection(self._footprint)
        elif layer_idx >= self.ramp_layers:
            # function layers (pre-layers and full layers): partial while their surface still dips below the bed
            # somewhere (outside the footprint, where the function takes over from the ramp)
            zp = self.start_z + (layer_idx - self.ramp_layers) * self.layer_height     # z' of the layer top
            if zp >= self.clear_z - 1e-9 and not self._is_pre(layer_idx):
                return None
            X, Y, F = self._region_grid()
            level = self.ramp_base_z + self.layer_height                               # surface at least one layer above the bed
            region = region_polygon(
                X, Y, zp - F - level, lambda x, y: float(zp - self._f(np.array(x), np.array(y)) - level))
            if self._is_pre(layer_idx) and self._footprint is not None and region is not None and not region.is_empty:
                region = region.difference(self._footprint)                            # under the footprint the ramp is built
        else:
            return None
        self._regions[layer_idx] = region
        return region

    # -- point maps: ramp layers are planar, full layers are the standard maths shifted by K -----------
    def undeform_point(self, pt, layer_idx: int):
        if self._is_ramp(layer_idx):
            return (float(pt[0]), float(pt[1]), float(pt[2]))
        return super().undeform_point(pt, layer_idx - self.ramp_layers)

    def project_to_layer(self, x: float, y: float, layer_idx: int):
        if self._is_ramp(layer_idx):
            return (float(x), float(y), float(self._ramp_top(layer_idx)))
        j = layer_idx - self.ramp_layers
        return super().undeform_point((x, y, self.start_z + j * self.layer_height), j)

    def compute_normal(self, pt, layer_idx: int):
        if self._is_ramp(layer_idx):
            return (0.0, 0.0, 1.0)
        return super().compute_normal(pt, layer_idx - self.ramp_layers)

    def compute_thickness(self, pt, layer_idx: int) -> float:
        if self._is_ramp(layer_idx):
            return float(self.ramp_layer_height)
        if layer_idx == self.ramp_layers + self.pre_layers:
            return float(self.first_gap(pt[0], pt[1]))
        return super().compute_thickness(pt, layer_idx - self.ramp_layers)

    def layer_gaps(self, x: float, y: float, layer_idx: int, eps: float = 1e-4):
        if self._is_ramp(layer_idx):
            return float(self.ramp_layer_height), float(self.ramp_layer_height)
        return super().layer_gaps(x, y, layer_idx, eps)

    # -- batch versions --------------------------------------------------------------------------
    def undeform_points(self, pts, layer_idx: int) -> np.ndarray:
        if self._is_ramp(layer_idx):
            return np.asarray(pts, float).reshape(-1, 3).copy()
        return super().undeform_points(pts, layer_idx - self.ramp_layers)

    def project_points(self, xy, layer_idx: int) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        if self._is_ramp(layer_idx):
            return np.column_stack([xy, np.full(len(xy), self._ramp_top(layer_idx))])
        return super().project_points(xy, layer_idx - self.ramp_layers)

    def normals_at(self, pts, layer_idx: int) -> np.ndarray:
        pts = np.asarray(pts, float).reshape(-1, 3)
        if self._is_ramp(layer_idx):
            return np.tile([0.0, 0.0, 1.0], (len(pts), 1))
        return super().normals_at(pts, layer_idx - self.ramp_layers)

    def thicknesses_at(self, pts, layer_idx: int) -> np.ndarray:
        pts = np.asarray(pts, float).reshape(-1, 3)
        if self._is_ramp(layer_idx):
            return np.full(len(pts), float(self.ramp_layer_height))
        if layer_idx == self.ramp_layers + self.pre_layers:
            return self.first_gap(pts[:, 0], pts[:, 1])
        return super().thicknesses_at(pts, layer_idx - self.ramp_layers)

    def layer_gaps_batch(self, xy, layer_idx: int, eps: float = 1e-4):
        xy = np.asarray(xy, float).reshape(-1, 2)
        if self._is_ramp(layer_idx):
            g = np.full(len(xy), float(self.ramp_layer_height))
            return g, g.copy()
        return super().layer_gaps_batch(xy, layer_idx, eps)


class RampCustomExpressionStrategy(RampMixin, CustomExpressionStrategy):
    pass


class RampGridFieldStrategy(RampMixin, GridFieldStrategy):
    pass
