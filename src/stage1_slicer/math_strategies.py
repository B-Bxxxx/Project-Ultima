import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import trimesh
from src.stage1_slicer.safe_expr import compile_expression, evaluate_compiled, expression_names

logger = logging.getLogger(__name__)


class BaseSlicingStrategy(ABC):
    def __init__(self, params: Dict[str, Any] = None):
        if params is None: params = {}
        self.transition_height = params.get("transition_height", 0.0)

    # -- layer grid ------------------------------------------------------------------------
    # (self.start_z is z' of layer 0; the parameter 'start_z' is the base, one layer lower.)
    # Layer i is printed at z' = self.start_z + i*h (the TOP of the layer; the first layer therefore
    # sits at one layer height above the bed, never at z = 0) and its contours are cut from the
    # mid-plane of the layer, z' - h/2 (so a slice never coincides with a top/bottom face).
    def _first_layer_z(self, params: Dict[str, Any], bottom_z: float) -> float:
        h = params.get("layer_height", 0.2)
        # the parameter `start_z` is the BASE of the print (bed level of the part); the first
        # layer is printed one layer height above it
        return float(params.get("start_z", bottom_z)) + h

    def _count_layers(self, start_z: float, end_z: float, h: float) -> int:
        return max(1, int(np.ceil((end_z - start_z) / h - 1e-9)) + 1)

    def _mid_offset(self) -> float:
        return 0.5 * getattr(self, "layer_height", 0.2)

    @abstractmethod
    def get_layer_count(self) -> int:
        pass

    @abstractmethod
    def get_slice_plane(self, layer_idx: int) -> Tuple[List[float], List[float]]:
        """Returns (plane_origin, plane_normal) for the deformed slicing plane."""
        pass

    @abstractmethod
    def deform_mesh(self) -> trimesh.Trimesh:
        """Deforms the mesh vertices for slicing."""
        pass

    def get_blend_weight(self, z_val: float) -> float:
        """Calculates vertical blending weight for transition zones (0 to 1)."""
        if self.transition_height <= 0.0:
            return 1.0
        start = getattr(self, "start_z", 0.0)
        # z_val is the nominal deformed plane Z
        w = (z_val - start) / self.transition_height
        return float(np.clip(w, 0.0, 1.0))

    @abstractmethod
    def undeform_point(self, pt: Tuple[float, float, float], layer_idx: int) -> Tuple[float, float, float]:
        """Undeforms a single point extracted from the deformed contour."""
        pass

    @abstractmethod
    def compute_normal(self, undeformed_pt: Tuple[float, float, float], layer_idx: int) -> Tuple[float, float, float]:
        """Computes the analytical or numerical gradient surface normal at the undeformed point."""
        pass

    def _refine_mesh(self, max_edge: float = None):
        import trimesh.remesh
        if max_edge is None:
            max_edge = getattr(self, "layer_height", 0.2) * 2.0  # reasonable default

        vertices, faces = trimesh.remesh.subdivide_to_size(
            self.mesh.vertices,
            self.mesh.faces,
            max_edge=max_edge,
            max_iter=10
        )
        if len(vertices) > 500_000:
            raise ValueError(f"Mesh refinement produced too many vertices: {len(vertices)}")
        self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        self.deformed = self.mesh.copy()
    def _solve_forward_z(self, z, f_val):
        """
        Solves z' = z + w(z') * f for z' (exact inverse of undeform_point: z = z' - w(z') * f), where
        w(z') = clip((z' - start_z) / T, 0, 1) is the blend weight of the transition zone (height T).

        w is piecewise linear, so there is a closed form - no iteration, no convergence limit:
          * z' <= start_z      : z' = z                              (valid when z <= start_z)
          * z' >= start_z + T  : z' = z + f                          (valid when z + f >= start_z + T)
          * inside the zone    : z' = (z - start_z * f / T) / (1 - f / T)
        For f < T the map z -> z' is strictly increasing (this includes ANY negative f, however
        large); for f >= T it folds over (several z' belong to one z) - the upper branch is taken
        and a warning is logged, because the layers then really cross each other.
        """
        z_arr = np.asarray(z, dtype=float)
        f_arr = np.broadcast_to(np.asarray(f_val, dtype=float), z_arr.shape)
        T = float(self.transition_height)
        if T <= 0.0:
            z_prime = z_arr + f_arr
        else:
            if f_arr.size and float(np.max(f_arr)) >= T:
                logger.warning("max(f) = %.3g >= transition_height = %.3g: the deformation folds the layers over; "
                               "increase transition_height above max(f) or reduce the deformation.",
                               float(np.max(f_arr)), T)
            s0 = float(getattr(self, "start_z", 0.0))
            denom = 1.0 - f_arr / T
            safe = np.where(np.abs(denom) > 1e-12, denom, 1.0)
            inside = (z_arr - s0 * f_arr / T) / safe
            z_prime = np.where(z_arr <= s0, z_arr, np.where(z_arr + f_arr >= s0 + T, z_arr + f_arr, inside))
        return z_prime if isinstance(z, np.ndarray) else float(z_prime)

    def get_resolved_params(self) -> Dict[str, Any]:
        """
        Returns a copy of the parameters AFTER defaults were resolved (e.g. the
        mesh-derived start_z / end_z). Stage 1 stores this in the model metadata so
        Stage 2 rebuilds the strategy with exactly the same values.
        """
        return {
            "layer_height": float(getattr(self, "layer_height", 0.2)),
            "transition_height": float(self.transition_height),
            # base of the print (the first layer is printed at start_z + layer_height)
            "start_z": float(getattr(self, "start_z", 0.0)) - float(getattr(self, "layer_height", 0.2)),
            "end_z": float(getattr(self, "end_z", 0.0)),
        }

    def project_to_layer(self, x: float, y: float, layer_idx: int) -> Tuple[float, float, float]:
        """
        Lifts a 2D point (x, y) of layer `layer_idx` to its true 3D position.
        Used by Stage 2 after it re-generated 2D features (walls, infill).
        Default: the layer lives on the deformed-space plane z' = start_z + idx*h,
        so the point is mapped back with undeform_point.
        """
        z_prime = getattr(self, "start_z", 0.0) + layer_idx * getattr(self, "layer_height", 0.2)
        return self.undeform_point((x, y, z_prime), layer_idx)

    def layer_gaps(self, x: float, y: float, layer_idx: int, eps: float = 1e-4) -> Tuple[float, float]:
        """
        (vertical gap, normal gap) between layer `layer_idx` and the next one at (x, y), measured on the
        real layer surfaces (project_to_layer) - the same definition for every strategy.
          vertical = Z_{i+1}(x, y) - Z_i(x, y)                 (what a vertical nozzle sees)
          normal   = vertical * cos(slope of layer i)           (what a nozzle along the normal sees)
        The slope comes from finite differences of the surface itself, so it does not depend on how a
        strategy blends its normals.
        """
        z0 = self.project_to_layer(x, y, layer_idx)[2]
        z1 = self.project_to_layer(x, y, layer_idx + 1)[2]
        dzdx = (self.project_to_layer(x + eps, y, layer_idx)[2] - self.project_to_layer(x - eps, y, layer_idx)[2]) / (2 * eps)
        dzdy = (self.project_to_layer(x, y + eps, layer_idx)[2] - self.project_to_layer(x, y - eps, layer_idx)[2]) / (2 * eps)
        vertical = float(z1 - z0)
        return vertical, float(vertical / np.sqrt(1.0 + dzdx * dzdx + dzdy * dzdy))

    def compute_thickness(self, undeformed_pt: Tuple[float, float, float], layer_idx: int) -> float:
        """Computes local layer thickness (adaptive). Defaults to nominal layer_height."""
        return getattr(self, "layer_height", 0.2)

    def _compute_custom_field_thickness(self, undeformed_pt: Tuple[float, float, float], layer_idx: int, f_val: float) -> float:
        zp = getattr(self, "start_z", 0.0) + layer_idx * getattr(self, "layer_height", 0.2)
        if self.transition_height <= 0.0 or zp > (getattr(self, "start_z", 0.0) + self.transition_height):
            return getattr(self, "layer_height", 0.2)

        dw_dzp = 1.0 / self.transition_height
        local_thickness_ratio = 1.0 - dw_dzp * f_val
        t = getattr(self, "layer_height", 0.2) * local_thickness_ratio
        return float(max(0.01, t))


class PlanarStrategy(BaseSlicingStrategy):
    def __init__(self, mesh: trimesh.Trimesh, params: Dict[str, Any]):
        super().__init__(params)
        self.mesh = mesh.copy()
        self.layer_height = params.get("layer_height", 0.2)
        if len(self.mesh.vertices) > 0:
            self.start_z = self._first_layer_z(params, self.mesh.bounds[0, 2])
            end_z = params.get("end_z", self.mesh.bounds[1, 2])
        else:
            self.start_z = float(params.get("start_z", 0.0)) + self.layer_height
            end_z = params.get("end_z", 10.0)
        self.end_z = end_z
        self.num_layers = self._count_layers(self.start_z, end_z, self.layer_height)

    def get_layer_count(self) -> int:
        return self.num_layers

    def get_slice_plane(self, layer_idx: int):
        z = self.start_z + layer_idx * self.layer_height - self._mid_offset()
        return [0, 0, z], [0, 0, 1]

    def deform_mesh(self) -> trimesh.Trimesh:
        return self.mesh

    def undeform_point(self, pt: Tuple[float, float, float], layer_idx: int):
        return pt

    def compute_normal(self, undeformed_pt: Tuple[float, float, float], layer_idx: int):
        return (0.0, 0.0, 1.0)


class ProgressiveTiltStrategy(BaseSlicingStrategy):
    def __init__(self, mesh: trimesh.Trimesh, params: Dict[str, Any]):
        super().__init__(params)
        self.mesh = mesh.copy()
        self.layer_height = params.get("layer_height", 0.2)
        if len(self.mesh.vertices) > 0:
            self.start_z = self._first_layer_z(params, self.mesh.bounds[0, 2])
            end_z = params.get("end_z", self.mesh.bounds[1, 2])
        else:
            self.start_z = float(params.get("start_z", 0.0)) + self.layer_height
            end_z = params.get("end_z", 10.0)
        self.end_z = end_z
        self.num_layers = self._count_layers(self.start_z, end_z, self.layer_height)
        self.start_tilt = np.radians(params.get("start_tilt_deg", 0.0))
        self.end_tilt = np.radians(params.get("end_tilt_deg", 0.0))

        # Tilt given as an expression of the layer position t = 0..1 (function files, "tilt_x_deg").
        # Without it the linear start_tilt_deg -> end_tilt_deg interpolation is used.
        self.tilt_expr = params.get("tilt_expression")
        self.expr_params = {k: float(v) for k, v in dict(params.get("expression_params", {})).items()}
        self._tilt_code = None
        y_min = float(self.mesh.bounds[0, 1]) if len(self.mesh.vertices) > 0 else 0.0
        if self.tilt_expr is not None:
            missing = expression_names(self.tilt_expr) - set(self.expr_params) - {"t"}
            if missing:
                raise ValueError(f"Tilt expression uses undefined parameter(s) {sorted(missing)}")
            self._tilt_code = compile_expression(self.tilt_expr, set(self.expr_params) | {"t"})

        # Pivot line (prevents negative Z folding): numeric `pivot_y`, or an expression of y_min,
        # default the lowest Y edge of the part.
        if params.get("pivot_y") is not None:
            self.pivot_y = float(params["pivot_y"])
        else:
            pexpr = params.get("pivot_expression", "y_min")
            pcode = compile_expression(pexpr, set(self.expr_params) | {"y_min"})
            self.pivot_y = float(evaluate_compiled(pcode, 0.0, 0.0, 0.0, {**self.expr_params, "y_min": y_min}))

    def get_layer_count(self) -> int:
        return self.num_layers

    def get_resolved_params(self) -> Dict[str, Any]:
        resolved = super().get_resolved_params()
        resolved["pivot_y"] = float(self.pivot_y)
        if self._tilt_code is not None:
            resolved["tilt_expression"] = self.tilt_expr
            resolved["expression_params"] = dict(self.expr_params)
        else:
            resolved["start_tilt_deg"] = float(np.degrees(self.start_tilt))
            resolved["end_tilt_deg"] = float(np.degrees(self.end_tilt))
        return resolved

    def project_to_layer(self, x: float, y: float, layer_idx: int) -> Tuple[float, float, float]:
        """
        The layer lies on the tilted plane through [0, pivot_y, z0] with normal
        [0, -sin(t), cos(t)]  =>  z = z0 + (y - pivot_y) * tan(t).
        (undeform_point stays a pass-through: Stage 1 feeds it points that are
        already real 3D points of the tilted plane.)
        """
        tilt = self._get_tilt_for_layer(layer_idx)
        z0 = self.start_z + layer_idx * self.layer_height
        return (float(x), float(y), float(z0 + (y - self.pivot_y) * np.tan(tilt)))

    def _get_tilt_for_layer(self, layer_idx: int) -> float:
        t = layer_idx / max(1, (self.num_layers - 1))
        if self._tilt_code is not None:
            deg = float(evaluate_compiled(self._tilt_code, 0.0, 0.0, 0.0, {**self.expr_params, "t": t}))
            return float(np.radians(deg))
        return self.start_tilt + t * (self.end_tilt - self.start_tilt)

    def get_slice_plane(self, layer_idx: int):
        tilt = self._get_tilt_for_layer(layer_idx)
        z = self.start_z + layer_idx * self.layer_height

        # Tilt around X-axis. If we pivot at pivot_y, the plane origin must be shifted
        # so that at y = pivot_y, the height is exactly z.
        # Plane eq: ny * (y - oy) + nz * (z_p - oz) = 0
        # If origin is [0, pivot_y, z], it satisfies this cleanly.
        plane_normal = [0, -np.sin(tilt), np.cos(tilt)]
        # cut from the mid-plane of the layer: move the origin half a layer down along the normal
        plane_origin = [0, self.pivot_y - self._mid_offset() * plane_normal[1],
                        z - self._mid_offset() * plane_normal[2]]
        return plane_origin, plane_normal

    def compute_thickness(self, undeformed_pt: Tuple[float, float, float], layer_idx: int) -> float:
        # Distance from pivot determines local thickness spread
        # thickness = layer_height / cos(tilt) + dy * sin(dtilt)
        # For simplicity, if tilting, outer radii are thicker.
        # dtilt = tilt(layer) - tilt(layer-1)
        if layer_idx == 0:
            dtilt = self._get_tilt_for_layer(0)
        else:
            dtilt = self._get_tilt_for_layer(layer_idx) - self._get_tilt_for_layer(layer_idx - 1)

        tilt = self._get_tilt_for_layer(layer_idx)

        x, y, z = undeformed_pt
        dy = y - self.pivot_y

        # Geometric local thickness approximation
        # Base thickness along normal = layer_height * cos(tilt)
        # Plus arc length due to tilt differential
        t = (self.layer_height * np.cos(tilt)) + (dy * np.sin(dtilt))
        return float(max(0.01, t))

    def deform_mesh(self) -> trimesh.Trimesh:
        return self.mesh

    def undeform_point(self, pt: Tuple[float, float, float], layer_idx: int):
        x, y, zp = pt
        w = self.get_blend_weight(zp)
        # Actually tilt strategy just shifts points if pivoting, but we keep pt mapping simple:
        # Since undeform_point is pass-through for progressive tilt in this simple geometric model,
        # blend weight doesn't affect coordinates, just normals
        return pt

    def compute_normal(self, undeformed_pt: Tuple[float, float, float], layer_idx: int):
        tilt = self._get_tilt_for_layer(layer_idx)
        x, y, z = undeformed_pt
        w = self.get_blend_weight(self.start_z + layer_idx * self.layer_height)

        n_x, n_y, n_z = 0.0, float(-np.sin(tilt)), float(np.cos(tilt))

        # Blend with flat [0, 0, 1]
        n_x = w * n_x
        n_y = w * n_y
        n_z = w * n_z + (1.0 - w) * 1.0

        norm = np.linalg.norm([n_x, n_y, n_z])
        return (n_x/norm, n_y/norm, n_z/norm)


class ConicalStrategy(BaseSlicingStrategy):
    def __init__(self, mesh: trimesh.Trimesh, params: Dict[str, Any]):
        super().__init__(params)
        self.mesh = mesh.copy()
        self.layer_height = params.get("layer_height", 0.2)
        self.cone_angle = np.radians(params.get("cone_angle_deg", 15.0))
        self.tan_alpha = np.tan(self.cone_angle)

        # Precompute deformed mesh
        if params.get("refine_mesh", True) and len(self.mesh.vertices) > 0:
            wavelength = params.get("feature_scale", self.mesh.extents.max() if self.mesh.extents is not None else 10.0)
            self._refine_mesh(max_edge=wavelength * params.get("refine_fraction", 1.0/8.0))
        else:
            self.deformed = self.mesh.copy()
        self.start_z = self._first_layer_z(params, self.deformed.bounds[0, 2] if len(self.deformed.vertices) > 0 else 0.0)
        if len(self.deformed.vertices) > 0:
            r = np.linalg.norm(self.deformed.vertices[:, :2], axis=1)
            f_val = r * self.tan_alpha
            self.deformed.vertices[:, 2] = self._solve_forward_z(self.deformed.vertices[:, 2], f_val)
            end_z = params.get("end_z", self.deformed.bounds[1, 2])
        else:
            self.start_z = float(params.get("start_z", 0.0)) + self.layer_height
            end_z = params.get("end_z", 10.0)
        self.end_z = end_z
        self.num_layers = self._count_layers(self.start_z, end_z, self.layer_height)

    def get_resolved_params(self) -> Dict[str, Any]:
        resolved = super().get_resolved_params()
        resolved["cone_angle_deg"] = float(np.degrees(self.cone_angle))
        return resolved

    def get_layer_count(self) -> int:
        return self.num_layers

    def get_slice_plane(self, layer_idx: int):
        z_prime = self.start_z + layer_idx * self.layer_height - self._mid_offset()
        return [0, 0, z_prime], [0, 0, 1]

    def deform_mesh(self) -> trimesh.Trimesh:
        return self.deformed

    def undeform_point(self, pt: Tuple[float, float, float], layer_idx: int):
        x, y, zp = pt
        rad = np.sqrt(x**2 + y**2)

        w = self.get_blend_weight(zp)
        # Undoing the deformation: If w=1, subtract r*tan_alpha
        # If w=0 (flat), we don't subtract anything? Wait.
        # If the mesh was fully deformed, and we slice at Z, but we want flat layer 0:
        # A fully deformed mesh sliced at flat Z produces a shifted contour.
        # If we use w, we pull it back conditionally.
        real_z = zp - w * rad * self.tan_alpha
        return (float(x), float(y), float(real_z))

    def compute_normal(self, undeformed_pt: Tuple[float, float, float], layer_idx: int):
        x, y, z = undeformed_pt
        zp = self.start_z + layer_idx * self.layer_height
        w = self.get_blend_weight(zp)

        rad = np.sqrt(x**2 + y**2)
        if rad > 1e-6:
            df_dx = w * (x / rad) * self.tan_alpha
            df_dy = w * (y / rad) * self.tan_alpha
        else:
            df_dx = 0.0
            df_dy = 0.0

        norm_vec = np.array([df_dx, df_dy, 1.0])
        norm_vec /= np.linalg.norm(norm_vec)
        return (float(norm_vec[0]), float(norm_vec[1]), float(norm_vec[2]))

    def compute_thickness(self, undeformed_pt: Tuple[float, float, float], layer_idx: int) -> float:
        x, y, z = undeformed_pt
        zp = self.start_z + layer_idx * self.layer_height

        if self.transition_height <= 0.0 or zp > (self.start_z + self.transition_height):
            return self.layer_height

        # The true deformation is z' = z + w(z') * f(x, y)
        # Therefore z = z' - w(z') * f(x, y)
        # We want the local vertical thickness mapping: dz / dz'
        # dz/dz' = 1 - dw/dz' * f(x, y)
        # where w(z') = (z' - start_z) / transition_height  =>  dw/dz' = 1 / transition_height

        rad = np.sqrt(x**2 + y**2)
        f_val = rad * self.tan_alpha

        dw_dzp = 1.0 / self.transition_height
        local_thickness_ratio = 1.0 - dw_dzp * f_val

        t = self.layer_height * local_thickness_ratio
        return float(max(0.01, t))



class CustomExpressionStrategy(BaseSlicingStrategy):
    def __init__(self, mesh: trimesh.Trimesh, params: Dict[str, Any]):
        super().__init__(params)
        self.mesh = mesh.copy()
        self.layer_height = params.get("layer_height", 0.2)
        self._setup_field(params)

        # Precompute deformed mesh (refined first, a coarse mesh would deform only at its corners)
        if params.get("refine_mesh", True) and len(self.mesh.vertices) > 0:
            wavelength = params.get("feature_scale", self.mesh.extents.max() if self.mesh.extents is not None else 10.0)
            self._refine_mesh(max_edge=wavelength * params.get("refine_fraction", 1.0 / 8.0))
        else:
            self.deformed = self.mesh.copy()
        self.start_z = self._first_layer_z(params, self.deformed.bounds[0, 2] if len(self.deformed.vertices) > 0 else 0.0)
        if len(self.deformed.vertices) > 0:
            x = self.deformed.vertices[:, 0]
            y = self.deformed.vertices[:, 1]
            f_val = self._evaluate(x, y)
            self.deformed.vertices[:, 2] = self._solve_forward_z(self.deformed.vertices[:, 2], f_val)
            end_z = params.get("end_z", self.deformed.bounds[1, 2])
        else:
            self.start_z = float(params.get("start_z", 0.0)) + self.layer_height
            end_z = params.get("end_z", 10.0)
        self.end_z = end_z
        self.num_layers = self._count_layers(self.start_z, end_z, self.layer_height)

    def _setup_field(self, params: Dict[str, Any]) -> None:
        self.expr = params.get("expression", "0.0")
        # named numeric parameters of the expression (function files); {} for a plain expression
        self.expr_params = {k: float(v) for k, v in dict(params.get("expression_params", {})).items()}
        missing = expression_names(self.expr) - set(self.expr_params)
        if missing:
            raise ValueError(f"Expression uses undefined parameter(s) {sorted(missing)}")
        # Security (AGENTS.md "No Blind Eval"): the expression is validated against an
        # AST whitelist and compiled once; invalid/unsafe input raises ValueError here.
        self._code = compile_expression(self.expr, self.expr_params.keys())

    def get_resolved_params(self) -> Dict[str, Any]:
        resolved = super().get_resolved_params()
        resolved["expression"] = self.expr
        resolved["expression_params"] = dict(self.expr_params)
        return resolved

    def _evaluate(self, x, y):
        r = np.sqrt(x**2 + y**2)
        # Vectorized evaluation of the validated, pre-compiled expression
        return evaluate_compiled(self._code, x, y, r, self.expr_params)

    def get_layer_count(self) -> int:
        return self.num_layers

    def get_slice_plane(self, layer_idx: int):
        z_prime = self.start_z + layer_idx * self.layer_height - self._mid_offset()
        return [0, 0, z_prime], [0, 0, 1]

    def deform_mesh(self) -> trimesh.Trimesh:
        return self.deformed

    def undeform_point(self, pt: Tuple[float, float, float], layer_idx: int):
        x, y, zp = pt
        w = self.get_blend_weight(zp)
        f_val = self._evaluate(x, y)
        real_z = zp - w * f_val
        return (float(x), float(y), float(real_z))

    def compute_normal(self, undeformed_pt: Tuple[float, float, float], layer_idx: int):
        x, y, z = undeformed_pt
        zp = self.start_z + layer_idx * self.layer_height
        w = self.get_blend_weight(zp)

        eps = 1e-5
        f0 = self._evaluate(x, y)
        fx = self._evaluate(x + eps, y)
        fy = self._evaluate(x, y + eps)

        df_dx = w * (fx - f0) / eps
        df_dy = w * (fy - f0) / eps

        norm_vec = np.array([df_dx, df_dy, 1.0])
        norm_vec /= np.linalg.norm(norm_vec)
        return (float(norm_vec[0]), float(norm_vec[1]), float(norm_vec[2]))

    def compute_thickness(self, undeformed_pt: Tuple[float, float, float], layer_idx: int) -> float:
        x, y, z = undeformed_pt
        f_val = self._evaluate(x, y)
        return self._compute_custom_field_thickness(undeformed_pt, layer_idx, f_val)


class GridFieldStrategy(CustomExpressionStrategy):
    """
    f(x, y) given as a regular grid (imported from a data file, see function_import.py) with
    bilinear interpolation and clamping outside the grid. Parameters: grid_x, grid_y (ascending
    1D lists) and grid_values (len(grid_y) rows of len(grid_x) values) - all plain lists, so
    they travel through the Stage 1 metadata to Stage 2.
    """

    def _setup_field(self, params: Dict[str, Any]) -> None:
        from src.stage1_slicer.function_import import GridField
        self.field = GridField.from_params(params)
        self.expr = "grid_field"
        self.expr_params = {}

    def _evaluate(self, x, y):
        return self.field.evaluate(x, y)


class ExternalScalarFieldStrategy(BaseSlicingStrategy):
    def __init__(self, mesh: trimesh.Trimesh, params: Dict[str, Any]):
        super().__init__(params)
        self.mesh = mesh.copy()
        n_vert = len(self.mesh.vertices)
        # Accepts a numpy array or a plain list. In Stage 2 the mesh is empty and the
        # field is rebuilt afterwards from the layer contours (see Stage 2).
        self.vertex_deformations = np.asarray(
            params.get("vertex_deformations", np.zeros(n_vert)), dtype=float)
        self.layer_height = params.get("layer_height", 0.2)

        # Precompute deformed mesh
        self.deformed = self.mesh.copy()
        if n_vert > 0:
            self.start_z = self._first_layer_z(params, self.deformed.bounds[0, 2])
            self.deformed.vertices[:, 2] = self._solve_forward_z(
                self.deformed.vertices[:, 2], self.vertex_deformations)
            end_z = params.get("end_z", self.deformed.bounds[1, 2])
        else:
            self.start_z = float(params.get("start_z", 0.0)) + self.layer_height
            end_z = params.get("end_z", 10.0)
        self.end_z = end_z
        self.num_layers = self._count_layers(self.start_z, end_z, self.layer_height)

        # Setup for pure Numpy IDW (Inverse Distance Weighting) interpolation
        self.pts_2d = self.mesh.vertices[:, :2]

    def _interpolate_idw(self, x, y):
        # K-nearest or simple IDW using numpy
        pt = np.array([x, y])
        dists = np.linalg.norm(self.pts_2d - pt, axis=1)

        # If exact match
        min_idx = np.argmin(dists)
        if dists[min_idx] < 1e-6:
            return self.vertex_deformations[min_idx]

        # Use K=5 nearest neighbors
        k = min(5, len(dists))
        k_indices = np.argpartition(dists, k)[:k]

        weights = 1.0 / (dists[k_indices] ** 2)
        f_val = np.sum(weights * self.vertex_deformations[k_indices]) / np.sum(weights)
        return float(f_val)

    def get_layer_count(self) -> int:
        return self.num_layers

    def get_slice_plane(self, layer_idx: int):
        z_prime = self.start_z + layer_idx * self.layer_height - self._mid_offset()
        return [0, 0, z_prime], [0, 0, 1]

    def deform_mesh(self) -> trimesh.Trimesh:
        return self.deformed

    def undeform_point(self, pt: Tuple[float, float, float], layer_idx: int):
        x, y, zp = pt
        w = self.get_blend_weight(zp)
        f_val = self._interpolate_idw(x, y)
        real_z = zp - w * f_val
        return (float(x), float(y), float(real_z))

    def compute_normal(self, undeformed_pt: Tuple[float, float, float], layer_idx: int):
        x, y, z = undeformed_pt
        zp = self.start_z + layer_idx * self.layer_height
        w = self.get_blend_weight(zp)

        eps = 1e-4

        f0 = self._interpolate_idw(x, y)
        fx = self._interpolate_idw(x + eps, y)
        fy = self._interpolate_idw(x, y + eps)

        df_dx = w * (fx - f0) / eps
        df_dy = w * (fy - f0) / eps

        norm_vec = np.array([df_dx, df_dy, 1.0])
        norm_vec /= np.linalg.norm(norm_vec)
        return (float(norm_vec[0]), float(norm_vec[1]), float(norm_vec[2]))

    def compute_thickness(self, undeformed_pt: Tuple[float, float, float], layer_idx: int) -> float:
        x, y, z = undeformed_pt
        f_val = self._interpolate_idw(x, y)
        return self._compute_custom_field_thickness(undeformed_pt, layer_idx, f_val)


def build_strategy(name: str, mesh: trimesh.Trimesh, params: Dict[str, Any]) -> BaseSlicingStrategy:
    """Single factory used by Stage 1 (real mesh) and Stage 2 (empty mesh)."""
    if name == "planar":
        return PlanarStrategy(mesh, params)
    if name == "progressive_tilt":
        return ProgressiveTiltStrategy(mesh, params)
    if name in ("conical", "conical_or_curved"):
        return ConicalStrategy(mesh, params)
    if name == "custom_expr":
        return CustomExpressionStrategy(mesh, params)
    if name == "grid_field":
        return GridFieldStrategy(mesh, params)
    if name == "external_field":
        return ExternalScalarFieldStrategy(mesh, params)
    raise ValueError(f"Unknown strategy: {name}")
