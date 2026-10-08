import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import trimesh

class BaseSlicingStrategy(ABC):
    def __init__(self, params: Dict[str, Any] = None):
        if params is None: params = {}
        self.transition_height = params.get("transition_height", 0.0)

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
            self.start_z = params.get("start_z", self.mesh.bounds[0, 2])
            end_z = params.get("end_z", self.mesh.bounds[1, 2])
        else:
            self.start_z = params.get("start_z", 0.0)
            end_z = params.get("end_z", 10.0)
        self.num_layers = max(1, int(np.ceil((end_z - self.start_z) / self.layer_height)))

    def get_layer_count(self) -> int:
        return self.num_layers

    def get_slice_plane(self, layer_idx: int):
        z = self.start_z + layer_idx * self.layer_height
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
            self.start_z = params.get("start_z", self.mesh.bounds[0, 2])
            end_z = params.get("end_z", self.mesh.bounds[1, 2])
            self.pivot_y = params.get("pivot_y", self.mesh.bounds[0, 1])
        else:
            self.start_z = params.get("start_z", 0.0)
            end_z = params.get("end_z", 10.0)
            self.pivot_y = params.get("pivot_y", 0.0)
        self.num_layers = max(1, int(np.ceil((end_z - self.start_z) / self.layer_height)))
        self.start_tilt = np.radians(params.get("start_tilt_deg", 0.0))
        self.end_tilt = np.radians(params.get("end_tilt_deg", 0.0))

    def get_layer_count(self) -> int:
        return self.num_layers

    def _get_tilt_for_layer(self, layer_idx: int) -> float:
        t = layer_idx / max(1, (self.num_layers - 1))
        return self.start_tilt + t * (self.end_tilt - self.start_tilt)

    def get_slice_plane(self, layer_idx: int):
        tilt = self._get_tilt_for_layer(layer_idx)
        z = self.start_z + layer_idx * self.layer_height

        # Tilt around X-axis. If we pivot at pivot_y, the plane origin must be shifted
        # so that at y = pivot_y, the height is exactly z.
        # Plane eq: ny * (y - oy) + nz * (z_p - oz) = 0
        # If origin is [0, pivot_y, z], it satisfies this cleanly.
        plane_origin = [0, self.pivot_y, z]
        plane_normal = [0, -np.sin(tilt), np.cos(tilt)]
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
        tilt = self._get_tilt_for_layer(layer_idx)
        # Progressive tilt deforms by pivoting around pivot_y
        rad = y - self.pivot_y
        real_z = zp - w * rad * np.tan(tilt)
        return (float(x), float(y), float(real_z))

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
        self.deformed = self.mesh.copy()
        if len(self.deformed.vertices) > 0:
            r = np.linalg.norm(self.deformed.vertices[:, :2], axis=1)
            self.deformed.vertices[:, 2] += r * self.tan_alpha
            self.start_z = params.get("start_z", self.deformed.bounds[0, 2])
            end_z = params.get("end_z", self.deformed.bounds[1, 2])
        else:
            self.start_z = params.get("start_z", 0.0)
            end_z = params.get("end_z", 10.0)
        self.num_layers = max(1, int(np.ceil((end_z - self.start_z) / self.layer_height)))

    def get_layer_count(self) -> int:
        return self.num_layers

    def get_slice_plane(self, layer_idx: int):
        z_prime = self.start_z + layer_idx * self.layer_height
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

        norm_vec = np.array([-df_dx, -df_dy, 1.0])
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
        self.expr = params.get("expression", "0.0")

        # Security: evaluate expression in strictly restricted namespace
        self.safe_dict = {"np": np, "math": __import__("math"), "__builtins__": {}}

        # Precompute deformed mesh
        self.deformed = self.mesh.copy()
        if len(self.deformed.vertices) > 0:
            x = self.deformed.vertices[:, 0]
            y = self.deformed.vertices[:, 1]
            f_val = self._evaluate(x, y)
            self.deformed.vertices[:, 2] += f_val
            self.start_z = params.get("start_z", self.deformed.bounds[0, 2])
            end_z = params.get("end_z", self.deformed.bounds[1, 2])
        else:
            self.start_z = params.get("start_z", 0.0)
            end_z = params.get("end_z", 10.0)
        self.num_layers = max(1, int(np.ceil((end_z - self.start_z) / self.layer_height)))

    def _evaluate(self, x, y):
        r = np.sqrt(x**2 + y**2)
        # Allows vectorized evaluation via np
        local_dict = {"x": x, "y": y, "r": r}
        return eval(self.expr, self.safe_dict, local_dict)

    def get_layer_count(self) -> int:
        return self.num_layers

    def get_slice_plane(self, layer_idx: int):
        z_prime = self.start_z + layer_idx * self.layer_height
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

        norm_vec = np.array([-df_dx, -df_dy, 1.0])
        norm_vec /= np.linalg.norm(norm_vec)
        return (float(norm_vec[0]), float(norm_vec[1]), float(norm_vec[2]))

    def compute_thickness(self, undeformed_pt: Tuple[float, float, float], layer_idx: int) -> float:
        x, y, z = undeformed_pt
        f_val = self._evaluate(x, y)
        return self._compute_custom_field_thickness(undeformed_pt, layer_idx, f_val)


class ExternalScalarFieldStrategy(BaseSlicingStrategy):
    def __init__(self, mesh: trimesh.Trimesh, params: Dict[str, Any]):
        super().__init__(params)
        self.mesh = mesh.copy()
        self.vertex_deformations = np.array(params.get("vertex_deformations", []))
        if len(self.vertex_deformations) == 0:
            self.vertex_deformations = np.zeros(len(self.mesh.vertices))

        self.layer_height = params.get("layer_height", 0.2)

        # Precompute deformed mesh
        self.deformed = self.mesh.copy()
        if len(self.deformed.vertices) > 0 and len(self.deformed.vertices) == len(self.vertex_deformations):
            self.deformed.vertices[:, 2] += self.vertex_deformations
            self.start_z = params.get("start_z", self.deformed.bounds[0, 2])
            end_z = params.get("end_z", self.deformed.bounds[1, 2])
        else:
            self.start_z = params.get("start_z", 0.0)
            end_z = params.get("end_z", 10.0)
        self.num_layers = max(1, int(np.ceil((end_z - self.start_z) / self.layer_height)))

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
        z_prime = self.start_z + layer_idx * self.layer_height
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

        norm_vec = np.array([-df_dx, -df_dy, 1.0])
        norm_vec /= np.linalg.norm(norm_vec)
        return (float(norm_vec[0]), float(norm_vec[1]), float(norm_vec[2]))

    def compute_thickness(self, undeformed_pt: Tuple[float, float, float], layer_idx: int) -> float:
        x, y, z = undeformed_pt
        f_val = self._interpolate_idw(x, y)
        return self._compute_custom_field_thickness(undeformed_pt, layer_idx, f_val)
