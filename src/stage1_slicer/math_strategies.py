import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import trimesh

class BaseSlicingStrategy(ABC):
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

    @abstractmethod
    def undeform_point(self, pt: Tuple[float, float, float], layer_idx: int) -> Tuple[float, float, float]:
        """Undeforms a single point extracted from the deformed contour."""
        pass

    @abstractmethod
    def compute_normal(self, undeformed_pt: Tuple[float, float, float], layer_idx: int) -> Tuple[float, float, float]:
        """Computes the analytical or numerical gradient surface normal at the undeformed point."""
        pass


class PlanarStrategy(BaseSlicingStrategy):
    def __init__(self, mesh: trimesh.Trimesh, params: Dict[str, Any]):
        self.mesh = mesh.copy()
        self.layer_height = params.get("layer_height", 0.2)
        self.start_z = params.get("start_z", self.mesh.bounds[0, 2])
        end_z = params.get("end_z", self.mesh.bounds[1, 2])
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
        self.mesh = mesh.copy()
        self.layer_height = params.get("layer_height", 0.2)
        self.start_z = params.get("start_z", self.mesh.bounds[0, 2])
        end_z = params.get("end_z", self.mesh.bounds[1, 2])
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
        plane_normal = [0, -np.sin(tilt), np.cos(tilt)]
        return [0, 0, z], plane_normal

    def deform_mesh(self) -> trimesh.Trimesh:
        return self.mesh

    def undeform_point(self, pt: Tuple[float, float, float], layer_idx: int):
        return pt

    def compute_normal(self, undeformed_pt: Tuple[float, float, float], layer_idx: int):
        tilt = self._get_tilt_for_layer(layer_idx)
        return (0.0, float(-np.sin(tilt)), float(np.cos(tilt)))


class ConicalStrategy(BaseSlicingStrategy):
    def __init__(self, mesh: trimesh.Trimesh, params: Dict[str, Any]):
        self.mesh = mesh.copy()
        self.layer_height = params.get("layer_height", 0.2)
        self.cone_angle = np.radians(params.get("cone_angle_deg", 15.0))
        self.tan_alpha = np.tan(self.cone_angle)

        # Precompute deformed mesh
        self.deformed = self.mesh.copy()
        r = np.linalg.norm(self.deformed.vertices[:, :2], axis=1)
        self.deformed.vertices[:, 2] += r * self.tan_alpha

        self.start_z = params.get("start_z", self.deformed.bounds[0, 2])
        end_z = params.get("end_z", self.deformed.bounds[1, 2])
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
        real_z = zp - rad * self.tan_alpha
        return (float(x), float(y), float(real_z))

    def compute_normal(self, undeformed_pt: Tuple[float, float, float], layer_idx: int):
        x, y, z = undeformed_pt
        rad = np.sqrt(x**2 + y**2)
        if rad > 1e-6:
            df_dx = (x / rad) * self.tan_alpha
            df_dy = (y / rad) * self.tan_alpha
        else:
            df_dx = 0.0
            df_dy = 0.0

        norm_vec = np.array([-df_dx, -df_dy, 1.0])
        norm_vec /= np.linalg.norm(norm_vec)
        return (float(norm_vec[0]), float(norm_vec[1]), float(norm_vec[2]))


class CustomExpressionStrategy(BaseSlicingStrategy):
    def __init__(self, mesh: trimesh.Trimesh, params: Dict[str, Any]):
        self.mesh = mesh.copy()
        self.layer_height = params.get("layer_height", 0.2)
        self.expr = params.get("expression", "0.0")

        # Security: evaluate expression in strictly restricted namespace
        self.safe_dict = {"np": np, "math": __import__("math"), "__builtins__": {}}

        # Precompute deformed mesh
        self.deformed = self.mesh.copy()
        x = self.deformed.vertices[:, 0]
        y = self.deformed.vertices[:, 1]
        f_val = self._evaluate(x, y)
        self.deformed.vertices[:, 2] += f_val

        self.start_z = params.get("start_z", self.deformed.bounds[0, 2])
        end_z = params.get("end_z", self.deformed.bounds[1, 2])
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
        f_val = self._evaluate(x, y)
        real_z = zp - f_val
        return (float(x), float(y), float(real_z))

    def compute_normal(self, undeformed_pt: Tuple[float, float, float], layer_idx: int):
        x, y, z = undeformed_pt
        eps = 1e-5
        f0 = self._evaluate(x, y)
        fx = self._evaluate(x + eps, y)
        fy = self._evaluate(x, y + eps)

        df_dx = (fx - f0) / eps
        df_dy = (fy - f0) / eps

        norm_vec = np.array([-df_dx, -df_dy, 1.0])
        norm_vec /= np.linalg.norm(norm_vec)
        return (float(norm_vec[0]), float(norm_vec[1]), float(norm_vec[2]))


class ExternalScalarFieldStrategy(BaseSlicingStrategy):
    def __init__(self, mesh: trimesh.Trimesh, params: Dict[str, Any]):
        self.mesh = mesh.copy()
        self.vertex_deformations = np.array(params.get("vertex_deformations", np.zeros(len(self.mesh.vertices))))
        self.layer_height = params.get("layer_height", 0.2)

        # Precompute deformed mesh
        self.deformed = self.mesh.copy()
        self.deformed.vertices[:, 2] += self.vertex_deformations

        self.start_z = params.get("start_z", self.deformed.bounds[0, 2])
        end_z = params.get("end_z", self.deformed.bounds[1, 2])
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
        f_val = self._interpolate_idw(x, y)
        real_z = zp - f_val
        return (float(x), float(y), float(real_z))

    def compute_normal(self, undeformed_pt: Tuple[float, float, float], layer_idx: int):
        x, y, z = undeformed_pt
        eps = 1e-4

        f0 = self._interpolate_idw(x, y)
        fx = self._interpolate_idw(x + eps, y)
        fy = self._interpolate_idw(x, y + eps)

        df_dx = (fx - f0) / eps
        df_dy = (fy - f0) / eps

        norm_vec = np.array([-df_dx, -df_dy, 1.0])
        norm_vec /= np.linalg.norm(norm_vec)
        return (float(norm_vec[0]), float(norm_vec[1]), float(norm_vec[2]))
