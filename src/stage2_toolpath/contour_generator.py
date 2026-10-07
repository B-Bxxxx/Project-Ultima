import math
import numpy as np
from typing import List
from src.common.schemas import UniversalSlicedModel, MinimalPrintProfile, CLDataTrajectory, CLDataWaypoint
from src.stage2_toolpath.base import BaseToolpathGenerator

import shapely.geometry as sg

class StandardToolpathGenerator(BaseToolpathGenerator):
    """
    Stage 2 Toolpath Generator.
    Converts UniversalSlicedModel to CLDataTrajectory by inserting waypoints, calculating travel moves,
    and calculating extrusion volumes.
    """

    def _subdivide_segment(self, p1, p2, max_len):
        import math
        dist = math.dist(p1, p2)
        if dist <= max_len:
            return [p2]

        num_segments = int(math.ceil(dist / max_len))
        pts = []
        for i in range(1, num_segments + 1):
            t = i / num_segments
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            pts.append((x, y))
        return pts

    def _generate_2d_features(self, layer, profile: MinimalPrintProfile, strategy=None):
        if not layer.contours:
            return layer.contours

        polygons = []
        for contour in layer.contours:
            if len(contour.points) > 2 and contour.points[0] == contour.points[-1]:
                pts_2d = [(p[0], p[1]) for p in contour.points]
                try:
                    poly = sg.Polygon(pts_2d)
                    if poly.is_valid:
                        polygons.append(poly)
                except:
                    pass

        if not polygons:
            return layer.contours

        polygons.sort(key=lambda p: p.area, reverse=True)

        valid_polygons = []
        for p in polygons:
            is_hole = False
            for vp in valid_polygons:
                if vp.contains(p):
                    try:
                        diff = vp.difference(p)
                        if diff.is_valid:
                            idx = valid_polygons.index(vp)
                            valid_polygons[idx] = diff
                            is_hole = True
                            break
                    except:
                        pass
            if not is_hole:
                valid_polygons.append(p)

        new_contours = []
        offset_dist = profile.nozzle_diameter

        from src.common.schemas import SpatialContour
        import math

        def add_contour(poly_coords, f_type):
            if len(poly_coords) < 2:
                return

            # Subdivision logic for 2D points to conform to non-planar shapes
            subdivided_coords = [poly_coords[0]]
            for i in range(1, len(poly_coords)):
                subdivided_coords.extend(self._subdivide_segment(subdivided_coords[-1], poly_coords[i], profile.max_segment_length))

            pts = []
            normals = []
            thicknesses = []

            for x, y in subdivided_coords:
                if strategy:
                    z_prime = layer.z_prime if hasattr(layer, "z_prime") and layer.z_prime is not None else layer.z_height
                    layer_idx = int(round((z_prime - strategy.start_z) / strategy.layer_height))
                    pt_3d = strategy.undeform_point((x, y, z_prime), layer_idx)
                    norm = strategy.compute_normal(pt_3d, layer_idx)
                    thick = strategy.compute_thickness(pt_3d, layer_idx)
                else:
                    pt_3d = (x, y, layer.z_height)
                    norm = (0.0, 0.0, 1.0)
                    thick = profile.layer_height
                pts.append(pt_3d)
                normals.append(norm)
                thicknesses.append(thick)

            new_contours.append(SpatialContour(
                points=pts,
                normals=normals,
                thicknesses=thicknesses,
                feature_type=f_type
            ))

        for poly in valid_polygons:
            if poly.geom_type == 'Polygon':
                polys = [poly]
            elif poly.geom_type == 'MultiPolygon':
                polys = list(poly.geoms)
            else:
                polys = []

            for p in polys:
                # 1. Concentric Perimeters
                current_poly = p
                for i in range(profile.num_perimeters):
                    if current_poly.is_empty:
                        break

                    if current_poly.geom_type == 'Polygon':
                        c_polys = [current_poly]
                    elif current_poly.geom_type == 'MultiPolygon':
                        c_polys = list(current_poly.geoms)
                    else:
                        c_polys = []

                    for cp in c_polys:
                        f_type = "outer_wall" if i == 0 else "inner_wall"
                        if cp.exterior:
                            add_contour(list(cp.exterior.coords), f_type)
                        for interior in cp.interiors:
                            add_contour(list(interior.coords), f_type)

                    # Offset inward for next perimeter
                    try:
                        current_poly = current_poly.buffer(-offset_dist)
                    except:
                        break

                # 2. Infill inside the remaining current_poly
                if current_poly.is_empty or profile.infill_density <= 0:
                    continue

                bounds = current_poly.bounds
                if bounds:
                    minx, miny, maxx, maxy = bounds
                    # Adjust spacing based on density
                    # 100% density = offset_dist spacing
                    # If density is <= 1.0, treat it as a fraction (e.g. 0.2 = 20%). If > 1.0, treat as percentage.
                    density_frac = profile.infill_density if profile.infill_density <= 1.0 else profile.infill_density / 100.0
                    spacing = offset_dist / density_frac if density_frac > 0 else offset_dist * 4

                    x = minx + offset_dist
                    infill_lines = []
                    # Alternating rectilinear: vary angle based on layer index
                    angle = profile.infill_angle_deg if layer.layer_index % 2 == 0 else -profile.infill_angle_deg

                    import math as m
                    rad = m.radians(angle)
                    cos_a, sin_a = m.cos(rad), m.sin(rad)

                    # Compute rotated bounding box
                    rot_coords = [(px*cos_a - py*sin_a, px*sin_a + py*cos_a) for px, py in current_poly.exterior.coords]
                    minx_r = min(r[0] for r in rot_coords)
                    maxx_r = max(r[0] for r in rot_coords)
                    miny_r = min(r[1] for r in rot_coords)
                    maxy_r = max(r[1] for r in rot_coords)

                    # Adjust spacing to avoid infinite loop
                    x_r = minx_r + spacing
                    infill_lines = []
                    while x_r < maxx_r:
                        # Inverse rotation for line endpoints
                        p1_x = x_r*cos_a + miny_r*sin_a
                        p1_y = -x_r*sin_a + miny_r*cos_a
                        p2_x = x_r*cos_a + maxy_r*sin_a
                        p2_y = -x_r*sin_a + maxy_r*cos_a

                        line = sg.LineString([(p1_x, p1_y), (p2_x, p2_y)])
                        try:
                            inter = current_poly.intersection(line)
                            if inter.geom_type == 'LineString':
                                infill_lines.append(inter)
                            elif inter.geom_type == 'MultiLineString':
                                infill_lines.extend(list(inter.geoms))
                        except:
                            pass
                        x_r += spacing

                    for line in infill_lines:
                        add_contour(list(line.coords), "infill")

        if not new_contours:
            return layer.contours
        return new_contours

    def generate_toolpath(self, model: UniversalSlicedModel, profile: MinimalPrintProfile) -> CLDataTrajectory:
        waypoints = []

        # We will keep track of the last point to calculate distance (dL)
        last_pt = None

        # Load strategy to re-project generated 2D shapes
        strategy = None
        if "strategy" in model.metadata:
            strat_name = model.metadata["strategy"]
            # To reconstruct strategy we need a mesh, but we don't have it here.
            # But the math strategies don't actually need the mesh *if* we only call undeform/compute.
            # We can recreate it with an empty mesh just for math.
            import trimesh
            from src.stage1_slicer.math_strategies import (
                PlanarStrategy, ProgressiveTiltStrategy, ConicalStrategy,
                CustomExpressionStrategy, ExternalScalarFieldStrategy
            )
            dummy_mesh = trimesh.Trimesh()
            if strat_name == "progressive_tilt":
                strategy = ProgressiveTiltStrategy(dummy_mesh, model.metadata)
            elif strat_name == "custom_expr":
                strategy = CustomExpressionStrategy(dummy_mesh, model.metadata)
            elif strat_name == "conical" or strat_name == "conical_or_curved":
                strategy = ConicalStrategy(dummy_mesh, model.metadata)
            elif strat_name == "external_field":
                strategy = ExternalScalarFieldStrategy(dummy_mesh, model.metadata)

                # Rebuild IDW from ALL layer boundaries
                import numpy as np
                pts_2d = []
                vertex_deformations = []
                for lyr in model.layers:
                    for contour in lyr.contours:
                        for idx, pt in enumerate(contour.points):
                            z_prime = lyr.z_prime if hasattr(lyr, "z_prime") and lyr.z_prime is not None else lyr.z_height
                            w = strategy.get_blend_weight(z_prime) if hasattr(strategy, 'get_blend_weight') else 1.0
                            if w > 1e-6:
                                f_val = (z_prime - pt[2]) / w
                                pts_2d.append([pt[0], pt[1]])
                                vertex_deformations.append(f_val)

                if pts_2d:
                    strategy.pts_2d = np.array(pts_2d)
                    strategy.vertex_deformations = np.array(vertex_deformations)
            else:
                strategy = PlanarStrategy(dummy_mesh, model.metadata)

        for layer in model.layers:
            layer.contours = self._generate_2d_features(layer, profile, strategy)
            for contour in layer.contours:
                if not contour.points:
                    continue

                # First point of the contour: is it a travel move?
                first_pt = contour.points[0]
                first_norm = contour.normals[0]

                # If we are jumping from a previous contour/layer and not continuous spiral, insert a travel move
                if last_pt is not None and not profile.continuous_spiral:
                    dist_to_start = math.dist(last_pt, first_pt)
                    if dist_to_start > 1e-6:
                        # Insert a travel move to the first point of this contour
                        waypoints.append(CLDataWaypoint(
                            x=first_pt[0], y=first_pt[1], z=first_pt[2],
                            i=first_norm[0], j=first_norm[1], k=first_norm[2],
                            extrusion_volume=0.0,
                            feedrate=3000.0, # default travel feedrate
                            is_travel_move=True,
                            feature_type=contour.feature_type
                        ))

                # For continuous spiral or normal extrusion, add the first point of the contour
                # If it's the very first point of the print, it's a travel move to get there
                if not waypoints:
                    waypoints.append(CLDataWaypoint(
                        x=first_pt[0], y=first_pt[1], z=first_pt[2],
                        i=first_norm[0], j=first_norm[1], k=first_norm[2],
                        extrusion_volume=0.0,
                        feedrate=3000.0,
                        is_travel_move=True,
                        feature_type=contour.feature_type
                    ))
                elif profile.continuous_spiral and last_pt is not None:
                    # In continuous spiral mode, extrude directly to the first point of the new contour
                    dl = math.dist(last_pt, first_pt)
                    if dl > 1e-6:
                        local_h = profile.layer_height
                        if contour.thicknesses and len(contour.thicknesses) > 0:
                            local_h = contour.thicknesses[0]
                        vol = dl * local_h * profile.nozzle_diameter
                        waypoints.append(CLDataWaypoint(
                            x=first_pt[0], y=first_pt[1], z=first_pt[2],
                            i=first_norm[0], j=first_norm[1], k=first_norm[2],
                            extrusion_volume=vol,
                            feedrate=1500.0,
                            is_travel_move=False,
                            feature_type=contour.feature_type
                        ))

                last_pt = first_pt

                # Now iterate through the rest of the contour points and extrude
                for idx, (pt, norm) in enumerate(zip(contour.points[1:], contour.normals[1:])):
                    # offset by 1 since we are iterating from 1:
                    real_idx = idx + 1

                    dl = math.dist(last_pt, pt)

                    if dl > 1e-6:
                        # Use local thickness if available, else nominal profile layer_height
                        local_h = profile.layer_height
                        if contour.thicknesses and len(contour.thicknesses) > real_idx:
                            local_h = contour.thicknesses[real_idx]

                        # V_ext = dL * local_layer_height * nozzle_diameter
                        vol = dl * local_h * profile.nozzle_diameter

                        waypoints.append(CLDataWaypoint(
                            x=pt[0], y=pt[1], z=pt[2],
                            i=norm[0], j=norm[1], k=norm[2],
                            extrusion_volume=vol,
                            feedrate=1500.0, # default extrusion feedrate
                            is_travel_move=False,
                            feature_type=contour.feature_type
                        ))
                        last_pt = pt

        return CLDataTrajectory(waypoints=waypoints)
