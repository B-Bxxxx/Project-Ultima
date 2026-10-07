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
        import shapely.geometry as sg
        if not layer.contours:
            return layer.contours

        rings = []
        for contour in layer.contours:
            if len(contour.points) > 2 and contour.points[0] == contour.points[-1]:
                pts_2d = [(p[0], p[1]) for p in contour.points]
                try:
                    ring = sg.LinearRing(pts_2d)
                    if ring.is_valid:
                        poly = sg.Polygon(ring)
                        if poly.is_valid:
                            rings.append((poly.area, ring, poly))
                except:
                    pass

        if not rings:
            return layer.contours

        rings.sort(key=lambda r: r[0], reverse=True)

        valid_exteriors = []

        # 1. Group rings into shells and holes
        for area, ring, poly in rings:
            is_hole = False
            for ext_poly, ext_ring, holes in valid_exteriors:
                if ext_poly.contains(poly):
                    holes.append(ring)
                    is_hole = True
                    break
            if not is_hole:
                valid_exteriors.append((poly, ring, []))

        # 2. Build final polygons with holes explicit parameter
        valid_polygons = []
        for ext_poly, ext_ring, holes in valid_exteriors:
            try:
                p = sg.Polygon(shell=ext_ring, holes=holes)
                if p.is_valid:
                    valid_polygons.append(p)
            except:
                pass

        new_contours = []
        offset_dist = profile.nozzle_diameter

        from src.common.schemas import SpatialContour
        import math

        def add_contour(poly_coords, f_type):
            if len(poly_coords) < 2:
                return

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

                    try:
                        current_poly = current_poly.buffer(-offset_dist)
                    except:
                        break

                if current_poly.is_empty or profile.infill_density <= 0:
                    continue

                bounds = current_poly.bounds
                if bounds:
                    minx, miny, maxx, maxy = bounds
                    density_frac = profile.infill_density if profile.infill_density <= 1.0 else profile.infill_density / 100.0
                    spacing = offset_dist / density_frac if density_frac > 0 else offset_dist * 4

                    angle = profile.infill_angle_deg if layer.layer_index % 2 == 0 else -profile.infill_angle_deg
                    import math as m
                    rad = m.radians(angle)
                    cos_a, sin_a = m.cos(rad), m.sin(rad)

                    # We compute the bounding box of the rotated polygon
                    # Handle MultiPolygons safely
                    if current_poly.geom_type == 'MultiPolygon':
                        pts = []
                        for geom in current_poly.geoms:
                            pts.extend(list(geom.exterior.coords))
                    else:
                        pts = list(current_poly.exterior.coords)

                    rot_coords = [(px*cos_a - py*sin_a, px*sin_a + py*cos_a) for px, py in pts]
                    minx_r = min(r[0] for r in rot_coords)
                    maxx_r = max(r[0] for r in rot_coords)
                    miny_r = min(r[1] for r in rot_coords)
                    maxy_r = max(r[1] for r in rot_coords)

                    x_r = minx_r + spacing / 2.0
                    infill_segments = []
                    while x_r < maxx_r:
                        p1_x = x_r*cos_a + miny_r*sin_a
                        p1_y = -x_r*sin_a + miny_r*cos_a
                        p2_x = x_r*cos_a + maxy_r*sin_a
                        p2_y = -x_r*sin_a + maxy_r*cos_a

                        line = sg.LineString([(p1_x, p1_y), (p2_x, p2_y)])
                        try:
                            inter = current_poly.intersection(line)
                            if inter.geom_type == 'LineString':
                                infill_segments.append(list(inter.coords))
                            elif inter.geom_type == 'MultiLineString':
                                for l in inter.geoms:
                                    infill_segments.append(list(l.coords))
                        except:
                            pass
                        x_r += spacing

                    # Now sort segments to form a zigzag path
                    # We alternate endpoints
                    if infill_segments:
                        # Sort by their rotated x-coordinate (which is the orthogonal distance)
                        def get_orthogonal_distance(seg):
                            # The line equation is roughly x_r = px * cos_a + py * sin_a
                            # We can just pick the first point
                            px, py = seg[0]
                            return px * cos_a + py * sin_a

                        infill_segments.sort(key=get_orthogonal_distance)

                        current_contour = []

                        # Build contiguous zigzags
                        unvisited = list(infill_segments)

                        while unvisited:
                            if not current_contour:
                                # Start a new contour with the first available segment
                                current_contour = unvisited.pop(0)
                            else:
                                last_pt = current_contour[-1]

                                # Find the closest segment endpoint among unvisited segments
                                best_dist = float('inf')
                                best_idx = -1
                                best_seg_oriented = None

                                # Only look at nearby segments to maintain rectilinear pattern
                                # We assume they are roughly sorted by orthogonal distance
                                search_limit = min(10, len(unvisited))
                                for i in range(search_limit):
                                    seg = unvisited[i]
                                    d1 = m.hypot(last_pt[0] - seg[0][0], last_pt[1] - seg[0][1])
                                    d2 = m.hypot(last_pt[0] - seg[-1][0], last_pt[1] - seg[-1][1])

                                    if d1 < best_dist:
                                        best_dist = d1
                                        best_idx = i
                                        best_seg_oriented = seg
                                    if d2 < best_dist:
                                        best_dist = d2
                                        best_idx = i
                                        best_seg_oriented = seg[::-1]

                                if best_dist <= spacing * 2.1:
                                    # Join them!
                                    current_contour.extend(best_seg_oriented)
                                    unvisited.pop(best_idx)
                                else:
                                    # Too far, finish this contour
                                    add_contour(current_contour, "infill")
                                    current_contour = []

                        if current_contour:
                            add_contour(current_contour, "infill")

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
