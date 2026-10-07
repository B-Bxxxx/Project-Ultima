import re

with open("src/stage2_toolpath/contour_generator.py", "r") as f:
    code = f.read()

# Replace `_generate_2d_features` to include concentric offsets and segment subdivisions

search = """    def _generate_2d_features(self, layer, profile: MinimalPrintProfile, strategy=None):"""

# We'll replace the entire `_generate_2d_features` method

replace = """    def _subdivide_segment(self, p1, p2, max_len):
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
                    z_prime = getattr(layer, "z_prime", layer.z_height)
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
                    spacing = offset_dist / (profile.infill_density / 100.0) if profile.infill_density > 0 else offset_dist * 4

                    x = minx + offset_dist
                    infill_lines = []
                    while x < maxx:
                        line = sg.LineString([(x, miny), (x, maxy)])
                        try:
                            inter = current_poly.intersection(line)
                            if inter.geom_type == 'LineString':
                                infill_lines.append(inter)
                            elif inter.geom_type == 'MultiLineString':
                                infill_lines.extend(list(inter.geoms))
                        except:
                            pass
                        x += spacing

                    for line in infill_lines:
                        add_contour(list(line.coords), "infill")

        if not new_contours:
            return layer.contours
        return new_contours"""

# Extract the existing method to replace
import re
match = re.search(r"    def _generate_2d_features.*?return new_contours", code, re.DOTALL)
if match:
    code = code[:match.start()] + replace + code[match.end():]

with open("src/stage2_toolpath/contour_generator.py", "w") as f:
    f.write(code)
