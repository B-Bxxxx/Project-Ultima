import math
import numpy as np
from typing import List
from src.common.schemas import UniversalSlicedModel, MinimalPrintProfile, CLDataTrajectory, CLDataWaypoint
from src.stage2_toolpath.base import BaseToolpathGenerator

class StandardToolpathGenerator(BaseToolpathGenerator):
    """
    Stage 2 Toolpath Generator.
    Converts UniversalSlicedModel to CLDataTrajectory by inserting waypoints, calculating travel moves,
    and calculating extrusion volumes.
    """

    def generate_toolpath(self, model: UniversalSlicedModel, profile: MinimalPrintProfile) -> CLDataTrajectory:
        waypoints = []

        # We will keep track of the last point to calculate distance (dL)
        last_pt = None

        for layer in model.layers:
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
                            is_travel_move=True
                        ))

                # For continuous spiral or normal extrusion, add the first point of the contour
                # If it's the very first point of the print, it's a travel move to get there
                if not waypoints:
                    waypoints.append(CLDataWaypoint(
                        x=first_pt[0], y=first_pt[1], z=first_pt[2],
                        i=first_norm[0], j=first_norm[1], k=first_norm[2],
                        extrusion_volume=0.0,
                        feedrate=3000.0,
                        is_travel_move=True
                    ))
                elif profile.continuous_spiral and last_pt is not None:
                    # In continuous spiral mode, extrude directly to the first point of the new contour
                    dl = math.dist(last_pt, first_pt)
                    if dl > 1e-6:
                        vol = dl * profile.layer_height * profile.nozzle_diameter
                        waypoints.append(CLDataWaypoint(
                            x=first_pt[0], y=first_pt[1], z=first_pt[2],
                            i=first_norm[0], j=first_norm[1], k=first_norm[2],
                            extrusion_volume=vol,
                            feedrate=1500.0,
                            is_travel_move=False
                        ))

                last_pt = first_pt

                # Now iterate through the rest of the contour points and extrude
                for pt, norm in zip(contour.points[1:], contour.normals[1:]):
                    dl = math.dist(last_pt, pt)

                    if dl > 1e-6:
                        # V_ext = dL * layer_height * nozzle_diameter
                        vol = dl * profile.layer_height * profile.nozzle_diameter

                        waypoints.append(CLDataWaypoint(
                            x=pt[0], y=pt[1], z=pt[2],
                            i=norm[0], j=norm[1], k=norm[2],
                            extrusion_volume=vol,
                            feedrate=1500.0, # default extrusion feedrate
                            is_travel_move=False
                        ))
                        last_pt = pt

        return CLDataTrajectory(waypoints=waypoints)
