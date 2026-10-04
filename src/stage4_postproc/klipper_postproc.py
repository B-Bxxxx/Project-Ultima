import math
import os
from jinja2 import Environment, FileSystemLoader
from typing import List
from src.common.schemas import MachineTrajectory, MinimalPrintProfile
from src.stage4_postproc.base import BasePostProcessor

class Klipper5AxisPostProcessor(BasePostProcessor):
    """
    Stage 4 Post-Processor for Klipper 5-axis.
    Converts extrusion volume to linear E delta, templates G-code.
    """

    def generate_gcode(self, trajectory: MachineTrajectory, profile: MinimalPrintProfile) -> str:
        # Filament cross-sectional area
        r = profile.filament_diameter / 2.0
        area = math.pi * (r ** 2)

        # Prepare context
        template_states = []
        for state in trajectory.states:
            # E step calculation from volume
            if state.is_travel_move:
                e_val = 0.0
            else:
                e_val = state.extrusion_volume / area

            template_states.append({
                "x": state.x,
                "y": state.y,
                "z": state.z,
                "b": state.b,
                "c": state.c,
                "e": e_val,
                "feedrate": state.feedrate,
                "is_travel_move": state.is_travel_move,
                "feature_type": getattr(state, "feature_type", "outer_wall")
            })

        # Load Jinja2 template
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("klipper_5axis.gcode.j2")

        # Render G-code
        gcode = template.render(states=template_states, profile=profile)
        return gcode
