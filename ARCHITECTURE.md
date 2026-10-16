# Modular 5-Axis Additive Manufacturing CAM Architecture

## System Pipeline Diagram
```mermaid
flowchart TD
    A[3D Mesh / Parametric Def] -->|Universal Slicer Plugins| B[Stage 1: Slicer]
    B -->|UniversalSlicedModel| C[Stage 2: Toolpath Generator]
    C -->|CLDataTrajectory| D[Stage 3: Kinematic Solver]
    D -->|MachineTrajectory| E[Stage 4: Post-Processor]
    E -->|G-code Strings| F[Firmware]

    style B fill:#e1f5fe,stroke:#01579b
    style C fill:#e8f5e9,stroke:#1b5e20
    style D fill:#fff3e0,stroke:#e65100
    style E fill:#fce4ec,stroke:#880e4f
```

## Data Contracts Registry
All inter-stage data exchanges are strongly typed using Pydantic in `src/common/schemas.py` and serialize to JSON files for standalone execution.

| Contract Schema | Origin -> Destination | Key Fields |
| --- | --- | --- |
| `UniversalSlicedModel` | Stage 1 -> Stage 2 | `layers` (List of `UniversalLayer`: `layer_index`, `z_height`, `z_prime`, `contours`), each contour with `points` (X,Y,Z), `normals` (I,J,K), `thicknesses`, `vertical_gaps` / `normal_gaps` (per point gap to the next layer, see layer-gap check) and `feature_type` (`boundary` straight from Stage 1). `metadata`: the slicing parameters plus the **resolved** values (`start_z`, `end_z`, `pivot_y`, `transition_height`, ...) written by `strategy.get_resolved_params()`; Stage 2 rebuilds the strategy from exactly these. Large numpy fields (e.g. `vertex_deformations`) are never stored. |
| `MinimalPrintProfile` | Config -> Stage 2/4 | `layer_height`, `nozzle_diameter`, `num_perimeters`, `top_solid_layers`, `bottom_solid_layers` (0 disables the skin), `infill_density` (fraction 0..1), `infill_pattern` (`lines`), `infill_angle_deg`, `max_segment_length`, `continuous_spiral`, `filament_diameter`, speeds (`print_speed`, `travel_speed`, `z_speed`, `retract_speed`, `unretract_speed`), `park_lift`, `infill_overlap`, `thickness_model` (`stage1` | `nozzle_axis`), `retract_distance`, `unretract_distance` (retraction before a travel and the prime right before printing resumes are separate; both default to the retract values), `z_hop`, `min_retract_travel`, `travel_command` (`G0`/`G1`), `hop_short_travels` (Z-hop also on travels below `min_retract_travel`, never with retract). `wall_count` is accepted as an alias of `num_perimeters`. |
| `CLDataTrajectory` | Stage 2 -> Stage 3 | `waypoints` (List of `CLDataWaypoint`): `x`, `y`, `z`, `i`, `j`, `k`, `extrusion_volume`, `feedrate`, `is_travel_move`, `feature_type`, `layer_index` (the layer a waypoint belongs to; a travel move belongs to the layer it travels INTO). |
| `MachineConfig` | Config -> Stage 3 | `kinematic_topology`, pivot offsets, axis limits, `max_3axis_tilt_deg`. |
| `MachineTrajectory` | Stage 3 -> Stage 4 | `states` (List of `MachineStateVector`): `x`, `y`, `z`, `b`, `c`, `extrusion_volume`, `feedrate`, `is_travel_move`, `feature_type`, `layer_index` (copied unchanged from the waypoint, so a state keeps its layer in machine space). |

## Stage 1 Mathematical Slicing Strategies (`src/stage1_slicer/math_strategies.py`)

*Note: All strategies support adaptive local layer thickness and dynamic mesh bounds.*

| Strategy ID | Description | Parameters |
| --- | --- | --- |
| `planar` | Standard planar slicing ($Z = const$) | `start_z`, `end_z`, `layer_height` |
| `progressive_tilt` | Linearly tilts normal around X axis across layers. | `start_tilt_deg`, `end_tilt_deg` |
| `conical` | Deform-Slice-Undeform for support-free overhangs ($z' = z + r \cdot \tan\alpha$) | `cone_angle_deg` |
| `custom_expr` | Deform-Slice-Undeform of $f(x, y)$. The expression is validated against an AST whitelist (`src/stage1_slicer/safe_expr.py`), never evaluated blindly. | `expression` |
| `grid_field` | Same deform-slice-undeform as `custom_expr`, but f(x, y) is a regular grid (imported from a data file, bilinear interpolation, clamped outside). All parameters are plain lists, so they travel through the JSON metadata to Stage 2. | `grid_x`, `grid_y`, `grid_values` (`len(grid_y)` rows of `len(grid_x)` values) |
| `external_field` | Point cloud spatial interpolation (pure Numpy IDW) | `vertex_deformations` array or list (not stored in the JSON metadata; Stage 2 rebuilds the field from the layer contours) |

Every strategy implements the same interface used by Stage 1 **and** Stage 2 (`build_strategy(name, mesh, params)` is the single factory; Stage 2 passes an empty mesh and the resolved metadata):

| Method | Purpose |
| --- | --- |
| `get_resolved_params()` | Copy of the parameters after defaults were resolved (mesh-derived `start_z`, `end_z`, `pivot_y`, ...). Stored in the Stage 1 metadata. |
| `project_to_layer(x, y, layer_idx)` | Lifts a 2D point of a layer to its true 3D position. Planar/deforming strategies use `undeform_point` at `z' = start_z + idx*h`; `progressive_tilt` returns the point on the tilted plane `z = z0 + (y - pivot_y) tan(t)`. |
| `compute_normal`, `compute_thickness` | Normal and local layer thickness at a 3D point. |
| `_solve_forward_z(z, f)` | Solves `z' = z + w(z') f` in closed form (w is piecewise linear): exact for every f < transition_height, however negative; f >= transition_height folds the layers over and logs a warning. |

## Function files and machine files (`config/functions`, `config/machines`)
Both are JSON (`format` + `version` + `name`), validated strictly (unknown fields, missing parameters, out-of-range values are errors that name the element). Parameters are declared as `"name": {"label", "unit", "default" (mandatory), "min", "max", "step", "type": number|int|bool|choice}` (`src/common/param_spec.py`); the GUI builds one named input field per declared parameter and can save the current values back as new defaults.
* **Function file** (`src/stage1_slicer/function_file.py`, `"version": 2`): the file CONTAINS the maths - there is no `type` that selects hidden code. Exactly one of: `"expression"` (f(x, y), layers follow z = z0 - f; flat = `"0"`, cone = `"r * tan(radians(cone_angle_deg))"`), `"data_file"` (f(x, y) from a grid file, path relative to the file) or `"tilt_x_deg"` (every layer is a plane tilted about X by this expression of `t` = layer position 0..1, with the pivot line `"pivot_y"`, default `"y_min"` = lowest Y of the part; `t` and `y_min` are the only names the program supplies). Every other name must be declared in `parameters` (nothing predefined, missing = error; `t`, `y_min`, `x`, `y`, `r`, `np`, `math` are reserved). `tilt_x_deg` / `pivot_y` are one number per layer, so `x`, `y`, `r` are rejected there. Version 1 files are rejected. Shipped in `config/functions/` (all listed in the GUI, Planar is the default): `planar`, `progressive_tilt` (the template), `wave`, `conical`, `rounded_ridge` (inverted V with a rounded apex: `k * (sqrt(x**2 + (R*k)**2) - R*k)`), `ripple` (concentric waves), `dome` (spherical cap, a button top). The strategies behind it: `custom_expr` (expression, now with mesh refinement like the conical strategy), `grid_field`, `progressive_tilt` (`tilt_expression` + `expression_params`; the old `start_tilt_deg`/`end_tilt_deg` interpolation stays for code/CLI use). Tests prove that the file based tilt, planar and cone slice like the original strategies.
* **Machine file** (`src/stage3_kinematics/machine_file.py`): `kinematics` (registered solver), `axes` (only these appear in the G-code; X Y Z, plus B and C for rotary machines), optional `build_volume`, `parameters` (MachineConfig fields; a rotary machine must declare the B/C limits). Shipped: a generic 3-axis Cartesian machine and a basic 5-axis trunnion table (placeholder values).
* **JSON editors** (`src/gui_editors.py`, Edit... / New... in the Function and Machine sections): one window with one text field - the JSON file is the whole description. A status line validates the text with the real loaders on every keystroke (`FunctionSpec.from_dict` / `MachineSpec.from_dict`), shows the first error and disables Apply / Save as until the file is valid. Apply adds the spec to the session combo box; Save as writes the file. The deformed (layered) body is checked in the Stage 1 tab, there is no separate preview in the editor.
* **Layer-gap check** (`src/stage1_slicer/thickness_check.py`, label under *Transition H*): no function specific maths. `BaseSlicingStrategy.layer_gaps(x, y, i)` gives, for every strategy alike, the *vertical* gap to the next layer (what a vertical nozzle works in) and the *normal* gap (vertical * cos slope, what a nozzle along the surface normal works in; normal <= vertical). Stage 1 stores both per point (`SpatialContour.vertical_gaps` / `normal_gaps`). The nozzle window is 0.25 .. 0.80 of the nozzle diameter. `check()` decides by the machine mode (rotary axes -> normal, otherwise vertical) and always reports both values and whether the part is printable in both directions. `probe()` evaluates the strategy on sample points (live label, no slicing), `measure_model()` uses the real slice (warning after the run), `suggest_transition()` bisects the smallest passing transition height on click and says when the transition has no effect (tilting layers) or cannot help (layer height itself outside the window). The exact along-nozzle gap after Stage 3 is a later step.

## Stage 0 Function Import & Placement
* `src/stage1_slicer/function_import.py` - user deformation functions f(x, y). **Coordinate reference:** x, y are relative to the *centre of the part's footprint* (XY bounding-box centre of the placed mesh): the strategies subtract `origin_x` / `origin_y`, which Stage 1 resolves from the mesh and stores in the metadata so Stage 2 uses the same values (an explicit `origin_x` / `origin_y` parameter overrides). Moving the part on the bed therefore does not change the slice; grid data files are read in the same part-relative coordinates; layers follow `z = z0 - w(z)*f(x, y)` (w = transition blend), so a positive f lowers the layer surface.
  * A. expression file (`.txt`: one expression, `#` / `//` comment lines ignored, lines joined; `.json`: `{"expression": "..."}`), validated by the AST whitelist at load time.
  * B. data file (`x, y, f` columns, CSV / whitespace / `;`, optional header): a complete regular grid is used as is, scattered points are resampled to a regular grid with pure-numpy IDW. Result: `GridField` -> `grid_field` strategy. `preview_surface` samples either kind of field for the Stage 0 preview.
* `src/common/placement.py` - `place_mesh`: rotate about the bounding-box centre (X, Y, Z order), centre in XY, offset X/Y, then either drop to z = 0 or (drop off) move by `offset_z`. Returns a copy.
* `src/common/view_model.py` - Qt-free view logic: layer range (`effective_layers`, `move_range`) and the `Playback` state machine; `feature_styles.group_lines_by_feature` supports `layer_min` and `move_limit`.

## Stage 2 Toolpath Generation (`src/stage2_toolpath/contour_generator.py`)

`StandardToolpathGenerator.generate_toolpath(model, profile)` never modifies the model. Steps:

1. **Outlines**: closed contours become shapely polygons; nesting-depth parity decides hole (odd) vs. outer boundary / island (even). Open, self-intersecting or degenerate contours are skipped with a warning (the raw `boundary` contour is never emitted).
2. **Walls and infill region**: `num_perimeters` walls offset by `nozzle_diameter`; the infill region is the outline eroded by all walls.
3. **Skin**: layers are compared **by `layer_index`** (missing neighbour = exposed). `bottom_solid` = region minus the intersection of the `bottom_solid_layers` regions below, `top_solid` = region minus the intersection of the `top_solid_layers` regions above, `infill` (sparse) = the rest. Solid skin uses 100 % density at +-45 deg alternating per layer.
4. **Fill**: scanlines per connected polygon, joined into zigzag chains only when the straight link stays inside `region.buffer(0.1 * spacing)`; otherwise a new chain starts (a real travel).
5. **Trajectory**: 2D points are projected with `strategy.project_to_layer`; travel moves are inserted between non-adjacent contours.

| `feature_type` | Meaning |
| --- | --- |
| `inner_wall`, `outer_wall` | Wall rings, printed per outline from the innermost wall to the outer wall |
| `bottom_solid`, `top_solid` | Solid skin |
| `infill` | Sparse infill |

Print order inside a layer is `FEATURE_ORDER = (inner_wall, outer_wall, bottom_solid, top_solid, infill)`.
`generate_features(model, profile)` returns the per-layer feature contours, `compute_layer_regions(model, profile)` the shapely regions (for tests and previews).

## Stage 3 Kinematic Solvers (`src/stage3_kinematics/registry.py`)

| Topology ID | Transformation |
| --- | --- |
| `trunnion_table_xyzbc` | $P_m = P_p + R_B(B) \cdot (R_C(C) \cdot P_e - P_p)$ |
| `swivel_head_xyzbc` | $P_m = P + P_p - R_C(C) \cdot R_B(B) \cdot P_p$ |
| `planar_3axis_xyz` | $P_m = P$ (Pass-through). Logs warnings if surface tilt exceeds `max_3axis_tilt_deg`. |

## User Interfaces

| Interface | Command | Description |
| --- | --- | --- |
| **Interactive GUI (ULTIMA slicer)** | `python3 src/gui_app.py` | Native PyQt6 with a charcoal theme with coral and amber accents (`src/gui_theme.py`, flat cards, drawn arrow images) + pyqtgraph desktop application to generate/upload meshes, configure slicing/kinematics parameters, preview 3D layers/toolpaths and export G-code. The settings column is a scroll area with collapsible sections. Part placement (rotate / centre / offset / drop to bed), layer range sliders, playback (play/pause, speed, scrub; nozzle marker), Stage 0 function import with a surface preview and a Stage 2/3 status line (layer, feature, X Y Z, B C) are available. Skin layers (top/bottom), travel command, retract and Z-hop are editable; the view checkboxes double as the colour legend (`src/common/feature_styles.py` is the single source of line-type colours: outer wall green, inner wall yellow, bottom skin orange, top skin magenta, sparse infill blue, travel red). |
| **CLI Runner** | `python3 src/pipeline_cli.py run-all` | Headless execution of all stages sequentially using JSON configs. |

## Validation & Test Matrix
Execute with `pytest tests/`

| File | Subsystem | Validation |
| --- | --- | --- |
| `test_stage1_math_slicer.py` | Stage 1 Plugin | Validates discrete plugin parameter parsing & default vertical normals. |
| `test_stage1_field_slicer.py` | Stage 1 Engine | Evaluates output structure of mathematical fields. |
| `test_mathematical_validation.py` | Stage 1 & 3 | **Mathematical Exactness**: Validates numerical derivatives against analytical expressions. Validates Kinematic FK/IK Roundtrip error $< 1e-5$. |
| `test_multi_kinematics.py` | Stage 3 Solvers | Tests topology configurations. |
| `test_stage4_postproc.py` | Stage 4 Post-proc | Parses the G-code (`tests/gcode_tools.py`): E = volume/area, retract -> Z-hop -> travel -> Z-hop -> prime sequence, short travels direct, G0/G1 selection, no start heaters, `;LAYER:n` / `; TYPE:` markers, `split_gcode_by_layer`, profile validation. |
| `test_feature_styles.py` | GUI line types | Distinct colour per line type, every Stage 2 feature has a style, grouping honours visibility and `layer_index`; offscreen GUI smoke test (skipped without PyQt6). |
| `test_deformation_independent.py` | Stage 1 math | Independent checks: outward cone normal vs. finite differences, deform -> undeform round trip, refinement vs. the analytic surface. |
| `test_safe_expr.py` | Stage 1 security | AST whitelist: malicious / DoS expressions are rejected, valid ones match a numpy reference. |
| `test_stage12_consistency.py` | Stage 1 -> 2 | Resolved metadata, identical Stage 1 / Stage 2 strategy results per layer, tilted-plane z, `external_field` end to end (Stage 1 -> JSON -> Stage 2). |
| `test_stage2_toolpath.py` | Stage 2 | Top/bottom skin, zigzag linking, holes and islands, immutability, travel counts, `layer_index` through Stage 3. |
| `test_visual_cases.py` | E2E CLI | Asserts all JSON and HTML files are generated correctly. |
| `test_function_machine_files.py` | JSON files | ParamSpec rules, strict function/machine file validation, shipped templates, round trips, `expression_params` end to end, per-state `tool_alignment` of all solvers. |
| `test_first_layer.py` | Layer convention | First layer at one layer height above the bed (never z = 0), last layer at the top, `start_z` = base of the print, every strategy's first layer flat. |
| `test_phase5_models.py` | Phase 5 logic | Placement (incl. Z offset), layer range, playback state machine, grouping by range. |
| `test_function_import.py` | Stage 0 import | Expression/data file parsing and rejection of unsafe input, GridField interpolation, `grid_field` equals the equivalent expression end to end, JSON-safe metadata. |
| `test_gui_smoke.py` | GUI | Offscreen: placement (Z offset only without drop-to-bed) invalidates results, camera framing, mouse wheel never edits a parameter, planar G-code has no B/C, separate retract/unretract, layer range / playback / status, function import (skipped without PyQt6). |
| `test_contour_assembly.py` | Stage 1 contours | Zero-length segments, touching loops and open polylines are traced completely; both rings of a wave-deformed hollow cylinder are closed with the right length in every wall layer. |
| `test_input_widgets.py` | GUI input | With a `hu_HU` locale: `.` and `,` both work, arrows / clicks / typing work in every visible spin box, a typed placement value is applied once on Enter. |
| `test_function_origin.py` | Function origin | The same part at (0,0) and (150,150) gives the same Stage 1 contours and Stage 2 walls for every shipped function (top 3 layers excluded), origin stored in the metadata, explicit override, preview origin. |
| `test_thickness_check.py` | Layer-gap check | Window, planar gap = h, normal <= vertical, slice vs probe agreement, machine-mode direction with both values, suggestion bisection, tilt without effect, impossible layer height, GUI label. |
| `test_editors.py` | GUI editors | Offscreen: JSON text is validated live, errors block Apply, Apply/Save as, Edit/New buttons add to the combos. |

## Layer convention (Stage 1)
Layer `i` is printed at `z' = start_z + (i + 1) * h` where the parameter `start_z` is the base of the print (default: the lowest point of the mesh) - the first layer therefore sits one layer height above the bed, never at z = 0. Its contours are cut from the mid-plane of the layer (`z' - h/2`), so a slice never coincides with a top or bottom face. `UniversalLayer.z_height` is that mid-plane height; the printed Z comes from `project_to_layer`. The deformation blend `w` starts at the first layer (flat first layer).

## Stage 4 output conventions
* End block: lift `park_lift` mm above the highest point (no fixed Z); `B0 C0` follows only on rotary machines. `thickness_model='nozzle_axis'` scales E by 1/cos(nozzle axis vs layer normal) using `MachineStateVector.tool_alignment`.
* Relative extrusion (M83), absolute positioning; no heater commands at the start (the end block still turns heaters off).
* `;LAYER_COUNT:N` in the header, `;LAYER:n` before the first move of each layer, `; TYPE: <feature>` per layer/feature change.
* Every travel >= `min_retract_travel`: retract, Z-hop up (highest Z of the layer the travel runs in - of both layers at a layer change - plus `z_hop`, so wavy layers are cleared too), travel, Z-hop down, prime. Travel/hop moves use `travel_command` (G0 or G1). The first move lifts without retracting.
