# Pitfalls log

Problems and traps met while building the pipeline - raw material for the documentation and for the
"known pitfalls" chapter. Newest entries at the bottom of each section. **Status:** fixed / open / by design.

## Slicing and geometry
| # | Pitfall | Status | Notes |
|---|---------|--------|-------|
| 1 | **Slice plane on a face.** Slicing at z = 0 (first layer) or exactly at the top coincides with the bottom/top face; the result depends on the mesh library. | fixed | Layers are cut from the mid-plane (`z' - h/2`) and printed at the layer top (`z'`). See ARCHITECTURE "Layer convention". |
| 2 | **First layer at z = 0.** The nozzle would be on the bed. | fixed | The `start_z` parameter is the BASE of the print; layer 0 is printed at `start_z + h`. |
| 3 | **Fold-over of the deformation.** z' = z + w(z')·f is only monotonic while f < transition_height; a larger positive f can converge to a folded solution without any error. | warning | `_solve_forward_z` logs a warning when max(f) >= T; a hard preflight check is still to do. |
| 4 | **Non-convergence** of the forward iteration for steep negative fields. | fixed | Raises a ValueError with the residual. |
| 5 | **Stage 1 vs Stage 2 drift.** Stage 2 rebuilt the strategy with different defaults (start_z, pivot_y) than Stage 1 used. | fixed | The RESOLVED parameters are stored in the model metadata; a consistency test compares both stages layer by layer. |
| 6 | **Stage 2 mutated its input contours**, so tests passed vacuously. | fixed | Stage 2 never modifies the model; tests assert immutability and `checked > N` guards against empty checks. |
| 7 | **Tilt normals** are blended with the flat normal in the transition zone, while the slice plane is not blended. | open | Only matters with `transition_height > 0` and tilt. |
| 8 | **Tilt pivot** at the lowest-Y edge keeps z >= 0 but makes the tilt direction fixed (about X). | by design | Other tilt axes need a function file type of their own. |

## Toolpath (Stage 2)
| # | Pitfall | Status | Notes |
|---|---------|--------|-------|
| 9 | **Infill links.** A zigzag link that leaves the region must not be drawn through the wall; it starts a new chain instead (extra travels). | fixed | `fill_region(..., link_tol)`; chains do not follow the boundary yet (open improvement). |
| 10 | **Skin / core / wall seams.** Regions only touch, which leaves a hairline gap. | fixed | `infill_overlap` (default 0.15 x nozzle) grows the filled region; it never reaches beyond the centre line of the innermost wall. |
| 11 | **Nesting parity.** Holes/islands must follow the nesting depth of the contours, not their orientation. | fixed | |
| 12 | **Skin detection** compares each layer with its neighbours by `layer_index`, not by Z (tilted layers have no common Z). | fixed | A gap in the numbering is treated as a surface. |
| 13 | `infill_pattern` accepts only `lines`; others fall back with a warning. | open | |

## Extrusion
| # | Pitfall | Status | Notes |
|---|---------|--------|-------|
| 14 | **Which thickness?** Stage 1 gives the layer thickness NORMAL to the layer. With a vertical nozzle on a tilted layer the gap along the nozzle is t / cos(tilt) (about 8 % more at 20 degrees, measured). | by design | `thickness_model`: `stage1` (default for 3-axis) or `nozzle_axis` (E scaled by 1 / cos(angle between nozzle axis and layer normal), capped at cos 78 degrees). The GUI chooses automatically (`auto`): rotary machine -> `nozzle_axis`. To be validated with real prints. |
| 15 | **Relative E and retract accounting.** Every retract must be matched by a prime before the next extrusion; retract and unretract are separate parameters. | fixed | Tests parse the G-code and check the balance. |
| 16 | **G-code text tests** break with every template change. | fixed | `tests/gcode_tools.py` parses the semantics (moves, axes, E) instead. |
| 17 | Travel at z = 0 / no Z-hop on short travels looked like a bug. | by design | Short travels (< `min_retract_travel`) are direct; `hop_short_travels` adds the hop. |
| 18 | **Fixed park position** (Z150) can be inside the part or below the build volume. | fixed | End block lifts `park_lift` above the highest point of the print. |
| 19 | **Axes in the G-code.** B/C were always written, even for a 3-axis machine. | fixed | The machine file's `axes` decide; `include_rotary_axes` is derived from it. |

## Machine and function files
| # | Pitfall | Status | Notes |
|---|---------|--------|-------|
| 20 | Expressions come from files: never `eval` blindly (AGENTS.md rule 5). | fixed | AST whitelist; user parameters must be plain identifiers that shadow nothing. |
| 21 | **Silent defaults hide mistakes.** A function that needs a parameter it does not declare, or a machine with rotary axes but no limits, is an error. | by design | `FunctionFileError` / `MachineFileError` name the missing element. |
| 22 | **Coordinate reference of f(x, y).** Which x, y does the function see? | decided | The centre of the part's footprint (bbox centre; was: the part as placed on the bed, changed because moving the part changed the slice). `origin_x/origin_y` are resolved in Stage 1 and travel in the metadata. The infill lattice is still anchored to the bed grid. Re-check when 5-axis work starts. |
| 23 | Rotary kinematics are not validated (first tests are 3-axis). | open | Tool-axis alignment is recorded per state (`tool_alignment`) for later checks. |

## GUI
| # | Pitfall | Status | Notes |
|---|---------|--------|-------|
| 24 | Many fields in a fixed-height column overlap each other. | fixed | Scroll area + collapsible sections. |
| 25 | **Mouse wheel changed parameters** while scrolling the menu. | fixed | Spin boxes, combo boxes and sliders ignore the wheel. |
| 26 | `QCheckBox` does not render HTML (the legend showed raw `<span>`). | fixed | Plain text, coloured by stylesheet. |
| 27 | A dropped `QApplication` reference aborts Python in tests. | fixed | Kept alive in the test module. |
| 28 | A far-away part (offset 120 / 80) left the camera behind. | fixed | The camera and the floor grid follow the part in X/Y; Z is never faked. |
| 29 | `trimesh` boolean (hollow cube) needs an external engine that is not always installed. | removed | The hollow cube was dropped; the hollow cylinder needs no engine. |
| 30 | GL rendering cannot be verified in the headless test environment. | open | Offscreen smoke tests only check the logic and that nothing raises. |
| 31 | Function files used a `type` label while the real maths (tilt, cone) lived in Python - the file did not describe the function. | fixed | Version 2: the maths (expression / tilt_x_deg + pivot_y) is in the JSON; the editor is just a validated JSON text field. |
| 32 | `custom_expr` did not refine the mesh, so a coarse mesh (a cube) was deformed only at its corners. | fixed | Same refinement as the conical strategy (`refine_mesh`, `refine_fraction`). |
| 33 | **Contour assembly lost parts of a ring**: `nx.find_cycle` returned only the first cycle - with a plane through a mesh vertex (degree 4 node) or a zero-length segment (two points rounding to one key) the contour skipped a piece (jagged / cut-off rings) or the whole ring was replaced by a 1-edge loop. Seen on a hollow cylinder with the wave function. | fixed | Segments are traced (every segment used once, zero-length ones dropped); `tests/test_contour_assembly.py`. |
| 34 | **Hungarian locale swallowed the decimal point**: typing `0.5` in a spin box gave 5, `12.75` gave 1275. | fixed | Decimal spin boxes use the C locale and accept `.` and `,`; keyboard tracking is off (value is committed on Enter / Tab / arrows, so typing no longer rebuilds the scene for every digit). `tests/test_input_widgets.py` runs with `hu_HU`. |
| 35 | Spin box / combo arrows disappeared with the neon stylesheet (QSS cannot draw triangles from borders). | fixed | Arrow images are drawn once into a temp folder and referenced by the stylesheet. |
| 36 | **Flat top face vs. non-planar layers**: the layers follow z = z0 - f(x, y) but the part's top is a flat plane, so the last layers are only partly there (terraces / a serrated rim). Expected geometry, not a slicing error. | open | Needs a top transition (the deformation fades out towards the top) - proposed, not implemented. |
| 37 | **"did not converge" on a harmless negative field** (V-groove f = -k*abs(x), |f| 7.5 > transition_height 5): the old fixed-point iteration for z' = z + w(z')*f needs |f| < T although the equation always has an exact solution for f < T. | fixed | Closed-form solution (w is piecewise linear), no iteration; only f >= T (real fold) is warned about. Note: a negative f inside the transition zone makes the real layer spacing larger, up to h * (1 + |f| / T). |
| 38 | **Mesh refinement ignores the function**: edges are split to extents/8 whatever the wavelength, and the deformed mesh is only linear between vertices. Measured on a 30x20 cylinder: Ripple b=3: error 0.07 mm; Ripple b=1: 0.50 mm (amplitude 0.4!); Dome Rs=20: 0.21 mm. Caps and sloped faces take the wrong shape, vertical walls are fine. | fixed | Adaptive conforming edge refinement (`refine_tolerance`, `mesh_refine.py`); the uniform split is only used when `feature_scale` / `refine_fraction` are given. Note: vertical walls are exact whatever the refinement - only caps and sloped faces depend on it. |
| 39 | The 3-axis tilt warning (`max_3axis_tilt_deg`) goes to the logger only - the GUI user never sees it, steep templates (k = 0.5 is 27 deg) can pass silently. | open | Show it in the status / metrics line. |
| 40 | Default layer height 0.4 on a 0.4 nozzle is above the 0.8 x nozzle window; with any function the advice flags it. | fixed | GUI default layer height is now 0.2 mm. |
| 41 | Real layer spacing and extrusion thickness are computed for the vertical direction only; on steep flanks the thickness along the normal is smaller by cos(slope). | open | Decide with the first test prints (material excess is deferred). |
| 42 | **Top layers are ill-conditioned.** The last 2-3 layers cut the deformed top cap at a grazing angle, so float noise alone moves the contour by 0.04-0.2 mm (the same slice shifted by 150 mm differs only there); the function-origin tests skip the top 3 layers. | open | Likely cause of the zig-zag top layer; to be measured and fixed. |
| 43 | **Wall direction and seam were not normalised.** Stage 1 traces each ring from the first segment trimesh returns (triangle order, changes with the layer height), so direction and start point flipped between layers, and the outer wall could run opposite to the inner walls (`buffer` orientation). | fixed | Stage 2 orients every wall ring (outer CCW, holes CW) and rotates it to start at the vertex farthest in `seam_angle_deg` (default back). The skin and infill chains are not aligned. |
| 44 | **Too many, too short segments** (13 MB G-code, 37% of the segments < 0.1 mm, 95th percentile needs > 1000 lines/s at 25 mm/s) make 8/32-bit Marlin boards stutter, the nozzle speeds up and down and the wall looks like fuzzy skin. | fixed | Stage 2 path simplification (`path_tolerance`, `infill_path_tolerance`, `min_segment_length`). The line-rate limit of the boards is a rule of thumb (8 bit ~100, 32 bit ~300 lines/s), to be tuned with test prints. Stage 1 contours are thinned too (`contour_tolerance`) and the per-point work is vectorized (120-section ring: Stage 1 45.6 s -> 0.6 s, Stage 2 36 s -> 1.1 s). |
| 45 | **The standard transition cannot reach a large function value within a few millimetres**: the layer thickness `h (1 - f/T)` leaves the printable band, only warned about. | workaround | Optional ramp mode (`transition_mode: ramp`): planar partial base layers up to the function surface, then exact-f layers. Limits: interior walls along the clipped stair edges, mostly solid skin on the first full layer (see ARCHITECTURE.md). |

* The thickness check is estimated (vertical and normal gap to the next layer, per Stage 1 point). The exact gap along the nozzle (normal / tool alignment) needs Stage 3 and is not checked yet; a layer height of 0.4 on a 0.4 nozzle fails the check by design (window 0.10 .. 0.32); the GUI default is 0.2.

## Process
* A long single prompt overwhelms an assistant; split into small, verifiable tasks.
* A branch that deletes unrelated files can look like a successful implementation - always diff against main before merging.
* Update ARCHITECTURE.md in the same commit as the code (AGENTS.md rule 2).

* `pipeline_cli.py` is unused by the author and still hardcodes a 10x10 cylinder and the old machine config; it is not maintained (decision: leave it).

* Ramp contact-only without pre-layers dropped overhang material below the first full layer (Part69: ~1650 mm3 of 26900) - fixed with pre-layers. Slicing in a plane that contains a flat face gave open contours ("skipping an open contour") - fixed by nudging the plane. Known limit: non-contact material lower than one layer height above the bed (e.g. a chamfered underside) is not printed by the pre-layers.

## Machine origin / rotary axis position (`c_axis_x/y/z`)
- Optional machine parameters (default 0 = old behaviour): machine position of the C axis (table centre) at B = 0. The part-frame origin lies there; Placement offsets are relative to it. Trunnion: `machine = B_axis + R_B · (c_axis + R_C · part − B_axis)` – the offset is applied BEFORE the B tilt (it rotates with B). B axis = `pivot_offset_*`.
- 3-axis / swivel head: `c_axis_*` is simply the machine position of the part origin on the bed.
- `MachineSpec.range_warnings(states)` compares all machine states (travel included) with `build_volume` (e.g. X 0..220, Klipper cannot go below the end stop); non-blocking, shown as "Warning (machine range)" in the GUI note and logged.
- G-code format is unchanged.
