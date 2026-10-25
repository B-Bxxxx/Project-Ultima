# Kinematics: current state and the generic chain description

## Status

| Solver (`kinematics` in the machine file) | Machine | State |
|---|---|---|
| `planar_3axis_xyz` | XYZ printer | stable |
| `trunnion_table_xyzbc` | tilting (B) + rotating (C) table | stable, used for the real tests |
| `swivel_head_xyzbc` | tilting + rotating head | stable |
| `generic_chain` | **any** XYZ + up to two rotary axes, described in the machine file | **new, parallel test** - the three solvers above are untouched |

Rotary axis model of the system: **A about X, B about Y, C about Z** (fixed by the letter; `sign` flips the turning direction).
`generic_chain` supports A, B, C (at most TWO of them: 5 axis, or one = 4 axis); the hand-written solvers only B and C.
A third rotary joint (6 axis) is refused with an explicit message - in progress, not connected to Stage 4. Who-carries-whom
is the list order in `generic_chain`; the hand-written solvers have it fixed (table: B carries C; head: C carries B).

G-code: the letters written are the machine's rotary axes in the order A B C (`MinimalPrintProfile.rotary_axes`, set from the machine
file); a classic B+C machine writes exactly the same G-code as before. `MachineStateVector.a` is 0 for machines without A.

## The `chain` object (kinematics `generic_chain`)

```json
"kinematics": "generic_chain",
"axes": ["X", "Y", "Z", "B", "C"],
"chain": {
  "tool":  [ {"axis": "X", "type": "linear", "direction": [1, 0, 0]}, ... ],      // machine base -> nozzle, OUTERMOST FIRST
  "table": [ {"axis": "B", "type": "rotary", "origin": ["pivot_offset_x", "pivot_offset_z"],
              "sign": "b_axis_dir", "min": "b_axis_min", "max": "b_axis_max"},
             {"axis": "C", "type": "rotary", "origin": ["c_axis_x", "c_axis_y"],
              "sign": "c_axis_dir", "min": "c_axis_min", "max": "c_axis_max"} ],   // base -> part
  "part_origin": ["c_axis_x + eccentricity_x", "c_axis_y + eccentricity_y", "c_axis_z"],   // part (0,0,0) in the zero pose
  "tip": [0, 0, 0],            // nozzle tip in the zero pose
  "tool_axis": [0, 0, 1]       // nozzle direction in the zero pose
}
```

* **Zero pose**: all joints at 0, everything in the machine base frame - the axes are measured on the machine.
  A rotary axis' direction comes from its letter (a different `direction` is an error); its **origin** is TWO numbers, the
  coordinates of the axis line (A: y z, B: x z, C: x y; the third one is irrelevant). `point` (3D) is still accepted.
  Linear joints have a free `direction`.
* **Who carries whom** = list order, outermost first. A linear axis in `table` is a moving bed (Y carrying B and C); in
  `tool` it moves the nozzle. `"sign": -1` flips the direction of the machine value relative to the mathematical one.
* `min`/`max` are limits of the **machine value**. `continuous` (may wind around freely) follows from the range unless given:
  range >= 360 deg or no limits = can turn around (inside a move chain the angle stays continuous, at every travel move it is
  re-chosen closest to 0); a smaller range cannot, the angle is fitted into it and clamped with a warning if the normal needs more.
  The "keep the previous azimuth at zero tilt" rule applies only to a continuous axis that turns about the nozzle axis (C).
* Numbers may be expressions of the machine file's `parameters` (`+ - * /`, numbers, parameter names - nothing else).
  The parameters can have any name; they show up as input fields in the GUI as before.
* Rules: exactly one X, Y, Z linear joint (independent directions), at most two rotary joints, every axis once,
  `axes` must list exactly the chain's axes.

## Maths (per waypoint, `chain_solver.py`)

Product of exponentials: `T_branch = T_1(q_1) ... T_n(q_n)`. Orientation: `R_tool tool_axis = R_table N` (N = layer normal in the
part frame). Two rotary joints are solved in closed form (two planes `a1.v = alpha`, `a2.v = beta`, `|v| = 1`, two branches
- the one inside the limits wins, tie: positive angle of the limited joint); one rotary joint = best fit; none = fixed nozzle.
Then `tool(tip) = table(P + part_origin)` is linear in X, Y, Z (3x3 solve). `to_part_batch` is the exact inverse, so the
joint-space refinement (`solve_refined`) works unchanged.

## Parity with the hand-written solvers

`tests/test_generic_chain.py` solves the same trajectories (vertical, tilted, spiral, random tilts, travels, singular
tilt) with the old solver and with the chain description in `config/machines_experimental/chain_*.json (copy one into config/machines to see it in the GUI)` (variants: origin offsets,
pivot, eccentricity, `b_axis_dir`/`c_axis_dir` = -1): poses, tool alignment and inverse agree to < 2e-6.
One deliberate difference: with `b_axis_dir = -1` and one-sided limits the old solver clamps B to 0 (it always takes B >= 0 in the
program convention); the chain solver takes the other branch if it lies inside the limits.

## Not done (in progress)

Third rotary joint (6 axis); a GUI editor for the chain (edit the JSON); ranking
branches by travel distance instead of "positive limited joint"; firmware-side issues are in PITFALLS.md.

## 4 axis and unreachable normals

With one rotary joint (or limits that stop the nozzle short of the normal) the nozzle is not along the layer normal. Stage 3
reports the cosine in `tool_alignment`; with `thickness_model = nozzle_axis` Stage 4 divides the extruded volume by it
(floor `MIN_TOOL_ALIGNMENT`), so the deposited cross-section stays right. With the other thickness model the volume is NOT corrected.
Some structures are weak by design: a vertical-axis joint outside a tilt joint (C carrying A) cannot tilt the nozzle at all
(only normals with N_x = 0 are reachable).
