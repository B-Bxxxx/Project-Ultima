# Kinematics: current state and the generic chain description

## Status

| Solver (`kinematics` in the machine file) | Machine | State |
|---|---|---|
| `planar_3axis_xyz` | XYZ printer | stable |
| `trunnion_table_xyzbc` | tilting (B) + rotating (C) table | stable, used for the real tests |
| `swivel_head_xyzbc` | tilting + rotating head | stable |
| `generic_chain` | **any** XYZ + up to two rotary axes, described in the machine file | **new, parallel test** - the three solvers above are untouched |

Rotary axis model of the system: **A about X, B about Y, C about Z**. Only B and C exist in the G-code today; A is
reserved (the `axis` name is rejected with a clear message) and a third rotary joint (6 axis, "C only theoretical") is
in progress. Who-carries-whom between B and C may be described in `generic_chain`; the hand-written solvers have it fixed
(table: B carries C; head: C carries B).

## The `chain` object (kinematics `generic_chain`)

```json
"kinematics": "generic_chain",
"axes": ["X", "Y", "Z", "B", "C"],
"chain": {
  "tool":  [ {"axis": "X", "type": "linear", "direction": [1, 0, 0]}, ... ],      // machine base -> nozzle, OUTERMOST FIRST
  "table": [ {"axis": "B", "type": "rotary", "direction": [0, 1, 0],
              "point": ["pivot_offset_x", "pivot_offset_y", "pivot_offset_z"],
              "sign": "b_axis_dir", "min": "b_axis_min", "max": "b_axis_max"},
             {"axis": "C", "type": "rotary", "direction": [0, 0, 1], "point": ["c_axis_x", "c_axis_y", "c_axis_z"],
              "sign": "c_axis_dir", "min": "c_axis_min", "max": "c_axis_max", "continuous": true} ],   // base -> part
  "part_origin": ["c_axis_x + eccentricity_x", "c_axis_y + eccentricity_y", "c_axis_z"],   // part (0,0,0) in the zero pose
  "tip": [0, 0, 0],            // nozzle tip in the zero pose
  "tool_axis": [0, 0, 1]       // nozzle direction in the zero pose
}
```

* **Zero pose**: all joints at 0. `direction` (right-handed axis / translation direction) and `point` (a point on a rotation
  axis) are given in the machine base frame in that pose - the axes are measured on the machine, not derived from formulas.
* **Who carries whom** = list order, outermost first. A linear axis in `table` is a moving bed (Y carrying B and C); in
  `tool` it moves the nozzle. `"sign": -1` flips the direction of the machine value relative to the mathematical one.
* `min`/`max` are limits of the **machine value**; `continuous` marks an axis that may wind around freely (C): inside
  a move chain it stays continuous, at every travel move it is re-chosen closest to 0 (same rule as the other solvers).
* Numbers may be expressions of the machine file's `parameters` (`+ - * /`, numbers, parameter names - nothing else).
  The parameters can have any name; they show up as input fields in the GUI as before.
* Rules: exactly one X, Y, Z linear joint (independent directions), at most two rotary joints (not parallel), every axis once,
  `axes` must list exactly the chain's axes.

## Maths (per waypoint, `chain_solver.py`)

Product of exponentials: `T_branch = T_1(q_1) ... T_n(q_n)`. Orientation: `R_tool tool_axis = R_table N` (N = layer normal in the
part frame). Two rotary joints are solved in closed form (two planes `a1.v = alpha`, `a2.v = beta`, `|v| = 1`, two branches
- the one inside the limits wins, tie: positive angle of the limited joint); one rotary joint = best fit; none = fixed nozzle.
Then `tool(tip) = table(P + part_origin)` is linear in X, Y, Z (3x3 solve). `to_part_batch` is the exact inverse, so the
joint-space refinement (`solve_refined`) works unchanged.

## Parity with the hand-written solvers

`tests/test_generic_chain.py` solves the same trajectories (vertical, tilted, spiral, random tilts, travels, singular
tilt) with the old solver and with the chain description in `config/machines/chain_*.json` (variants: origin offsets,
pivot, eccentricity, `b_axis_dir`/`c_axis_dir` = -1): poses, tool alignment and inverse agree to < 2e-6.
One deliberate difference: with `b_axis_dir = -1` and one-sided limits the old solver clamps B to 0 (it always takes B >= 0 in the
program convention); the chain solver takes the other branch if it lies inside the limits.

## Not done (in progress)

A axis in the G-code / `MachineStateVector`; third rotary joint (6 axis); a GUI editor for the chain (edit the JSON); ranking
branches by travel distance instead of "positive limited joint"; firmware-side issues are in PITFALLS.md.
