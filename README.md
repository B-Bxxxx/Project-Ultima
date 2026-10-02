```text
██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗
██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝
██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║
██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║
██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║
╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝

██╗   ██╗██╗  ████████╗██╗███╗   ███╗ █████╗
██║   ██║██║  ╚══██╔══╝██║████╗ ████║██╔══██╗
██║   ██║██║     ██║   ██║██╔████╔██║███████║
╚██╗ ██╔╝██║     ██║   ██║██║╚██╔╝██║██╔══██║
 ╚████╔╝ ██║     ██║   ██║██║ ╚═╝ ██║██║  ██║
  ╚═══╝  ╚═╝     ╚═╝   ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝
```
# Modular 5-Axis Additive Manufacturing CAM Framework

## 1. Project Overview
This repository hosts an offline, modular Computer-Aided Manufacturing (CAM) and slicing framework designed for multi-axis (specifically 5-axis XYZBC) Material Extrusion (ME) 3D printing. 

Because target microcontroller architectures (such as modified 5-axis Klipper setups) lack the computational capacity for real-time Tool Center Point Control (TCPC) and dynamic coordinate transformations, this framework handles all spatial slicing, inverse kinematics (IK), pivot-offset compensations, and eccentricity corrections offline prior to G-code generation.

### Core Design Principles
* **Decoupled Architecture:** Slicing, toolpath generation, kinematic transformations, and G-code post-processing are strictly separated into a 4-stage pipeline.
* **Structural Interchangeability:** Layering algorithms, toolpath strategies, kinematic machine models, and post-processors can be swapped independently without altering the core pipeline.
* **Machine Agnosticism (Intermediate Representation):** Early stages generate universal Cutter Location Data (CL-Data) completely independent of the physical printer's kinematic chain.

---

## 2. System Pipeline & Data Contracts

The framework operates as a sequential 4-stage pipeline. Each stage communicates exclusively through standardized data structures (Data Contracts).

```text
[3D Model (.stl/.obj) / Parametric Math] 
       │
       ▼
┌──────────────────────────────┐
│ STAGE 1: Slicing Engine      │ ──► Output: Sliced Layers / Spatial Contours
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ STAGE 2: Toolpath Generator  │ ──► Output: Universal CL-Data (JSON/CSV/DataClass)
└──────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐     ┌──────────────────────────────┐
│ STAGE 3: Kinematic Solver    │ ◄── │ Machine Config (JSON/YAML)   │
└──────────────────────────────┘     └──────────────────────────────┘
       │                             (Kinematic chain, pivot offsets, limits)
       ▼
┌──────────────────────────────┐     ┌──────────────────────────────┐
│ STAGE 4: Post-Processor      │ ◄── │ Post-Processor Profile       │
└──────────────────────────────┘     └──────────────────────────────┘
       │
       ▼
[Executable Firmware G-Code (.gcode)]
