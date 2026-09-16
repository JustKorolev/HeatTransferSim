# HeatTransferSim

A control-oriented thermal simulator for cryogenic instrument assemblies. It
turns a CAD assembly into a lumped thermal graph, simulates it closed-loop
against a MIMO controller, and exports that controller as constants you can
flash.

Three subsystems, in the order a graph moves through them:

1. `octree_graph/` -- CAD (GLB or STEP) to an octree to a lumped RC graph.
   Writes `graphs/<name>/` with `graph.json` plus the `C`, `L` and `G_rad`
   matrices.
2. `graph_visualizer/` -- the application: inspect and edit the graph in 3D,
   run it live or headless, validate it against analytical solutions, and
   design and export the controller.
3. `export_controller.py` -- the controller as a C header plus a JSON twin.

## Install

```powershell
python -m pip install -r requirements.txt
```

The STEP/B-rep pipeline needs OpenCASCADE, which has no pip wheel. For that,
use the conda environment instead:

```powershell
conda env create -f environment.yml
conda activate heatsim
```

## Launch the application

```powershell
python -m graph_visualizer.main
```

Five tabs: `3D Octree Graph Editor`, `2D Network Graph`,
`Heat Transfer Simulation`, `Thermal Validation`, and `Headless Run`.

The visualizer loads octree `graph.json` folders and legacy `graph3d.json`
folders. In octree mode, geometry and topology are read-only: select cells in
the 3D cuboid view or the 2D network view, then edit heater/sensor tags,
materials and notes. Autosave writes changes back to `graph.json` and
`ui_state.json`.

## Run a simulation without the UI

```powershell
python run_simulation.py --graph graphs/CRYOSTAT_V2 --setpoint 80 --duration 3600 --dt 1
```

Each run writes a timestamped folder under `simulations/<graph>/` holding the
series, plots, `status.json`, `events.log` and periodic checkpoints. Runs are
not tracked in git.

## Export the controller

```powershell
python export_controller.py --graph graphs/CRYOSTAT_V2
```

Writes `controller_constants.h` and `controller_constants.json`: the DC gain
`G`, its regularized inverse, per-sensor Kp/Ki and setpoints, per-heater power
and slew limits, and every loop-shaping constant, with provenance naming the
gain matrix they came from. `--list` shows the available gain matrices.

The same export is on the `Export Controller Constants` button in both
simulation tabs. It needs a gain matrix: the decoupling lives in `G`, so Kp and
Ki alone do not define this controller.

## Build an octree graph from SolidWorks GLB exports

```powershell
python build_octree_graph.py `
  --mesh-dir meshes\assembly_export `
  --graph-name hispec_test_octree `
  --output-root graphs `
  --min-cell-size-mm 5 `
  --max-cell-size-mm 50 `
  --max-depth 8 `
  --dominant-fraction-accept 0.95 `
  --minority-fraction-ignore 0.02 `
  --material-contrast-refine-threshold 5 `
  --contact-refine-distance-mm 10 `
  --crowded-component-refine-count 3 `
  --crowded-component-refine-distance-mm 2 `
  --samples-per-cell 9 `
  --voxel-workers 0
```

The converter assumes glTF/GLB coordinates are millimeters, finds the single
embedded `.glb` file in `--mesh-dir`, uses glTF material names from that scene,
and reads material properties from the project-level `materials.json` file by
default. The mesh directory must contain exactly one `.glb` scene file.
External-buffer `.gltf`/`.bin` exports are rejected because missing or mismatched
buffers can collapse CAD geometry during loading.
CAD components are recognized as heater/sensor geometry only when you provide
matching names with `--heater-name-substring` or `--sensor-name-substring`;
repeat a flag to add multiple matches. (The `--heater-name-pattern` and
`--sensor-name-pattern` regex flags are accepted for backward compatibility but
currently ignored — role detection uses substring matching only.) Matched components remain in voxelization so their occupied
octree cells first receive normal graph connections, then cells from the same
detected heater/sensor part are consolidated into one role node with the union
of those external connections. If no heater or sensor match is configured, no
cells are assigned those roles.
If `materials.xlsx` exists in `--mesh-dir`, it maps SolidWorks part instance
names to material names. Contact checking is handled separately in Python by
exact shared voxel faces plus a voxel-surface contact-distance pass.
`--voxel-workers` enables multiprocessing for octree cell classification:
`1` is sequential, `0` uses conservative auto-selection capped at 2 worker
processes, and an explicit integer uses that many workers. Large CAD assemblies
copy triangle data into each worker process, so increase this gradually if
memory pressure is high. `--voxel-batch-size` controls how many queued octree
cells are classified per worker batch.
Each run writes `conversion.log` in the graph output folder with phase changes,
periodic voxelization progress, memory estimates, and Python tracebacks. If a
run exits without a terminal error, inspect that log first.
The `Voxelizing octree` phase depends on the mesh geometry, any configured
heater/sensor component exclusions, material lookup used for material-contrast
refinement, and octree/refinement parameters such as cell sizes, depth,
sampling, and boundary/contact refinement distance. It does not depend on
graph-only settings such as `--contact-detection-distance-mm` or radiation
reference temperature.
For dense regions with many small nearby parts, use
`--crowded-component-refine-count N` with
`--crowded-component-refine-distance-mm D` to force additional local refinement
where a cell's padded bounds overlap at least `N` CAD components. This helps
preserve small air gaps between nearby parts; keep `--max-leaf-cells` high
enough for the extra local cells.
The builder writes:

```text
graphs/<graph_name>/
  graph.json
  nodes.csv
  edges.csv
  params.json
  materials_used.json
  material_warnings.csv
  validation_report.txt
  conversion.log
  node_ids.npy
  C.npy
  L.npy
  G_rad.npy
  initial_temperature_K.npy
  G.npy            # dense conductance matrix, only for small graphs
  C_diag.json      # browser matrix exports
  G_rad_diag.json
  L_sparse.json
  ui_state.json
```

Matrix rows and columns follow `node_ids`. `C` is the per-node thermal
capacitance vector and `G_rad` the per-node linearized radiation conductance.
The Laplacian is built as `L[i, i] = sum_j G[i, j]`, `L[i, j] = -G[i, j]`. For
small graphs (at or below `--dense-matrix-node-limit`, default 6000 nodes) a
dense symmetric conductance matrix `G` is also written, where `G` stores
conductances in W/K with zeros for non-edges; larger graphs write only the
sparse Laplacian. No `A = -C^{-1}L` dynamics matrix is written — the simulator
forms that operator internally from `C` and `L` at run time.

### Export SolidWorks materials for octree lookup

Run `tools/ExportAssemblyMaterialsToExcel.bas` from SolidWorks with the assembly
open to create a two-column workbook. Save it as `materials.xlsx` in the same
folder as the exported `.glb` mesh:

- `Part Name`: SolidWorks component instance name.
- `Material Name`: SolidWorks material assigned to that part/configuration.

Use the generated workbook during graph construction:

```powershell
python build_octree_graph.py `
  --mesh-dir meshes\assembly_export `
  --graph-name hispec_test_octree `
  --output-root graphs
```
