# HeatTransferSim

Simple control-oriented thermal RC model for two solid cubes exchanging heat
through one thermal contact resistance.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run the example

```powershell
python main.py
```

Run without opening plots:

```powershell
python main.py --no-plot
```

## Launch the interactive UI

```powershell
python main.py --ui
```

The UI uses PyVista/Qt for CAD-like orbit, pan, and zoom around two stationary
cubes. The control panel edits cube, contact, heater, and simulation parameters.
Set `contact area override, 0 = auto [m^2]` to zero to compute contact area from
the overlap of touching cube faces.
Cube position fields are minimum-corner coordinates in meters. For example, a
`0.1 m` cube with `min corner x = -0.1` spans `x = -0.1` to `x = 0.0`.

UI parameters are persisted in `simulation_parameters.json` at the project root.
Changing a parameter in the UI updates that file, and the next launch restores
the saved values.

Live mode has separate display and solver timing controls:

- `simulated seconds per display update [s]`: how far the simulation advances
  each time the 3D view refreshes.
- `display update interval [ms]`: real wall-clock time between screen updates.
- `max solver internal step [s]`: maximum adaptive ODE step used inside each
  display update for accuracy.

## Model

```text
C1 dT1/dt = -(T1 - T2) / R12 + P1(t)
C2 dT2/dt =  (T1 - T2) / R12 + P2(t)
R12 = (s1 / 2) / (k1 A) + R_interface + (s2 / 2) / (k2 A)
C_i = m_i cp_i
```

Positive heat flow `Qdot_1_to_2 = (T1 - T2) / R12` means heat leaves cube 1 and
enters cube 2.

`extra interface resistance` is `R_interface`, an optional resistance for
imperfect face-to-face contact. Set it to `0` for ideal contact where the only
resistance is conduction from each cube center to its contacting face.

## Launch the sparse graph visualizer

The lumped thermal graph inspector/editor is separate from the original two-cube UI:

```powershell
python -m graph_visualizer.main
```

The existing two-cube UI is also available through the clearer compatibility
name:

```powershell
python -m heat_transfer_visualizer.main
```

The visualizer can load legacy `graph3d.json` folders and new octree
`graph.json` folders. In octree mode, geometry/topology are read-only:
select cells in the 3D cuboid view or 2D network view, then edit heater/sensor
tags and notes. Autosave writes tag changes back to `graph.json` and
`ui_state.json`.

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
  --samples-per-cell 9 `
  --voxel-workers 0
```

The converter assumes glTF/GLB coordinates are millimeters, finds the single
embedded `.glb` file in `--mesh-dir`, uses glTF material names from that scene,
and reads material properties from the project-level `materials.json` file by
default. The mesh directory must contain exactly one `.glb` scene file.
External-buffer `.gltf`/`.bin` exports are rejected because missing or mismatched
buffers can collapse CAD geometry during loading.
If `materials.xlsx` exists in `--mesh-dir`, it maps SolidWorks part instance
names to material names. Contact checking is handled separately in Python by
exact shared voxel faces plus a voxel-surface contact-distance pass.
`--voxel-workers` enables multiprocessing for octree cell classification:
`1` is sequential, `0` uses conservative auto-selection capped at 2 worker
processes, and an explicit integer uses that many workers. Large CAD assemblies
copy triangle data into each worker process, so increase this gradually if
memory pressure is high. `--voxel-batch-size` controls how many queued octree
cells are classified per worker batch.
For bbox-fallback graphs, `--contact-detection-distance-mm` defaults to
`--min-cell-size-mm` so near but non-face-adjacent cells can still be connected.
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
  C.npy
  G.npy
  L.npy
  A.npy
  ui_state.json
```

Matrix rows and columns follow `node_ids`. `G` stores symmetric conductances in
W/K, zeros for non-edges, and the Laplacian is built as
`L[i, i] = sum_j G[i, j]`, `L[i, j] = -G[i, j]`. The optional dynamics matrix is
`A = -C^{-1}L`.

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
