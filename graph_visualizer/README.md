# graph_visualizer

Run the sparse 3D lumped thermal graph editor with:

```powershell
python -m graph_visualizer.main
```

The original two-cube heat-transfer UI remains available as:

```powershell
python -m heat_transfer_visualizer.main
```

## Visualization Tabs

The main viewer has two tabs:

- `3D View`: the PyVista octree/cell scene with cuboid cell rendering,
  selection, labels, edges, heater markers, and sensor markers.
- `2D View`: a read-only flattened adjacency graph for inspecting nodes and
  conductive links. It does not edit graph topology or node/edge properties.

The 2D view uses a spring layout by default. The layout dropdown can also show
coordinate projections: `XY`, `XZ`, and `YZ`, where coordinates come from each
node's sparse `(i, j, k)` grid coordinate. `Refresh 2D Layout` recomputes and
redraws the 2D graph.

Hovering over a node in either view shows a compact node summary: cell ID,
center/size in millimeters, component, material, level, volume, mass, `C`,
heater status, and sensor status. Heater and sensor details are included when
present. Hovering over a 2D edge shows its conductance `Gij_W_K`, edge type,
area, distance, and source/mode.

## Graph Folder Structure

Legacy graphs are saved as folders:

```text
thermal_graphs/
  my_graph_name/
    graph3d.json
    matrices.npz
    metadata.json
    material_library.json
```

In `Save As`, choose the parent folder. The app creates a child graph folder
using the current Graph Settings name.

Autosave is enabled only after `Save As` or after loading an existing graph
folder. Once enabled, graph changes schedule a debounced autosave of the full
graph folder: `graph3d.json`, `matrices.npz`, `metadata.json`, and
`material_library.json`. Before a save folder exists, the app shows an unsaved
graph status instead of writing files.

`graph3d.json` stores node and edge metadata in a NetworkX-compatible shape:
nodes have an `id` field plus node attributes, and edges have `source`,
`target`, and `Gij_W_K`.

Octree conversion folders use the newer layout:

```text
graphs/
  my_graph_name/
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

In this mode the visualizer is an inspection and tagging tool. Geometry and
topology fields are read-only; selecting a leaf cell allows editing heater,
sensor, ID, and notes metadata only. The filters panel can restrict the 3D and
2D views by material, component, octree level, heater/sensor tags, or
contact/boundary cells.

`metadata.json` stores graph-level settings such as `T_sur_K`, edge mode,
timestamps, graph name, and notes.

`material_library.json` stores the material defaults used by the graph.

## Matrix Conventions

`matrices.npz` includes:

```text
node_ids
coords
C
Grad
G
L
```

`node_ids` is the single source of truth for row and column ordering. By
default, saves use numerically sorted node IDs.

`G` is a symmetric pairwise conduction conductance matrix in W/K. `G[i, j] > 0`
means a conductive link exists. `G[i, j] = 0` means no link. Non-edges are zero,
not infinity.

`Grad` is different from `G`: `Grad` is the per-node linearized radiative
conductance to surroundings. `C` is the per-node thermal capacitance.

`L` is the graph Laplacian:

```text
L[i, i] = sum_j G[i, j]
L[i, j] = -G[i, j] for i != j
```

## Mutually Exclusive Conduction Modes

The UI exposes a `Conduction Model` radio-button group. Exactly one mode is
active at a time:

- `Auto-estimate G from geometry/materials`
- `Load/use G from matrices.npz`

Auto-estimated conductance mode connects only 6-neighbor face-adjacent cells:


```python
abs(i1 - i2) + abs(j1 - j2) + abs(k1 - k2) == 1
```

Diagonal, edge-touching, and corner-touching cells are not connected.

Loaded-G mode reads `G` and `node_ids` from the current graph folder's
`matrices.npz`. The loaded `node_ids` must match the current graph node IDs,
`G` must be square, symmetric within tolerance, and nonnegative.

Topology edits invalidate loaded `G`. Adding, deleting, renumbering, or moving
nodes switches the graph back to auto-estimated mode, regenerates face-adjacent
edges, and rebuilds `G` and `L`.

## Octree Graph Conversion

Use the project-level wrapper to convert a SolidWorks-exported embedded GLB by
pointing at the one directory containing the `.glb` file:

```powershell
python build_octree_graph.py `
  --mesh-dir meshes\assembly_export `
  --graph-name hispec_test_octree `
  --output-root graphs
```

The converter preserves millimeter coordinates for visualization, computes
thermal quantities in SI units, generates `C`, `G`, `L`, and `A`, and writes a
validation report with unknown materials, non-watertight mesh warnings, matrix
checks, and tolerance-accepted boundary cells. Material properties are read from
the project-level `materials.json` by default. The mesh directory must contain
exactly one embedded `.glb` scene file. External-buffer `.gltf`/`.bin` exports
are rejected for octree conversion.
If `materials.xlsx` exists in the mesh directory, it is used as a two-column
Excel lookup for part names and material names. Contact checking is handled in
Python by exact shared voxel faces plus a voxel-surface contact-distance pass.
Heater and sensor CAD components are converted into dedicated role nodes only
when you provide matches with `--heater-name-substring`,
`--heater-name-pattern`, `--sensor-name-substring`, or
`--sensor-name-pattern`.
Use `--voxel-workers 0` for conservative multiprocessing capped at 2 worker
processes during octree classification, or pass an explicit worker count. Large
CAD assemblies copy triangle data into each worker process, so increase the
count gradually. Conversion runs write `conversion.log` in the graph output
folder with phase changes, progress, memory estimates, and Python tracebacks.
