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

- `3D View`: the editable PyVista cube scene with selection, Draw Mode,
  labels, edges, heater markers, and sensor markers.
- `2D View`: a read-only flattened adjacency graph for inspecting nodes and
  conductive links. It does not edit graph topology or node/edge properties.

The 2D view uses a spring layout by default. The layout dropdown can also show
coordinate projections: `XY`, `XZ`, and `YZ`, where coordinates come from each
node's sparse `(i, j, k)` grid coordinate. `Refresh 2D Layout` recomputes and
redraws the 2D graph.

Hovering over a node in either view shows a compact node summary: coordinate,
material, mass, `C`, `Grad`, heater status, and sensor status. Heater and
sensor details are included when present. Hovering over a 2D edge shows its
conductance `Gij_W_K` and source/mode.

## Graph Folder Structure

Graphs are saved as folders:

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

## Draw Mode

`Draw Mode` is a toolbar toggle for straight face-normal extrusion.

1. Enable `Draw Mode`.
2. Click a face of an existing cube cell.
3. Drag to preview transparent ghost cells.
4. Click again to commit the preview.

The extrusion direction comes from the clicked cube face normal, not from
screen-space direction. For example, clicking the `+x` face of `(i, j, k)`
creates `(i+1, j, k)`, `(i+2, j, k)`, and so on.

Preview cells stop at the first occupied coordinate. If the immediately adjacent
cell is occupied, no new cells are created and the UI shows a non-fatal message.
Press `Escape` or disable `Draw Mode` to cancel a preview.

New draw-created nodes receive IDs starting at `max(existing_node_ids) + 1`, or
`0` for an empty graph. They copy material, dimension, mass, and heat-capacity
fields from the source cell. Heater, sensor, and radiation fields reset:
`Grad_W_K = 0`, no heater, no sensor, and heater/sensor IDs default to the new
node ID.
