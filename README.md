# HeatTransferSim

A control-oriented thermal simulator for cryogenic instrument assemblies. It
turns a CAD assembly into a lumped thermal graph, simulates it closed-loop
against a MIMO controller, and exports that controller as constants you can
flash.

Three subsystems, in the order a graph moves through them:

1. `octree_graph/` -- a STEP assembly to an octree to a lumped RC graph.
   Writes `graphs/<name>/` with `graph.json` plus the `C`, `L` and `G_rad`
   matrices.
2. `graph_visualizer/` -- the application: inspect and edit the graph in 3D,
   run it live or headless, validate it against analytical solutions, and
   design and export the controller.
3. `hts-export-controller` -- the controller as a C header plus a JSON twin.

## Install

Input is **STEP** (`.step` / `.stp`), which needs OpenCASCADE. OpenCASCADE has no
pip wheel on any platform, so conda is the install that gets you the whole
pipeline:

```powershell
git clone https://github.com/JustKorolev/HeatTransferSim
cd HeatTransferSim
conda env create -f environment.yml
conda activate heatsim
python -m pip install -e .
```

There is a lighter pip install, which needs Python 3.11 or newer:

```powershell
python -m pip install git+https://github.com/JustKorolev/HeatTransferSim
```

It can simulate existing graphs, run the validation cases and export
controllers. It **cannot build a graph from CAD**, because that is the part that
needs OpenCASCADE. The builder says so plainly rather than failing obscurely.

`requirements.txt` is not how you install this either way. It pins exact versions
for reproducing a particular run; the installs above resolve against whatever
else is in your environment.

### Why STEP only

A mesh export (GLB/glTF) carries surfaces, not solids, so the voxelizer shells a
part instead of filling it. `CRYOSTAT_V2` was built from a mesh, and its DC gain
is a rank-1 ~1e9 K/W artifact of those shells -- a number that looks like a plant
model and is not one. STEP carries the B-rep solids, which fill.

## Launch the application

```powershell
heattransfersim
```

From a clone without installing, the equivalent is
`python -m graph_visualizer.main`.

`Ctrl+C` in the launching terminal closes the window. A second `Ctrl+C` exits
immediately without waiting for anything to shut down. Detached headless runs are
not affected either way -- they are launched into their own process group so that
closing the app, or losing the terminal, does not take a multi-hour run with it.

Five tabs: `3D Octree Graph Editor`, `2D Network Graph`,
`Heat Transfer Simulation`, `Thermal Validation`, and `Headless Run`.

The visualizer loads octree `graph.json` folders and legacy `graph3d.json`
folders. In octree mode, geometry and topology are read-only: select cells in
the 3D cuboid view or the 2D network view, then edit heater/sensor tags,
materials and notes. Autosave writes changes back to `graph.json` and
`ui_state.json`.

## Run a simulation without the UI

```powershell
hts-run --graph graphs/CRYOSTAT_V2 --setpoint 80 --duration 3600 --dt 1
```

Each run writes a timestamped folder under `simulations/<graph>/` holding the
series, plots, `status.json`, `events.log` and periodic checkpoints. Runs are
not tracked in git.

## Export the controller

```powershell
hts-export-controller --graph graphs/CRYOSTAT_V2
```

Writes `controller_constants.h` and `controller_constants.json`: the DC gain
`G`, its regularized inverse, per-sensor Kp/Ki and setpoints, per-heater power
and slew limits, and every loop-shaping constant, with provenance naming the
gain matrix they came from. `--list` shows the available gain matrices.

The same export is on the `Export Controller Constants` button in both
simulation tabs. It needs a gain matrix: the decoupling lives in `G`, so Kp and
Ki alone do not define this controller.

## Build a thermal graph from a STEP assembly

The easiest way is the **Build Graph** tab, which is the first tab in the
application: pick an assembly folder, adjust the parameters, press Build. It
shows the exact command it will run and tails the builder's log.

A sample assembly ships with the repository, so there is something to try
immediately:

```text
meshes/step_test/HISPEC_FEA.STEP     30 named solids, 2.9 MB
meshes/step_test/Materials.xlsx      part name -> material
```

From a shell, the same thing. These are the settings that produced the reference
graph:

```powershell
hts-build-graph `
  --mesh-dir meshes\step_test `
  --graph-name STEP_SAMPLE `
  --output-root graphs `
  --min-cell-size-mm 10 `
  --max-cell-size-mm 20 `
  --max-depth 10 `
  --samples-per-cell 27 `
  --step-deflection-mm 1.5 `
  --voxel-workers 8 `
  --voxel-worker-memory-fraction 0.8 `
  --low-k-refine-threshold-w-mk 10 `
  --no-boundary-refine `
  --contact-detection-distance-mm 2 `
  --contact-gap-tolerance-mm 0.2 `
  --heater-name-substring SAFE-HEATER `
  --sensor-name-substring COO-0001-P0003 `
  --max-heater-sensor-pair-distance-mm 50 `
  --max-heaters-per-sensor 2 `
  --role-refine-distance-mm 20 `
  --role-refine-max-depth 10 `
  --max-leaf-cells 2000000
```

`--mesh-dir` must contain exactly one `.step`/`.stp` file. Coordinates are
assumed to be millimetres. Material properties come from the table shipped with
the package (`graph_visualizer/data/materials.json`), or from a `materials.json`
in the working directory if there is one.

**A build is memory-hungry.** The voxelizer copies triangle data into every
worker process, so `--voxel-workers` and `--max-leaf-cells` are the two settings
that decide whether the machine survives a full assembly. Start conservative.

### Heaters and sensors

Components become heaters or sensors only when you name them with
`--heater-name-substring` or `--sensor-name-substring`; repeat either flag for
several matches. Matching is by **substring**, not pattern. Without them the
graph has no heaters and no sensors, so there is nothing to control.

Matched components stay in voxelization, so their occupied cells first get normal
graph connections; cells belonging to one detected part are then consolidated
into a single role node carrying the union of those connections.

### The part -> material lookup

Run `tools/ExportAssemblyMaterialsToExcel.bas` from SolidWorks with the assembly
open to produce a two-column workbook, and save it as `Materials.xlsx` beside the
STEP file:

- `Part Name`: the component instance name.
- `Material Name`: the material assigned to that part.

Without it there is nothing to identify engineering materials by, and every part
falls back to the unassigned default -- the conductances will not describe your
assembly. `meshes/step_test/Materials.xlsx` is a worked example.

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

