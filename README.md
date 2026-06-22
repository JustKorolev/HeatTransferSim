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

The new lumped thermal graph editor is separate from the original two-cube UI:

```powershell
python -m graph_visualizer.main
```

The existing two-cube UI is also available through the clearer compatibility
name:

```powershell
python -m heat_transfer_visualizer.main
```

`graph_visualizer` saves each graph as a folder containing `graph3d.json`,
`matrices.npz`, `metadata.json`, and `material_library.json`. Matrix rows and
columns follow the saved `node_ids` array. The pairwise conduction matrix `G`
stores conductances in W/K, zeros for non-edges, and the Laplacian is built as
`L[i, i] = sum_j G[i, j]`, `L[i, j] = -G[i, j]`.
