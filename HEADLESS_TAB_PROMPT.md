# Task: make the Headless Run tab identical to the Heat Transfer Simulation tab (minus the 3D viewer)

Repo: `C:\Users\andre\Documents\2025-2026 Academic Year\SURF\HeatTransferSim`

## Goal

The **Headless Run** tab must look and behave like the **Heat Transfer Simulation** tab.
Same control panel, same sections, **same order, same positions, same labels and
tooltips**. The only differences:

1. no 3D graph viewer / no graph is ever loaded into the GUI process, and
2. it launches the run headlessly (separate process) instead of playing it live.

There must be **one** simulation control panel implementation shared by both tabs —
not two that drift apart. Today the two panels are built by different code and look
nothing alike; that is the bug to fix.

## Current state (already done, do not redo)

* `graph_visualizer/headless_run_tab.py` — the Headless Run tab. It already:
  * lists graphs from `graphs/` **without loading them** (node count read from
    `node_ids.npy`'s header),
  * launches `run_simulation.py` as a **detached subprocess** (so the GUI holds no
    graph, stays responsive, and the run survives closing the app),
  * monitors the run by polling the run's own `status.json` and tailing
    `events.log`,
  * writes the chosen parameters to `simulation_parameters.json` in the run folder
    and passes them via `--sim-params`.
  * **Keep all of this behaviour.** Only the *layout/---panel composition* is wrong.
* `graph_visualizer/simulation_parameter_panel.py` — a first attempt at a shared
  panel. It covers the parameter sections but **not** the rest of the tab, and the
  simulation tab does **not** use it yet. Extend or replace it as needed.

## What the Heat Transfer Simulation tab looks like

`graph_visualizer/heat_transfer_simulation_tab.py`, `_build_layout()` builds a left
`QScrollArea` (`self.controls_scroll`, `setMinimumWidth(320)`) holding a
`QFormLayout`, filled **in this order**:

1. graph row (`graph` combo + `Refresh`), `Load Selected Graph`, `Use Current Editor Graph`
2. `self._add_parameter_controls(form)` — which adds, in order, the `QGroupBox`
   sections built by `self._section(title)`:
   * **Run** — `dt_s`, `t_final_s`, `playback speed`, `history limit`,
     `Loop playback`, `input mode` combo, `controller` combo, plus an `Initialize` button
   * **Environment** — `exterior / ambient T K`, `interior (cryo) T K`,
     `Use ambient radiation`, `Surface-to-surface radiative coupling (ray-traced)`,
     `initial T (all) K` + `Set all components`, `randomize setpoints` row
   * **Material Properties** — `Temperature-dependent cp(T)/k(T)`, `Copper RRR`,
     `Midpoint property/radiation coupling`
   * **Cryocooler** — `Model` label, `Maximum cooling power W`, `Capacity scale`, `Enabled`
   * **Controller (global limits)** — `max heater power W`, `hard slew W/s`, `max rate cmd K/s`
   * **Modal LQR Design** (`_build_modal_design_controls`) — operating T, slow modes,
     reduced order r, LQR effort weight, integral gain, adaptive feedforward group,
     `Build & Use Modal Control` button, `status`
   * **MIMO Thermal-Rate QP** — 11 double fields + `Freeze integral when saturated`
   * **Display** — autoscale, color min/max K, colormap
3. `self._add_playback_controls(form)` — Play/Pause/Reset/Step+/Step-, time slider,
   `Save / Export Trajectory`, `Reset MIMO Integrators`, and the existing
   `Run Headless (save, no viz)` / `Stop Headless Run` buttons
4. `self._add_enabled_io_controls(form)` — **Enabled Simulation I/O** table
5. `self._add_sys_id_controls(form)` — **Simulation Sys ID for Controller Gain Matrix**
6. `self._add_stepper_diagnostic_controls(form)` — **Solver Diagnostic**
7. `self._add_component_temperature_controls(form)`
8. warning label, stats label, controller status label, sensor readout box, legend

The right-hand side is the viewer panel (`self.viewer.interactor`) plus the view
toggles row (Heaters/Sensors/Cryocoolers/Opacity/Depth/Cross section).

## What to build

1. **Extract the whole control panel into one shared class**, e.g.
   `graph_visualizer/simulation_controls_panel.py`, that builds sections 2–8 above
   (everything except the graph-loading row and the viewer). Give it:
   * `build(form)` — add every section in the same order,
   * `set_params(params)` / `read(base)` — populate from and read back into
     `SimulationParameters`, using `dataclasses.replace(base, ...)` so fields with
     no widget keep their saved values,
   * a `mode` or feature-flag mechanism so a section that is meaningless without a
     loaded graph can be hidden or disabled rather than duplicated (see point 3).
2. **Make BOTH tabs use it.** `HeatTransferSimulationTab` must keep working exactly
   as it does now — it is the primary workflow. Its existing per-widget behaviour
   (`self.inputs[...]`, `_handle_parameter_change`, `_read_params`, marking the
   controller stale, deferred re-initialise) must be preserved; wire it through a
   callback the panel invokes on change.
3. **Headless differences** — in the headless tab:
   * hide/disable: `playback speed`, `history limit`, `Loop playback`, the whole
     **Display** section, Play/Pause/Step/time-slider, and anything needing a loaded
     model (`Set all components`, `randomize setpoints`, Enabled Simulation I/O
     table, Sys ID, Solver Diagnostic, component temperatures, sensor readout).
     Prefer `setVisible(False)` on the same widgets over building a different layout,
     so positions stay identical.
   * keep instead: graph picker (no load), controller-artifact picker, `setpoint K`,
     `initial T K`, snapshot/checkpoint intervals, `Start Headless Run` / `Stop Run`
     / `Open Output Folder`, progress bar, status line, live log view.
   * a **Solver** section (implicit method, rtol, maxiter, adaptive substeps,
     residual check, GPU) is wanted here even though the live tab does not show it.

## Constraints

* **PySide6 is NOT installed in this environment**, so the GUI cannot be launched or
  screenshotted here. Verify by: `python -c "import ast; ast.parse(open(F).read())"`
  on every edited file, plus a Qt-stub unit test that builds the panel and
  round-trips `set_params`/`read` (see the stub pattern used previously: distinct
  stub classes per widget type, or `isinstance` checks collapse).
* Run `python -m pytest tests/ -q -k "not graph_visualizer"` — **199 tests must
  pass**. (`tests/test_graph_visualizer.py` fails in this env only because Qt is
  missing; 17 known failures.)
* Use the `heatsim` conda env. In git-bash:
  `export PATH="/c/Users/andre/miniconda3/envs/heatsim:/c/Users/andre/miniconda3/envs/heatsim/Library/bin:/c/Users/andre/miniconda3/envs/heatsim/Scripts:$PATH"`
  and `export PYTHONPATH="<repo root>"`.
* Do **not** change the physics/runner behaviour; this is a UI-composition task.
* Do not regress the memory work: the headless tab must never load a graph into the
  GUI process.

## Definition of done

Opening the two tabs side by side, the control panel reads the same top to bottom —
same section titles in the same order, same field labels, same positions — with the
headless tab simply missing the viewer and the graph-dependent/playback controls,
and carrying the run/monitor controls instead. One shared panel class; no duplicated
section-building code.
