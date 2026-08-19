"""Headless run tab: configure and launch a simulation without loading the graph.

The other tabs hold the whole ThermalGraphModel in memory to draw it. For a
multi-million-cell overnight run that is the problem, not a feature: the model is
tens of GB, and running the simulation in the GUI process meant a second copy plus
a Python-level load that blocks the Qt event loop.

This tab never loads the graph. It only lists the folders under ``graphs/``,
collects run parameters, and launches ``run_simulation.py`` as a SEPARATE PROCESS.
Consequences that matter for overnight runs:

* the GUI holds none of the graph -- memory is the run's alone,
* the window stays responsive; nothing shares this process's GIL,
* the run survives closing the GUI (it is detached, not a child thread),
* progress is read from the run's own ``status.json``, so the same view works
  whether the run was started here or from the command line.

The controls come from :class:`SimulationControlsPanel`, the same class the Heat
Transfer Simulation tab uses, so the two tabs offer the same sections, labels and
tooltips in the same order. This tab only adds the graph picker at the top and,
in place of the 3D viewer, the progress / status / log view on the right.
"""

from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .diagnostics import log_event
from .graph_roles import load_role_manifest
from .modal_reduction import list_modal_artifacts
from .simulation_controls_panel import MODE_HEADLESS, PID_QP_LABEL, SimulationControlsPanel
from .simulation_runner import STOP_REQUEST_FILENAME

# What every sensor's target starts at. Matches NodeProperties' own default, so a
# prefilled table asks for exactly what an untouched graph would have run anyway.
DEFAULT_SETPOINT_K = 293.15


class HeadlessRunTab:
    """Configure + launch + monitor a headless simulation (no graph in memory)."""

    def __init__(
        self,
        qt: Any,
        parent: Any,
        *,
        on_status: Callable[[str, bool], None] | None = None,
        graphs_root: Callable[[], Path] | None = None,
    ) -> None:
        self._qt = qt
        self.QtCore = qt.QtCore
        self.QtWidgets = qt.QtWidgets
        self.on_status = on_status
        self._graphs_root = graphs_root or (lambda: Path.cwd() / "graphs")
        self.process: subprocess.Popen | None = None
        self.run_dir: Path | None = None
        self._stop_requested = False
        self._refresh_process: subprocess.Popen | None = None
        self._refresh_folder: Path | None = None
        self._modal_build_process: subprocess.Popen | None = None
        self._modal_build_folder: Path | None = None
        self._gain_build_process: subprocess.Popen | None = None
        self._gain_build_folder: Path | None = None
        self._log_size = 0
        self._params_source = "defaults"
        self._pending_log_note = ""
        self.widget = self.QtWidgets.QWidget(parent)
        self._build_layout()
        self.refresh_graphs()
        self._poll_timer = self.QtCore.QTimer(self.widget)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start(1000)

    # -- layout ------------------------------------------------------------- #
    def _build_layout(self) -> None:
        outer = self.QtWidgets.QHBoxLayout(self.widget)
        # Like the simulation tab, the controls live in the window's left side
        # panel (app.py puts controls_scroll in side_panel_stack) and the tab body
        # holds only the right-hand view. Same minimum width, so the two panels are
        # the same size and their rows line up when the tabs are compared.
        self.controls_scroll = self.QtWidgets.QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setMinimumWidth(320)
        controls = self.QtWidgets.QWidget()
        self.controls_scroll.setWidget(controls)
        form = self.QtWidgets.QFormLayout(controls)

        intro = self.QtWidgets.QLabel(
            "Runs a simulation in a separate process. The graph is never loaded into "
            "this window, so a multi-million-cell run uses only the run's own memory "
            "and the GUI stays responsive. The run keeps going if you close the app."
        )
        intro.setWordWrap(True)
        form.addRow(intro)

        # The graph row is this tab's own: it picks a folder rather than loading it.
        graph_row = self.QtWidgets.QHBoxLayout()
        self.graph_combo = self.QtWidgets.QComboBox()
        self.graph_combo.currentTextChanged.connect(self._handle_graph_changed)
        refresh = self.QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_graphs)
        graph_row.addWidget(self.graph_combo, 1)
        graph_row.addWidget(refresh)
        form.addRow("graph", graph_row)
        self.graph_info = self.QtWidgets.QLabel("")
        self.graph_info.setWordWrap(True)
        form.addRow(self.graph_info)
        self.update_graph_button = self.QtWidgets.QPushButton("Update graph (rebuild nodes.csv + edges.npz)")
        self.update_graph_button.setToolTip(
            "Regenerate the fast-load artifacts (nodes.csv + edges.npz) from graph.json in "
            "a SEPARATE process (the graph is never loaded into this window). Run this once "
            "after editing a graph so later runs load fast and use less memory.\n\n"
            "edges.npz carries the conduction edges. Without it, temperature-dependent "
            "properties cannot rebuild L(T) and the run silently falls back to constant "
            "properties, so this is required for an accurate cryogenic run."
        )
        self.update_graph_button.clicked.connect(self.update_graph)
        form.addRow(self.update_graph_button)
        # Resume: a run that died (or was stopped) keeps checkpoints, so it can be
        # continued instead of restarting from t=0. Picking a run here reuses its
        # directory; the runner then resumes from that directory's newest
        # checkpoint. Defaults to a fresh run so the normal path is unchanged.
        resume_row = self.QtWidgets.QHBoxLayout()
        self.resume_combo = self.QtWidgets.QComboBox()
        self.resume_combo.setToolTip(
            "Continue a previous run from its last checkpoint instead of starting over.\n"
            "The run keeps its original output directory, so its plots, events.log and "
            "timeseries continue in place: the earlier rows are reloaded and appended to, "
            "and the clock picks up from the checkpoint, so t_final_s still means total "
            "elapsed rather than another full duration.\n\n"
            "The checkpoint supplies the starting state, so 'initial T K' is IGNORED on a "
            "resume. provenance.json records that it was overridden and keeps the earlier "
            "leg under 'previous'; the earlier parameter file is kept as "
            "simulation_parameters.leg<N>.json.\n\n"
            "Everything else below still applies, so this is also how to resume with "
            "something changed -- turn the GPU solver off after a crash, extend the "
            "duration, retune the controller. Kp/Ki/filter/hold are read fresh every step "
            "and are safe to change. dt and the CONTROLLER are not: dt changes the loop "
            "gain, and a G rebuilt with a different sensor count discards the restored "
            "integral (logged, so check events.log)."
        )
        resume_refresh = self.QtWidgets.QPushButton("Refresh")
        resume_refresh.setMaximumWidth(80)
        resume_refresh.clicked.connect(self.refresh_resume_runs)
        # Redraw a finished run's figures without re-running it -- useful when the
        # plotting itself changed (the controlled-sensor filter, say), since the
        # timeseries is already on disk and the plots are the only stale part.
        self.replot_button = self.QtWidgets.QPushButton("Generate plots")
        self.replot_button.setMaximumWidth(130)
        self.replot_button.setToolTip(
            "Redraw plots/ for the run selected above, from its saved timeseries.npz.\n"
            "Does not re-run the simulation and does not touch its data -- only the "
            "figures are rewritten."
        )
        self.replot_button.clicked.connect(self.regenerate_run_plots)
        resume_row.addWidget(self.resume_combo, 1)
        resume_row.addWidget(resume_refresh)
        resume_row.addWidget(self.replot_button)
        form.addRow("resume", resume_row)
        self.resume_combo.currentIndexChanged.connect(self._sync_replot_enabled)
        self.resume_combo.currentIndexChanged.connect(self.load_resume_run_settings)
        self.resume_combo.currentIndexChanged.connect(self._sync_initial_temperature_enabled)
        self._sync_replot_enabled()
        # MIMO PI needs one object: the plant's DC gain. In simulation that is an
        # exact solve of L T = P, not a step-test campaign -- 74.3% of G lives in
        # modes slower than 10,000 s here, so a short identification understates it
        # badly (-96.6% at the sys-ID default of 300 s).
        self.build_gain_button = self.QtWidgets.QPushButton("Generate G matrix (MIMO PI)")
        self.build_gain_button.setToolTip(
            "Solve the plant's DC gain exactly from the operator, in a SEPARATE process, "
            "WITHOUT loading graph.json.\n\n"
            "G is a LINEARIZATION, so it is computed at the 'operating T K' field in the "
            "Controller Design section below (the same field the modal build uses) - "
            "conductance depends on temperature through k(T) and h(T), and a gain taken at the "
            "wrong background is systematically wrong.\n\n"
            "The result appears in the controller list as a MIMO PI entry. Progress goes to "
            "build_g_matrix.log in the graph folder."
        )
        self.build_gain_button.clicked.connect(self.build_gain_matrix)
        form.addRow(self.build_gain_button)
        # Almost every claim anyone makes about this plant is a claim about G: how
        # coupled it is, how many directions it really has, whether it could be
        # paired one heater per sensor, which channels can be shaped independently.
        # Those were answered one at a time in whatever notebook was open, so the
        # answers drifted and none were reproducible. This computes them together
        # from the one artifact and writes a report that stands on its own.
        self.plant_analysis_button = self.QtWidgets.QPushButton("Analyse plant (full G report)")
        self.plant_analysis_button.setToolTip(
            "Analyse the G matrix selected in the controller row and write "
            "analysis/ beside it: plant_analysis.md, plant_analysis.json, per-channel and "
            "per-heater CSVs, and figures for the gain structure, singular spectrum, "
            "dominant directions, RGA, pairing, actuator redundancy, per-channel "
            "reachability and the achievable operating point.\n\n"
            "Everything is derived from G, so the report is reproducible from the artifact "
            "alone. The operating-point section additionally uses the setpoint table and "
            "the Controller section's max heater power, and says so when the artifact "
            "carries no passive reference to measure a deviation against.\n\n"
            "The 'controlled' ticks in the setpoint table are applied: dropping a channel "
            "moves every result, because each one runs through the pseudo-inverse."
        )
        self.plant_analysis_button.clicked.connect(self.analyse_plant)
        form.addRow(self.plant_analysis_button)
        self.notes_edit = self.QtWidgets.QLineEdit()
        self.notes_edit.setPlaceholderText("optional note stored with the run")
        form.addRow("notes", self.notes_edit)

        # Everything else is the shared panel -- the same sections, in the same
        # order, as the Heat Transfer Simulation tab, with the graph-dependent and
        # playback rows hidden and the Solver section shown.
        self.panel = SimulationControlsPanel(
            self._qt,
            mode=MODE_HEADLESS,
            actions={
                "start_headless": self.start_run,
                "stop_headless": self.stop_run,
                "open_output": self.open_output,
                # The sim tab's set-all / randomize buttons edit a loaded model.
                # Headless has none, so here they drive the whole-run initial
                # temperature / setpoint that get passed to the subprocess instead.
                "set_all_initial_temperatures": self._set_all_initial_temperatures,
                "randomize_setpoints": self._randomize_setpoints,
                # The panel's Modal LQR Design section is shown in headless mode
                # too: its spins set what gets built, and this runs the same
                # reduction in a separate process instead of in this window.
                "build_modal_controller": self.build_modal_controller,
            },
            # Autosave: every edit goes straight back to the graph's
            # simulation_parameters.json, so settings survive closing the app.
            on_parameter_change=self._handle_parameter_change,
        )
        self.panel.build(form)
        # The simulation tab's per-node "Parameters" editor is deliberately NOT
        # built here (see build_readout_editor): without a model to write to every
        # field was inert or duplicated the sections above. Per-heater limits are
        # the one thing that was genuinely per-node, and they now live in the
        # override table below, which can show every heater at once.
        self.panel.export_to(self)
        # controls_scroll is NOT added here: app.py puts it in the window's shared
        # side-panel stack, exactly as it does for the simulation tab. The tab body
        # holds the per-role tables (setpoints, heater overrides, PI gains) and, in
        # place of the 3D viewer, the run's progress / status / log.
        # Per-sensor setpoints. The simulation tab edits these through its readout
        # tables, which need a loaded graph; here the sensor list comes from the
        # build's own nodes.csv (see graph_roles), so the graph still never enters
        # this process and the list exists before the first run has ever happened.
        #
        # This table is the ONLY source of setpoints for the run: every row is
        # prefilled with the default, and Randomize / Set all / hand-editing all
        # work on top of that. There is no global setpoint to fall back to, so a
        # row's value is the whole answer for that sensor.
        setpoint_box = self.QtWidgets.QGroupBox("Per-sensor setpoints")
        setpoint_layout = self.QtWidgets.QVBoxLayout(setpoint_box)
        setpoint_help = self.QtWidgets.QLabel(
            f"Every sensor the graph declares, prefilled at {DEFAULT_SETPOINT_K:g} K. Edit a "
            "row to give that sensor its own target, or use Randomize / Set all. A row left "
            "blank keeps whatever setpoint the graph itself holds."
        )
        setpoint_help.setWordWrap(True)
        setpoint_layout.addWidget(setpoint_help)
        self.setpoint_table = self.QtWidgets.QTableWidget(0, 3)
        self.setpoint_table.setHorizontalHeaderLabels(["sensor", "setpoint K", "controlled"])
        self.setpoint_table.horizontalHeader().setStretchLastSection(True)
        self.setpoint_table.setMinimumHeight(200)
        setpoint_layout.addWidget(self.setpoint_table, 1)
        setpoint_buttons = self.QtWidgets.QHBoxLayout()
        # "Set all" needs its own value now that there is no global setpoint row.
        self.setpoint_all_spin = self.panel.double_spin(0.0, 1.0e6, DEFAULT_SETPOINT_K, 1.0)
        self.setpoint_all_spin.setToolTip("The value 'Set all' writes into every sensor row.")
        setpoint_buttons.addWidget(self.setpoint_all_spin)
        for label, slot in (
            ("Load sensors", lambda: self.load_sensor_setpoints(announce=True)),
            ("Set all", self.apply_setpoint_to_all_sensors),
            ("Clear", self.clear_setpoint_overrides),
        ):
            button = self.QtWidgets.QPushButton(label)
            button.clicked.connect(slot)
            setpoint_buttons.addWidget(button)
        setpoint_layout.addLayout(setpoint_buttons)

        # Per-sensor PI gains, beneath the setpoint table. Channels come from the
        # SELECTED G matrix: after decoupling a channel owns a controlled sensor,
        # and G's sensor_ids are exactly those channels. Blank = use the global.
        gain_box = self.QtWidgets.QGroupBox("Per-sensor PI gains (MIMO PI)")
        gain_layout = self.QtWidgets.QVBoxLayout(gain_box)
        gain_help = self.QtWidgets.QLabel(
            "Blank uses the global Kp/Ki. A row needs BOTH values to count as an "
            "override. Channels come from the selected G matrix; 'Save preset' stores "
            "the tuning beside that matrix, since gains do not transfer to a different G."
        )
        gain_help.setWordWrap(True)
        gain_layout.addWidget(gain_help)
        self.gain_table = self.QtWidgets.QTableWidget(0, 3)
        self.gain_table.setHorizontalHeaderLabels(["sensor", "Kp", "Ki"])
        self.gain_table.horizontalHeader().setStretchLastSection(True)
        self.gain_table.setMinimumHeight(180)
        gain_layout.addWidget(self.gain_table, 1)
        gain_buttons = self.QtWidgets.QHBoxLayout()
        for label, slot in (
            ("Load gains", lambda: self.load_pi_gains(announce=True)),
            ("Set all", self.apply_pi_gains_to_all),
            ("Clear", self.clear_pi_gain_overrides),
            ("Save preset", self.save_pi_preset),
        ):
            button = self.QtWidgets.QPushButton(label)
            button.clicked.connect(slot)
            gain_buttons.addWidget(button)
        gain_layout.addLayout(gain_buttons)

        # Per-heater limit overrides. Every heater runs on the Controller section's
        # defaults (default max heater power, slew rate) unless it is given its own
        # value here -- a driver limit belongs to the hardware, so heaters on
        # different drivers can differ, while the common case stays one number on
        # the left. Blank = use the default; only filled cells are sent.
        heater_box = self.QtWidgets.QGroupBox("Per-heater overrides")
        heater_layout = self.QtWidgets.QVBoxLayout(heater_box)
        heater_help = self.QtWidgets.QLabel(
            "Every heater uses the Controller defaults on the left. Fill a cell to "
            "override that heater's own limit; blank leaves it on the default. "
            "Filling 'manual W' instead drives that heater OPEN-LOOP at a fixed "
            "wattage -- it stops taking controller commands, which is what an "
            "open-loop step test against a column of the gain matrix needs."
        )
        heater_help.setWordWrap(True)
        heater_layout.addWidget(heater_help)
        self.heater_table = self.QtWidgets.QTableWidget(0, 5)
        self.heater_table.setHorizontalHeaderLabels(
            ["heater", "max power W", "slew W/s", "efficiency", "manual W"]
        )
        self.heater_table.horizontalHeader().setStretchLastSection(True)
        self.heater_table.setMinimumHeight(180)
        heater_layout.addWidget(self.heater_table, 1)
        heater_buttons = self.QtWidgets.QHBoxLayout()
        for label, slot in (
            ("Load heaters", lambda: self.load_heaters(announce=True)),
            ("Clear", self.clear_heater_overrides),
        ):
            button = self.QtWidgets.QPushButton(label)
            button.clicked.connect(slot)
            heater_buttons.addWidget(button)
        heater_layout.addLayout(heater_buttons)

        editor_column = self.QtWidgets.QWidget()
        editor_layout = self.QtWidgets.QVBoxLayout(editor_column)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.addWidget(setpoint_box, 1)
        editor_layout.addWidget(heater_box, 1)
        editor_layout.addWidget(gain_box, 1)
        outer.addWidget(editor_column, 0, self.QtCore.Qt.AlignTop)

        # In place of the 3D viewer: what the launched run is doing.
        right = self.QtWidgets.QWidget()
        right_layout = self.QtWidgets.QVBoxLayout(right)
        self.progress = self.QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        right_layout.addWidget(self.progress)
        self.summary_label = self.QtWidgets.QLabel("No run started.")
        self.summary_label.setWordWrap(True)
        right_layout.addWidget(self.summary_label)
        right_layout.addWidget(self.QtWidgets.QLabel("Run log"))
        self.log_view = self.QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        right_layout.addWidget(self.log_view, 1)
        outer.addWidget(right, 1)

    # -- update graph (rebuild fast-load nodes.csv from graph.json) ---------- #
    def update_graph(self) -> None:
        if self._refresh_process is not None and self._refresh_process.poll() is None:
            self._status("A graph update is already running.", True)
            return
        folder = self._selected_folder()
        if folder is None:
            self._status("Select a graph first.", True)
            return
        if not (folder / "graph.json").exists():
            self._status(f"No graph.json in {folder}.", True)
            return
        try:
            from .fast_graph_io import edges_only_refresh_is_enough, launch_refresh_subprocess

            # When edges.npz is the only thing missing (a graph built before that
            # artifact existed), stream it out of graph.json instead of doing the
            # full parse -- bounded memory, so it works on graphs whose full load
            # would not fit in RAM.
            edges_only = edges_only_refresh_is_enough(folder)
            self._refresh_process = launch_refresh_subprocess(folder, edges_only=edges_only)
        except Exception as exc:  # noqa: BLE001
            self._status(f"Could not start the graph update: {exc}", True)
            return
        self._refresh_folder = folder
        self.update_graph_button.setEnabled(False)
        detail = (
            "streaming edges.npz out of graph.json (low memory)"
            if edges_only
            else "rebuilding nodes.csv + edges.npz from graph.json"
        )
        self._status(
            f"Updating {folder.name}: {detail} "
            "(separate process; this window stays responsive)…",
            False,
        )

    def _poll_refresh(self) -> None:
        proc = self._refresh_process
        if proc is None or proc.poll() is None:
            return
        code = proc.returncode
        folder = self._refresh_folder
        self._refresh_process = None
        self.update_graph_button.setEnabled(True)
        if code == 0 and folder is not None:
            from .fast_graph_io import REFRESH_LOG_FILENAME, can_load_fast

            usable, reason = can_load_fast(folder)
            self._handle_graph_changed()  # refresh the node-count / params line
            if usable:
                self._status(f"Graph updated — future runs will load fast and lean.", False)
            else:
                self._status(
                    f"Graph update ran but fast load is still unavailable: {reason} "
                    f"(see {folder / REFRESH_LOG_FILENAME}).",
                    True,
                )
        else:
            log = (folder / "refresh_fast_load.log") if folder is not None else "the run folder"
            self._status(f"Graph update failed (exit {code}); see {log}.", True)

    # -- set-all / randomize (headless equivalents) -------------------------- #
    def _set_all_initial_temperatures(self) -> None:
        """The sim tab sets every component's initial temperature on the loaded
        model. Headless holds no model, so this sets the whole-run initial-T
        override that ``run_simulation.py`` applies to every cell instead."""
        value = float(self.initial_temperature_all_spin.value())
        self.initial_spin.setValue(value)
        self.use_initial.setChecked(True)
        self._status(f"Run will start every cell at {value:g} K.", False)

    def _randomize_setpoints(self) -> None:
        """Give every sensor its own target drawn from center +/- spread.

        This matches the simulation tab, which randomizes per sensor on the loaded
        model. Here the draw fills the per-sensor table, so each sensor gets an
        INDEPENDENT value rather than the whole run sharing one -- the point of
        randomizing is the spread between sensors, and a single shared value has
        none.

        With no sensor rows there is nothing to spread across, and (since the global
        setpoint row was removed) nowhere else to put a value, so it says so instead
        of silently doing nothing.
        """
        center = float(self.sensor_random_center_spin.value())
        spread_K = float(self.sensor_random_spread_mK_spin.value()) * 1.0e-3
        table = getattr(self, "setpoint_table", None)
        rows = getattr(self, "_sensor_rows_manifest", None) or []
        if table is None or not rows:
            self._status(
                "No sensors to randomize. Press 'Load sensors' first; if the list stays "
                "empty this graph's nodes.csv declares no sensors.",
                True,
            )
            return
        drawn: list[float] = []
        for index in range(len(rows)):
            value = center + random.uniform(-spread_K, spread_K)
            drawn.append(value)
            table.setItem(index, 1, self.QtWidgets.QTableWidgetItem(f"{value:.6g}"))
        self._status(
            f"Randomized {len(drawn)} sensor setpoints around {center:g} K "
            f"+/- {spread_K * 1.0e3:g} mK "
            f"(actual span {min(drawn):g} to {max(drawn):g} K).",
            False,
        )

    # -- parameters ---------------------------------------------------------- #
    def _load_parameters_for_graph(self, folder: Path) -> None:
        """Populate the form from the graph's saved parameters, so a headless run
        starts from exactly what the Heat Transfer Simulation tab last saved."""
        from .simulation_parameters import SimulationParameters, load_simulation_parameters

        path = folder / "simulation_parameters.json"
        if path.is_file():
            params, _extras = load_simulation_parameters(path)
            source = f"loaded from {path.name}"
        else:
            params = SimulationParameters()
            source = "defaults (graph has no saved simulation_parameters.json)"
        self.panel.set_params(params)
        self._params_source = source

    @staticmethod
    def _preserve_prior_parameters(params_path: Path) -> Path | None:
        """Copy a resumed run's parameter file aside before it is overwritten.

        Numbered rather than a single .prev so a run resumed more than once keeps
        every leg. Best-effort: failing to archive the old file must not stop the
        run, but it is reported so it is not discovered later by its absence.
        """
        import shutil

        for index in range(1, 1000):
            candidate = params_path.with_name(f"simulation_parameters.leg{index}.json")
            if not candidate.exists():
                try:
                    shutil.copy2(params_path, candidate)
                except OSError:
                    return None
                return candidate
        return None

    def _collect_parameters(self):
        """Current form values as a SimulationParameters (unsupported fields keep
        whatever the graph's saved file had)."""
        from .simulation_parameters import SimulationParameters, load_simulation_parameters

        folder = self._selected_folder()
        # Fields with no widget keep the base's value, so on a resume the base has to
        # be the RUN's own file. Taking the graph's instead would quietly revert every
        # such field to whatever the graph was last saved with -- a "resume" that
        # continues the state but not the physics.
        resume_dir = self.selected_resume_dir()
        candidates = [
            path
            for path in (
                (resume_dir / "simulation_parameters.json") if resume_dir else None,
                (folder / "simulation_parameters.json") if folder else None,
            )
            if path is not None and path.is_file()
        ]
        if candidates:
            base, _extras = load_simulation_parameters(candidates[0])
        else:
            base = SimulationParameters()
        # Widgets win; fields without a widget keep the graph's saved value.
        params = self.panel.read(base)
        # Except this one. The enabled-I/O table is a LIVE tab control (its section
        # is hidden here), so an unticked heater there would ride into an overnight
        # run through the saved file with nothing in this tab to show it -- the run
        # would quietly drive fewer heaters than the controller was designed for.
        # Every heater runs headless; None is the "no filter" convention.
        # Sensors, unlike heaters, ARE filterable headless: unticking one is how a
        # channel the plant cannot serve gets out of the loop, and the column that
        # does it lives in this tab where the run can see it.
        return replace(
            params,
            enabled_heater_node_ids=None,
            enabled_sensor_node_ids=self.collect_enabled_sensors(),
        )

    def persist_parameters(self) -> bool:
        """Write the current form back to <graph>/simulation_parameters.json.

        The tab always LOADED this file but only ever wrote into the run directory,
        so every edit was lost when the app closed and the panel had to be set up
        again from scratch each session. Saving to the graph folder makes the graph's
        file the single place the settings live -- the same file the loader already
        reads on startup and on every graph switch.

        Extras (unknown keys from other tools) are preserved: the loader hands them
        back and they are written out again rather than being silently dropped.
        """
        folder = self._selected_folder()
        if folder is None or not folder.is_dir():
            return False
        from .simulation_parameters import (
            load_simulation_parameters,
            save_simulation_parameters,
        )

        path = folder / "simulation_parameters.json"
        extras: dict[str, Any] = {}
        if path.is_file():
            try:
                _base, extras = load_simulation_parameters(path)
            except Exception:  # noqa: BLE001 - a corrupt file must not block saving
                extras = {}
        try:
            save_simulation_parameters(path, self._collect_parameters(), extras)
        except OSError as exc:
            self._status(f"Could not save parameters: {exc}", True)
            return False
        return True

    def _handle_parameter_change(self, *_: Any) -> None:
        """Autosave on every edit. The file is small and writes are atomic enough
        that doing this per keystroke is cheaper than making the user remember."""
        if getattr(self, "_loading_parameters", False):
            return          # do not write back while populating the form from disk
        self.persist_parameters()

    # -- graph discovery (metadata only -- never loads a graph) -------------- #
    def refresh_graphs(self) -> None:
        root = Path(self._graphs_root())
        current = self.graph_combo.currentText()
        names = []
        if root.is_dir():
            names = sorted(
                entry.name
                for entry in root.iterdir()
                if entry.is_dir() and (entry / "node_ids.npy").exists()
            )
        self.graph_combo.blockSignals(True)
        self.graph_combo.clear()
        self.graph_combo.addItems(names)
        if current in names:
            self.graph_combo.setCurrentText(current)
        self.graph_combo.blockSignals(False)
        self._handle_graph_changed()

    def _selected_folder(self) -> Path | None:
        name = self.graph_combo.currentText().strip()
        if not name:
            return None
        return Path(self._graphs_root()) / name

    def _handle_graph_changed(self, *_: Any) -> None:
        # Switching graphs repopulates every widget, which fires the change hook on
        # each one. Without this guard the newly-selected graph's settings would be
        # written back over it mid-population, one field at a time.
        self._loading_parameters = True
        try:
            self._populate_from_graph()
        finally:
            self._loading_parameters = False

    def _populate_from_graph(self, *_: Any) -> None:
        folder = self._selected_folder()
        self.controller_scheme_combo.clear()
        if folder is None or not folder.is_dir():
            self.graph_info.setText("")
            return
        # Node count comes from node_ids.npy's header, not by loading the graph.
        node_count = "?"
        try:
            import numpy as np

            node_count = f"{int(np.load(folder / 'node_ids.npy', mmap_mode='r').shape[0]):,}"
        except Exception:  # noqa: BLE001
            pass
        # Same list, from the same helper, as the simulation tab's controller row:
        # a controller artifact validated by its contents, not its filename.
        # The list mixes both schemes, so each entry carries its own scheme rather
        # than having it inferred from "is a path selected".
        from .sys_id_artifacts import list_sys_id_gain_matrices

        artifacts = list_modal_artifacts(folder)
        gains = list_sys_id_gain_matrices(folder)
        self.controller_scheme_combo.addItem(PID_QP_LABEL, ("none", ""))
        for info in gains:
            self.controller_scheme_combo.addItem(f"MIMO PI - {info.name}", ("mimo_pi", str(info.path)))
        for info in artifacts:
            self.controller_scheme_combo.addItem(info.label, ("modal_lqr", str(info.path)))
        if gains or artifacts:
            self.controller_scheme_combo.setCurrentIndex(1)
        self._load_parameters_for_graph(folder)
        self.refresh_resume_runs()
        # Each loader is isolated: one that fails must not take the rest of the
        # refresh -- and the graph info line -- with it. An empty table with a stale
        # header is indistinguishable from "this graph has no sensors".
        for name, loader in (
            ("sensor setpoints", self.load_sensor_setpoints),
            ("heaters", self.load_heaters),
            ("PI gains", self.load_pi_gains),
        ):
            try:
                loader()
            except Exception as exc:  # noqa: BLE001 - a bad file must not blank the tab
                self._status(f"Could not load {name} for {folder.name}: {exc}", True)
        # Say plainly whether the run will take the fast path and whether the edge
        # data is there. Without edges.npz, temperature-dependent properties would
        # have to fall back to constant properties (L(T) is rebuilt from the edges),
        # and the run silently models different physics -- so surface it up front
        # rather than leaving it to be discovered in events.log.
        self.graph_info.setText(
            f"{node_count} nodes • {len(artifacts)} controller artifact(s) • {folder}\n"
            f"parameters: {self._params_source}\n"
            f"roles: {self._role_status(folder)}\n"
            f"{self._fast_load_status(folder)}"
        )

    def _role_status(self, folder: Path) -> str:
        """How many heaters/sensors the tables found, and where they came from.

        Both tables are fed from one place, so when they come up empty this line is
        the difference between "this graph declares no roles", "its nodes.csv is too
        old to say" and "the file could not be read at all".
        """
        try:
            manifest = load_role_manifest(folder)
        except Exception as exc:  # noqa: BLE001
            return f"could not read roles: {exc}"
        heaters, sensors = len(manifest.heaters), len(manifest.sensors)
        if heaters or sensors:
            return f"{heaters} heater(s), {sensors} sensor(s) from {manifest.source}"
        fallback = len(getattr(self, "_sensor_rows_manifest", None) or [])
        detail = f"{manifest.source}"
        if fallback:
            return f"0 from {detail}; {fallback} sensor(s) from the last run's sensors.csv"
        return f"no heaters or sensors found ({detail})"

    # -- modal controller build (separate process, no graph in this window) --- #
    def build_modal_controller(self) -> None:
        if self._modal_build_process is not None and self._modal_build_process.poll() is None:
            self._status("A modal controller build is already running.", True)
            return
        folder = self._selected_folder()
        if folder is None:
            self._status("Select a graph first.", True)
            return
        from .fast_graph_io import can_load_fast, launch_modal_build_subprocess

        usable, reason = can_load_fast(folder)
        if not usable:
            self._status(
                f"Cannot build without the fast-load artifacts ({reason}). Press "
                "'Update graph' first -- the alternative is the graph.json loader, which "
                "needs roughly 45 GB on a graph this size.",
                True,
            )
            return
        # Every design parameter comes from the panel's Modal LQR Design section --
        # the same spins the simulation tab uses -- so the two tabs build the same
        # controller from the same numbers.
        t_op = float(self.modal_temp_spin.value())
        modes = int(self.modal_modes_spin.value())
        order = int(self.modal_order_spin.value())
        effort = float(self.modal_effort_spin.value())
        integral = float(self.modal_integral_spin.value())
        # Design the gain at the dt this tab will actually run at: an LQR gain is
        # only valid at the sample rate it was solved for.
        design_dt = float(getattr(self._collect_parameters(), "dt_s", 1.0) or 1.0)
        if order > modes:
            self._status(
                f"Reduced order r={order} cannot exceed the {modes} slow modes it is "
                "truncated from.",
                True,
            )
            return
        try:
            # Design at the dt this tab will actually run, so the sampled gain is
            # correct out of the box instead of needing a runtime re-solve.
            self._modal_build_process = launch_modal_build_subprocess(
                folder, t_op_K=t_op, n_modes=modes, order=order,
                effort=effort, integral_gain=integral, design_dt_s=design_dt,
            )
        except Exception as exc:  # noqa: BLE001
            self._status(f"Could not start the controller build: {exc}", True)
            return
        self._modal_build_folder = folder
        self.modal_design_button.setEnabled(False)
        message = (
            f"Building r={order} from {modes} modes at T_op={t_op:g} K "
            f"(effort {effort:g}, integral {integral:g}, dt {design_dt:g} s) for {folder.name}."
        )
        self.modal_design_status_label.setText(message)
        self._status(
            f"{message} Separate process; this window stays responsive. Progress: "
            "build_modal_controller.log in the graph folder.",
            False,
        )

    def _poll_modal_build(self) -> None:
        proc = getattr(self, "_modal_build_process", None)
        if proc is None or proc.poll() is None:
            return
        code = proc.returncode
        folder = self._modal_build_folder
        self._modal_build_process = None
        self.modal_design_button.setEnabled(True)
        from .fast_graph_io import MODAL_BUILD_LOG_FILENAME

        log = (folder / MODAL_BUILD_LOG_FILENAME) if folder is not None else "the graph folder"
        if code == 0 and folder is not None:
            # Repopulate the controller list so the new artifact is selectable.
            self._handle_graph_changed()
            self.modal_design_status_label.setText("Built. Select it in the controller row.")
            self._status(f"Modal controller built; see {log}.", False)
        else:
            self.modal_design_status_label.setText(f"Failed (exit {code}).")
            self._status(f"Modal controller build failed (exit {code}); see {log}.", True)

    # -- DC gain generation (separate process, no graph in this window) ------- #
    def build_gain_matrix(self) -> None:
        if self._gain_build_process is not None and self._gain_build_process.poll() is None:
            self._status("A G matrix build is already running.", True)
            return
        folder = self._selected_folder()
        if folder is None:
            self._status("Select a graph first.", True)
            return
        from .fast_graph_io import can_load_fast, launch_gain_build_subprocess

        usable, reason = can_load_fast(folder)
        if not usable:
            self._status(
                f"Cannot build without the fast-load artifacts ({reason}). Press "
                "'Update graph' first.",
                True,
            )
            return
        # Same operating point the modal design uses, so the two artifacts describe
        # the same linearization and can be compared directly.
        t_op = float(self.modal_temp_spin.value())
        try:
            self._gain_build_process = launch_gain_build_subprocess(folder, t_op_K=t_op)
        except Exception as exc:  # noqa: BLE001
            self._status(f"Could not start the G matrix build: {exc}", True)
            return
        self._gain_build_folder = folder
        self.build_gain_button.setEnabled(False)
        self._status(
            f"Solving the exact DC gain for {folder.name} at T_op={t_op:g} K "
            "(separate process; this window stays responsive). Progress: "
            "build_g_matrix.log in the graph folder.",
            False,
        )

    def _poll_gain_build(self) -> None:
        proc = getattr(self, "_gain_build_process", None)
        if proc is None or proc.poll() is None:
            return
        code = proc.returncode
        folder = self._gain_build_folder
        self._gain_build_process = None
        self.build_gain_button.setEnabled(True)
        from .fast_graph_io import GAIN_BUILD_LOG_FILENAME

        log = (folder / GAIN_BUILD_LOG_FILENAME) if folder is not None else "the graph folder"
        if code == 0 and folder is not None:
            self._handle_graph_changed()   # the new matrix joins the controller list
            self._status(f"G matrix built; select it in the controller row. See {log}.", False)
        else:
            self._status(f"G matrix build failed (exit {code}); see {log}.", True)

    # -- full plant analysis of the selected G matrix ------------------------ #
    def analyse_plant(self) -> None:
        """Analyse the selected G matrix and write the report beside it.

        In-process rather than detached, like 'Generate plots': the whole cost is
        an SVD and a pseudo-inverse of a matrix with one row per controlled sensor,
        which is milliseconds at these sizes -- there is nothing here worth a
        subprocess and its startup. Figures render through the Agg canvas directly
        so they cannot disturb the Qt backend the 2D graph view is using.
        """
        scheme, path = self.panel.selected_controller()
        if scheme != "mimo_pi" or not path:
            self._status(
                "Select a MIMO PI entry in the controller row first -- the analysis is "
                "computed from its G matrix. Press 'Generate G matrix' if this graph has "
                "none yet.",
                True,
            )
            return
        if not Path(path).is_dir():
            self._status(f"The selected G matrix is gone: {path}", True)
            return
        try:
            from .plant_report import write_plant_analysis

            analysis = write_plant_analysis(
                Path(path),
                # The run's own channel selection, so the report describes the loop
                # this tab would actually close rather than a different one.
                enabled_sensor_ids=self.collect_enabled_sensors(),
                # The operating-point section needs a target and a budget, and both
                # live in this tab rather than in the artifact.
                setpoints_K=self.collect_setpoint_overrides() or None,
                heater_max_power_W=self._analysis_heater_caps(),
            )
        except Exception as exc:  # noqa: BLE001 - report rather than kill the tab
            self._status(f"Could not analyse the plant: {exc}", True)
            return
        self._status(self._describe_analysis(analysis), False)

    def _analysis_heater_caps(self) -> dict[int, float] | float:
        """Per-heater max power, defaulting to the Controller section's value.

        Mirrors what the run itself does (``_controller_heater_max_power``): the
        Controller section's figure is a CEILING on every heater, a heater rated
        below it keeps its lower rating, and only a per-heater entry in the heater
        table may exceed it. Analysing against a different budget than the run would
        use is how a report ends up disagreeing with the loop it describes.
        """
        from .simulation_parameters import SimulationParameters

        widget = (getattr(self, "inputs", None) or {}).get("mimo_default_heater_max_power_W")
        default = (
            float(widget.value())
            if widget is not None
            else SimulationParameters().mimo_default_heater_max_power_W
        )
        overrides = {
            node_id: float(fields["heater_max_power_W"])
            for node_id, fields in self.collect_heater_overrides().items()
            if "heater_max_power_W" in fields
        }
        if not overrides:
            return default
        rows = getattr(self, "_heater_rows_manifest", None) or []
        caps = {int(row["node_id"]): default for row in rows if row.get("node_id")}
        caps.update(overrides)
        return caps

    @staticmethod
    def _describe_analysis(analysis: Any) -> str:
        """The headline numbers, so the status bar says what was found rather than
        only that something was written."""
        stats = analysis.stats
        spectrum = stats["spectrum"]
        summary = stats["pairing"]["rga_summary"]
        parts = [
            f"{stats['n_sensors']}x{stats['n_heaters']} G",
            f"cond {spectrum['condition_number']:.4g}",
            f"sigma_1 carries {spectrum['top_energy_fraction'] * 100:.1f}%",
        ]
        if summary.get("rga_diag_negative") is not None:
            parts.append(f"{summary['rga_diag_negative']} negative RGA diagonal entries")
        bounded = (stats["uniform_lift"] or {}).get("nonnegative")
        if bounded:
            parts.append(f"uniform-lift residual {bounded['residual_rms_K_per_K']:.3g} K/K")
        skipped = stats.get("skipped_figures") or []
        tail = f" ({len(skipped)} figure(s) skipped)" if skipped else ""
        return (
            f"{'; '.join(parts)}. Wrote {len(analysis.figures)} figure(s) and "
            f"{len(analysis.tables)} table(s) to {analysis.out_dir}{tail}"
        )

    # -- per-sensor setpoints ------------------------------------------------ #
    def sensor_manifest(self, graph_name: str) -> list[dict[str, str]]:
        """Sensor rows (node_id, component_name, monitor_only) for ``graph_name``.

        Primary source is the BUILD: nodes.csv's is_sensor column, via graph_roles,
        which caches the scan beside the graph. That matters because this table is
        now the only place a run's setpoints come from -- taking the list from a
        previous run's sensors.csv meant a graph that had never been run had no
        sensors to target, and no way to get any.

        Falls back to the newest run's sensors.csv, which covers a graph whose
        nodes.csv predates those columns, and a sensor assigned in the app after the
        build (those edits live in graph.json, not nodes.csv).
        """
        folder = Path(self._graphs_root()) / graph_name
        manifest = load_role_manifest(folder)
        if manifest.sensors:
            return [
                {
                    "node_id": str(row.node_id),
                    "component_name": row.component_name,
                    "monitor_only": "true" if row.monitor_only else "false",
                }
                for row in manifest.sensors
            ]
        root = self.simulations_root() / graph_name
        if not root.is_dir():
            return []
        for run_dir in sorted(root.iterdir(), reverse=True):
            csv_path = run_dir / "sensors.csv"
            if not csv_path.is_file():
                continue
            try:
                with csv_path.open("r", newline="", encoding="utf-8") as handle:
                    rows = [row for row in csv.DictReader(handle) if row.get("node_id")]
            except OSError:
                continue
            if rows:
                return rows
        return []

    def heater_manifest(self, graph_name: str) -> list[dict[str, str]]:
        """Heater rows (node_id, component_name) for ``graph_name``.

        Same build-time source as the sensors. Deliberately NOT the selected
        controller artifact's heater_ids: that would show only the heaters the
        current scheme's matrix happens to cover, and the point of this table is
        every heater the run will actually drive.
        """
        folder = Path(self._graphs_root()) / graph_name
        manifest = load_role_manifest(folder)
        return [
            {"node_id": str(row.node_id), "component_name": row.component_name}
            for row in manifest.heaters
        ]

    def load_sensor_setpoints(self, announce: bool = False) -> None:
        """Fill the table from the newest run's sensor manifest.

        ``announce`` reports an empty result to the status bar. It stays False for
        the automatic refresh on graph change, which happens while this tab is being
        constructed inside the main window's own layout pass -- the host's status
        widget may not exist yet, and "no manifest" is not news the user asked for.
        """
        folder = self._selected_folder()
        table = getattr(self, "setpoint_table", None)
        if table is None:
            return
        rows = self.sensor_manifest(folder.name) if folder is not None else []
        table.setRowCount(len(rows))
        self._sensor_rows_manifest = rows
        for index, row in enumerate(rows):
            node_id = str(row.get("node_id", ""))
            name = str(row.get("component_name", ""))
            monitor = str(row.get("monitor_only", "")).lower() == "true"
            label = f"{node_id}  {name}" + ("  (monitor-only)" if monitor else "")
            item = self.QtWidgets.QTableWidgetItem(label)
            item.setFlags(self.QtCore.Qt.ItemIsEnabled)
            table.setItem(index, 0, item)
            # Prefilled, not blank: this table is the run's only source of
            # setpoints, so an unedited row still has to say what it wants.
            table.setItem(index, 1, self.QtWidgets.QTableWidgetItem(f"{DEFAULT_SETPOINT_K:g}"))
            # Untick to drop a channel from the loop entirely. A sensor the plant
            # cannot serve does not just track badly: it holds the largest error, so
            # it demands the largest deviation, and the allocator's least-squares fit
            # pushes every other channel above setpoint trying to help it.
            control = self.QtWidgets.QTableWidgetItem("")
            control.setFlags(
                self.QtCore.Qt.ItemIsEnabled | self.QtCore.Qt.ItemIsUserCheckable
            )
            control.setCheckState(
                self.QtCore.Qt.Unchecked if monitor else self.QtCore.Qt.Checked
            )
            table.setItem(index, 2, control)
        if announce and not rows and folder is not None:
            self._status(
                f"{folder.name} declares no sensors: its nodes.csv has no is_sensor rows "
                "and no previous run wrote a sensors.csv.",
                True,
            )

    # -- per-heater overrides ------------------------------------------------- #
    def load_heaters(self, announce: bool = False) -> None:
        """Fill the heater table from the graph's build-time heater list.

        Values start blank -- blank means "use the Controller section's defaults",
        which is what every heater does unless the user says otherwise.
        """
        table = getattr(self, "heater_table", None)
        if table is None:
            return
        folder = self._selected_folder()
        rows = self.heater_manifest(folder.name) if folder is not None else []
        table.setRowCount(len(rows))
        self._heater_rows_manifest = rows
        for index, row in enumerate(rows):
            label = f"{row.get('node_id', '')}  {row.get('component_name', '')}".strip()
            item = self.QtWidgets.QTableWidgetItem(label)
            item.setFlags(self.QtCore.Qt.ItemIsEnabled)
            table.setItem(index, 0, item)
            for column in (1, 2, 3, 4):
                table.setItem(index, column, self.QtWidgets.QTableWidgetItem(""))
        if announce and not rows and folder is not None:
            self._status(f"{folder.name} declares no heaters in its nodes.csv.", True)

    def collect_heater_overrides(self) -> dict[int, dict[str, float]]:
        """{node_id: {field: value}} for the cells the user actually filled.

        A heater with no filled cell is absent entirely, so the run applies the
        Controller defaults to it exactly as before this table existed.
        """
        table = getattr(self, "heater_table", None)
        rows = getattr(self, "_heater_rows_manifest", None) or []
        overrides: dict[int, dict[str, float]] = {}
        if table is None:
            return overrides
        columns = (
            (1, "heater_max_power_W"),
            (2, "heater_slew_rate_W_per_s"),
            (3, "heater_efficiency"),
            (4, "sensor_manual_power_W"),
        )
        for index, row in enumerate(rows):
            try:
                node_id = int(row["node_id"])
            except (KeyError, TypeError, ValueError):
                continue
            fields: dict[str, float] = {}
            for column, name in columns:
                cell = table.item(index, column)
                text = (cell.text() if cell is not None else "").strip()
                if not text:
                    continue
                try:
                    fields[name] = float(text)
                except (TypeError, ValueError):
                    continue
            if fields:
                overrides[node_id] = fields
        return overrides

    def clear_heater_overrides(self) -> None:
        table = getattr(self, "heater_table", None)
        if table is None:
            return
        for index in range(table.rowCount()):
            for column in (1, 2, 3, 4):
                table.setItem(index, column, self.QtWidgets.QTableWidgetItem(""))

    def collect_setpoint_overrides(self) -> dict[int, float]:
        """{node_id: setpoint_K} for rows the user actually filled in."""
        table = getattr(self, "setpoint_table", None)
        rows = getattr(self, "_sensor_rows_manifest", None) or []
        overrides: dict[int, float] = {}
        if table is None:
            return overrides
        for index, row in enumerate(rows):
            cell = table.item(index, 1)
            text = (cell.text() if cell is not None else "").strip()
            if not text:
                continue
            try:
                overrides[int(row["node_id"])] = float(text)
            except (TypeError, ValueError):
                continue
        return overrides

    def collect_enabled_sensors(self):
        """Ticked sensor ids, or None when nothing has been unticked.

        None is the "no filter" convention, so a tab nobody has touched behaves
        exactly as before. Monitor-only sensors start unticked and are irrelevant to
        the controller either way, so they are not counted as a filter on their own.
        """
        table = getattr(self, "setpoint_table", None)
        rows = getattr(self, "_sensor_rows_manifest", None) or []
        if table is None or not rows:
            return None
        enabled: list[int] = []
        unticked_controllable = False
        for index, row in enumerate(rows):
            monitor = str(row.get("monitor_only", "")).lower() == "true"
            cell = table.item(index, 2)
            ticked = bool(cell is not None and cell.checkState() == self.QtCore.Qt.Checked)
            try:
                node_id = int(row["node_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if ticked:
                enabled.append(node_id)
            elif not monitor:
                unticked_controllable = True
        return enabled if unticked_controllable else None

    def apply_setpoint_to_all_sensors(self) -> None:
        """Write the 'set all' value into every row, as a starting point to edit."""
        table = getattr(self, "setpoint_table", None)
        if table is None:
            return
        value = f"{self.setpoint_all_spin.value():g}"
        for index in range(table.rowCount()):
            table.setItem(index, 1, self.QtWidgets.QTableWidgetItem(value))

    def clear_setpoint_overrides(self) -> None:
        table = getattr(self, "setpoint_table", None)
        if table is None:
            return
        for index in range(table.rowCount()):
            table.setItem(index, 1, self.QtWidgets.QTableWidgetItem(""))

    # -- per-sensor MIMO PI gains -------------------------------------------- #
    def load_pi_gains(self, announce: bool = False) -> None:
        """Fill the gain table from the SELECTED G matrix.

        The channel list comes from G, not from a run manifest: after decoupling a
        PI channel owns a controlled sensor, and G's own sensor_ids are exactly
        those channels. Any preset saved beside that matrix is loaded with it.
        """
        table = getattr(self, "gain_table", None)
        if table is None:
            return
        scheme, path = self.panel.selected_controller()
        self._gain_rows = []
        if scheme != "mimo_pi" or not path:
            table.setRowCount(0)
            if announce:
                self._status(
                    "Select a MIMO PI entry in the controller row first; its G matrix "
                    "defines the channels these gains apply to.",
                    True,
                )
            return
        try:
            from .sys_id_artifacts import load_mimo_pi_preset, load_sys_id_gain_matrix_data

            data = load_sys_id_gain_matrix_data(Path(path))
            preset = load_mimo_pi_preset(Path(path)) or {}
        except Exception as exc:  # noqa: BLE001
            table.setRowCount(0)
            self._status(f"Could not read the gain matrix: {exc}", True)
            return
        sensors = [int(v) for v in data.sensor_ids]
        per_sensor = preset.get("per_sensor", {})
        if preset:
            # A saved preset is the tuning this matrix was built with, so show it.
            self.mimo_pi_kp_spin.setValue(float(preset.get("kp", self.mimo_pi_kp_spin.value())))
            self.mimo_pi_ki_spin.setValue(float(preset.get("ki", self.mimo_pi_ki_spin.value())))
        self._gain_rows = sensors
        table.setRowCount(len(sensors))
        for index, sensor_id in enumerate(sensors):
            item = self.QtWidgets.QTableWidgetItem(str(sensor_id))
            item.setFlags(self.QtCore.Qt.ItemIsEnabled)
            table.setItem(index, 0, item)
            override = per_sensor.get(int(sensor_id), {})
            table.setItem(index, 1, self.QtWidgets.QTableWidgetItem(
                f"{override['kp']:g}" if "kp" in override else ""))
            table.setItem(index, 2, self.QtWidgets.QTableWidgetItem(
                f"{override['ki']:g}" if "ki" in override else ""))

    def collect_pi_gain_overrides(self) -> dict[int, dict[str, float]]:
        """{sensor_id: {kp, ki}} for rows the user actually filled in. A row needs
        BOTH values, since a half-specified channel would silently take a global
        for the other gain and read as intentional."""
        table = getattr(self, "gain_table", None)
        rows = getattr(self, "_gain_rows", None) or []
        overrides: dict[int, dict[str, float]] = {}
        if table is None:
            return overrides
        for index, sensor_id in enumerate(rows):
            texts = []
            for column in (1, 2):
                cell = table.item(index, column)
                texts.append((cell.text() if cell is not None else "").strip())
            if not all(texts):
                continue
            try:
                overrides[int(sensor_id)] = {"kp": float(texts[0]), "ki": float(texts[1])}
            except (TypeError, ValueError):
                continue
        return overrides

    def apply_pi_gains_to_all(self) -> None:
        table = getattr(self, "gain_table", None)
        if table is None:
            return
        kp = f"{self.mimo_pi_kp_spin.value():g}"
        ki = f"{self.mimo_pi_ki_spin.value():g}"
        for index in range(table.rowCount()):
            table.setItem(index, 1, self.QtWidgets.QTableWidgetItem(kp))
            table.setItem(index, 2, self.QtWidgets.QTableWidgetItem(ki))

    def clear_pi_gain_overrides(self) -> None:
        table = getattr(self, "gain_table", None)
        if table is None:
            return
        for index in range(table.rowCount()):
            table.setItem(index, 1, self.QtWidgets.QTableWidgetItem(""))
            table.setItem(index, 2, self.QtWidgets.QTableWidgetItem(""))

    def save_pi_preset(self) -> None:
        """Write the current gains beside the selected G matrix.

        Gains are meaningless with a different G -- the unit-DC-gain assumption a
        channel is tuned against comes from that specific decoupler -- so the
        preset lives with the matrix rather than in the graph's parameters.
        """
        scheme, path = self.panel.selected_controller()
        if scheme != "mimo_pi" or not path:
            self._status("Select a MIMO PI entry before saving its tuning.", True)
            return
        try:
            from .sys_id_artifacts import save_mimo_pi_preset

            target = save_mimo_pi_preset(
                Path(path),
                kp=float(self.mimo_pi_kp_spin.value()),
                ki=float(self.mimo_pi_ki_spin.value()),
                per_sensor=self.collect_pi_gain_overrides(),
            )
        except Exception as exc:  # noqa: BLE001
            self._status(f"Could not save the tuning: {exc}", True)
            return
        overrides = len(self.collect_pi_gain_overrides())
        self._status(
            f"Saved Kp={self.mimo_pi_kp_spin.value():g}, Ki={self.mimo_pi_ki_spin.value():g} "
            f"and {overrides} per-sensor override(s) to {target.name}.",
            False,
        )

    # -- resume (continue a previous run from its last checkpoint) ----------- #
    def simulations_root(self) -> Path:
        """Where runs are written. Mirrors start_run's ``simulations/<graph>``."""
        return Path("simulations").resolve()

    @staticmethod
    def describe_checkpoint(run_dir: Path) -> tuple[int, float] | None:
        """(step, sim_time_s) of a run's newest checkpoint, or None if it has none.

        Reads only the two scalar arrays, so listing many runs stays cheap even
        though each checkpoint holds a full multi-million-node temperature field.
        """
        # The SAME selection the engine will make. This used to sort by filename and
        # take the last, which is the stale-checkpoint bug: the dropdown advertised
        # "resume at step 1977, t=59310s" while a t=100020 s checkpoint sat beside
        # it. The label is the only thing the user sees before committing hours.
        try:
            from .simulation_runner import newest_checkpoint

            found = newest_checkpoint(run_dir / "checkpoints")
        except Exception:  # noqa: BLE001 - a truncated checkpoint just is not offered
            return None
        if found is None:
            return None
        _path, step, time_s, _n = found
        return step, time_s

    def resumable_runs(self, graph_name: str) -> list[tuple[Path, int, float]]:
        """Runs of ``graph_name`` that have a checkpoint, newest first."""
        root = self.simulations_root() / graph_name
        if not root.is_dir():
            return []
        found: list[tuple[Path, int, float]] = []
        for run_dir in sorted(root.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            described = self.describe_checkpoint(run_dir)
            if described is not None:
                found.append((run_dir, described[0], described[1]))
        return found

    def refresh_resume_runs(self) -> None:
        combo = getattr(self, "resume_combo", None)
        if combo is None:
            return
        previous = combo.currentData()
        combo.clear()
        combo.addItem("(start a new run)", None)
        folder = self._selected_folder()
        if folder is None:
            return
        for run_dir, step, time_s in self.resumable_runs(folder.name):
            combo.addItem(f"{run_dir.name} - resume at step {step}, t={time_s:g}s", str(run_dir))
        if previous:
            index = combo.findData(previous)
            if index >= 0:
                combo.setCurrentIndex(index)

    def _sync_replot_enabled(self, *_: Any) -> None:
        """The button only means anything with a run selected."""
        button = getattr(self, "replot_button", None)
        if button is not None:
            button.setEnabled(self.selected_resume_dir() is not None)

    def _sync_initial_temperature_enabled(self, *_: Any) -> None:
        """Grey out the initial temperature while a resume target is selected.

        A resume takes its starting state from the checkpoint and ignores this field
        entirely (see the runner's _prepare). Leaving it live invites the reading
        that a resumed run starts from it -- exactly the question it raises when the
        form still shows 50.1 K beside a run that had got to 46 K.
        """
        resuming = self.selected_resume_dir() is not None
        for name in ("initial_spin", "use_initial", "initial_temperature_all_spin"):
            widget = getattr(self, name, None)
            if widget is None:
                continue
            widget.setEnabled(not resuming)
            if resuming:
                widget.setToolTip(
                    "Ignored while resuming: the checkpoint supplies the starting state. "
                    "Select '(fresh run)' in the resume row to use this."
                )

    def load_resume_run_settings(self, *_: Any) -> None:
        """Load the selected resume target's OWN settings into this form.

        Without this, picking a run to resume left the form showing the graph's saved
        parameters and a default setpoint table -- so continuing a run silently
        changed the setpoints, the controlled-sensor filter and the heater limits it
        had been running with. Everything needed is already written beside the run.

        Each part is loaded independently and reported, because a run may predate any
        of these files and a partial load beats none as long as it says what it
        could not do.
        """
        run_dir = self.selected_resume_dir()
        if run_dir is None:
            return
        # Only on a genuine change of target. refresh_resume_runs() rebuilds the combo
        # and re-selects what was selected before, which fires this again -- and a
        # second load would silently discard whatever the user had tuned since the
        # first one. Refreshing the list is not a request to revert the form.
        if getattr(self, "_resume_settings_loaded_from", None) == run_dir:
            return
        self._resume_settings_loaded_from = run_dir
        loaded: list[str] = []
        missing: list[str] = []

        params = None
        params_path = run_dir / "simulation_parameters.json"
        if params_path.is_file():
            try:
                from .simulation_parameters import load_simulation_parameters

                params, _extras = load_simulation_parameters(params_path)
                self.panel.set_params(params)
                self._params_source = f"loaded from {run_dir.name}/{params_path.name}"
                loaded.append("parameters")
            except Exception as exc:  # noqa: BLE001 - report, do not kill the tab
                missing.append(f"parameters ({exc})")
        else:
            missing.append("parameters")

        config: dict = {}
        config_path = run_dir / "config.json"
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8")) or {}
            except Exception:  # noqa: BLE001
                config = {}
        # These two live in config.json, not the parameter file, so set_params above
        # cannot have restored them.
        for key, attribute in (
            ("snapshot_interval_s", "snapshot_spin"),
            ("checkpoint_interval_s", "checkpoint_spin"),
        ):
            widget = getattr(self, attribute, None)
            if widget is not None and isinstance(config.get(key), (int, float)):
                widget.setValue(float(config[key]))
                loaded.append(key)

        if self._select_controller_for_resume(params, config):
            loaded.append("controller")

        setpoints = self._read_id_map(run_dir / "setpoints.json")
        if not setpoints:
            try:
                setpoints = {
                    int(k): float(v) for k, v in (config.get("setpoints_K") or {}).items()
                }
            except (TypeError, ValueError):
                setpoints = {}
        if setpoints and self._apply_setpoints(setpoints):
            loaded.append(f"{len(setpoints)} setpoint(s)")
        elif not setpoints:
            missing.append("setpoints")

        # None is the no-filter convention, so it is NOT "load nothing": it means
        # every controllable sensor was in the loop, and the ticks have to be put
        # back to that rather than left wherever the form happened to be.
        if params is not None:
            count = self._apply_enabled_sensors(params.enabled_sensor_node_ids)
            if count is not None:
                loaded.append(f"{count} controlled sensor(s)")

        overrides = self._read_heater_overrides(run_dir)
        if not overrides:
            overrides = config.get("heater_overrides") or {}
        if overrides and self._apply_heater_overrides(overrides):
            loaded.append(f"{len(overrides)} heater override(s)")

        self._sync_initial_temperature_enabled()
        summary = ", ".join(loaded) if loaded else "nothing"
        message = f"Loaded from {run_dir.name}: {summary}"
        if missing:
            message += f". NOT found: {', '.join(missing)} (the form's values stand)."
        self._status(message, bool(missing) and not loaded)

    @staticmethod
    def _read_id_map(path: Path) -> dict:
        """{int node id: float} from a run's json sidecar; {} if absent or unreadable."""
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8")) or {}
            return {int(k): float(v) for k, v in raw.items()}
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _read_heater_overrides(run_dir: Path) -> dict:
        path = run_dir / "heater_overrides.json"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            return {}

    def _select_controller_for_resume(self, params, config) -> bool:
        """Re-select the artifact the run used, matched by NAME rather than by path.

        A run's recorded path is the path on the machine that produced it, so
        matching it literally fails for any run copied between machines. The
        artifact's folder name is what actually identifies it.
        """
        combo = getattr(self, "controller_scheme_combo", None)
        if combo is None:
            return False
        wanted = ""
        for candidate in (
            getattr(params, "mimo_pi_gain_matrix_path", "") if params is not None else "",
            getattr(params, "modal_controller_path", "") if params is not None else "",
            config.get("controller_path") or "",
        ):
            if candidate:
                wanted = Path(str(candidate)).name
                break
        if not wanted:
            return False
        for index in range(combo.count()):
            data = combo.itemData(index)
            path = data[1] if isinstance(data, (tuple, list)) and len(data) > 1 else ""
            if path and Path(str(path)).name == wanted:
                combo.setCurrentIndex(index)
                return True
        self._status(
            f"The run used controller '{wanted}', which is not in this graph's list; "
            "the current selection stands.",
            True,
        )
        return False

    def _apply_setpoints(self, setpoints: dict) -> bool:
        table = getattr(self, "setpoint_table", None)
        rows = getattr(self, "_sensor_rows_manifest", None) or []
        if table is None or not rows:
            return False
        applied = 0
        for index, row in enumerate(rows):
            try:
                node_id = int(row["node_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if node_id in setpoints:
                value = f"{setpoints[node_id]:g}"
                table.setItem(index, 1, self.QtWidgets.QTableWidgetItem(value))
                applied += 1
        return applied > 0

    def _apply_enabled_sensors(self, enabled):
        """Restore the controlled ticks. Returns how many ended ticked, or None."""
        table = getattr(self, "setpoint_table", None)
        rows = getattr(self, "_sensor_rows_manifest", None) or []
        if table is None or not rows:
            return None
        allowed = None if enabled is None else {int(v) for v in enabled}
        ticked = 0
        for index, row in enumerate(rows):
            cell = table.item(index, 2)
            if cell is None:
                continue
            monitor = str(row.get("monitor_only", "")).lower() == "true"
            try:
                node_id = int(row["node_id"])
            except (KeyError, TypeError, ValueError):
                continue
            # No filter means every controllable sensor was in the loop; monitor-only
            # rows are not controlled under either convention.
            on = (not monitor) if allowed is None else (node_id in allowed)
            cell.setCheckState(
                self.QtCore.Qt.Checked if on else self.QtCore.Qt.Unchecked
            )
            ticked += int(on)
        return ticked

    def _apply_heater_overrides(self, overrides: dict) -> bool:
        table = getattr(self, "heater_table", None)
        rows = getattr(self, "_heater_rows_manifest", None) or []
        if table is None or not rows:
            return False
        columns = (
            (1, "heater_max_power_W"),
            (2, "heater_slew_rate_W_per_s"),
            (3, "heater_efficiency"),
            (4, "sensor_manual_power_W"),
        )
        by_id: dict = {}
        for key, fields in overrides.items():
            try:
                by_id[int(key)] = dict(fields or {})
            except (TypeError, ValueError):
                continue
        applied = 0
        for index, row in enumerate(rows):
            try:
                node_id = int(row["node_id"])
            except (KeyError, TypeError, ValueError):
                continue
            fields = by_id.get(node_id, {})
            # Cleared, not merged: a blank cell means "use the Controller defaults",
            # so a heater the run did not override must come back blank rather than
            # keeping an override from whatever the form showed before.
            for column, name in columns:
                value = fields.get(name)
                text = "" if value is None else f"{float(value):g}"
                table.setItem(index, column, self.QtWidgets.QTableWidgetItem(text))
            applied += int(bool(fields))
        return applied > 0

    def regenerate_run_plots(self) -> None:
        run_dir = self.selected_resume_dir()
        if run_dir is None:
            self._status("Select a run in the resume row first.", True)
            return
        try:
            from .simulation_runner import regenerate_plots

            written = regenerate_plots(run_dir)
        except FileNotFoundError as exc:
            self._status(str(exc), True)
            return
        except Exception as exc:  # noqa: BLE001 - report rather than kill the tab
            self._status(f"Could not regenerate plots: {exc}", True)
            return
        self._status(f"Rewrote {len(written)} plot(s) in {run_dir / 'plots'}.")

    def selected_resume_dir(self) -> Path | None:
        combo = getattr(self, "resume_combo", None)
        value = combo.currentData() if combo is not None else None
        return Path(value) if value else None

    def _fast_load_status(self, folder: Path) -> str:
        try:
            from .fast_edge_io import EDGES_FILENAME
            from .fast_graph_io import can_load_fast, edges_only_refresh_is_enough

            if (folder / EDGES_FILENAME).exists():
                usable, reason = can_load_fast(folder)
                if usable:
                    return (
                        f"fast load: READY - {EDGES_FILENAME} present "
                        "(temperature-dependent properties supported)"
                    )
                return f"fast load: unavailable - {reason} (runs use the full graph.json loader)"
            if edges_only_refresh_is_enough(folder):
                return (
                    f"fast load: MISSING {EDGES_FILENAME} - press 'Update graph'. Without it "
                    "runs fall back to the full graph.json loader (very large RAM)."
                )
            usable, reason = can_load_fast(folder)
            return f"fast load: unavailable - {reason} (press 'Update graph')"
        except Exception as exc:  # noqa: BLE001 - status line must never break the tab
            return f"fast load: unknown ({exc})"

    def _confirm_controller_ok(self, artifact: str) -> bool:
        """The runner enables the heater controller only when a controller artifact
        is present, so anything else is an open-loop run -- say so before launching
        an overnight job that never tracks the setpoint."""
        if artifact and Path(artifact).exists():
            return True
        detail = (
            f"The selected controller file is gone:\n\n{artifact}\n\n"
            if artifact
            else "No controller artifact is selected.\n\n"
        )
        reply = self.QtWidgets.QMessageBox.question(
            self.widget,
            "No controller",
            detail
            + "The run will be OPEN-LOOP: heaters stay off and nothing tracks the "
            "setpoint.\n\nRun anyway?",
            self.QtWidgets.QMessageBox.Yes | self.QtWidgets.QMessageBox.No,
            self.QtWidgets.QMessageBox.No,
        )
        return reply == self.QtWidgets.QMessageBox.Yes

    # -- run ---------------------------------------------------------------- #
    def start_run(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self._status("A headless run is already in progress.", True)
            return
        folder = self._selected_folder()
        if folder is None:
            self._status("Select a graph first.", True)
            return
        artifact = self.panel.selected_controller_artifact()
        if not self._confirm_controller_ok(artifact):
            return
        open_loop = not (artifact and Path(artifact).exists())
        # Resuming reuses the previous run's directory: run_simulation.py's
        # _resume_if_checkpoint picks up that directory's newest checkpoint, and the
        # run's plots / events.log / timeseries continue in place rather than
        # restarting at t=0. The parameters below are still written, so a resume is
        # also the way to continue with something changed (GPU off, longer duration).
        resume_dir = self.selected_resume_dir()
        if resume_dir is not None and not resume_dir.is_dir():
            self._status(f"Resume target no longer exists: {resume_dir}", True)
            return
        resume_at = self.describe_checkpoint(resume_dir) if resume_dir is not None else None
        if resume_dir is not None and resume_at is None:
            self._status(
                f"{resume_dir.name} has no usable checkpoint to resume from.", True
            )
            return
        run_dir = (
            resume_dir.resolve()
            if resume_dir is not None
            else (Path("simulations") / folder.name / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
        )
        # Persist the full parameter set beside the run and hand it to the process,
        # so the headless run uses exactly the physics shown in this tab (and the
        # file doubles as a record of what was run).
        params = self._collect_parameters()
        run_dir.mkdir(parents=True, exist_ok=True)
        params_path = run_dir / "simulation_parameters.json"
        # A resume reuses the directory, so writing the form's parameters here would
        # destroy the record of what the EARLIER leg ran with -- and that record is
        # the only way to interpret its half of the timeseries. Set it aside first.
        # (The runner does the same for provenance.json, but it cannot help here:
        # this write happens before the process is even launched.)
        if resume_dir is not None and params_path.is_file():
            kept = self._preserve_prior_parameters(params_path)
            if kept is not None:
                self._status(f"Earlier parameters kept as {kept.name}.", False)
        try:
            from .simulation_parameters import save_simulation_parameters

            save_simulation_parameters(params_path, params)
        except Exception as exc:  # noqa: BLE001
            self._status(f"Could not write the parameter file: {exc}", True)
            return
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "run_simulation.py"),
            "--graph", str(folder),
            "--run-dir", str(run_dir),
            "--sim-params", str(params_path),
            "--dt", f"{float(params.dt_s):g}",
            "--duration", f"{float(params.t_final_s):g}",
            "--snapshot-interval-s", f"{self.snapshot_spin.value():g}",
            "--checkpoint-interval-s", f"{self.checkpoint_spin.value():g}",
        ]
        if open_loop:
            command.append("--allow-no-controller")
        else:
            command += ["--controller", artifact]
        # The per-sensor table is the run's only source of setpoints -- there is no
        # global to fall back on -- so every filled row is written. Saved beside the
        # run so the file records what was actually asked for, the same way
        # simulation_parameters.json does.
        overrides = self.collect_setpoint_overrides()
        if overrides:
            overrides_path = run_dir / "setpoints.json"
            try:
                overrides_path.write_text(
                    json.dumps({str(k): v for k, v in sorted(overrides.items())}, indent=2),
                    encoding="utf-8",
                )
            except OSError as exc:
                self._status(f"Could not write per-sensor setpoints: {exc}", True)
                return
            command += ["--setpoints-json", str(overrides_path)]
        # Per-heater limit overrides, same idea: only heaters the user actually gave
        # a value are named, and the rest run on the Controller section's defaults.
        heater_overrides = self.collect_heater_overrides()
        if heater_overrides:
            heater_path = run_dir / "heater_overrides.json"
            try:
                heater_path.write_text(
                    json.dumps(
                        {str(k): v for k, v in sorted(heater_overrides.items())}, indent=2
                    ),
                    encoding="utf-8",
                )
            except OSError as exc:
                self._status(f"Could not write per-heater overrides: {exc}", True)
                return
            command += ["--heater-overrides-json", str(heater_path)]
        if self.use_initial.isChecked() and resume_dir is None:
            command += ["--initial-temp", f"{self.initial_spin.value():g}"]
        elif self.use_initial.isChecked():
            self._pending_log_note = (
                "resume: ignoring the initial-temperature override; the checkpoint "
                "supplies the starting state."
            )
        if not bool(getattr(params, "gpu_solver_enabled", True)):
            command.append("--no-gpu")
        if self.notes_edit.text().strip():
            command += ["--notes", self.notes_edit.text().strip()]

        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self._stop_requested = False
        # On a resume the run appends to the SAME events.log, so start reading from
        # the end of what is already there rather than replaying the whole history.
        events = run_dir / "events.log"
        self._log_size = events.stat().st_size if (resume_dir is not None and events.exists()) else 0
        self.log_view.clear()
        if resume_dir is not None and resume_at is not None:
            self.log_view.appendPlainText(
                f"--- resuming {run_dir.name} from step {resume_at[0]} (t={resume_at[1]:g}s) ---"
            )
        note = getattr(self, "_pending_log_note", "")
        if note:
            self.log_view.appendPlainText(note)
            self._pending_log_note = ""
        self.progress.setValue(0)
        try:
            # Detached: the run must outlive this window, and must not inherit the
            # GUI's console/stdin.
            creation = 0
            if os.name == "nt":
                creation = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
                    subprocess, "DETACHED_PROCESS", 0
                )
            self.process = subprocess.Popen(  # noqa: S603
                command,
                cwd=str(Path(__file__).resolve().parent.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=creation,
            )
        except Exception as exc:  # noqa: BLE001
            self._status(f"Could not start the run: {exc}", True)
            return
        log_event("headless tab started run", pid=self.process.pid, out=str(run_dir))
        self.run_headless_button.setEnabled(False)
        self.stop_headless_button.setEnabled(True)
        self.open_output_button.setEnabled(True)
        verb = (
            f"Resumed from step {resume_at[0]} (t={resume_at[1]:g}s)"
            if resume_dir is not None and resume_at is not None
            else "Started"
        )
        self.summary_label.setText(f"{verb} (pid {self.process.pid}) -> {run_dir}")
        self._status(f"Headless run {verb.lower()} (pid {self.process.pid}) -> {run_dir}", False)
        self.refresh_resume_runs()

    def stop_run(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        if self.run_dir is not None and not self._stop_requested:
            # Graceful: ask the run to stop at the next step boundary so it still
            # runs _finalize (plots, report, timeseries). A hard terminate() here
            # would skip all of that -- on Windows it is an uncatchable kill.
            self._stop_requested = True
            try:
                (self.run_dir / STOP_REQUEST_FILENAME).write_text("stop", encoding="utf-8")
                self._status(
                    "Stop requested — the run will finish the current step, then save its "
                    "plots and report and exit. Click Stop again to force-kill.",
                    False,
                )
                return
            except OSError as exc:
                self._status(f"Could not request a graceful stop ({exc}); force-killing.", True)
        # Second click, or the request file could not be written: hard kill. The run
        # keeps whatever it already wrote (snapshots/checkpoints) but skips finalize.
        try:
            self.process.terminate()
            self._status("Force-stopping the run (plots/report may be incomplete).", True)
        except Exception as exc:  # noqa: BLE001
            self._status(f"Could not stop the run: {exc}", True)

    def open_output(self) -> None:
        if self.run_dir is None or not self.run_dir.exists():
            return
        try:
            if os.name == "nt":
                os.startfile(str(self.run_dir))  # noqa: S606
            else:
                subprocess.Popen(["xdg-open", str(self.run_dir)])  # noqa: S603,S607
        except Exception as exc:  # noqa: BLE001
            self._status(f"Could not open the folder: {exc}", True)

    # -- monitoring (reads the run's own artifacts) ------------------------- #
    def _poll(self) -> None:
        self._poll_refresh()
        self._poll_modal_build()
        self._poll_gain_build()
        if self.run_dir is None:
            return
        self._tail_log()
        status_path = self.run_dir / "status.json"
        if status_path.exists():
            try:
                data = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = None
            if data:
                progress = float(data.get("progress", 0.0) or 0.0)
                self.progress.setValue(int(max(0.0, min(1.0, progress)) * 1000))
                eta = data.get("eta_s")
                eta_text = (
                    f", ETA {float(eta) / 60.0:.1f} min"
                    if isinstance(eta, (int, float)) and eta == eta and eta not in (float("inf"),)
                    else ""
                )
                rebuild = data.get("property_rebuild_ms")
                solve = data.get("model_solve_ms")
                profile_text = (
                    f"\nper step: rebuild {float(rebuild):.0f} ms, solve {float(solve):.0f} ms"
                    if isinstance(rebuild, (int, float)) and isinstance(solve, (int, float))
                    else ""
                )
                self.summary_label.setText(
                    f"{data.get('status', '?')} — t={float(data.get('sim_time_s', 0.0)):.1f}"
                    f"/{float(data.get('t_final_s', 0.0)):.0f} s"
                    f" ({progress * 100:.1f}%), step {data.get('step', 0)}"
                    f", RSS {data.get('rss_gib', 0.0)} GiB{eta_text}{profile_text}"
                )
        if self.process is not None and self.process.poll() is not None:
            code = self.process.returncode
            self.process = None
            self.run_headless_button.setEnabled(True)
            self.stop_headless_button.setEnabled(False)
            self._tail_log()
            self._status(f"Headless run finished (exit {code}); outputs in {self.run_dir}", code != 0)

    def _tail_log(self) -> None:
        if self.run_dir is None:
            return
        path = self.run_dir / "events.log"
        if not path.exists():
            return
        try:
            size = path.stat().st_size
            if size <= self._log_size:
                return
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self._log_size)
                new_text = handle.read()
            self._log_size = size
        except OSError:
            return
        for line in new_text.splitlines():
            if line.strip():
                self.log_view.appendPlainText(line.rstrip())

    def _status(self, message: str, is_error: bool) -> None:
        if self.on_status is None:
            return
        try:
            self.on_status(message, is_error)
        except Exception as exc:  # noqa: BLE001
            # This tab is constructed DURING the main window's _build_layout, so a
            # status raised from __init__ can reach a host whose status widget does
            # not exist yet (AttributeError: no attribute 'status_label') and take
            # the whole app down before it opens. Showing a status must never be
            # able to do that.
            log_event("headless tab status unavailable", error=repr(exc), status_text=message)
