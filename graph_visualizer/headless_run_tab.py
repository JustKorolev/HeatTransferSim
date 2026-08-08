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

import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .diagnostics import log_event
from .modal_reduction import list_modal_artifacts
from .simulation_controls_panel import MODE_HEADLESS, PID_QP_LABEL, SimulationControlsPanel
from .simulation_runner import STOP_REQUEST_FILENAME


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
        self._log_size = 0
        self._params_source = "defaults"
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
            },
        )
        self.panel.build(form)
        # The same per-heater/sensor/cryocooler "Parameters" editor the simulation
        # tab shows beside its viewer. No readout tables here to drive per-row
        # selection, so the shared panel shows it as a static defaults block.
        self.panel.build_readout_editor()
        self.panel.export_to(self)
        # controls_scroll is NOT added here: app.py puts it in the window's shared
        # side-panel stack, exactly as it does for the simulation tab. The tab body
        # holds only what sits beside the viewer there -- the "Parameters" editor
        # and, in place of the 3D viewer, the run's progress / status / log.
        outer.addWidget(self.readout_editor_box, 0, self.QtCore.Qt.AlignTop)

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
        """The sim tab assigns each sensor a random setpoint on the loaded model.
        The headless subprocess only takes one constant ``--setpoint`` for all
        sensors, so this draws one value from center +/- spread and applies it to
        the whole run. Per-sensor randomization needs the graph loaded (which this
        tab deliberately never does)."""
        center = float(self.sensor_random_center_spin.value())
        spread_K = float(self.sensor_random_spread_mK_spin.value()) * 1.0e-3
        value = center + random.uniform(-spread_K, spread_K)
        self.setpoint_spin.setValue(value)
        self.use_setpoint.setChecked(True)
        self._status(
            f"Run setpoint randomized to {value:g} K "
            f"(one value for all sensors; per-sensor needs the graph loaded).",
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

    def _collect_parameters(self):
        """Current form values as a SimulationParameters (unsupported fields keep
        whatever the graph's saved file had)."""
        from .simulation_parameters import SimulationParameters, load_simulation_parameters

        folder = self._selected_folder()
        path = (folder / "simulation_parameters.json") if folder else None
        if path is not None and path.is_file():
            base, _extras = load_simulation_parameters(path)
        else:
            base = SimulationParameters()
        # Widgets win; fields without a widget keep the graph's saved value.
        return self.panel.read(base)

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
        artifacts = list_modal_artifacts(folder)
        self.controller_scheme_combo.addItem(PID_QP_LABEL, None)
        for info in artifacts:
            self.controller_scheme_combo.addItem(info.label, str(info.path))
        if artifacts:
            self.controller_scheme_combo.setCurrentIndex(1)
        self._load_parameters_for_graph(folder)
        # Say plainly whether the run will take the fast path and whether the edge
        # data is there. Without edges.npz, temperature-dependent properties would
        # have to fall back to constant properties (L(T) is rebuilt from the edges),
        # and the run silently models different physics -- so surface it up front
        # rather than leaving it to be discovered in events.log.
        self.graph_info.setText(
            f"{node_count} nodes • {len(artifacts)} controller artifact(s) • {folder}\n"
            f"parameters: {self._params_source}\n"
            f"{self._fast_load_status(folder)}"
        )

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
        run_dir = (
            Path("simulations") / folder.name / datetime.now().strftime("%Y%m%d-%H%M%S")
        ).resolve()
        # Persist the full parameter set beside the run and hand it to the process,
        # so the headless run uses exactly the physics shown in this tab (and the
        # file doubles as a record of what was run).
        params = self._collect_parameters()
        run_dir.mkdir(parents=True, exist_ok=True)
        params_path = run_dir / "simulation_parameters.json"
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
        if self.use_setpoint.isChecked():
            command += ["--setpoint", f"{self.setpoint_spin.value():g}"]
        if self.use_initial.isChecked():
            command += ["--initial-temp", f"{self.initial_spin.value():g}"]
        if not bool(getattr(params, "gpu_solver_enabled", True)):
            command.append("--no-gpu")
        if self.notes_edit.text().strip():
            command += ["--notes", self.notes_edit.text().strip()]

        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self._stop_requested = False
        self._log_size = 0
        self.log_view.clear()
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
        self.summary_label.setText(f"Started (pid {self.process.pid}) -> {run_dir}")
        self._status(f"Headless run started (pid {self.process.pid}) -> {run_dir}", False)

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
        if self.on_status is not None:
            self.on_status(message, is_error)
