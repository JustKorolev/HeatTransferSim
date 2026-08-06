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
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .diagnostics import log_event


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
        self.QtCore = qt.QtCore
        self.QtWidgets = qt.QtWidgets
        self.on_status = on_status
        self._graphs_root = graphs_root or (lambda: Path.cwd() / "graphs")
        self.process: subprocess.Popen | None = None
        self.run_dir: Path | None = None
        self._log_size = 0
        self.widget = self.QtWidgets.QWidget(parent)
        self._build_layout()
        self.refresh_graphs()
        self._poll_timer = self.QtCore.QTimer(self.widget)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start(1000)

    # -- layout ------------------------------------------------------------- #
    def _build_layout(self) -> None:
        outer = self.QtWidgets.QHBoxLayout(self.widget)
        scroll = self.QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)
        controls = self.QtWidgets.QWidget()
        scroll.setWidget(controls)
        form = self.QtWidgets.QFormLayout(controls)

        intro = self.QtWidgets.QLabel(
            "Runs a simulation in a separate process. The graph is never loaded into "
            "this window, so a multi-million-cell run uses only the run's own memory "
            "and the GUI stays responsive. The run keeps going if you close the app."
        )
        intro.setWordWrap(True)
        form.addRow(intro)

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

        self.controller_combo = self.QtWidgets.QComboBox()
        self.controller_combo.setToolTip(
            "Controller artifact for the run. 'none (open-loop)' runs with no heater "
            "control -- useful for a passive cooldown, not for controller validation."
        )
        form.addRow("controller", self.controller_combo)

        self.setpoint_spin = self._double(0.0, 1.0e6, 293.15, 1.0)
        self.setpoint_spin.setToolTip(
            "Constant setpoint applied to EVERY sensor. Leave the 'use setpoint' box "
            "unchecked to keep whatever the graph already has."
        )
        self.use_setpoint = self.QtWidgets.QCheckBox("use setpoint")
        self.use_setpoint.setChecked(True)
        setpoint_row = self.QtWidgets.QHBoxLayout()
        setpoint_row.addWidget(self.setpoint_spin, 1)
        setpoint_row.addWidget(self.use_setpoint)
        form.addRow("setpoint K", setpoint_row)

        self.initial_spin = self._double(0.0, 1.0e6, 293.15, 1.0)
        self.use_initial = self.QtWidgets.QCheckBox("override")
        initial_row = self.QtWidgets.QHBoxLayout()
        initial_row.addWidget(self.initial_spin, 1)
        initial_row.addWidget(self.use_initial)
        form.addRow("initial T K", initial_row)

        self.snapshot_spin = self._double(0.0, 1.0e12, 300.0, 60.0)
        form.addRow("snapshot every s", self.snapshot_spin)
        self.checkpoint_spin = self._double(0.0, 1.0e12, 600.0, 60.0)
        self.checkpoint_spin.setToolTip("Wall-clock seconds between resume checkpoints.")
        form.addRow("checkpoint every s", self.checkpoint_spin)

        # FULL simulation parameters. A headless run must use the same physics as the
        # Heat Transfer Simulation tab -- radiation, temperature-dependent properties,
        # cryocooler, solver settings, controller gains -- not the runner's defaults,
        # or it silently simulates a different system. The form is generated from the
        # SimulationParameters dataclass so every field is exposed and it cannot drift
        # as fields are added; values load from the graph's own
        # simulation_parameters.json (what the GUI saved) and are written back out for
        # the run.
        params_box = self.QtWidgets.QGroupBox("Simulation parameters (full set)")
        params_outer = self.QtWidgets.QVBoxLayout(params_box)
        self.params_filter = self.QtWidgets.QLineEdit()
        self.params_filter.setPlaceholderText("filter parameters…")
        self.params_filter.textChanged.connect(self._filter_params)
        params_outer.addWidget(self.params_filter)
        params_scroll = self.QtWidgets.QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setMinimumHeight(280)
        params_host = self.QtWidgets.QWidget()
        self.params_form = self.QtWidgets.QFormLayout(params_host)
        params_scroll.setWidget(params_host)
        params_outer.addWidget(params_scroll)
        self.param_widgets: dict[str, Any] = {}
        self.param_rows: dict[str, tuple[Any, Any]] = {}
        self._build_parameter_widgets()
        form.addRow(params_box)

        self.notes_edit = self.QtWidgets.QLineEdit()
        self.notes_edit.setPlaceholderText("optional note stored with the run")
        form.addRow("notes", self.notes_edit)

        button_row = self.QtWidgets.QHBoxLayout()
        self.run_button = self.QtWidgets.QPushButton("Start Headless Run")
        self.run_button.clicked.connect(self.start_run)
        self.stop_button = self.QtWidgets.QPushButton("Stop Run")
        self.stop_button.clicked.connect(self.stop_run)
        self.stop_button.setEnabled(False)
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.stop_button)
        form.addRow(button_row)

        self.open_button = self.QtWidgets.QPushButton("Open Output Folder")
        self.open_button.clicked.connect(self.open_output)
        self.open_button.setEnabled(False)
        form.addRow(self.open_button)
        outer.addWidget(scroll)

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

    # -- full parameter form (generated from the dataclass) ----------------- #
    _CHOICES = {
        "input_mode": ["zero", "heater_inputs"],
        "mimo_controller_scheme": ["pid_qp", "modal_lqr"],
        "implicit_sparse_simulation_method": ["tr_bdf2", "backward_euler"],
    }
    # Set by the run itself; editing them here would be ignored or misleading.
    _SKIP_FIELDS = {
        "playback_speed", "loop_playback", "save_trajectory", "autoscale_temperature",
        "color_min_K", "color_max_K", "colormap", "simulation_history_limit",
        "live_step_profiling_enabled", "live_step_profile_threshold_ms",
        "browser_simulation_size_warning", "display_update_interval_ms",
        "enabled_heater_node_ids", "enabled_sensor_node_ids",
    }

    def _build_parameter_widgets(self) -> None:
        from dataclasses import fields as dataclass_fields

        from .simulation_parameters import SimulationParameters

        defaults = SimulationParameters()
        for spec in dataclass_fields(SimulationParameters):
            if spec.name in self._SKIP_FIELDS:
                continue
            value = getattr(defaults, spec.name)
            type_text = str(spec.type)
            if spec.name in self._CHOICES:
                widget = self.QtWidgets.QComboBox()
                widget.addItems(self._CHOICES[spec.name])
                widget.setCurrentText(str(value))
            elif type_text == "bool":
                widget = self.QtWidgets.QCheckBox()
                widget.setChecked(bool(value))
            elif type_text == "int":
                widget = self.QtWidgets.QSpinBox()
                widget.setRange(-2_000_000_000, 2_000_000_000)
                widget.setValue(int(value))
            elif type_text == "float":
                widget = self.QtWidgets.QDoubleSpinBox()
                widget.setDecimals(9)
                widget.setRange(-1.0e12, 1.0e12)
                widget.setValue(float(value))
            elif type_text == "str":
                widget = self.QtWidgets.QLineEdit(str(value))
            else:
                continue  # unsupported (e.g. tuple|None) -- left at its saved value
            label = self.QtWidgets.QLabel(spec.name)
            self.params_form.addRow(label, widget)
            self.param_widgets[spec.name] = widget
            self.param_rows[spec.name] = (label, widget)

    def _filter_params(self, text: str) -> None:
        needle = (text or "").strip().lower()
        for name, (label, widget) in self.param_rows.items():
            visible = needle in name.lower() if needle else True
            label.setVisible(visible)
            widget.setVisible(visible)

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
        for name, widget in self.param_widgets.items():
            value = getattr(params, name, None)
            if value is None:
                continue
            if isinstance(widget, self.QtWidgets.QComboBox):
                widget.setCurrentText(str(value))
            elif isinstance(widget, self.QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, self.QtWidgets.QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, self.QtWidgets.QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, self.QtWidgets.QLineEdit):
                widget.setText(str(value))
        self._params_source = source

    def _collect_parameters(self):
        """Current form values as a SimulationParameters (unsupported fields keep
        whatever the graph's saved file had)."""
        from .simulation_parameters import SimulationParameters, load_simulation_parameters

        folder = self._selected_folder()
        path = (folder / "simulation_parameters.json") if folder else None
        if path is not None and path.is_file():
            params, _extras = load_simulation_parameters(path)
        else:
            params = SimulationParameters()
        for name, widget in self.param_widgets.items():
            if isinstance(widget, self.QtWidgets.QComboBox):
                setattr(params, name, widget.currentText())
            elif isinstance(widget, self.QtWidgets.QCheckBox):
                setattr(params, name, bool(widget.isChecked()))
            elif isinstance(widget, self.QtWidgets.QSpinBox):
                setattr(params, name, int(widget.value()))
            elif isinstance(widget, self.QtWidgets.QDoubleSpinBox):
                setattr(params, name, float(widget.value()))
            elif isinstance(widget, self.QtWidgets.QLineEdit):
                setattr(params, name, widget.text())
        return params

    def _double(self, minimum: float, maximum: float, value: float, step: float) -> Any:
        spin = self.QtWidgets.QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

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
        self.controller_combo.clear()
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
        controllers = sorted(p.name for p in folder.glob("*.npz") if "controller" in p.name.lower())
        self.controller_combo.addItems(controllers or [])
        self.controller_combo.addItem("none (open-loop)")
        if controllers:
            self.controller_combo.setCurrentIndex(0)
        self._load_parameters_for_graph(folder)
        self.graph_info.setText(
            f"{node_count} nodes • {len(controllers)} controller artifact(s) • {folder}\n"
            f"parameters: {getattr(self, '_params_source', 'defaults')}"
        )

    # -- run ---------------------------------------------------------------- #
    def start_run(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self._status("A headless run is already in progress.", True)
            return
        folder = self._selected_folder()
        if folder is None:
            self._status("Select a graph first.", True)
            return
        controller = self.controller_combo.currentText()
        open_loop = (not controller) or controller.startswith("none")
        if open_loop:
            reply = self.QtWidgets.QMessageBox.question(
                self.widget,
                "No controller",
                "No controller artifact is selected, so the run will be OPEN-LOOP: "
                "heaters stay off and nothing tracks the setpoint.\n\nRun anyway?",
                self.QtWidgets.QMessageBox.Yes | self.QtWidgets.QMessageBox.No,
                self.QtWidgets.QMessageBox.No,
            )
            if reply != self.QtWidgets.QMessageBox.Yes:
                return
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
            command += ["--controller", str(folder / controller)]
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
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.summary_label.setText(f"Started (pid {self.process.pid}) -> {run_dir}")
        self._status(f"Headless run started (pid {self.process.pid}) -> {run_dir}", False)

    def stop_run(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        # terminate() lets the runner's signal handler finalise its artifacts.
        try:
            self.process.terminate()
            self._status("Stopping the headless run (it will finalise its outputs)...", False)
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
                self.summary_label.setText(
                    f"{data.get('status', '?')} — t={float(data.get('sim_time_s', 0.0)):.1f}"
                    f"/{float(data.get('t_final_s', 0.0)):.0f} s"
                    f" ({progress * 100:.1f}%), step {data.get('step', 0)}"
                    f", RSS {data.get('rss_gib', 0.0)} GiB{eta_text}"
                )
        if self.process is not None and self.process.poll() is not None:
            code = self.process.returncode
            self.process = None
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
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
