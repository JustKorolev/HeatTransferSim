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

        self.dt_spin = self._double(1.0e-6, 1.0e6, 1.0, 0.1)
        form.addRow("dt s", self.dt_spin)
        self.duration_spin = self._double(1.0e-3, 1.0e12, 3600.0, 60.0)
        form.addRow("duration s", self.duration_spin)
        self.snapshot_spin = self._double(0.0, 1.0e12, 300.0, 60.0)
        form.addRow("snapshot every s", self.snapshot_spin)
        self.checkpoint_spin = self._double(0.0, 1.0e12, 600.0, 60.0)
        self.checkpoint_spin.setToolTip("Wall-clock seconds between resume checkpoints.")
        form.addRow("checkpoint every s", self.checkpoint_spin)

        self.gpu_checkbox = self.QtWidgets.QCheckBox("Use GPU solver when available")
        self.gpu_checkbox.setChecked(True)
        form.addRow(self.gpu_checkbox)

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
        self.graph_info.setText(
            f"{node_count} nodes • {len(controllers)} controller artifact(s) • {folder}"
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
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "run_simulation.py"),
            "--graph", str(folder),
            "--run-dir", str(run_dir),
            "--dt", f"{self.dt_spin.value():g}",
            "--duration", f"{self.duration_spin.value():g}",
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
        if not self.gpu_checkbox.isChecked():
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
