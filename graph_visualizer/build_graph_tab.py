"""Build a thermal graph from a CAD assembly, from inside the application.

This was CLI-only, which meant the first step of the pipeline was the one step a
new user could not reach from the app: they had to find the README, assemble a
twenty-five flag command, and get every flag right before they had anything to
open. The parameters are the same ones ``hts-build-graph`` takes -- the spec in
:mod:`build_graph_params` is checked against the real argument parser by a test,
so the form cannot drift from the builder.

The build runs as a DETACHED subprocess, like the modal-controller and G-matrix
builds. Three reasons, all learned the hard way: a full assembly takes a long
time, the voxelizer is memory-hungry enough to take a machine down with it, and a
build that dies with the window is a build you have to start again. The tab holds
no geometry itself and only tails the log the builder writes.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

from .build_graph_params import (
    BUILD_FIELDS,
    MESH_ROOT,
    build_argv,
    default_values,
    step_folders,
)
from .ui_theme import configure_double_spin, widen_decimals_for, wrapping_label

#: The builder, invoked as a module so it resolves from a wheel as well as a
#: checkout (see graph_visualizer.cli for why a script path does not).
BUILDER_MODULE = "octree_graph.cli"
#: The builder writes this into the output folder; it is the whole progress story.
CONVERSION_LOG = "conversion.log"


class BuildGraphTab:
    """CAD -> octree -> thermal graph, with the builder's parameters exposed."""

    def __init__(
        self,
        qt: Any,
        parent: Any,
        *,
        on_status: Any = None,
        mesh_root: Any = None,
        on_graph_built: Any = None,
    ) -> None:
        self.QtCore = qt.QtCore
        self.QtWidgets = qt.QtWidgets
        self.on_status = on_status
        self.on_graph_built = on_graph_built
        self._mesh_root = mesh_root or (lambda: Path.cwd() / MESH_ROOT)
        self.process: subprocess.Popen | None = None
        self.inputs: dict[str, Any] = {}
        self._log_size = 0
        self._output_folder: Path | None = None

        self.widget = self.QtWidgets.QWidget(parent)
        self._build_layout()
        self.refresh_folders()

        self._poll_timer = self.QtCore.QTimer(self.widget)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start(1000)

    # -- layout -------------------------------------------------------------- #
    def _build_layout(self) -> None:
        outer = self.QtWidgets.QHBoxLayout(self.widget)

        # Controls live in the window's shared side panel, like the simulation and
        # headless tabs; app.py puts controls_scroll into side_panel_stack.
        self.controls_scroll = self.QtWidgets.QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setMinimumWidth(400)
        controls = self.QtWidgets.QWidget()
        form = self.QtWidgets.QFormLayout(controls)
        self.controls_scroll.setWidget(controls)

        intro = wrapping_label(
            self.QtCore,
            self.QtWidgets,
            "Builds a lumped thermal graph from a STEP assembly. Put the assembly in "
            f"a folder under {MESH_ROOT}/ with its Materials.xlsx beside it, and it "
            "appears below.",
        )
        form.addRow(intro)

        source_box, source_form = self._section("Source")
        form.addRow(source_box)
        self.folder_combo = self.QtWidgets.QComboBox()
        self.folder_combo.currentTextChanged.connect(self._handle_folder_changed)
        refresh = self.QtWidgets.QPushButton("Rescan")
        refresh.clicked.connect(lambda: self.refresh_folders(announce=True))
        source_form.addRow("assembly folder", self._hrow(self.folder_combo, refresh))
        self.source_info = wrapping_label(self.QtCore, self.QtWidgets, "")
        source_form.addRow(self.source_info)
        self.graph_name_input = self.QtWidgets.QLineEdit()
        self.graph_name_input.setToolTip(
            "Folder name under the output root. An existing graph of the same name is "
            "overwritten, so name variants rather than rebuilding over a good graph."
        )
        source_form.addRow("graph name", self.graph_name_input)
        self.output_root_input = self.QtWidgets.QLineEdit("graphs")
        source_form.addRow("output root", self.output_root_input)

        # Every editable parameter, grouped as the build proceeds.
        for title, fields in BUILD_FIELDS:
            box, section_form = self._section(title)
            form.addRow(box)
            for field in fields:
                widget = self._widget_for(field)
                self.inputs[field.dest] = widget
                if field.tooltip:
                    widget.setToolTip(field.tooltip)
                if field.kind == "bool":
                    # A checkbox carries its own text, so a row label beside it
                    # just prints the name twice. Matches the simulation panel.
                    section_form.addRow(widget)
                else:
                    section_form.addRow(field.label, widget)

        action_box, action_form = self._section("Build")
        form.addRow(action_box)
        self.build_button = self.QtWidgets.QPushButton("Build Graph")
        self.build_button.setToolTip(
            "Runs the builder as a separate, detached process: it survives closing "
            "this window, and a crash in it cannot take the application down."
        )
        self.build_button.clicked.connect(self.start_build)
        self.stop_button = self.QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_build)
        action_form.addRow(self._hrow(self.build_button, self.stop_button))
        self.status_label = wrapping_label(self.QtCore, self.QtWidgets, "Idle.")
        action_form.addRow("status", self.status_label)

        outer.addWidget(self._log_panel(), 1)

    def _log_panel(self) -> Any:
        panel = self.QtWidgets.QWidget(self.widget)
        layout = self.QtWidgets.QVBoxLayout(panel)

        self.command_view = self.QtWidgets.QPlainTextEdit()
        self.command_view.setReadOnly(True)
        self.command_view.setMaximumHeight(110)
        self.command_view.setPlaceholderText(
            "The exact command this tab will run appears here, so it can be checked, "
            "copied, or re-run in a terminal."
        )
        layout.addWidget(self.QtWidgets.QLabel("Command"))
        layout.addWidget(self.command_view)

        layout.addWidget(self.QtWidgets.QLabel(f"Progress ({CONVERSION_LOG})"))
        self.log_view = self.QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        layout.addWidget(self.log_view, 1)
        return panel

    def _section(self, title: str) -> tuple[Any, Any]:
        box = self.QtWidgets.QGroupBox(title)
        box.setStyleSheet("QGroupBox { font-weight: 700; margin-top: 8px; }")
        return box, self.QtWidgets.QFormLayout(box)

    def _hrow(self, *widgets: Any) -> Any:
        container = self.QtWidgets.QWidget()
        layout = self.QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            layout.addWidget(widget)
        return container

    def _widget_for(self, field: Any) -> Any:
        if field.kind == "bool":
            widget = self.QtWidgets.QCheckBox(field.label)
            widget.setChecked(bool(field.default))
            widget.toggled.connect(self._refresh_command)
            return widget
        if field.kind == "text":
            widget = self.QtWidgets.QLineEdit(str(field.default or ""))
            widget.textChanged.connect(self._refresh_command)
            return widget
        if field.kind == "int":
            widget = self.QtWidgets.QSpinBox()
            widget.setRange(int(field.minimum), int(min(field.maximum, 2_147_483_647)))
            widget.setSingleStep(int(field.step) or 1)
            widget.setValue(int(field.default))
            widget.valueChanged.connect(self._refresh_command)
            return widget

        class _NoWheelDoubleSpinBox(self.QtWidgets.QDoubleSpinBox):
            def wheelEvent(inner_self, event: Any) -> None:  # noqa: N802 - Qt name
                event.ignore()

            def setValue(inner_self, number: Any) -> None:  # noqa: N802 - Qt name
                widen_decimals_for(inner_self, number)
                super().setValue(number)

        widget = _NoWheelDoubleSpinBox()
        widget.setRange(float(field.minimum), float(field.maximum))
        widget.setSingleStep(float(field.step))
        configure_double_spin(widget, field.step, field.default)
        widget.setValue(float(field.default))
        widget.valueChanged.connect(self._refresh_command)
        return widget

    # -- source selection ---------------------------------------------------- #
    def refresh_folders(self, announce: bool = False) -> None:
        root = Path(self._mesh_root())
        previous = self.folder_combo.currentText()
        self.folder_combo.blockSignals(True)
        self.folder_combo.clear()
        folders = step_folders(root)
        self.folder_combo.addItems(folders)
        if previous in folders:
            self.folder_combo.setCurrentText(previous)
        self.folder_combo.blockSignals(False)
        if not folders:
            self.source_info.setText(
                f"No STEP assemblies found in {root}. Put a .step or .stp file in a "
                f"subfolder of {MESH_ROOT}/, with its Materials.xlsx beside it."
            )
            self._refresh_command()
            if announce:
                self._status(f"No STEP assemblies under {root}.", True)
            return
        self._handle_folder_changed(self.folder_combo.currentText())
        if announce:
            self._status(f"{len(folders)} assembly folder(s) under {root}.")

    def _handle_folder_changed(self, name: str) -> None:
        if not name:
            return
        folder = Path(self._mesh_root()) / name
        steps = [
            item
            for item in sorted(folder.glob("*"))
            if item.is_file() and item.suffix.lower() in (".step", ".stp")
        ]
        lookup = [
            item for item in folder.glob("*") if item.suffix.lower() in (".xlsx", ".xls")
        ]
        size_mb = sum(item.stat().st_size for item in steps) / 1048576 if steps else 0.0
        parts = [f"{steps[0].name} ({size_mb:.1f} MB)" if steps else "no STEP file"]
        if lookup:
            parts.append(f"materials: {lookup[0].name}")
        else:
            # Worth saying plainly: without the lookup every part falls back to the
            # unassigned default, and the resulting conductances are fiction.
            parts.append(
                "no Materials.xlsx -- every part will use the unassigned default "
                "material, so the conductances will not describe your assembly"
            )
        self.source_info.setText(". ".join(parts) + ".")
        if not self.graph_name_input.text().strip():
            self.graph_name_input.setText(name.upper())
        self._refresh_command()

    # -- the command --------------------------------------------------------- #
    def current_argv(self) -> list[str]:
        values = {}
        for dest, widget in self.inputs.items():
            if hasattr(widget, "isChecked"):
                values[dest] = bool(widget.isChecked())
            elif hasattr(widget, "value"):
                values[dest] = widget.value()
            else:
                values[dest] = widget.text()
        return build_argv(
            str(Path(self._mesh_root()) / self.folder_combo.currentText()),
            self.graph_name_input.text().strip(),
            self.output_root_input.text().strip() or "graphs",
            values,
        )

    def _refresh_command(self, *_: Any) -> None:
        if not hasattr(self, "command_view"):
            return
        argv = self.current_argv()
        rendered = " ".join(_quote(part) for part in ["hts-build-graph", *argv])
        self.command_view.setPlainText(rendered)

    # -- running ------------------------------------------------------------- #
    def start_build(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self._status("A build is already running.", True)
            return
        folder = self.folder_combo.currentText().strip()
        if not folder:
            self._status("Select an assembly folder first.", True)
            return
        graph_name = self.graph_name_input.text().strip()
        if not graph_name:
            self._status("Give the graph a name.", True)
            return

        output_root = Path(self.output_root_input.text().strip() or "graphs")
        self._output_folder = output_root / graph_name
        self._output_folder.mkdir(parents=True, exist_ok=True)
        self._log_size = 0
        self.log_view.clear()

        command = [sys.executable, "-m", BUILDER_MODULE, *self.current_argv()]
        try:
            creation = 0
            if sys.platform == "win32":
                # Its own process group, so a Ctrl+C in the launching terminal --
                # or closing this window -- does not kill a build that may have
                # hours left in it.
                creation = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
                    subprocess, "DETACHED_PROCESS", 0
                )
            self.process = subprocess.Popen(  # noqa: S603
                command,
                cwd=str(Path.cwd()),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=creation,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self._status(f"Could not start the build: {exc}", True)
            self.process = None
            return

        self.build_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText(f"Building {graph_name}...")
        self._status(
            f"Building {graph_name} from {folder}. Separate process; progress in "
            f"{self._output_folder / CONVERSION_LOG}."
        )

    def stop_build(self) -> None:
        proc = self.process
        if proc is None or proc.poll() is not None:
            self._status("No build is running.", True)
            return
        try:
            proc.terminate()
        except Exception as exc:  # noqa: BLE001
            self._status(f"Could not stop the build: {exc}", True)
            return
        self.status_label.setText("Stopping...")
        self._status("Asked the build to stop; the partial output folder is left in place.")

    # -- monitoring ---------------------------------------------------------- #
    def _poll(self) -> None:
        self._tail_log()
        proc = self.process
        if proc is None or proc.poll() is None:
            return
        code = proc.returncode
        self.process = None
        self.build_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._tail_log()
        if code == 0:
            name = self.graph_name_input.text().strip()
            self.status_label.setText(f"Built {name}.")
            self._status(f"Built {name} in {self._output_folder}.")
            if self.on_graph_built is not None:
                try:
                    self.on_graph_built(self._output_folder)
                except Exception:  # noqa: BLE001 - a listener must not break the tab
                    pass
        else:
            self.status_label.setText(f"Build failed (exit {code}).")
            self._status(
                f"Build failed (exit {code}); the reason is at the end of "
                f"{CONVERSION_LOG}, shown on the right.",
                True,
            )

    def _tail_log(self) -> None:
        """Append whatever the builder has written since the last poll.

        Reads from a byte offset rather than re-reading the file: a long build's
        log grows to megabytes, and re-reading it every second would make the UI
        the slowest part of the build.
        """
        if self._output_folder is None:
            return
        path = self._output_folder / CONVERSION_LOG
        if not path.is_file():
            return
        try:
            size = path.stat().st_size
            if size < self._log_size:
                # Truncated or replaced -- a new build into the same folder.
                self._log_size = 0
                self.log_view.clear()
            if size == self._log_size:
                return
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self._log_size)
                chunk = handle.read()
                self._log_size = handle.tell()
        except OSError:
            return
        for line in chunk.splitlines():
            if line.strip():
                self.log_view.appendPlainText(line)

    def shutdown(self) -> None:
        """Stop polling. The BUILD is deliberately left running -- it is detached,
        and a user closing the window did not ask to discard hours of work."""
        try:
            self._poll_timer.stop()
        except Exception:  # noqa: BLE001
            pass

    def _status(self, message: str, error: bool = False) -> None:
        if self.on_status is not None:
            self.on_status(message, error)


def _quote(part: str) -> str:
    """Quote a command part only when it needs it, so the line stays readable."""
    text = str(part)
    return f'"{text}"' if (not text or " " in text) else text
