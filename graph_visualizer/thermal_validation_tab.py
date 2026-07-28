"""Qt tab for built-in thermal validation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

try:  # pragma: no cover - import path depends on the installed Qt binding.
    from PySide6 import QtGui
except Exception:  # pragma: no cover
    from qtpy import QtGui

from .material_library import default_material_library
from .pyvista_widget import GraphPyVistaWidget
from .thermal_validation import (
    DISTRIBUTED_ROD,
    GEOMETRY_CONTACT_PAIR,
    ONE_D_PRISM,
    TWO_NODE_LUMPED,
    TWO_BLOCK_EXCHANGE,
    VALIDATION_EXPERIMENTS,
    ThermalValidationParameters,
    ValidationBuildResult,
    ValidationRunResult,
    experiments_by_name,
    export_validation_result,
)


class ThermalValidationTab:
    """Built-in analytical validation workflow."""

    def __init__(
        self,
        qt: Any,
        parent: Any,
        *,
        on_status: Callable[[str, bool], None] | None = None,
    ) -> None:
        self.QtCore = qt.QtCore
        self.QtGui = getattr(qt, "QtGui", None)
        self.QtWidgets = qt.QtWidgets
        self.on_status = on_status
        self.experiments = experiments_by_name()
        self.build_result: ValidationBuildResult | None = None
        self.run_result: ValidationRunResult | None = None
        self.stop_requested = False
        self.widget = self.QtWidgets.QWidget(parent)
        self.controls_scroll = self.QtWidgets.QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setMinimumWidth(340)
        self.inputs: dict[str, Any] = {}
        self._build_layout()
        self._handle_experiment_changed()

    def _build_layout(self) -> None:
        controls = self.QtWidgets.QWidget()
        self.controls_scroll.setWidget(controls)
        form = self.QtWidgets.QFormLayout(controls)

        self.experiment_combo = self.QtWidgets.QComboBox()
        self.experiment_combo.addItems(list(VALIDATION_EXPERIMENTS))
        self.experiment_combo.currentTextChanged.connect(self._handle_experiment_changed)
        form.addRow("experiment", self.experiment_combo)

        self.description_label = self.QtWidgets.QLabel("")
        self.description_label.setWordWrap(True)
        form.addRow(self.description_label)
        self.equation_label = self.QtWidgets.QLabel("")
        self.equation_label.setWordWrap(True)
        form.addRow("reference", self.equation_label)

        self.material_combo = self.QtWidgets.QComboBox()
        materials = sorted(default_material_library())
        self.material_combo.addItems(materials)
        if "Copper" in materials:
            self.material_combo.setCurrentText("Copper")
        elif "copper" in materials:
            self.material_combo.setCurrentText("copper")
        self.material_combo.currentTextChanged.connect(self._refresh_material_properties)
        form.addRow("material", self.material_combo)
        self.material_label = self.QtWidgets.QLabel("")
        self.material_label.setWordWrap(True)
        form.addRow("properties", self.material_label)
        self.reference_temperature = self._double_spin(0.0, 1.0e6, 293.15, 1.0)
        form.addRow("reference T K", self.reference_temperature)

        for name, label, value, step in (
            ("length_mm", "length mm", 100.0, 1.0),
            ("width_mm", "width mm", 20.0, 1.0),
            ("height_mm", "height mm", 20.0, 1.0),
            ("initial_temperature_K", "initial T K", 293.15, 1.0),
            ("duration_s", "duration s", 100.0, 1.0),
            ("dt_s", "dt s", 0.1, 0.01),
            ("output_sample_interval_s", "sample interval s", 1.0, 0.1),
            ("voxel_min_size_mm", "voxel min mm", 5.0, 1.0),
            ("voxel_max_size_mm", "voxel max mm", 10.0, 1.0),
            ("absolute_tolerance_K", "abs tol K", 0.05, 0.01),
            ("relative_tolerance", "rel tol", 1.0e-3, 1.0e-4),
        ):
            self.inputs[name] = self._double_spin(0.0 if "temperature" not in name else 1.0e-9, 1.0e9, value, step)
            form.addRow(label, self.inputs[name])
        self.inputs["max_octree_depth"] = self._int_spin(1, 32, 8)
        self.inputs["samples_per_cell"] = self._int_spin(1, 1000, 9)
        form.addRow("max octree depth", self.inputs["max_octree_depth"])
        form.addRow("samples per cell", self.inputs["samples_per_cell"])
        self.use_octree_checkbox = self.QtWidgets.QCheckBox("Use generated GLB and octree importer")
        self.use_octree_checkbox.setChecked(True)
        form.addRow(self.use_octree_checkbox)

        solver_box = self.QtWidgets.QGroupBox("Solver (mirrors the live simulator)")
        solver_form = self.QtWidgets.QFormLayout(solver_box)
        self.inputs["solver_adaptive_max_substeps"] = self._int_spin(1, 100000, 4)
        self.inputs["solver_adaptive_target_delta_K"] = self._double_spin(1.0e-6, 1.0e9, 1.0, 0.1)
        self.inputs["solver_rtol"] = self._double_spin(0.0, 1.0, 1.0e-6, 1.0e-6)
        self.inputs["copper_rrr"] = self._int_spin(1, 100000, 100)
        solver_form.addRow("max substeps", self.inputs["solver_adaptive_max_substeps"])
        solver_form.addRow("target dT/substep K", self.inputs["solver_adaptive_target_delta_K"])
        solver_form.addRow("linear rtol", self.inputs["solver_rtol"])
        self.gpu_solver_checkbox = self.QtWidgets.QCheckBox("Use GPU solver when available")
        self.gpu_solver_checkbox.setChecked(True)
        solver_form.addRow(self.gpu_solver_checkbox)
        self.radiation_checkbox = self.QtWidgets.QCheckBox("Enable ambient radiation")
        solver_form.addRow(self.radiation_checkbox)
        self.tdep_checkbox = self.QtWidgets.QCheckBox("Temperature-dependent cp(T)/k(T)")
        solver_form.addRow(self.tdep_checkbox)
        self.midpoint_checkbox = self.QtWidgets.QCheckBox(
            "Midpoint property/radiation coupling (2nd-order splitting)"
        )
        self.midpoint_checkbox.setChecked(True)
        solver_form.addRow(self.midpoint_checkbox)
        solver_form.addRow("copper RRR", self.inputs["copper_rrr"])
        form.addRow(solver_box)

        self.specific_stack = self.QtWidgets.QStackedWidget()
        self.block_panel = self._block_specific_panel()
        self.two_block_panel = self._two_block_specific_panel()
        self.prism_panel = self._prism_specific_panel()
        self.specific_stack.addWidget(self.block_panel)
        self.specific_stack.addWidget(self.two_block_panel)
        self.specific_stack.addWidget(self.prism_panel)
        form.addRow(self.specific_stack)

        button_row = self.QtWidgets.QHBoxLayout()
        for text, callback in (
            ("Build Experiment", self.build_experiment),
            ("Run Validation", self.run_validation),
            ("Stop", self.stop),
            ("Reset", self.reset),
        ):
            button = self.QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            button_row.addWidget(button)
        form.addRow(button_row)
        export = self.QtWidgets.QPushButton("Export Results")
        export.clicked.connect(self.export_results)
        form.addRow(export)

        self.status_label = self.QtWidgets.QLabel("Select an experiment and build it.")
        self.status_label.setWordWrap(True)
        form.addRow(self.status_label)

        layout = self.QtWidgets.QVBoxLayout(self.widget)
        self.viewer = GraphPyVistaWidget(
            self.widget,
            on_pick_node=None,
            tooltip_for_node=lambda _node_id: "",
        )
        self.viewer.set_toggles(False, False, True, False, False)
        self.plot = ValidationPlotWidget(self.QtWidgets, self.QtCore, self.widget)
        self.summary_label = self.QtWidgets.QLabel("No validation run.")
        self.summary_label.setWordWrap(True)
        self.metrics_table = self.QtWidgets.QTableWidget(0, 4)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Measured value", "Tolerance", "Status"])
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(self.QtWidgets.QAbstractItemView.NoEditTriggers)
        self.metrics_table.setSelectionBehavior(self.QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(self.viewer.interactor, 2)
        layout.addWidget(self.plot.widget, 1)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.metrics_table, 1)

    def _block_specific_panel(self) -> Any:
        box = self.QtWidgets.QGroupBox("Insulated Block")
        form = self.QtWidgets.QFormLayout(box)
        self.inputs["heater_power_W"] = self._double_spin(0.0, 1.0e9, 10.0, 1.0)
        form.addRow("heater power W", self.inputs["heater_power_W"])
        return box

    def _two_block_specific_panel(self) -> Any:
        box = self.QtWidgets.QGroupBox("Two-Block Exchange")
        form = self.QtWidgets.QFormLayout(box)
        self.inputs["hot_initial_temperature_K"] = self._double_spin(0.0, 1.0e6, 300.0, 1.0)
        self.inputs["cold_initial_temperature_K"] = self._double_spin(0.0, 1.0e6, 200.0, 1.0)
        self.inputs["interface_conductance_W_K"] = self._double_spin(0.0, 1.0e9, 0.1, 0.01)
        self.interface_model = self.QtWidgets.QComboBox()
        self.interface_model.addItems(["explicit_total_conductance", "geometry_derived_conductance"])
        form.addRow("hot initial K", self.inputs["hot_initial_temperature_K"])
        form.addRow("cold initial K", self.inputs["cold_initial_temperature_K"])
        form.addRow("interface G W/K", self.inputs["interface_conductance_W_K"])
        form.addRow("interface model", self.interface_model)
        return box

    def _prism_specific_panel(self) -> Any:
        box = self.QtWidgets.QGroupBox("One-Dimensional Prism")
        form = self.QtWidgets.QFormLayout(box)
        self.inputs["surface_temperature_K"] = self._double_spin(0.0, 1.0e6, 200.0, 1.0)
        # Floor at 20 terms: fewer makes the truncated Fourier references
        # (prism, Sandia challenge) Gibbs-oscillate badly near t=0.
        self.inputs["analytical_series_terms"] = self._int_spin(20, 10000, 100)
        self.probe_positions_input = self.QtWidgets.QLineEdit("0.25, 0.50, 0.75, 1.00")
        form.addRow("fixed face T K", self.inputs["surface_temperature_K"])
        form.addRow("series terms", self.inputs["analytical_series_terms"])
        form.addRow("x/L probes", self.probe_positions_input)
        return box

    def _handle_experiment_changed(self, *_: Any) -> None:
        name = self.experiment_combo.currentText()
        experiment = self.experiments[name]
        params = experiment.default_parameters()
        self._apply_params(params)
        self.description_label.setText(experiment.description)
        self.equation_label.setText(experiment.equation)
        if name in {TWO_BLOCK_EXCHANGE, TWO_NODE_LUMPED, GEOMETRY_CONTACT_PAIR}:
            self.specific_stack.setCurrentWidget(self.two_block_panel)
        elif name in {ONE_D_PRISM, DISTRIBUTED_ROD}:
            self.specific_stack.setCurrentWidget(self.prism_panel)
        else:
            self.specific_stack.setCurrentWidget(self.block_panel)
        self._refresh_material_properties()

    def _apply_params(self, params: ThermalValidationParameters) -> None:
        self.material_combo.setCurrentText(params.material)
        self.reference_temperature.setValue(float(params.reference_temperature_K))
        for key, widget in self.inputs.items():
            if not hasattr(params, key):
                continue
            value = getattr(params, key)
            widget.blockSignals(True)
            if hasattr(widget, "setValue"):
                widget.setValue(float(value) if not isinstance(value, int) else int(value))
            widget.blockSignals(False)
        self.interface_model.setCurrentText(params.interface_model)
        self.use_octree_checkbox.setChecked(bool(params.use_octree_pipeline))
        self.gpu_solver_checkbox.setChecked(bool(params.gpu_solver_enabled))
        self.radiation_checkbox.setChecked(bool(params.use_ambient_radiation))
        self.tdep_checkbox.setChecked(bool(params.use_temperature_dependent_properties))
        self.midpoint_checkbox.setChecked(bool(getattr(params, "use_midpoint_property_coupling", True)))
        self.probe_positions_input.setText(", ".join(f"{value:.2f}" for value in params.probe_positions))

    def _params_from_ui(self) -> ThermalValidationParameters:
        params = ThermalValidationParameters(experiment_name=self.experiment_combo.currentText())
        params.material = self.material_combo.currentText()
        params.reference_temperature_K = float(self.reference_temperature.value())
        for key, widget in self.inputs.items():
            if not hasattr(params, key):
                continue
            value = widget.value() if hasattr(widget, "value") else None
            if isinstance(getattr(params, key), int):
                setattr(params, key, int(value))
            else:
                setattr(params, key, float(value))
        params.interface_model = self.interface_model.currentText()
        params.use_octree_pipeline = bool(self.use_octree_checkbox.isChecked())
        params.gpu_solver_enabled = bool(self.gpu_solver_checkbox.isChecked())
        params.use_ambient_radiation = bool(self.radiation_checkbox.isChecked())
        params.use_temperature_dependent_properties = bool(self.tdep_checkbox.isChecked())
        params.use_midpoint_property_coupling = bool(self.midpoint_checkbox.isChecked())
        params.probe_positions = self._parse_probe_positions(self.probe_positions_input.text())
        return params

    def _parse_probe_positions(self, text: str) -> tuple[float, ...]:
        values: list[float] = []
        for raw in text.replace(";", ",").split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                values.append(min(max(float(raw), 0.0), 1.0))
            except ValueError:
                pass
        return tuple(values or [0.25, 0.5, 0.75, 1.0])

    def _refresh_material_properties(self, *_: Any) -> None:
        library = default_material_library()
        material = self.material_combo.currentText()
        values = library.get(material) or library.get("Copper") or next(iter(library.values()))
        rho = float(values["rho_kg_m3"])
        cp = float(values["cp_J_kgK"])
        k = float(values["k_W_mK"])
        alpha = k / max(rho * cp, 1.0e-30)
        self.material_label.setText(
            f"k={k:.6g} W/m/K, rho={rho:.6g} kg/m3, cp={cp:.6g} J/kg/K, alpha={alpha:.6g} m2/s"
        )

    def build_experiment(self) -> None:
        self.stop_requested = False
        params = self._params_from_ui()
        experiment = self.experiments[params.experiment_name]
        try:
            self._set_status("Building validation experiment...")
            self.build_result = experiment.build(params, Path.cwd() / "validation-assets")
            self.run_result = None
            temperatures = {
                int(node_id): float(node.initial_temperature_K)
                for node_id, node in self.build_result.model.nodes.items()
            }
            self.viewer.draw(
                self.build_result.model,
                reset_camera=True,
                node_scalar_values=temperatures,
                scalar_cmap="jet",
                scalar_bar_title="Temperature [K]",
            )
            warning_text = f" Warnings: {len(self.build_result.warnings)}." if self.build_result.warnings else ""
            self._set_status(
                f"Ready. Nodes={len(self.build_result.model.nodes)}, edges={len(self.build_result.model.edges)}, "
                f"volume error={self.build_result.volume_error_fraction:.3%}.{warning_text}"
            )
        except Exception as exc:
            self._set_status(f"Build failed: {exc}", error=True)

    def run_validation(self) -> None:
        if self.build_result is None:
            self.build_experiment()
        if self.build_result is None:
            return
        self.stop_requested = False
        params = self._params_from_ui()
        try:
            self._set_status("Running validation...")
            self.run_result = self.build_result.experiment.run(self.build_result, params)
            self._show_result(self.run_result)
            self._set_status(f"Validation {self.run_result.status}.")
        except Exception as exc:
            self._set_status(f"Validation failed: {exc}", error=True)

    def stop(self) -> None:
        self.stop_requested = True
        self._set_status("Stop requested.")

    def reset(self) -> None:
        if self.build_result is None:
            self.plot.set_result(None)
            self.summary_label.setText("No validation run.")
            self.metrics_table.setRowCount(0)
            return
        temperatures = {
            int(node_id): float(node.initial_temperature_K)
            for node_id, node in self.build_result.model.nodes.items()
        }
        self.viewer.update_node_scalars(temperatures)
        self.plot.set_result(None)
        self.summary_label.setText("Ready.")
        self.metrics_table.setRowCount(0)

    def export_results(self) -> None:
        if self.run_result is None:
            self._set_status("Run validation before exporting results.", error=True)
            return
        path, _filter = self.QtWidgets.QFileDialog.getSaveFileName(
            self.widget,
            "Export Validation Results",
            str(Path.cwd() / "validation-assets" / "thermal_validation_results.json"),
            "JSON files (*.json);;CSV files (*.csv)",
        )
        if not path:
            return
        try:
            export_validation_result(self.run_result, Path(path))
            self._set_status(f"Exported validation results to {path}.")
        except Exception as exc:
            self._set_status(f"Export failed: {exc}", error=True)

    def _show_result(self, result: ValidationRunResult) -> None:
        self.plot.set_result(result)
        self.summary_label.setText(
            f"{result.status}: {result.experiment_name}. "
            f"{len(result.metrics)} metrics, {len(result.warnings)} warning(s)."
        )
        self.metrics_table.setRowCount(len(result.metrics))
        for row, metric in enumerate(result.metrics):
            values = [
                metric.name,
                f"{metric.value:.6g} {metric.units}".strip(),
                "" if metric.tolerance is None else f"{metric.tolerance:.6g} {metric.units}".strip(),
                metric.status,
            ]
            for col, value in enumerate(values):
                self.metrics_table.setItem(row, col, self.QtWidgets.QTableWidgetItem(value))
        final_temps = self._final_temperature_values(result)
        if final_temps and self.build_result is not None:
            updated = self.viewer.update_node_scalars(final_temps)
            if not updated:
                self.viewer.draw(
                    self.build_result.model,
                    reset_camera=False,
                    node_scalar_values=final_temps,
                    scalar_cmap="jet",
                    scalar_bar_title="Temperature [K]",
                )

    def _final_temperature_values(self, result: ValidationRunResult) -> dict[int, float]:
        if self.build_result is None:
            return {}
        final: dict[int, float] = {
            int(node_id): float(node.initial_temperature_K)
            for node_id, node in self.build_result.model.nodes.items()
        }
        if result.experiment_name in {TWO_BLOCK_EXCHANGE, TWO_NODE_LUMPED, GEOMETRY_CONTACT_PAIR}:
            hot = result.simulated.get("hot_average_temperature_K", [])
            cold = result.simulated.get("cold_average_temperature_K", [])
            if hot and cold:
                for node_id, node in self.build_result.model.nodes.items():
                    if str(node.component_name).startswith("VALIDATION_HOT_BLOCK"):
                        final[int(node_id)] = float(hot[-1])
                    if str(node.component_name).startswith("VALIDATION_COLD_BLOCK"):
                        final[int(node_id)] = float(cold[-1])
        elif result.experiment_name == DISTRIBUTED_ROD:
            for node_id in final:
                values = result.simulated.get(f"node_{int(node_id)}_temperature_K", [])
                if values:
                    final[int(node_id)] = float(values[-1])
        elif result.simulated:
            first = next((values for values in result.simulated.values() if isinstance(values, list) and values), None)
            if first:
                for node_id in final:
                    final[node_id] = float(first[-1])
        return final

    def _set_status(self, message: str, error: bool = False) -> None:
        self.status_label.setText(message)
        if self.on_status is not None:
            self.on_status(message, error)

    def shutdown(self) -> None:
        self.viewer.close()

    def _int_spin(self, minimum: int, maximum: int, value: int) -> Any:
        class NoWheelSpinBox(self.QtWidgets.QSpinBox):
            def wheelEvent(inner_self, event: Any) -> None:  # noqa: N802
                event.ignore()

        widget = NoWheelSpinBox()
        widget.setRange(int(minimum), int(maximum))
        widget.setValue(int(value))
        return widget

    def _double_spin(self, minimum: float, maximum: float, value: float, step: float) -> Any:
        class NoWheelDoubleSpinBox(self.QtWidgets.QDoubleSpinBox):
            def wheelEvent(inner_self, event: Any) -> None:  # noqa: N802
                event.ignore()

        widget = NoWheelDoubleSpinBox()
        widget.setDecimals(8)
        widget.setRange(float(minimum), float(maximum))
        widget.setSingleStep(float(step))
        widget.setValue(float(value))
        return widget


class ValidationPlotWidget:
    """Small Qt-painted line plot for validation curves."""

    def __init__(self, QtWidgets: Any, QtCore: Any, parent: Any) -> None:
        class PlotWidget(QtWidgets.QWidget):
            def __init__(inner_self) -> None:
                super().__init__(parent)
                inner_self.result: ValidationRunResult | None = None
                inner_self.setMinimumHeight(180)

            def set_result(inner_self, result: ValidationRunResult | None) -> None:
                inner_self.result = result
                inner_self.update()

            def paintEvent(inner_self, _event: Any) -> None:  # noqa: N802
                painter = inner_self._painter()
                try:
                    inner_self._paint(painter)
                finally:
                    painter.end()

            def _painter(inner_self) -> Any:
                return QtGui.QPainter(inner_self)

            def _paint(inner_self, painter: Any) -> None:
                rect = inner_self.rect().adjusted(44, 12, -16, -28)
                painter.fillRect(inner_self.rect(), inner_self.palette().base())
                painter.setPen(QtGui.QPen(QtGui.QColor("#6b7280"), 1))
                painter.drawRect(rect)
                if inner_self.result is None or not inner_self.result.times_s:
                    painter.drawText(rect, QtCore.Qt.AlignCenter, "No validation curves")
                    return
                times = np.asarray(inner_self.result.times_s, dtype=float)
                series = _plot_series(inner_self.result)
                if not series:
                    painter.drawText(rect, QtCore.Qt.AlignCenter, "No plottable result")
                    return
                y_values = np.concatenate([np.asarray(values, dtype=float) for _name, values, _color in series])
                x_min, x_max = float(np.min(times)), float(np.max(times))
                y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
                if abs(x_max - x_min) <= 0.0:
                    x_max = x_min + 1.0
                if abs(y_max - y_min) <= 0.0:
                    y_max = y_min + 1.0
                for name, values, color in series:
                    points = []
                    vals = np.asarray(values, dtype=float)
                    for x, y in zip(times, vals):
                        px = rect.left() + (float(x) - x_min) / (x_max - x_min) * rect.width()
                        py = rect.bottom() - (float(y) - y_min) / (y_max - y_min) * rect.height()
                        points.append(QtCore.QPointF(px, py))
                    painter.setPen(QtGui.QPen(QtGui.QColor(color), 2))
                    for start, end in zip(points[:-1], points[1:]):
                        painter.drawLine(start, end)
                painter.setPen(QtGui.QPen(QtGui.QColor("#111827"), 1))
                painter.drawText(8, inner_self.height() - 8, f"time {x_min:.3g}-{x_max:.3g} s")
                legend_x = rect.left() + 8
                legend_y = rect.top() + 16
                for name, _values, color in series[:6]:
                    painter.setPen(QtGui.QPen(QtGui.QColor(color), 2))
                    painter.drawLine(legend_x, legend_y - 4, legend_x + 18, legend_y - 4)
                    painter.setPen(QtGui.QPen(QtGui.QColor("#111827"), 1))
                    painter.drawText(legend_x + 24, legend_y, name)
                    legend_y += 16

        self._widget = PlotWidget()

    @property
    def widget(self) -> Any:
        return self._widget

    def __getattr__(self, name: str) -> Any:
        return getattr(self._widget, name)


def _plot_series(result: ValidationRunResult) -> list[tuple[str, list[float], str]]:
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    series: list[tuple[str, list[float], str]] = []
    for key, values in result.simulated.items():
        if not isinstance(values, list) or len(values) != len(result.times_s) or "energy" in key:
            continue
        series.append((f"sim {key}", values, colors[len(series) % len(colors)]))
        analytical = result.analytical.get(key)
        if isinstance(analytical, list) and len(analytical) == len(result.times_s):
            series.append((f"ref {key}", analytical, colors[len(series) % len(colors)]))
        if len(series) >= 6:
            break
    return series
