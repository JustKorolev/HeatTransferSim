"""The Heat Transfer Simulation tab's parameter panel, usable without a graph.

Same sections, labels, ranges and tooltips as that tab -- Run, Environment,
Material Properties, Cryocooler, Controller (global limits), MIMO Thermal-Rate QP,
Modal LQR, Solver -- so a headless run is configured exactly the way the live
simulation is. Only the parts that need a loaded graph (set-all initial
temperature, randomize setpoints) and the display/playback controls are left out;
the headless tab provides its own setpoint / initial-temperature fields, which
apply to the whole run rather than to an in-memory model.

Values are read back with ``replace()`` on the caller's parameters, so any field
without a widget keeps its saved value instead of silently reverting to a default.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .simulation_parameters import SimulationParameters

CONTROLLER_SCHEME_LABELS = {
    "pid_qp": "PID + QP allocator",
    "modal_lqr": "Modal LQR (reduced-model)",
}


class SimulationParameterPanel:
    """Builds the parameter widgets and reads them back into SimulationParameters."""

    def __init__(self, qt: Any, params: SimulationParameters | None = None) -> None:
        self.QtWidgets = qt.QtWidgets
        self.QtCore = qt.QtCore
        self.params = params or SimulationParameters()
        self.inputs: dict[str, Any] = {}

    # -- widget helpers (match the simulation tab's) ------------------------ #
    def _double_spin(self, minimum: float, maximum: float, value: float, step: float) -> Any:
        class NoWheelDoubleSpinBox(self.QtWidgets.QDoubleSpinBox):
            def wheelEvent(inner_self, event: Any) -> None:  # noqa: N802 - Qt override
                event.ignore()

        widget = NoWheelDoubleSpinBox()
        widget.setDecimals(8)
        widget.setRange(float(minimum), float(maximum))
        widget.setSingleStep(float(step))
        widget.setValue(float(value))
        return widget

    def _int_spin(self, minimum: int, maximum: int, value: int, step: int) -> Any:
        class NoWheelSpinBox(self.QtWidgets.QSpinBox):
            def wheelEvent(inner_self, event: Any) -> None:  # noqa: N802 - Qt override
                event.ignore()

        widget = NoWheelSpinBox()
        widget.setRange(int(minimum), int(maximum))
        widget.setSingleStep(int(step))
        widget.setValue(int(value))
        return widget

    def _checkbox(self, text: str, checked: bool) -> Any:
        widget = self.QtWidgets.QCheckBox(text)
        widget.setChecked(bool(checked))
        return widget

    def _section(self, title: str) -> tuple[Any, Any]:
        box = self.QtWidgets.QGroupBox(title)
        box.setStyleSheet("QGroupBox { font-weight: 700; margin-top: 8px; }")
        return box, self.QtWidgets.QFormLayout(box)

    def _add_double(self, form: Any, name: str, label: str, minimum: float, maximum: float, step: float) -> None:
        widget = self._double_spin(minimum, maximum, getattr(self.params, name), step)
        self.inputs[name] = widget
        form.addRow(label, widget)

    def _add_int(self, form: Any, name: str, label: str, minimum: int, maximum: int, step: int) -> None:
        widget = self._int_spin(minimum, maximum, int(getattr(self.params, name)), step)
        self.inputs[name] = widget
        form.addRow(label, widget)

    # -- panel -------------------------------------------------------------- #
    def build(self, form: Any) -> None:
        run_box, run_form = self._section("Run")
        for name, label, minimum, maximum, step in (
            ("dt_s", "dt_s", 1.0e-9, 1.0e9, 1.0),
            ("t_final_s", "t_final_s", 0.0, 1.0e12, 60.0),
        ):
            self._add_double(run_form, name, label, minimum, maximum, step)
        self.input_mode = self.QtWidgets.QComboBox()
        self.input_mode.addItems(["zero", "heater_inputs"])
        self.input_mode.setCurrentText(self.params.input_mode)
        self.input_mode.setToolTip(
            "'heater_inputs' runs the heater controller; 'zero' leaves heaters off "
            "(passive cooldown)."
        )
        run_form.addRow("input mode", self.input_mode)
        self.controller_scheme_combo = self.QtWidgets.QComboBox()
        self.controller_scheme_combo.addItems(list(CONTROLLER_SCHEME_LABELS.values()))
        self.controller_scheme_combo.setCurrentText(
            CONTROLLER_SCHEME_LABELS.get(
                str(getattr(self.params, "mimo_controller_scheme", "pid_qp")),
                CONTROLLER_SCHEME_LABELS["pid_qp"],
            )
        )
        self.controller_scheme_combo.setToolTip(
            "Heater controller for 'heater_inputs' mode.\n"
            "- PID + QP allocator: the standard controller.\n"
            "- Modal LQR (reduced-model): needs a controller artifact for this graph; "
            "falls back to PID+QP if it is missing or built for a different graph."
        )
        run_form.addRow("controller", self.controller_scheme_combo)
        form.addRow(run_box)

        environment_box, environment_form = self._section("Environment")
        self._add_double(environment_form, "T_env_K", "exterior / ambient T K", 0.0, 1.0e6, 1.0)
        self.inputs["T_env_K"].setToolTip(
            "Radiative background for the OUTSIDE of the assembly (room / ambient surroundings)."
        )
        self._add_double(
            environment_form, "interior_environment_temperature_K", "interior (cryo) T K", 0.0, 1.0e6, 1.0
        )
        self.inputs["interior_environment_temperature_K"].setToolTip(
            "Radiative background for the INSIDE of the assembly (cryocooled vacuum enclosure)."
        )
        self.inputs["use_ambient_radiation"] = self._checkbox(
            "Use ambient radiation", self.params.use_ambient_radiation
        )
        environment_form.addRow(self.inputs["use_ambient_radiation"])
        self.inputs["use_radiative_coupling"] = self._checkbox(
            "Surface-to-surface radiative coupling (ray-traced)",
            getattr(self.params, "use_radiative_coupling", False),
        )
        self.inputs["use_radiative_coupling"].setToolTip(
            "Ray-trace view factors so parts exchange radiation with each other, not just with "
            "the background. One-time precompute at prepare; skipped for very large graphs."
        )
        environment_form.addRow(self.inputs["use_radiative_coupling"])
        form.addRow(environment_box)

        properties_box, properties_form = self._section("Material Properties")
        self.inputs["use_temperature_dependent_properties"] = self._checkbox(
            "Temperature-dependent cp(T)/k(T)", self.params.use_temperature_dependent_properties
        )
        self.inputs["use_temperature_dependent_properties"].setToolTip(
            "Recompute per-node C(T)=m*cp(T) and conduction/contact from NIST cryogenic curves "
            "each step, instead of constant room-temperature properties."
        )
        properties_form.addRow(self.inputs["use_temperature_dependent_properties"])
        self._add_int(properties_form, "copper_rrr", "Copper RRR", 1, 100000, 10)
        self.inputs["copper_rrr"].setToolTip(
            "Residual resistivity ratio for OFHC copper k(T). NIST fits exist for "
            "50/100/150/300/500 (nearest is used). Only affects temperature-dependent runs."
        )
        self.inputs["use_midpoint_property_coupling"] = self._checkbox(
            "Midpoint property/radiation coupling",
            getattr(self.params, "use_midpoint_property_coupling", True),
        )
        self.inputs["use_midpoint_property_coupling"].setToolTip(
            "Evaluate temperature-dependent properties and radiation at a predicted midpoint "
            "(2nd-order-in-dt splitting) instead of the step-start temperature."
        )
        properties_form.addRow(self.inputs["use_midpoint_property_coupling"])
        self._add_double(
            properties_form, "default_bolted_contact_conductance_W_m2K",
            "bolted contact W/m2K", 0.0, 1.0e9, 100.0,
        )
        self._add_double(
            properties_form, "contact_conductance_temp_exponent",
            "contact temp exponent", 0.0, 1.0e3, 0.1,
        )
        self._add_double(
            properties_form, "contact_conductance_reference_temperature_K",
            "contact reference T K", 0.0, 1.0e6, 1.0,
        )
        form.addRow(properties_box)

        cooler_box, cooler_form = self._section("Cryocooler")
        cooler_form.addRow("Model", self.QtWidgets.QLabel("PT60 measured lift curve"))
        for name, label, minimum, maximum, step in (
            ("cryocooler_max_power_W", "Maximum cooling power W", 0.0, 1.0e9, 1.0),
            ("cryocooler_capacity_scale", "Capacity scale", 0.0, 1.0e9, 0.05),
        ):
            self._add_double(cooler_form, name, label, minimum, maximum, step)
        self.inputs["cryocooler_enabled"] = self._checkbox("Enabled", self.params.cryocooler_enabled)
        cooler_form.addRow(self.inputs["cryocooler_enabled"])
        form.addRow(cooler_box)

        controller_box, controller_form = self._section("Controller (global limits)")
        for name, label, minimum, maximum, step in (
            ("mimo_default_heater_max_power_W", "max heater power W", 0.0, 1.0e9, 1.0),
            ("mimo_heater_slew_rate_W_per_s", "hard slew W/s", 0.0, 1.0e9, 1.0),
            ("mimo_v_cmd_abs_max_K_per_s", "max rate cmd K/s", 0.0, 1.0e9, 0.01),
        ):
            self._add_double(controller_form, name, label, minimum, maximum, step)
        form.addRow(controller_box)

        modal_box, modal_form = self._section("Modal LQR")
        self._add_double(modal_form, "modal_integral_gain", "integral gain", 0.0, 1.0e6, 0.01)
        self.inputs["modal_integral_gain"].setToolTip(
            "Offset-free integral action for the modal controller; 0 disables it."
        )
        form.addRow(modal_box)

        mimo_box, mimo_form = self._section("MIMO Thermal-Rate QP")
        for name, label, minimum, maximum, step in (
            ("mimo_hold_threshold_K", "enter hold below K", 0.0, 1.0e6, 0.1),
            ("mimo_coarse_threshold_K", "return coarse above K", 0.0, 1.0e6, 0.1),
            ("mimo_lambda_u", "lambda_u heater effort", 0.0, 1.0e9, 0.001),
            ("mimo_rho_du", "rho_du power change", 0.0, 1.0e9, 0.01),
            ("heater_sensor_pair_alpha", "pair alpha", 0.0, 1.0e9, 0.01),
            ("role_contact_tolerance_mm", "role contact tol mm", 0.0, 1.0e9, 1.0e-6),
            ("role_contact_tolerance_max_mm", "role contact max mm", 0.0, 1.0e9, 0.1),
            ("role_contact_tolerance_growth_factor", "role contact growth", 1.01, 1.0e6, 0.1),
            ("drift_lpf_tau_s", "drift LPF tau s", 0.0, 1.0e9, 0.1),
            ("derivative_dt_floor_s", "derivative dt floor s", 0.0, 1.0e9, 1.0e-6),
            ("mimo_integral_abs_max", "integral abs max", 0.0, 1.0e12, 1.0),
        ):
            self._add_double(mimo_form, name, label, minimum, maximum, step)
        self.inputs["mimo_freeze_integral_when_saturated"] = self._checkbox(
            "Freeze integral when saturated", self.params.mimo_freeze_integral_when_saturated
        )
        mimo_form.addRow(self.inputs["mimo_freeze_integral_when_saturated"])
        form.addRow(mimo_box)

        # Not shown in the live tab (it keeps its saved values), but an overnight run
        # is exactly where the solver settings matter.
        solver_box, solver_form = self._section("Solver")
        self.solver_method = self.QtWidgets.QComboBox()
        self.solver_method.addItems(["tr_bdf2", "backward_euler"])
        self.solver_method.setCurrentText(str(self.params.implicit_sparse_simulation_method))
        solver_form.addRow("implicit method", self.solver_method)
        self._add_double(solver_form, "implicit_sparse_simulation_rtol", "rtol", 1.0e-14, 1.0, 1.0e-6)
        self._add_int(solver_form, "implicit_sparse_simulation_maxiter", "max iterations", 1, 100000, 10)
        self.inputs["implicit_sparse_adaptive_substeps_enabled"] = self._checkbox(
            "Adaptive substeps", self.params.implicit_sparse_adaptive_substeps_enabled
        )
        solver_form.addRow(self.inputs["implicit_sparse_adaptive_substeps_enabled"])
        self._add_double(
            solver_form, "implicit_sparse_adaptive_target_delta_K", "substep target dT K", 0.0, 1.0e6, 0.1
        )
        self._add_int(
            solver_form, "implicit_sparse_adaptive_max_substeps", "max substeps", 1, 1000, 1
        )
        self.inputs["implicit_sparse_residual_check_enabled"] = self._checkbox(
            "Residual check", self.params.implicit_sparse_residual_check_enabled
        )
        solver_form.addRow(self.inputs["implicit_sparse_residual_check_enabled"])
        self.inputs["gpu_solver_enabled"] = self._checkbox(
            "Use GPU solver when available", self.params.gpu_solver_enabled
        )
        solver_form.addRow(self.inputs["gpu_solver_enabled"])
        form.addRow(solver_box)

    # -- state -------------------------------------------------------------- #
    def set_params(self, params: SimulationParameters) -> None:
        """Repopulate every widget from ``params`` (e.g. after switching graph)."""
        self.params = params
        for name, widget in self.inputs.items():
            value = getattr(params, name, None)
            if value is None:
                continue
            if isinstance(widget, self.QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, self.QtWidgets.QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, self.QtWidgets.QDoubleSpinBox):
                widget.setValue(float(value))
        self.input_mode.setCurrentText(str(params.input_mode))
        self.controller_scheme_combo.setCurrentText(
            CONTROLLER_SCHEME_LABELS.get(
                str(getattr(params, "mimo_controller_scheme", "pid_qp")),
                CONTROLLER_SCHEME_LABELS["pid_qp"],
            )
        )
        self.solver_method.setCurrentText(str(params.implicit_sparse_simulation_method))

    def read(self, base: SimulationParameters | None = None) -> SimulationParameters:
        """Current widget values, applied on top of ``base`` so fields without a
        widget keep their saved values."""
        values: dict[str, Any] = {}
        for name, widget in self.inputs.items():
            if isinstance(widget, self.QtWidgets.QCheckBox):
                values[name] = bool(widget.isChecked())
            elif isinstance(widget, self.QtWidgets.QSpinBox):
                values[name] = int(widget.value())
            elif isinstance(widget, self.QtWidgets.QDoubleSpinBox):
                values[name] = float(widget.value())
        values["input_mode"] = self.input_mode.currentText()
        labels_to_key = {v: k for k, v in CONTROLLER_SCHEME_LABELS.items()}
        values["mimo_controller_scheme"] = labels_to_key.get(
            self.controller_scheme_combo.currentText(), "pid_qp"
        )
        values["implicit_sparse_simulation_method"] = self.solver_method.currentText()
        return replace(base or self.params, **values)
