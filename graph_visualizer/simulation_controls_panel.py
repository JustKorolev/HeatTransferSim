"""The simulation control panel, built once and shared by both simulation tabs.

The Heat Transfer Simulation tab and the Headless Run tab configure the same
simulation, so they must offer the same controls, with the same labels and
tooltips, in the same order, at the same positions. They used to be built by two
unrelated pieces of code and drifted apart immediately; everything below the
graph-loading row is now built here, once.

Mode differences are expressed by HIDING rows, never by building a different
layout. Every widget exists in both modes and :meth:`_apply_mode` calls
``setVisible(False)`` on the ones that do not apply. That is what keeps the two
panels lined up -- a row added for one tab cannot silently shift the other one.

What differs, and why:

* live-only rows need a model loaded in the GUI process (Initialize, set-all
  initial temperature, randomize setpoints, the enabled-I/O table, sys ID, the
  solver diagnostic, per-component temperatures) or only mean something during
  live playback (playback speed, history limit, loop, Play/Pause/Step, the time
  slider, the Display colouring section);
* headless-only rows describe a run that is launched rather than played: the
  snapshot/checkpoint cadence, a whole-run setpoint and initial temperature, the
  output folder button, and a Solver section -- the live tab keeps its saved
  solver settings without showing them, but an overnight run is exactly where
  they matter.

The owning tab keeps direct handles on the widgets via :meth:`export_to`, so its
existing per-widget code (``self.inputs[...]``, ``self.time_slider``, ...) is
unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from .simulation_parameters import SimulationParameters

MODE_LIVE = "live"
MODE_HEADLESS = "headless"

# Placeholder shown when a graph has no controller artifact yet. The PID+QP
# allocator it used to name is gone: its per-pair PID could not work on a plant
# whose RGA diagonal is negative on 26 of 27 pairings. Kept as a name so the combo
# always has a "nothing selected" row rather than being empty.
NO_CONTROLLER_LABEL = "(no controller selected)"
PID_QP_LABEL = NO_CONTROLLER_LABEL  # back-compat for importers

# Rows hidden per mode, by row key (see the _row(...) calls below).
_HEADLESS_HIDDEN_ROWS = frozenset(
    {
        "initialize",
        "playback_speed",
        "simulation_history_limit",
        "loop_playback",
        "transport",
        "time_slider",
        "save_trajectory",
        "reset_integrators",
        "component",
        "component_initial_K",
        "component_apply",
        "warning",
        "stats",
        "controller_status",
        "sensor_readouts",
        "legend",
    }
)
_HEADLESS_HIDDEN_SECTIONS = frozenset(
    {"display", "enabled_io", "sys_id", "stepper_diagnostic"}
)
# Sections/rows the live tab has never shown; they exist so the headless tab does
# not have to build a second panel to get them.
_LIVE_HIDDEN_ROWS = frozenset(
    {
        "snapshot_interval_s",
        "checkpoint_interval_s",
        "run_setpoint_K",
        "run_initial_temperature_K",
        "open_output",
    }
)
_LIVE_HIDDEN_SECTIONS = frozenset({"solver"})

# Widgets the live tab holds by name. Built here, handed back by export_to() so
# the tab's existing code keeps working against plain attributes.
_EXPORTED_LIVE_ONLY = (
    "initial_temperature_all_spin",
    "sensor_random_center_spin",
    "sensor_random_spread_mK_spin",
    "modal_temp_spin",
    "modal_modes_spin",
    "modal_order_spin",
    "modal_effort_spin",
    "modal_integral_spin",
    "modal_design_button",
    "modal_design_status_label",
    "time_slider",
    "enabled_io_table",
    "sys_id_matrix_combo",
    "sys_id_step_power",
    "sys_id_global_temperature_K",
    "sys_id_duration_s",
    "sys_id_baseline_window_s",
    "sys_id_final_window_s",
    "sys_id_restore_between_tests",
    "sys_id_keep_cryocooler_active",
    "sys_id_uniform_baseline",
    "run_sys_id_button",
    "cancel_sys_id_button",
    "sys_id_progress_label",
    "sys_id_status_label",
    "stepper_diagnostic_save",
    "stepper_diagnostic_button",
    "stepper_diagnostic_target_label",
    "stepper_diagnostic_status_label",
    "component_combo",
    "component_temperature",
    "warning_label",
    "stats_label",
    "controller_status_label",
    "sensor_readout_box",
    "cooling_readout_box",
    "cooling_readout_table",
    "heating_readout_box",
    "heating_readout_tree",
    "legend_label",
)
_EXPORTED_SHARED = (
    "input_mode",
    "controller_scheme_combo",
    "mimo_pi_kp_spin",
    "mimo_pi_ki_spin",
    "solver_method_combo",
    "run_headless_button",
    "stop_headless_button",
    "open_output_button",
    "snapshot_spin",
    "checkpoint_spin",
    "setpoint_spin",
    "use_setpoint",
    "initial_spin",
    "use_initial",
)
_EXPORTED = _EXPORTED_LIVE_ONLY + _EXPORTED_SHARED
# The per-node "Parameters" editor, exported only once build_readout_editor ran.
_EXPORTED_READOUT = (
    "readout_editor_box",
    "readout_editor_title",
    "readout_sensor_editor",
    "readout_heater_editor",
    "readout_cooling_editor",
)


class SimulationControlsPanel:
    """Builds every control below the graph row, for both simulation tabs."""

    def __init__(
        self,
        qt: Any,
        *,
        params: SimulationParameters | None = None,
        mode: str = MODE_LIVE,
        actions: dict[str, Callable[..., Any]] | None = None,
        on_parameter_change: Callable[..., None] | None = None,
        legend_text: str = "",
        modal_operating_temperature_K: float = 293.15,
    ) -> None:
        self.QtCore = qt.QtCore
        self.QtWidgets = qt.QtWidgets
        self.mode = mode
        self.params = params if params is not None else SimulationParameters()
        self._actions = dict(actions or {})
        self._on_parameter_change = on_parameter_change
        self._legend_text = legend_text
        self._modal_operating_temperature_K = float(modal_operating_temperature_K)
        # name -> widget for every field backed by a SimulationParameters field.
        self.inputs: dict[str, Any] = {}
        # The per-node "Parameters" editor's widgets (built by build_readout_editor).
        self.readout_editor_inputs: dict[str, Any] = {}
        # row key -> (form, widget) so a row can be hidden with its label.
        self._rows: dict[str, tuple[Any, Any]] = {}
        self._sections: dict[str, Any] = {}
        self._section_forms: dict[str, Any] = {}

    # -- public widget helpers (the tabs reuse these outside the panel) ------- #
    def double_spin(self, minimum: float, maximum: float, value: float, step: float) -> Any:
        class NoWheelDoubleSpinBox(self.QtWidgets.QDoubleSpinBox):
            def wheelEvent(inner_self, event: Any) -> None:  # noqa: N802 - Qt override name.
                event.ignore()

        widget = NoWheelDoubleSpinBox()
        widget.setDecimals(8)
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setValue(float(value))
        return widget

    def int_spin(self, minimum: int, maximum: int, value: int, step: int) -> Any:
        class NoWheelSpinBox(self.QtWidgets.QSpinBox):
            def wheelEvent(inner_self, event: Any) -> None:  # noqa: N802 - Qt override name.
                event.ignore()

        widget = NoWheelSpinBox()
        widget.setRange(int(minimum), int(maximum))
        widget.setSingleStep(int(step))
        widget.setValue(int(value))
        return widget

    def checkbox(self, text: str, checked: bool, callback: Any | None = None) -> Any:
        widget = self.QtWidgets.QCheckBox(text)
        widget.setChecked(bool(checked))
        if callback is not None:
            widget.stateChanged.connect(callback)
        return widget

    def section(self, title: str) -> tuple[Any, Any]:
        box = self.QtWidgets.QGroupBox(title)
        box.setStyleSheet("QGroupBox { font-weight: 700; margin-top: 8px; }")
        return box, self.QtWidgets.QFormLayout(box)

    def pin_two_line_label(self, label: Any) -> None:
        """Lock a status label to a fixed two-line height so runtime messages of
        varying length can't change its size and shove the rest of the panel around.
        Text longer than two lines wraps then clips (is cut off), not expands."""
        label.setWordWrap(True)
        label.setAlignment(self.QtCore.Qt.AlignTop | self.QtCore.Qt.AlignLeft)
        two_lines = label.fontMetrics().lineSpacing() * 2 + 6
        label.setFixedHeight(int(two_lines))
        label.setSizePolicy(self.QtWidgets.QSizePolicy.Preferred, self.QtWidgets.QSizePolicy.Fixed)

    # -- internal building blocks ------------------------------------------- #
    def _act(self, name: str) -> Callable[..., Any] | None:
        return self._actions.get(name)

    def _connect(self, signal: Any, action: str) -> None:
        callback = self._act(action)
        if callback is not None:
            signal.connect(callback)

    def _changed(self, field: str) -> Callable[..., None]:
        """A slot that reports ``field`` to the owner, or does nothing when the
        owner has no live simulation to update (the headless tab reads the widgets
        once, at launch)."""

        def _slot(*_args: Any) -> None:
            if self._on_parameter_change is not None:
                self._on_parameter_change(field)

        return _slot

    def _row(self, form: Any, key: str, widget: Any, label: str | None = None) -> Any:
        if label is None:
            form.addRow(widget)
        else:
            form.addRow(label, widget)
        self._rows[key] = (form, widget)
        return widget

    def _hrow(self, *widgets: Any) -> Any:
        """A horizontal row of widgets wrapped in a container, so the whole row can
        be hidden as one (a bare QLayout added to a QFormLayout cannot be)."""
        container = self.QtWidgets.QWidget()
        layout = self.QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            layout.addWidget(widget)
        return container

    def _add_double(
        self, form: Any, name: str, label: str, minimum: float, maximum: float, step: float
    ) -> Any:
        widget = self.double_spin(minimum, maximum, getattr(self.params, name), step)
        widget.valueChanged.connect(self._changed(name))
        self.inputs[name] = widget
        return self._row(form, name, widget, label)

    def _add_int(
        self, form: Any, name: str, label: str, minimum: int, maximum: int, step: int
    ) -> Any:
        widget = self.int_spin(minimum, maximum, int(getattr(self.params, name)), step)
        widget.valueChanged.connect(self._changed(name))
        self.inputs[name] = widget
        return self._row(form, name, widget, label)

    def _add_checkbox(self, form: Any, name: str, text: str, default: Any = None) -> Any:
        current = getattr(self.params, name, default)
        widget = self.checkbox(text, bool(current), self._changed(name))
        self.inputs[name] = widget
        return self._row(form, name, widget)

    def _add_section(self, form: Any, key: str, title: str) -> tuple[Any, Any]:
        box, section_form = self.section(title)
        self._sections[key] = box
        self._section_forms[key] = section_form
        form.addRow(box)
        return box, section_form

    def _button(self, text: str, action: str, tooltip: str | None = None) -> Any:
        button = self.QtWidgets.QPushButton(text)
        if tooltip:
            button.setToolTip(tooltip)
        self._connect(button.clicked, action)
        return button

    # -- panel --------------------------------------------------------------- #
    def build(self, form: Any) -> None:
        """Add every section, in the order both tabs show them."""
        self._build_parameter_controls(form)
        self._build_playback_controls(form)
        self._build_enabled_io_controls(form)
        self._build_sys_id_controls(form)
        self._build_stepper_diagnostic_controls(form)
        self._build_component_temperature_controls(form)
        self._build_status_controls(form)
        self._apply_mode()

    def _build_parameter_controls(self, form: Any) -> None:
        run_box, run_form = self._add_section(form, "run", "Run")
        self._row(
            run_form,
            "initialize",
            self._button(
                "Initialize",
                "initialize",
                "Build the simulation using the latest graph, matrices, and controller settings.",
            ),
        )
        for name, label, minimum, maximum, step in (
            ("dt_s", "dt_s", 1.0e-9, 1.0e9, 1.0),
            ("t_final_s", "t_final_s", 0.0, 1.0e12, 60.0),
            ("playback_speed", "playback speed", 0.01, 1.0e6, 0.25),
        ):
            self._add_double(run_form, name, label, minimum, maximum, step)
        self._add_int(run_form, "simulation_history_limit", "history limit", 0, 1_000_000, 1)
        self._add_checkbox(run_form, "loop_playback", "Loop playback")
        self.input_mode = self.QtWidgets.QComboBox()
        self.input_mode.addItems(["zero", "heater_inputs"])
        self.input_mode.setCurrentText(self.params.input_mode)
        self.input_mode.currentTextChanged.connect(self._changed("input_mode"))
        self._row(run_form, "input_mode", self.input_mode, "input mode")
        self.controller_scheme_combo = self.QtWidgets.QComboBox()
        self.controller_scheme_combo.setToolTip(
            "Heater controller for 'heater_inputs' mode.\n"
            "- MIMO PI entries: static-decoupling PI, one per G matrix built for this graph. This is the controller to use; tune it in the per-sensor gain table.\n"
            "- Modal LQR entries: one per controller actually built for this graph, labelled with "
            "the reduced order, mode count and design operating point that distinguish them. "
            "Build one with the 'Modal LQR Design' panel below (or tools/analyze_plant_modes.py); "
            "until then there is nothing to select.\n"
            "- MIMO PI entries: one per DC gain matrix built for this graph. The gain is the only "
            "model this scheme needs, and it decouples through it -- per-pair PID is not viable "
            "here, since the RGA diagonal is negative on 26 of 27 pairings."
        )
        self._connect(self.controller_scheme_combo.currentTextChanged, "controller_scheme_selected")
        self._row(run_form, "controller", self.controller_scheme_combo, "controller")
        # MIMO PI gains. After decoupling every channel has unit DC gain, so one
        # pair of numbers is a defensible default for all of them; the per-sensor
        # table beside the Parameters editor overrides individual channels.
        self.mimo_pi_kp_spin = self.double_spin(0.0, 1.0e9,
                                                float(getattr(self.params, "mimo_pi_kp", 0.0)), 0.01)
        self.mimo_pi_kp_spin.setToolTip(
            "Proportional gain per CONTROLLED SENSOR, applied in the decoupled space "
            "(Kp = tau/lambda for a channel of time constant tau). 0 gives feedforward + "
            "integral, which is what the modal scheme effectively ran."
        )
        self._row(run_form, "mimo_pi_kp", self.mimo_pi_kp_spin, "MIMO PI Kp")
        self.mimo_pi_ki_spin = self.double_spin(0.0, 1.0e9,
                                                float(getattr(self.params, "mimo_pi_ki", 1.0e-3)), 1.0e-4)
        self.mimo_pi_ki_spin.setToolTip(
            "Integral gain per CONTROLLED SENSOR: Ki = 1/lambda for a desired closed-loop time "
            "constant lambda. Do NOT ask for lambda faster than the plant's fastest retained mode "
            "(1182 s on no_mli_high_res, so Ki <~ 8.5e-4) or the command chases dynamics the "
            "model does not contain."
        )
        self._row(run_form, "mimo_pi_ki", self.mimo_pi_ki_spin, "MIMO PI Ki")
        # Headless only: how often the launched run writes snapshots/checkpoints.
        # Live playback has no such artifacts.
        self.snapshot_spin = self.double_spin(0.0, 1.0e12, 300.0, 60.0)
        self.snapshot_spin.setToolTip("Simulated seconds between saved snapshots.")
        self._row(run_form, "snapshot_interval_s", self.snapshot_spin, "snapshot every s")
        self.checkpoint_spin = self.double_spin(0.0, 1.0e12, 600.0, 60.0)
        self.checkpoint_spin.setToolTip("Wall-clock seconds between resume checkpoints.")
        self._row(run_form, "checkpoint_interval_s", self.checkpoint_spin, "checkpoint every s")

        environment_box, environment_form = self._add_section(form, "environment", "Environment")
        self._add_double(environment_form, "T_env_K", "exterior / ambient T K", 0.0, 1.0e6, 1.0)
        self.inputs["T_env_K"].setToolTip(
            "Radiative background for the OUTSIDE of the assembly (room / ambient surroundings)."
        )
        self._add_double(
            environment_form, "interior_environment_temperature_K", "interior (cryo) T K", 0.0, 1.0e6, 1.0
        )
        self.inputs["interior_environment_temperature_K"].setToolTip(
            "Radiative background for the INSIDE of the assembly (cryocooled vacuum enclosure). "
            "Inward-facing surfaces radiate to this once view-factor classification assigns them."
        )
        self._add_checkbox(environment_form, "use_ambient_radiation", "Use ambient radiation")
        self._add_checkbox(
            environment_form,
            "use_radiative_coupling",
            "Surface-to-surface radiative coupling (ray-traced)",
            default=False,
        )
        self.inputs["use_radiative_coupling"].setToolTip(
            "Ray-trace view factors over the exposed faces so parts exchange radiation with "
            "each other (a hot part can warm a cold part), not just with the background. "
            "One-time precompute when the simulation is prepared; skipped for very large graphs."
        )
        # Bulk initial-temperature control: set initial_temperature_K on EVERY
        # component at once (the state the simulation starts from / resets to).
        self.initial_temperature_all_spin = self.double_spin(0.0, 1.0e6, 293.15, 1.0)
        set_all_initial = self._button(
            "Set all components",
            "set_all_initial_temperatures",
            "Set the initial temperature of EVERY component to this value. Updates the loaded "
            "graph; if a simulation is already initialized it resets to this immediately, "
            "otherwise it takes effect on the next Initialize.",
        )
        self._row(
            environment_form,
            "initial_temperature_all",
            self._hrow(self.initial_temperature_all_spin, set_all_initial),
            "initial T (all) K",
        )
        # Testing helper: assign each sensor a random controller SETPOINT (desired
        # temperature) = center +/- a random mK-scale spread (e.g. ~50 K +/- tens of mK).
        self.sensor_random_center_spin = self.double_spin(0.0, 1.0e6, 50.0, 1.0)
        self.sensor_random_spread_mK_spin = self.double_spin(0.0, 1.0e6, 50.0, 1.0)
        randomize_setpoints = self._button(
            "Randomize setpoints",
            "randomize_setpoints",
            "Assign each sensor a random controller setpoint (desired temperature) = center +/- a "
            "uniform random offset within the spread (mK). For testing how the controller drives "
            "the sensors to distinct targets. Applied live (the controller reads setpoints each step).",
        )
        self._row(
            environment_form,
            "randomize_setpoints",
            self._hrow(
                self.sensor_random_center_spin,
                self.QtWidgets.QLabel("K  ±"),
                self.sensor_random_spread_mK_spin,
                self.QtWidgets.QLabel("mK"),
                randomize_setpoints,
            ),
            "randomize setpoints",
        )
        # Headless equivalents of the two rows above: they apply to the whole run
        # rather than to a graph held in this process.
        self.setpoint_spin = self.double_spin(0.0, 1.0e6, 293.15, 1.0)
        self.use_setpoint = self.QtWidgets.QCheckBox("use setpoint")
        self.use_setpoint.setChecked(True)
        self.setpoint_spin.setToolTip(
            "Constant setpoint applied to EVERY sensor. Leave the 'use setpoint' box "
            "unchecked to keep whatever the graph already has."
        )
        self._row(
            environment_form,
            "run_setpoint_K",
            self._hrow(self.setpoint_spin, self.use_setpoint),
            "setpoint K",
        )
        self.initial_spin = self.double_spin(0.0, 1.0e6, 293.15, 1.0)
        self.use_initial = self.QtWidgets.QCheckBox("override")
        self.initial_spin.setToolTip(
            "Start every cell at this temperature. Leave 'override' unchecked to use the "
            "initial temperatures saved with the graph."
        )
        self._row(
            environment_form,
            "run_initial_temperature_K",
            self._hrow(self.initial_spin, self.use_initial),
            "initial T K",
        )

        properties_box, properties_form = self._add_section(form, "properties", "Material Properties")
        self._add_checkbox(
            properties_form, "use_temperature_dependent_properties", "Temperature-dependent cp(T)/k(T)"
        )
        self.inputs["use_temperature_dependent_properties"].setToolTip(
            "Recompute per-node C(T)=m*cp(T) and conduction/contact from NIST cryogenic "
            "curves each step, instead of using constant room-temperature properties."
        )
        self._add_double(
            properties_form, "tdep_rebuild_delta_K", "rebuild properties above K", 0.0, 1.0e6, 0.05
        )
        self.inputs["tdep_rebuild_delta_K"].setToolTip(
            "Reuse C(T)/L(T) until the largest per-cell temperature change since the LAST "
            "REBUILD exceeds this many K. 0 rebuilds every step (the old behaviour).\n\n"
            "Worth far more than the rebuild's own cost: recomputing the operator also "
            "invalidates the implicit stepper's stage-matrix cache, so the CG solve restarts "
            "from scratch every step, and it churns several sparse matrices the size of the "
            "graph's edge list.\n\n"
            "This is not new lag. The properties are ALREADY evaluated at the step-start "
            "temperature (semi-implicit), so this replaces an implicit one-step lag with an "
            "explicit bounded one. 0.25 K is far inside the uncertainty of the NIST curves "
            "themselves. Larger is faster and staler."
        )
        self._add_int(properties_form, "copper_rrr", "Copper RRR", 1, 100000, 10)
        self.inputs["copper_rrr"].setToolTip(
            "Residual resistivity ratio for OFHC copper thermal conductivity k(T). "
            "NIST fits exist for 50/100/150/300/500 (the nearest is used). "
            "Only affects runs with temperature-dependent properties enabled."
        )
        self._add_checkbox(
            properties_form,
            "use_midpoint_property_coupling",
            "Midpoint property/radiation coupling",
            default=True,
        )
        self.inputs["use_midpoint_property_coupling"].setToolTip(
            "Evaluate the temperature-dependent properties and radiation at a "
            "predicted midpoint (2nd-order-in-dt splitting) instead of the "
            "step-start temperature. More accurate during fast transients; adds "
            "one operator rebuild per step and only when those terms are active."
        )

        cooler_box, cooler_form = self._add_section(form, "cryocooler", "Cryocooler")
        cooler_form.addRow("Model", self.QtWidgets.QLabel("PT60 measured lift curve"))
        for name, label, minimum, maximum, step in (
            ("cryocooler_max_power_W", "Maximum cooling power W", 0.0, 1.0e9, 1.0),
            ("cryocooler_capacity_scale", "Capacity scale", 0.0, 1.0e9, 0.05),
        ):
            self._add_double(cooler_form, name, label, minimum, maximum, step)
        self._add_checkbox(cooler_form, "cryocooler_enabled", "Enabled")

        # Global controller limits: enforced by BOTH the MIMO PI and modal-LQR
        # schemes (absolute heater-power clamp + hard slew rate). "max rate cmd"
        # additionally bounded the removed PID+QP rate command; it is inert now
        # (which has no rate command) but kept here as a global controller knob.
        controller_box, controller_form = self._add_section(
            form, "controller_limits", "Controller (global limits)"
        )
        for name, label, minimum, maximum, step in (
            ("mimo_default_heater_max_power_W", "max heater power W", 0.0, 1.0e9, 1.0),
            ("mimo_heater_slew_rate_W_per_s", "hard slew W/s", 0.0, 1.0e9, 1.0),
        ):
            self._add_double(controller_form, name, label, minimum, maximum, step)

        self._build_modal_design_controls(form)

        mimo_box, mimo_form = self._add_section(form, "mimo", "MIMO Thermal-Rate QP")
        for name, label, minimum, maximum, step in (
            ("mimo_lambda_u", "lambda_u heater effort", 0.0, 1.0e9, 0.001),
            ("mimo_rho_du", "rho_du power change", 0.0, 1.0e9, 0.01),
            ("role_contact_tolerance_mm", "role contact tol mm", 0.0, 1.0e9, 1.0e-6),
            ("role_contact_tolerance_max_mm", "role contact max mm", 0.0, 1.0e9, 0.1),
            ("role_contact_tolerance_growth_factor", "role contact growth", 1.01, 1.0e6, 0.1),
            ("mimo_integral_abs_max", "integral abs max", 0.0, 1.0e12, 1.0),
        ):
            self._add_double(mimo_form, name, label, minimum, maximum, step)

        self._build_solver_controls(form)

        display_box, display_form = self._add_section(form, "display", "Display")
        self._add_checkbox(display_form, "autoscale_temperature", "Autoscale temperature")
        self._add_double(display_form, "color_min_K", "color min K", 0.0, 1.0e6, 1.0)
        self._add_double(display_form, "color_max_K", "color max K", 0.0, 1.0e6, 1.0)

    def _build_modal_design_controls(self, form: Any) -> None:
        box, design_form = self._add_section(form, "modal_design", "Controller Design (modal LQR / MIMO PI G)")
        self.modal_temp_spin = self.double_spin(
            0.0, 1.0e6, self._modal_operating_temperature_K, 1.0
        )
        self.modal_temp_spin.setToolTip(
            "Operating temperature to linearize the plant about. BOTH controller builds read "
            "this one field, so the modal artifact and the MIMO PI G matrix describe the same "
            "linearization and can be compared directly.\n\n"
            "It matters because conductance is temperature dependent through k(T) and h(T): a "
            "gain taken at the wrong background is systematically wrong, not just noisy. Set it "
            "to the temperature the plant will actually sit at.\n\n"
            "The modal controller additionally offsets its measurements and setpoints from this."
        )
        self._row(design_form, "modal_operating_temperature", self.modal_temp_spin, "operating T K")
        self.modal_modes_spin = self.int_spin(2, 100000, 120, 1)
        self.modal_modes_spin.setToolTip(
            "Number of slowest thermal modes solved in stage 1 (before balanced truncation). "
            "Clamped to fit the graph."
        )
        self._row(design_form, "modal_modes", self.modal_modes_spin, "slow modes")
        self.modal_order_spin = self.int_spin(1, 100000, 40, 1)
        self.modal_order_spin.setToolTip(
            "Reduced model order r after balanced truncation -- the controller's state dimension "
            "(kept small so it runs on the microcontroller)."
        )
        self._row(design_form, "modal_order", self.modal_order_spin, "reduced order r")
        self.modal_effort_spin = self.double_spin(1.0e-9, 1.0e9, 1.0, 0.1)
        self.modal_effort_spin.setToolTip(
            "LQR control-effort weight rho (R = rho*I, Q = C^T C). Larger rho = gentler, less "
            "aggressive heating; smaller = faster, higher-power response."
        )
        self._row(design_form, "modal_effort", self.modal_effort_spin, "LQR effort weight")
        # A LIVE parameter, unlike the order/effort design knobs beside it: the
        # runtime reads params.modal_integral_gain fresh every step and never reads
        # the artifact's stored integral_gain, so retuning it is decoupled from the
        # LQR build and does not need a rebuild.
        self.modal_integral_spin = self.double_spin(
            0.0, 1.0e9, float(getattr(self.params, "modal_integral_gain", 0.0)), 0.01
        )
        self.modal_integral_spin.setToolTip(
            "Offset-free integral gain the modal controller uses to supply the operating holding "
            "power the linearized model omits. Applies immediately during a run -- no controller "
            "rebuild needed. Roughly 1/tau_dominant is a sane scale; far above that the integrator "
            "commands most of the steady-state correction within a single step and overshoots."
        )
        self.modal_integral_spin.valueChanged.connect(self._changed("modal_integral_gain"))
        self.inputs["modal_integral_gain"] = self.modal_integral_spin
        self._row(design_form, "modal_integral_gain", self.modal_integral_spin, "integral gain")
        # Adaptive (learning) feedforward: online RLS correction of the exact-DC-gain
        # feedforward from the integral's steady-state holding power. Off by default;
        # all of these hot-swap during a running sim (they only change controller
        # behavior, not the plant matrices).
        self._add_checkbox(
            design_form, "modal_adaptive_ff_enabled", "Adaptive feedforward (RLS)", default=False
        )
        self.inputs["modal_adaptive_ff_enabled"].setToolTip(
            "Learn the DC-gain error the model got wrong: regress the integral's steady-state "
            "holding power against the setpoint (recursive least squares) and fold the correction "
            "into the feedforward, so revisited setpoints get the right holding power immediately "
            "instead of waiting for the integral. Bumpless; in-memory only (reset on re-prepare)."
        )
        self._add_double(
            design_form, "modal_adaptive_ff_forgetting", "adaptive forgetting", 0.5, 1.0, 0.001
        )
        self.inputs["modal_adaptive_ff_forgetting"].setToolTip(
            "RLS forgetting factor in (0, 1]. 1 = growing-window (exact, ever-more-confident); "
            "<1 lets a stale estimate fade for a slowly time-varying plant. Keep near 1."
        )
        self._add_double(
            design_form, "modal_adaptive_ff_error_tol_K", "adaptive error tol K", 0.0, 1.0e6, 0.01
        )
        self._add_double(
            design_form, "modal_adaptive_ff_rate_tol_K_per_s", "adaptive rate tol K/s", 0.0, 1.0e6, 1.0e-4
        )
        self.inputs["modal_adaptive_ff_error_tol_K"].setToolTip(
            "Steady-state gate: a learning sample is only taken when every controlled sensor's "
            "tracking error is below this AND its |dT/dt| is below the rate tolerance -- so "
            "transient or saturated data never corrupts the static-map regression."
        )
        self._add_double(
            design_form, "modal_adaptive_ff_max_correction_frac", "adaptive max corr frac", 0.0, 1.0e6, 0.1
        )
        self.inputs["modal_adaptive_ff_max_correction_frac"].setToolTip(
            "Projection guard: the learned feedforward correction is clamped, per heater, to this "
            "fraction of its max power (the effective command is clamped to [0, max] regardless)."
        )
        self.modal_design_button = self._button(
            "Build && Use Modal Controller",
            "build_modal_controller",
            "Reduce the CURRENT graph to a reduced-order LQR controller and load it into the "
            "modal-LQR scheme automatically (saved as modal_controller.npz in the graph folder). "
            "Runs in the background.",
        )
        self._row(design_form, "modal_build", self.modal_design_button)
        self.modal_design_status_label = self.QtWidgets.QLabel("Idle.")
        self.pin_two_line_label(self.modal_design_status_label)
        self._row(design_form, "modal_status", self.modal_design_status_label, "status")

    def _build_solver_controls(self, form: Any) -> None:
        """Implicit-solver knobs. Hidden in the live tab (which keeps whatever the
        graph saved) and shown for a headless run, where they decide whether an
        overnight job converges or crawls."""
        box, solver_form = self._add_section(form, "solver", "Solver")
        self.solver_method_combo = self.QtWidgets.QComboBox()
        self.solver_method_combo.addItems(["tr_bdf2", "backward_euler"])
        self.solver_method_combo.setCurrentText(str(self.params.implicit_sparse_simulation_method))
        self.solver_method_combo.currentTextChanged.connect(
            self._changed("implicit_sparse_simulation_method")
        )
        self._row(solver_form, "implicit_method", self.solver_method_combo, "implicit method")
        self._add_double(solver_form, "implicit_sparse_simulation_rtol", "rtol", 1.0e-14, 1.0, 1.0e-6)
        self._add_int(
            solver_form, "implicit_sparse_simulation_maxiter", "max iterations", 1, 100000, 10
        )
        self._add_checkbox(
            solver_form, "implicit_sparse_adaptive_substeps_enabled", "Adaptive substeps"
        )
        self._add_double(
            solver_form, "implicit_sparse_adaptive_target_delta_K", "substep target dT K", 0.0, 1.0e6, 0.1
        )
        self._add_int(
            solver_form, "implicit_sparse_adaptive_max_substeps", "max substeps", 1, 1000, 1
        )
        self._add_checkbox(
            solver_form, "implicit_sparse_residual_check_enabled", "Residual check"
        )
        self._add_double(
            solver_form, "implicit_capacitance_floor_J_K", "capacitance floor J/K", 0.0, 1.0e9, 0.001
        )
        self.inputs["implicit_capacitance_floor_J_K"].setToolTip(
            "Fixed floor for per-node heat capacity. Degenerate near-zero-capacitance "
            "cells (thin-shell / marker mesh artifacts) blow up the solve's condition number, "
            "so the linear solver returns an inaccurate result that overshoots temperatures "
            "(negative on cooling, runaway-hot on heating). Raise this (e.g. 0.1-1.0) on a graph "
            "that diverges; too high slows the smallest cells' response. 0 disables."
        )
        self._add_double(
            solver_form, "implicit_capacitance_condition_cap", "auto floor: max C ratio", 0.0, 1.0e12, 10.0
        )
        self.inputs["implicit_capacitance_condition_cap"].setToolTip(
            "Automatic capacitance floor scaled to the graph: also floor capacity at "
            "max(C)/this, capping the capacitance spread (a proxy for the solve's condition "
            "number). Only bites on pathological graphs (tiny cells beside bulk); leaves "
            "well-conditioned graphs untouched. Lower = more aggressive (more stable, less "
            "accurate on tiny cells). 0 disables. Default 100 handles most divergences without "
            "hand-tuning the fixed floor above."
        )
        self._add_double(
            solver_form, "implicit_temperature_floor_K", "temperature floor K", 0.0, 1.0e6, 0.001
        )
        self.inputs["implicit_temperature_floor_K"].setToolTip(
            "Clamp every cell to at least this temperature after each implicit step, so a "
            "residual solver error can never leave a non-physical negative temperature. Keep "
            "well below any real cryogenic temperature (e.g. 1e-3 K)."
        )
        self._add_double(
            solver_form, "implicit_temperature_ceiling_K", "temperature ceiling K", 0.0, 1.0e9, 1.0
        )
        self.inputs["implicit_temperature_ceiling_K"].setToolTip(
            "Optional upper clamp (0 = off). Isolated / tiny-capacitance artifact cells (thin "
            "shells, stranded nodes) can absorb heat they can't shed and run away to thousands "
            "of K, aborting the run, while the connected body is fine. Set this well above your "
            "operating regime (e.g. a few hundred K for a cryostat) to pin those artifacts "
            "without affecting real cells."
        )
        self._add_checkbox(solver_form, "gpu_solver_enabled", "Use GPU solver when available")

    def _build_playback_controls(self, form: Any) -> None:
        buttons = []
        for text, action in (
            ("Play", "play"),
            ("Pause", "pause"),
            ("Reset", "reset"),
            ("Step +", "step_forward"),
            ("Step -", "step_backward"),
        ):
            tooltip = None
            if text == "Play":
                tooltip = "Start live playback using the precomputed transition matrix."
            elif text == "Reset":
                tooltip = "Return the simulation to each cell's initial_temperature_K."
            buttons.append(self._button(text, action, tooltip))
        self._row(form, "transport", self._hrow(*buttons))
        self.time_slider = self.QtWidgets.QSlider(self.QtCore.Qt.Horizontal)
        self.time_slider.setRange(0, 0)
        self._connect(self.time_slider.valueChanged, "time_slider_changed")
        self._row(form, "time_slider", self.time_slider, "time")
        self._row(
            form, "save_trajectory", self._button("Save / Export Trajectory", "save_trajectory")
        )
        self._row(
            form,
            "reset_integrators",
            self._button("Reset MIMO Integrators", "reset_controller_integrators"),
        )
        # Headless overnight run: no live visualization, everything saved to
        # simulations/<graph>/<timestamp>/ (recommended for large graphs).
        headless = self.mode == MODE_HEADLESS
        self.run_headless_button = self._button(
            "Start Headless Run" if headless else "Run Headless (save, no viz)",
            "start_headless",
            "Run the full closed-loop simulation with NO live visualization, saving all data, "
            "plots, checkpoints and a report to simulations/<graph>/<timestamp>/. "
            "The window stays responsive; tail status.json for progress. Recommended for large graphs.",
        )
        self.stop_headless_button = self._button(
            "Stop Run" if headless else "Stop Headless Run", "stop_headless"
        )
        self.stop_headless_button.setEnabled(False)
        self._row(
            form, "headless_run", self._hrow(self.run_headless_button, self.stop_headless_button)
        )
        self.open_output_button = self._button("Open Output Folder", "open_output")
        self.open_output_button.setEnabled(False)
        self._row(form, "open_output", self.open_output_button)

    def _build_enabled_io_controls(self, form: Any) -> None:
        box, layout = self._add_section(form, "enabled_io", "Enabled Simulation I/O")
        layout.addRow(
            self._hrow(
                self._button("Enable All", "enable_all_io"),
                self._button("Disable All", "disable_all_io"),
            )
        )
        self.enabled_io_table = self.QtWidgets.QTableWidget(0, 3)
        self.enabled_io_table.setHorizontalHeaderLabels(["cell/node", "heater", "sensor"])
        self.enabled_io_table.verticalHeader().setVisible(False)
        self.enabled_io_table.setEditTriggers(self.QtWidgets.QAbstractItemView.NoEditTriggers)
        self.enabled_io_table.setSelectionBehavior(self.QtWidgets.QAbstractItemView.SelectRows)
        self.enabled_io_table.setMaximumHeight(170)
        self._connect(self.enabled_io_table.itemChanged, "enabled_io_item_changed")
        layout.addRow(self.enabled_io_table)

    def _build_sys_id_controls(self, form: Any) -> None:
        box, sysid_form = self._add_section(
            form, "sys_id", "Simulation Sys ID for Controller Gain Matrix"
        )
        self.sys_id_matrix_combo = self.QtWidgets.QComboBox()
        self._connect(self.sys_id_matrix_combo.currentIndexChanged, "sys_id_matrix_selected")
        sysid_form.addRow(
            "active G matrix",
            self._hrow(
                self.sys_id_matrix_combo, self._button("Refresh Matrices", "refresh_sys_id_matrices")
            ),
        )
        self.sys_id_step_power = self.double_spin(0.0, 1.0e9, 1.0, 0.1)
        self.sys_id_global_temperature_K = self.double_spin(0.0, 1.0e6, 293.15, 1.0)
        self.sys_id_duration_s = self.double_spin(0.0, 1.0e9, 300.0, 10.0)
        self.sys_id_baseline_window_s = self.double_spin(0.0, 1.0e9, 10.0, 1.0)
        self.sys_id_final_window_s = self.double_spin(0.0, 1.0e9, 10.0, 1.0)
        self.sys_id_restore_between_tests = self.checkbox("Restore baseline between heater tests", True)
        self.sys_id_keep_cryocooler_active = self.checkbox("Keep cryocooler active during sys ID", True)
        self.sys_id_uniform_baseline = self.checkbox("Start from uniform baseline temperature", True)
        for label, widget in (
            ("step power Delta P W", self.sys_id_step_power),
            ("background T K", self.sys_id_global_temperature_K),
            ("experiment duration s", self.sys_id_duration_s),
            ("baseline averaging window s", self.sys_id_baseline_window_s),
            ("final averaging window s", self.sys_id_final_window_s),
        ):
            sysid_form.addRow(label, widget)
        sysid_form.addRow(self.sys_id_restore_between_tests)
        sysid_form.addRow(self.sys_id_keep_cryocooler_active)
        sysid_form.addRow(self.sys_id_uniform_baseline)
        self.run_sys_id_button = self._button("Run G_ctrl Sys ID", "run_sys_id")
        self.cancel_sys_id_button = self._button("Cancel Sys ID", "cancel_sys_id")
        self.cancel_sys_id_button.setEnabled(False)
        sysid_form.addRow(self._hrow(self.run_sys_id_button, self.cancel_sys_id_button))
        self.sys_id_progress_label = self.QtWidgets.QLabel("Idle.")
        self.sys_id_progress_label.setWordWrap(True)
        self.sys_id_status_label = self.QtWidgets.QLabel("")
        self.sys_id_status_label.setWordWrap(True)
        sysid_form.addRow("progress", self.sys_id_progress_label)
        sysid_form.addRow(self.sys_id_status_label)

    def _build_stepper_diagnostic_controls(self, form: Any) -> None:
        box, diag_form = self._add_section(form, "stepper_diagnostic", "Solver Diagnostic")
        self.stepper_diagnostic_save = self.checkbox("Save matrices", True)
        self.stepper_diagnostic_button = self._button(
            "Compare Current vs Reference",
            "run_stepper_diagnostic",
            "Compare the current simulation state against one expm_multiply reference solve "
            "to the same time.",
        )
        self.stepper_diagnostic_target_label = self.QtWidgets.QLabel("Uses the current simulation time.")
        self.stepper_diagnostic_target_label.setWordWrap(True)
        diag_form.addRow("target", self.stepper_diagnostic_target_label)
        diag_form.addRow(self._hrow(self.stepper_diagnostic_button, self.stepper_diagnostic_save))
        self.stepper_diagnostic_status_label = self.QtWidgets.QLabel("Idle.")
        self.stepper_diagnostic_status_label.setWordWrap(True)
        diag_form.addRow("result", self.stepper_diagnostic_status_label)

    def _build_component_temperature_controls(self, form: Any) -> None:
        self.component_combo = self.QtWidgets.QComboBox()
        self.component_temperature = self.double_spin(0.0, 1.0e6, 293.15, 1.0)
        self._row(form, "component", self.component_combo, "component")
        self._row(form, "component_initial_K", self.component_temperature, "initial K")
        self._row(
            form,
            "component_apply",
            self._button("Apply To Component", "apply_component_initial_temperature"),
        )

    def _build_status_controls(self, form: Any) -> None:
        self.warning_label = self.QtWidgets.QLabel("")
        self.pin_two_line_label(self.warning_label)
        self._row(form, "warning", self.warning_label)
        self.stats_label = self.QtWidgets.QLabel("No simulation initialized.")
        self.stats_label.setWordWrap(True)
        self._row(form, "stats", self.stats_label)
        self.controller_status_label = self.QtWidgets.QLabel("")
        self.pin_two_line_label(self.controller_status_label)
        self._row(form, "controller_status", self.controller_status_label)

        self.sensor_readout_box = self.QtWidgets.QGroupBox("Thermal I/O Readouts")
        readout_layout = self.QtWidgets.QVBoxLayout(self.sensor_readout_box)
        self.cooling_readout_box = self.QtWidgets.QGroupBox("Cooling")
        cooling_layout = self.QtWidgets.QVBoxLayout(self.cooling_readout_box)
        self.cooling_readout_table = self.QtWidgets.QTableWidget(0, 7)
        self.cooling_readout_table.setHorizontalHeaderLabels(
            [
                "cryocooler",
                "cold-tip temperature",
                "base capacity",
                "scale",
                "applied cooling",
                "receiving nodes",
                "enabled",
            ]
        )
        self.cooling_readout_table.verticalHeader().setVisible(False)
        self.cooling_readout_table.setEditTriggers(self.QtWidgets.QAbstractItemView.NoEditTriggers)
        self.cooling_readout_table.setSelectionBehavior(self.QtWidgets.QAbstractItemView.SelectRows)
        self.cooling_readout_table.setMaximumHeight(130)
        self._connect(self.cooling_readout_table.itemSelectionChanged, "cooling_table_selected")
        cooling_layout.addWidget(self.cooling_readout_table)
        readout_layout.addWidget(self.cooling_readout_box)
        self.heating_readout_box = self.QtWidgets.QGroupBox("Heating")
        heating_layout = self.QtWidgets.QVBoxLayout(self.heating_readout_box)
        self.heating_readout_tree = self.QtWidgets.QTreeWidget()
        self.heating_readout_tree.setHeaderLabels(
            [
                "role",
                "cell/node",
                "measured temperature",
                "desired temperature",
                "error",
                "heater power",
            ]
        )
        self.heating_readout_tree.setSelectionMode(self.QtWidgets.QAbstractItemView.SingleSelection)
        self.heating_readout_tree.setMaximumHeight(220)
        self._connect(self.heating_readout_tree.itemSelectionChanged, "heating_tree_selected")
        heating_layout.addWidget(self.heating_readout_tree)
        readout_layout.addWidget(self.heating_readout_box)
        self.sensor_readout_box.setVisible(False)
        self._row(form, "sensor_readouts", self.sensor_readout_box)
        self.legend_label = self.QtWidgets.QLabel(self._legend_text)
        self.legend_label.setWordWrap(True)
        self._row(form, "legend", self.legend_label)

    # -- the per-node "Parameters" editor ------------------------------------ #
    def _readout_slot(self, action: str, field: str) -> Callable[..., None]:
        callback = self._act(action)
        if callback is None:
            return lambda *_args: None
        return lambda *_args, f=field: callback(f)

    def _readout_field_defaults(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """(node-level, heater-hardware) default values a fresh heater/sensor gets
        in the simulation tab, read straight from the model dataclasses so the
        headless editor -- which has no graph to read from -- shows the same
        reasonable presets (e.g. heater max power 30 W, efficiency 1.0, setpoint
        293.15 K) rather than zeros. Falls back to hardcoded values if models can't
        be imported (e.g. a minimal test stub)."""
        try:
            import dataclasses

            from .models import HeaterProperties, NodeProperties

            node_defaults = {
                f.name: f.default
                for f in dataclasses.fields(NodeProperties)
                if f.default is not dataclasses.MISSING
            }
            heater_defaults = dataclasses.asdict(HeaterProperties())
            return node_defaults, heater_defaults
        except Exception:  # noqa: BLE001 - defaults are a convenience, never fatal
            return (
                {"controller_setpoint_K": 293.15},
                {"heater_min_power_W": 0.0, "heater_max_power_W": 30.0, "heater_efficiency": 1.0},
            )

    def build_readout_editor(self) -> Any:
        """Build the per-heater/sensor/cryocooler "Parameters" box.

        This is the sim tab's heater/sensor editor (setpoint K, heater mode /
        power / efficiency / PID, cryocooler). It sits beside the viewer in the
        live tab and beside the log in the headless tab, so both tabs offer the
        same heater/sensor options. Its ``valueChanged`` slots call back into the
        owning tab via the ``readout_*_change`` actions; the headless tab, which
        has no per-node model to write to, simply passes no such actions.

        Field initial values are the model's own heater/sensor defaults, so the
        headless tab (which can't read a graph) still shows sensible presets. The
        live tab overwrites them from the selected node on every row selection, so
        these seeds only surface where there is no graph to read.
        """
        node_defaults, heater_defaults = self._readout_field_defaults()
        box = self.QtWidgets.QGroupBox("Parameters")
        box.setMinimumWidth(260)
        box.setMaximumWidth(340)
        box.setSizePolicy(self.QtWidgets.QSizePolicy.Fixed, self.QtWidgets.QSizePolicy.Preferred)
        layout = self.QtWidgets.QVBoxLayout(box)
        self.readout_editor_box = box
        self.readout_editor_title = self.QtWidgets.QLabel("Select a readout row.")
        self.readout_editor_title.setWordWrap(True)
        layout.addWidget(self.readout_editor_title)

        self.readout_sensor_editor = self.QtWidgets.QWidget()
        sensor_form = self.QtWidgets.QFormLayout(self.readout_sensor_editor)
        widget = self.double_spin(
            0.0, 1.0e6, float(node_defaults.get("controller_setpoint_K", 293.15)), 1.0
        )
        widget.valueChanged.connect(self._readout_slot("readout_sensor_change", "controller_setpoint_K"))
        self.readout_editor_inputs["controller_setpoint_K"] = widget
        # Named "default" because this is the value applied to the SELECTED/new role,
        # not the target the run tracks. Reading it as the run's setpoint is an easy
        # mistake: it shows 293.15 while a cryogenic run is tracking ~50 K from the
        # "use setpoint" row above.
        widget.setToolTip(
            "Default setpoint for a sensor edited here. This is NOT the run's target: "
            "the headless run uses the 'use setpoint' value above, plus any per-sensor "
            "overrides in the 'Per-sensor setpoints' table."
        )
        sensor_form.addRow("default setpoint K", widget)
        layout.addWidget(self.readout_sensor_editor)

        self.readout_heater_editor = self.QtWidgets.QWidget()
        heater_form = self.QtWidgets.QFormLayout(self.readout_heater_editor)
        mode = self.QtWidgets.QComboBox()
        mode.addItems(["manual", "mimo"])
        mode.currentTextChanged.connect(self._readout_slot("readout_heater_change", "sensor_control_mode"))
        self.readout_editor_inputs["sensor_control_mode"] = mode
        heater_form.addRow("mode", mode)
        for name, label, minimum, maximum, step in (
            ("heater_id", "heater id", -1, 1_000_000_000, 1),
            ("heater_min_power_W", "min power W", 0.0, 1.0e9, 1.0),
            ("heater_max_power_W", "max power W", 0.0, 1.0e9, 1.0),
            ("heater_efficiency", "efficiency", 0.0, 1.0e6, 0.05),
        ):
            if name == "heater_id":
                widget = self.int_spin(int(minimum), int(maximum), 0, int(step))
            else:
                widget = self.double_spin(
                    float(minimum), float(maximum), float(heater_defaults.get(name, 0.0)), float(step)
                )
            widget.valueChanged.connect(self._readout_slot("readout_heater_change", name))
            self.readout_editor_inputs[name] = widget
            heater_form.addRow(label, widget)
        for name, label, minimum, maximum, step in (
            ("sensor_manual_power_W", "manual power W", 0.0, 1.0e9, 1.0),
            ("controller_weight", "weight", 0.0, 1.0e9, 0.1),
            ("sensor_settling_time_s", "settling time s", 0.0, 1.0e9, 1.0),
        ):
            widget = self.double_spin(minimum, maximum, float(node_defaults.get(name, 0.0)), step)
            widget.valueChanged.connect(self._readout_slot("readout_heater_change", name))
            self.readout_editor_inputs[name] = widget
            heater_form.addRow(label, widget)
        layout.addWidget(self.readout_heater_editor)

        self.readout_cooling_editor = self.QtWidgets.QWidget()
        cooling_form = self.QtWidgets.QFormLayout(self.readout_cooling_editor)
        cooling_form.addRow("Model", self.QtWidgets.QLabel("PT60 measured lift curve"))
        for name, label, minimum, maximum, step in (
            ("cryocooler_max_power_W", "Maximum cooling power W", 0.0, 1.0e9, 1.0),
            ("cryocooler_capacity_scale", "Capacity scale", 0.0, 1.0e9, 0.05),
        ):
            widget = self.double_spin(minimum, maximum, float(getattr(self.params, name)), step)
            widget.valueChanged.connect(self._readout_slot("readout_cooling_change", name))
            self.readout_editor_inputs[name] = widget
            cooling_form.addRow(label, widget)
        enabled_widget = self.checkbox(
            "Enabled",
            self.params.cryocooler_enabled,
            self._readout_slot("readout_cooling_change", "cryocooler_enabled"),
        )
        self.readout_editor_inputs["cryocooler_enabled"] = enabled_widget
        cooling_form.addRow(enabled_widget)
        layout.addWidget(self.readout_cooling_editor)
        layout.addStretch(1)

        if self.mode == MODE_HEADLESS:
            # No readout tables to select a row from, so show the whole editor as a
            # static defaults block rather than hiding it until a selection.
            self.readout_editor_title.setText(
                "Heater / sensor / cryocooler defaults (applied to the run's graph)."
            )
        else:
            # Live tab drives it from readout-row selection; hidden until then.
            box.setVisible(False)
        return box

    # -- mode ---------------------------------------------------------------- #
    def _set_row_visible(self, key: str, visible: bool) -> None:
        entry = self._rows.get(key)
        if entry is None:
            return
        form, widget = entry
        widget.setVisible(visible)
        label = None
        label_for_field = getattr(form, "labelForField", None)
        if label_for_field is not None:
            try:
                label = label_for_field(widget)
            except Exception:  # noqa: BLE001 - Qt raises if the row was never added.
                label = None
        if label is not None:
            label.setVisible(visible)

    def _apply_mode(self) -> None:
        if self.mode == MODE_HEADLESS:
            hidden_rows, hidden_sections = _HEADLESS_HIDDEN_ROWS, _HEADLESS_HIDDEN_SECTIONS
        else:
            hidden_rows, hidden_sections = _LIVE_HIDDEN_ROWS, _LIVE_HIDDEN_SECTIONS
        for key in hidden_rows:
            self._set_row_visible(key, False)
        for key in hidden_sections:
            box = self._sections.get(key)
            if box is not None:
                box.setVisible(False)
            # Qt hides a group box's children with it; hide them explicitly too so
            # "is this row shown in this mode?" has one answer, not two.
            section_form = self._section_forms.get(key)
            for row_key, (row_form, _widget) in self._rows.items():
                if row_form is section_form:
                    self._set_row_visible(row_key, False)

    # -- handing the widgets to the owning tab -------------------------------- #
    def export_to(self, owner: Any) -> None:
        """Give ``owner`` a direct attribute for every named widget (and share the
        ``inputs`` dict), so the tab's existing per-widget code needs no changes."""
        owner.inputs = self.inputs
        for name in _EXPORTED:
            setattr(owner, name, getattr(self, name))
        if hasattr(self, "readout_editor_box"):
            owner.readout_editor_inputs = self.readout_editor_inputs
            for name in _EXPORTED_READOUT:
                setattr(owner, name, getattr(self, name))

    # -- parameters ----------------------------------------------------------- #
    def set_params(self, params: SimulationParameters) -> None:
        """Repopulate every parameter widget from ``params``.

        The controller dropdown is left alone: it is populated from the artifacts
        on disk by whichever tab owns the graph folder.
        """
        self.params = params
        for name, widget in self.inputs.items():
            if not hasattr(params, name):
                continue
            value = getattr(params, name)
            widget.blockSignals(True)
            try:
                if hasattr(widget, "setChecked") and isinstance(value, bool):
                    widget.setChecked(bool(value))
                elif hasattr(widget, "setValue"):
                    if isinstance(value, int) and not isinstance(value, bool):
                        widget.setValue(int(value))
                    else:
                        widget.setValue(float(value))
                elif hasattr(widget, "setChecked"):
                    widget.setChecked(bool(value))
            finally:
                widget.blockSignals(False)
        self.input_mode.blockSignals(True)
        self.input_mode.setCurrentText(str(params.input_mode))
        self.input_mode.blockSignals(False)
        self.solver_method_combo.blockSignals(True)
        self.solver_method_combo.setCurrentText(str(params.implicit_sparse_simulation_method))
        self.solver_method_combo.blockSignals(False)
        self.modal_integral_spin.blockSignals(True)
        self.modal_integral_spin.setValue(float(getattr(params, "modal_integral_gain", 0.0)))
        self.modal_integral_spin.blockSignals(False)

    def selected_controller(self) -> tuple[str, str]:
        """(scheme, artifact path) for the selected controller.

        Entries carry their scheme rather than having it inferred, because the list
        now mixes two kinds of artifact: modal-LQR .npz files and MIMO PI DC-gain
        matrices. Older entries that stored a bare path are read as modal-LQR so a
        saved selection keeps working.
        """
        combo = self.controller_scheme_combo
        data = combo.currentData() if hasattr(combo, "currentData") else None
        if isinstance(data, tuple) and len(data) == 2:
            scheme, path = data
            return str(scheme), str(path or "")
        if data:
            return "modal_lqr", str(data)
        return "none", ""

    def selected_controller_artifact(self) -> str:
        """Path of the selected artifact, or "" when none is selected."""
        return self.selected_controller()[1]

    def read(self, base: SimulationParameters | None = None) -> SimulationParameters:
        """Current widget values applied on top of ``base``.

        Uses ``replace()`` so any parameter without a widget here (colormap,
        block-Jacobi knobs, fields added later) keeps its saved value instead of
        silently reverting to a dataclass default.
        """
        values: dict[str, Any] = {}
        for name, widget in self.inputs.items():
            current = getattr(self.params, name, None)
            if hasattr(widget, "isChecked") and isinstance(current, bool):
                values[name] = bool(widget.isChecked())
            elif hasattr(widget, "value"):
                value = widget.value()
                if isinstance(current, int) and not isinstance(current, bool):
                    values[name] = int(value)
                else:
                    values[name] = float(value)
            elif hasattr(widget, "isChecked"):
                values[name] = bool(widget.isChecked())
        values["input_mode"] = self.input_mode.currentText()
        values["implicit_sparse_simulation_method"] = self.solver_method_combo.currentText()
        values["modal_integral_gain"] = float(self.modal_integral_spin.value())
        scheme, artifact = self.selected_controller()
        values["mimo_controller_scheme"] = scheme
        # Each scheme reads its own artifact path, and the other is cleared so a
        # stale path cannot quietly reactivate a scheme that is not selected.
        values["modal_controller_path"] = artifact if scheme == "modal_lqr" else ""
        values["mimo_pi_gain_matrix_path"] = artifact if scheme == "mimo_pi" else ""
        if hasattr(self, "mimo_pi_kp_spin"):
            values["mimo_pi_kp"] = float(self.mimo_pi_kp_spin.value())
            values["mimo_pi_ki"] = float(self.mimo_pi_ki_spin.value())
        return replace(base if base is not None else self.params, **values)
