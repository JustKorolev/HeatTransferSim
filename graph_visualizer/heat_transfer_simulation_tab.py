"""Qt tab for live octree heat-transfer simulation."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable

import numpy as np

try:  # pragma: no cover - import path depends on the installed Qt binding.
    from PySide6 import QtGui
except Exception:  # pragma: no cover
    from qtpy import QtGui

from .diagnostics import log_event, log_exception
from .graph_io import has_generated_role_contact_edges, load_graph_folder, save_graph_folder
from .matrix_builder import build_matrices, refresh_geometry_edges, refresh_radiation_from_exposed_faces
from .modal_reduction import design_modal_controller
from .models import EdgeMode, ThermalGraphModel
from .material_library import is_unassigned_material
from .pyvista_widget import GraphPyVistaWidget
from .role_pairing import sensor_readout_temperature_K
from .simulation_model import PreparedSimulation, prepare_simulation, save_trajectory
from .simulation_runner import RunConfig, SimulationRunner
from .simulation_parameters import (
    SimulationParameters,
    apply_initial_temperature_parameter_payload,
    initial_temperature_parameter_payload,
    load_simulation_parameters,
    save_simulation_parameters,
)
from .simulation_diagnostics import compare_current_state_to_expm_multiply, save_current_state_comparison
from .sys_id_artifacts import (
    list_sys_id_gain_matrices,
    load_sys_id_gain_matrix,
    save_sys_id_gain_matrix,
    update_sys_id_gain_matrix,
)


QT_SLIDER_MAXIMUM = 2_147_483_647
_REINITIALIZE_PARAMETER_FIELDS = {
    "dt_s",
    "use_ambient_radiation",
    "T_env_K",
    "interior_environment_temperature_K",
    "use_radiative_coupling",
    "input_mode",
    "cryocooler_max_power_W",
    "cryocooler_capacity_scale",
    "cryocooler_enabled",
    "use_temperature_dependent_properties",
    "use_midpoint_property_coupling",
    "copper_rrr",
}
_DISPLAY_PARAMETER_FIELDS = {
    "autoscale_temperature",
    "color_min_K",
    "color_max_K",
}
_CONTROLLER_PARAMETER_FIELDS = {
    "mimo_controller_enabled",
    "mimo_hold_threshold_K",
    "mimo_coarse_threshold_K",
    "mimo_default_heater_max_power_W",
    "mimo_lambda_u",
    "mimo_rho_du",
    "mimo_heater_slew_rate_W_per_s",
    "mimo_v_cmd_abs_max_K_per_s",
    "heater_sensor_pair_alpha",
    "drift_lpf_tau_s",
    "derivative_dt_floor_s",
    "mimo_integral_abs_max",
    "mimo_freeze_integral_when_saturated",
    # Adaptive feedforward: pure controller behavior (no plant-matrix change), so
    # these hot-swap during a running sim like the other controller knobs.
    "modal_adaptive_ff_enabled",
    "modal_adaptive_ff_forgetting",
    "modal_adaptive_ff_p0",
    "modal_adaptive_ff_rate_tol_K_per_s",
    "modal_adaptive_ff_error_tol_K",
    "modal_adaptive_ff_max_correction_frac",
}
_CONTROLLER_RUNTIME_HOTSWAP_FIELDS = set(_CONTROLLER_PARAMETER_FIELDS)
_LIGHTWEIGHT_RUNTIME_PARAMETER_FIELDS = {
    "playback_speed",
    "loop_playback",
}
_NONBLOCKING_PARAMETER_FIELDS = _LIGHTWEIGHT_RUNTIME_PARAMETER_FIELDS | _DISPLAY_PARAMETER_FIELDS
_READOUT_SENSOR_CONTROLLER_FIELDS = (
    "controller_setpoint_K",
)
_READOUT_HEATER_CONTROLLER_FIELDS = (
    "sensor_manual_power_W",
    "controller_weight",
    "sensor_settling_time_s",
    "controller_kp_coarse",
    "controller_ki_coarse",
    "controller_kd_coarse",
    "controller_kp_hold",
    "controller_ki_hold",
    "controller_kd_hold",
)
_READOUT_HEATER_HARDWARE_FIELDS = (
    "heater_id",
    "heater_min_power_W",
    "heater_max_power_W",
    "heater_efficiency",
)


class HeatTransferSimulationTab:
    """Live matrix-exponential heat-transfer simulation view."""

    def __init__(
        self,
        qt: Any,
        parent: Any,
        current_model: Callable[[], ThermalGraphModel],
        current_folder: Callable[[], Path | None],
        on_select_node: Callable[[int], None] | None = None,
        on_status: Callable[[str, bool], None] | None = None,
        on_controller_gain_matrix_changed: Callable[[], None] | None = None,
        hide_unassigned_getter: Callable[[], bool] | None = None,
        on_hide_unassigned_toggled: Callable[[bool], None] | None = None,
    ) -> None:
        self.QtCore = qt.QtCore
        self.QtGui = QtGui
        self.QtWidgets = qt.QtWidgets
        self.current_model = current_model
        self.current_folder = current_folder
        self.on_select_node = on_select_node
        self.on_status = on_status
        self.on_controller_gain_matrix_changed = on_controller_gain_matrix_changed
        self._hide_unassigned_getter = hide_unassigned_getter or (lambda: True)
        self.on_hide_unassigned_toggled = on_hide_unassigned_toggled
        self.model: ThermalGraphModel | None = None
        self.folder: Path | None = None
        self.matrices: dict[str, np.ndarray] = {}
        self.params = SimulationParameters()
        self.parameter_extras: dict[str, Any] = {}
        self.prepared: PreparedSimulation | None = None
        self.temperature_by_node: dict[int, float] = {}
        self.inputs: dict[str, Any] = {}
        self._refreshing_sys_id_matrix_combo = False
        self.enabled_heater_node_ids: set[int] = set()
        self.enabled_sensor_node_ids: set[int] = set()
        self._known_heater_node_ids: set[int] = set()
        self._known_sensor_node_ids: set[int] = set()
        self._enabled_io_initialized = False
        self._syncing_enabled_io_table = False
        self._simulation_reinitialize_pending = False
        self.widget = self.QtWidgets.QWidget(parent)
        self.timer = self.QtCore.QTimer(self.widget)
        self.timer.timeout.connect(self.step_forward)
        self.simulation_worker_timer = self.QtCore.QTimer(self.widget)
        self.simulation_worker_timer.timeout.connect(self._poll_simulation_worker)
        self.simulation_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="HeatTransferSimulation")
        self.simulation_future: Future[dict[str, Any]] | None = None
        self.simulation_cancel_event: threading.Event | None = None
        self._simulation_worker_mode: str | None = None
        # Render throttle + cached role-node lists so the GUI thread stays light
        # while the solver worker runs (keeps the parameter window responsive).
        self._last_render_end_s: float = 0.0
        self._last_render_duration_s: float = 0.0
        self._role_cache_model_id: int | None = None
        self._cooling_nodes_cache: list[Any] = []
        self._heating_sensor_nodes_cache: list[Any] = []
        self._heaters_by_sensor_cache: dict[int, set[int]] = {}
        self.stepper_diagnostic_future: Future[dict[str, Any]] | None = None
        self.stepper_diagnostic_timer = self.QtCore.QTimer(self.widget)
        self.stepper_diagnostic_timer.timeout.connect(self._poll_stepper_diagnostic_worker)
        self.modal_design_future: Future[Any] | None = None
        self.modal_design_timer = self.QtCore.QTimer(self.widget)
        self.modal_design_timer.timeout.connect(self._poll_modal_design_worker)
        self._modal_design_progress: dict[str, str] = {"message": ""}
        self._readout_editor_syncing = False
        self._readout_editor_kind: str | None = None
        self._readout_editor_node_id: int | None = None
        self._readout_editor_sensor_id: int | None = None
        self.readout_editor_inputs: dict[str, Any] = {}
        self._pending_controller_runtime_params: SimulationParameters | None = None
        self._pending_controller_runtime_fields: set[str] = set()
        self._pending_editor_controller_refresh: tuple[ThermalGraphModel, Path | None] | None = None
        self.parameter_save_timer = self.QtCore.QTimer(self.widget)
        self.parameter_save_timer.setSingleShot(True)
        self.parameter_save_timer.timeout.connect(self._flush_deferred_parameter_save)
        self.sys_id_timer = self.QtCore.QTimer(self.widget)
        self.sys_id_timer.timeout.connect(self._step_sys_id)
        self.sys_id_state: dict[str, Any] | None = None
        self._build_layout()
        self.refresh_graph_list()

    def _build_layout(self) -> None:
        layout = self.QtWidgets.QHBoxLayout(self.widget)
        self.controls_scroll = self.QtWidgets.QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setMinimumWidth(320)
        controls = self.QtWidgets.QWidget()
        self.controls_scroll.setWidget(controls)
        form = self.QtWidgets.QFormLayout(controls)

        self.graph_combo = self.QtWidgets.QComboBox()
        refresh = self.QtWidgets.QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_graph_list)
        graph_row = self.QtWidgets.QHBoxLayout()
        graph_row.addWidget(self.graph_combo, 1)
        graph_row.addWidget(refresh)
        form.addRow("graph", graph_row)
        load_selected = self.QtWidgets.QPushButton("Load Selected Graph")
        load_selected.clicked.connect(self.load_selected_graph)
        load_current = self.QtWidgets.QPushButton("Use Current Editor Graph")
        load_current.clicked.connect(self.use_current_graph)
        form.addRow(load_selected)
        form.addRow(load_current)

        self._add_parameter_controls(form)
        self._add_playback_controls(form)
        self._add_enabled_io_controls(form)
        self._add_sys_id_controls(form)
        self._add_stepper_diagnostic_controls(form)
        self._add_component_temperature_controls(form)

        self.warning_label = self.QtWidgets.QLabel("")
        self._pin_two_line_label(self.warning_label)
        form.addRow(self.warning_label)
        self.stats_label = self.QtWidgets.QLabel("No simulation initialized.")
        self.stats_label.setWordWrap(True)
        form.addRow(self.stats_label)
        self.controller_status_label = self.QtWidgets.QLabel("")
        self._pin_two_line_label(self.controller_status_label)
        form.addRow(self.controller_status_label)
        self.sensor_readout_box = self.QtWidgets.QGroupBox("Thermal I/O Readouts")
        readout_layout = self.QtWidgets.QVBoxLayout(self.sensor_readout_box)
        self.cooling_readout_box = self.QtWidgets.QGroupBox("Cooling")
        cooling_layout = self.QtWidgets.QVBoxLayout(self.cooling_readout_box)
        self.cooling_readout_table = self.QtWidgets.QTableWidget(0, 7)
        self.cooling_readout_table.setHorizontalHeaderLabels(
            ["cryocooler", "cold-tip temperature", "base capacity", "scale", "applied cooling", "receiving nodes", "enabled"]
        )
        self.cooling_readout_table.verticalHeader().setVisible(False)
        self.cooling_readout_table.setEditTriggers(self.QtWidgets.QAbstractItemView.NoEditTriggers)
        self.cooling_readout_table.setSelectionBehavior(self.QtWidgets.QAbstractItemView.SelectRows)
        self.cooling_readout_table.setMaximumHeight(130)
        self.cooling_readout_table.itemSelectionChanged.connect(self._handle_cooling_table_selection)
        cooling_layout.addWidget(self.cooling_readout_table)
        readout_layout.addWidget(self.cooling_readout_box)
        self.heating_readout_box = self.QtWidgets.QGroupBox("Heating")
        heating_layout = self.QtWidgets.QVBoxLayout(self.heating_readout_box)
        self.heating_readout_tree = self.QtWidgets.QTreeWidget()
        self.heating_readout_tree.setHeaderLabels(["role", "cell/node", "measured temperature", "desired temperature", "error", "heater power"])
        self.heating_readout_tree.setSelectionMode(self.QtWidgets.QAbstractItemView.SingleSelection)
        self.heating_readout_tree.setMaximumHeight(220)
        self.heating_readout_tree.itemSelectionChanged.connect(self._handle_heating_tree_selection)
        heating_layout.addWidget(self.heating_readout_tree)
        readout_layout.addWidget(self.heating_readout_box)
        self.sensor_readout_box.setVisible(False)
        form.addRow(self.sensor_readout_box)
        self.legend_label = self.QtWidgets.QLabel(self._legend_text())
        self.legend_label.setWordWrap(True)
        form.addRow(self.legend_label)

        self._build_readout_parameter_editor()
        self.viewer = GraphPyVistaWidget(
            self.widget,
            on_pick_node=self._handle_pick,
            tooltip_for_node=self._tooltip_for_node,
        )
        viewer_panel = self.QtWidgets.QWidget(self.widget)
        viewer_layout = self.QtWidgets.QVBoxLayout(viewer_panel)
        toggles = self.QtWidgets.QHBoxLayout()
        self.show_heaters = self._checkbox("Heaters", True, self._handle_marker_toggle)
        self.show_sensors = self._checkbox("Sensors", True, self._handle_marker_toggle)
        self.show_coolers = self._checkbox("Cryocoolers", True, self._handle_marker_toggle)
        self.hide_unassigned_checkbox = self._checkbox(
            "Hide unassigned",
            bool(self._hide_unassigned_getter()),
            self._handle_hide_unassigned_toggled,
        )
        self.hide_unassigned_checkbox.setToolTip(
            "Hide cells with no assigned material (e.g. 'Unassigned (ignored)', 'Not assigned'). "
            "'ZERO MATTER' cells stay visible. Synced with the editor; view-only, the simulation "
            "is unaffected."
        )
        toggles.addWidget(self.show_heaters)
        toggles.addWidget(self.show_sensors)
        toggles.addWidget(self.show_coolers)
        toggles.addWidget(self.hide_unassigned_checkbox)
        toggles.addWidget(self.QtWidgets.QLabel("Opacity"))
        self.opacity_slider = self._view_slider(5, 100, 34, self._handle_visual_control_changed)
        toggles.addWidget(self.opacity_slider)
        self.depth_focus_toggle = self._checkbox("Depth", False, self._handle_visual_control_changed)
        toggles.addWidget(self.depth_focus_toggle)
        self.depth_axis_combo = self.QtWidgets.QComboBox()
        self.depth_axis_combo.addItems(["X", "Y", "Z"])
        self.depth_axis_combo.setCurrentText("Z")
        self.depth_axis_combo.currentTextChanged.connect(self._handle_visual_control_changed)
        toggles.addWidget(self.depth_axis_combo)
        self.depth_slider = self._view_slider(0, 100, 50, self._handle_visual_control_changed)
        toggles.addWidget(self.depth_slider)
        toggles.addWidget(self.QtWidgets.QLabel("Width"))
        self.depth_width_slider = self._view_slider(1, 100, 12, self._handle_visual_control_changed)
        toggles.addWidget(self.depth_width_slider)
        toggles.addStretch(1)
        viewer_layout.addLayout(toggles)
        cross_section_row = self.QtWidgets.QHBoxLayout()
        self.cross_section_toggle = self._checkbox("Cross section", False, self._handle_visual_control_changed)
        self.cross_section_toggle.setToolTip("Clip away cells below a movable axis-aligned plane.")
        cross_section_row.addWidget(self.cross_section_toggle)
        cross_section_row.addWidget(self.QtWidgets.QLabel("Axis"))
        self.cross_section_axis_combo = self.QtWidgets.QComboBox()
        self.cross_section_axis_combo.addItems(["X", "Y", "Z"])
        self.cross_section_axis_combo.setCurrentText("Z")
        self.cross_section_axis_combo.currentTextChanged.connect(self._handle_visual_control_changed)
        cross_section_row.addWidget(self.cross_section_axis_combo)
        cross_section_row.addWidget(self.QtWidgets.QLabel("Cut"))
        self.cross_section_slider = self._view_slider(0, 100, 50, self._handle_visual_control_changed)
        self.cross_section_slider.setToolTip("Move the cut from the minimum to maximum axis coordinate.")
        self.cross_section_slider.setFixedWidth(220)
        cross_section_row.addWidget(self.cross_section_slider)
        self.cross_section_value_label = self.QtWidgets.QLabel("50%")
        self.cross_section_value_label.setMinimumWidth(110)
        cross_section_row.addWidget(self.cross_section_value_label)
        cross_section_row.addStretch(1)
        viewer_layout.addLayout(cross_section_row)
        self.viewer.set_toggles(
            False,
            False,
            self.show_heaters.isChecked(),
            self.show_sensors.isChecked(),
            self.show_coolers.isChecked(),
        )
        self._sync_view_controls_to_viewer()
        viewer_layout.addWidget(self.viewer.interactor, 1)
        layout.addWidget(self.readout_editor_box, 0, self.QtCore.Qt.AlignBottom)
        layout.addWidget(viewer_panel, 1)

    def _add_parameter_controls(self, form: Any) -> None:
        run_box, run_form = self._section("Run")
        run = self.QtWidgets.QPushButton("Initialize")
        run.setToolTip("Build the simulation using the latest graph, matrices, and controller settings.")
        run.clicked.connect(self.initialize_simulation)
        run_form.addRow(run)
        for name, label, minimum, maximum, step in (
            ("dt_s", "dt_s", 1.0e-9, 1.0e9, 1.0),
            ("t_final_s", "t_final_s", 0.0, 1.0e12, 60.0),
            ("playback_speed", "playback speed", 0.01, 1.0e6, 0.25),
        ):
            self._add_double_parameter(run_form, name, label, minimum, maximum, step)
        self._add_int_parameter(run_form, "simulation_history_limit", "history limit", 0, 1_000_000, 1)
        self.inputs["loop_playback"] = self._checkbox(
            "Loop playback", self.params.loop_playback, lambda *_: self._handle_parameter_change("loop_playback")
        )
        run_form.addRow(self.inputs["loop_playback"])
        self.input_mode = self.QtWidgets.QComboBox()
        self.input_mode.addItems(["zero", "heater_inputs"])
        self.input_mode.setCurrentText(self.params.input_mode)
        self.input_mode.currentTextChanged.connect(lambda *_: self._handle_parameter_change("input_mode"))
        run_form.addRow("input mode", self.input_mode)
        self._controller_scheme_labels = {
            "pid_qp": "PID + QP allocator",
            "modal_lqr": "Modal LQR (reduced-model)",
        }
        self.controller_scheme_combo = self.QtWidgets.QComboBox()
        self.controller_scheme_combo.addItems(list(self._controller_scheme_labels.values()))
        current_scheme = str(getattr(self.params, "mimo_controller_scheme", "pid_qp"))
        self.controller_scheme_combo.setCurrentText(
            self._controller_scheme_labels.get(current_scheme, "PID + QP allocator")
        )
        self.controller_scheme_combo.setToolTip(
            "Heater controller for 'heater_inputs' mode.\n"
            "- PID + QP allocator: the standard controller.\n"
            "- Modal LQR (reduced-model): reduced-order LQR + regularized static state estimate; "
            "needs 'modal_controller.npz' for this graph (build it with the 'Modal LQR Design' panel "
            "below, or tools/analyze_plant_modes.py). "
            "Falls back to PID+QP if the artifact is missing or built for a different graph."
        )
        self.controller_scheme_combo.currentTextChanged.connect(
            lambda *_: self._handle_parameter_change("mimo_controller_scheme")
        )
        run_form.addRow("controller", self.controller_scheme_combo)
        form.addRow(run_box)

        environment_box, environment_form = self._section("Environment")
        self._add_double_parameter(environment_form, "T_env_K", "exterior / ambient T K", 0.0, 1.0e6, 1.0)
        self.inputs["T_env_K"].setToolTip(
            "Radiative background for the OUTSIDE of the assembly (room / ambient surroundings)."
        )
        self._add_double_parameter(
            environment_form, "interior_environment_temperature_K", "interior (cryo) T K", 0.0, 1.0e6, 1.0
        )
        self.inputs["interior_environment_temperature_K"].setToolTip(
            "Radiative background for the INSIDE of the assembly (cryocooled vacuum enclosure). "
            "Inward-facing surfaces radiate to this once view-factor classification assigns them."
        )
        self.inputs["use_ambient_radiation"] = self._checkbox(
            "Use ambient radiation",
            self.params.use_ambient_radiation,
            lambda *_: self._handle_parameter_change("use_ambient_radiation"),
        )
        environment_form.addRow(self.inputs["use_ambient_radiation"])
        self.inputs["use_radiative_coupling"] = self._checkbox(
            "Surface-to-surface radiative coupling (ray-traced)",
            getattr(self.params, "use_radiative_coupling", False),
            lambda *_: self._handle_parameter_change("use_radiative_coupling"),
        )
        self.inputs["use_radiative_coupling"].setToolTip(
            "Ray-trace view factors over the exposed faces so parts exchange radiation with "
            "each other (a hot part can warm a cold part), not just with the background. "
            "One-time precompute when the simulation is prepared; skipped for very large graphs."
        )
        environment_form.addRow(self.inputs["use_radiative_coupling"])
        # Bulk initial-temperature control: set initial_temperature_K on EVERY
        # component at once (the state the simulation starts from / resets to).
        self.initial_temperature_all_spin = self._double_spin(0.0, 1.0e6, 293.15, 1.0)
        set_all_initial = self.QtWidgets.QPushButton("Set all components")
        set_all_initial.setToolTip(
            "Set the initial temperature of EVERY component to this value. Updates the loaded "
            "graph; if a simulation is already initialized it resets to this immediately, "
            "otherwise it takes effect on the next Initialize."
        )
        set_all_initial.clicked.connect(self._set_all_initial_temperatures)
        initial_temp_container = self.QtWidgets.QWidget()
        initial_temp_layout = self.QtWidgets.QHBoxLayout(initial_temp_container)
        initial_temp_layout.setContentsMargins(0, 0, 0, 0)
        initial_temp_layout.addWidget(self.initial_temperature_all_spin, 1)
        initial_temp_layout.addWidget(set_all_initial)
        environment_form.addRow("initial T (all) K", initial_temp_container)
        # Testing helper: assign each sensor a random controller SETPOINT (desired
        # temperature) = center +/- a random mK-scale spread (e.g. ~50 K +/- tens of mK).
        self.sensor_random_center_spin = self._double_spin(0.0, 1.0e6, 50.0, 1.0)
        self.sensor_random_spread_mK_spin = self._double_spin(0.0, 1.0e6, 50.0, 1.0)
        randomize_setpoints = self.QtWidgets.QPushButton("Randomize setpoints")
        randomize_setpoints.setToolTip(
            "Assign each sensor a random controller setpoint (desired temperature) = center +/- a "
            "uniform random offset within the spread (mK). For testing how the controller drives "
            "the sensors to distinct targets. Applied live (the controller reads setpoints each step)."
        )
        randomize_setpoints.clicked.connect(self._randomize_sensor_setpoints)
        rand_container = self.QtWidgets.QWidget()
        rand_layout = self.QtWidgets.QHBoxLayout(rand_container)
        rand_layout.setContentsMargins(0, 0, 0, 0)
        rand_layout.addWidget(self.sensor_random_center_spin, 1)
        rand_layout.addWidget(self.QtWidgets.QLabel("K  ±"))
        rand_layout.addWidget(self.sensor_random_spread_mK_spin, 1)
        rand_layout.addWidget(self.QtWidgets.QLabel("mK"))
        rand_layout.addWidget(randomize_setpoints)
        environment_form.addRow("randomize setpoints", rand_container)
        form.addRow(environment_box)

        properties_box, properties_form = self._section("Material Properties")
        self.inputs["use_temperature_dependent_properties"] = self._checkbox(
            "Temperature-dependent cp(T)/k(T)",
            self.params.use_temperature_dependent_properties,
            lambda *_: self._handle_parameter_change("use_temperature_dependent_properties"),
        )
        self.inputs["use_temperature_dependent_properties"].setToolTip(
            "Recompute per-node C(T)=m*cp(T) and conduction/contact from NIST cryogenic "
            "curves each step, instead of using constant room-temperature properties."
        )
        properties_form.addRow(self.inputs["use_temperature_dependent_properties"])
        self._add_int_parameter(properties_form, "copper_rrr", "Copper RRR", 1, 100000, 10)
        self.inputs["copper_rrr"].setToolTip(
            "Residual resistivity ratio for OFHC copper thermal conductivity k(T). "
            "NIST fits exist for 50/100/150/300/500 (the nearest is used). "
            "Only affects runs with temperature-dependent properties enabled."
        )
        self.inputs["use_midpoint_property_coupling"] = self._checkbox(
            "Midpoint property/radiation coupling",
            getattr(self.params, "use_midpoint_property_coupling", True),
            lambda *_: self._handle_parameter_change("use_midpoint_property_coupling"),
        )
        self.inputs["use_midpoint_property_coupling"].setToolTip(
            "Evaluate the temperature-dependent properties and radiation at a "
            "predicted midpoint (2nd-order-in-dt splitting) instead of the "
            "step-start temperature. More accurate during fast transients; adds "
            "one operator rebuild per step and only when those terms are active."
        )
        properties_form.addRow(self.inputs["use_midpoint_property_coupling"])
        form.addRow(properties_box)

        cooler_box, cooler_form = self._section("Cryocooler")
        cooler_model = self.QtWidgets.QLabel("PT60 measured lift curve")
        cooler_form.addRow("Model", cooler_model)
        for name, label, minimum, maximum, step in (
            ("cryocooler_max_power_W", "Maximum cooling power W", 0.0, 1.0e9, 1.0),
            ("cryocooler_capacity_scale", "Capacity scale", 0.0, 1.0e9, 0.05),
        ):
            self._add_double_parameter(cooler_form, name, label, minimum, maximum, step)
        self.inputs["cryocooler_enabled"] = self._checkbox(
            "Enabled",
            self.params.cryocooler_enabled,
            lambda *_: self._handle_parameter_change("cryocooler_enabled"),
        )
        cooler_form.addRow(self.inputs["cryocooler_enabled"])
        form.addRow(cooler_box)

        # Global controller limits: enforced by BOTH the PID+QP and modal-LQR
        # schemes (absolute heater-power clamp + hard slew rate). "max rate cmd"
        # additionally bounds the PID+QP rate command; it is inert for modal-LQR
        # (which has no rate command) but kept here as a global controller knob.
        controller_box, controller_form = self._section("Controller (global limits)")
        for name, label, minimum, maximum, step in (
            ("mimo_default_heater_max_power_W", "max heater power W", 0.0, 1.0e9, 1.0),
            ("mimo_heater_slew_rate_W_per_s", "hard slew W/s", 0.0, 1.0e9, 1.0),
            ("mimo_v_cmd_abs_max_K_per_s", "max rate cmd K/s", 0.0, 1.0e9, 0.01),
        ):
            self._add_double_parameter(controller_form, name, label, minimum, maximum, step)
        form.addRow(controller_box)

        self._build_modal_design_controls(form)

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
            self._add_double_parameter(mimo_form, name, label, minimum, maximum, step)
        self.inputs["mimo_freeze_integral_when_saturated"] = self._checkbox(
            "Freeze integral when saturated",
            self.params.mimo_freeze_integral_when_saturated,
            lambda *_: self._handle_parameter_change("mimo_freeze_integral_when_saturated"),
        )
        mimo_form.addRow(self.inputs["mimo_freeze_integral_when_saturated"])
        form.addRow(mimo_box)

        display_box, display_form = self._section("Display")
        self.inputs["autoscale_temperature"] = self._checkbox(
            "Autoscale temperature",
            self.params.autoscale_temperature,
            lambda *_: self._handle_parameter_change("autoscale_temperature"),
        )
        display_form.addRow(self.inputs["autoscale_temperature"])
        self._add_double_parameter(display_form, "color_min_K", "color min K", 0.0, 1.0e6, 1.0)
        self._add_double_parameter(display_form, "color_max_K", "color max K", 0.0, 1.0e6, 1.0)
        form.addRow(display_box)

    def _add_playback_controls(self, form: Any) -> None:
        row = self.QtWidgets.QHBoxLayout()
        for text, callback in (
            ("Play", self.play),
            ("Pause", self.pause),
            ("Reset", self.reset),
            ("Step +", self.step_forward),
            ("Step -", self.step_backward),
        ):
            button = self.QtWidgets.QPushButton(text)
            if text == "Play":
                button.setToolTip("Start live playback using the precomputed transition matrix.")
            elif text == "Reset":
                button.setToolTip("Return the simulation to each cell's initial_temperature_K.")
            button.clicked.connect(callback)
            row.addWidget(button)
        form.addRow(row)
        self.time_slider = self.QtWidgets.QSlider(self.QtCore.Qt.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_slider.valueChanged.connect(self._handle_time_slider)
        form.addRow("time", self.time_slider)
        save = self.QtWidgets.QPushButton("Save / Export Trajectory")
        save.clicked.connect(self.save_current_trajectory)
        form.addRow(save)
        reset_controller = self.QtWidgets.QPushButton("Reset MIMO Integrators")
        reset_controller.clicked.connect(self.reset_controller_integrators)
        form.addRow(reset_controller)
        # Headless overnight run: no live visualization, everything saved to
        # simulations/<graph>/<timestamp>/ (recommended for large graphs).
        headless_row = self.QtWidgets.QHBoxLayout()
        self.run_headless_button = self.QtWidgets.QPushButton("Run Headless (save, no viz)")
        self.run_headless_button.setToolTip(
            "Run the full closed-loop simulation with NO live visualization, saving all data, "
            "plots, checkpoints and a report to simulations/<graph>/<timestamp>/. "
            "The window stays responsive; tail status.json for progress. Recommended for large graphs."
        )
        self.run_headless_button.clicked.connect(self.run_headless_simulation)
        self.stop_headless_button = self.QtWidgets.QPushButton("Stop Headless Run")
        self.stop_headless_button.clicked.connect(self.stop_headless_simulation)
        self.stop_headless_button.setEnabled(False)
        headless_row.addWidget(self.run_headless_button)
        headless_row.addWidget(self.stop_headless_button)
        form.addRow(headless_row)

    def _add_stepper_diagnostic_controls(self, form: Any) -> None:
        box, diag_form = self._section("Solver Diagnostic")
        self.stepper_diagnostic_save = self._checkbox("Save matrices", True)
        self.stepper_diagnostic_button = self.QtWidgets.QPushButton("Compare Current vs Reference")
        self.stepper_diagnostic_button.setToolTip(
            "Compare the current simulation state against one expm_multiply reference solve to the same time."
        )
        self.stepper_diagnostic_button.clicked.connect(self.run_stepper_diagnostic)
        button_row = self.QtWidgets.QHBoxLayout()
        button_row.addWidget(self.stepper_diagnostic_button)
        button_row.addWidget(self.stepper_diagnostic_save)
        self.stepper_diagnostic_target_label = self.QtWidgets.QLabel("Uses the current simulation time.")
        self.stepper_diagnostic_target_label.setWordWrap(True)
        diag_form.addRow("target", self.stepper_diagnostic_target_label)
        diag_form.addRow(button_row)
        self.stepper_diagnostic_status_label = self.QtWidgets.QLabel("Idle.")
        self.stepper_diagnostic_status_label.setWordWrap(True)
        diag_form.addRow("result", self.stepper_diagnostic_status_label)
        form.addRow(box)

    def _build_readout_parameter_editor(self) -> None:
        self.readout_editor_box = self.QtWidgets.QGroupBox("Parameters")
        self.readout_editor_box.setVisible(False)
        self.readout_editor_box.setMinimumWidth(260)
        self.readout_editor_box.setMaximumWidth(340)
        self.readout_editor_box.setSizePolicy(
            self.QtWidgets.QSizePolicy.Fixed,
            self.QtWidgets.QSizePolicy.Preferred,
        )
        layout = self.QtWidgets.QVBoxLayout(self.readout_editor_box)
        self.readout_editor_title = self.QtWidgets.QLabel("Select a readout row.")
        self.readout_editor_title.setWordWrap(True)
        layout.addWidget(self.readout_editor_title)

        self.readout_sensor_editor = self.QtWidgets.QWidget()
        sensor_form = self.QtWidgets.QFormLayout(self.readout_sensor_editor)
        widget = self._double_spin(0.0, 1.0e6, 293.15, 1.0)
        widget.valueChanged.connect(lambda *_args: self._apply_readout_sensor_editor_change("controller_setpoint_K"))
        self.readout_editor_inputs["controller_setpoint_K"] = widget
        sensor_form.addRow("setpoint K", widget)
        layout.addWidget(self.readout_sensor_editor)

        self.readout_heater_editor = self.QtWidgets.QWidget()
        heater_form = self.QtWidgets.QFormLayout(self.readout_heater_editor)
        mode = self.QtWidgets.QComboBox()
        mode.addItems(["manual", "mimo"])
        mode.currentTextChanged.connect(lambda *_: self._apply_readout_heater_editor_change("sensor_control_mode"))
        self.readout_editor_inputs["sensor_control_mode"] = mode
        heater_form.addRow("mode", mode)
        for name, label, minimum, maximum, step in (
            ("heater_id", "heater id", -1, 1_000_000_000, 1),
            ("heater_min_power_W", "min power W", 0.0, 1.0e9, 1.0),
            ("heater_max_power_W", "max power W", 0.0, 1.0e9, 1.0),
            ("heater_efficiency", "efficiency", 0.0, 1.0e6, 0.05),
        ):
            if name == "heater_id":
                widget = self._int_spin(int(minimum), int(maximum), 0, int(step))
                widget.valueChanged.connect(lambda *_args, field=name: self._apply_readout_heater_editor_change(field))
            else:
                widget = self._double_spin(float(minimum), float(maximum), 0.0, float(step))
                widget.valueChanged.connect(lambda *_args, field=name: self._apply_readout_heater_editor_change(field))
            self.readout_editor_inputs[name] = widget
            heater_form.addRow(label, widget)
        for name, label, minimum, maximum, step in (
            ("sensor_manual_power_W", "manual power W", 0.0, 1.0e9, 1.0),
            ("controller_weight", "weight", 0.0, 1.0e9, 0.1),
            ("sensor_settling_time_s", "settling time s", 0.0, 1.0e9, 1.0),
            ("controller_kp_coarse", "coarse kP", 0.0, 1.0e9, 0.1),
            ("controller_ki_coarse", "coarse kI", 0.0, 1.0e9, 0.1),
            ("controller_kd_coarse", "coarse kD", 0.0, 1.0e9, 0.1),
            ("controller_kp_hold", "hold kP", 0.0, 1.0e9, 0.1),
            ("controller_ki_hold", "hold kI", 0.0, 1.0e9, 0.1),
            ("controller_kd_hold", "hold kD", 0.0, 1.0e9, 0.1),
        ):
            widget = self._double_spin(minimum, maximum, 0.0, step)
            widget.valueChanged.connect(lambda *_args, field=name: self._apply_readout_heater_editor_change(field))
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
            widget = self._double_spin(minimum, maximum, float(getattr(self.params, name)), step)
            widget.valueChanged.connect(lambda *_args, field=name: self._apply_readout_cooling_editor_change(field))
            self.readout_editor_inputs[name] = widget
            cooling_form.addRow(label, widget)
        enabled_widget = self._checkbox(
            "Enabled",
            self.params.cryocooler_enabled,
            lambda *_args: self._apply_readout_cooling_editor_change("cryocooler_enabled"),
        )
        self.readout_editor_inputs["cryocooler_enabled"] = enabled_widget
        cooling_form.addRow(enabled_widget)
        layout.addWidget(self.readout_cooling_editor)
        layout.addStretch(1)

    def _add_enabled_io_controls(self, form: Any) -> None:
        box, layout = self._section("Enabled Simulation I/O")
        button_row = self.QtWidgets.QHBoxLayout()
        enable_all = self.QtWidgets.QPushButton("Enable All")
        enable_all.clicked.connect(self._enable_all_simulation_io)
        disable_all = self.QtWidgets.QPushButton("Disable All")
        disable_all.clicked.connect(self._disable_all_simulation_io)
        button_row.addWidget(enable_all)
        button_row.addWidget(disable_all)
        layout.addRow(button_row)
        self.enabled_io_table = self.QtWidgets.QTableWidget(0, 3)
        self.enabled_io_table.setHorizontalHeaderLabels(["cell/node", "heater", "sensor"])
        self.enabled_io_table.verticalHeader().setVisible(False)
        self.enabled_io_table.setEditTriggers(self.QtWidgets.QAbstractItemView.NoEditTriggers)
        self.enabled_io_table.setSelectionBehavior(self.QtWidgets.QAbstractItemView.SelectRows)
        self.enabled_io_table.setMaximumHeight(170)
        self.enabled_io_table.itemChanged.connect(self._handle_enabled_io_item_changed)
        layout.addRow(self.enabled_io_table)
        form.addRow(box)

    def _add_component_temperature_controls(self, form: Any) -> None:
        self.component_combo = self.QtWidgets.QComboBox()
        self.component_temperature = self._double_spin(0.0, 1.0e6, 293.15, 1.0)
        apply_component = self.QtWidgets.QPushButton("Apply To Component")
        apply_component.clicked.connect(self.apply_component_initial_temperature)
        form.addRow("component", self.component_combo)
        form.addRow("initial K", self.component_temperature)
        form.addRow(apply_component)

    def _add_sys_id_controls(self, form: Any) -> None:
        box, sysid_form = self._section("Simulation Sys ID for Controller Gain Matrix")
        self.sys_id_matrix_combo = self.QtWidgets.QComboBox()
        self.sys_id_matrix_combo.currentIndexChanged.connect(self._handle_sys_id_matrix_selection)
        refresh_matrix_list = self.QtWidgets.QPushButton("Refresh Matrices")
        refresh_matrix_list.clicked.connect(lambda: self._refresh_sys_id_matrix_list())
        matrix_row = self.QtWidgets.QHBoxLayout()
        matrix_row.addWidget(self.sys_id_matrix_combo, 1)
        matrix_row.addWidget(refresh_matrix_list)
        sysid_form.addRow("active G matrix", matrix_row)
        self.sys_id_step_power = self._double_spin(0.0, 1.0e9, 1.0, 0.1)
        self.sys_id_global_temperature_K = self._double_spin(0.0, 1.0e6, 293.15, 1.0)
        self.sys_id_duration_s = self._double_spin(0.0, 1.0e9, 300.0, 10.0)
        self.sys_id_baseline_window_s = self._double_spin(0.0, 1.0e9, 10.0, 1.0)
        self.sys_id_final_window_s = self._double_spin(0.0, 1.0e9, 10.0, 1.0)
        self.sys_id_restore_between_tests = self._checkbox("Restore baseline between heater tests", True)
        self.sys_id_keep_cryocooler_active = self._checkbox("Keep cryocooler active during sys ID", True)
        self.sys_id_uniform_baseline = self._checkbox("Start from uniform baseline temperature", True)
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
        button_row = self.QtWidgets.QHBoxLayout()
        self.run_sys_id_button = self.QtWidgets.QPushButton("Run G_ctrl Sys ID")
        self.run_sys_id_button.clicked.connect(self.run_simulation_sys_id_for_G_ctrl)
        self.cancel_sys_id_button = self.QtWidgets.QPushButton("Cancel Sys ID")
        self.cancel_sys_id_button.clicked.connect(self.cancel_sys_id)
        self.cancel_sys_id_button.setEnabled(False)
        button_row.addWidget(self.run_sys_id_button)
        button_row.addWidget(self.cancel_sys_id_button)
        sysid_form.addRow(button_row)
        self.sys_id_progress_label = self.QtWidgets.QLabel("Idle.")
        self.sys_id_progress_label.setWordWrap(True)
        self.sys_id_status_label = self.QtWidgets.QLabel("")
        self.sys_id_status_label.setWordWrap(True)
        sysid_form.addRow("progress", self.sys_id_progress_label)
        sysid_form.addRow(self.sys_id_status_label)
        form.addRow(box)

    def refresh_graph_list(self) -> None:
        self.graph_combo.clear()
        root = Path.cwd() / "graphs"
        if not root.exists():
            return
        self.graph_combo.addItems([path.name for path in sorted(root.iterdir()) if (path / "graph.json").exists()])

    def _refresh_sys_id_matrix_list(self, select_path: Path | str | None = None) -> None:
        if not hasattr(self, "sys_id_matrix_combo"):
            return
        if isinstance(select_path, bool):
            select_path = None
        selected = Path(select_path) if select_path is not None else self._selected_sys_id_matrix_path()
        infos = list_sys_id_gain_matrices(self.folder)
        self._refreshing_sys_id_matrix_combo = True
        try:
            self.sys_id_matrix_combo.clear()
            self.sys_id_matrix_combo.addItem("Embedded graph matrix", None)
            target_index = 0
            for info in infos:
                label = info.name
                if info.created_at:
                    label = f"{info.name} ({info.created_at})"
                self.sys_id_matrix_combo.addItem(label, str(info.path))
                if selected is not None and info.path == selected:
                    target_index = self.sys_id_matrix_combo.count() - 1
            self.sys_id_matrix_combo.setCurrentIndex(target_index)
        finally:
            self._refreshing_sys_id_matrix_combo = False

    def _selected_sys_id_matrix_path(self) -> Path | None:
        if not hasattr(self, "sys_id_matrix_combo"):
            return None
        data = self.sys_id_matrix_combo.currentData()
        if not data:
            return None
        return Path(str(data))

    def _handle_sys_id_matrix_selection(self, *_: Any) -> None:
        if self._refreshing_sys_id_matrix_combo:
            return
        run_path = self._selected_sys_id_matrix_path()
        if run_path is None:
            return
        if self.model is None:
            self._status("Load a graph before selecting a saved G matrix.", True)
            return
        try:
            self.model.controller_gain_matrix = load_sys_id_gain_matrix(run_path)
            self.model.prune_controller_gain_matrix()
            if self.prepared is not None:
                self.prepared.mark_controller_stale()
            if self.on_controller_gain_matrix_changed is not None:
                self.on_controller_gain_matrix_changed()
            self._refresh_stats()
            self._refresh_sensor_readouts()
            self._status(f"Using saved G matrix '{run_path.name}'.")
        except Exception as exc:
            self._status(f"Could not load saved G matrix: {exc}", True)

    def _tagged_heater_node_ids(self) -> set[int]:
        if self.model is None:
            return set()
        return {int(node_id) for node_id, node in self.model.nodes.items() if node.is_heater}

    def _tagged_sensor_node_ids(self) -> set[int]:
        if self.model is None:
            return set()
        return {int(node_id) for node_id, node in self.model.nodes.items() if node.is_sensor}

    def _reset_enabled_io_from_params(self) -> None:
        self._enabled_io_initialized = False
        self._sync_enabled_io_table(use_saved_params=True)

    def _sync_enabled_io_table(self, *, use_saved_params: bool = False) -> None:
        if not hasattr(self, "enabled_io_table"):
            return
        tagged_heaters = self._tagged_heater_node_ids()
        tagged_sensors = self._tagged_sensor_node_ids()
        if use_saved_params or not self._enabled_io_initialized:
            self.enabled_heater_node_ids = (
                set(tagged_heaters)
                if self.params.enabled_heater_node_ids is None
                else {int(node_id) for node_id in self.params.enabled_heater_node_ids} & tagged_heaters
            )
            self.enabled_sensor_node_ids = (
                set(tagged_sensors)
                if self.params.enabled_sensor_node_ids is None
                else {int(node_id) for node_id in self.params.enabled_sensor_node_ids} & tagged_sensors
            )
            self._enabled_io_initialized = True
        else:
            self.enabled_heater_node_ids &= tagged_heaters
            self.enabled_sensor_node_ids &= tagged_sensors
            self.enabled_heater_node_ids |= tagged_heaters - self._known_heater_node_ids
            self.enabled_sensor_node_ids |= tagged_sensors - self._known_sensor_node_ids
        self._known_heater_node_ids = set(tagged_heaters)
        self._known_sensor_node_ids = set(tagged_sensors)
        nodes = sorted(tagged_heaters | tagged_sensors)
        self._syncing_enabled_io_table = True
        try:
            self.enabled_io_table.setRowCount(len(nodes))
            for row, node_id in enumerate(nodes):
                id_item = self.QtWidgets.QTableWidgetItem(str(node_id))
                id_item.setData(self.QtCore.Qt.UserRole, int(node_id))
                id_item.setFlags(self.QtCore.Qt.ItemIsEnabled | self.QtCore.Qt.ItemIsSelectable)
                self.enabled_io_table.setItem(row, 0, id_item)
                self.enabled_io_table.setItem(
                    row,
                    1,
                    self._enabled_io_checkbox_item(
                        "heater",
                        node_id,
                        node_id in tagged_heaters,
                        node_id in self.enabled_heater_node_ids,
                    ),
                )
                self.enabled_io_table.setItem(
                    row,
                    2,
                    self._enabled_io_checkbox_item(
                        "sensor",
                        node_id,
                        node_id in tagged_sensors,
                        node_id in self.enabled_sensor_node_ids,
                    ),
                )
            self.enabled_io_table.resizeColumnsToContents()
        finally:
            self._syncing_enabled_io_table = False
        self._apply_enabled_io_to_params(save=False)

    def _enabled_io_checkbox_item(self, role: str, node_id: int, available: bool, checked: bool) -> Any:
        item = self.QtWidgets.QTableWidgetItem("")
        item.setData(self.QtCore.Qt.UserRole, (role, int(node_id)))
        if not available:
            item.setFlags(
                self.QtCore.Qt.ItemFlag.NoItemFlags
                if hasattr(self.QtCore.Qt, "ItemFlag")
                else self.QtCore.Qt.NoItemFlags
            )
            return item
        item.setFlags(
            self.QtCore.Qt.ItemIsEnabled
            | self.QtCore.Qt.ItemIsSelectable
            | self.QtCore.Qt.ItemIsUserCheckable
        )
        item.setCheckState(self._qt_checked_state() if checked else self._qt_unchecked_state())
        return item

    def _qt_checked_state(self) -> Any:
        return (
            self.QtCore.Qt.CheckState.Checked
            if hasattr(self.QtCore.Qt, "CheckState")
            else self.QtCore.Qt.Checked
        )

    def _qt_unchecked_state(self) -> Any:
        return (
            self.QtCore.Qt.CheckState.Unchecked
            if hasattr(self.QtCore.Qt, "CheckState")
            else self.QtCore.Qt.Unchecked
        )

    def _handle_enabled_io_item_changed(self, item: Any) -> None:
        if self._syncing_enabled_io_table:
            return
        payload = item.data(self.QtCore.Qt.UserRole)
        if not isinstance(payload, tuple) or len(payload) != 2:
            return
        role, node_id = payload
        checked = item.checkState() == self._qt_checked_state()
        if role == "heater":
            if checked:
                self.enabled_heater_node_ids.add(int(node_id))
            else:
                self.enabled_heater_node_ids.discard(int(node_id))
        elif role == "sensor":
            if checked:
                self.enabled_sensor_node_ids.add(int(node_id))
            else:
                self.enabled_sensor_node_ids.discard(int(node_id))
        else:
            return
        self._apply_enabled_io_to_params(save=True)

    def _enable_all_simulation_io(self) -> None:
        self.enabled_heater_node_ids = self._tagged_heater_node_ids()
        self.enabled_sensor_node_ids = self._tagged_sensor_node_ids()
        self._enabled_io_initialized = True
        self._sync_enabled_io_table()
        self._apply_enabled_io_to_params(save=True)

    def _disable_all_simulation_io(self) -> None:
        self.enabled_heater_node_ids = set()
        self.enabled_sensor_node_ids = set()
        self._enabled_io_initialized = True
        self._sync_enabled_io_table()
        self._apply_enabled_io_to_params(save=True)

    def _apply_enabled_io_to_params(self, *, save: bool) -> None:
        previous_heaters = tuple(sorted(int(node_id) for node_id in (self.params.enabled_heater_node_ids or ())))
        previous_sensors = tuple(sorted(int(node_id) for node_id in (self.params.enabled_sensor_node_ids or ())))
        self.params = self._read_params()
        changed = (
            previous_heaters != tuple(sorted(int(node_id) for node_id in (self.params.enabled_heater_node_ids or ())))
            or previous_sensors != tuple(sorted(int(node_id) for node_id in (self.params.enabled_sensor_node_ids or ())))
        )
        if save:
            self._save_params_to_folder()
        if self.prepared is not None:
            self.prepared.params = self.params
            if changed:
                self.prepared.mark_controller_stale()
                self.prepared.reset_controller_integrators()
        self._refresh_stats()
        self._refresh_sensor_readouts()

    def _heater_enabled_for_simulation(self, node_id: int) -> bool:
        return int(node_id) in self.enabled_heater_node_ids

    def _sensor_enabled_for_simulation(self, node_id: int) -> bool:
        return int(node_id) in self.enabled_sensor_node_ids

    def use_current_graph(self) -> None:
        if self._simulation_worker_active():
            self.pause()
            self._status("Simulation worker is stopping; load the current graph after the current compute step finishes.")
            return
        self.model = self.current_model()
        self.folder = self.current_folder()
        self.matrices = build_matrices(self.model)
        self._load_params_from_folder()
        self._reset_enabled_io_from_params()
        self._refresh_sys_id_matrix_list()
        self._sync_component_options()
        self._reset_to_model_initial_temperatures()
        self._simulation_reinitialize_pending = False
        self._draw_current(reset_camera=True)
        self._refresh_sensor_readouts()
        self._status("Using current editor graph.")

    def load_selected_graph(self) -> None:
        if self._simulation_worker_active():
            self.pause()
            self._status("Simulation worker is stopping; load the graph after the current compute step finishes.")
            return
        name = self.graph_combo.currentText()
        if not name:
            self._status("No graph selected.", True)
            return
        try:
            self.folder = Path.cwd() / "graphs" / name
            log_event("simulation load_selected_graph start", folder=str(self.folder))
            self.model, self.matrices = load_graph_folder(self.folder)
            log_event(
                "simulation load_selected_graph loaded folder",
                nodes=len(self.model.nodes),
                edges=len(self.model.edges),
                matrix_keys=sorted(self.matrices),
            )
            self._load_params_from_folder()
            self._reset_enabled_io_from_params()
            self._refresh_sys_id_matrix_list()
            self._sync_component_options()
            self._reset_to_model_initial_temperatures()
            self._simulation_reinitialize_pending = False
            log_event("simulation load_selected_graph before draw")
            self._draw_current(reset_camera=True)
            log_event("simulation load_selected_graph after draw")
            self._refresh_sensor_readouts()
            self._status(f"Loaded simulation graph {name}.")
        except Exception as exc:
            log_exception("simulation load_selected_graph failed", exc)
            self._status(str(exc), True)

    def initialize_simulation(self) -> None:
        if self._simulation_worker_active():
            self.pause()
            self._status("Simulation worker is stopping; initialize after the current compute step finishes.")
            return
        if self.model is None:
            self.use_current_graph()
        if self.model is None:
            return
        try:
            self._refresh_matrices_for_run()
            self.params = self._read_params()
            self._save_params_to_folder()
            self.prepared = prepare_simulation(self.model, self.matrices, self.params)
            self.temperature_by_node = {
                int(node_id): float(temp)
                for node_id, temp in zip(self.prepared.node_ids, self.prepared.temperatures_K)
            }
            self._reset_time_slider()
            self._refresh_initialized_view()
            self._refresh_stats()
            self._set_warnings(self.prepared.warnings)
            self._simulation_reinitialize_pending = False
        except Exception as exc:
            self._status(str(exc), True)

    def _set_all_initial_temperatures(self) -> None:
        """Set initial_temperature_K on EVERY component to the spin-box value.

        Updates the loaded graph (the source of truth for the next Initialize) and,
        if a simulation is already prepared and idle, applies it live by resetting
        to the new uniform initial state."""
        value = float(self.initial_temperature_all_spin.value())
        if self.model is None:
            self.use_current_graph()
        if self.model is None:
            self._status("Load a graph before setting initial temperatures.", True)
            return
        # Stop any running (or blowing-up) simulation first. Otherwise the worker
        # owns the live state and the new temperature would be silently ignored,
        # leaving the run at the old temperature when the controller next steps.
        worker_was_active = self._simulation_worker_active()
        if worker_was_active:
            self.pause()
        for node in self.model.nodes.values():
            node.initial_temperature_K = value
        count = len(self.model.nodes)
        # Keep any cached initial-temperature vector on the matrices consistent so a
        # later prepare/reset agrees with the model.
        if isinstance(self.matrices, dict) and "initial_temperature_K" in self.matrices:
            shape = np.asarray(self.matrices["initial_temperature_K"], dtype=float).shape
            self.matrices["initial_temperature_K"] = np.full(shape, value, dtype=float)
        applied_live = False
        if self.prepared is not None and not worker_was_active:
            # Idle prepared sim: reset its live state in place (cheap and exact),
            # so the current state is authoritative at the set temperature.
            self.prepared.initial_temperatures_K[:] = value
            self.prepared.reset()
            self.temperature_by_node = {int(node_id): value for node_id in self.prepared.node_ids}
            self._reset_time_slider()
            self._refresh_initialized_view()
            self._refresh_stats()
            applied_live = True
        else:
            # No prepared sim yet, or we just stopped a worker (don't race the
            # still-finishing thread by mutating its state). Force a clean
            # re-initialize from the model on the next run so the simulation is
            # GUARANTEED to start at the set temperature before the controller
            # takes its first step.
            self._simulation_reinitialize_pending = True
        if worker_was_active:
            suffix = " (stopped the run; it re-initializes to this on the next Play)"
        elif applied_live:
            suffix = " (reset the prepared simulation to this state)"
        else:
            suffix = " (initializes to this on the next Play)"
        self._status(f"Set initial temperature of all {count} components to {value:g} K{suffix}.")

    def _randomize_sensor_setpoints(self) -> None:
        """Assign each sensor a random controller setpoint (desired temperature) =
        center +/- a uniform mK-scale offset. Testing helper; applies live because
        the controller reads controller_setpoint_K each step (no re-init needed)."""
        center = float(self.sensor_random_center_spin.value())
        spread_K = float(self.sensor_random_spread_mK_spin.value()) * 1.0e-3
        if self.model is None:
            self.use_current_graph()
        if self.model is None:
            self._status("Load a graph before randomizing sensor setpoints.", True)
            return
        rng = np.random.default_rng()
        sensors = [node for node in self.model.nodes.values() if bool(getattr(node, "is_sensor", False))]
        for sensor in sensors:
            offset = float(rng.uniform(-spread_K, spread_K)) if spread_K > 0.0 else 0.0
            sensor.controller_setpoint_K = float(center + offset)
        if self.prepared is not None:
            self.prepared.mark_controller_stale()
        self._status(
            f"Randomized {len(sensors)} sensor setpoint(s) around {center:g} K "
            f"+/- {spread_K * 1.0e3:g} mK (applied live)."
        )

    def play(self) -> None:
        if not self._simulation_worker_active():
            # Guard the live path: large graphs are slow to visualize, and a
            # modal-LQR run without its controller artifact should be confirmed.
            if not self._large_graph_viz_warn_ok():
                return
            if not self._confirm_controller_ok():
                return
            self._apply_pending_runtime_changes()
        if self.prepared is None or self._simulation_reinitialize_pending:
            self.initialize_simulation()
        if self.prepared is None:
            self.pause()
            self._status("Simulation did not initialize; playback was not started.", True)
            return
        self.timer.start(self._playback_timer_interval_ms())
        self.step_forward()

    # -- large-graph / controller guards + headless runs -------------------- #
    def _large_graph_viz_warn_ok(self) -> bool:
        warn_nodes = 250_000
        n = len(self.model.nodes) if self.model is not None else 0
        if n <= warn_nodes or getattr(self, "_large_graph_viz_ack", False):
            return True
        reply = self.QtWidgets.QMessageBox.question(
            self.widget,
            "Large graph",
            f"This graph has {n:,} nodes. Live visualization will be slow and memory-heavy.\n\n"
            "For large / overnight runs use 'Run Headless (save, no viz)' instead.\n\n"
            "Play WITH visualization anyway?",
            self.QtWidgets.QMessageBox.Yes | self.QtWidgets.QMessageBox.No,
            self.QtWidgets.QMessageBox.No,
        )
        if reply != self.QtWidgets.QMessageBox.Yes:
            return False
        self._large_graph_viz_ack = True  # don't nag again this session
        return True

    def _controller_selected_scheme(self) -> str:
        labels_to_key = {v: k for k, v in getattr(self, "_controller_scheme_labels", {}).items()}
        return labels_to_key.get(self.controller_scheme_combo.currentText(), "pid_qp")

    def _modal_controller_path(self) -> Path | None:
        explicit = str(getattr(self.params, "modal_controller_path", "") or "")
        if explicit and Path(explicit).exists():
            return Path(explicit)
        if self.folder is not None and (self.folder / "modal_controller.npz").exists():
            return self.folder / "modal_controller.npz"
        return None

    def _confirm_controller_ok(self) -> bool:
        if self.input_mode.currentText() != "heater_inputs":
            return True  # controller isn't used in this input mode
        if self._controller_selected_scheme() != "modal_lqr":
            return True  # PID+QP needs no artifact
        if self._modal_controller_path() is not None:
            return True
        reply = self.QtWidgets.QMessageBox.question(
            self.widget,
            "No controller selected",
            "Modal LQR is selected but no 'modal_controller.npz' was found for this graph.\n\n"
            "The run will fall back to the PID+QP controller. Run anyway?",
            self.QtWidgets.QMessageBox.Yes | self.QtWidgets.QMessageBox.No,
            self.QtWidgets.QMessageBox.No,
        )
        return reply == self.QtWidgets.QMessageBox.Yes

    def run_headless_simulation(self) -> None:
        """Launch the full closed-loop run with no visualization, saving everything
        to simulations/<graph>/<timestamp>/. Runs in a background thread; the window
        stays responsive and a poll timer re-enables the button when it finishes."""
        if getattr(self, "_headless_thread", None) is not None and self._headless_thread.is_alive():
            self._status("A headless run is already in progress.", True)
            return
        if self.model is None or self.folder is None:
            self._status("Load a graph before running.", True)
            return
        if not self._confirm_controller_ok():
            return
        self.params = self._read_params()
        setpoints = {
            int(nid): float(getattr(node, "controller_setpoint_K", 293.15))
            for nid, node in self.model.nodes.items()
            if getattr(node, "is_sensor", False)
        }
        # Capture the CURRENT in-memory starting temperatures so the headless run
        # begins from what the user is looking at (e.g. "Set all -> 50 K"). The
        # runner reloads the graph folder from disk, which would otherwise revert to
        # the saved-on-disk temps; passing the state explicitly keeps them in sync.
        node_ids = np.asarray(self.model.ordered_node_ids(), dtype=int)
        init_temps = np.array(
            [float(getattr(self.model.nodes[int(nid)], "initial_temperature_K", 293.15)) for nid in node_ids],
            dtype=float,
        )
        ctrl = self._modal_controller_path()
        cfg = RunConfig(
            graph_folder=str(self.folder),
            controller_path=str(ctrl) if ctrl is not None else None,
            allow_no_controller=ctrl is None,
            setpoints_K=setpoints,
            dt_s=float(self.params.dt_s),
            t_final_s=float(self.params.t_final_s),
            gpu_solver_enabled=bool(self.params.gpu_solver_enabled),
            # Forward the FULL parameter set so the headless run uses the same physics
            # (tdep properties, radiation, cryocooler, substeps, ...) as the GUI --
            # not the runner's minimal defaults.
            params=self.params,
        )
        self._headless_cancel = threading.Event()
        self._headless_runner = SimulationRunner(
            cfg, cancel_event=self._headless_cancel, initial_state=(node_ids, init_temps)
        )
        out_dir = self._headless_runner.out_dir

        def _target() -> None:
            try:
                self._headless_runner.run()
                log_event("headless simulation complete", out=str(out_dir))
            except Exception as exc:  # noqa: BLE001
                log_exception("headless simulation failed", exc)

        self._headless_thread = threading.Thread(target=_target, name="headless-sim", daemon=True)
        self._headless_thread.start()
        self.run_headless_button.setEnabled(False)
        self.stop_headless_button.setEnabled(True)
        self._status(f"Headless run started -> {out_dir}  (tail status.json for progress)", False)
        if getattr(self, "_headless_poll_timer", None) is None:
            self._headless_poll_timer = self.QtCore.QTimer(self.widget)
            self._headless_poll_timer.timeout.connect(self._poll_headless_status)
        self._headless_poll_timer.start(1000)

    def stop_headless_simulation(self) -> None:
        cancel = getattr(self, "_headless_cancel", None)
        if cancel is not None:
            cancel.set()
            self._status("Stopping headless run (finalizing artifacts)...", False)

    def _poll_headless_status(self) -> None:
        thread = getattr(self, "_headless_thread", None)
        runner = getattr(self, "_headless_runner", None)
        if thread is None or not thread.is_alive():
            if getattr(self, "_headless_poll_timer", None) is not None:
                self._headless_poll_timer.stop()
            self.run_headless_button.setEnabled(True)
            self.stop_headless_button.setEnabled(False)
            if runner is not None:
                self._status(f"Headless run finished: {runner._exit_status}  ({runner.out_dir})", False)
            return
        if runner is not None:
            try:
                import json as _json

                data = _json.loads(runner.status_path.read_text(encoding="utf-8"))
                self._status(
                    f"Headless: t={data.get('sim_time_s', 0):.0f}/{data.get('t_final_s', 0):.0f}s "
                    f"({100 * data.get('progress', 0):.0f}%) step={data.get('step', 0)}",
                    False,
                )
            except Exception:
                pass

    def _refresh_matrices_for_run(self) -> None:
        if self.model is None:
            return
        if self._can_reuse_loaded_octree_matrices_for_run():
            self.matrices = self._runtime_matrices_from_loaded_octree()
            log_event(
                "simulation refresh_matrices_for_run reused loaded octree matrices",
                nodes=len(self.model.nodes),
                matrix_keys=sorted(self.matrices),
                L_type=type(self.matrices.get("L")).__name__ if "L" in self.matrices else None,
            )
            return
        if (
            EdgeMode.normalize(self.model.metadata.edge_mode) == EdgeMode.AUTO.value
            and all(node.center_mm is not None and node.size_mm is not None for node in self.model.nodes.values())
            and not has_generated_role_contact_edges(self.model)
        ):
            refresh_geometry_edges(self.model)
            refresh_radiation_from_exposed_faces(self.model)
        self.matrices = build_matrices(self.model)

    def _can_reuse_loaded_octree_matrices_for_run(self) -> bool:
        if self.model is None or not self.model.octree_graph_data:
            return False
        if not isinstance(self.matrices, dict) or "L" not in self.matrices:
            return False
        try:
            matrix_node_ids = np.asarray(self.matrices.get("node_ids"), dtype=int).reshape(-1)
        except Exception:
            return False
        expected_node_ids = np.asarray(self.model.ordered_node_ids(), dtype=int)
        if matrix_node_ids.shape != expected_node_ids.shape:
            return False
        if not np.array_equal(matrix_node_ids, expected_node_ids):
            return False
        L = self.matrices.get("L")
        return bool(getattr(L, "shape", None) == (len(expected_node_ids), len(expected_node_ids)))

    def _runtime_matrices_from_loaded_octree(self) -> dict[str, Any]:
        assert self.model is not None
        node_ids = np.asarray(self.matrices["node_ids"], dtype=int).reshape(-1)
        matrices = dict(self.matrices)
        matrices["node_ids"] = node_ids
        matrices["coords"] = np.array(
            [self.model.nodes[int(node_id)].coord for node_id in node_ids],
            dtype=int,
        )
        matrices["C"] = np.array(
            [float(self.model.nodes[int(node_id)].C_J_K) for node_id in node_ids],
            dtype=float,
        )
        matrices["Grad"] = np.array(
            [float(self.model.nodes[int(node_id)].Grad_W_K) for node_id in node_ids],
            dtype=float,
        )
        matrices["G_rad"] = np.array(
            [
                float(self.model.nodes[int(node_id)].G_rad_W_K)
                if float(self.model.nodes[int(node_id)].G_rad_W_K) > 0.0
                else float(self.model.nodes[int(node_id)].Grad_W_K)
                for node_id in node_ids
            ],
            dtype=float,
        )
        matrices["initial_temperature_K"] = np.array(
            [float(self.model.nodes[int(node_id)].initial_temperature_K) for node_id in node_ids],
            dtype=float,
        )
        return matrices

    def pause(self) -> None:
        self.timer.stop()
        self._cancel_simulation_worker()

    def shutdown(self) -> None:
        self.timer.stop()
        diagnostic_timer = getattr(self, "stepper_diagnostic_timer", None)
        if diagnostic_timer is not None:
            diagnostic_timer.stop()
        parameter_save_timer = getattr(self, "parameter_save_timer", None)
        if parameter_save_timer is not None:
            parameter_save_timer.stop()
            self._flush_deferred_parameter_save()
        self._cancel_simulation_worker()
        executor = getattr(self, "simulation_executor", None)
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
        try:
            self.viewer.close()
        except Exception:
            pass

    def reset(self) -> None:
        if self._simulation_worker_active():
            self.pause()
            self._status("Simulation worker is stopping; reset after the current compute step finishes.")
            return
        self._apply_pending_runtime_changes()
        if self.prepared is None or self._simulation_reinitialize_pending:
            self.initialize_simulation()
            return
        self.timer.stop()
        self.prepared.reset()
        self._after_state_change()
        self._status("Simulation reset to initial temperatures.")

    def step_forward(self) -> None:
        if not self._simulation_worker_active():
            self._apply_pending_runtime_changes()
        if self.prepared is None or self._simulation_reinitialize_pending:
            self.initialize_simulation()
            return
        if self._simulation_worker_active():
            return
        mode = "play" if self.timer.isActive() else "step"
        self._start_simulation_worker(mode=mode, steps=self._playback_steps_per_tick() if mode == "play" else 1)

    def _playback_target_step_interval_ms(self) -> float:
        return 100.0 / max(float(self.params.playback_speed), 1.0e-9)

    def _playback_timer_interval_ms(self) -> int:
        target_step_interval = self._playback_target_step_interval_ms()
        display_interval = max(10.0, float(getattr(self.params, "display_update_interval_ms", 100.0)))
        return max(10, int(round(max(target_step_interval, display_interval))))

    def _playback_steps_per_tick(self) -> int:
        interval = float(self._playback_timer_interval_ms())
        target_step_interval = max(1.0e-9, self._playback_target_step_interval_ms())
        return max(1, int(round(interval / target_step_interval)))

    def step_backward(self) -> None:
        if self._simulation_worker_active():
            self.pause()
            self._status("Simulation worker is stopping; step backward after the current compute step finishes.")
            return
        if self.prepared is None:
            return
        if self._simulation_reinitialize_pending:
            self.initialize_simulation()
            return
        self.prepared.step_backward()
        self._after_state_change()

    def run_stepper_diagnostic(self) -> None:
        if self._simulation_worker_active():
            self.pause()
            self._status("Simulation worker is stopping; run the solver diagnostic after the current compute step finishes.")
            return
        if self._stepper_diagnostic_worker_active():
            self._status("Solver diagnostic is already running.", True)
            return
        if self.sys_id_state is not None:
            self._status("Finish or cancel G_ctrl sys ID before running the solver diagnostic.", True)
            return
        self.pause()
        if self.model is None or self.prepared is None:
            self._status("Initialize or run a simulation before running the solver diagnostic.", True)
            return
        current_time_s = float(self.prepared.time_s)
        if current_time_s <= 0.0:
            self._status("Advance the simulation before running the solver diagnostic.", True)
            return
        try:
            params = self.prepared.params
            output_dir = self._stepper_diagnostic_output_dir() if self.stepper_diagnostic_save.isChecked() else None
            current_profile = dict(getattr(self.prepared, "last_step_profile_ms", {}) or {})
            self.stepper_diagnostic_status_label.setText(
                f"Running current-state comparison at t = {current_time_s:.6g} s."
            )
            self.stepper_diagnostic_target_label.setText(
                f"Current simulation time: {current_time_s:.6g} s."
            )
            self.stepper_diagnostic_button.setEnabled(False)
            worker_args = (
                self.model,
                dict(self.matrices),
                params,
                np.asarray(self.prepared.node_ids, dtype=int).copy(),
                np.asarray(self.prepared.initial_temperatures_K, dtype=float).copy(),
                np.asarray(self.prepared.temperatures_K, dtype=float).copy(),
                current_time_s,
                _last_prepared_solver_name(self.prepared),
                float(current_profile.get("total_ms", 0.0)) / 1000.0,
                current_profile,
                output_dir,
            )
            executor = getattr(self, "simulation_executor", None)
            if executor is None:
                result = _run_stepper_diagnostic_worker(*worker_args)
                self._apply_stepper_diagnostic_result(result)
                return
            self.stepper_diagnostic_future = executor.submit(_run_stepper_diagnostic_worker, *worker_args)
            self.stepper_diagnostic_timer.start(50)
            self._status("Solver diagnostic running in background.")
        except Exception as exc:
            self.stepper_diagnostic_button.setEnabled(True)
            self.stepper_diagnostic_status_label.setText(f"Failed: {exc}")
            self._status(f"Solver diagnostic failed: {exc}", True)
            log_exception("solver diagnostic failed", exc)

    def _build_modal_design_controls(self, form: Any) -> None:
        box, design_form = self._section("Modal LQR Design")
        self.modal_temp_spin = self._double_spin(0.0, 1.0e6, self._default_modal_operating_temperature_K(), 1.0)
        self.modal_temp_spin.setToolTip(
            "Operating temperature to linearize the plant about. The controller offsets its "
            "measurements and setpoints from this."
        )
        design_form.addRow("operating T K", self.modal_temp_spin)
        self.modal_modes_spin = self._int_spin(2, 100000, 120, 1)
        self.modal_modes_spin.setToolTip(
            "Number of slowest thermal modes solved in stage 1 (before balanced truncation). "
            "Clamped to fit the graph."
        )
        design_form.addRow("slow modes", self.modal_modes_spin)
        self.modal_order_spin = self._int_spin(1, 100000, 40, 1)
        self.modal_order_spin.setToolTip(
            "Reduced model order r after balanced truncation -- the controller's state dimension "
            "(kept small so it runs on the microcontroller)."
        )
        design_form.addRow("reduced order r", self.modal_order_spin)
        self.modal_effort_spin = self._double_spin(1.0e-9, 1.0e9, 1.0, 0.1)
        self.modal_effort_spin.setToolTip(
            "LQR control-effort weight rho (R = rho*I, Q = C^T C). Larger rho = gentler, less "
            "aggressive heating; smaller = faster, higher-power response."
        )
        design_form.addRow("LQR effort weight", self.modal_effort_spin)
        self.modal_integral_spin = self._double_spin(0.0, 1.0e9, float(getattr(self.params, "modal_integral_gain", 0.0)), 0.01)
        self.modal_integral_spin.setToolTip(
            "Offset-free integral gain the modal controller uses to supply the operating holding "
            "power the linearized model omits."
        )
        design_form.addRow("integral gain", self.modal_integral_spin)
        # Adaptive (learning) feedforward: online RLS correction of the exact-DC-gain
        # feedforward from the integral's steady-state holding power. Off by default;
        # all of these hot-swap during a running sim (they only change controller
        # behavior, not the plant matrices).
        self.inputs["modal_adaptive_ff_enabled"] = self._checkbox(
            "Adaptive feedforward (RLS)",
            bool(getattr(self.params, "modal_adaptive_ff_enabled", False)),
            lambda *_: self._handle_parameter_change("modal_adaptive_ff_enabled"),
        )
        self.inputs["modal_adaptive_ff_enabled"].setToolTip(
            "Learn the DC-gain error the model got wrong: regress the integral's steady-state "
            "holding power against the setpoint (recursive least squares) and fold the correction "
            "into the feedforward, so revisited setpoints get the right holding power immediately "
            "instead of waiting for the integral. Bumpless; in-memory only (reset on re-prepare)."
        )
        design_form.addRow(self.inputs["modal_adaptive_ff_enabled"])
        self._add_double_parameter(
            design_form, "modal_adaptive_ff_forgetting", "adaptive forgetting", 0.5, 1.0, 0.001
        )
        self.inputs["modal_adaptive_ff_forgetting"].setToolTip(
            "RLS forgetting factor in (0, 1]. 1 = growing-window (exact, ever-more-confident); "
            "<1 lets a stale estimate fade for a slowly time-varying plant. Keep near 1."
        )
        self._add_double_parameter(
            design_form, "modal_adaptive_ff_error_tol_K", "adaptive error tol K", 0.0, 1.0e6, 0.01
        )
        self._add_double_parameter(
            design_form, "modal_adaptive_ff_rate_tol_K_per_s", "adaptive rate tol K/s", 0.0, 1.0e6, 1.0e-4
        )
        self.inputs["modal_adaptive_ff_error_tol_K"].setToolTip(
            "Steady-state gate: a learning sample is only taken when every controlled sensor's "
            "tracking error is below this AND its |dT/dt| is below the rate tolerance -- so "
            "transient or saturated data never corrupts the static-map regression."
        )
        self._add_double_parameter(
            design_form, "modal_adaptive_ff_max_correction_frac", "adaptive max corr frac", 0.0, 1.0e6, 0.1
        )
        self.inputs["modal_adaptive_ff_max_correction_frac"].setToolTip(
            "Projection guard: the learned feedforward correction is clamped, per heater, to this "
            "fraction of its max power (the effective command is clamped to [0, max] regardless)."
        )
        self.modal_design_button = self.QtWidgets.QPushButton("Build && Use Modal Controller")
        self.modal_design_button.setToolTip(
            "Reduce the CURRENT graph to a reduced-order LQR controller and load it into the "
            "modal-LQR scheme automatically (saved as modal_controller.npz in the graph folder). "
            "Runs in the background."
        )
        self.modal_design_button.clicked.connect(self.build_modal_controller)
        design_form.addRow(self.modal_design_button)
        self.modal_design_status_label = self.QtWidgets.QLabel("Idle.")
        self._pin_two_line_label(self.modal_design_status_label)
        design_form.addRow("status", self.modal_design_status_label)
        form.addRow(box)

    def _default_modal_operating_temperature_K(self) -> float:
        """Median initial temperature of the sensor cells (else all cells) as a
        sensible default operating point; falls back to room temperature."""
        if self.model is None:
            return 293.15
        sensor_temps = [
            float(node.initial_temperature_K)
            for node in self.model.nodes.values()
            if bool(getattr(node, "is_sensor", False))
        ]
        temps = sensor_temps or [float(node.initial_temperature_K) for node in self.model.nodes.values()]
        return float(np.median(temps)) if temps else 293.15

    def _modal_design_worker_active(self) -> bool:
        future = getattr(self, "modal_design_future", None)
        return future is not None and not future.done()

    def build_modal_controller(self) -> None:
        """Run the modal/balanced-truncation reduction + LQR design on the current
        graph in the background, then load the result into the modal-LQR scheme."""
        if self._simulation_worker_active():
            self.pause()
            self._status("Simulation worker is stopping; build the modal controller after the current step finishes.")
            return
        if self._modal_design_worker_active():
            self._status("Modal controller build is already running.", True)
            return
        if self.sys_id_state is not None:
            self._status("Finish or cancel G_ctrl sys ID before building the modal controller.", True)
            return
        if self.model is None:
            self._status("Load a graph before building the modal controller.", True)
            return
        try:
            if getattr(self, "matrices", None) is None:
                self.matrices = build_matrices(self.model)
            matrices = self.matrices
            if "L" not in matrices or "C" not in matrices or "node_ids" not in matrices:
                self._status("Loaded graph is missing the operator matrices needed for reduction.", True)
                return
            out_path = self._modal_controller_output_path()
            args = (
                np.asarray(matrices["C"], dtype=float).copy(),
                matrices["L"],
                np.asarray(matrices.get("G_rad", np.zeros(len(matrices["C"]))), dtype=float).copy(),
                np.asarray(matrices["node_ids"], dtype=int).copy(),
                self.model,
                float(self.modal_temp_spin.value()),
                int(self.modal_modes_spin.value()),
                int(self.modal_order_spin.value()),
                float(self.modal_effort_spin.value()),
                float(self.modal_integral_spin.value()),
                str(out_path),
                str(getattr(self.model.metadata, "graph_name", "") or ""),
                self._modal_design_progress,
            )
            self._modal_design_progress["message"] = "Starting reduction…"
            self.modal_design_status_label.setText("Starting reduction…")
            self.modal_design_button.setEnabled(False)
            executor = getattr(self, "simulation_executor", None)
            if executor is None:  # headless / no executor: run inline
                result = _run_modal_design_worker(*args)
                self._apply_modal_design_result(result)
                return
            self.modal_design_future = executor.submit(_run_modal_design_worker, *args)
            self.modal_design_timer.start(150)
            self._status("Modal controller build running in background.")
        except Exception as exc:  # noqa: BLE001 - surface to the panel
            self.modal_design_button.setEnabled(True)
            self.modal_design_status_label.setText(f"Failed: {exc}")
            self._status(f"Modal controller build failed: {exc}", True)
            log_exception("modal controller build failed", exc)

    def _modal_controller_output_path(self) -> Path:
        # Save next to the graph so _modal_controller_path_value auto-discovers it;
        # otherwise fall back to the scratch/session temp dir.
        if self.folder is not None:
            return self.folder / "modal_controller.npz"
        return Path(tempfile.gettempdir()) / "modal_controller.npz"

    def _poll_modal_design_worker(self) -> None:
        future = getattr(self, "modal_design_future", None)
        if future is None:
            self.modal_design_timer.stop()
            return
        # Reflect the worker's latest progress message while it runs.
        message = self._modal_design_progress.get("message", "")
        if message:
            self.modal_design_status_label.setText(message)
        if not future.done():
            return
        self.modal_design_timer.stop()
        self.modal_design_future = None
        self.modal_design_button.setEnabled(True)
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            self.modal_design_status_label.setText(f"Failed: {exc}")
            self._status(f"Modal controller build failed: {exc}", True)
            log_exception("modal controller build failed", exc)
            return
        self._apply_modal_design_result(result)

    def _apply_modal_design_result(self, result: Any) -> None:
        self.modal_design_button.setEnabled(True)
        self.modal_design_status_label.setText(result.summary())
        # Wire the freshly-built artifact into the modal-LQR scheme and switch to it.
        self.params = replace(
            self.params,
            mimo_controller_scheme="modal_lqr",
            modal_controller_path=result.path,
            modal_integral_gain=float(self.modal_integral_spin.value()),
        )
        combo = getattr(self, "controller_scheme_combo", None)
        if combo is not None:
            combo.blockSignals(True)
            combo.setCurrentText(self._controller_scheme_labels.get("modal_lqr", combo.currentText()))
            combo.blockSignals(False)
        # Rebuild the prepared simulation so the controller picks up the artifact.
        if self.model is not None:
            self.sync_from_editor(self.model, self.folder, reinitialize=self.prepared is not None)
        self._status(
            f"Modal LQR controller built and loaded (order r={result.reduced_order}). {result.summary()}"
        )

    def _stepper_diagnostic_output_dir(self) -> Path | None:
        if self.folder is None:
            return None
        name = "stepper_compare_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.folder / "simulations" / name

    def _stepper_diagnostic_worker_active(self) -> bool:
        future = getattr(self, "stepper_diagnostic_future", None)
        return future is not None and not future.done()

    def _poll_stepper_diagnostic_worker(self) -> None:
        future = getattr(self, "stepper_diagnostic_future", None)
        if future is None:
            self.stepper_diagnostic_timer.stop()
            return
        if not future.done():
            return
        self.stepper_diagnostic_timer.stop()
        self.stepper_diagnostic_future = None
        self.stepper_diagnostic_button.setEnabled(True)
        try:
            result = future.result()
        except Exception as exc:
            self.stepper_diagnostic_status_label.setText(f"Failed: {exc}")
            self._status(f"Solver diagnostic failed: {exc}", True)
            log_exception("solver diagnostic failed", exc)
            return
        self._apply_stepper_diagnostic_result(result)

    def _apply_stepper_diagnostic_result(self, result: dict[str, Any]) -> None:
        self.stepper_diagnostic_button.setEnabled(True)
        summary = _format_stepper_diagnostic_summary(result)
        self.stepper_diagnostic_status_label.setText(summary)
        metrics = result.get("metrics", {})
        if isinstance(metrics, dict):
            self._status(
                "Solver diagnostic complete: "
                f"max error = {float(metrics.get('max_abs_error_K', 0.0)):.4g} K, "
                f"RMSE = {float(metrics.get('rmse_K', 0.0)):.4g} K."
            )
        else:
            self._status("Solver diagnostic complete.")

    def _simulation_worker_active(self) -> bool:
        future = getattr(self, "simulation_future", None)
        return future is not None and not future.done()

    def _cancel_simulation_worker(self) -> None:
        event = getattr(self, "simulation_cancel_event", None)
        if event is not None:
            event.set()
        future = getattr(self, "simulation_future", None)
        if future is not None and not future.done():
            future.cancel()

    def _start_simulation_worker(self, *, mode: str, steps: int) -> None:
        if self.prepared is None or self._simulation_worker_active():
            return
        event = threading.Event()
        self.simulation_cancel_event = event
        self._simulation_worker_mode = str(mode)
        worker_args = (
            self.prepared,
            self.params,
            max(1, int(steps)),
            bool(self.params.loop_playback and mode == "play"),
            event,
            bool(self._live_step_profiling_enabled()),
        )
        executor = getattr(self, "simulation_executor", None)
        if executor is None:
            result = _run_simulation_worker_batch(*worker_args)
            self._apply_simulation_worker_result(result)
            return
        self.simulation_future = executor.submit(_run_simulation_worker_batch, *worker_args)
        self.simulation_worker_timer.start(20)
        if mode == "step":
            self._status("Simulation step running in background.")

    def _poll_simulation_worker(self) -> None:
        future = getattr(self, "simulation_future", None)
        if future is None:
            self.simulation_worker_timer.stop()
            return
        if not future.done():
            return
        self.simulation_worker_timer.stop()
        self.simulation_future = None
        self.simulation_cancel_event = None
        if future.cancelled():
            self._simulation_worker_mode = None
            self._apply_pending_runtime_changes()
            self._status("Simulation worker stopped.")
            return
        try:
            result = future.result()
        except Exception as exc:
            self.timer.stop()
            self._simulation_worker_mode = None
            self._status(f"Simulation worker failed: {exc}", True)
            log_exception("simulation worker failed", exc)
            return
        self._apply_simulation_worker_result(result)

    def _apply_simulation_worker_result(self, result: dict[str, Any]) -> None:
        if self.prepared is None:
            self._simulation_worker_mode = None
            return
        steps_completed = int(result.get("steps_completed", 0))
        mode = str(result.get("mode") or self._simulation_worker_mode or "")
        self._simulation_worker_mode = None
        if bool(result.get("cancelled", False)):
            self._apply_pending_runtime_changes()
            self._status("Simulation worker stopped.")
            return
        if steps_completed <= 0:
            if bool(result.get("done", False)):
                self.timer.stop()
            return
        profile = result.get("profile")
        profile = profile if isinstance(profile, dict) else None
        payload = result.get("readout") if isinstance(result.get("readout"), dict) else {}
        # Throttle the (main-thread, VTK) render/readout while playing so the Qt
        # event loop stays responsive between frames. Stepping/paused always
        # renders. Skipped frames still advance state cheaply.
        now = time.perf_counter()
        # Render throttle. While playing, require at least as much idle time as the
        # last render took (>=50% duty cycle) AND the configured display interval,
        # so the Qt event loop (parameter window, camera) stays responsive even
        # when a single 3D redraw is expensive. Stepping/paused always renders.
        render_floor_s = max(0.0, float(getattr(self.params, "display_update_interval_ms", 100.0))) / 1000.0
        min_gap_s = max(render_floor_s, float(getattr(self, "_last_render_duration_s", 0.0)))
        render_due = (not self.timer.isActive()) or (now - float(getattr(self, "_last_render_end_s", 0.0))) >= min_gap_s
        ui_start = time.perf_counter()
        if render_due:
            render_start = time.perf_counter()
            self._after_worker_state_change(profile, payload)
            self._last_render_end_s = time.perf_counter()
            self._last_render_duration_s = self._last_render_end_s - render_start
        else:
            temperature_by_node = payload.get("temperature_by_node") if isinstance(payload, dict) else None
            if temperature_by_node:
                self.temperature_by_node = temperature_by_node
            self._sync_time_slider_to_history()
        if profile is not None:
            profile["total_ms"] = float(profile.get("step_loop_ms", 0.0)) + (time.perf_counter() - ui_start) * 1000.0
        max_delta_K = float(result.get("max_delta_K", 0.0))
        done = bool(result.get("done", False))
        if self.timer.isActive():
            status = (
                f"Playing simulation: t = {self.prepared.time_s:.3g} s, "
                f"steps/update = {steps_completed}, max dT/update = {max_delta_K:.3e} K."
            )
            if max_delta_K <= 1.0e-12:
                status += " No temperature change is being produced by the current inputs/initial conditions."
            self._status(status)
            if done and not self.params.loop_playback:
                self.timer.stop()
        else:
            self._status(
                f"Simulation step complete: t = {self.prepared.time_s:.3g} s, "
                f"max dT = {max_delta_K:.3e} K."
            )
        if profile is not None:
            self._report_live_step_profile(profile, steps_completed, max_delta_K)
        self._apply_pending_runtime_changes()

    def _live_step_profiling_enabled(self) -> bool:
        return True

    def _after_worker_state_change(
        self,
        profile: dict[str, float] | None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        try:
            self._after_state_change(profile, payload)
        except TypeError:
            callback = getattr(self, "_after_state_change", None)
            if getattr(callback, "__name__", "") == "<lambda>":
                self._after_state_change()
                return
            raise

    def _live_step_profile_threshold_ms(self) -> float:
        return max(0.0, float(getattr(self.params, "live_step_profile_threshold_ms", 200.0)))

    def _report_live_step_profile(
        self,
        profile: dict[str, float],
        steps_completed: int,
        max_delta_K: float,
    ) -> None:
        total_ms = float(profile.get("total_ms", 0.0))
        if total_ms < self._live_step_profile_threshold_ms():
            return
        fields = {
            key: round(float(value), 3)
            for key, value in profile.items()
            if key.endswith("_ms")
        }
        fields.update(
            steps=int(steps_completed),
            nodes=0 if self.prepared is None else int(len(self.prepared.node_ids)),
            max_delta_K=f"{float(max_delta_K):.6g}",
        )
        # Record the per-step breakdown to the diagnostics log only. It is
        # intentionally NOT pushed to the on-screen status so the status stays
        # on the "Playing simulation" line rather than being overwritten.
        log_event("simulation live step profile", **fields)

    def save_current_trajectory(self) -> None:
        if self._simulation_worker_active():
            self.pause()
            self._status("Simulation worker is stopping; save the trajectory after the current compute step finishes.")
            return
        if self.prepared is None or self.folder is None:
            self._status("Initialize a graph simulation before saving.", True)
            return
        name = "simulation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        target = save_trajectory(self.folder, name, self.prepared)
        self._save_params_to_folder(target / "simulation_parameters.json", include_initial_temperatures=True)
        self._status(f"Saved trajectory to {target}.")

    def apply_component_initial_temperature(self) -> None:
        if self._simulation_worker_active():
            self.pause()
            self._status("Simulation worker is stopping; apply component temperature after the current compute step finishes.")
            return
        if self.model is None:
            self.use_current_graph()
        if self.model is None:
            return
        component = self.component_combo.currentText()
        temperature = float(self.component_temperature.value())
        count = 0
        for node in self.model.nodes.values():
            if node.component_name == component:
                node.initial_temperature_K = temperature
                count += 1
        self.pause()
        self.prepared = None
        self._reset_to_model_initial_temperatures()
        self._save_params_to_folder(include_initial_temperatures=True)
        self._update_colors()
        self._refresh_stats()
        self._status(f"Updated {count} cells in component {component}.")

    def sync_from_editor(
        self,
        model: ThermalGraphModel,
        folder: Path | None,
        reinitialize: bool = False,
    ) -> None:
        self._invalidate_role_caches()
        if self._simulation_worker_active():
            self.pause()
            self._status("Simulation worker is stopping; sync the editor graph after the current compute step finishes.")
            return
        if self.sys_id_state is not None:
            self.cancel_sys_id("Graph changed while sys ID was running; run cancelled.")
        if self.model is model:
            self.folder = folder
            self.matrices = build_matrices(model)
            self._sync_enabled_io_table()
            self._refresh_sys_id_matrix_list()
            if reinitialize and self.prepared is not None:
                was_playing = self.timer.isActive()
                self.timer.stop()
                self.initialize_simulation()
                if was_playing and self.prepared is not None:
                    self.play()
                return
            if self.prepared is None:
                self._reset_to_model_initial_temperatures()
            elif self.prepared is not None:
                self.prepared.mark_controller_stale()
            self._refresh_sensor_readouts()

    def refresh_live_readouts_from_editor(self, model: ThermalGraphModel, folder: Path | None) -> None:
        """Refresh cheap editor-driven readouts without rebuilding matrices or restarting playback."""
        self._invalidate_role_caches()
        if self._simulation_worker_active():
            return
        if self.sys_id_state is not None:
            self.cancel_sys_id("Graph changed while sys ID was running; run cancelled.")
        if self.model is model:
            self.folder = folder
            self._sync_enabled_io_table()
            if self.prepared is not None:
                self.prepared.mark_controller_stale()
            self._refresh_sensor_readouts()

    def refresh_cryocoolers_from_editor(self, model: ThermalGraphModel, folder: Path | None) -> None:
        """Apply a cryocooler (re)assignment without rebuilding matrices/radiation.

        A cryocooler is a runtime heat-sink source, so only the prepared sim's
        cryocooler devices need rebuilding -- a ~30 ms operation vs. a multi-second
        full re-prepare on a large graph. Falls back to the standard reinitialise
        path when the worker is mid-step (mutating the prepared sim then is unsafe)."""
        self._invalidate_role_caches()
        if self._simulation_worker_active():
            # Can't safely touch the prepared sim mid-step; use the existing safe
            # (deferred/pause) reinitialise path instead.
            self.sync_from_editor(model, folder, reinitialize=True)
            return
        if self.sys_id_state is not None:
            self.cancel_sys_id("Graph changed while sys ID was running; run cancelled.")
        if self.model is model:
            self.folder = folder
            self._sync_enabled_io_table()
            if self.prepared is not None:
                self.prepared.refresh_cryocoolers()
                self.prepared.mark_controller_stale()
            self._refresh_sensor_readouts()

    def refresh_controller_settings_from_editor(self, model: ThermalGraphModel, folder: Path | None) -> None:
        """Apply editor-side MIMO/controller edits without rebuilding matrices or temperatures."""
        if self._simulation_worker_active():
            self._pending_editor_controller_refresh = (model, folder)
            self.pause()
            self._status(
                "Simulation paused; controller edits will apply after the current compute step finishes."
            )
            return
        self._apply_editor_controller_refresh(model, folder)

    def _apply_editor_controller_refresh(self, model: ThermalGraphModel, folder: Path | None) -> None:
        if self.sys_id_state is not None:
            self.cancel_sys_id("Controller settings changed while sys ID was running; run cancelled.")
        if self.model is not model:
            return
        self.folder = folder
        self._sync_enabled_io_table()
        if self.prepared is not None:
            self.prepared.mark_controller_stale()
            self.prepared.reset_controller_integrators()
        self._simulation_reinitialize_pending = False
        self._refresh_sensor_readouts()

    def save_active_controller_gain_matrix_from_editor(self, model: ThermalGraphModel) -> None:
        if self.model is not model:
            return
        run_path = self._selected_sys_id_matrix_path()
        if run_path is None:
            return
        try:
            update_sys_id_gain_matrix(run_path, self.model.controller_gain_matrix)
            self._status(f"Updated active G matrix '{run_path.name}'.")
        except Exception as exc:
            self._status(f"Could not update active G matrix: {exc}", True)

    def reset_controller_integrators(self) -> None:
        if self._simulation_worker_active():
            self.pause()
            self._status("Simulation worker is stopping; reset MIMO integrators after the current compute step finishes.")
            return
        if self.prepared is None:
            self._status("Initialize the simulation before resetting MIMO integrators.", True)
            return
        self.prepared.reset_controller_integrators()
        self._refresh_stats()
        self._refresh_sensor_readouts()
        self._status("MIMO controller integrators reset.")

    def run_simulation_sys_id_for_G_ctrl(self) -> None:
        if self._simulation_worker_active():
            self.pause()
            self._status("Simulation worker is stopping; run G_ctrl sys ID after the current compute step finishes.")
            return
        if self.sys_id_state is not None:
            self._status("G_ctrl sys ID is already running.", True)
            return
        self.pause()
        if self.prepared is None:
            self.initialize_simulation()
        if self.prepared is None or self.model is None:
            self._status("Initialize a simulation before running G_ctrl sys ID.", True)
            return
        sensors = self._ordered_sys_id_sensors()
        heaters = self._ordered_sys_id_heaters()
        if not sensors or not heaters:
            self._status("Cannot run G_ctrl sys ID: tag at least one sensor and one heater.", True)
            return
        self.params = self._read_params()
        self.prepared.params = self.params
        original_snapshot = self.prepared.snapshot_state()
        global_temperature_K = self._sys_id_uniform_baseline_temperature()
        if self.sys_id_uniform_baseline.isChecked():
            self.prepared.set_temperatures(self._sys_id_baseline_temperatures(sensors, global_temperature_K))
            self._update_after_sys_id_step()
        global_snapshot = self.prepared.snapshot_state()
        self.sys_id_state = {
            "sensors": sensors,
            "heaters": heaters,
            "heater_index": 0,
            "phase": "start_heater",
            "elapsed_s": 0.0,
            "baseline_samples": [],
            "final_samples": [],
            "T0": None,
            "baseline_powers": {},
            "test_powers": {},
            "active_heater_powers": {},
            "G": np.zeros((len(sensors), len(heaters)), dtype=float),
            "warnings": [],
            "original_snapshot": original_snapshot,
            "global_snapshot": global_snapshot,
            "restore_between": bool(self.sys_id_restore_between_tests.isChecked()),
            "uniform_baseline": bool(self.sys_id_uniform_baseline.isChecked()),
            "keep_cryocooler": bool(self.sys_id_keep_cryocooler_active.isChecked()),
            "global_temperature_K": float(global_temperature_K),
            "background_temperature_K": float(global_temperature_K),
            "sensor_setpoint_baseline": bool(self.sys_id_uniform_baseline.isChecked()),
            "sensor_setpoints_K": [
                float(getattr(self.model.nodes[sensor_id], "controller_setpoint_K", global_temperature_K))
                for sensor_id in sensors
            ],
            "requested_delta_power": float(self.sys_id_step_power.value()),
            "duration_s": max(0.0, float(self.sys_id_duration_s.value())),
            "baseline_window_s": max(0.0, float(self.sys_id_baseline_window_s.value())),
            "final_window_s": max(0.0, float(self.sys_id_final_window_s.value())),
            "cancelled": False,
        }
        self.run_sys_id_button.setEnabled(False)
        self.cancel_sys_id_button.setEnabled(True)
        self.sys_id_status_label.setText("")
        self._set_sys_id_progress()
        self.sys_id_timer.start(0)

    def cancel_sys_id(self, message: Any = "G_ctrl sys ID cancelled.") -> None:
        if self.sys_id_state is None:
            return
        if not isinstance(message, str):
            message = "G_ctrl sys ID cancelled."
        self.sys_id_state["cancelled"] = True
        self._finish_sys_id(cancelled=True, message=message)

    def _step_sys_id(self) -> None:
        if self.sys_id_state is None or self.prepared is None or self.model is None:
            self.sys_id_timer.stop()
            return
        state = self.sys_id_state
        if state.get("cancelled"):
            self._finish_sys_id(cancelled=True, message="G_ctrl sys ID cancelled.")
            return
        heaters = state["heaters"]
        heater_index = int(state["heater_index"])
        if heater_index >= len(heaters):
            self._finish_sys_id(cancelled=False, message="G_ctrl sys ID complete.")
            return
        if state["phase"] == "start_heater":
            self._start_sys_id_heater()
            return
        if state["phase"] == "baseline":
            self._step_sys_id_baseline()
            return
        if state["phase"] == "experiment":
            self._step_sys_id_experiment()
            return

    def _start_sys_id_heater(self) -> None:
        assert self.sys_id_state is not None and self.prepared is not None
        state = self.sys_id_state
        if state["restore_between"]:
            self.prepared.restore_state(state["global_snapshot"])
        state["elapsed_s"] = 0.0
        state["baseline_samples"] = []
        state["final_samples"] = []
        state["T0"] = None
        # Sys ID is open-loop for G_ctrl: all heaters are held at 0 W except the one being stepped.
        state["baseline_powers"] = {int(heater_id): 0.0 for heater_id in state["heaters"]}
        state["active_heater_powers"] = dict(state["baseline_powers"])
        state["phase"] = "baseline"
        self._set_sys_id_progress()
        if float(state["baseline_window_s"]) <= 0.0:
            self._finish_sys_id_baseline()

    def _step_sys_id_baseline(self) -> None:
        assert self.sys_id_state is not None and self.prepared is not None
        state = self.sys_id_state
        state["baseline_samples"].append(self._collect_sensor_temperatures(state["sensors"]))
        self.prepared.step_with_forced_heater_powers(
            state["baseline_powers"],
            keep_cryocoolers_active=bool(state["keep_cryocooler"]),
        )
        state["elapsed_s"] = float(state["elapsed_s"]) + float(self.params.dt_s)
        self._update_after_sys_id_step()
        if float(state["elapsed_s"]) >= float(state["baseline_window_s"]):
            self._finish_sys_id_baseline()

    def _finish_sys_id_baseline(self) -> None:
        assert self.sys_id_state is not None and self.prepared is not None and self.model is not None
        state = self.sys_id_state
        if not state["baseline_samples"]:
            state["baseline_samples"].append(self._collect_sensor_temperatures(state["sensors"]))
        T0 = np.nanmean(np.vstack(state["baseline_samples"]), axis=0)
        state["T0"] = T0
        heater_id = int(state["heaters"][int(state["heater_index"])])
        baseline_powers = dict(state["baseline_powers"])
        baseline_power = float(baseline_powers.get(heater_id, 0.0))
        requested = baseline_power + float(state["requested_delta_power"])
        applied = min(max(requested, 0.0), self._heater_max_power(heater_id))
        delta_actual = applied - baseline_power
        if abs(delta_actual) <= 1.0e-12 or not np.isfinite(delta_actual):
            state["warnings"].append(f"Heater {heater_id} skipped: actual applied step was too small.")
            state["G"][:, int(state["heater_index"])] = 0.0
            self._advance_sys_id_heater()
            return
        test_powers = dict(baseline_powers)
        test_powers[heater_id] = applied
        state["test_powers"] = test_powers
        state["active_heater_powers"] = dict(test_powers)
        state["delta_actual"] = float(delta_actual)
        state["elapsed_s"] = 0.0
        state["final_samples"] = []
        state["phase"] = "experiment"
        self._set_sys_id_progress()

    def _step_sys_id_experiment(self) -> None:
        assert self.sys_id_state is not None and self.prepared is not None
        state = self.sys_id_state
        duration = float(state["duration_s"])
        final_window = float(state["final_window_s"])
        final_start = max(0.0, duration - final_window)
        self.prepared.step_with_forced_heater_powers(
            state["test_powers"],
            keep_cryocoolers_active=bool(state["keep_cryocooler"]),
        )
        state["elapsed_s"] = float(state["elapsed_s"]) + float(self.params.dt_s)
        if float(state["elapsed_s"]) >= final_start:
            state["final_samples"].append(self._collect_sensor_temperatures(state["sensors"]))
        self._update_after_sys_id_step()
        if float(state["elapsed_s"]) >= duration:
            self._finish_sys_id_experiment()

    def _finish_sys_id_experiment(self) -> None:
        assert self.sys_id_state is not None and self.prepared is not None
        state = self.sys_id_state
        if not state["final_samples"]:
            state["final_samples"].append(self._collect_sensor_temperatures(state["sensors"]))
        Tinf = np.nanmean(np.vstack(state["final_samples"]), axis=0)
        T0 = np.asarray(state["T0"], dtype=float)
        delta_actual = float(state["delta_actual"])
        column = (Tinf - T0) / delta_actual
        column = np.where(np.isfinite(column), column, 0.0)
        state["G"][:, int(state["heater_index"])] = column
        self._advance_sys_id_heater()

    def _advance_sys_id_heater(self) -> None:
        assert self.sys_id_state is not None
        self.sys_id_state["heater_index"] = int(self.sys_id_state["heater_index"]) + 1
        self.sys_id_state["active_heater_powers"] = {}
        self.sys_id_state["phase"] = "start_heater"
        self._set_sys_id_progress()

    def _finish_sys_id(self, *, cancelled: bool, message: str) -> None:
        if self.sys_id_state is None:
            return
        state = self.sys_id_state
        self.sys_id_timer.stop()
        restore_key = "original_snapshot" if state.get("uniform_baseline", False) else "global_snapshot"
        if self.prepared is not None and (restore_key in state) and (
            cancelled or state.get("restore_between", True) or state.get("uniform_baseline", False)
        ):
            self.prepared.restore_state(state[restore_key])
            self._update_after_sys_id_step()
        if not cancelled and self.model is not None:
            self._populate_G_ctrl_matrix(
                state["sensors"],
                state["heaters"],
                np.asarray(state["G"], dtype=float),
            )
            if self.prepared is not None:
                self.prepared.mark_controller_stale()
            self._save_sys_id_results()
            if self.on_controller_gain_matrix_changed is not None:
                self.on_controller_gain_matrix_changed()
        warnings = list(state.get("warnings", []))
        self.sys_id_state = None
        self.run_sys_id_button.setEnabled(True)
        self.cancel_sys_id_button.setEnabled(False)
        self.sys_id_progress_label.setText("Idle." if cancelled else "Complete.")
        status = message
        if warnings:
            status += "\n" + "\n".join(warnings[:6])
        self.sys_id_status_label.setText(status)
        self._status(message, cancelled)

    def _ordered_sys_id_sensors(self) -> list[int]:
        if self.model is None:
            return []
        return [
            int(node_id)
            for node_id, node in sorted(self.model.nodes.items(), key=lambda item: int(item[0]))
            if node.is_sensor and self._sensor_enabled_for_simulation(int(node_id))
        ]

    def _ordered_sys_id_heaters(self) -> list[int]:
        if self.model is None:
            return []
        return [
            int(node_id)
            for node_id, node in sorted(self.model.nodes.items(), key=lambda item: int(item[0]))
            if node.is_heater and self._heater_enabled_for_simulation(int(node_id))
        ]

    def _collect_sensor_temperatures(self, sensor_ids: list[int]) -> np.ndarray:
        if self.prepared is None:
            return np.zeros(len(sensor_ids), dtype=float)
        index = {int(node_id): row for row, node_id in enumerate(self.prepared.node_ids)}
        values = []
        for sensor_id in sensor_ids:
            values.append(
                sensor_readout_temperature_K(
                    self.model,
                    index,
                    self.prepared.temperatures_K,
                    int(sensor_id),
                )
            )
        return np.asarray(values, dtype=float)

    def _populate_G_ctrl_matrix(self, sensor_ids: list[int], heater_ids: list[int], G: np.ndarray) -> None:
        if self.model is None:
            return
        for i, sensor_id in enumerate(sensor_ids):
            for j, heater_id in enumerate(heater_ids):
                self.model.set_controller_gain(int(sensor_id), int(heater_id), float(G[i, j]))

    def _save_sys_id_results(self) -> None:
        if self.model is None or self.folder is None or self.sys_id_state is None:
            return
        state = self.sys_id_state
        run_name = self._sys_id_run_name(state)
        metadata = {
            "graph_name": self.model.metadata.graph_name,
            "requested_delta_power_W": float(state.get("requested_delta_power", 0.0)),
            "duration_s": float(state.get("duration_s", 0.0)),
            "baseline_window_s": float(state.get("baseline_window_s", 0.0)),
            "final_window_s": float(state.get("final_window_s", 0.0)),
            "restore_between": bool(state.get("restore_between", True)),
            "uniform_baseline": bool(state.get("uniform_baseline", True)),
            "global_temperature_K": float(state.get("global_temperature_K", np.nan)),
            "background_temperature_K": float(state.get("background_temperature_K", np.nan)),
            "sensor_setpoint_baseline": bool(state.get("sensor_setpoint_baseline", False)),
            "sensor_setpoints_K": [float(value) for value in state.get("sensor_setpoints_K", [])],
            "keep_cryocooler": bool(state.get("keep_cryocooler", True)),
            "warnings": list(state.get("warnings", [])),
        }
        try:
            run_path = save_sys_id_gain_matrix(
                self.folder,
                run_name,
                [int(value) for value in state["sensors"]],
                [int(value) for value in state["heaters"]],
                np.asarray(state["G"], dtype=float),
                metadata,
            )
            save_graph_folder(self.model, self.folder)
            self._refresh_sys_id_matrix_list(select_path=run_path)
        except Exception as exc:
            self.sys_id_status_label.setText(
                self.sys_id_status_label.text() + f"\nG_ctrl populated but sys ID artifact/graph save failed: {exc}"
            )

    def _heater_max_power(self, heater_id: int) -> float:
        if self.model is None or heater_id not in self.model.nodes:
            return 0.0
        node = self.model.nodes[int(heater_id)]
        return max(0.0, float(node.heater.heater_max_power_W) * float(node.heater.heater_efficiency))

    def _sys_id_uniform_baseline_temperature(self) -> float:
        return float(self.sys_id_global_temperature_K.value())

    def _sys_id_baseline_temperatures(self, sensor_ids: list[int], background_temperature_K: float) -> np.ndarray:
        if self.prepared is None or self.model is None:
            return np.zeros(0, dtype=float)
        temperatures = np.full(len(self.prepared.node_ids), float(background_temperature_K), dtype=float)
        node_index = {int(node_id): row for row, node_id in enumerate(self.prepared.node_ids)}
        for sensor_id in sensor_ids:
            row = node_index.get(int(sensor_id))
            if row is None:
                continue
            node = self.model.nodes.get(int(sensor_id))
            if node is None:
                continue
            temperatures[row] = float(getattr(node, "controller_setpoint_K", background_temperature_K))
        return temperatures

    def _sys_id_run_name(self, state: dict[str, Any]) -> str:
        temp = self._name_number(float(state.get("global_temperature_K", 293.15)), precision=2)
        step = self._name_number(float(state.get("requested_delta_power", 0.0)), precision=3)
        duration = self._name_number(float(state.get("duration_s", 0.0)), precision=1)
        cooler = "cooler_on" if bool(state.get("keep_cryocooler", True)) else "cooler_off"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"sys_id_T{temp}K_{cooler}_dP{step}W_dur{duration}s_{timestamp}"

    def _name_number(self, value: float, precision: int = 3) -> str:
        text = f"{float(value):.{precision}f}".rstrip("0").rstrip(".")
        if text == "-0":
            text = "0"
        return text.replace("-", "m").replace(".", "p")

    def _set_sys_id_progress(self) -> None:
        if self.sys_id_state is None:
            self.sys_id_progress_label.setText("Idle.")
            return
        state = self.sys_id_state
        total = len(state["heaters"])
        current = min(int(state["heater_index"]) + 1, total)
        heater = state["heaters"][int(state["heater_index"])] if int(state["heater_index"]) < total else "done"
        self.sys_id_progress_label.setText(
            f"heater {current}/{total}: {heater}, phase: {state['phase']}"
        )

    def _update_after_sys_id_step(self) -> None:
        if self.prepared is None:
            return
        self.temperature_by_node = {
            int(node_id): float(temp)
            for node_id, temp in zip(self.prepared.node_ids, self.prepared.temperatures_K)
        }
        self._update_colors()
        self._refresh_stats()
        self._refresh_sensor_readouts()

    def _after_state_change(
        self,
        profile: dict[str, float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        assert self.prepared is not None
        payload = payload or {}
        start = time.perf_counter()
        temperature_by_node = payload.get("temperature_by_node")
        if not temperature_by_node:
            # Non-worker paths (step back, initialize) recompute on the main thread.
            temperature_by_node = {
                int(node_id): float(temp)
                for node_id, temp in zip(self.prepared.node_ids, self.prepared.temperatures_K)
            }
        self.temperature_by_node = temperature_by_node
        _record_profile_ms(profile, "temperature_map_ms", start)
        start = time.perf_counter()
        self._update_colors()
        _record_profile_ms(profile, "color_update_render_ms", start)
        start = time.perf_counter()
        self._refresh_stats(power_balance=payload.get("power_balance"))
        _record_profile_ms(profile, "stats_refresh_ms", start)
        start = time.perf_counter()
        self._refresh_sensor_readouts(
            heater_powers=payload.get("heater_powers"),
            cryocooler_diagnostics=payload.get("cryocooler_diagnostics"),
        )
        _record_profile_ms(profile, "sensor_readouts_ms", start)
        start = time.perf_counter()
        self._sync_time_slider_to_history()
        _record_profile_ms(profile, "time_slider_ms", start)

    def _reset_time_slider(self) -> None:
        self.time_slider.blockSignals(True)
        self.time_slider.setRange(0, 0)
        self.time_slider.setValue(0)
        self.time_slider.blockSignals(False)

    def _sync_time_slider_to_history(self) -> None:
        if self.prepared is None:
            self._reset_time_slider()
            return
        history_max = max(0, len(self.prepared.history) - 1)
        slider_max = min(QT_SLIDER_MAXIMUM, history_max)
        slider_value = min(slider_max, max(0, int(getattr(self.prepared, "history_index", history_max))))
        self.time_slider.blockSignals(True)
        if self.time_slider.maximum() != slider_max:
            self.time_slider.setRange(0, slider_max)
        self.time_slider.setValue(slider_value)
        self.time_slider.blockSignals(False)

    def _handle_time_slider(self, value: int) -> None:
        if self.prepared is None or not self.prepared.history:
            return
        value = max(0, min(int(value), len(self.prepared.history) - 1))
        self.prepared.seek(value)
        self._after_state_change()

    def _handle_parameter_change(self, changed_field: str | None = None, *_: Any) -> None:
        if self.sys_id_state is not None:
            self.cancel_sys_id("Simulation parameter changed while sys ID was running; run cancelled.")
            return
        previous_params = self.params
        if isinstance(changed_field, str) and changed_field:
            self.params = self._params_with_widget_value(changed_field)
        else:
            self.params = self._read_params()
        changed = _changed_parameter_names(previous_params, self.params)
        if not changed:
            return
        if self._simulation_worker_active() and changed <= _CONTROLLER_RUNTIME_HOTSWAP_FIELDS:
            self._queue_controller_runtime_parameter_change(changed)
            return
        if self._simulation_worker_active() and not changed <= _NONBLOCKING_PARAMETER_FIELDS:
            self.params = previous_params
            self.pause()
            self._status(
                "Simulation worker is stopping; parameter changes will apply after the current compute step finishes."
            )
            return
        if changed <= _CONTROLLER_RUNTIME_HOTSWAP_FIELDS:
            self._apply_controller_runtime_parameter_change(changed)
            return
        if changed <= _LIGHTWEIGHT_RUNTIME_PARAMETER_FIELDS:
            self._apply_lightweight_runtime_parameter_change(changed)
            return
        if changed <= _DISPLAY_PARAMETER_FIELDS:
            self._apply_display_parameter_change()
            return
        self._save_params_to_folder()
        if self.prepared is not None:
            self.prepared.params = self.params
            if changed & _CONTROLLER_PARAMETER_FIELDS:
                self.prepared.mark_controller_stale()
                self.prepared.reset_controller_integrators()
            if changed & _REINITIALIZE_PARAMETER_FIELDS:
                self._simulation_reinitialize_pending = True
                if self.timer.isActive():
                    self.pause()
                self._status("Simulation parameters saved. Reinitialize, play, or step to apply matrix/stepper changes.")
            elif changed & _DISPLAY_PARAMETER_FIELDS:
                self._update_colors()
            else:
                if "playback_speed" in changed and self.timer.isActive():
                    self.timer.start(self._playback_timer_interval_ms())
                self._refresh_stats()
                self._refresh_sensor_readouts()

    def _queue_controller_runtime_parameter_change(self, changed: set[str]) -> None:
        self._pending_controller_runtime_params = self.params
        self._pending_controller_runtime_fields = set(changed)
        self.pause()
        self._schedule_parameter_save()
        self._status(
            "Simulation paused; controller parameter changes will apply after the current compute step finishes."
        )

    def _apply_controller_runtime_parameter_change(self, changed: set[str]) -> None:
        self._save_params_to_folder()
        if self.prepared is not None:
            self.prepared.params = self.params
            self.prepared.mark_controller_stale()
            self.prepared.reset_controller_integrators()
        self._simulation_reinitialize_pending = False
        self._refresh_stats()
        self._refresh_sensor_readouts()

    def _apply_pending_runtime_changes(self) -> bool:
        applied = False
        pending_params = getattr(self, "_pending_controller_runtime_params", None)
        if pending_params is not None:
            pending_fields = set(getattr(self, "_pending_controller_runtime_fields", set()) or set())
            self.params = pending_params
            self._pending_controller_runtime_params = None
            self._pending_controller_runtime_fields = set()
            self._apply_controller_runtime_parameter_change(pending_fields)
            applied = True
        pending_editor = getattr(self, "_pending_editor_controller_refresh", None)
        if pending_editor is not None:
            self._pending_editor_controller_refresh = None
            model, folder = pending_editor
            self._apply_editor_controller_refresh(model, folder)
            applied = True
        return applied

    def _apply_lightweight_runtime_parameter_change(self, changed: set[str]) -> None:
        if self.prepared is not None:
            self.prepared.params = self.params
        if "playback_speed" in changed and self.timer.isActive():
            self.timer.start(self._playback_timer_interval_ms())
        self._schedule_parameter_save()

    def _apply_display_parameter_change(self) -> None:
        if self.prepared is not None:
            self.prepared.params = self.params
        self._update_colors()
        self._schedule_parameter_save()

    def _params_with_widget_value(self, name: str) -> SimulationParameters:
        if name == "input_mode":
            return replace(self.params, input_mode=self.input_mode.currentText())
        widget = self.inputs.get(name)
        if widget is None or not hasattr(self.params, name):
            return self._read_params()
        if hasattr(widget, "isChecked"):
            value = bool(widget.isChecked())
        elif hasattr(widget, "value"):
            value = widget.value()
            current = getattr(self.params, name)
            if isinstance(current, int) and not isinstance(current, bool):
                value = int(value)
            else:
                value = float(value)
        else:
            return self._read_params()
        return replace(self.params, **{name: value})

    def _schedule_parameter_save(self) -> None:
        timer = getattr(self, "parameter_save_timer", None)
        if timer is not None:
            timer.start(500)
            return
        self._save_params_to_folder()

    def _flush_deferred_parameter_save(self) -> None:
        self._save_params_to_folder()

    def _read_params(self) -> SimulationParameters:
        # Start from the current params and override ONLY the fields the UI exposes.
        # This guarantees every selected setting is applied on Run AND that any
        # param without a widget (e.g. solver knobs, colormap, passive-drift source,
        # or fields added later) is carried through rather than silently reset to a
        # dataclass default.
        return replace(
            self.params,
            dt_s=float(self.inputs["dt_s"].value()),
            t_final_s=float(self.inputs["t_final_s"].value()),
            playback_speed=float(self.inputs["playback_speed"].value()),
            use_ambient_radiation=bool(self.inputs["use_ambient_radiation"].isChecked()),
            T_env_K=float(self.inputs["T_env_K"].value()),
            interior_environment_temperature_K=float(
                self.inputs["interior_environment_temperature_K"].value()
            ),
            use_radiative_coupling=bool(self.inputs["use_radiative_coupling"].isChecked()),
            input_mode=self.input_mode.currentText(),
            cryocooler_max_power_W=float(self.inputs["cryocooler_max_power_W"].value()),
            cryocooler_capacity_scale=float(self.inputs["cryocooler_capacity_scale"].value()),
            cryocooler_enabled=bool(self.inputs["cryocooler_enabled"].isChecked()),
            autoscale_temperature=bool(self.inputs["autoscale_temperature"].isChecked()),
            color_min_K=float(self.inputs["color_min_K"].value()),
            color_max_K=float(self.inputs["color_max_K"].value()),
            loop_playback=bool(self.inputs["loop_playback"].isChecked()),
            use_temperature_dependent_properties=bool(
                self.inputs["use_temperature_dependent_properties"].isChecked()
            ),
            use_midpoint_property_coupling=bool(
                self.inputs["use_midpoint_property_coupling"].isChecked()
            ),
            copper_rrr=int(self.inputs["copper_rrr"].value()),
            simulation_history_limit=int(self.inputs["simulation_history_limit"].value()),
            live_step_profiling_enabled=True,
            mimo_controller_enabled=self._mimo_controller_should_run(),
            mimo_hold_threshold_K=float(self.inputs["mimo_hold_threshold_K"].value()),
            mimo_coarse_threshold_K=float(self.inputs["mimo_coarse_threshold_K"].value()),
            mimo_default_heater_max_power_W=float(self.inputs["mimo_default_heater_max_power_W"].value()),
            mimo_lambda_u=float(self.inputs["mimo_lambda_u"].value()),
            mimo_rho_du=float(self.inputs["mimo_rho_du"].value()),
            mimo_heater_slew_rate_W_per_s=float(self.inputs["mimo_heater_slew_rate_W_per_s"].value()),
            mimo_v_cmd_abs_max_K_per_s=float(self.inputs["mimo_v_cmd_abs_max_K_per_s"].value()),
            heater_sensor_pair_alpha=float(self.inputs["heater_sensor_pair_alpha"].value()),
            role_contact_tolerance_mm=float(self.inputs["role_contact_tolerance_mm"].value()),
            role_contact_tolerance_max_mm=float(self.inputs["role_contact_tolerance_max_mm"].value()),
            role_contact_tolerance_growth_factor=float(self.inputs["role_contact_tolerance_growth_factor"].value()),
            drift_lpf_tau_s=float(self.inputs["drift_lpf_tau_s"].value()),
            derivative_dt_floor_s=float(self.inputs["derivative_dt_floor_s"].value()),
            mimo_integral_abs_max=float(self.inputs["mimo_integral_abs_max"].value()),
            mimo_freeze_integral_when_saturated=bool(self.inputs["mimo_freeze_integral_when_saturated"].isChecked()),
            enabled_heater_node_ids=(
                tuple(sorted(int(node_id) for node_id in self.enabled_heater_node_ids))
                if self._enabled_io_initialized
                else None
            ),
            enabled_sensor_node_ids=(
                tuple(sorted(int(node_id) for node_id in self.enabled_sensor_node_ids))
                if self._enabled_io_initialized
                else None
            ),
            mimo_controller_scheme=self._controller_scheme_value(),
            modal_controller_path=self._modal_controller_path_value(),
        )

    def _controller_scheme_value(self) -> str:
        combo = getattr(self, "controller_scheme_combo", None)
        if combo is None:
            return str(getattr(self.params, "mimo_controller_scheme", "pid_qp"))
        label = combo.currentText()
        for key, text in self._controller_scheme_labels.items():
            if text == label:
                return key
        return "pid_qp"

    def _modal_controller_path_value(self) -> str:
        # The modal controller artifact travels with the graph folder.
        if self.folder is not None:
            candidate = self.folder / "modal_controller.npz"
            if candidate.exists():
                return str(candidate)
        return str(getattr(self.params, "modal_controller_path", "") or "")

    def _mimo_controller_should_run(self) -> bool:
        if self.input_mode.currentText() != "heater_inputs" or self.model is None:
            return False
        for heater_id in tuple(self.enabled_heater_node_ids or self._known_heater_node_ids):
            heater = self.model.nodes.get(int(heater_id))
            if heater is None or not _node_uses_mimo_controller(
                heater,
                heater_enabled=self._heater_enabled_for_simulation(int(heater_id)),
            ):
                continue
            sensor_id = getattr(heater, "assigned_sensor_id", None)
            if sensor_id is None and bool(getattr(heater, "is_sensor", False)):
                sensor_id = int(heater_id)
            if sensor_id is None:
                continue
            sensor = self.model.nodes.get(int(sensor_id))
            if sensor is not None and _node_uses_mimo_controller(
                sensor,
                sensor_enabled=self._sensor_enabled_for_simulation(int(sensor_id)),
            ):
                return True
        return False

    def _load_params_from_folder(self) -> None:
        path = self._params_path()
        if path is None:
            return
        self.params, self.parameter_extras = load_simulation_parameters(path)
        if self.model is not None:
            apply_initial_temperature_parameter_payload(self.model, self.parameter_extras)
        self._sync_params_to_widgets()

    def _save_params_to_folder(
        self,
        override_path: Path | None = None,
        include_initial_temperatures: bool = False,
    ) -> None:
        path = override_path or self._params_path()
        if path is None:
            if include_initial_temperatures:
                self._status("Initial temperatures are not saved yet because this graph has no folder.", True)
            return
        extras = dict(self.parameter_extras)
        if include_initial_temperatures and self.model is not None:
            extras.update(initial_temperature_parameter_payload(self.model))
            self.parameter_extras = dict(extras)
        save_simulation_parameters(path, self.params, extras)

    def _params_path(self) -> Path | None:
        if self.folder is None:
            return None
        return self.folder / "simulation_parameters.json"

    def _sync_params_to_widgets(self) -> None:
        for key, widget in self.inputs.items():
            if not hasattr(self.params, key):
                continue
            if hasattr(widget, "setValue"):
                widget.blockSignals(True)
                value = getattr(self.params, key)
                if isinstance(value, int) and not isinstance(value, bool):
                    widget.setValue(int(value))
                else:
                    widget.setValue(float(value))
                widget.blockSignals(False)
            elif hasattr(widget, "setChecked"):
                widget.blockSignals(True)
                widget.setChecked(bool(getattr(self.params, key)))
                widget.blockSignals(False)
        self.input_mode.blockSignals(True)
        self.input_mode.setCurrentText(self.params.input_mode)
        self.input_mode.blockSignals(False)
        combo = getattr(self, "controller_scheme_combo", None)
        if combo is not None:
            scheme = str(getattr(self.params, "mimo_controller_scheme", "pid_qp"))
            combo.blockSignals(True)
            combo.setCurrentText(self._controller_scheme_labels.get(scheme, "PID + QP allocator"))
            combo.blockSignals(False)

    def _visible_node_ids(self) -> set[int] | None:
        """Node set to draw. ``None`` means all; otherwise unassigned-material cells
        are filtered out (``ZERO MATTER`` stays visible via ``is_unassigned_material``)."""
        if self.model is None or not self._hide_unassigned_getter():
            return None
        return {
            int(node_id)
            for node_id, node in self.model.nodes.items()
            if not is_unassigned_material(node.material)
        }

    def _handle_hide_unassigned_toggled(self, *_: Any) -> None:
        if self.on_hide_unassigned_toggled is not None:
            self.on_hide_unassigned_toggled(bool(self.hide_unassigned_checkbox.isChecked()))

    def set_hide_unassigned_material(self, enabled: bool) -> None:
        """Reflect the shared hide-unassigned state and redraw the sim view."""
        if hasattr(self, "hide_unassigned_checkbox"):
            self.hide_unassigned_checkbox.blockSignals(True)
            self.hide_unassigned_checkbox.setChecked(bool(enabled))
            self.hide_unassigned_checkbox.blockSignals(False)
        if self.model is not None:
            self._draw_current(reset_camera=False)

    def _draw_current(self, reset_camera: bool) -> None:
        if self.model is None:
            return
        log_event(
            "simulation draw_current start",
            nodes=len(self.model.nodes),
            edges=len(self.model.edges),
            reset_camera=reset_camera,
        )
        self.viewer.set_toggles(
            False,
            False,
            self.show_heaters.isChecked(),
            self.show_sensors.isChecked(),
            self.show_coolers.isChecked(),
        )
        self._sync_view_controls_to_viewer()
        self.viewer.selected_node_id = None
        self.viewer.draw(
            self.model,
            reset_camera=reset_camera,
            visible_node_ids=self._visible_node_ids(),
            node_scalar_values=self._temperature_values(),
            scalar_cmap="jet",
            scalar_clim=self._temperature_clim(),
            scalar_bar_title="Temperature [K]",
        )
        self._update_cross_section_value_label()
        log_event("simulation draw_current viewer.draw complete")
        self._refresh_stats()
        self._refresh_sensor_readouts()
        log_event("simulation draw_current complete")

    def _update_colors(self) -> None:
        updated = self.viewer.update_node_scalars(
            self._temperature_values(),
            scalar_clim=self._temperature_clim(),
        )
        if not updated and self.prepared is None:
            self._draw_current(reset_camera=False)

    def _refresh_initialized_view(self) -> None:
        updated = self.viewer.update_node_scalars(
            self._temperature_values(),
            scalar_clim=self._temperature_clim(),
        )
        if updated:
            log_event("simulation initialize updated existing view scalars")
            return
        log_event("simulation initialize redraw view after scalar update miss")
        self._draw_current(reset_camera=False)

    def _temperature_values(self) -> dict[int, float]:
        if self.model is None:
            return {}
        return self.temperature_by_node or {
            int(node_id): float(node.initial_temperature_K)
            for node_id, node in self.model.nodes.items()
        }

    def _reset_to_model_initial_temperatures(self) -> None:
        if self.model is None:
            self.temperature_by_node = {}
            return
        self.temperature_by_node = {
            int(node_id): float(node.initial_temperature_K)
            for node_id, node in self.model.nodes.items()
        }

    def _temperature_clim(self) -> tuple[float, float]:
        temperatures = self._temperature_values()
        values = np.array(list(temperatures.values()), dtype=float)
        if self.params.autoscale_temperature and values.size:
            cmin = float(np.min(values))
            cmax = float(np.max(values))
            if cmax <= cmin:
                cmax = cmin + 1.0
        else:
            cmin, cmax = float(self.params.color_min_K), float(self.params.color_max_K)
            if cmax <= cmin:
                cmax = cmin + 1.0
        return (cmin, cmax)

    def _refresh_stats(self, power_balance: dict[str, float] | None = None) -> None:
        values = np.array(list((self.temperature_by_node or {}).values()), dtype=float)
        if values.size == 0 and self.model is not None:
            values = np.array([node.initial_temperature_K for node in self.model.nodes.values()], dtype=float)
        if values.size == 0:
            self.stats_label.setText("No simulation initialized.")
            self._refresh_sensor_readouts()
            return
        time_s = self.prepared.time_s if self.prepared is not None else 0.0
        power_text = ""
        if self.prepared is not None:
            try:
                balance = power_balance if power_balance is not None else self.prepared.power_balance_W()
                power_in = balance["heater_W"] + max(0.0, balance["radiation_W"])
                power_out = balance["cryocooler_W"] + max(0.0, -balance["radiation_W"])
                power_text = (
                    f"\npower in = {power_in:.3g} W  |  out = {power_out:.3g} W  |  net = {balance['net_W']:+.3g} W"
                    f"\n  heaters {balance['heater_W']:.3g} W · cryo {balance['cryocooler_W']:.3g} W · "
                    f"radiation {balance['radiation_W']:+.3g} W"
                )
            except Exception:
                power_text = ""
        self.stats_label.setText(
            f"t = {time_s:.3g} s\n"
            f"min = {np.min(values):.3f} K / {np.min(values) - 273.15:.3f} C\n"
            f"max = {np.max(values):.3f} K / {np.max(values) - 273.15:.3f} C\n"
            f"mean = {np.mean(values):.3f} K / {np.mean(values) - 273.15:.3f} C"
            + power_text
        )
        if self.prepared is not None and self._mimo_controller_should_run():
            if self.sys_id_state is not None:
                self.controller_status_label.setText("G_ctrl sys ID running open-loop; PID/MIMO/manual heater commands are bypassed.")
            else:
                rms = self.prepared.controller_weighted_rms_error
                rms_text = "?" if rms is None else f"{float(rms):.4g} K"
                warnings = "\n".join(self.prepared.controller_warnings[:4])
                diagnostics = self.prepared.controller_allocator_diagnostics
                allocation_text = ""
                if diagnostics:
                    b_sources = diagnostics.get("B_s_source", []) or []
                    source_text = ",".join(sorted({str(source) for source in b_sources if str(source)})) or "?"
                    allocation_text = (
                        f"\nthermal-rate QP: sensors={diagnostics.get('active_sensor_count', '?')}, "
                        f"heaters={diagnostics.get('active_heater_count', '?')}, "
                        f"||v_cmd||={float(diagnostics.get('rate_command_norm', 0.0)):.4g} K/s, "
                        f"||u_ff||={float(diagnostics.get('feedforward_hold_power_norm', 0.0)):.4g} W, "
                        f"||u||={float(diagnostics.get('heater_command_norm', 0.0)):.4g}, "
                        f"rate_resid={float(diagnostics.get('predicted_dTdt_residual_norm', 0.0)):.4g} K/s, "
                        f"B_s={source_text}"
                    )
                    if diagnostics.get("v_cmd_clipped"):
                        allocation_text += ", v_cmd clipped"
                    if diagnostics.get("bounds_active"):
                        allocation_text += ", bounds active"
                self.controller_status_label.setText(
                    f"MIMO thermal-rate QP mode = {self.prepared.controller_mode}, weighted RMS error = {rms_text}"
                    + allocation_text
                    + (f"\n{warnings}" if warnings else "")
                )
        else:
            self.controller_status_label.setText("MIMO controller disabled.")

    def _sys_id_readout_heater_powers(self) -> dict[int, float] | None:
        if self.sys_id_state is None:
            return None
        state = self.sys_id_state
        if state.get("phase") not in {"baseline", "experiment"}:
            return None
        active = state.get("active_heater_powers", {})
        return {
            int(heater_id): max(0.0, float(active.get(int(heater_id), 0.0)))
            for heater_id in state.get("heaters", [])
        }

    def _rebuild_role_caches(self) -> None:
        self._role_cache_model_id = id(self.model) if self.model is not None else None
        if self.model is None:
            self._cooling_nodes_cache = []
            self._heating_sensor_nodes_cache = []
            self._heaters_by_sensor_cache = {}
            return
        self._cooling_nodes_cache = [node for node in self.model.nodes.values() if node.has_cryocooler]
        self._heating_sensor_nodes_cache = self._heating_sensor_nodes()
        # Reverse map sensor_id -> {heater_ids that target it}, built in ONE pass so
        # the per-sensor readout lookup is O(1) instead of scanning every node (that
        # per-sensor-per-frame O(N) scan was the dominant live-step cost at ~471k nodes).
        heaters_by_sensor: dict[int, set[int]] = {}
        for node in self.model.nodes.values():
            if not bool(getattr(node, "is_heater", False)):
                continue
            assigned = getattr(node, "assigned_sensor_id", None)
            if assigned is not None:
                heaters_by_sensor.setdefault(int(assigned), set()).add(int(node.node_id))
        self._heaters_by_sensor_cache = heaters_by_sensor

    def _invalidate_role_caches(self) -> None:
        self._role_cache_model_id = None

    def _cached_cooling_nodes(self) -> list[Any]:
        if self._role_cache_model_id != (id(self.model) if self.model is not None else None):
            self._rebuild_role_caches()
        return self._cooling_nodes_cache

    def _cached_heating_sensor_nodes(self) -> list[Any]:
        if self._role_cache_model_id != (id(self.model) if self.model is not None else None):
            self._rebuild_role_caches()
        return self._heating_sensor_nodes_cache

    def _cached_heaters_by_sensor(self) -> dict[int, set[int]]:
        if getattr(self, "_role_cache_model_id", None) != (id(self.model) if self.model is not None else None):
            self._rebuild_role_caches()
        return self._heaters_by_sensor_cache

    def _refresh_sensor_readouts(
        self,
        heater_powers: dict[int, float] | None = None,
        cryocooler_diagnostics: list[dict[str, Any]] | None = None,
    ) -> None:
        if self.model is None:
            self.sensor_readout_box.setVisible(False)
            self.cooling_readout_table.setRowCount(0)
            self.heating_readout_tree.clear()
            return
        # Role-node lists are cached (they change only on model/role edits), and
        # heater powers / cryocooler diagnostics are supplied by the worker when
        # available so this method does no per-frame O(N) work on the GUI thread.
        cooling_nodes = self._cached_cooling_nodes()
        heating_sensors = self._cached_heating_sensor_nodes()
        self.sensor_readout_box.setVisible(bool(cooling_nodes or heating_sensors))
        self.cooling_readout_box.setVisible(bool(cooling_nodes))
        self.heating_readout_box.setVisible(bool(heating_sensors))
        temperatures = self._temperature_values()
        sys_id_heater_powers = self._sys_id_readout_heater_powers()
        if sys_id_heater_powers is not None:
            heater_powers = sys_id_heater_powers
        elif heater_powers is None:
            heater_powers = self.prepared.heater_actuator_power_by_node() if self.prepared is not None else {}
        if cryocooler_diagnostics is None:
            cryocooler_diagnostics = self.prepared.cryocooler_diagnostics() if self.prepared is not None else []
        node_index = (
            self.prepared.node_index_by_id
            if self.prepared is not None and self.prepared.node_index_by_id
            else (
                {int(node_id): row for row, node_id in enumerate(self.prepared.node_ids)}
                if self.prepared is not None
                else {}
            )
        )
        self._refresh_cooling_readouts(cooling_nodes, cryocooler_diagnostics)
        self._refresh_heating_readouts(heating_sensors, temperatures, heater_powers, node_index)

    def _refresh_cooling_readouts(
        self,
        cooling_nodes: list[Any],
        diagnostics: list[dict[str, Any]],
    ) -> None:
        if not diagnostics and cooling_nodes:
            diagnostics = [
                {
                    "cryocooler_id": str(getattr(node, "cryocooler_id", "") or node.component_name or node.node_id),
                    "source_node_ids": [int(node.node_id)],
                    "representative_temperature_K": None,
                    "base_curve_capacity_W": 0.0,
                    "capacity_scale": float(self.params.cryocooler_capacity_scale),
                    "applied_cooling_W": 0.0,
                    "receiving_node_count": 0,
                    "enabled": bool(self.params.cryocooler_enabled and getattr(node, "cryocooler_enabled", True)),
                }
                for node in sorted(cooling_nodes, key=lambda item: item.node_id)
            ]
        self.cooling_readout_table.setRowCount(len(diagnostics))
        for row, diagnostic in enumerate(diagnostics):
            source_ids = [int(value) for value in diagnostic.get("source_node_ids", []) or []]
            id_item = self.QtWidgets.QTableWidgetItem(str(diagnostic.get("cryocooler_id", "?")))
            if source_ids:
                id_item.setData(self.QtCore.Qt.UserRole, int(source_ids[0]))
            self.cooling_readout_table.setItem(row, 0, id_item)
            tip = diagnostic.get("representative_temperature_K")
            tip_text = _format_temperature(float(tip)) if tip is not None else "invalid"
            values = (
                tip_text,
                _format_power(float(diagnostic.get("base_curve_capacity_W", 0.0))),
                f"{float(diagnostic.get('capacity_scale', 1.0)):.3f}",
                _format_power(float(diagnostic.get("applied_cooling_W", 0.0))),
                str(int(diagnostic.get("receiving_node_count", 0))),
                "yes" if bool(diagnostic.get("enabled", False)) else "no",
            )
            for column, text in enumerate(values, start=1):
                self.cooling_readout_table.setItem(row, column, self.QtWidgets.QTableWidgetItem(text))
        self.cooling_readout_table.resizeColumnsToContents()

    def _refresh_heating_readouts(
        self,
        heating_sensors: list[Any],
        temperatures: dict[int, float],
        heater_powers: dict[int, float],
        node_index: dict[int, int],
    ) -> None:
        selected_kind = self._readout_editor_kind if self._readout_editor_kind in {"sensor", "heater"} else None
        selected_node_id = self._readout_editor_node_id if selected_kind is not None else None
        restore_item = None
        self.heating_readout_tree.blockSignals(True)
        self.heating_readout_tree.clear()
        for sensor in sorted(heating_sensors, key=lambda item: item.node_id):
            sensor_id = int(sensor.node_id)
            measured = self._sensor_measured_temperature(sensor_id, temperatures, node_index)
            desired = float(getattr(sensor, "controller_setpoint_K", 293.15))
            sensor_item = self.QtWidgets.QTreeWidgetItem(
                [
                    "sensor",
                    str(sensor_id),
                    _format_temperature(measured),
                    _format_temperature(desired),
                    _format_error(desired - measured),
                    "",
                ]
            )
            sensor_item.setData(1, self.QtCore.Qt.UserRole, sensor_id)
            if selected_kind == "sensor" and selected_node_id == sensor_id:
                restore_item = sensor_item
            font = sensor_item.font(0)
            font.setBold(True)
            for column in range(6):
                sensor_item.setFont(column, font)
            self.heating_readout_tree.addTopLevelItem(sensor_item)
            for heater_id in self._associated_heater_ids_for_sensor(sensor_id):
                heater = self.model.nodes.get(int(heater_id)) if self.model is not None else None
                if heater is None or not bool(getattr(heater, "is_heater", False)):
                    continue
                heater_temperature = float(temperatures.get(int(heater_id), heater.initial_temperature_K))
                power = self._heater_readout_power_for_sensor_heater(sensor_id, int(heater_id), heater_powers)
                heater_item = self.QtWidgets.QTreeWidgetItem(
                    [
                        "heater",
                        str(heater_id),
                        _format_temperature(heater_temperature),
                        "",
                        "",
                        _format_power(power),
                    ]
                )
                heater_item.setData(1, self.QtCore.Qt.UserRole, int(heater_id))
                sensor_item.addChild(heater_item)
                if selected_kind == "heater" and selected_node_id == int(heater_id):
                    restore_item = heater_item
            sensor_item.setExpanded(True)
        for column in range(6):
            self.heating_readout_tree.resizeColumnToContents(column)
        if restore_item is not None:
            self.heating_readout_tree.setCurrentItem(restore_item)
        self.heating_readout_tree.blockSignals(False)

    def _heater_readout_power_for_sensor_heater(
        self,
        sensor_id: int,
        heater_id: int,
        heater_powers: dict[int, float],
    ) -> float:
        if self.model is None:
            return 0.0
        if int(heater_id) in heater_powers:
            return float(heater_powers.get(int(heater_id), 0.0))
        sensor = self.model.nodes.get(int(sensor_id))
        heater = self.model.nodes.get(int(heater_id))
        if sensor is None or heater is None:
            return 0.0
        if not self._sensor_enabled_for_simulation(int(sensor_id)) or not self._heater_enabled_for_simulation(int(heater_id)):
            return 0.0
        heater_mode = str(getattr(heater, "sensor_control_mode", "manual"))
        if heater_mode != "manual":
            return 0.0
        max_power = max(
            0.0,
            float(getattr(getattr(heater, "heater", None), "heater_max_power_W", 0.0))
            * float(getattr(getattr(heater, "heater", None), "heater_efficiency", 1.0)),
        )
        manual_power = float(getattr(heater, "sensor_manual_power_W", 0.0))
        if manual_power == 0.0 and _readout_heater_controller_is_default(heater):
            manual_power = float(getattr(sensor, "sensor_manual_power_W", 0.0))
        return min(max(manual_power, 0.0), max_power)

    def _show_readout_sensor_editor(self, sensor_id: int, *, selected_node_id: int | None = None) -> None:
        if self.model is None or int(sensor_id) not in self.model.nodes:
            self._hide_readout_parameter_editor()
            return
        sensor = self.model.nodes[int(sensor_id)]
        self._readout_editor_syncing = True
        try:
            self._readout_editor_kind = "sensor"
            self._readout_editor_sensor_id = int(sensor_id)
            self._readout_editor_node_id = int(selected_node_id if selected_node_id is not None else sensor_id)
            self.readout_editor_box.setVisible(True)
            self.readout_sensor_editor.setVisible(True)
            self.readout_heater_editor.setVisible(False)
            self.readout_cooling_editor.setVisible(False)
            self.readout_editor_title.setText(f"Sensor {int(sensor_id)}")
            for field in _READOUT_SENSOR_CONTROLLER_FIELDS:
                widget = self.readout_editor_inputs.get(field)
                if widget is not None:
                    widget.setValue(float(getattr(sensor, field, 0.0)))
        finally:
            self._readout_editor_syncing = False

    def _show_readout_heater_editor(self, heater_id: int) -> None:
        if self.model is None or int(heater_id) not in self.model.nodes:
            self._hide_readout_parameter_editor()
            return
        heater = self.model.nodes[int(heater_id)]
        self._readout_editor_syncing = True
        try:
            self._readout_editor_kind = "heater"
            self._readout_editor_node_id = int(heater_id)
            self._readout_editor_sensor_id = (
                int(getattr(heater, "assigned_sensor_id"))
                if getattr(heater, "assigned_sensor_id", None) is not None
                else None
            )
            self.readout_editor_box.setVisible(True)
            self.readout_sensor_editor.setVisible(False)
            self.readout_heater_editor.setVisible(True)
            self.readout_cooling_editor.setVisible(False)
            title = f"Heater {int(heater_id)}"
            if self._readout_editor_sensor_id is not None:
                title = f"Heater {int(heater_id)} controlled by sensor {self._readout_editor_sensor_id}"
            self.readout_editor_title.setText(title)
            mode = "mimo" if str(getattr(heater, "sensor_control_mode", "manual")) == "mimo" else "manual"
            self.readout_editor_inputs["sensor_control_mode"].setCurrentText(mode)
            heater_props = getattr(heater, "heater", None)
            for field in _READOUT_HEATER_HARDWARE_FIELDS:
                widget = self.readout_editor_inputs.get(field)
                if widget is None:
                    continue
                value = getattr(heater_props, field, 0)
                if field == "heater_id":
                    widget.setValue(int(value or heater.node_id))
                else:
                    widget.setValue(float(value))
            for field in _READOUT_HEATER_CONTROLLER_FIELDS:
                widget = self.readout_editor_inputs.get(field)
                if widget is not None:
                    widget.setValue(float(getattr(heater, field, 0.0)))
            self._sync_readout_heater_editor_enabled()
        finally:
            self._readout_editor_syncing = False

    def _show_readout_cooling_editor(self, node_id: int) -> None:
        self._readout_editor_syncing = True
        try:
            self._readout_editor_kind = "cooling"
            self._readout_editor_node_id = int(node_id)
            self._readout_editor_sensor_id = None
            self.readout_editor_box.setVisible(True)
            self.readout_sensor_editor.setVisible(False)
            self.readout_heater_editor.setVisible(False)
            self.readout_cooling_editor.setVisible(True)
            identifier = str(getattr(self.model.nodes.get(int(node_id)), "cryocooler_id", "") or "").strip()
            if not identifier and self.model.nodes.get(int(node_id)) is not None:
                identifier = str(getattr(self.model.nodes[int(node_id)], "component_name", "") or int(node_id))
            self.readout_editor_title.setText(f"Cryocooler {identifier}")
            for field in ("cryocooler_max_power_W", "cryocooler_capacity_scale", "cryocooler_enabled"):
                widget = self.readout_editor_inputs.get(field)
                if widget is not None and hasattr(widget, "setValue"):
                    widget.setValue(float(getattr(self.params, field)))
                elif widget is not None and hasattr(widget, "setChecked"):
                    widget.setChecked(bool(getattr(self.params, field)))
        finally:
            self._readout_editor_syncing = False

    def _hide_readout_parameter_editor(self) -> None:
        self._readout_editor_kind = None
        self._readout_editor_node_id = None
        self._readout_editor_sensor_id = None
        if hasattr(self, "readout_editor_box"):
            self.readout_editor_box.setVisible(False)
        if hasattr(self, "readout_heater_editor"):
            self.readout_heater_editor.setVisible(False)

    def _sync_readout_sensor_editor_enabled(self) -> None:
        widget = self.readout_editor_inputs.get("controller_setpoint_K")
        if widget is not None:
            widget.setEnabled(True)

    def _sync_readout_heater_editor_enabled(self) -> None:
        mode_widget = self.readout_editor_inputs.get("sensor_control_mode")
        mode = mode_widget.currentText() if mode_widget is not None else "manual"
        manual = str(mode) == "manual"
        for field in _READOUT_HEATER_CONTROLLER_FIELDS:
            widget = self.readout_editor_inputs.get(field)
            if widget is None:
                continue
            if field == "sensor_manual_power_W":
                widget.setEnabled(manual)
            else:
                widget.setEnabled(not manual)

    def _apply_readout_sensor_editor_change(self, field: str) -> None:
        if self._readout_editor_syncing or self.model is None or self._readout_editor_sensor_id is None:
            return
        sensor = self.model.nodes.get(int(self._readout_editor_sensor_id))
        if sensor is None:
            return
        widget = self.readout_editor_inputs.get(field)
        if widget is None:
            return
        setattr(sensor, field, float(widget.value()))
        if self.prepared is not None:
            self.prepared.mark_controller_stale()
            self.prepared.reset_controller_integrators()
        self._simulation_reinitialize_pending = False
        self._refresh_stats()
        self._refresh_sensor_readouts()
        self._show_readout_sensor_editor(
            int(self._readout_editor_sensor_id),
            selected_node_id=self._readout_editor_node_id,
        )
        self._status(f"Updated controller parameters for sensor {int(sensor.node_id)}.")

    def _apply_readout_heater_editor_change(self, field: str) -> None:
        if self._readout_editor_syncing or self.model is None or self._readout_editor_node_id is None:
            return
        heater = self.model.nodes.get(int(self._readout_editor_node_id))
        if heater is None:
            return
        widget = self.readout_editor_inputs.get(field)
        if widget is None:
            return
        if field == "sensor_control_mode":
            value = "mimo" if widget.currentText() == "mimo" else "manual"
            heater.sensor_control_mode = value
            self._sync_readout_heater_editor_enabled()
        elif field in _READOUT_HEATER_HARDWARE_FIELDS:
            heater_props = getattr(heater, "heater", None)
            if heater_props is None:
                return
            if field == "heater_id":
                setattr(heater_props, field, int(widget.value()))
            else:
                setattr(heater_props, field, float(widget.value()))
        else:
            setattr(heater, field, float(widget.value()))
        if self.prepared is not None:
            self.prepared.mark_controller_stale()
            self.prepared.reset_controller_integrators()
        self._simulation_reinitialize_pending = False
        self._refresh_stats()
        self._refresh_sensor_readouts()
        self._show_readout_heater_editor(int(heater.node_id))
        self._status(f"Updated controller parameters for heater {int(heater.node_id)}.")

    def _apply_readout_cooling_editor_change(self, field: str) -> None:
        if self._readout_editor_syncing:
            return
        widget = self.readout_editor_inputs.get(field)
        if widget is None or not hasattr(self.params, field):
            return
        linked = self.inputs.get(field)
        if linked is not None and hasattr(linked, "setValue") and hasattr(widget, "value"):
            linked.blockSignals(True)
            linked.setValue(float(widget.value()))
            linked.blockSignals(False)
            self._handle_parameter_change(field)
        elif linked is not None and hasattr(linked, "setChecked") and hasattr(widget, "isChecked"):
            linked.blockSignals(True)
            linked.setChecked(bool(widget.isChecked()))
            linked.blockSignals(False)
            self._handle_parameter_change(field)
        else:
            value = bool(widget.isChecked()) if hasattr(widget, "isChecked") else float(widget.value())
            self.params = replace(self.params, **{field: value})
            self._save_params_to_folder()
        if self._readout_editor_node_id is not None:
            self._show_readout_cooling_editor(int(self._readout_editor_node_id))

    def _heating_sensor_nodes(self) -> list[Any]:
        if self.model is None:
            return []
        sensor_ids = {
            int(node_id)
            for node_id, node in self.model.nodes.items()
            if bool(getattr(node, "is_sensor", False))
            and (
                bool(getattr(node, "assigned_heater_ids", []) or [])
                or getattr(node, "assigned_heater_id", None) is not None
            )
        }
        for node in self.model.nodes.values():
            if bool(getattr(node, "is_heater", False)) and getattr(node, "assigned_sensor_id", None) is not None:
                sensor_ids.add(int(getattr(node, "assigned_sensor_id")))
        return [
            self.model.nodes[sensor_id]
            for sensor_id in sorted(sensor_ids)
            if sensor_id in self.model.nodes and bool(getattr(self.model.nodes[sensor_id], "is_sensor", False))
        ]

    def _associated_heater_ids_for_sensor(self, sensor_id: int) -> list[int]:
        if self.model is None or int(sensor_id) not in self.model.nodes:
            return []
        sensor = self.model.nodes[int(sensor_id)]
        heater_ids = {
            int(value)
            for value in getattr(sensor, "assigned_heater_ids", []) or []
            if int(value) in self.model.nodes
        }
        if getattr(sensor, "assigned_heater_id", None) is not None:
            heater_ids.add(int(getattr(sensor, "assigned_heater_id")))
        # Heaters that target this sensor, from the precomputed reverse map (avoids
        # rescanning every node once per sensor per frame).
        for heater_id in self._cached_heaters_by_sensor().get(int(sensor_id), ()):
            if int(heater_id) in self.model.nodes:
                heater_ids.add(int(heater_id))
        return sorted(heater_ids)

    def _sensor_measured_temperature(
        self,
        sensor_id: int,
        temperatures: dict[int, float],
        node_index: dict[int, int],
    ) -> float:
        if self.prepared is not None and self.model is not None:
            return float(
                sensor_readout_temperature_K(
                    self.model,
                    node_index,
                    self.prepared.temperatures_K,
                    int(sensor_id),
                )
            )
        if self.model is None or int(sensor_id) not in self.model.nodes:
            return float("nan")
        node = self.model.nodes[int(sensor_id)]
        return float(temperatures.get(int(sensor_id), node.initial_temperature_K))

    def _handle_cooling_table_selection(self) -> None:
        row = self.cooling_readout_table.currentRow()
        if row < 0:
            return
        id_item = self.cooling_readout_table.item(row, 0)
        if id_item is None:
            return
        node_id = id_item.data(self.QtCore.Qt.UserRole)
        if node_id is None:
            return
        self._handle_pick(int(node_id))
        self._show_readout_cooling_editor(int(node_id))

    def _handle_heating_tree_selection(self) -> None:
        item = self.heating_readout_tree.currentItem()
        if item is None:
            return
        node_id = item.data(1, self.QtCore.Qt.UserRole)
        if node_id is None:
            return
        self._handle_pick(int(node_id))
        if item.text(0) == "sensor":
            self._show_readout_sensor_editor(int(node_id))
            return
        self._show_readout_heater_editor(int(node_id))


    def _handle_visual_toggle(self, *_: Any) -> None:
        self._draw_current(reset_camera=False)

    def _handle_visual_control_changed(self, *_: Any) -> None:
        self._sync_view_controls_to_viewer()
        self.viewer.safe_render()

    def _sync_view_controls_to_viewer(self) -> None:
        if not hasattr(self, "viewer") or not hasattr(self, "opacity_slider"):
            return
        self.viewer.set_cell_opacity(float(self.opacity_slider.value()) / 100.0, render=False)
        self.viewer.set_depth_focus(
            self.depth_focus_toggle.isChecked(),
            float(self.depth_slider.value()) / 100.0,
            axis=self.depth_axis_combo.currentText().lower(),
            width=float(self.depth_width_slider.value()) / 100.0,
            render=False,
        )
        self.viewer.set_cross_section(
            self.cross_section_toggle.isChecked(),
            float(self.cross_section_slider.value()) / 100.0,
            axis=self.cross_section_axis_combo.currentText().lower(),
            render=False,
        )
        self._update_cross_section_value_label()

    def _update_cross_section_value_label(self) -> None:
        if not hasattr(self, "cross_section_value_label"):
            return
        coordinate = self.viewer.cross_section_coordinate()
        if coordinate is None:
            self.cross_section_value_label.setText(f"{self.cross_section_slider.value()}%")
        else:
            axis = self.cross_section_axis_combo.currentText().upper()
            self.cross_section_value_label.setText(f"{axis} = {coordinate:.4g} mm")

    def _handle_marker_toggle(self, *_: Any) -> None:
        self.viewer.update_io_marker_visibility(
            self.show_heaters.isChecked(),
            self.show_sensors.isChecked(),
            self.show_coolers.isChecked(),
        )

    def _set_warnings(self, warnings: list[str]) -> None:
        combined = list(warnings)
        if self.prepared is not None:
            combined.extend(self.prepared.controller_warnings)
        self.warning_label.setText("\n".join(combined[:8]))

    def _tooltip_for_node(self, node_id: int) -> str:
        if self.model is None or node_id not in self.model.nodes:
            return ""
        node = self.model.nodes[node_id]
        temperature = self.temperature_by_node.get(node_id, node.initial_temperature_K)
        return "\n".join(
            [
                f"Node {node_id}",
                f"part/component: {node.component_name or '?'}",
                f"material: {node.material}",
                f"temperature: {temperature:.3f} K / {temperature - 273.15:.3f} C",
                f"initial: {node.initial_temperature_K:.3f} K / {node.initial_temperature_K - 273.15:.3f} C",
                f"C: {node.C_J_K:.6g} J/K",
                f"mass: {node.mass_kg:.6g} kg",
                f"volume: {node.volume_m3:.6g} m^3",
                f"level: {node.level}",
                f"heater: {node.is_heater} id={node.heater.heater_id}",
                f"sensor: {node.is_sensor} id={node.sensor.sensor_id}",
                f"cryocooler: {node.has_cryocooler}",
                f"exposed: {node.is_exposed}",
                f"G_rad: {node.G_rad_W_K:.6g} W/K",
            ]
        )

    def _handle_pick(self, node_id: int, *_: Any) -> None:
        if self.on_select_node is not None:
            self.on_select_node(node_id)
        self._select_component_for_node(node_id)
        self.viewer.select_node(node_id)

    def _select_component_for_node(self, node_id: int) -> None:
        if self.model is None or node_id not in self.model.nodes:
            return
        node = self.model.nodes[node_id]
        component = node.component_name
        if not component:
            return
        index = self.component_combo.findText(component)
        if index < 0:
            self.component_combo.addItem(component)
            index = self.component_combo.findText(component)
        if index >= 0:
            self.component_combo.setCurrentIndex(index)
        self.component_temperature.blockSignals(True)
        self.component_temperature.setValue(float(node.initial_temperature_K))
        self.component_temperature.blockSignals(False)

    def _sync_component_options(self) -> None:
        self.component_combo.clear()
        if self.model is None:
            return
        self.component_combo.addItems(
            sorted({node.component_name for node in self.model.nodes.values() if node.component_name})
        )

    def _legend_text(self) -> str:
        return "3D legend: jet colormap, bottom right."

    def _pin_two_line_label(self, label: Any) -> None:
        """Lock a status label to a fixed two-line height so runtime messages of
        varying length can't change its size and shove the rest of the panel around.
        Text longer than two lines wraps then clips (is cut off), not expands."""
        label.setWordWrap(True)
        label.setAlignment(self.QtCore.Qt.AlignTop | self.QtCore.Qt.AlignLeft)
        # Two line-heights plus a little padding, from the label's own font metrics.
        two_lines = label.fontMetrics().lineSpacing() * 2 + 6
        label.setFixedHeight(int(two_lines))
        label.setSizePolicy(self.QtWidgets.QSizePolicy.Preferred, self.QtWidgets.QSizePolicy.Fixed)

    def _status(self, message: str, error: bool = False) -> None:
        if self.on_status is not None:
            self.on_status(message, error)
        else:
            self.warning_label.setText(message)

    def _checkbox(self, text: str, checked: bool, callback: Any | None = None) -> Any:
        widget = self.QtWidgets.QCheckBox(text)
        widget.setChecked(checked)
        if callback is not None:
            widget.stateChanged.connect(callback)
        return widget

    def _view_slider(self, minimum: int, maximum: int, value: int, callback: Any) -> Any:
        slider = self.QtWidgets.QSlider(self.QtCore.Qt.Horizontal)
        slider.setRange(int(minimum), int(maximum))
        slider.setValue(int(value))
        slider.setFixedWidth(110)
        slider.valueChanged.connect(callback)
        return slider

    def _section(self, title: str) -> tuple[Any, Any]:
        box = self.QtWidgets.QGroupBox(title)
        box.setStyleSheet("QGroupBox { font-weight: 700; margin-top: 8px; }")
        return box, self.QtWidgets.QFormLayout(box)

    def _add_double_parameter(
        self,
        form: Any,
        name: str,
        label: str,
        minimum: float,
        maximum: float,
        step: float,
    ) -> None:
        widget = self._double_spin(minimum, maximum, getattr(self.params, name), step)
        widget.valueChanged.connect(lambda *_args, field=name: self._handle_parameter_change(field))
        self.inputs[name] = widget
        form.addRow(label, widget)

    def _add_int_parameter(
        self,
        form: Any,
        name: str,
        label: str,
        minimum: int,
        maximum: int,
        step: int,
    ) -> None:
        widget = self._int_spin(minimum, maximum, int(getattr(self.params, name)), step)
        widget.valueChanged.connect(lambda *_args, field=name: self._handle_parameter_change(field))
        self.inputs[name] = widget
        form.addRow(label, widget)

    def _int_spin(self, minimum: int, maximum: int, value: int, step: int) -> Any:
        class NoWheelSpinBox(self.QtWidgets.QSpinBox):
            def wheelEvent(inner_self, event: Any) -> None:  # noqa: N802 - Qt override name.
                event.ignore()

        widget = NoWheelSpinBox()
        widget.setRange(int(minimum), int(maximum))
        widget.setSingleStep(int(step))
        widget.setValue(int(value))
        return widget

    def _double_spin(self, minimum: float, maximum: float, value: float, step: float) -> Any:
        class NoWheelDoubleSpinBox(self.QtWidgets.QDoubleSpinBox):
            def wheelEvent(inner_self, event: Any) -> None:  # noqa: N802 - Qt override name.
                event.ignore()

        widget = NoWheelDoubleSpinBox()
        widget.setDecimals(8)
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setValue(float(value))
        return widget


def _changed_parameter_names(before: SimulationParameters, after: SimulationParameters) -> set[str]:
    changed: set[str] = set()
    before_values = vars(before)
    after_values = vars(after)
    for name, after_value in after_values.items():
        before_value = before_values.get(name)
        if isinstance(before_value, (list, tuple)) or isinstance(after_value, (list, tuple)):
            if tuple(before_value or ()) != tuple(after_value or ()):
                changed.add(name)
            continue
        if before_value != after_value:
            changed.add(name)
    return changed


def _record_profile_ms(profile: dict[str, float] | None, key: str, start: float) -> None:
    if profile is None:
        return
    profile[key] = profile.get(key, 0.0) + (time.perf_counter() - start) * 1000.0


def _accumulate_profile_ms(target: dict[str, float], source: Any) -> None:
    if not isinstance(source, dict):
        return
    for key, value in source.items():
        if not key.endswith("_ms"):
            continue
        try:
            target[key] = target.get(key, 0.0) + float(value)
        except (TypeError, ValueError):
            continue


def _readout_heater_controller_is_default(heater: Any) -> bool:
    defaults = {
        "sensor_control_mode": "manual",
        "sensor_manual_power_W": 0.0,
        "controller_weight": 0.0,
        "sensor_settling_time_s": 0.0,
        "controller_kp_coarse": 0.0,
        "controller_ki_coarse": 0.0,
        "controller_kd_coarse": 0.0,
        "controller_kp_hold": 0.0,
        "controller_ki_hold": 0.0,
        "controller_kd_hold": 0.0,
    }
    for field, default in defaults.items():
        value = getattr(heater, field, default)
        if field == "sensor_control_mode":
            if str(value or "manual").strip().lower() == "mimo":
                return False
            continue
        try:
            if abs(float(value) - float(default)) > 1.0e-12:
                return False
        except (TypeError, ValueError):
            continue
    return True


def _format_temperature(value_K: float) -> str:
    try:
        value = float(value_K)
    except (TypeError, ValueError):
        return "invalid"
    if not np.isfinite(value):
        return "invalid"
    return f"{value:.3f} K / {value - 273.15:.3f} C"


def _format_error(value_K: float) -> str:
    try:
        value = float(value_K)
    except (TypeError, ValueError):
        return "invalid"
    if not np.isfinite(value):
        return "invalid"
    return f"{value:.3f} K"


def _format_power(value_W: float) -> str:
    try:
        value = float(value_W)
    except (TypeError, ValueError):
        return "invalid"
    if not np.isfinite(value):
        return "invalid"
    return f"{value:.3f} W"


def _run_modal_design_worker(
    C: np.ndarray,
    L: Any,
    Grad: np.ndarray,
    node_ids: np.ndarray,
    model: ThermalGraphModel,
    T_op_K: float,
    n_modes: int,
    r: int,
    effort_weight: float,
    integral_gain: float,
    out_path: str,
    graph_name: str,
    progress_holder: dict[str, str],
) -> Any:
    """Background job: reduce the plant + design the modal-LQR controller artifact.
    Writes progress messages into ``progress_holder`` (read by the poll timer)."""

    def _progress(message: str) -> None:
        progress_holder["message"] = message

    return design_modal_controller(
        C, L, Grad, node_ids, model,
        T_op_K=float(T_op_K), n_modes=int(n_modes), r=int(r),
        effort_weight=float(effort_weight), integral_gain=float(integral_gain),
        out_path=out_path, graph_name=graph_name, progress=_progress,
    )


def _run_stepper_diagnostic_worker(
    model: ThermalGraphModel,
    matrices: dict[str, Any],
    params: SimulationParameters,
    node_ids: np.ndarray,
    initial_temperatures_K: np.ndarray,
    current_temperatures_K: np.ndarray,
    current_time_s: float,
    current_stepper: str,
    current_elapsed_s: float,
    current_profile_ms: dict[str, float],
    output_dir: Path | None,
) -> dict[str, Any]:
    result = compare_current_state_to_expm_multiply(
        model,
        matrices,
        params,
        node_ids=node_ids,
        initial_temperatures_K=initial_temperatures_K,
        current_temperatures_K=current_temperatures_K,
        current_time_s=current_time_s,
        current_stepper=current_stepper,
        current_elapsed_s=current_elapsed_s,
    )
    result.implicit_profile_ms.update(current_profile_ms)
    saved_output_dir: str | None = None
    if output_dir is not None:
        saved_output_dir = str(save_current_state_comparison(result, output_dir))
    return {
        "mode": "current_state",
        "metrics": asdict(result.metrics),
        "implicit_profile_ms": dict(result.implicit_profile_ms),
        "reference_profile_ms": dict(result.reference_profile_ms),
        "implicit_warnings": list(result.implicit_warnings),
        "reference_warnings": list(result.reference_warnings),
        "output_dir": saved_output_dir,
    }


def _format_stepper_diagnostic_summary(result: dict[str, Any]) -> str:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return "Solver diagnostic complete."
    implicit_profile = result.get("implicit_profile_ms", {})
    reference_profile = result.get("reference_profile_ms", {})
    implicit_profile = implicit_profile if isinstance(implicit_profile, dict) else {}
    reference_profile = reference_profile if isinstance(reference_profile, dict) else {}
    output_dir = result.get("output_dir")
    mode = str(result.get("mode") or "")
    parts = [
        f"{metrics.get('implicit_stepper', 'current')} vs {metrics.get('reference_stepper', 'reference')}",
        (
            f"current time={float(metrics.get('worst_time_s', 0.0)):.6g} s, "
            f"nominal steps={int(metrics.get('steps', 0))}, nodes={int(metrics.get('node_count', 0))}, "
            f"dt={float(metrics.get('dt_s', 0.0)):.6g} s"
            if mode == "current_state"
            else f"steps={int(metrics.get('steps', 0))}, nodes={int(metrics.get('node_count', 0))}, dt={float(metrics.get('dt_s', 0.0)):.6g} s"
        ),
        (
            f"max abs error={float(metrics.get('max_abs_error_K', 0.0)):.6g} K, "
            f"mean abs={float(metrics.get('mean_abs_error_K', 0.0)):.6g} K, "
            f"RMSE={float(metrics.get('rmse_K', 0.0)):.6g} K"
        ),
        (
            f"final max={float(metrics.get('final_max_abs_error_K', 0.0)):.6g} K, "
            f"final RMSE={float(metrics.get('final_rmse_K', 0.0)):.6g} K, "
            f"relative Frobenius={float(metrics.get('relative_frobenius_error', 0.0)):.6g}"
        ),
        (
            f"worst node={int(metrics.get('worst_node_id', -1))} "
            f"at t={float(metrics.get('worst_time_s', 0.0)):.6g} s "
            f"(step {int(metrics.get('worst_step_index', 0))})"
        ),
        (
            f"solve time: current-last-step={float(metrics.get('implicit_elapsed_s', 0.0)):.3f} s, "
            f"reference={float(metrics.get('reference_elapsed_s', 0.0)):.3f} s"
            if mode == "current_state"
            else (
                f"solve time: implicit={float(metrics.get('implicit_elapsed_s', 0.0)):.3f} s, "
                f"reference={float(metrics.get('reference_elapsed_s', 0.0)):.3f} s"
            )
        ),
    ]
    substeps = implicit_profile.get("substeps")
    predicted_delta = implicit_profile.get("predicted_delta_K")
    if substeps is not None or predicted_delta is not None:
        details = []
        if substeps is not None:
            details.append(f"implicit substeps={int(float(substeps))}")
        if predicted_delta is not None:
            details.append(f"predicted dT={float(predicted_delta):.6g} K")
        parts.append(", ".join(details))
    if reference_profile:
        reference_ms = reference_profile.get("cpu_expm_multiply_ms")
        if reference_ms is not None:
            parts.append(f"reference expm_multiply={float(reference_ms):.1f} ms")
    if output_dir:
        parts.append(f"saved: {output_dir}")
    else:
        parts.append("matrices not saved")
    warnings = list(result.get("implicit_warnings") or []) + list(result.get("reference_warnings") or [])
    if warnings:
        parts.append("warnings: " + " | ".join(str(item) for item in warnings[:3]))
    return "\n".join(parts)


def _last_prepared_solver_name(prepared: PreparedSimulation) -> str:
    profile = getattr(prepared, "last_step_profile_ms", {}) or {}
    if "implicit_step_ms" in profile:
        return "implicit_gpu" if profile.get("implicit_backend_gpu", 0.0) >= 1.0 else "implicit_cpu"
    return "current"


def _run_simulation_worker_batch(
    prepared: PreparedSimulation,
    params: SimulationParameters,
    steps_requested: int,
    loop_playback: bool,
    cancel_event: threading.Event,
    profile_enabled: bool,
) -> dict[str, Any]:
    profile: dict[str, float] | None = {} if profile_enabled else None
    previous_temperatures = np.asarray(prepared.temperatures_K, dtype=float).copy()
    steps_completed = 0
    model_profile: dict[str, float] = {}
    step_loop_start = time.perf_counter()
    while not cancel_event.is_set():
        if prepared.time_s >= params.t_final_s:
            if loop_playback:
                prepared.reset()
            else:
                break
        prepared.step_forward()
        if profile is not None:
            _accumulate_profile_ms(model_profile, getattr(prepared, "last_step_profile_ms", None))
        steps_completed += 1
        if steps_completed >= max(1, int(steps_requested)):
            break
    if profile is not None:
        profile["step_loop_ms"] = (time.perf_counter() - step_loop_start) * 1000.0
        profile.update(model_profile)
    current_temperatures = np.asarray(prepared.temperatures_K, dtype=float)
    max_delta_K = (
        float(np.max(np.abs(current_temperatures - previous_temperatures)))
        if current_temperatures.size and previous_temperatures.size == current_temperatures.size
        else 0.0
    )
    # Compute all per-frame numerics HERE (worker thread) so the GUI thread only
    # has to draw and set labels. This includes the heavy readout work (MIMO
    # re-run for actuator powers, cryocooler diagnostics) that previously ran on
    # the main thread and blocked the parameter window.
    readout: dict[str, Any] = {}
    if steps_completed > 0 and not cancel_event.is_set():
        readout_start = time.perf_counter()
        try:
            readout = {
                "temperature_by_node": {
                    int(node_id): float(temperature)
                    for node_id, temperature in zip(prepared.node_ids, current_temperatures)
                },
                "power_balance": prepared.power_balance_W(),
                "heater_powers": dict(prepared.heater_actuator_power_by_node()),
                "cryocooler_diagnostics": prepared.cryocooler_diagnostics(),
            }
        except Exception:  # noqa: BLE001 - readout is best-effort; never fail the step
            readout = {}
        if profile is not None:
            profile["worker_readout_ms"] = (time.perf_counter() - readout_start) * 1000.0
    return {
        "steps_completed": int(steps_completed),
        "max_delta_K": float(max_delta_K),
        "done": bool(prepared.time_s >= params.t_final_s and not loop_playback),
        "cancelled": bool(cancel_event.is_set()),
        "profile": profile,
        "readout": readout,
    }


def _node_uses_mimo_controller(
    node: Any,
    *,
    heater_enabled: bool = True,
    sensor_enabled: bool = True,
) -> bool:
    if bool(getattr(node, "is_heater", False)):
        return (
            bool(heater_enabled)
            and str(getattr(node, "sensor_control_mode", "manual")) == "mimo"
            and (
                getattr(node, "assigned_sensor_id", None) is not None
                or str(getattr(getattr(node, "heater_control", None), "mode", "")) == "mimo"
            )
        )
    if bool(getattr(node, "is_sensor", False)):
        return (
            bool(sensor_enabled)
            and (
                bool(getattr(node, "assigned_heater_ids", []) or [])
                or getattr(node, "assigned_heater_id", None) is not None
                or bool(getattr(node, "is_heater", False))
            )
            and bool(getattr(node, "sensor_valid", True))
            and not bool(getattr(node, "sensor_monitor_only", False))
        )
    return False
