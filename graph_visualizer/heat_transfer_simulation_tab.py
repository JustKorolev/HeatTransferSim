"""Qt tab for live octree heat-transfer simulation."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
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
from .models import EdgeMode, ThermalGraphModel
from .pyvista_widget import GraphPyVistaWidget
from .role_pairing import sensor_readout_temperature_K
from .simulation_model import PreparedSimulation, prepare_simulation, save_trajectory
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
    "input_mode",
}
_DISPLAY_PARAMETER_FIELDS = {
    "autoscale_temperature",
    "color_min_K",
    "color_max_K",
}
_CONTROLLER_PARAMETER_FIELDS = {
    "Kp_cooler",
    "P_cooler_max",
    "T_cooler_setpoint",
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
}
_CONTROLLER_RUNTIME_HOTSWAP_FIELDS = set(_CONTROLLER_PARAMETER_FIELDS)
_LIGHTWEIGHT_RUNTIME_PARAMETER_FIELDS = {
    "playback_speed",
    "loop_playback",
}
_NONBLOCKING_PARAMETER_FIELDS = _LIGHTWEIGHT_RUNTIME_PARAMETER_FIELDS | _DISPLAY_PARAMETER_FIELDS
_READOUT_SENSOR_CONTROLLER_FIELDS = (
    "sensor_manual_power_W",
    "controller_setpoint_K",
    "controller_weight",
    "sensor_settling_time_s",
    "controller_kp_coarse",
    "controller_ki_coarse",
    "controller_kd_coarse",
    "controller_kp_hold",
    "controller_ki_hold",
    "controller_kd_hold",
    "controller_lambda_order",
    "controller_mu_order",
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
    ) -> None:
        self.QtCore = qt.QtCore
        self.QtGui = QtGui
        self.QtWidgets = qt.QtWidgets
        self.current_model = current_model
        self.current_folder = current_folder
        self.on_select_node = on_select_node
        self.on_status = on_status
        self.on_controller_gain_matrix_changed = on_controller_gain_matrix_changed
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
        self.stepper_diagnostic_future: Future[dict[str, Any]] | None = None
        self.stepper_diagnostic_timer = self.QtCore.QTimer(self.widget)
        self.stepper_diagnostic_timer.timeout.connect(self._poll_stepper_diagnostic_worker)
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
        self.controls_scroll.setMinimumWidth(390)
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
        self.warning_label.setWordWrap(True)
        form.addRow(self.warning_label)
        self.stats_label = self.QtWidgets.QLabel("No simulation initialized.")
        self.stats_label.setWordWrap(True)
        form.addRow(self.stats_label)
        self.controller_status_label = self.QtWidgets.QLabel("")
        self.controller_status_label.setWordWrap(True)
        form.addRow(self.controller_status_label)
        self.sensor_readout_box = self.QtWidgets.QGroupBox("Thermal I/O Readouts")
        readout_layout = self.QtWidgets.QVBoxLayout(self.sensor_readout_box)
        self.cooling_readout_box = self.QtWidgets.QGroupBox("Cooling")
        cooling_layout = self.QtWidgets.QVBoxLayout(self.cooling_readout_box)
        self.cooling_readout_table = self.QtWidgets.QTableWidget(0, 3)
        self.cooling_readout_table.setHorizontalHeaderLabels(["cell/node", "temperature", "cryocooler power"])
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
        toggles.addWidget(self.show_heaters)
        toggles.addWidget(self.show_sensors)
        toggles.addWidget(self.show_coolers)
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
        self.viewer.set_toggles(
            False,
            False,
            self.show_heaters.isChecked(),
            self.show_sensors.isChecked(),
            self.show_coolers.isChecked(),
        )
        self._sync_view_controls_to_viewer()
        viewer_layout.addWidget(self.viewer.interactor, 1)
        layout.addWidget(self.readout_editor_box, 0, self.QtCore.Qt.AlignTop)
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
        form.addRow(run_box)

        environment_box, environment_form = self._section("Environment")
        self._add_double_parameter(environment_form, "T_env_K", "ambient T K", 0.0, 1.0e6, 1.0)
        self.inputs["use_ambient_radiation"] = self._checkbox(
            "Use ambient radiation",
            self.params.use_ambient_radiation,
            lambda *_: self._handle_parameter_change("use_ambient_radiation"),
        )
        environment_form.addRow(self.inputs["use_ambient_radiation"])
        form.addRow(environment_box)

        cooler_box, cooler_form = self._section("Cryocooler")
        for name, label, minimum, maximum, step in (
            ("Kp_cooler", "Kp cooler W/K", 0.0, 1.0e9, 0.1),
            ("P_cooler_max", "max cooling W", 0.0, 1.0e9, 1.0),
            ("T_cooler_setpoint", "setpoint K", 0.0, 1.0e6, 1.0),
        ):
            self._add_double_parameter(cooler_form, name, label, minimum, maximum, step)
        form.addRow(cooler_box)

        mimo_box, mimo_form = self._section("MIMO Thermal-Rate QP")
        for name, label, minimum, maximum, step in (
            ("mimo_hold_threshold_K", "enter hold below K", 0.0, 1.0e6, 0.1),
            ("mimo_coarse_threshold_K", "return coarse above K", 0.0, 1.0e6, 0.1),
            ("mimo_default_heater_max_power_W", "default heater max W", 0.0, 1.0e9, 1.0),
            ("mimo_lambda_u", "lambda_u heater effort", 0.0, 1.0e9, 0.001),
            ("mimo_rho_du", "rho_du power change", 0.0, 1.0e9, 0.01),
            ("mimo_heater_slew_rate_W_per_s", "hard slew W/s", 0.0, 1.0e9, 1.0),
            ("mimo_v_cmd_abs_max_K_per_s", "max rate cmd K/s", 0.0, 1.0e9, 0.01),
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
        mode = self.QtWidgets.QComboBox()
        mode.addItems(["manual", "mimo"])
        mode.currentTextChanged.connect(lambda *_: self._apply_readout_sensor_editor_change("sensor_control_mode"))
        self.readout_editor_inputs["sensor_control_mode"] = mode
        sensor_form.addRow("mode", mode)
        for name, label, minimum, maximum, step in (
            ("sensor_manual_power_W", "manual power W", 0.0, 1.0e9, 1.0),
            ("controller_setpoint_K", "setpoint K", 0.0, 1.0e6, 1.0),
            ("controller_weight", "weight", 0.0, 1.0e9, 0.1),
            ("sensor_settling_time_s", "settling time s", 0.0, 1.0e9, 1.0),
            ("controller_kp_coarse", "coarse kP", 0.0, 1.0e9, 0.1),
            ("controller_ki_coarse", "coarse kI", 0.0, 1.0e9, 0.1),
            ("controller_kd_coarse", "coarse kD", 0.0, 1.0e9, 0.1),
            ("controller_kp_hold", "hold kP", 0.0, 1.0e9, 0.1),
            ("controller_ki_hold", "hold kI", 0.0, 1.0e9, 0.1),
            ("controller_kd_hold", "hold kD", 0.0, 1.0e9, 0.1),
            ("controller_lambda_order", "lambda", 0.0, 1.0e9, 0.1),
            ("controller_mu_order", "mu", 0.0, 1.0e9, 0.1),
        ):
            widget = self._double_spin(minimum, maximum, 0.0, step)
            widget.valueChanged.connect(lambda *_args, field=name: self._apply_readout_sensor_editor_change(field))
            self.readout_editor_inputs[name] = widget
            sensor_form.addRow(label, widget)
        layout.addWidget(self.readout_sensor_editor)

        self.readout_cooling_editor = self.QtWidgets.QWidget()
        cooling_form = self.QtWidgets.QFormLayout(self.readout_cooling_editor)
        for name, label, minimum, maximum, step in (
            ("Kp_cooler", "Kp cooler W/K", 0.0, 1.0e9, 0.1),
            ("P_cooler_max", "max cooling W", 0.0, 1.0e9, 1.0),
            ("T_cooler_setpoint", "setpoint K", 0.0, 1.0e6, 1.0),
        ):
            widget = self._double_spin(minimum, maximum, float(getattr(self.params, name)), step)
            widget.valueChanged.connect(lambda *_args, field=name: self._apply_readout_cooling_editor_change(field))
            self.readout_editor_inputs[name] = widget
            cooling_form.addRow(label, widget)
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

    def play(self) -> None:
        if self.prepared is None or self._simulation_reinitialize_pending:
            self.initialize_simulation()
        if self.prepared is None:
            self.pause()
            self._status("Simulation did not initialize; playback was not started.", True)
            return
        self.timer.start(self._playback_timer_interval_ms())
        self.step_forward()

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
        if self.prepared is None or self._simulation_reinitialize_pending:
            self.initialize_simulation()
            return
        self.timer.stop()
        self.prepared.reset()
        self._after_state_change()
        self._status("Simulation reset to initial temperatures.")

    def step_forward(self) -> None:
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
        ui_start = time.perf_counter()
        self._after_worker_state_change(profile)
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

    def _after_worker_state_change(self, profile: dict[str, float] | None) -> None:
        if profile is None:
            self._after_state_change()
            return
        try:
            self._after_state_change(profile)
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
        log_event("simulation live step profile", **fields)
        self._status(_format_live_step_profile(profile, steps_completed, max_delta_K))

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

    def _after_state_change(self, profile: dict[str, float] | None = None) -> None:
        assert self.prepared is not None
        start = time.perf_counter()
        self.temperature_by_node = {
            int(node_id): float(temp)
            for node_id, temp in zip(self.prepared.node_ids, self.prepared.temperatures_K)
        }
        _record_profile_ms(profile, "temperature_map_ms", start)
        start = time.perf_counter()
        self._update_colors()
        _record_profile_ms(profile, "color_update_render_ms", start)
        start = time.perf_counter()
        self._refresh_stats()
        _record_profile_ms(profile, "stats_refresh_ms", start)
        start = time.perf_counter()
        self._refresh_sensor_readouts()
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
        return SimulationParameters(
            dt_s=float(self.inputs["dt_s"].value()),
            t_final_s=float(self.inputs["t_final_s"].value()),
            playback_speed=float(self.inputs["playback_speed"].value()),
            use_ambient_radiation=bool(self.inputs["use_ambient_radiation"].isChecked()),
            T_env_K=float(self.inputs["T_env_K"].value()),
            input_mode=self.input_mode.currentText(),
            Kp_cooler=float(self.inputs["Kp_cooler"].value()),
            P_cooler_max=float(self.inputs["P_cooler_max"].value()),
            T_cooler_setpoint=float(self.inputs["T_cooler_setpoint"].value()),
            autoscale_temperature=bool(self.inputs["autoscale_temperature"].isChecked()),
            color_min_K=float(self.inputs["color_min_K"].value()),
            color_max_K=float(self.inputs["color_max_K"].value()),
            loop_playback=bool(self.inputs["loop_playback"].isChecked()),
            save_trajectory=bool(getattr(self.params, "save_trajectory", False)),
            gpu_simulation_enabled=True,
            gpu_simulation_max_substeps=int(getattr(self.params, "gpu_simulation_max_substeps", 128)),
            gpu_simulation_safety_factor=float(getattr(self.params, "gpu_simulation_safety_factor", 0.2)),
            fast_sparse_simulation_enabled=True,
            fast_sparse_simulation_max_substeps=int(getattr(self.params, "fast_sparse_simulation_max_substeps", 128)),
            fast_sparse_simulation_safety_factor=float(getattr(self.params, "fast_sparse_simulation_safety_factor", 0.2)),
            implicit_sparse_simulation_enabled=True,
            implicit_sparse_simulation_method=str(getattr(self.params, "implicit_sparse_simulation_method", "tr_bdf2")),
            implicit_sparse_simulation_rtol=float(getattr(self.params, "implicit_sparse_simulation_rtol", 1.0e-6)),
            implicit_sparse_simulation_maxiter=int(getattr(self.params, "implicit_sparse_simulation_maxiter", 300)),
            implicit_sparse_adaptive_substeps_enabled=bool(
                getattr(self.params, "implicit_sparse_adaptive_substeps_enabled", True)
            ),
            implicit_sparse_adaptive_target_delta_K=float(
                getattr(self.params, "implicit_sparse_adaptive_target_delta_K", 1.0)
            ),
            implicit_sparse_adaptive_max_substeps=int(
                getattr(self.params, "implicit_sparse_adaptive_max_substeps", 4)
            ),
            implicit_sparse_residual_check_enabled=bool(
                getattr(self.params, "implicit_sparse_residual_check_enabled", True)
            ),
            simulation_history_limit=int(self.inputs["simulation_history_limit"].value()),
            live_step_profiling_enabled=True,
            live_step_profile_threshold_ms=float(getattr(self.params, "live_step_profile_threshold_ms", 1000.0)),
            browser_simulation_size_warning=int(getattr(self.params, "browser_simulation_size_warning", 1000)),
            display_update_interval_ms=float(getattr(self.params, "display_update_interval_ms", 100.0)),
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
        )

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
            node_scalar_values=self._temperature_values(),
            scalar_cmap="jet",
            scalar_clim=self._temperature_clim(),
            scalar_bar_title="Temperature [K]",
        )
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

    def _refresh_stats(self) -> None:
        values = np.array(list((self.temperature_by_node or {}).values()), dtype=float)
        if values.size == 0 and self.model is not None:
            values = np.array([node.initial_temperature_K for node in self.model.nodes.values()], dtype=float)
        if values.size == 0:
            self.stats_label.setText("No simulation initialized.")
            self._refresh_sensor_readouts()
            return
        time_s = self.prepared.time_s if self.prepared is not None else 0.0
        self.stats_label.setText(
            f"t = {time_s:.3g} s\n"
            f"min = {np.min(values):.3f} K / {np.min(values) - 273.15:.3f} C\n"
            f"max = {np.max(values):.3f} K / {np.max(values) - 273.15:.3f} C\n"
            f"mean = {np.mean(values):.3f} K / {np.mean(values) - 273.15:.3f} C"
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
                    allocation_text = (
                        f"\nthermal-rate QP: sensors={diagnostics.get('active_sensor_count', '?')}, "
                        f"heaters={diagnostics.get('active_heater_count', '?')}, "
                        f"||v_cmd||={float(diagnostics.get('rate_command_norm', 0.0)):.4g} K/s, "
                        f"||u||={float(diagnostics.get('heater_command_norm', 0.0)):.4g}, "
                        f"rate_resid={float(diagnostics.get('predicted_dTdt_residual_norm', 0.0)):.4g} K/s"
                    )
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

    def _refresh_sensor_readouts(self) -> None:
        if self.model is None:
            self.sensor_readout_box.setVisible(False)
            self.cooling_readout_table.setRowCount(0)
            self.heating_readout_tree.clear()
            return
        cooling_nodes = [
            node
            for node in self.model.nodes.values()
            if node.has_cryocooler
        ]
        heating_sensors = self._heating_sensor_nodes()
        self.sensor_readout_box.setVisible(bool(cooling_nodes or heating_sensors))
        self.cooling_readout_box.setVisible(bool(cooling_nodes))
        self.heating_readout_box.setVisible(bool(heating_sensors))
        temperatures = self._temperature_values()
        sys_id_heater_powers = self._sys_id_readout_heater_powers()
        heater_powers = (
            sys_id_heater_powers
            if sys_id_heater_powers is not None
            else self.prepared.heater_actuator_power_by_node()
            if self.prepared is not None
            else {}
        )
        cryocooler_powers = self.prepared.cryocooler_power_by_node() if self.prepared is not None else {}
        node_index = (
            {int(node_id): row for row, node_id in enumerate(self.prepared.node_ids)}
            if self.prepared is not None
            else {}
        )
        self._refresh_cooling_readouts(cooling_nodes, temperatures, cryocooler_powers)
        self._refresh_heating_readouts(heating_sensors, temperatures, heater_powers, node_index)

    def _refresh_cooling_readouts(
        self,
        cooling_nodes: list[Any],
        temperatures: dict[int, float],
        cryocooler_powers: dict[int, float],
    ) -> None:
        self.cooling_readout_table.setRowCount(len(cooling_nodes))
        for row, node in enumerate(sorted(cooling_nodes, key=lambda item: item.node_id)):
            temperature = float(temperatures.get(int(node.node_id), node.initial_temperature_K))
            id_item = self.QtWidgets.QTableWidgetItem(str(node.node_id))
            id_item.setData(self.QtCore.Qt.UserRole, int(node.node_id))
            self.cooling_readout_table.setItem(row, 0, id_item)
            self.cooling_readout_table.setItem(row, 1, self.QtWidgets.QTableWidgetItem(_format_temperature(temperature)))
            self.cooling_readout_table.setItem(
                row,
                2,
                self.QtWidgets.QTableWidgetItem(_format_power(float(cryocooler_powers.get(int(node.node_id), 0.0)))),
            )
        self.cooling_readout_table.resizeColumnsToContents()

    def _refresh_heating_readouts(
        self,
        heating_sensors: list[Any],
        temperatures: dict[int, float],
        heater_powers: dict[int, float],
        node_index: dict[int, int],
    ) -> None:
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
            sensor_item.setExpanded(True)
        for column in range(6):
            self.heating_readout_tree.resizeColumnToContents(column)

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
        if str(getattr(sensor, "sensor_control_mode", "manual")) != "manual":
            return 0.0
        max_power = max(
            0.0,
            float(getattr(getattr(heater, "heater", None), "heater_max_power_W", 0.0))
            * float(getattr(getattr(heater, "heater", None), "heater_efficiency", 1.0)),
        )
        return min(max(float(getattr(sensor, "sensor_manual_power_W", 0.0)), 0.0), max_power)

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
            self.readout_cooling_editor.setVisible(False)
            title = f"Sensor {int(sensor_id)}"
            if selected_node_id is not None and int(selected_node_id) != int(sensor_id):
                title = f"Heater {int(selected_node_id)} controlled by sensor {int(sensor_id)}"
            self.readout_editor_title.setText(title)
            mode = "mimo" if str(getattr(sensor, "sensor_control_mode", "manual")) == "mimo" else "manual"
            self.readout_editor_inputs["sensor_control_mode"].setCurrentText(mode)
            for field in _READOUT_SENSOR_CONTROLLER_FIELDS:
                widget = self.readout_editor_inputs.get(field)
                if widget is not None:
                    widget.setValue(float(getattr(sensor, field, 0.0)))
            self._sync_readout_sensor_editor_enabled()
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
            self.readout_cooling_editor.setVisible(True)
            self.readout_editor_title.setText(f"Cryocooler cell {int(node_id)}")
            for field in ("Kp_cooler", "P_cooler_max", "T_cooler_setpoint"):
                widget = self.readout_editor_inputs.get(field)
                if widget is not None:
                    widget.setValue(float(getattr(self.params, field)))
        finally:
            self._readout_editor_syncing = False

    def _hide_readout_parameter_editor(self) -> None:
        self._readout_editor_kind = None
        self._readout_editor_node_id = None
        self._readout_editor_sensor_id = None
        if hasattr(self, "readout_editor_box"):
            self.readout_editor_box.setVisible(False)

    def _sync_readout_sensor_editor_enabled(self) -> None:
        mode_widget = self.readout_editor_inputs.get("sensor_control_mode")
        mode = mode_widget.currentText() if mode_widget is not None else "manual"
        manual = str(mode) == "manual"
        for field in _READOUT_SENSOR_CONTROLLER_FIELDS:
            widget = self.readout_editor_inputs.get(field)
            if widget is None:
                continue
            if field == "sensor_manual_power_W":
                widget.setEnabled(manual)
            elif field == "controller_setpoint_K":
                widget.setEnabled(True)
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
        if field == "sensor_control_mode":
            value = "mimo" if widget.currentText() == "mimo" else "manual"
            sensor.sensor_control_mode = value
            self._sync_readout_sensor_editor_enabled()
        else:
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

    def _apply_readout_cooling_editor_change(self, field: str) -> None:
        if self._readout_editor_syncing:
            return
        widget = self.readout_editor_inputs.get(field)
        if widget is None or not hasattr(self.params, field):
            return
        linked = self.inputs.get(field)
        if linked is not None and hasattr(linked, "setValue"):
            linked.blockSignals(True)
            linked.setValue(float(widget.value()))
            linked.blockSignals(False)
            self._handle_parameter_change(field)
        else:
            self.params = replace(self.params, **{field: float(widget.value())})
            self._apply_controller_runtime_parameter_change({field})
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
        for heater in self.model.nodes.values():
            if bool(getattr(heater, "is_heater", False)) and getattr(heater, "assigned_sensor_id", None) == int(sensor_id):
                heater_ids.add(int(heater.node_id))
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
        if self.model is None:
            self._hide_readout_parameter_editor()
            return
        heater = self.model.nodes.get(int(node_id))
        sensor_id = getattr(heater, "assigned_sensor_id", None) if heater is not None else None
        if sensor_id is None:
            self._hide_readout_parameter_editor()
            return
        self._show_readout_sensor_editor(int(sensor_id), selected_node_id=int(node_id))


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


def _format_live_step_profile(profile: dict[str, float], steps_completed: int, max_delta_K: float) -> str:
    labels = {
        "step_loop_ms": "step loop",
        "model_solve_ms": "solve/controller",
        "controller_mode_check_ms": "controller check",
        "controller_mimo_ms": "MIMO controller",
        "controller_heater_power_ms": "heater controller",
        "zero_power_vector_ms": "zero power vector",
        "gpu_step_ms": "GPU step",
        "cpu_sparse_implicit_step_ms": "TR-BDF2 sparse step",
        "cpu_fast_sparse_step_ms": "fast sparse step",
        "state_vector_update_ms": "state vector",
        "radiation_source_ms": "radiation source",
        "source_vector_build_ms": "source vector",
        "affine_matrix_build_ms": "affine build",
        "cpu_expm_multiply_ms": "CPU expm_multiply",
        "dense_phi_matvec_ms": "dense Phi matvec",
        "dense_affine_matrix_build_ms": "dense affine build",
        "dense_expm_matvec_ms": "dense expm",
        "state_copy_ms": "state copy",
        "history_append_ms": "history",
        "temperature_map_ms": "temp map",
        "color_update_render_ms": "colors/render",
        "stats_refresh_ms": "stats",
        "sensor_readouts_ms": "sensors",
        "time_slider_ms": "slider",
        "seek_ms": "seek",
    }
    total_ms = float(profile.get("total_ms", 0.0))
    parts = [
        f"{label}={float(profile[key]):.1f} ms"
        for key, label in labels.items()
        if key in profile
    ]
    contributors = [
        (labels.get(key, key[:-3].replace("_", " ")), float(value))
        for key, value in profile.items()
        if key.endswith("_ms") and key != "total_ms"
    ]
    contributors.sort(key=lambda item: item[1], reverse=True)
    bottleneck = contributors[0][0] if contributors else "unknown"
    detail = ", ".join(parts)
    if detail:
        detail = " " + detail
    return (
        f"Live step profile: total={total_ms:.1f} ms, steps={int(steps_completed)}, "
        f"max dT={float(max_delta_K):.3e} K, largest={bottleneck}.{detail}"
    )


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
    if "gpu_step_ms" in profile:
        return "gpu_sparse"
    if "cpu_sparse_implicit_step_ms" in profile:
        return "implicit_sparse_cpu"
    if "cpu_fast_sparse_step_ms" in profile:
        return "fast_sparse_cpu"
    if "cpu_expm_multiply_ms" in profile:
        return "expm_multiply"
    if "dense_phi_matvec_ms" in profile:
        return "dense_phi_matvec"
    if "dense_expm_matvec_ms" in profile:
        return "dense_expm_matvec"
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
    return {
        "steps_completed": int(steps_completed),
        "max_delta_K": float(max_delta_K),
        "done": bool(prepared.time_s >= params.t_final_s and not loop_playback),
        "cancelled": bool(cancel_event.is_set()),
        "profile": profile,
    }


def _node_uses_mimo_controller(
    node: Any,
    *,
    heater_enabled: bool = True,
    sensor_enabled: bool = True,
) -> bool:
    if bool(getattr(node, "is_sensor", False)):
        return (
            bool(sensor_enabled)
            and (
                str(getattr(node, "sensor_control_mode", "manual")) == "mimo"
                or str(getattr(getattr(node, "heater_control", None), "mode", "")) == "mimo"
            )
            and (
                bool(getattr(node, "assigned_heater_ids", []) or [])
                or getattr(node, "assigned_heater_id", None) is not None
                or bool(getattr(node, "is_heater", False))
            )
            and bool(getattr(node, "sensor_valid", True))
            and not bool(getattr(node, "sensor_monitor_only", False))
        )
    if bool(getattr(node, "is_heater", False)):
        return bool(heater_enabled) and (
            getattr(node, "assigned_sensor_id", None) is not None
            or str(getattr(getattr(node, "heater_control", None), "mode", "")) == "mimo"
        )
    return False
