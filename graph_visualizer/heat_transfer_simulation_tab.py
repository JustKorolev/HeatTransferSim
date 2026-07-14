"""Qt tab for live octree heat-transfer simulation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:  # pragma: no cover - import path depends on the installed Qt binding.
    from PySide6 import QtGui
except Exception:  # pragma: no cover
    from qtpy import QtGui

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
from .sys_id_artifacts import (
    list_sys_id_gain_matrices,
    load_sys_id_gain_matrix,
    save_sys_id_gain_matrix,
    update_sys_id_gain_matrix,
)


QT_SLIDER_MAXIMUM = 2_147_483_647


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
        self.widget = self.QtWidgets.QWidget(parent)
        self.timer = self.QtCore.QTimer(self.widget)
        self.timer.timeout.connect(self.step_forward)
        self.sys_id_timer = self.QtCore.QTimer(self.widget)
        self.sys_id_timer.timeout.connect(self._step_sys_id)
        self.sys_id_state: dict[str, Any] | None = None
        self._build_layout()
        self.refresh_graph_list()

    def _build_layout(self) -> None:
        layout = self.QtWidgets.QVBoxLayout(self.widget)
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
        self.sensor_readout_box = self.QtWidgets.QGroupBox("Sensor / Heater / Cooler Readouts")
        sensor_layout = self.QtWidgets.QVBoxLayout(self.sensor_readout_box)
        self.sensor_readout_table = self.QtWidgets.QTableWidget(0, 6)
        self.sensor_readout_table.setHorizontalHeaderLabels(
            ["cell/node", "temperature", "net power", "cryocooler power", "desired temperature", "error"]
        )
        self.sensor_readout_table.verticalHeader().setVisible(False)
        self.sensor_readout_table.setEditTriggers(self.QtWidgets.QAbstractItemView.NoEditTriggers)
        self.sensor_readout_table.setSelectionBehavior(self.QtWidgets.QAbstractItemView.SelectRows)
        self.sensor_readout_table.setMaximumHeight(170)
        self.sensor_readout_table.itemSelectionChanged.connect(self._handle_sensor_table_selection)
        sensor_layout.addWidget(self.sensor_readout_table)
        self.sensor_readout_box.setVisible(False)
        form.addRow(self.sensor_readout_box)
        self.legend_label = self.QtWidgets.QLabel(self._legend_text())
        self.legend_label.setWordWrap(True)
        form.addRow(self.legend_label)

        self.viewer = GraphPyVistaWidget(
            self.widget,
            on_pick_node=self._handle_pick,
            tooltip_for_node=self._tooltip_for_node,
        )
        toggles = self.QtWidgets.QHBoxLayout()
        self.show_heaters = self._checkbox("Heaters", True, self._handle_marker_toggle)
        self.show_sensors = self._checkbox("Sensors", True, self._handle_marker_toggle)
        self.show_coolers = self._checkbox("Cryocoolers", True, self._handle_marker_toggle)
        toggles.addWidget(self.show_heaters)
        toggles.addWidget(self.show_sensors)
        toggles.addWidget(self.show_coolers)
        toggles.addStretch(1)
        layout.addLayout(toggles)
        self.viewer.set_toggles(
            False,
            False,
            self.show_heaters.isChecked(),
            self.show_sensors.isChecked(),
            self.show_coolers.isChecked(),
        )
        layout.addWidget(self.viewer.interactor, 1)

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
        self.inputs["loop_playback"] = self._checkbox(
            "Loop playback", self.params.loop_playback, self._handle_parameter_change
        )
        run_form.addRow(self.inputs["loop_playback"])
        self.input_mode = self.QtWidgets.QComboBox()
        self.input_mode.addItems(["zero", "heater_inputs"])
        self.input_mode.setCurrentText(self.params.input_mode)
        self.input_mode.currentTextChanged.connect(self._handle_parameter_change)
        run_form.addRow("input mode", self.input_mode)
        form.addRow(run_box)

        environment_box, environment_form = self._section("Environment")
        self._add_double_parameter(environment_form, "T_env_K", "ambient T K", 0.0, 1.0e6, 1.0)
        self.inputs["use_ambient_radiation"] = self._checkbox(
            "Use ambient radiation", self.params.use_ambient_radiation, self._handle_parameter_change
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
            self._handle_parameter_change,
        )
        mimo_form.addRow(self.inputs["mimo_freeze_integral_when_saturated"])
        form.addRow(mimo_box)

        display_box, display_form = self._section("Display")
        self.inputs["autoscale_temperature"] = self._checkbox(
            "Autoscale temperature", self.params.autoscale_temperature, self._handle_parameter_change
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
        self.model = self.current_model()
        self.folder = self.current_folder()
        self.matrices = build_matrices(self.model)
        self._load_params_from_folder()
        self._reset_enabled_io_from_params()
        self._refresh_sys_id_matrix_list()
        self._sync_component_options()
        self._reset_to_model_initial_temperatures()
        self._draw_current(reset_camera=True)
        self._refresh_sensor_readouts()
        self._status("Using current editor graph.")

    def load_selected_graph(self) -> None:
        name = self.graph_combo.currentText()
        if not name:
            self._status("No graph selected.", True)
            return
        try:
            self.folder = Path.cwd() / "graphs" / name
            self.model, self.matrices = load_graph_folder(self.folder)
            self._load_params_from_folder()
            self._reset_enabled_io_from_params()
            self._refresh_sys_id_matrix_list()
            self._sync_component_options()
            self._reset_to_model_initial_temperatures()
            self._draw_current(reset_camera=True)
            self._refresh_sensor_readouts()
            self._status(f"Loaded simulation graph {name}.")
        except Exception as exc:
            self._status(str(exc), True)

    def initialize_simulation(self) -> None:
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
            self._draw_current(reset_camera=False)
            self._refresh_stats()
            self._set_warnings(self.prepared.warnings)
        except Exception as exc:
            self._status(str(exc), True)

    def play(self) -> None:
        if self.prepared is None:
            self.initialize_simulation()
        if self.prepared is None:
            self.pause()
            self._status("Simulation did not initialize; playback was not started.", True)
            return
        interval = max(10, int(100.0 / max(float(self.params.playback_speed), 1.0e-9)))
        self.timer.start(interval)
        self.step_forward()

    def _refresh_matrices_for_run(self) -> None:
        if self.model is None:
            return
        if (
            EdgeMode.normalize(self.model.metadata.edge_mode) == EdgeMode.AUTO.value
            and all(node.center_mm is not None and node.size_mm is not None for node in self.model.nodes.values())
            and not has_generated_role_contact_edges(self.model)
        ):
            refresh_geometry_edges(self.model)
            refresh_radiation_from_exposed_faces(self.model)
        self.matrices = build_matrices(self.model)

    def pause(self) -> None:
        self.timer.stop()

    def shutdown(self) -> None:
        self.timer.stop()
        try:
            self.viewer.close()
        except Exception:
            pass

    def reset(self) -> None:
        if self.prepared is None:
            self.initialize_simulation()
            return
        self.timer.stop()
        self.prepared.reset()
        self._after_state_change()
        self._status("Simulation reset to initial temperatures.")

    def step_forward(self) -> None:
        if self.prepared is None:
            self.initialize_simulation()
            return
        if self.prepared.time_s >= self.params.t_final_s:
            if self.params.loop_playback:
                self.prepared.reset()
            else:
                self.pause()
                return
        previous_temperatures = np.asarray(self.prepared.temperatures_K, dtype=float).copy()
        self.prepared.step_forward()
        current_temperatures = np.asarray(self.prepared.temperatures_K, dtype=float)
        if current_temperatures.size and previous_temperatures.size == current_temperatures.size:
            max_delta_K = float(np.max(np.abs(current_temperatures - previous_temperatures)))
        else:
            max_delta_K = 0.0
        self._after_state_change()
        if self.timer.isActive():
            status = f"Playing simulation: t = {self.prepared.time_s:.3g} s, max dT/step = {max_delta_K:.3e} K."
            if max_delta_K <= 1.0e-12:
                status += " No temperature change is being produced by the current inputs/initial conditions."
            self._status(status)

    def step_backward(self) -> None:
        if self.prepared is None:
            return
        self.prepared.step_backward()
        self._after_state_change()

    def save_current_trajectory(self) -> None:
        if self.prepared is None or self.folder is None:
            self._status("Initialize a graph simulation before saving.", True)
            return
        name = "simulation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        target = save_trajectory(self.folder, name, self.prepared)
        self._save_params_to_folder(target / "simulation_parameters.json", include_initial_temperatures=True)
        self._status(f"Saved trajectory to {target}.")

    def apply_component_initial_temperature(self) -> None:
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
        if self.sys_id_state is not None:
            self.cancel_sys_id("Graph changed while sys ID was running; run cancelled.")
        if self.model is model:
            self.folder = folder
            self._sync_enabled_io_table()
            if self.prepared is not None:
                self.prepared.mark_controller_stale()
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
        if self.prepared is None:
            self._status("Initialize the simulation before resetting MIMO integrators.", True)
            return
        self.prepared.reset_controller_integrators()
        self._refresh_stats()
        self._refresh_sensor_readouts()
        self._status("MIMO controller integrators reset.")

    def run_simulation_sys_id_for_G_ctrl(self) -> None:
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

    def _after_state_change(self) -> None:
        assert self.prepared is not None
        self.temperature_by_node = {
            int(node_id): float(temp)
            for node_id, temp in zip(self.prepared.node_ids, self.prepared.temperatures_K)
        }
        self._update_colors()
        self._refresh_stats()
        self._refresh_sensor_readouts()
        self._sync_time_slider_to_history()

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

    def _handle_parameter_change(self, *_: Any) -> None:
        if self.sys_id_state is not None:
            self.cancel_sys_id("Simulation parameter changed while sys ID was running; run cancelled.")
            return
        self.params = self._read_params()
        self._save_params_to_folder()
        if self.prepared is not None:
            self.initialize_simulation()

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
        is_sensor = any(
            _node_uses_mimo_controller(node, sensor_enabled=self._sensor_enabled_for_simulation(int(node_id)))
            for node_id, node in self.model.nodes.items()
        )
        is_heater = any(
            _node_uses_mimo_controller(node, heater_enabled=self._heater_enabled_for_simulation(int(node_id)))
            for node_id, node in self.model.nodes.items()
        )
        return is_sensor and is_heater

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
                widget.setValue(float(getattr(self.params, key)))
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
        self.viewer.set_toggles(
            False,
            False,
            self.show_heaters.isChecked(),
            self.show_sensors.isChecked(),
            self.show_coolers.isChecked(),
        )
        self.viewer.selected_node_id = None
        self.viewer.draw(
            self.model,
            reset_camera=reset_camera,
            node_scalar_values=self._temperature_values(),
            scalar_cmap="jet",
            scalar_clim=self._temperature_clim(),
            scalar_bar_title="Temperature [K]",
        )
        self._refresh_stats()
        self._refresh_sensor_readouts()

    def _update_colors(self) -> None:
        updated = self.viewer.update_node_scalars(
            self._temperature_values(),
            scalar_clim=self._temperature_clim(),
        )
        if not updated and self.prepared is None:
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
            self.sensor_readout_table.setRowCount(0)
            return
        readout_nodes = [
            node
            for node in self.model.nodes.values()
            if node.is_sensor or node.is_heater or node.has_cryocooler
        ]
        self.sensor_readout_box.setVisible(bool(readout_nodes))
        self.sensor_readout_table.setRowCount(len(readout_nodes))
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
        for row, node in enumerate(sorted(readout_nodes, key=lambda item: item.node_id)):
            if node.is_sensor and self.prepared is not None:
                temperature = sensor_readout_temperature_K(
                    self.model,
                    node_index,
                    self.prepared.temperatures_K,
                    int(node.node_id),
                )
            else:
                temperature = float(temperatures.get(node.node_id, node.initial_temperature_K))
            id_item = self.QtWidgets.QTableWidgetItem(str(node.node_id))
            temp_item = self.QtWidgets.QTableWidgetItem(
                "invalid"
                if not np.isfinite(float(temperature))
                else f"{temperature:.3f} K / {temperature - 273.15:.3f} C"
            )
            power_text = (
                f"{float(heater_powers.get(node.node_id, 0.0)):.3f} W"
                if node.is_heater or node.has_cryocooler
                else ""
            )
            power_item = self.QtWidgets.QTableWidgetItem(power_text)
            cooler_text = (
                f"{float(cryocooler_powers.get(node.node_id, 0.0)):.3f} W"
                if node.has_cryocooler
                else ""
            )
            cooler_item = self.QtWidgets.QTableWidgetItem(cooler_text)
            desired_text = ""
            error_text = ""
            if (
                self._mimo_controller_should_run()
                and _node_uses_mimo_controller(
                    node,
                    sensor_enabled=self._sensor_enabled_for_simulation(int(node.node_id)),
                )
            ):
                desired_temperature = float(getattr(node, "controller_setpoint_K", 293.15))
                error = desired_temperature - temperature
                desired_text = f"{desired_temperature:.3f} K / {desired_temperature - 273.15:.3f} C"
                error_text = "invalid" if not np.isfinite(float(error)) else f"{error:.3f} K"
            desired_item = self.QtWidgets.QTableWidgetItem(desired_text)
            error_item = self.QtWidgets.QTableWidgetItem(error_text)
            id_item.setData(self.QtCore.Qt.UserRole, int(node.node_id))
            self.sensor_readout_table.setItem(row, 0, id_item)
            self.sensor_readout_table.setItem(row, 1, temp_item)
            self.sensor_readout_table.setItem(row, 2, power_item)
            self.sensor_readout_table.setItem(row, 3, cooler_item)
            self.sensor_readout_table.setItem(row, 4, desired_item)
            self.sensor_readout_table.setItem(row, 5, error_item)
        self.sensor_readout_table.resizeColumnsToContents()

    def _handle_sensor_table_selection(self) -> None:
        row = self.sensor_readout_table.currentRow()
        if row < 0:
            return
        id_item = self.sensor_readout_table.item(row, 0)
        if id_item is None:
            return
        node_id = id_item.data(self.QtCore.Qt.UserRole)
        if node_id is None:
            return
        self._handle_pick(int(node_id))

    def _handle_visual_toggle(self, *_: Any) -> None:
        self._draw_current(reset_camera=False)

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
        widget.valueChanged.connect(self._handle_parameter_change)
        self.inputs[name] = widget
        form.addRow(label, widget)

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
