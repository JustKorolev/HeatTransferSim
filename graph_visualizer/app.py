"""Qt main window for constructing sparse 3D thermal graph networks."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import os
import re
from pathlib import Path
import tempfile
from typing import Any

from .draw_tools import (
    clone_node_for_extrusion,
    compute_face_normal,
    extrusion_count_from_projected_pixel_drag,
    next_node_id,
    preview_coords,
)
from .graph_io import (
    load_conductance_matrix_from_folder,
    load_graph_folder,
    save_graph_folder,
)
from .heat_transfer_simulation_tab import HeatTransferSimulationTab
from .material_library import default_material_library
from .matrix_builder import refresh_auto_edges
from .models import (
    EdgeMode,
    GraphMetadata,
    HeaterControl,
    HeaterProperties,
    ManualHeaterSettings,
    NodeProperties,
    PIDControlSettings,
    PIDState,
    SensorProperties,
    ThermalGraphModel,
)
from .pyvista_widget import GraphPyVistaWidget
from .role_assignment import assign_matching_nodes_to_role
from .tooltip_formatters import format_node_tooltip
from .two_d_graph_widget import TwoDGraphWidget
from .validation import raise_if_errors, validate_model


class GraphVisualizerApp:
    """Interactive Qt/PyVista editor for lumped thermal graph folders."""

    def __init__(self) -> None:
        self._load_qt()
        self.model = ThermalGraphModel(
            metadata=GraphMetadata(), material_library=default_material_library()
        )
        self.current_folder: Path | None = None
        self.autosave_enabled = False
        self.selected_node_id: int | None = None
        self.selected_node_ids: set[int] = set()
        self.inputs: dict[str, Any] = {}
        self._building_form = False
        self._syncing_conduction_ui = False
        self._syncing_metadata_ui = False
        self._applying_node_form = False
        self.dirty = False
        self.draw_mode_enabled = False
        self.draw_active = False
        self.draw_start_node_id: int | None = None
        self.draw_start_coord: tuple[int, int, int] | None = None
        self.draw_start_pixel: tuple[int, int] | None = None
        self.draw_normal_grid: tuple[int, int, int] | None = None
        self.draw_screen_direction: tuple[float, float] | None = None
        self.draw_pixels_per_cell: float | None = None
        self.draw_preview_coords: list[tuple[int, int, int]] = []
        self._last_shown_preview_coords: list[tuple[int, int, int]] = []
        self._suppress_next_draw_pick = False
        self.dark_mode = False

        self.app = self.QtWidgets.QApplication.instance() or self.QtWidgets.QApplication([])
        self.window = self.QtWidgets.QMainWindow()
        self.window.setWindowTitle("Graph Visualizer - Sparse Thermal Lump Network")
        self.window.closeEvent = self._handle_close_event
        self.autosave_timer = self.QtCore.QTimer(self.window)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.timeout.connect(self._autosave_now)
        self._build_layout()
        self._apply_theme()
        self._refresh_all(reset_camera=True)

    def _load_qt(self) -> None:
        try:
            from PySide6 import QtCore, QtWidgets
        except ImportError:
            try:
                from qtpy import QtCore, QtWidgets
            except ImportError as exc:
                raise RuntimeError(
                    "graph_visualizer requires PySide6 or another Qt binding. "
                    "Install UI dependencies later with pip install -r requirements.txt."
                ) from exc
        self.QtCore = QtCore
        self.QtWidgets = QtWidgets

    def _build_layout(self) -> None:
        central = self.QtWidgets.QWidget()
        layout = self.QtWidgets.QHBoxLayout(central)

        self.left_scroll = self.QtWidgets.QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setMinimumWidth(390)
        left_content = self.QtWidgets.QWidget()
        self.left_scroll.setWidget(left_content)
        self.left_layout = self.QtWidgets.QVBoxLayout(left_content)

        self._build_file_controls()
        self._build_global_controls()
        self._build_search_controls()
        self._build_filter_controls()
        self._build_node_form()
        self._build_bulk_role_assignment_controls()
        self._build_details_panel()
        self.left_layout.addStretch(1)

        right_panel = self.QtWidgets.QWidget()
        right_layout = self.QtWidgets.QVBoxLayout(right_panel)
        self.view_tabs = self.QtWidgets.QTabWidget()
        self.view_tabs.currentChanged.connect(self._handle_tab_changed)

        self.three_d_tab = self.QtWidgets.QWidget()
        three_d_layout = self.QtWidgets.QVBoxLayout(self.three_d_tab)
        toggles = self.QtWidgets.QHBoxLayout()
        self.show_labels = self._checkbox("Labels", False, self._handle_visual_toggle)
        self.show_edges = self._checkbox("Edges", False, self._handle_visual_toggle)
        self.show_heaters = self._checkbox("Heaters", True, self._handle_marker_toggle)
        self.show_sensors = self._checkbox("Sensors", True, self._handle_marker_toggle)
        self.show_coolers = self._checkbox("Cryocoolers", True, self._handle_marker_toggle)
        for widget in (
            self.show_labels,
            self.show_edges,
            self.show_heaters,
            self.show_sensors,
            self.show_coolers,
        ):
            toggles.addWidget(widget)
        toggles.addStretch(1)
        three_d_layout.addLayout(toggles)
        self.viewer = GraphPyVistaWidget(
            self.three_d_tab,
            on_pick_node=self._handle_viewer_pick,
            tooltip_for_node=self._tooltip_for_node,
        )
        three_d_layout.addWidget(self.viewer.interactor, 1)

        self.two_d_view = TwoDGraphWidget(right_panel, on_select_node=self.select_node)
        self.simulation_tab = HeatTransferSimulationTab(
            self,
            right_panel,
            current_model=lambda: self.model,
            current_folder=lambda: self.current_folder,
            on_select_node=self.select_node,
            on_status=self._set_status,
            on_controller_gain_matrix_changed=self._handle_controller_gain_matrix_changed,
        )
        self.view_tabs.addTab(self.three_d_tab, "3D Octree Graph Editor")
        self.view_tabs.addTab(self.two_d_view.widget, "2D Network Graph")
        self.view_tabs.addTab(self.simulation_tab.widget, "Heat Transfer Simulation")
        right_layout.addWidget(self.view_tabs, 1)

        self.status_label = self.QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        right_layout.addWidget(self.status_label)

        self.side_panel_stack = self.QtWidgets.QStackedWidget()
        self.side_panel_stack.addWidget(self.left_scroll)
        self.side_panel_stack.addWidget(self.simulation_tab.controls_scroll)
        self.side_panel_stack.setCurrentWidget(self.left_scroll)

        layout.addWidget(self.side_panel_stack, 0)
        layout.addWidget(right_panel, 1)
        self.window.setCentralWidget(central)
        self.window.resize(1380, 820)

    def _build_file_controls(self) -> None:
        row = self.QtWidgets.QHBoxLayout()
        for text, callback in (
            ("New", self.new_graph),
            ("Load", self.load_graph),
            ("Save", self.save_graph),
            ("Save As", self.save_graph_as),
        ):
            button = self.QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)
        self.theme_toggle = self._checkbox("Dark", self.dark_mode, self._handle_theme_toggle)
        row.addWidget(self.theme_toggle)
        self.left_layout.addLayout(row)

    def _build_global_controls(self) -> None:
        box = self._group_box("Graph Settings")
        form = self.QtWidgets.QFormLayout(box)
        self.graph_name_input = self.QtWidgets.QLineEdit(self.model.metadata.graph_name)
        self.graph_name_input.editingFinished.connect(self._handle_metadata_changed)
        self.T_sur_input = self._double_spin(0.0, 1.0e9, self.model.metadata.T_sur_K, 0.5)
        self.T_sur_input.valueChanged.connect(self._handle_metadata_changed)
        self.notes_input = self.QtWidgets.QPlainTextEdit()
        self.notes_input.setMaximumHeight(60)
        self.notes_input.textChanged.connect(self._handle_metadata_changed)
        self.auto_conduction_radio = self.QtWidgets.QRadioButton(
            "Auto-estimate G from geometry/materials"
        )
        self.loaded_conduction_radio = self.QtWidgets.QRadioButton(
            "Load/use G from matrices.npz"
        )
        self.conduction_button_group = self.QtWidgets.QButtonGroup(self.window)
        self.conduction_button_group.setExclusive(True)
        self.conduction_button_group.addButton(self.auto_conduction_radio)
        self.conduction_button_group.addButton(self.loaded_conduction_radio)
        self.auto_conduction_radio.setChecked(True)
        self.auto_conduction_radio.toggled.connect(self._handle_conduction_radio_change)
        self.loaded_conduction_radio.toggled.connect(self._handle_conduction_radio_change)
        form.addRow("name", self.graph_name_input)
        form.addRow("T_sur_K", self.T_sur_input)
        form.addRow("Conduction Model", self.auto_conduction_radio)
        form.addRow("", self.loaded_conduction_radio)
        form.addRow("notes", self.notes_input)
        row = self.QtWidgets.QHBoxLayout()
        recompute = self.QtWidgets.QPushButton("Recompute Auto Edges")
        recompute.clicked.connect(self.recompute_auto_edges)
        load_matrix = self.QtWidgets.QPushButton("Reload Loaded G")
        load_matrix.clicked.connect(self.load_matrix_edges)
        row.addWidget(recompute)
        row.addWidget(load_matrix)
        form.addRow(row)
        self.left_layout.addWidget(box)

    def _build_search_controls(self) -> None:
        box = self._group_box("Search")
        row = self.QtWidgets.QHBoxLayout(box)
        self.search_input = self.QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("node_id or (i, j, k)")
        search_button = self.QtWidgets.QPushButton("Find")
        search_button.clicked.connect(self.search_node)
        row.addWidget(self.search_input, 1)
        row.addWidget(search_button)
        self.left_layout.addWidget(box)

    def _build_filter_controls(self) -> None:
        box = self._group_box("Filters")
        form = self.QtWidgets.QFormLayout(box)
        self.filter_material = self.QtWidgets.QComboBox()
        self.filter_component = self.QtWidgets.QComboBox()
        self.filter_level_min = self._int_spin(0, 99, 0)
        self.filter_level_max = self._int_spin(0, 99, 99)
        self.filter_heater_sensor = self._checkbox("heater or sensor only", False, self._handle_visual_toggle)
        self.filter_boundary = self._checkbox("contact/boundary only", False, self._handle_visual_toggle)
        for combo in (self.filter_material, self.filter_component):
            combo.currentTextChanged.connect(self._handle_visual_toggle)
        self.filter_level_min.valueChanged.connect(self._handle_visual_toggle)
        self.filter_level_max.valueChanged.connect(self._handle_visual_toggle)
        form.addRow("material", self.filter_material)
        form.addRow("component", self.filter_component)
        form.addRow("min level", self.filter_level_min)
        form.addRow("max level", self.filter_level_max)
        form.addRow("", self.filter_heater_sensor)
        form.addRow("", self.filter_boundary)
        self.left_layout.addWidget(box)

    def _build_node_form(self) -> None:
        box = self._group_box("Selected Cell Tags")
        layout = self.QtWidgets.QVBoxLayout(box)

        form = self.QtWidgets.QFormLayout()
        self.inputs["part_code"] = self.QtWidgets.QLineEdit()
        self.inputs["part_code"].setReadOnly(True)
        self.inputs["part_code"].setStyleSheet("font-weight: 700;")
        self.inputs["component_name"] = self.QtWidgets.QLineEdit()
        self.inputs["component_name"].setReadOnly(True)
        self.inputs["node_id"] = self._int_spin(-10**9, 10**9, 0)
        for axis in ("i", "j", "k"):
            self.inputs[f"coord_{axis}"] = self._int_spin(-10**6, 10**6, 0)
        self.inputs["side_length_m"] = self._double_spin(1.0e-9, 1.0e9, 1.0, 0.1)
        self.inputs["material"] = self.QtWidgets.QComboBox()
        self.inputs["material"].addItems(sorted(self.model.material_library))
        self.inputs["material"].currentTextChanged.connect(self._apply_selected_material)
        self.inputs["rho_kg_m3"] = self._double_spin(0.0, 1.0e12, 2200.0, 10.0)
        self.inputs["cp_J_kgK"] = self._double_spin(0.0, 1.0e9, 800.0, 10.0)
        self.inputs["k_W_mK"] = self._double_spin(0.0, 1.0e9, 2.0, 1.0)
        self.inputs["emissivity"] = self._double_spin(0.0, 1.0, 0.85, 0.05)
        self.inputs["mass_kg"] = self._double_spin(0.0, 1.0e9, 1.0, 0.1)
        self.inputs["C_manual_override"] = self._checkbox("", False, self._update_C_enabled)
        self.inputs["C_J_K"] = self._double_spin(0.0, 1.0e15, 800.0, 10.0)
        self.inputs["Grad_W_K"] = self._double_spin(0.0, 1.0e12, 0.0, 0.01)
        self.inputs["initial_temperature_K"] = self._double_spin(0.0, 1.0e6, 293.15, 1.0)
        self.inputs["notes"] = self.QtWidgets.QPlainTextEdit()
        self.inputs["notes"].setMaximumHeight(64)
        for field_name in ("mass_kg", "cp_J_kgK"):
            self.inputs[field_name].valueChanged.connect(self._auto_update_C)

        for label, key in (
            ("part code", "part_code"),
            ("component", "component_name"),
            ("node_id", "node_id"),
            ("coord i", "coord_i"),
            ("coord j", "coord_j"),
            ("coord k", "coord_k"),
            ("side_length_m", "side_length_m"),
            ("material", "material"),
            ("rho_kg_m3", "rho_kg_m3"),
            ("cp_J_kgK", "cp_J_kgK"),
            ("k_W_mK", "k_W_mK"),
            ("emissivity", "emissivity"),
            ("mass_kg", "mass_kg"),
            ("manual C", "C_manual_override"),
            ("C_J_K", "C_J_K"),
            ("Grad_W_K", "Grad_W_K"),
            ("initial_temperature_K", "initial_temperature_K"),
            ("notes", "notes"),
        ):
            form.addRow(label, self.inputs[key])

        for key in (
            "node_id",
            "coord_i",
            "coord_j",
            "coord_k",
            "side_length_m",
            "material",
            "rho_kg_m3",
            "cp_J_kgK",
            "k_W_mK",
            "emissivity",
            "mass_kg",
            "C_manual_override",
            "C_J_K",
            "Grad_W_K",
        ):
            self.inputs[key].setEnabled(False)

        self.inputs["role"] = self.QtWidgets.QComboBox()
        self.inputs["role"].addItems(["Body", "Heater", "Sensor"])
        self.inputs["role"].currentTextChanged.connect(self._handle_role_changed)
        self.inputs["is_heater"] = self._checkbox("is heater", False, self._update_optional_sections)
        self.inputs["is_heater"].setVisible(False)
        self.node_role_label = self.QtWidgets.QLabel("role: body cell")
        self.node_role_label.setWordWrap(True)
        self.heater_box = self._group_box("Heater")
        heater_form = self.QtWidgets.QFormLayout(self.heater_box)
        self.inputs["heater_id"] = self._int_spin(-1, 10**9, 0)
        self.inputs["heater_min_power_W"] = self._double_spin(0.0, 1.0e9, 0.0, 1.0)
        self.inputs["heater_max_power_W"] = self._double_spin(0.0, 1.0e9, 30.0, 1.0)
        self.inputs["heater_efficiency"] = self._double_spin(0.0, 1.0e6, 1.0, 0.05)
        for key in ("heater_id", "heater_min_power_W", "heater_max_power_W", "heater_efficiency"):
            heater_form.addRow(key, self.inputs[key])
        self.inputs["heater_mode_pid"] = self.QtWidgets.QRadioButton("PID control (per cell)")
        self.inputs["heater_mode_mimo"] = self.QtWidgets.QRadioButton("MIMO PID")
        self.inputs["heater_mode_manual"] = self.QtWidgets.QRadioButton("Manual override")
        self.heater_mode_group = self.QtWidgets.QButtonGroup(self.window)
        self.heater_mode_group.setExclusive(True)
        self.heater_mode_group.addButton(self.inputs["heater_mode_pid"])
        self.heater_mode_group.addButton(self.inputs["heater_mode_mimo"])
        self.heater_mode_group.addButton(self.inputs["heater_mode_manual"])
        self.inputs["heater_mode_manual"].setChecked(True)
        self.inputs["heater_mode_pid"].toggled.connect(self._handle_heater_mode_change)
        self.inputs["heater_mode_mimo"].toggled.connect(self._handle_heater_mode_change)
        self.inputs["heater_mode_manual"].toggled.connect(self._handle_heater_mode_change)
        heater_form.addRow("control", self.inputs["heater_mode_pid"])
        heater_form.addRow("", self.inputs["heater_mode_mimo"])
        heater_form.addRow("", self.inputs["heater_mode_manual"])
        self.inputs["heater_pid_kp"] = self._double_spin(-1.0e12, 1.0e12, 0.0, 0.1)
        self.inputs["heater_pid_ki"] = self._double_spin(-1.0e12, 1.0e12, 0.0, 0.1)
        self.inputs["heater_pid_kd"] = self._double_spin(-1.0e12, 1.0e12, 0.0, 0.1)
        self.inputs["heater_pid_lambda_order"] = self._double_spin(0.0, 2.0, 1.0, 0.05)
        self.inputs["heater_pid_mu_order"] = self._double_spin(0.0, 2.0, 1.0, 0.05)
        self.inputs["heater_pid_setpoint"] = self._double_spin(0.0, 1.0e6, 293.15, 1.0)
        self.inputs["heater_manual_power"] = self._double_spin(0.0, 1.0e9, 0.0, 1.0)
        self.inputs["controller_setpoint_K"] = self._double_spin(0.0, 1.0e6, 293.15, 1.0)
        self.inputs["controller_weight"] = self._double_spin(0.0, 1.0e9, 0.0, 0.1)
        self.inputs["sensor_settling_time_s"] = self._double_spin(0.0, 1.0e9, 0.0, 0.1)
        self.inputs["controller_kp_coarse"] = self._double_spin(0.0, 1.0e9, 0.0, 0.1)
        self.inputs["controller_ki_coarse"] = self._double_spin(0.0, 1.0e9, 0.0, 0.01)
        self.inputs["controller_kp_hold"] = self._double_spin(0.0, 1.0e9, 0.0, 0.1)
        self.inputs["controller_ki_hold"] = self._double_spin(0.0, 1.0e9, 0.0, 0.01)
        self.inputs["controller_kd_coarse"] = self._double_spin(0.0, 1.0e9, 0.0, 0.1)
        self.inputs["controller_kd_hold"] = self._double_spin(0.0, 1.0e9, 0.0, 0.1)
        self.inputs["controller_lambda_order"] = self._double_spin(0.0, 2.0, 1.0, 0.05)
        self.inputs["controller_mu_order"] = self._double_spin(0.0, 2.0, 1.0, 0.05)
        self.inputs["sensor_settling_time_s"].setToolTip(
            "MIMO sensor settling time used for lag compensation. For a first-order sensor, settling time is roughly 5*tau."
        )
        self.inputs["heater_pid_setpoint"].valueChanged.connect(self._handle_heater_setpoint_change)
        for key in (
            "heater_pid_kp",
            "heater_pid_ki",
            "heater_pid_kd",
            "heater_pid_lambda_order",
            "heater_pid_mu_order",
            "heater_pid_setpoint",
            "heater_manual_power",
        ):
            heater_form.addRow(key, self.inputs[key])

        self.inputs["is_sensor"] = self._checkbox("is sensor", False, self._update_optional_sections)
        self.inputs["is_sensor"].setVisible(False)
        self.sensor_box = self._group_box("Sensor")
        sensor_form = self.QtWidgets.QFormLayout(self.sensor_box)
        self.inputs["sensor_id"] = self._int_spin(-1, 10**9, 0)
        self.inputs["sensor_noise_std_K"] = self._double_spin(0.0, 1.0e9, 0.0, 0.01)
        self.inputs["sensor_bias_K"] = self._double_spin(-1.0e9, 1.0e9, 0.0, 0.01)
        self.inputs["sensor_time_constant_s"] = self._double_spin(0.0, 1.0e9, 0.0, 0.01)
        self.inputs["sensor_time_constant_s"].setToolTip(
            "Physical first-order sensor lag tau. This is the time to move about 63% of the way to a temperature step."
        )
        for label, key in (
            ("sensor_id", "sensor_id"),
            ("noise std K", "sensor_noise_std_K"),
            ("bias K", "sensor_bias_K"),
            ("time constant tau s", "sensor_time_constant_s"),
        ):
            sensor_form.addRow(label, self.inputs[key])
        for label, key in (
            ("MIMO setpoint K", "controller_setpoint_K"),
            ("MIMO sensor weight", "controller_weight"),
            ("MIMO settling time s", "sensor_settling_time_s"),
            ("MIMO coarse kP", "controller_kp_coarse"),
            ("MIMO coarse kI", "controller_ki_coarse"),
            ("MIMO coarse kD", "controller_kd_coarse"),
            ("MIMO hold kP", "controller_kp_hold"),
            ("MIMO hold kI", "controller_ki_hold"),
            ("MIMO hold kD", "controller_kd_hold"),
            ("MIMO lambda", "controller_lambda_order"),
            ("MIMO mu", "controller_mu_order"),
        ):
            sensor_form.addRow(label, self.inputs[key])

        self.controller_gain_box = self._group_box("MIMO Controller G Row")
        self.controller_gain_form = self.QtWidgets.QFormLayout(self.controller_gain_box)
        self.controller_gain_inputs: dict[int, Any] = {}

        self.inputs["has_cryocooler"] = self._checkbox("has cryocooler", False, self._update_optional_sections)

        layout.addLayout(form)
        layout.addWidget(self.inputs["role"])
        layout.addWidget(self.node_role_label)
        layout.addWidget(self.heater_box)
        layout.addWidget(self.sensor_box)
        layout.addWidget(self.controller_gain_box)
        layout.addWidget(self.inputs["has_cryocooler"])
        self._connect_node_form_autosave()
        self.left_layout.addWidget(box)
        self._build_component_temperature_controls()
        self._update_optional_sections()
        self._update_C_enabled()

    def _build_component_temperature_controls(self) -> None:
        box = self._group_box("Component Initial Temperature")
        form = self.QtWidgets.QFormLayout(box)
        self.component_temp_combo = self.QtWidgets.QComboBox()
        self.component_temp_input = self._double_spin(0.0, 1.0e6, 293.15, 1.0)
        self.component_temp_combo.currentTextChanged.connect(self._sync_component_temperature_input)
        self.component_temp_input.valueChanged.connect(self._handle_component_initial_temperature_changed)
        form.addRow("component", self.component_temp_combo)
        form.addRow("initial_temperature_K", self.component_temp_input)
        self.left_layout.addWidget(box)

    def _build_bulk_role_assignment_controls(self) -> None:
        box = self._group_box("Recognize Existing Cells")
        layout = self.QtWidgets.QVBoxLayout(box)
        form = self.QtWidgets.QFormLayout()
        self.bulk_role_substring_input = self.QtWidgets.QLineEdit()
        self.bulk_role_substring_input.setPlaceholderText("substring in component/source name")
        self.bulk_role_combo = self.QtWidgets.QComboBox()
        self.bulk_role_combo.addItems(["Heater", "Sensor"])
        form.addRow("substring", self.bulk_role_substring_input)
        form.addRow("assign as", self.bulk_role_combo)
        layout.addLayout(form)
        button = self.QtWidgets.QPushButton("Assign Matching Cells")
        button.clicked.connect(self.apply_bulk_role_assignment)
        layout.addWidget(button)
        self.left_layout.addWidget(box)

    def _build_details_panel(self) -> None:
        box = self._group_box("Selected Cell")
        layout = self.QtWidgets.QVBoxLayout(box)
        self.details_label = self.QtWidgets.QLabel("No cell selected.")
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)
        self.left_layout.addWidget(box)

    def _build_draw_controls(self, layout: Any) -> None:
        self.draw_mode_button = self.QtWidgets.QPushButton("Draw Mode")
        self.draw_mode_button.setCheckable(True)
        self.draw_mode_button.setToolTip("Extrude cells by clicking a cube face and dragging outward.")
        self.draw_mode_button.toggled.connect(self.enable_draw_mode)
        layout.addWidget(self.draw_mode_button)

    def prepare_new_node(self) -> None:
        next_id = 0
        while next_id in self.model.nodes:
            next_id += 1
        coord = (0, 0, 0)
        while coord in self.model.coord_index():
            coord = (coord[0] + 1, coord[1], coord[2])
        node = NodeProperties.with_material(
            next_id, coord, library=self.model.material_library
        )
        self.selected_node_id = None
        self.selected_node_ids = set()
        self._load_node_into_form(node)
        self._set_status("Ready to add a new cell.")

    def apply_node_form(self) -> None:
        self._apply_node_form_to_selected(show_status=True)

    def _apply_node_form_to_selected(
        self,
        show_status: bool = False,
        lightweight: bool = False,
    ) -> None:
        if self._building_form or self._applying_node_form:
            return
        self._applying_node_form = True
        try:
            node = self._node_from_form()
            target_ids = self._selected_target_ids()
            if not target_ids:
                raise ValueError("Select one or more existing octree cells before saving tags.")
            visual_tags_before = self._visual_tag_snapshot(target_ids)
            for target_id in target_ids:
                self._apply_node_template_to_existing_node(target_id, node)
            self.model.prune_controller_gain_matrix()
            self._mark_dirty()
            visual_tags_changed = visual_tags_before != self._visual_tag_snapshot(target_ids)
            if lightweight and not visual_tags_changed:
                self._refresh_details()
                self._refresh_simulation_readouts_from_editor()
            else:
                self._refresh_all(reset_camera=False)
                self._sync_simulation_from_editor(reinitialize=visual_tags_changed or not lightweight)
            self._rebuild_controller_gain_fields()
            if show_status:
                if len(target_ids) == 1:
                    self._set_status(f"Saved tags for cell {target_ids[0]}.")
                else:
                    self._set_status(f"Saved tags for {len(target_ids)} selected cells.")
        except Exception as exc:
            self._set_status(str(exc), error=True)
        finally:
            self._applying_node_form = False

    def _connect_node_form_autosave(self) -> None:
        for key in (
            "initial_temperature_K",
            "heater_id",
            "heater_min_power_W",
            "heater_max_power_W",
            "heater_efficiency",
            "heater_pid_kp",
            "heater_pid_ki",
            "heater_pid_kd",
            "heater_pid_lambda_order",
            "heater_pid_mu_order",
            "heater_manual_power",
            "heater_pid_setpoint",
            "sensor_id",
            "sensor_noise_std_K",
            "sensor_bias_K",
            "sensor_time_constant_s",
            "controller_setpoint_K",
            "controller_weight",
            "sensor_settling_time_s",
            "controller_kp_coarse",
            "controller_ki_coarse",
            "controller_kp_hold",
            "controller_ki_hold",
            "controller_kd_coarse",
            "controller_kd_hold",
            "controller_lambda_order",
            "controller_mu_order",
        ):
            self.inputs[key].valueChanged.connect(self._handle_node_form_changed)
        self.inputs["notes"].textChanged.connect(self._handle_node_form_changed)

    def _handle_node_form_changed(self, *_: Any) -> None:
        if self._building_form or not self._selected_target_ids():
            return
        self._apply_node_form_to_selected(show_status=False, lightweight=True)

    def _selected_target_ids(self) -> list[int]:
        target_ids = sorted(node_id for node_id in self.selected_node_ids if node_id in self.model.nodes)
        if not target_ids and self.selected_node_id in self.model.nodes:
            target_ids = [int(self.selected_node_id)]
        return target_ids

    def _apply_node_template_to_existing_node(self, target_id: int, template: NodeProperties) -> None:
        target = self.model.nodes[target_id]
        existing_heater_id = target.heater.heater_id or target.node_id
        existing_sensor_id = target.sensor.sensor_id or target.node_id
        target.is_heater = bool(template.is_heater)
        target.is_sensor = bool(template.is_sensor)
        if target.is_heater:
            target.heater = deepcopy(template.heater)
            if target_id != self.selected_node_id:
                target.heater.heater_id = existing_heater_id
            target.heater_control = deepcopy(template.heater_control)
        else:
            target.heater_control.reset_pid_state()
        if target.is_sensor:
            target.sensor = deepcopy(template.sensor)
            if target_id != self.selected_node_id:
                target.sensor.sensor_id = existing_sensor_id
        target.has_cryocooler = bool(template.has_cryocooler)
        target.controller_setpoint_K = template.controller_setpoint_K
        target.controller_weight = template.controller_weight
        target.sensor_settling_time_s = template.sensor_settling_time_s
        target.controller_kp_coarse = template.controller_kp_coarse
        target.controller_ki_coarse = template.controller_ki_coarse
        target.controller_kp_hold = template.controller_kp_hold
        target.controller_ki_hold = template.controller_ki_hold
        target.controller_kd_coarse = template.controller_kd_coarse
        target.controller_kd_hold = template.controller_kd_hold
        target.controller_lambda_order = template.controller_lambda_order
        target.controller_mu_order = template.controller_mu_order
        target.notes = template.notes
        target.initial_temperature_K = template.initial_temperature_K

    def apply_bulk_role_assignment(self) -> None:
        substring = self.bulk_role_substring_input.text() if hasattr(self, "bulk_role_substring_input") else ""
        role = self.bulk_role_combo.currentText().lower() if hasattr(self, "bulk_role_combo") else ""
        try:
            matched = assign_matching_nodes_to_role(self.model, substring, role)
        except ValueError as exc:
            self._set_status(str(exc), error=True)
            return
        if not matched:
            self._set_status(f"No cells matched substring {substring!r}.", error=True)
            return
        self.model.prune_controller_gain_matrix()
        self._mark_dirty()
        if self.selected_node_id in self.model.nodes:
            self._load_node_into_form(self.model.nodes[int(self.selected_node_id)])
        self._refresh_all(reset_camera=False)
        self._sync_simulation_from_editor(reinitialize=True)
        label = "heater" if role == "heater" else "sensor"
        self._set_status(
            f"Assigned {len(matched)} existing cell(s) as {label}s from substring {substring!r}."
        )

    def _visual_tag_snapshot(self, node_ids: list[int]) -> tuple[tuple[int, bool, bool, bool], ...]:
        return tuple(
            (
                int(node_id),
                bool(self.model.nodes[int(node_id)].is_heater),
                bool(self.model.nodes[int(node_id)].is_sensor),
                bool(self.model.nodes[int(node_id)].has_cryocooler),
            )
            for node_id in node_ids
            if int(node_id) in self.model.nodes
        )

    def _refresh_simulation_readouts_from_editor(self) -> None:
        if hasattr(self, "simulation_tab"):
            self.simulation_tab.refresh_live_readouts_from_editor(self.model, self.current_folder)

    def _rebuild_controller_gain_fields(self) -> None:
        if not hasattr(self, "controller_gain_form"):
            return
        while self.controller_gain_form.rowCount():
            self.controller_gain_form.removeRow(0)
        self.controller_gain_inputs = {}
        active_id = self.selected_node_id
        if active_id is None or active_id not in self.model.nodes:
            self.controller_gain_box.setVisible(False)
            return
        active_node = self.model.nodes[int(active_id)]
        if not active_node.is_sensor:
            self.controller_gain_box.setVisible(False)
            return
        self.controller_gain_box.setVisible(True)
        heater_ids = [
            int(node_id)
            for node_id, node in sorted(self.model.nodes.items(), key=lambda item: int(item[0]))
            if node.is_heater
        ]
        if not heater_ids:
            self.controller_gain_form.addRow(self.QtWidgets.QLabel("No heater cells are tagged."))
            return
        for heater_id in heater_ids:
            widget = self._double_spin(-1.0e12, 1.0e12, self.model.controller_gain(active_id, heater_id), 0.01)
            widget.valueChanged.connect(
                lambda value, heater_id=heater_id: self._handle_controller_gain_changed(heater_id, value)
            )
            self.controller_gain_inputs[heater_id] = widget
            self.controller_gain_form.addRow(f"G[{active_id},{heater_id}]", widget)

    def _handle_controller_gain_changed(self, heater_id: int, value: float) -> None:
        if self._building_form or self.selected_node_id is None:
            return
        sensor_id = int(self.selected_node_id)
        if sensor_id not in self.model.nodes:
            return
        self.model.set_controller_gain(sensor_id, int(heater_id), float(value))
        self._mark_dirty()
        if hasattr(self, "simulation_tab"):
            self.simulation_tab.save_active_controller_gain_matrix_from_editor(self.model)
        self._refresh_simulation_readouts_from_editor()

    def _handle_controller_gain_matrix_changed(self) -> None:
        self._mark_dirty()
        self._rebuild_controller_gain_fields()
        self._refresh_details()

    def delete_selected_node(self) -> None:
        if self.selected_node_id is None:
            self._set_status("No selected cell to delete.", error=True)
            return
        self.model.delete_node(self.selected_node_id)
        self.selected_node_ids.discard(self.selected_node_id)
        self.selected_node_id = None
        self._handle_topology_changed()
        self._refresh_all(reset_camera=False)
        self._set_status("Deleted selected cell.")

    def select_node(self, node_id: int, additive: bool = False) -> None:
        if node_id not in self.model.nodes:
            return
        if additive:
            if node_id in self.selected_node_ids:
                self.selected_node_ids.remove(node_id)
                if self.selected_node_id == node_id:
                    self.selected_node_id = next(iter(sorted(self.selected_node_ids)), None)
            else:
                self.selected_node_ids.add(node_id)
                self.selected_node_id = node_id
        else:
            self.selected_node_ids = {node_id}
            self.selected_node_id = node_id
        if self.selected_node_id is None or self.selected_node_id not in self.model.nodes:
            self._refresh_all(reset_camera=False)
            self._set_status("No cell selected.")
            return
        node = self.model.nodes[self.selected_node_id]
        self._load_node_into_form(node)
        self._refresh_details()
        self.viewer.select_nodes(set(self.selected_node_ids), active_node_id=self.selected_node_id)
        self.two_d_view.selected_node_ids = set(self.selected_node_ids)
        if self.view_tabs.currentWidget() is self.two_d_view.widget:
            self.two_d_view.refresh()
        component = self._component_display(node)
        if len(self.selected_node_ids) > 1:
            self._set_status(
                f"Selected {len(self.selected_node_ids)} cells. Active cell {self.selected_node_id}: {component}."
            )
        else:
            self._set_status(f"Selected cell {self.selected_node_id}: {component}.")

    def _handle_viewer_pick(
        self,
        node_id: int,
        picked_point: tuple[float, float, float] | None = None,
        mouse_position: tuple[int, int] | None = None,
        additive: bool = False,
    ) -> None:
        self.select_node(node_id, additive=additive)

    def search_node(self) -> None:
        text = self.search_input.text().strip()
        try:
            coord = self._parse_coord(text)
            node = self.model.find_by_coord(coord)
            if node is None:
                self._set_status(f"No node found at coordinate {coord}.", error=True)
                return
            self.select_node(node.node_id)
            self._set_status(f"Selected node {node.node_id}.")
            return
        except ValueError:
            pass
        try:
            node_id = int(text)
        except ValueError:
            self._set_status("Search must be a node_id or coordinate like (1, 2, 3).", error=True)
            return
        if node_id not in self.model.nodes:
            self._set_status(f"No node found with node_id {node_id}.", error=True)
            return
        self.select_node(node_id)
        self._set_status(f"Selected node {node_id}.")

    def recompute_auto_edges(self) -> None:
        refresh_auto_edges(self.model)
        self._sync_conduction_ui(EdgeMode.AUTO.value)
        self._mark_dirty()
        self._refresh_all(reset_camera=False)
        self._set_status(f"Auto-estimated {len(self.model.edges)} face-adjacent edges.")

    def load_matrix_edges(self) -> None:
        folder = self.current_folder
        if folder is None:
            self._set_status(
                "Save or load a graph folder before selecting loaded-G mode.",
                error=True,
            )
            self._sync_conduction_ui(EdgeMode.AUTO.value)
            return
        try:
            load_conductance_matrix_from_folder(self.model, folder)
            self.model.metadata.edge_mode = EdgeMode.LOADED_G.value
            self._sync_conduction_ui(EdgeMode.LOADED_G.value)
            self._mark_dirty()
            self._refresh_all(reset_camera=False)
            self._set_status(f"Loaded G conductance matrix from {folder / 'matrices.npz'}.")
        except Exception as exc:
            self._sync_conduction_ui(EdgeMode.AUTO.value)
            self._set_status(str(exc), error=True)

    def new_graph(self) -> None:
        self.model = ThermalGraphModel(
            metadata=GraphMetadata(), material_library=default_material_library()
        )
        self.current_folder = None
        self.autosave_enabled = False
        self.selected_node_id = None
        self.selected_node_ids = set()
        self.dirty = False
        self.cancel_draw_preview()
        self._sync_metadata_widgets()
        self.prepare_new_node()
        self._refresh_all(reset_camera=True)
        self._set_status("Started a new graph.")

    def load_graph(self) -> None:
        folder = self._choose_existing_folder("Load graph folder")
        if folder is None:
            return
        try:
            self.model, _matrices = load_graph_folder(folder)
            self.current_folder = folder
            self.autosave_enabled = True
            self.selected_node_id = None
            self.selected_node_ids = set()
            self.dirty = False
            self.cancel_draw_preview()
            self._sync_metadata_widgets()
            self.prepare_new_node()
            self._load_ui_state(folder)
            self._refresh_all(reset_camera=True)
            self._set_status(f"Loaded graph folder {folder}.")
        except Exception as exc:
            self._set_status(str(exc), error=True)

    def save_graph(self) -> None:
        if self.current_folder is None:
            self.save_graph_as()
            return
        self._save_to_folder(self.current_folder)

    def save_graph_as(self) -> None:
        parent_folder = self._choose_save_folder("Choose parent folder for graph")
        if parent_folder is None:
            return
        self._update_metadata_from_inputs()
        graph_folder_name = self._safe_graph_folder_name(self.model.metadata.graph_name)
        if graph_folder_name != self.model.metadata.graph_name:
            self.model.metadata.graph_name = graph_folder_name
            self.graph_name_input.setText(graph_folder_name)
        folder = parent_folder / graph_folder_name
        self.current_folder = folder
        self.autosave_enabled = True
        self._save_to_folder(folder)

    def _save_to_folder(self, folder: Path) -> None:
        try:
            self._update_metadata_from_inputs()
            errors = validate_model(self.model)
            raise_if_errors(errors, "Cannot save graph")
            matrices = save_graph_folder(self.model, folder)
            self._save_ui_state(folder)
            self._set_status(
                f"Saved {len(self.model.nodes)} nodes, {len(self.model.edges)} edges, "
                f"matrix shape {matrices['G'].shape}."
            )
            self.dirty = False
            self._update_window_title()
            self._refresh_all(reset_camera=False)
        except Exception as exc:
            self._set_status(str(exc), error=True)

    def enable_draw_mode(self, enabled: bool) -> None:
        self.draw_mode_enabled = bool(enabled)
        self.viewer.set_draw_mode(self.draw_mode_enabled)
        if not self.draw_mode_enabled:
            self.cancel_draw_preview()
            self._set_status("Draw Mode disabled.")
        else:
            self._set_status("Draw Mode enabled. Click a cube face and drag to extrude cells.")

    def start_draw_from_face(
        self,
        node_id: int,
        picked_point: tuple[float, float, float] | None,
        mouse_position: tuple[int, int] | None,
    ) -> None:
        if node_id not in self.model.nodes:
            self._set_status("Draw start missed a valid cell.", error=True)
            return
        node = self.model.nodes[node_id]
        side_length = float(node.side_length_m)
        if side_length <= 0.0:
            self._set_status("Cannot draw from a cell with invalid side_length_m.", error=True)
            return
        point = picked_point or node.center
        self.draw_active = True
        self.draw_start_node_id = node_id
        self.draw_start_coord = node.coord
        self.draw_start_pixel = mouse_position
        self.draw_normal_grid = compute_face_normal(node.center, point)
        self.draw_screen_direction, self.draw_pixels_per_cell = (
            self.viewer.screen_step_for_grid_normal(node.center, self.draw_normal_grid)
        )
        self.draw_preview_coords = []
        self._last_shown_preview_coords = []
        self.viewer.clear_preview()
        self._set_status(
            f"Drawing from node {node_id}, normal {self.draw_normal_grid}. Drag to preview cells."
        )

    def update_draw_preview(self, mouse_position: tuple[int, int] | None) -> None:
        if not self.draw_mode_enabled or not self.draw_active:
            return
        if (
            self.draw_start_node_id is None
            or self.draw_start_node_id not in self.model.nodes
            or self.draw_start_coord is None
            or self.draw_normal_grid is None
        ):
            self.cancel_draw_preview()
            self._set_status("Draw preview cancelled because the start cell is no longer valid.", error=True)
            return
        count = extrusion_count_from_projected_pixel_drag(
            self.draw_start_pixel,
            mouse_position,
            self.draw_screen_direction,
            pixels_per_cell=max(12.0, float(self.draw_pixels_per_cell or 80.0)),
        )
        occupied = set(self.model.coord_index())
        self.draw_preview_coords = preview_coords(
            self.draw_start_coord, self.draw_normal_grid, count, occupied
        )
        if self.draw_preview_coords == self._last_shown_preview_coords:
            return
        self._last_shown_preview_coords = list(self.draw_preview_coords)
        source_node = self.model.nodes[self.draw_start_node_id]
        self.viewer.show_preview(self.draw_preview_coords, source_node.side_length_m)
        if count > 0 and not self.draw_preview_coords:
            self._set_status("Adjacent coordinate is occupied; extrusion would create zero cells.", error=True)

    def commit_draw_preview_if_active(self) -> None:
        if self.draw_mode_enabled and self.draw_active:
            self._suppress_next_draw_pick = True
            self.commit_draw_preview()

    def commit_draw_preview(self) -> None:
        if not self.draw_active:
            return
        coords = list(self.draw_preview_coords)
        start_node_id = self.draw_start_node_id
        self.clear_draw_preview()
        if not coords:
            self._set_status("Draw finished with no new cells.")
            return
        if start_node_id is None or start_node_id not in self.model.nodes:
            self._set_status("Draw commit cancelled because the start cell was deleted.", error=True)
            return
        self.viewer.clear_preview(render=False)
        source = self.model.nodes[start_node_id]
        node_id = next_node_id(self.model.nodes)
        for coord in coords:
            self.model.add_node(clone_node_for_extrusion(source, node_id, coord))
            node_id += 1
        invalidated_loaded_g = self._handle_topology_changed()
        self.selected_node_id = node_id - 1
        self.selected_node_ids = {self.selected_node_id}
        self._refresh_all(reset_camera=False)
        if not invalidated_loaded_g:
            self._set_status(f"Created {len(coords)} extruded cell(s).")

    def cancel_draw_preview(self) -> None:
        self.clear_draw_preview()
        self.viewer.clear_preview()

    def clear_draw_preview(self) -> None:
        self.draw_active = False
        self.draw_start_node_id = None
        self.draw_start_coord = None
        self.draw_start_pixel = None
        self.draw_normal_grid = None
        self.draw_screen_direction = None
        self.draw_pixels_per_cell = None
        self.draw_preview_coords = []
        self._last_shown_preview_coords = []

    def _handle_topology_changed(self) -> bool:
        """Refresh topology-dependent matrices and invalidate loaded G when needed."""
        was_loaded = EdgeMode.normalize(self.model.metadata.edge_mode) == EdgeMode.LOADED_G.value
        if was_loaded:
            self.model.metadata.edge_mode = EdgeMode.AUTO.value
            self._sync_conduction_ui(EdgeMode.AUTO.value)
            refresh_auto_edges(self.model)
            self._mark_dirty()
            self._set_status(
                "Topology changed, so the loaded conductance matrix G is no longer valid. "
                "Switching to auto-estimated conductance mode.",
                error=True,
            )
            return True
        if EdgeMode.normalize(self.model.metadata.edge_mode) == EdgeMode.AUTO.value:
            refresh_auto_edges(self.model)
        self._mark_dirty()
        return False

    def _handle_conduction_radio_change(self, *_: Any) -> None:
        if self._syncing_conduction_ui:
            return
        if self.auto_conduction_radio.isChecked():
            refresh_auto_edges(self.model)
            self.model.metadata.edge_mode = EdgeMode.AUTO.value
            self._mark_dirty()
            self._refresh_all(reset_camera=False)
            self._set_status(f"Auto-estimated {len(self.model.edges)} face-adjacent edges.")
        elif self.loaded_conduction_radio.isChecked():
            self.load_matrix_edges()

    def _sync_conduction_ui(self, mode: str) -> None:
        normalized = EdgeMode.normalize(mode)
        self._syncing_conduction_ui = True
        self.auto_conduction_radio.setChecked(normalized == EdgeMode.AUTO.value)
        self.loaded_conduction_radio.setChecked(normalized == EdgeMode.LOADED_G.value)
        self.model.metadata.edge_mode = normalized
        self._syncing_conduction_ui = False

    def _mark_dirty(self) -> None:
        self.dirty = True
        self.model.touch()
        self._update_window_title()
        self._schedule_autosave()

    def _schedule_autosave(self) -> None:
        if not self.autosave_enabled or self.current_folder is None:
            self._set_status("Unsaved graph - use Save As to enable autosave.", error=True)
            return
        was_active = self.autosave_timer.isActive()
        self.autosave_timer.start(5000)
        if not was_active:
            self._set_status("Autosave scheduled...")

    def _autosave_now(self) -> None:
        if not self.dirty or self.current_folder is None:
            return
        try:
            self._update_metadata_from_inputs()
            errors = validate_model(self.model)
            raise_if_errors(errors, "Cannot autosave graph")
            save_graph_folder(self.model, self.current_folder)
            self._save_ui_state(self.current_folder)
            self.dirty = False
            self._update_window_title()
            self._set_status(f"Autosaved at {datetime.now().strftime('%H:%M:%S')}.")
        except Exception as exc:
            self._set_status(f"Autosave failed: {exc}", error=True)

    def _update_window_title(self) -> None:
        marker = "*" if self.dirty else ""
        self.window.setWindowTitle(
            f"Graph Visualizer - Sparse Thermal Lump Network{marker}"
        )

    def _save_ui_state(self, folder: Path) -> None:
        state = {
            "selected_node_id": self.selected_node_id,
            "selected_node_ids": sorted(self.selected_node_ids),
            "filters": {
                "material": self.filter_material.currentText() if hasattr(self, "filter_material") and self.filter_material.count() else "All",
                "component": self.filter_component.currentText() if hasattr(self, "filter_component") and self.filter_component.count() else "All",
                "level_min": int(self.filter_level_min.value()) if hasattr(self, "filter_level_min") else 0,
                "level_max": int(self.filter_level_max.value()) if hasattr(self, "filter_level_max") else 99,
                "heater_sensor_only": self.filter_heater_sensor.isChecked() if hasattr(self, "filter_heater_sensor") else False,
                "boundary_only": self.filter_boundary.isChecked() if hasattr(self, "filter_boundary") else False,
            },
            "dark_mode": self.dark_mode,
        }
        self._atomic_write_json(folder / "ui_state.json", state, indent=2)

    def _load_ui_state(self, folder: Path) -> None:
        path = folder / "ui_state.json"
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        if "dark_mode" in state:
            self.dark_mode = bool(state["dark_mode"])
            if hasattr(self, "theme_toggle"):
                self.theme_toggle.blockSignals(True)
                self.theme_toggle.setChecked(self.dark_mode)
                self.theme_toggle.blockSignals(False)
            self._apply_theme()
        selected = state.get("selected_node_id")
        try:
            selected_id = int(selected)
        except (TypeError, ValueError):
            selected_id = None
        if selected_id in self.model.nodes:
            self.selected_node_id = selected_id
        selected_ids: set[int] = set()
        for raw_node_id in state.get("selected_node_ids", []):
            try:
                node_id = int(raw_node_id)
            except (TypeError, ValueError):
                continue
            if node_id in self.model.nodes:
                selected_ids.add(node_id)
        if selected_ids:
            self.selected_node_ids = selected_ids
            if self.selected_node_id not in self.selected_node_ids:
                self.selected_node_id = next(iter(sorted(self.selected_node_ids)), None)
        elif self.selected_node_id in self.model.nodes:
            self.selected_node_ids = {int(self.selected_node_id)}
        if self.selected_node_id in self.model.nodes:
            self._load_node_into_form(self.model.nodes[int(self.selected_node_id)])

    def _node_from_form(self) -> NodeProperties:
        node_id = int(self.inputs["node_id"].value())
        role = self.inputs["role"].currentText().lower()
        is_heater = role == "heater"
        is_sensor = role == "sensor"
        has_cryocooler = self.inputs["has_cryocooler"].isChecked()
        node = NodeProperties(
            node_id=node_id,
            coord=(
                int(self.inputs["coord_i"].value()),
                int(self.inputs["coord_j"].value()),
                int(self.inputs["coord_k"].value()),
            ),
            side_length_m=float(self.inputs["side_length_m"].value()),
            material=self.inputs["material"].currentText(),
            rho_kg_m3=float(self.inputs["rho_kg_m3"].value()),
            cp_J_kgK=float(self.inputs["cp_J_kgK"].value()),
            k_W_mK=float(self.inputs["k_W_mK"].value()),
            emissivity=float(self.inputs["emissivity"].value()),
            mass_kg=float(self.inputs["mass_kg"].value()),
            C_J_K=float(self.inputs["C_J_K"].value()),
            C_manual_override=self.inputs["C_manual_override"].isChecked(),
            Grad_W_K=float(self.inputs["Grad_W_K"].value()),
            initial_temperature_K=float(self.inputs["initial_temperature_K"].value()),
            is_heater=is_heater,
            heater=HeaterProperties(
                heater_id=int(self.inputs["heater_id"].value()),
                heater_min_power_W=float(self.inputs["heater_min_power_W"].value()),
                heater_max_power_W=float(self.inputs["heater_max_power_W"].value()),
                heater_efficiency=float(self.inputs["heater_efficiency"].value()),
            ),
            heater_control=HeaterControl(
                mode=(
                    "pid"
                    if self.inputs["heater_mode_pid"].isChecked()
                    else "mimo"
                    if self.inputs["heater_mode_mimo"].isChecked()
                    else "manual"
                ),
                pid=PIDControlSettings(
                    kp=float(self.inputs["heater_pid_kp"].value()),
                    ki=float(self.inputs["heater_pid_ki"].value()),
                    kd=float(self.inputs["heater_pid_kd"].value()),
                    lambda_order=float(self.inputs["heater_pid_lambda_order"].value()),
                    mu_order=float(self.inputs["heater_pid_mu_order"].value()),
                    setpoint=float(self.inputs["heater_pid_setpoint"].value()),
                ),
                manual=ManualHeaterSettings(power=float(self.inputs["heater_manual_power"].value())),
                pid_state=PIDState(),
            ),
            is_sensor=is_sensor,
            sensor=SensorProperties(
                sensor_id=int(self.inputs["sensor_id"].value()),
                sensor_noise_std_K=float(self.inputs["sensor_noise_std_K"].value()),
                sensor_bias_K=float(self.inputs["sensor_bias_K"].value()),
                sensor_time_constant_s=float(self.inputs["sensor_time_constant_s"].value()),
            ),
            has_cryocooler=has_cryocooler,
            controller_setpoint_K=float(self.inputs["controller_setpoint_K"].value()),
            controller_weight=float(self.inputs["controller_weight"].value()),
            sensor_settling_time_s=float(self.inputs["sensor_settling_time_s"].value()),
            controller_kp_coarse=float(self.inputs["controller_kp_coarse"].value()),
            controller_ki_coarse=float(self.inputs["controller_ki_coarse"].value()),
            controller_kp_hold=float(self.inputs["controller_kp_hold"].value()),
            controller_ki_hold=float(self.inputs["controller_ki_hold"].value()),
            controller_kd_coarse=float(self.inputs["controller_kd_coarse"].value()),
            controller_kd_hold=float(self.inputs["controller_kd_hold"].value()),
            controller_lambda_order=float(self.inputs["controller_lambda_order"].value()),
            controller_mu_order=float(self.inputs["controller_mu_order"].value()),
            notes=self.inputs["notes"].toPlainText(),
        )
        if not node.C_manual_override:
            node.recompute_heat_capacity()
        if not is_heater:
            node.heater.heater_id = node_id
            node.heater_control.reset_pid_state()
        if not is_sensor:
            node.sensor.sensor_id = node_id
        return node

    def _load_node_into_form(self, node: NodeProperties) -> None:
        self._building_form = True
        self.inputs["part_code"].setText(self._part_code(node.component_name))
        self.inputs["component_name"].setText(node.component_name or "")
        self.inputs["node_id"].setValue(node.node_id)
        self.inputs["coord_i"].setValue(node.coord[0])
        self.inputs["coord_j"].setValue(node.coord[1])
        self.inputs["coord_k"].setValue(node.coord[2])
        self.inputs["side_length_m"].setValue(node.side_length_m)
        if node.material in self.model.material_library:
            self.inputs["material"].setCurrentText(node.material)
        self.inputs["rho_kg_m3"].setValue(node.rho_kg_m3)
        self.inputs["cp_J_kgK"].setValue(node.cp_J_kgK)
        self.inputs["k_W_mK"].setValue(node.k_W_mK)
        self.inputs["emissivity"].setValue(node.emissivity)
        self.inputs["mass_kg"].setValue(node.mass_kg)
        self.inputs["C_manual_override"].setChecked(node.C_manual_override)
        self.inputs["C_J_K"].setValue(node.C_J_K)
        self.inputs["Grad_W_K"].setValue(node.Grad_W_K)
        self.inputs["initial_temperature_K"].setValue(node.initial_temperature_K)
        self.inputs["notes"].setPlainText(node.notes)
        self.inputs["role"].setCurrentText("Heater" if node.is_heater else "Sensor" if node.is_sensor else "Body")
        self.inputs["is_heater"].setChecked(node.is_heater)
        self.inputs["heater_id"].setValue(node.heater.heater_id or node.node_id)
        self.inputs["heater_min_power_W"].setValue(node.heater.heater_min_power_W)
        self.inputs["heater_max_power_W"].setValue(node.heater.heater_max_power_W)
        self.inputs["heater_efficiency"].setValue(node.heater.heater_efficiency)
        if isinstance(node.heater_control, HeaterControl):
            heater_control = node.heater_control
        else:
            heater_control = HeaterControl.from_dict(
                {},
                initial_temperature_K=node.initial_temperature_K,
                default_manual_power_W=float(node.heater.heater_max_power_W) * float(node.heater.heater_efficiency),
            )
        self.inputs["heater_mode_pid"].setChecked(heater_control.mode == "pid")
        self.inputs["heater_mode_mimo"].setChecked(heater_control.mode == "mimo")
        self.inputs["heater_mode_manual"].setChecked(heater_control.mode not in {"pid", "mimo"})
        self.inputs["heater_pid_kp"].setValue(heater_control.pid.kp)
        self.inputs["heater_pid_ki"].setValue(heater_control.pid.ki)
        self.inputs["heater_pid_kd"].setValue(heater_control.pid.kd)
        self.inputs["heater_pid_lambda_order"].setValue(float(getattr(heater_control.pid, "lambda_order", 1.0)))
        self.inputs["heater_pid_mu_order"].setValue(float(getattr(heater_control.pid, "mu_order", 1.0)))
        self.inputs["heater_pid_setpoint"].setValue(heater_control.pid.setpoint)
        manual_power = heater_control.manual.power
        if node.is_heater and heater_control.mode == "manual" and manual_power <= 0.0:
            manual_power = self._default_heater_manual_power()
        self.inputs["heater_manual_power"].setValue(manual_power)
        self.inputs["is_sensor"].setChecked(node.is_sensor)
        self.inputs["sensor_id"].setValue(node.sensor.sensor_id or node.node_id)
        self.inputs["sensor_noise_std_K"].setValue(node.sensor.sensor_noise_std_K)
        self.inputs["sensor_bias_K"].setValue(node.sensor.sensor_bias_K)
        self.inputs["sensor_time_constant_s"].setValue(node.sensor.sensor_time_constant_s)
        self.inputs["controller_setpoint_K"].setValue(float(getattr(node, "controller_setpoint_K", 293.15)))
        self.inputs["controller_weight"].setValue(float(getattr(node, "controller_weight", 0.0)))
        self.inputs["sensor_settling_time_s"].setValue(float(getattr(node, "sensor_settling_time_s", 0.0)))
        self.inputs["controller_kp_coarse"].setValue(float(getattr(node, "controller_kp_coarse", 0.0)))
        self.inputs["controller_ki_coarse"].setValue(float(getattr(node, "controller_ki_coarse", 0.0)))
        self.inputs["controller_kp_hold"].setValue(float(getattr(node, "controller_kp_hold", 0.0)))
        self.inputs["controller_ki_hold"].setValue(float(getattr(node, "controller_ki_hold", 0.0)))
        self.inputs["controller_kd_coarse"].setValue(float(getattr(node, "controller_kd_coarse", 0.0)))
        self.inputs["controller_kd_hold"].setValue(float(getattr(node, "controller_kd_hold", 0.0)))
        self.inputs["controller_lambda_order"].setValue(float(getattr(node, "controller_lambda_order", 1.0)))
        self.inputs["controller_mu_order"].setValue(float(getattr(node, "controller_mu_order", 1.0)))
        self.inputs["has_cryocooler"].setChecked(node.has_cryocooler)
        self._sync_node_role_label(node)
        self._update_optional_sections()
        self._update_C_enabled()
        self._building_form = False
        self._rebuild_controller_gain_fields()

    def _handle_role_changed(self, *_: Any) -> None:
        if "role" not in self.inputs:
            return
        role = self.inputs["role"].currentText().lower()
        self.inputs["is_heater"].blockSignals(True)
        self.inputs["is_sensor"].blockSignals(True)
        self.inputs["is_heater"].setChecked(role == "heater")
        self.inputs["is_sensor"].setChecked(role == "sensor")
        self.inputs["is_heater"].blockSignals(False)
        self.inputs["is_sensor"].blockSignals(False)
        self._update_optional_sections()

    def _apply_selected_material(self, *_: Any) -> None:
        if self._building_form:
            return
        material = self.inputs["material"].currentText()
        defaults = self.model.material_library.get(material, {})
        for key in ("rho_kg_m3", "cp_J_kgK", "k_W_mK", "emissivity"):
            if key in defaults:
                self.inputs[key].setValue(float(defaults[key]))
        self._auto_update_C()

    def _auto_update_C(self, *_: Any) -> None:
        if self._building_form or self.inputs["C_manual_override"].isChecked():
            return
        self.inputs["C_J_K"].setValue(
            float(self.inputs["mass_kg"].value()) * float(self.inputs["cp_J_kgK"].value())
        )

    def _update_C_enabled(self, *_: Any) -> None:
        manual = self.inputs["C_manual_override"].isChecked()
        self.inputs["C_J_K"].setEnabled(manual)
        if not manual:
            self._auto_update_C()

    def _update_optional_sections(self, *_: Any) -> None:
        if "is_heater" not in self.inputs:
            return
        if (
            not self._building_form
            and self.inputs["is_heater"].isChecked()
            and self.inputs["heater_mode_manual"].isChecked()
            and self.inputs["heater_manual_power"].value() <= 0.0
        ):
            self.inputs["heater_manual_power"].setValue(self._default_heater_manual_power())
        self.heater_box.setVisible(self.inputs["is_heater"].isChecked())
        self.sensor_box.setVisible(self.inputs["is_sensor"].isChecked())
        self.controller_gain_box.setVisible(
            self.inputs["is_sensor"].isChecked()
        )
        node_id = int(self.inputs["node_id"].value())
        if self.inputs["heater_id"].value() == 0:
            self.inputs["heater_id"].setValue(node_id)
        if self.inputs["sensor_id"].value() == 0:
            self.inputs["sensor_id"].setValue(node_id)
        self._sync_heater_control_enabled()
        if not self._building_form:
            self._rebuild_controller_gain_fields()
        self._handle_node_form_changed()

    def _handle_heater_mode_change(self, *_: Any) -> None:
        if self._building_form:
            self._sync_heater_control_enabled()
            return
        self._reset_form_pid_state()
        self._sync_heater_control_enabled()
        self._handle_node_form_changed()

    def _handle_heater_setpoint_change(self, *_: Any) -> None:
        if not self._building_form:
            self._reset_form_pid_state()
            self._handle_node_form_changed()

    def _reset_form_pid_state(self) -> None:
        for node_id in self._selected_target_ids():
            self.model.nodes[int(node_id)].heater_control.reset_pid_state()

    def _sync_heater_control_enabled(self) -> None:
        if "heater_mode_pid" not in self.inputs:
            return
        pid_active = self.inputs["is_heater"].isChecked() and self.inputs["heater_mode_pid"].isChecked()
        mimo_active = self.inputs["is_sensor"].isChecked()
        manual_active = self.inputs["is_heater"].isChecked() and self.inputs["heater_mode_manual"].isChecked()
        for key in (
            "heater_pid_kp",
            "heater_pid_ki",
            "heater_pid_kd",
            "heater_pid_lambda_order",
            "heater_pid_mu_order",
            "heater_pid_setpoint",
        ):
            self.inputs[key].setEnabled(pid_active)
            self.inputs[key].setSpecialValueText("" if not pid_active else "")
        for key in (
            "controller_setpoint_K",
            "controller_weight",
            "sensor_settling_time_s",
            "controller_kp_coarse",
            "controller_ki_coarse",
            "controller_kp_hold",
            "controller_ki_hold",
            "controller_kd_coarse",
            "controller_kd_hold",
            "controller_lambda_order",
            "controller_mu_order",
        ):
            self.inputs[key].setEnabled(mimo_active)
            self.inputs[key].setSpecialValueText("" if not mimo_active else "")
        self.inputs["heater_manual_power"].setEnabled(manual_active)
        if hasattr(self, "controller_gain_box"):
            self.controller_gain_box.setEnabled(mimo_active)

    def _default_heater_manual_power(self) -> float:
        return max(
            0.0,
            float(self.inputs["heater_max_power_W"].value())
            * float(self.inputs["heater_efficiency"].value()),
        )

    def _sync_simulation_from_editor(self, reinitialize: bool = False) -> None:
        if hasattr(self, "simulation_tab"):
            self.simulation_tab.sync_from_editor(
                self.model,
                self.current_folder,
                reinitialize=reinitialize,
            )

    def _refresh_all(self, reset_camera: bool = False) -> None:
        self._sync_filter_options()
        visible_node_ids = self._filtered_node_ids()
        self.viewer.set_hover_tooltips_enabled(
            not (hasattr(self, "filter_heater_sensor") and self.filter_heater_sensor.isChecked())
        )
        self.viewer.set_toggles(
            self.show_labels.isChecked(),
            self.show_edges.isChecked(),
            self.show_heaters.isChecked(),
            self.show_sensors.isChecked(),
            self.show_coolers.isChecked(),
        )
        self.viewer.set_draw_mode(self.draw_mode_enabled)
        self.viewer.selected_node_id = self.selected_node_id
        self.viewer.selected_node_ids = set(self.selected_node_ids)
        self.viewer.draw(self.model, reset_camera=reset_camera, visible_node_ids=visible_node_ids)
        self.two_d_view.selected_node_ids = set(self.selected_node_ids)
        self.two_d_view.set_model(
            self.model,
            visible_node_ids=visible_node_ids,
            auto_refresh=self.view_tabs.currentWidget() is self.two_d_view.widget,
        )
        self._sync_simulation_from_editor()
        self._refresh_details()

    def _sync_filter_options(self) -> None:
        if not hasattr(self, "filter_material"):
            return
        for combo, values in (
            (self.filter_material, sorted({node.material for node in self.model.nodes.values()})),
            (self.filter_component, sorted({node.component_name for node in self.model.nodes.values() if node.component_name})),
        ):
            current = combo.currentText() if combo.count() else "All"
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All")
            combo.addItems(values)
            combo.setCurrentText(current if current in ["All", *values] else "All")
            combo.blockSignals(False)
        if hasattr(self, "component_temp_combo"):
            current = self.component_temp_combo.currentText() if self.component_temp_combo.count() else ""
            values = sorted({node.component_name for node in self.model.nodes.values() if node.component_name})
            self.component_temp_combo.blockSignals(True)
            self.component_temp_combo.clear()
            self.component_temp_combo.addItems(values)
            if current in values:
                self.component_temp_combo.setCurrentText(current)
            self.component_temp_combo.blockSignals(False)
            self._sync_component_temperature_input()

    def _filtered_node_ids(self) -> set[int]:
        if not hasattr(self, "filter_material"):
            return set(self.model.nodes)
        material = self.filter_material.currentText() if self.filter_material.count() else "All"
        component = self.filter_component.currentText() if self.filter_component.count() else "All"
        min_level = int(self.filter_level_min.value())
        max_level = int(self.filter_level_max.value())
        contact_nodes = {
            endpoint
            for edge in self.model.edges.values()
            if edge.edge_type not in {"internal_conduction", "near_internal_conduction", "same_material_spatial"}
            for endpoint in (edge.source, edge.target)
        }
        visible: set[int] = set()
        for node_id, node in self.model.nodes.items():
            if material != "All" and node.material != material:
                continue
            if component != "All" and node.component_name != component:
                continue
            if not (min_level <= int(node.level) <= max_level):
                continue
            if self.filter_heater_sensor.isChecked() and not (
                node.is_heater or node.is_sensor or node.has_cryocooler
            ):
                continue
            if self.filter_boundary.isChecked() and node.confidence == "high" and node_id not in contact_nodes:
                continue
            visible.add(node_id)
        return visible

    def _refresh_details(self) -> None:
        selected_count = len(self.selected_node_ids)
        if self.selected_node_id is None or self.selected_node_id not in self.model.nodes:
            self.details_label.setText("No cell selected.")
            return
        node = self.model.nodes[self.selected_node_id]
        incident = [
            edge for edge in self.model.edges.values()
            if edge.source == node.node_id or edge.target == node.node_id
        ]
        prefix = f"selected cells: {selected_count}\nactive " if selected_count > 1 else ""
        self.details_label.setText(
            f"{prefix}node_id: {node.node_id}\n"
            f"cell_id: {node.cell_id or node.coord}\n"
            f"part code: {self._part_code(node.component_name)}\n"
            f"component: {node.component_name or '?'}\n"
            f"center_mm: {node.center}\n"
            f"size_mm: {node.size_mm or node.side_length_m}\n"
            f"material: {node.material}\n"
            f"level: {node.level}, confidence: {node.confidence}\n"
            f"C: {node.C_J_K:.6g} J/K, Grad: {node.Grad_W_K:.6g} W/K\n"
            f"initial T: {node.initial_temperature_K:.3f} K / {node.initial_temperature_K - 273.15:.3f} C\n"
            f"exposed: {node.is_exposed}, G_rad: {node.G_rad_W_K:.6g} W/K\n"
            f"role: {self._node_role_text(node)}, cryocooler: {node.has_cryocooler}\n"
            f"incident conductive edges: {len(incident)}"
        )

    def _sync_node_role_label(self, node: NodeProperties) -> None:
        if hasattr(self, "node_role_label"):
            self.node_role_label.setText(f"role: {self._node_role_text(node)}")

    @staticmethod
    def _node_role_text(node: NodeProperties) -> str:
        if node.is_heater and node.is_sensor:
            return "invalid heater/sensor node"
        if node.is_heater:
            return "heater node" if node.is_cad_role_node else "user heater node"
        if node.is_sensor:
            return "sensor node" if node.is_cad_role_node else "user sensor node"
        return "body cell"

    @staticmethod
    def _part_code(component_name: str) -> str:
        match = re.search(r"P\d{3,5}", component_name or "")
        return match.group(0) if match else ""

    def _component_display(self, node: NodeProperties) -> str:
        part_code = self._part_code(node.component_name)
        if part_code and node.component_name:
            return f"{part_code} ({node.component_name})"
        return node.component_name or "unknown component"

    def _handle_visual_toggle(self, *_: Any) -> None:
        self._refresh_all(reset_camera=False)

    def _handle_marker_toggle(self, *_: Any) -> None:
        if hasattr(self, "viewer"):
            self.viewer.update_io_marker_visibility(
                self.show_heaters.isChecked(),
                self.show_sensors.isChecked(),
                self.show_coolers.isChecked(),
            )

    def apply_component_initial_temperature(self) -> None:
        if not hasattr(self, "component_temp_combo"):
            return
        component = self.component_temp_combo.currentText()
        if not component:
            self._set_status("Choose a component before applying initial temperature.", error=True)
            return
        temperature = float(self.component_temp_input.value())
        count = 0
        for node in self.model.nodes.values():
            if node.component_name == component:
                node.initial_temperature_K = temperature
                count += 1
        self._mark_dirty()
        self._refresh_all(reset_camera=False)
        self._sync_simulation_from_editor(reinitialize=True)
        self._set_status(f"Updated initial_temperature_K for {count} cells in {component}.")

    def _sync_component_temperature_input(self, *_: Any) -> None:
        if not hasattr(self, "component_temp_combo"):
            return
        component = self.component_temp_combo.currentText()
        if not component:
            return
        for node in self.model.nodes.values():
            if node.component_name == component:
                self.component_temp_input.blockSignals(True)
                self.component_temp_input.setValue(float(node.initial_temperature_K))
                self.component_temp_input.blockSignals(False)
                return

    def _handle_component_initial_temperature_changed(self, *_: Any) -> None:
        if self._building_form:
            return
        self.apply_component_initial_temperature()

    def _handle_theme_toggle(self, *_: Any) -> None:
        self.dark_mode = bool(self.theme_toggle.isChecked())
        self._apply_theme()
        self._refresh_all(reset_camera=False)
        if self.current_folder is not None:
            self._save_ui_state(self.current_folder)
        self._set_status("Dark mode enabled." if self.dark_mode else "Light mode enabled.")

    def _apply_theme(self) -> None:
        self.app.setStyleSheet(self._dark_stylesheet() if self.dark_mode else "")
        if hasattr(self, "viewer"):
            self.viewer.set_dark_mode(self.dark_mode)
        if hasattr(self, "two_d_view"):
            self.two_d_view.set_dark_mode(self.dark_mode)

    def _handle_tab_changed(self, index: int) -> None:
        current = self.view_tabs.widget(index)
        if hasattr(self, "side_panel_stack"):
            if current is self.simulation_tab.widget:
                self.side_panel_stack.setCurrentWidget(self.simulation_tab.controls_scroll)
            else:
                self.simulation_tab.pause()
                self.side_panel_stack.setCurrentWidget(self.left_scroll)
        if current is self.two_d_view.widget:
            self.two_d_view.visible_node_ids = self._filtered_node_ids()
            self.two_d_view.refresh()

    def _handle_close_event(self, event: Any) -> None:
        self.autosave_timer.stop()
        if hasattr(self, "simulation_tab"):
            self.simulation_tab.shutdown()
        if hasattr(self, "viewer"):
            self.viewer.close()
        event.accept()

    def _update_metadata_from_inputs(self, *_: Any) -> None:
        self.model.metadata.graph_name = self.graph_name_input.text().strip() or "untitled_graph"
        self.model.metadata.T_sur_K = float(self.T_sur_input.value())
        self.model.metadata.notes = self.notes_input.toPlainText()
        self.model.metadata.edge_mode = (
            EdgeMode.LOADED_G.value
            if self.loaded_conduction_radio.isChecked()
            else EdgeMode.AUTO.value
        )
        self.model.touch()

    def _handle_metadata_changed(self, *_: Any) -> None:
        if self._syncing_metadata_ui:
            return
        self._update_metadata_from_inputs()
        self._mark_dirty()

    def _sync_metadata_widgets(self) -> None:
        self._syncing_metadata_ui = True
        self.graph_name_input.setText(self.model.metadata.graph_name)
        self.T_sur_input.setValue(float(self.model.metadata.T_sur_K))
        self.notes_input.setPlainText(self.model.metadata.notes)
        self._syncing_metadata_ui = False
        self._sync_conduction_ui(self.model.metadata.edge_mode)

    def _choose_existing_folder(self, title: str) -> Path | None:
        folder = self.QtWidgets.QFileDialog.getExistingDirectory(self.window, title)
        return Path(folder) if folder else None

    def _choose_save_folder(self, title: str) -> Path | None:
        folder = self.QtWidgets.QFileDialog.getExistingDirectory(self.window, title)
        return Path(folder) if folder else None

    @staticmethod
    def _safe_graph_folder_name(graph_name: str) -> str:
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", graph_name.strip())
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
        return cleaned or "untitled_graph"

    def _set_status(self, message: str, error: bool = False) -> None:
        if self.dark_mode:
            color = "#fca5a5" if error else "#86efac"
        else:
            color = "#b00020" if error else "#1f6f3f"
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_label.setText(message)

    def _tooltip_for_node(self, node_id: int) -> str:
        node = self.model.nodes.get(node_id)
        return format_node_tooltip(node_id, node) if node is not None else ""

    def show(self) -> None:
        self.window.show()
        self._refresh_all(reset_camera=True)
        self.app.exec()

    def _group_box(self, title: str) -> Any:
        box = self.QtWidgets.QGroupBox(title)
        box.setStyleSheet("QGroupBox { font-weight: 700; margin-top: 8px; }")
        return box

    @staticmethod
    def _dark_stylesheet() -> str:
        return """
        QWidget {
            background-color: #111827;
            color: #e5e7eb;
            selection-background-color: #2563eb;
            selection-color: #ffffff;
        }
        QScrollArea, QTabWidget::pane {
            border: 1px solid #374151;
        }
        QGroupBox {
            border: 1px solid #374151;
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 10px;
            font-weight: 700;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
        }
        QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            background-color: #1f2937;
            color: #f9fafb;
            border: 1px solid #4b5563;
            border-radius: 4px;
            padding: 3px;
        }
        QLineEdit:read-only {
            color: #d1d5db;
            background-color: #172033;
        }
        QPushButton {
            background-color: #1f2937;
            color: #f9fafb;
            border: 1px solid #4b5563;
            border-radius: 4px;
            padding: 5px 8px;
        }
        QPushButton:hover {
            background-color: #374151;
        }
        QPushButton:pressed {
            background-color: #2563eb;
        }
        QTabBar::tab {
            background: #1f2937;
            color: #d1d5db;
            border: 1px solid #374151;
            padding: 6px 10px;
        }
        QTabBar::tab:selected {
            background: #111827;
            color: #ffffff;
            border-bottom-color: #111827;
        }
        QToolTip {
            background-color: #111827;
            color: #f9fafb;
            border: 1px solid #4b5563;
        }
        """

    def _checkbox(self, text: str, checked: bool, callback: Any | None = None) -> Any:
        widget = self.QtWidgets.QCheckBox(text)
        widget.setChecked(checked)
        if callback is not None:
            widget.stateChanged.connect(callback)
        return widget

    def _int_spin(self, minimum: int, maximum: int, value: int) -> Any:
        class NoWheelSpinBox(self.QtWidgets.QSpinBox):
            def wheelEvent(inner_self, event: Any) -> None:  # noqa: N802 - Qt override name.
                event.ignore()

        widget = NoWheelSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    def _double_spin(self, minimum: float, maximum: float, value: float, step: float) -> Any:
        class NoWheelDoubleSpinBox(self.QtWidgets.QDoubleSpinBox):
            def wheelEvent(inner_self, event: Any) -> None:  # noqa: N802 - Qt override name.
                event.ignore()

        widget = NoWheelDoubleSpinBox()
        widget.setDecimals(8)
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setValue(value)
        return widget

    @staticmethod
    def _parse_coord(text: str) -> tuple[int, int, int]:
        cleaned = text.strip()
        match = re.fullmatch(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", cleaned)
        if not match:
            raise ValueError("Not a coordinate.")
        return tuple(int(match.group(index)) for index in (1, 2, 3))

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any, indent: int | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_name = handle.name
                json.dump(payload, handle, indent=indent)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            if tmp_name:
                try:
                    Path(tmp_name).unlink(missing_ok=True)
                except OSError:
                    pass
