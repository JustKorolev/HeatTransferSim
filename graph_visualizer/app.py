"""Qt main window for constructing sparse 3D thermal graph networks."""

from __future__ import annotations

from datetime import datetime
import json
import re
from pathlib import Path
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
from .material_library import default_material_library
from .matrix_builder import refresh_auto_edges
from .models import (
    EdgeMode,
    GraphMetadata,
    HeaterProperties,
    NodeProperties,
    SensorProperties,
    ThermalGraphModel,
)
from .pyvista_widget import GraphPyVistaWidget
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
        self.inputs: dict[str, Any] = {}
        self._building_form = False
        self._syncing_conduction_ui = False
        self._syncing_metadata_ui = False
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
        self.show_heaters = self._checkbox("Heaters", True, self._handle_visual_toggle)
        self.show_sensors = self._checkbox("Sensors", True, self._handle_visual_toggle)
        for widget in (self.show_labels, self.show_edges, self.show_heaters, self.show_sensors):
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
        self.view_tabs.addTab(self.three_d_tab, "3D View")
        self.view_tabs.addTab(self.two_d_view.widget, "2D View")
        right_layout.addWidget(self.view_tabs, 1)

        self.status_label = self.QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        right_layout.addWidget(self.status_label)

        layout.addWidget(self.left_scroll, 0)
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

        self.inputs["has_heater"] = self._checkbox("has heater", False, self._update_optional_sections)
        self.heater_box = self._group_box("Heater")
        heater_form = self.QtWidgets.QFormLayout(self.heater_box)
        self.inputs["heater_id"] = self._int_spin(-1, 10**9, 0)
        self.inputs["heater_min_power_W"] = self._double_spin(0.0, 1.0e9, 0.0, 1.0)
        self.inputs["heater_max_power_W"] = self._double_spin(0.0, 1.0e9, 30.0, 1.0)
        self.inputs["heater_efficiency"] = self._double_spin(0.0, 1.0e6, 1.0, 0.05)
        for key in ("heater_id", "heater_min_power_W", "heater_max_power_W", "heater_efficiency"):
            heater_form.addRow(key, self.inputs[key])

        self.inputs["has_sensor"] = self._checkbox("has sensor", False, self._update_optional_sections)
        self.sensor_box = self._group_box("Sensor")
        sensor_form = self.QtWidgets.QFormLayout(self.sensor_box)
        self.inputs["sensor_id"] = self._int_spin(-1, 10**9, 0)
        self.inputs["sensor_noise_std_K"] = self._double_spin(0.0, 1.0e9, 0.0, 0.01)
        self.inputs["sensor_bias_K"] = self._double_spin(-1.0e9, 1.0e9, 0.0, 0.01)
        self.inputs["sensor_time_constant_s"] = self._double_spin(0.0, 1.0e9, 0.0, 0.01)
        for key in ("sensor_id", "sensor_noise_std_K", "sensor_bias_K", "sensor_time_constant_s"):
            sensor_form.addRow(key, self.inputs[key])

        layout.addLayout(form)
        layout.addWidget(self.inputs["has_heater"])
        layout.addWidget(self.heater_box)
        layout.addWidget(self.inputs["has_sensor"])
        layout.addWidget(self.sensor_box)

        button_row = self.QtWidgets.QHBoxLayout()
        self.apply_button = self.QtWidgets.QPushButton("Save Tags")
        self.apply_button.clicked.connect(self.apply_node_form)
        button_row.addWidget(self.apply_button)
        layout.addLayout(button_row)
        self.left_layout.addWidget(box)
        self._update_optional_sections()
        self._update_C_enabled()

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
        self._load_node_into_form(node)
        self._set_status("Ready to add a new cell.")

    def apply_node_form(self) -> None:
        try:
            node = self._node_from_form()
            if self.selected_node_id is None or self.selected_node_id not in self.model.nodes:
                raise ValueError("Select an existing octree cell before saving tags.")
            old_node = self.model.nodes[self.selected_node_id]
            old_node.has_heater = node.has_heater
            old_node.heater = node.heater
            old_node.has_sensor = node.has_sensor
            old_node.sensor = node.sensor
            old_node.notes = node.notes
            self._mark_dirty()
            self._refresh_all(reset_camera=False)
            self._set_status(f"Saved tags for cell {old_node.node_id}.")
        except Exception as exc:
            self._set_status(str(exc), error=True)

    def delete_selected_node(self) -> None:
        if self.selected_node_id is None:
            self._set_status("No selected cell to delete.", error=True)
            return
        self.model.delete_node(self.selected_node_id)
        self.selected_node_id = None
        self._handle_topology_changed()
        self._refresh_all(reset_camera=False)
        self._set_status("Deleted selected cell.")

    def select_node(self, node_id: int) -> None:
        if node_id not in self.model.nodes:
            return
        self.selected_node_id = node_id
        node = self.model.nodes[node_id]
        self._load_node_into_form(node)
        self._refresh_details()
        self.viewer.select_node(node_id)
        component = self._component_display(node)
        self._set_status(f"Selected cell {node_id}: {component}.")

    def _handle_viewer_pick(
        self,
        node_id: int,
        picked_point: tuple[float, float, float] | None = None,
        mouse_position: tuple[int, int] | None = None,
    ) -> None:
        if self._suppress_next_draw_pick:
            self._suppress_next_draw_pick = False
            return
        if self.draw_mode_enabled:
            if self.draw_active:
                self._suppress_next_draw_pick = True
                self.commit_draw_preview()
                return
            self.start_draw_from_face(node_id, picked_point, mouse_position)
        else:
            self.select_node(node_id)

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
        self.autosave_timer.start(1500)
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
        with (folder / "ui_state.json").open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)

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

    def _node_from_form(self) -> NodeProperties:
        node_id = int(self.inputs["node_id"].value())
        has_heater = self.inputs["has_heater"].isChecked()
        has_sensor = self.inputs["has_sensor"].isChecked()
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
            has_heater=has_heater,
            heater=HeaterProperties(
                heater_id=int(self.inputs["heater_id"].value()),
                heater_min_power_W=float(self.inputs["heater_min_power_W"].value()),
                heater_max_power_W=float(self.inputs["heater_max_power_W"].value()),
                heater_efficiency=float(self.inputs["heater_efficiency"].value()),
            ),
            has_sensor=has_sensor,
            sensor=SensorProperties(
                sensor_id=int(self.inputs["sensor_id"].value()),
                sensor_noise_std_K=float(self.inputs["sensor_noise_std_K"].value()),
                sensor_bias_K=float(self.inputs["sensor_bias_K"].value()),
                sensor_time_constant_s=float(self.inputs["sensor_time_constant_s"].value()),
            ),
            notes=self.inputs["notes"].toPlainText(),
        )
        if not node.C_manual_override:
            node.recompute_heat_capacity()
        if not has_heater:
            node.heater.heater_id = node_id
        if not has_sensor:
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
        self.inputs["notes"].setPlainText(node.notes)
        self.inputs["has_heater"].setChecked(node.has_heater)
        self.inputs["heater_id"].setValue(node.heater.heater_id or node.node_id)
        self.inputs["heater_min_power_W"].setValue(node.heater.heater_min_power_W)
        self.inputs["heater_max_power_W"].setValue(node.heater.heater_max_power_W)
        self.inputs["heater_efficiency"].setValue(node.heater.heater_efficiency)
        self.inputs["has_sensor"].setChecked(node.has_sensor)
        self.inputs["sensor_id"].setValue(node.sensor.sensor_id or node.node_id)
        self.inputs["sensor_noise_std_K"].setValue(node.sensor.sensor_noise_std_K)
        self.inputs["sensor_bias_K"].setValue(node.sensor.sensor_bias_K)
        self.inputs["sensor_time_constant_s"].setValue(node.sensor.sensor_time_constant_s)
        self._building_form = False
        self._update_optional_sections()
        self._update_C_enabled()

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
        if "has_heater" not in self.inputs:
            return
        self.heater_box.setVisible(self.inputs["has_heater"].isChecked())
        self.sensor_box.setVisible(self.inputs["has_sensor"].isChecked())
        node_id = int(self.inputs["node_id"].value())
        if self.inputs["heater_id"].value() == 0:
            self.inputs["heater_id"].setValue(node_id)
        if self.inputs["sensor_id"].value() == 0:
            self.inputs["sensor_id"].setValue(node_id)

    def _refresh_all(self, reset_camera: bool = False) -> None:
        self._sync_filter_options()
        visible_node_ids = self._filtered_node_ids()
        self.viewer.set_toggles(
            self.show_labels.isChecked(),
            self.show_edges.isChecked(),
            self.show_heaters.isChecked(),
            self.show_sensors.isChecked(),
        )
        self.viewer.set_draw_mode(self.draw_mode_enabled)
        self.viewer.selected_node_id = self.selected_node_id
        self.viewer.draw(self.model, reset_camera=reset_camera, visible_node_ids=visible_node_ids)
        self.two_d_view.set_model(
            self.model,
            visible_node_ids=visible_node_ids,
            auto_refresh=self.view_tabs.currentWidget() is self.two_d_view.widget,
        )
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
            if edge.edge_type not in {"internal_conduction", "same_material_spatial"}
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
            if self.filter_heater_sensor.isChecked() and not (node.has_heater or node.has_sensor):
                continue
            if self.filter_boundary.isChecked() and node.confidence == "high" and node_id not in contact_nodes:
                continue
            visible.add(node_id)
        return visible

    def _refresh_details(self) -> None:
        if self.selected_node_id is None or self.selected_node_id not in self.model.nodes:
            self.details_label.setText("No cell selected.")
            return
        node = self.model.nodes[self.selected_node_id]
        incident = [
            edge for edge in self.model.edges.values()
            if edge.source == node.node_id or edge.target == node.node_id
        ]
        self.details_label.setText(
            f"node_id: {node.node_id}\n"
            f"cell_id: {node.cell_id or node.coord}\n"
            f"part code: {self._part_code(node.component_name)}\n"
            f"component: {node.component_name or '?'}\n"
            f"center_mm: {node.center}\n"
            f"size_mm: {node.size_mm or node.side_length_m}\n"
            f"material: {node.material}\n"
            f"level: {node.level}, confidence: {node.confidence}\n"
            f"C: {node.C_J_K:.6g} J/K, Grad: {node.Grad_W_K:.6g} W/K\n"
            f"heater: {node.has_heater}, sensor: {node.has_sensor}\n"
            f"incident conductive edges: {len(incident)}"
        )

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
        if self.view_tabs.widget(index) is self.two_d_view.widget:
            self.two_d_view.visible_node_ids = self._filtered_node_ids()
            self.two_d_view.refresh()

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
        widget = self.QtWidgets.QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    def _double_spin(self, minimum: float, maximum: float, value: float, step: float) -> Any:
        widget = self.QtWidgets.QDoubleSpinBox()
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
