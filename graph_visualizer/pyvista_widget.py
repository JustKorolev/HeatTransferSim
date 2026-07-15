"""PyVista/PyVistaQt widget for sparse thermal graph visualization."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from matplotlib import colormaps

from .diagnostics import log_event
from .models import ThermalGraphModel
from .role_warnings import has_role_warning


class GraphPyVistaWidget:
    """Thin wrapper around QtInteractor that draws graph cells and edges."""

    def __init__(
        self,
        parent: Any,
        on_pick_node: Callable[
            [int, tuple[float, float, float] | None, tuple[int, int] | None], None
        ]
        | None = None,
        on_drag_update: Callable[[tuple[int, int] | None], None] | None = None,
        on_left_click: Callable[[], None] | None = None,
        on_drag_release: Callable[[], None] | None = None,
        on_escape: Callable[[], None] | None = None,
        tooltip_for_node: Callable[[int], str] | None = None,
    ) -> None:
        self._load_dependencies()
        self.parent = parent
        self.on_pick_node = on_pick_node
        self.on_drag_update = on_drag_update
        self.on_left_click = on_left_click
        self.on_drag_release = on_drag_release
        self.on_escape = on_escape
        self.tooltip_for_node = tooltip_for_node
        self.plotter = self.QtInteractor(parent)
        self.plotter.set_background("white")
        self.dark_mode = False
        self.selected_node_id: int | None = None
        self.selected_node_ids: set[int] = set()
        self.draw_mode_enabled = False
        self.show_labels = True
        self.show_edges = True
        self.show_heaters = True
        self.show_sensors = True
        self.show_coolers = True
        self.shader_mode_enabled = False
        self.cell_opacity = 0.34
        self.depth_focus_enabled = False
        self.depth_focus_axis = "z"
        self.depth_focus_fraction = 0.5
        self.depth_focus_width = 0.12
        self._node_actors: dict[int, Any] = {}
        self._node_meshes: dict[int, Any] = {}
        self._actor_node_ids: dict[str, int] = {}
        self._label_actors: list[Any] = []
        self._marker_actors: list[Any] = []
        self._marker_actors_by_kind: dict[str, list[Any]] = {"heater": [], "sensor": [], "cooler": []}
        self._edge_actors: list[Any] = []
        self._role_overlay_actors: list[Any] = []
        self._preview_actors: list[Any] = []
        self._batched_actor: Any | None = None
        self._batched_mesh: Any | None = None
        self._batched_node_geometry: dict[int, tuple[np.ndarray, tuple[float, float, float]]] = {}
        self._batched_selected_actor: Any | None = None
        self._last_preview_coords: list[tuple[int, int, int]] = []
        self._last_preview_side = 1.0
        self._picking_enabled = False
        self._observers_enabled = False
        self._ignore_next_mesh_pick = False
        self._hover_node_id: int | None = None
        self._hover_tooltips_enabled = True
        self._material_colors: dict[str, str] = {}
        self._last_model_nodes: dict[int, Any] = {}
        self._last_committed_bounds: tuple[float, float, float, float, float, float] | None = None
        self._closed = False

    def _load_dependencies(self) -> None:
        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor
        except ImportError as exc:
            raise RuntimeError(
                "graph_visualizer requires pyvista and pyvistaqt for the 3D view. "
                "Install UI dependencies later with pip install -r requirements.txt."
            ) from exc
        try:
            from PySide6 import QtCore, QtGui, QtWidgets
        except ImportError:
            from qtpy import QtCore, QtGui, QtWidgets
        self.pv = pv
        self.QtInteractor = QtInteractor
        self.QtCore = QtCore
        self.QtGui = QtGui
        self.QtWidgets = QtWidgets

    @property
    def interactor(self) -> Any:
        return self.plotter.interactor

    def close(self) -> None:
        """Stop future renders before Qt destroys the OpenGL handle."""
        self._closed = True
        try:
            self.QtWidgets.QToolTip.hideText()
        except Exception:
            pass
        for method_name in ("close", "Finalize", "finalize"):
            method = getattr(self.plotter, method_name, None)
            if method is None:
                continue
            try:
                method()
                break
            except Exception:
                pass

    def safe_render(self) -> bool:
        """Render only while the Qt/VTK window handle is still valid."""
        if self._closed:
            return False
        try:
            widget = self.interactor
            if hasattr(widget, "isVisible") and not widget.isVisible():
                return False
            if hasattr(widget, "winId") and int(widget.winId()) == 0:
                return False
            self.plotter.render()
            return True
        except RuntimeError:
            self._closed = True
            return False
        except Exception:
            return False

    def set_toggles(
        self,
        show_labels: bool,
        show_edges: bool,
        show_heaters: bool,
        show_sensors: bool,
        show_coolers: bool = True,
    ) -> None:
        self.show_labels = show_labels
        self.show_edges = show_edges
        self.show_heaters = show_heaters
        self.show_sensors = show_sensors
        self.show_coolers = show_coolers

    def update_io_marker_visibility(
        self,
        show_heaters: bool,
        show_sensors: bool,
        show_coolers: bool,
        render: bool = True,
    ) -> bool:
        """Toggle existing I/O marker actors without rebuilding graph geometry."""
        self.show_heaters = bool(show_heaters)
        self.show_sensors = bool(show_sensors)
        self.show_coolers = bool(show_coolers)
        for kind, visible in (
            ("heater", self.show_heaters),
            ("sensor", self.show_sensors),
            ("cooler", self.show_coolers),
        ):
            for actor in self._marker_actors_by_kind.get(kind, []):
                self._set_actor_visible(actor, visible)
        return self.safe_render() if render else True

    def set_dark_mode(self, enabled: bool) -> None:
        self.dark_mode = bool(enabled)
        self.plotter.set_background("#111827" if self.dark_mode else "white")

    def set_shader_mode(self, enabled: bool, render: bool = True) -> bool:
        """Toggle opaque, lit cell rendering for geometry inspection."""
        self.shader_mode_enabled = bool(enabled)
        self._apply_shader_mode_to_scene()
        return self.safe_render() if render else True

    def toggle_shader_mode(self) -> bool:
        return self.set_shader_mode(not self.shader_mode_enabled)

    def set_cell_opacity(self, opacity: float, render: bool = True) -> bool:
        self.cell_opacity = max(0.02, min(1.0, float(opacity)))
        self._apply_shader_mode_to_scene()
        return self.safe_render() if render else True

    def set_depth_focus(
        self,
        enabled: bool,
        fraction: float | None = None,
        axis: str | None = None,
        width: float | None = None,
        render: bool = True,
    ) -> bool:
        self.depth_focus_enabled = bool(enabled)
        if fraction is not None:
            self.depth_focus_fraction = max(0.0, min(1.0, float(fraction)))
        if axis is not None:
            normalized_axis = str(axis).strip().lower()
            if normalized_axis in {"x", "y", "z"}:
                self.depth_focus_axis = normalized_axis
        if width is not None:
            self.depth_focus_width = max(0.01, min(1.0, float(width)))
        self._apply_shader_mode_to_scene()
        return self.safe_render() if render else True

    def draw(
        self,
        model: ThermalGraphModel,
        reset_camera: bool = False,
        visible_node_ids: set[int] | None = None,
        node_colors: dict[int, str] | None = None,
        node_scalar_values: dict[int, float] | None = None,
        scalar_cmap: str = "jet",
        scalar_clim: tuple[float, float] | None = None,
        scalar_bar_title: str = "Temperature [K]",
    ) -> None:
        log_event(
            "pyvista draw start",
            nodes=len(model.nodes),
            edges=len(model.edges),
            reset_camera=reset_camera,
        )
        camera_position = self.plotter.camera_position if not reset_camera else None
        preview_coords = list(getattr(self, "_last_preview_coords", []))
        preview_side = float(getattr(self, "_last_preview_side", 1.0))
        committed_bounds = self._committed_model_bounds(model)
        self._last_model_nodes = dict(model.nodes)
        self._last_committed_bounds = committed_bounds
        self.plotter.clear()
        self._node_actors = {}
        self._node_meshes = {}
        self._actor_node_ids = {}
        self._label_actors = []
        self._marker_actors = []
        self._marker_actors_by_kind = {"heater": [], "sensor": [], "cooler": []}
        self._edge_actors = []
        self._role_overlay_actors = []
        self._batched_actor = None
        self._batched_mesh = None
        self._batched_node_geometry = {}
        self._batched_selected_actor = None
        visible = visible_node_ids if visible_node_ids is not None else set(model.ordered_node_ids())
        log_event("pyvista draw visible prepared", visible=len(visible))
        if len(visible) > 1200:
            log_event("pyvista draw using batched renderer", visible=len(visible))
            self._draw_batched(
                model,
                visible,
                node_colors=node_colors,
                node_scalar_values=node_scalar_values,
                scalar_cmap=scalar_cmap,
                scalar_clim=scalar_clim,
                scalar_bar_title=scalar_bar_title,
                committed_bounds=committed_bounds,
            )
            log_event("pyvista draw batched mesh complete")
            self._draw_role_interface_overlays(model, visible)
            log_event("pyvista draw overlays complete")
            self._finish_scene(committed_bounds, camera_position, reset_camera)
            log_event("pyvista draw finish_scene complete")
            return
        scalar_bar_added = False
        for node_id in model.ordered_node_ids():
            if node_id not in visible:
                continue
            node = model.nodes[node_id]
            geometry = _safe_node_cube_geometry(node)
            if geometry is None:
                continue
            center, lengths = geometry
            depth_focused = self._node_in_depth_focus(center, committed_bounds)
            marker_side = max(lengths)
            mesh = self.pv.Cube(
                center=center,
                x_length=lengths[0],
                y_length=lengths[1],
                z_length=lengths[2],
            )
            mesh.field_data["node_id"] = np.array([node_id], dtype=int)
            selected = node_id == self.selected_node_id or node_id in self.selected_node_ids
            color = (node_colors or {}).get(node_id, self._color_for_material(node.material))
            mesh_kwargs = {
                "opacity": self._cell_opacity(node_scalar_values is not None, selected, depth_focused),
                "show_edges": True,
                "edge_color": "#f87171" if selected else self._node_edge_color(),
                "line_width": 3 if selected else 1,
                "pickable": True,
                **self._lit_mesh_kwargs(),
            }
            if node_scalar_values is not None and node_id in node_scalar_values:
                mesh.cell_data["temperature_K"] = np.full(mesh.n_cells, float(node_scalar_values[node_id]))
                actor = self.plotter.add_mesh(
                    mesh,
                    scalars="temperature_K",
                    cmap=scalar_cmap,
                    clim=scalar_clim,
                    show_scalar_bar=not scalar_bar_added,
                    scalar_bar_args=self._scalar_bar_args(scalar_bar_title),
                    **mesh_kwargs,
                )
                scalar_bar_added = True
            else:
                actor = self.plotter.add_mesh(
                    mesh,
                    color="#ffd166" if selected else color,
                    **mesh_kwargs,
                )
            self._node_actors[node_id] = actor
            self._node_meshes[node_id] = mesh
            self._actor_node_ids[self._actor_key(actor)] = node_id
            self._enable_actor_pick(actor)
            if self.show_labels:
                label_actor = self.plotter.add_point_labels(
                    np.array([center], dtype=float),
                    [str(node_id)],
                    font_size=13,
                    text_color=self._label_color(),
                    shape_opacity=0.0,
                    always_visible=True,
                )
                self._label_actors.append(label_actor)
            self._add_io_markers_for_node(node, center, marker_side)

        if self.show_edges:
            for edge in model.edges.values():
                if edge.source not in model.nodes or edge.target not in model.nodes:
                    continue
                if edge.source not in visible or edge.target not in visible:
                    continue
                p0 = np.array(model.nodes[edge.source].center, dtype=float)
                p1 = np.array(model.nodes[edge.target].center, dtype=float)
                line = self.pv.Line(p0, p1)
                actor = self.plotter.add_mesh(line, color=self._edge_color(), line_width=3)
                self._edge_actors.append(actor)

        self._draw_role_interface_overlays(model, visible)
        self._finish_scene(committed_bounds, camera_position, reset_camera)
        log_event("pyvista draw complete")
        if preview_coords:
            self.show_preview(preview_coords, preview_side)

    def select_node(self, node_id: int | None, model: ThermalGraphModel | None = None) -> None:
        self.select_nodes(set() if node_id is None else {int(node_id)}, active_node_id=node_id)

    def select_nodes(self, node_ids: set[int], active_node_id: int | None = None) -> None:
        previous = set(self.selected_node_ids)
        previous_active = self.selected_node_id
        self.selected_node_ids = {int(node_id) for node_id in node_ids}
        self.selected_node_id = active_node_id if active_node_id is not None else next(iter(self.selected_node_ids), None)
        if previous == self.selected_node_ids and previous_active == self.selected_node_id:
            return
        for node_id in previous | self.selected_node_ids:
            if node_id in self._node_actors:
                self._style_node_actor(self._node_actors[node_id], selected=node_id in self.selected_node_ids)
        if self._batched_node_geometry:
            self._show_batched_selection(self.selected_node_ids)
        self.safe_render()

    def update_node_colors(self, node_colors: dict[int, str]) -> None:
        """Update visible node actor colors without rebuilding geometry."""
        for node_id, actor in self._node_actors.items():
            if node_id == self.selected_node_id or node_id in self.selected_node_ids:
                continue
            color = node_colors.get(node_id)
            if color is None:
                continue
            try:
                actor.prop.color = color
                actor.GetProperty().Modified()
            except Exception:
                pass
        try:
            self.safe_render()
        except Exception:
            pass

    def update_node_scalars(
        self,
        node_scalar_values: dict[int, float],
        scalar_clim: tuple[float, float] | None = None,
    ) -> bool:
        """Update temperature scalars in-place without rebuilding axes or bounds."""
        if self._batched_actor is not None and self._batched_mesh is not None:
            try:
                node_ids = np.asarray(self._batched_mesh.cell_data["node_id"], dtype=int)
                scalars = np.array(
                    [float(node_scalar_values.get(int(node_id), np.nan)) for node_id in node_ids],
                    dtype=float,
                )
                self._update_actor_cell_scalars(
                    self._batched_actor,
                    self._batched_mesh,
                    scalars,
                    scalar_clim,
                )
                self.safe_render()
                return True
            except Exception:
                return self._update_actor_direct_colors(node_scalar_values, scalar_clim)

        if not self._node_meshes:
            return False
        updated = False
        for node_id, mesh in self._node_meshes.items():
            if node_id not in node_scalar_values:
                continue
            try:
                actor = self._node_actors.get(node_id)
                if actor is not None:
                    scalars = np.full(mesh.n_cells, float(node_scalar_values[node_id]))
                    updated = self._update_actor_cell_scalars(
                        actor,
                        mesh,
                        scalars,
                        scalar_clim,
                    ) or updated
            except Exception:
                continue
        if updated:
            self._update_actor_direct_colors(node_scalar_values, scalar_clim)
            self.safe_render()
        if not updated:
            return self._update_actor_direct_colors(node_scalar_values, scalar_clim)
        return True

    def set_draw_mode(self, enabled: bool) -> None:
        self.draw_mode_enabled = bool(enabled)

    def set_hover_tooltips_enabled(self, enabled: bool) -> None:
        self._hover_tooltips_enabled = bool(enabled)
        if not self._hover_tooltips_enabled and self._hover_node_id is not None:
            self._hover_node_id = None
            try:
                self.QtWidgets.QToolTip.hideText()
            except Exception:
                pass

    def show_preview(self, coords: list[tuple[int, int, int]], side_length_m: float) -> None:
        normalized_coords = list(coords)
        normalized_side = float(side_length_m) if side_length_m > 0.0 else 1.0
        if (
            normalized_coords == self._last_preview_coords
            and abs(normalized_side - self._last_preview_side) <= 1.0e-12
        ):
            return
        self.clear_preview(render=False)
        self._last_preview_coords = normalized_coords
        self._last_preview_side = normalized_side
        side = max(1.0e-6, self._last_preview_side)
        for coord in normalized_coords:
            center = np.array(coord, dtype=float)
            mesh = self.pv.Cube(center=center, x_length=side, y_length=side, z_length=side)
            actor = self._add_preview_mesh(mesh)
            self._exclude_actor_from_bounds(actor)
            self._preview_actors.append(actor)
        self.safe_render()

    def clear_preview(self, render: bool = True) -> None:
        self._last_preview_coords = []
        self._last_preview_side = 1.0
        for actor in self._preview_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self._preview_actors = []
        if render:
            try:
                self.safe_render()
            except Exception:
                pass

    def _remove_bounds_axes(self) -> None:
        for method_name in ("remove_bounds_axes", "remove_bounds_axis"):
            method = getattr(self.plotter, method_name, None)
            if method is None:
                continue
            try:
                method()
                return
            except Exception:
                pass

    def screen_direction_for_grid_normal(
        self, normal: tuple[int, int, int]
    ) -> tuple[float, float] | None:
        """Return the clicked face normal projected into screen x/y movement."""
        try:
            camera_position = self.plotter.camera_position
            camera_location = np.asarray(camera_position[0], dtype=float)
            focal_point = np.asarray(camera_position[1], dtype=float)
            view_up = np.asarray(camera_position[2], dtype=float)
        except Exception:
            return None

        view_direction = focal_point - camera_location
        view_norm = float(np.linalg.norm(view_direction))
        up_norm = float(np.linalg.norm(view_up))
        if view_norm <= 1.0e-9 or up_norm <= 1.0e-9:
            return None
        view_direction = view_direction / view_norm
        view_up = view_up / up_norm
        right = np.cross(view_direction, view_up)
        right_norm = float(np.linalg.norm(right))
        if right_norm <= 1.0e-9:
            return None
        right = right / right_norm
        up = np.cross(right, view_direction)
        up_norm = float(np.linalg.norm(up))
        if up_norm <= 1.0e-9:
            return None
        up = up / up_norm

        normal_world = np.asarray(normal, dtype=float)
        screen_direction = np.array(
            [float(np.dot(normal_world, right)), float(np.dot(normal_world, up))]
        )
        if float(np.linalg.norm(screen_direction)) <= 1.0e-9:
            return None
        return (float(screen_direction[0]), float(screen_direction[1]))

    def screen_step_for_grid_normal(
        self,
        center: tuple[float, float, float],
        normal: tuple[int, int, int],
        grid_spacing: float = 1.0,
    ) -> tuple[tuple[float, float] | None, float | None]:
        """Return screen direction and pixels for one grid step along a normal."""
        center_array = np.asarray(center, dtype=float)
        normal_array = np.asarray(normal, dtype=float)
        start = self._world_to_display(center_array)
        end = self._world_to_display(center_array + normal_array * float(grid_spacing))
        if start is None or end is None:
            return self.screen_direction_for_grid_normal(normal), None
        vector = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        length = float(np.linalg.norm(vector))
        if length <= 1.0e-9 or not np.isfinite(length):
            return self.screen_direction_for_grid_normal(normal), None
        direction = vector / length
        return (float(direction[0]), float(direction[1])), length

    def _add_marker(
        self,
        center: np.ndarray,
        color: str,
        label: str,
        kind: str,
        visible: bool,
        *,
        point_size: int = 15,
    ) -> None:
        actor = self.plotter.add_points(
            np.array([center], dtype=float),
            color=color,
            point_size=point_size,
            render_points_as_spheres=True,
        )
        self._set_actor_visible(actor, visible)
        self._marker_actors.append(actor)
        self._marker_actors_by_kind.setdefault(kind, []).append(actor)
        label_actor = self.plotter.add_point_labels(
            np.array([center], dtype=float),
            [label],
            font_size=10,
            text_color=color,
            shape_opacity=0.0,
            always_visible=True,
        )
        self._set_actor_visible(label_actor, visible)
        self._marker_actors.append(label_actor)
        self._marker_actors_by_kind.setdefault(kind, []).append(label_actor)

    def _add_io_markers_for_node(self, node: Any, center: np.ndarray, marker_side: float) -> None:
        if bool(getattr(node, "is_heater", False)):
            warning = has_role_warning(node)
            self._add_marker(
                center + np.array([0.0, 0.0, 0.36 * marker_side]),
                "#ef4444" if warning else "#ff6b35",
                "H!" if warning else "H",
                "heater",
                self.show_heaters,
                point_size=20 if warning else 15,
            )
        if bool(getattr(node, "is_sensor", False)):
            warning = has_role_warning(node)
            self._add_marker(
                center + np.array([0.0, 0.0, -0.36 * marker_side]),
                "#ef4444" if warning else "#2a9d8f",
                "S!" if warning else "S",
                "sensor",
                self.show_sensors,
                point_size=20 if warning else 15,
            )
        if bool(getattr(node, "has_cryocooler", False)):
            self._add_marker(
                center + np.array([-0.36 * marker_side, 0.0, 0.0]),
                "#06b6d4",
                "C",
                "cooler",
                self.show_coolers,
            )

    def _draw_role_interface_overlays(self, model: ThermalGraphModel, visible: set[int]) -> None:
        if self.show_heaters:
            for heater in model.nodes.values():
                if not bool(getattr(heater, "is_heater", False)):
                    continue
                for body_id in getattr(heater, "power_deposition_node_ids", []) or []:
                    self._add_node_outline(model, int(body_id), visible, "#f97316", 0.45)
                sensor_id = getattr(heater, "assigned_sensor_id", None)
                if sensor_id is not None and int(sensor_id) in model.nodes:
                    self._add_pair_line(heater, model.nodes[int(sensor_id)])
                if has_role_warning(heater):
                    self._add_node_outline(model, int(heater.node_id), visible, "#ef4444", 0.75)
        if self.show_sensors:
            for sensor in model.nodes.values():
                if not bool(getattr(sensor, "is_sensor", False)):
                    continue
                for body_id in (getattr(sensor, "readout_node_ids", []) or getattr(sensor, "sensor_connected_node_ids", []) or []):
                    self._add_node_outline(model, int(body_id), visible, "#14b8a6", 0.38)
                if has_role_warning(sensor):
                    self._add_node_outline(model, int(sensor.node_id), visible, "#ef4444", 0.75)

    def _add_node_outline(
        self,
        model: ThermalGraphModel,
        node_id: int,
        visible: set[int],
        color: str,
        opacity: float,
    ) -> None:
        if int(node_id) not in visible or int(node_id) not in model.nodes:
            return
        geometry = _safe_node_cube_geometry(model.nodes[int(node_id)])
        if geometry is None:
            return
        center, lengths = geometry
        inflated = tuple(max(float(length) * 1.06, float(length) + 1.0e-6) for length in lengths)
        mesh = self.pv.Cube(
            center=center,
            x_length=inflated[0],
            y_length=inflated[1],
            z_length=inflated[2],
        )
        actor = self.plotter.add_mesh(
            mesh,
            color=color,
            style="wireframe",
            line_width=2,
            opacity=float(opacity),
            pickable=False,
        )
        self._role_overlay_actors.append(actor)

    def _add_pair_line(self, heater: Any, sensor: Any) -> None:
        try:
            p0 = np.asarray(heater.center, dtype=float)
            p1 = np.asarray(sensor.center, dtype=float)
        except Exception:
            return
        if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(p1)):
            return
        line = self.pv.Line(p0, p1)
        distance = getattr(heater, "sensor_pair_distance_mm", None)
        color = "#eab308" if distance is not None and float(distance) > 0.0 else "#22c55e"
        actor = self.plotter.add_mesh(line, color=color, line_width=4, opacity=0.8, pickable=False)
        self._role_overlay_actors.append(actor)

    @staticmethod
    def _set_actor_visible(actor: Any, visible: bool) -> None:
        for method_name in ("SetVisibility", "SetPickable"):
            method = getattr(actor, method_name, None)
            if method is None:
                continue
            try:
                method(bool(visible))
            except Exception:
                pass
        prop = getattr(actor, "prop", None)
        if prop is not None:
            try:
                prop.visibility = bool(visible)
            except Exception:
                pass

    def _style_node_actor(self, actor: Any, selected: bool) -> None:
        prop = actor.GetProperty()
        if selected:
            prop.SetColor(1.0, 0.8196078431, 0.4)
            prop.SetOpacity(self._cell_opacity(False, True))
            prop.SetEdgeColor(0.9725490196, 0.4431372549, 0.4431372549)
            prop.SetLineWidth(3)
        else:
            prop.SetColor(0.3450980392, 0.6509803922, 1.0)
            prop.SetOpacity(self._cell_opacity(self._actor_has_temperature_scalars(actor), False))
            prop.SetEdgeColor(*(self._node_edge_rgb()))
            prop.SetLineWidth(1)
        self._apply_lighting_to_actor(actor)
        prop.Modified()

    def _enable_actor_pick(self, actor: Any) -> None:
        try:
            actor.SetPickable(True)
        except AttributeError:
            pass

    def _enable_mesh_picking_once(self) -> None:
        if self._picking_enabled:
            return
        try:
            self.plotter.enable_mesh_picking(
                callback=self._handle_mesh_pick,
                left_clicking=True,
                show=False,
            )
            self._picking_enabled = True
        except Exception:
            pass

    def _enable_interaction_observers_once(self) -> None:
        if self._observers_enabled:
            return
        try:
            interactor = self.plotter.iren.interactor
        except AttributeError:
            try:
                interactor = self.plotter.interactor
            except AttributeError:
                return
        try:
            interactor.AddObserver("LeftButtonPressEvent", self._handle_left_press)
            interactor.AddObserver("MouseMoveEvent", self._handle_mouse_move)
            interactor.AddObserver("LeftButtonReleaseEvent", self._handle_left_release)
            interactor.AddObserver("KeyPressEvent", self._handle_key_press)
            self._observers_enabled = True
        except Exception:
            pass

    def _handle_left_press(self, *_: Any) -> None:
        if self.on_left_click is not None:
            self.on_left_click()
            if self._ignore_next_mesh_pick:
                return
        pick = self._pick_node_at_mouse()
        if pick is None:
            return
        self._ignore_next_mesh_pick = True
        node_id, picked_point, mouse_position = pick
        self._handle_pick(node_id, picked_point, mouse_position)

    def _handle_mesh_pick(self, picked_mesh: Any, *_: Any) -> None:
        if self._ignore_next_mesh_pick:
            self._ignore_next_mesh_pick = False
            return
        try:
            node_id = int(np.asarray(picked_mesh.field_data["node_id"])[0])
        except Exception:
            return
        self._handle_pick(node_id, self._picked_point(), self._mouse_position())

    def _handle_pick(
        self,
        node_id: int,
        picked_point: tuple[float, float, float] | None,
        mouse_position: tuple[int, int] | None,
    ) -> None:
        if self.on_pick_node is not None:
            self.on_pick_node(node_id, picked_point, mouse_position, self._ctrl_modifier_active())

    def _handle_mouse_move(self, *_: Any) -> None:
        if self._closed:
            return
        if self.draw_mode_enabled and self.on_drag_update is not None:
            self.on_drag_update(self._mouse_position())
            return
        if not self._hover_tooltips_enabled:
            return
        self._update_hover_tooltip()

    def _handle_left_release(self, *_: Any) -> None:
        if self.draw_mode_enabled and self.on_drag_release is not None:
            self.on_drag_release()

    def _handle_key_press(self, *_: Any) -> None:
        key = ""
        try:
            key = self.plotter.iren.interactor.GetKeySym()
        except AttributeError:
            try:
                key = self.plotter.interactor.GetKeySym()
            except AttributeError:
                pass
        if key == "Escape" and self.on_escape is not None:
            self.on_escape()
        elif key in {"s", "S"}:
            self.toggle_shader_mode()

    def _ctrl_modifier_active(self) -> bool:
        try:
            modifiers = self.QtWidgets.QApplication.keyboardModifiers()
            return bool(modifiers & self.QtCore.Qt.ControlModifier)
        except Exception:
            return False

    def _picked_point(self) -> tuple[float, float, float] | None:
        for owner in (self.plotter, getattr(self.plotter, "iren", None)):
            point = getattr(owner, "picked_point", None)
            if point is not None:
                try:
                    values = tuple(float(v) for v in point)
                    if len(values) == 3:
                        return values
                except (TypeError, ValueError):
                    pass
        return None

    def _mouse_position(self) -> tuple[int, int] | None:
        try:
            x, y = self.plotter.iren.interactor.GetEventPosition()
            return (int(x), int(y))
        except AttributeError:
            try:
                x, y = self.plotter.interactor.GetEventPosition()
                return (int(x), int(y))
            except AttributeError:
                return None

    def _pick_node_at_mouse(
        self,
    ) -> tuple[int, tuple[float, float, float] | None, tuple[int, int] | None] | None:
        if self._closed:
            return None
        mouse_position = self._mouse_position()
        if mouse_position is None:
            return None
        interactor = self._vtk_interactor()
        renderer = self._renderer()
        if interactor is None or renderer is None:
            return None
        picker = self._picker(interactor)
        if picker is None:
            return None
        try:
            picked = picker.Pick(mouse_position[0], mouse_position[1], 0, renderer)
        except Exception:
            return None
        if not picked:
            return None
        try:
            actor = picker.GetActor()
        except Exception:
            actor = None
        node_id = self._node_id_for_actor(actor)
        if node_id is None and self._is_batched_actor(actor):
            try:
                cell_id = int(picker.GetCellId())
                if self._batched_mesh is not None and cell_id >= 0:
                    node_id = int(self._batched_mesh.cell_data["node_id"][cell_id])
            except Exception:
                node_id = None
        if node_id is None:
            return None
        picked_point = None
        try:
            picked_point = tuple(float(v) for v in picker.GetPickPosition())
        except Exception:
            pass
        return node_id, picked_point, mouse_position

    def _update_hover_tooltip(self) -> None:
        if self.tooltip_for_node is None:
            return
        pick = self._pick_node_at_mouse()
        if pick is None:
            if self._hover_node_id is not None:
                self._hover_node_id = None
                self.QtWidgets.QToolTip.hideText()
            return
        node_id, _picked_point, mouse_position = pick
        if mouse_position is None:
            return
        self._hover_node_id = node_id
        tooltip = self.tooltip_for_node(node_id)
        try:
            global_pos = self.QtGui.QCursor.pos() + self.QtCore.QPoint(14, 18)
            self.QtWidgets.QToolTip.showText(global_pos, tooltip, self.interactor)
        except Exception:
            self.interactor.setToolTip(tooltip)

    def _vtk_interactor(self) -> Any | None:
        try:
            return self.plotter.iren.interactor
        except AttributeError:
            try:
                return self.plotter.interactor
            except AttributeError:
                return None

    def _renderer(self) -> Any | None:
        for name in ("renderer", "ren"):
            renderer = getattr(self.plotter, name, None)
            if renderer is not None:
                return renderer
        try:
            return self.plotter.renderers[0]
        except Exception:
            return None

    def _picker(self, interactor: Any) -> Any | None:
        try:
            picker = interactor.GetPicker()
            if picker is not None:
                return picker
        except Exception:
            pass
        try:
            from vtkmodules.vtkRenderingCore import vtkCellPicker

            picker = vtkCellPicker()
            picker.SetTolerance(0.0005)
            interactor.SetPicker(picker)
            return picker
        except Exception:
            return None

    def _node_id_for_actor(self, actor: Any) -> int | None:
        if actor is None:
            return None
        key = self._actor_key(actor)
        if key in self._actor_node_ids:
            return self._actor_node_ids[key]
        for node_id, node_actor in self._node_actors.items():
            if actor is node_actor or actor == node_actor:
                return node_id
        return None

    def _is_batched_actor(self, actor: Any) -> bool:
        if actor is None or self._batched_actor is None:
            return False
        if actor is self._batched_actor or actor == self._batched_actor:
            return True
        return self._actor_key(actor) == self._actor_key(self._batched_actor)

    def _world_to_display(self, point: np.ndarray) -> tuple[float, float] | None:
        renderer = self._renderer()
        if renderer is None:
            return None
        try:
            renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
            renderer.WorldToDisplay()
            display = renderer.GetDisplayPoint()
            return (float(display[0]), float(display[1]))
        except Exception:
            return None

    @staticmethod
    def _committed_model_bounds(
        model: ThermalGraphModel,
    ) -> tuple[float, float, float, float, float, float] | None:
        """Bounds for real cells only; draw-mode preview cells must not affect the grid."""
        if not model.nodes:
            return None
        mins = np.array([np.inf, np.inf, np.inf], dtype=float)
        maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        for node in model.nodes.values():
            geometry = _safe_node_cube_geometry(node)
            if geometry is None:
                continue
            center, lengths = geometry
            half = np.array(lengths, dtype=float) * 0.5
            mins = np.minimum(mins, center - half)
            maxs = np.maximum(maxs, center + half)
        if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
            return None
        padding = np.maximum(0.5, 0.05 * np.maximum(maxs - mins, 1.0))
        mins -= padding
        maxs += padding
        return (
            float(mins[0]),
            float(maxs[0]),
            float(mins[1]),
            float(maxs[1]),
            float(mins[2]),
            float(maxs[2]),
        )

    @staticmethod
    def _exclude_actor_from_bounds(actor: Any) -> None:
        """Keep ghost previews from expanding bounds/grid/camera extents."""
        for method_name in ("SetUseBounds", "UseBoundsOff"):
            method = getattr(actor, method_name, None)
            if method is None:
                continue
            try:
                if method_name == "SetUseBounds":
                    method(False)
                else:
                    method()
                return
            except Exception:
                pass

    def _add_preview_mesh(self, mesh: Any) -> Any:
        kwargs = {
            "color": "#9b5de5",
            "opacity": 0.22,
            "show_edges": True,
            "edge_color": "#5a189a",
            "line_width": 2,
            "pickable": False,
            "reset_camera": False,
        }
        try:
            return self.plotter.add_mesh(mesh, use_bounds=False, **kwargs)
        except TypeError:
            actor = self.plotter.add_mesh(mesh, **kwargs)
            self._exclude_actor_from_bounds(actor)
            return actor

    def _finish_scene(
        self,
        committed_bounds: tuple[float, float, float, float, float, float] | None,
        camera_position: Any,
        reset_camera: bool,
    ) -> None:
        self.plotter.add_axes()
        self._remove_bounds_axes()
        show_bounds_kwargs = {
            "grid": "front",
            "location": "outer",
            "xtitle": "x [mm]",
            "ytitle": "y [mm]",
            "ztitle": "z [mm]",
            "color": self._label_color(),
        }
        if committed_bounds is not None:
            show_bounds_kwargs["bounds"] = committed_bounds
        try:
            self.plotter.show_bounds(**show_bounds_kwargs)
        except TypeError:
            show_bounds_kwargs.pop("color", None)
            try:
                self.plotter.show_bounds(**show_bounds_kwargs)
            except TypeError:
                legacy_kwargs = dict(show_bounds_kwargs)
                legacy_kwargs["xlabel"] = legacy_kwargs.pop("xtitle")
                legacy_kwargs["ylabel"] = legacy_kwargs.pop("ytitle")
                legacy_kwargs["zlabel"] = legacy_kwargs.pop("ztitle")
                self.plotter.show_bounds(**legacy_kwargs)
        if reset_camera:
            self.plotter.camera_position = "iso"
            self.plotter.reset_camera()
        elif camera_position is not None:
            self.plotter.camera_position = camera_position
        self._enable_mesh_picking_once()
        self._enable_interaction_observers_once()
        self.safe_render()

    def _draw_batched(
        self,
        model: ThermalGraphModel,
        visible: set[int],
        node_colors: dict[int, str] | None = None,
        node_scalar_values: dict[int, float] | None = None,
        scalar_cmap: str = "jet",
        scalar_clim: tuple[float, float] | None = None,
        scalar_bar_title: str = "Temperature [K]",
        committed_bounds: tuple[float, float, float, float, float, float] | None = None,
    ) -> None:
        log_event("pyvista draw_batched build mesh start", visible=len(visible))
        points: list[list[float]] = []
        faces: list[int] = []
        cell_node_ids: list[int] = []
        cell_colors: list[list[int]] = []
        cell_scalars: list[float] = []
        face_template = (
            (0, 1, 3, 2),
            (4, 6, 7, 5),
            (0, 4, 5, 1),
            (2, 3, 7, 6),
            (0, 2, 6, 4),
            (1, 5, 7, 3),
        )
        for node_id in model.ordered_node_ids():
            if node_id not in visible:
                continue
            node = model.nodes[node_id]
            geometry = _safe_node_cube_geometry(node)
            if geometry is None:
                continue
            center, lengths = geometry
            half = np.array(lengths, dtype=float) * 0.5
            base = len(points)
            for signs in (
                (-1, -1, -1),
                (1, -1, -1),
                (-1, 1, -1),
                (1, 1, -1),
                (-1, -1, 1),
                (1, -1, 1),
                (-1, 1, 1),
                (1, 1, 1),
            ):
                points.append((center + half * np.array(signs, dtype=float)).tolist())
            color = (node_colors or {}).get(node_id, self._color_for_material(node.material))
            depth_focused = self._node_in_depth_focus(center, committed_bounds)
            rgb = self._depth_adjust_rgb(self._hex_to_uint8_rgb(color), depth_focused)
            scalar_value = (
                float(node_scalar_values[node_id])
                if node_scalar_values is not None and node_id in node_scalar_values
                else float("nan")
            )
            for face in face_template:
                faces.extend([4, *(base + index for index in face)])
                cell_node_ids.append(node_id)
                if node_scalar_values is None:
                    cell_colors.append(rgb)
                else:
                    cell_scalars.append(scalar_value)
            self._batched_node_geometry[node_id] = (center, lengths)
        if not points:
            log_event("pyvista draw_batched no valid points")
            return
        log_event(
            "pyvista draw_batched create PolyData",
            points=len(points),
            faces=len(cell_node_ids),
        )
        mesh = self.pv.PolyData(
            np.asarray(points, dtype=float),
            np.asarray(faces, dtype=np.int64),
        )
        mesh.cell_data["node_id"] = np.asarray(cell_node_ids, dtype=int)
        if node_scalar_values is not None:
            mesh.cell_data["temperature_K"] = np.asarray(cell_scalars, dtype=float)
        else:
            mesh.cell_data["cell_rgb"] = np.asarray(cell_colors, dtype=np.uint8)
        self._batched_mesh = mesh
        mesh_kwargs = {
            "opacity": self._cell_opacity(node_scalar_values is not None, False, True),
            "show_edges": True,
            "edge_color": self._node_edge_color(),
            "line_width": 0.35,
            "pickable": True,
            **self._lit_mesh_kwargs(),
        }
        if node_scalar_values is not None:
            log_event("pyvista draw_batched add_mesh scalar")
            self._batched_actor = self.plotter.add_mesh(
                mesh,
                scalars="temperature_K",
                cmap=scalar_cmap,
                clim=scalar_clim,
                show_scalar_bar=True,
                scalar_bar_args=self._scalar_bar_args(scalar_bar_title),
                **mesh_kwargs,
            )
        else:
            log_event("pyvista draw_batched add_mesh rgb")
            self._batched_actor = self.plotter.add_mesh(
                mesh,
                scalars="cell_rgb",
                rgb=True,
                show_scalar_bar=False,
                **mesh_kwargs,
            )
        self._enable_actor_pick(self._batched_actor)
        self._show_batched_selection(self.selected_node_ids)
        if self.show_edges:
            log_event("pyvista draw_batched draw edges")
            self._draw_batched_edges(model, visible)
        log_event("pyvista draw_batched complete")

    def _draw_batched_edges(self, model: ThermalGraphModel, visible: set[int]) -> None:
        log_event("pyvista draw_batched_edges start", edges=len(model.edges), visible=len(visible))
        points: list[list[float]] = []
        lines: list[int] = []
        for edge in model.edges.values():
            if edge.source not in visible or edge.target not in visible:
                continue
            base = len(points)
            points.append(list(model.nodes[edge.source].center))
            points.append(list(model.nodes[edge.target].center))
            lines.extend([2, base, base + 1])
        if not points:
            log_event("pyvista draw_batched_edges no points")
            return
        log_event("pyvista draw_batched_edges create PolyData", line_points=len(points))
        mesh = self.pv.PolyData(
            np.asarray(points, dtype=float),
            lines=np.asarray(lines, dtype=np.int64),
        )
        self._edge_actors.append(
            self.plotter.add_mesh(mesh, color=self._edge_color(), line_width=1, opacity=0.55)
        )
        log_event("pyvista draw_batched_edges complete")

    def _show_batched_selection(self, node_ids: set[int] | int | None) -> None:
        if self._batched_selected_actor is not None:
            try:
                self.plotter.remove_actor(self._batched_selected_actor)
            except Exception:
                pass
            self._batched_selected_actor = None
        if node_ids is None:
            return
        selected_ids = {int(node_ids)} if isinstance(node_ids, int) else {int(node_id) for node_id in node_ids}
        meshes = []
        for node_id in selected_ids:
            if node_id not in self._batched_node_geometry:
                continue
            center, lengths = self._batched_node_geometry[node_id]
            meshes.append(
                self.pv.Cube(
                    center=center,
                    x_length=lengths[0],
                    y_length=lengths[1],
                    z_length=lengths[2],
                )
            )
        if not meshes:
            return
        mesh = meshes[0]
        for extra in meshes[1:]:
            mesh = mesh.merge(extra)
        self._batched_selected_actor = self.plotter.add_mesh(
            mesh,
            color="#ffd166",
            opacity=self._cell_opacity(False, True, True),
            show_edges=True,
            edge_color="#f87171",
            line_width=3,
            pickable=False,
            **self._lit_mesh_kwargs(),
        )

    def _cell_opacity(self, scalar_active: bool, selected: bool, depth_focused: bool = True) -> float:
        if self.shader_mode_enabled:
            return 1.0
        base = float(self.cell_opacity)
        if self.depth_focus_enabled and not selected and not depth_focused:
            base *= 0.22
        if scalar_active and not selected:
            return max(0.02, min(1.0, base))
        return max(base, min(1.0, base + 0.28)) if selected else max(0.02, min(1.0, base))

    def _node_in_depth_focus(
        self,
        center: np.ndarray,
        bounds: tuple[float, float, float, float, float, float] | None,
    ) -> bool:
        if not self.depth_focus_enabled or bounds is None:
            return True
        axis_index = {"x": 0, "y": 1, "z": 2}.get(str(self.depth_focus_axis).lower(), 2)
        bound_index = axis_index * 2
        axis_min = float(bounds[bound_index])
        axis_max = float(bounds[bound_index + 1])
        span = max(axis_max - axis_min, 1.0e-9)
        depth = (float(center[axis_index]) - axis_min) / span
        return abs(depth - float(self.depth_focus_fraction)) <= float(self.depth_focus_width) * 0.5

    def _depth_adjust_rgb(self, rgb: list[int], depth_focused: bool) -> list[int]:
        if not self.depth_focus_enabled or depth_focused:
            return rgb
        background = 24 if self.dark_mode else 238
        return [int(round(background + 0.28 * (int(value) - background))) for value in rgb]

    def _lit_mesh_kwargs(self) -> dict[str, Any]:
        if not self.shader_mode_enabled:
            return {"lighting": True}
        return {
            "lighting": True,
            "ambient": 0.22,
            "diffuse": 0.78,
            "specular": 0.25,
            "specular_power": 24.0,
            "smooth_shading": False,
        }

    def _apply_shader_mode_to_scene(self) -> None:
        for node_id, actor in self._node_actors.items():
            selected = node_id in self.selected_node_ids
            depth_focused = True
            geometry = self._batched_node_geometry.get(node_id) or _safe_node_cube_geometry(
                self._last_model_nodes.get(node_id)
            )
            if geometry is not None:
                depth_focused = self._node_in_depth_focus(geometry[0], self._last_committed_bounds)
            self._set_actor_opacity(
                actor,
                self._cell_opacity(self._actor_has_temperature_scalars(actor), selected, depth_focused),
            )
            self._apply_lighting_to_actor(actor)
        for actor in (self._batched_actor, self._batched_selected_actor):
            if actor is None:
                continue
            self._set_actor_opacity(actor, self._batched_actor_opacity(actor))
            self._apply_lighting_to_actor(actor)

    def _batched_actor_opacity(self, actor: Any) -> float:
        selected = (
            self._batched_selected_actor is not None
            and self._actor_key(actor) == self._actor_key(self._batched_selected_actor)
        )
        scalar_active = False
        if self._batched_mesh is not None and not selected:
            try:
                scalar_active = "temperature_K" in self._batched_mesh.cell_data
            except Exception:
                scalar_active = False
        return self._cell_opacity(scalar_active, selected, True)

    @staticmethod
    def _set_actor_opacity(actor: Any, opacity: float) -> None:
        try:
            prop = actor.GetProperty()
            prop.SetOpacity(float(opacity))
            prop.Modified()
        except Exception:
            pass

    def _apply_lighting_to_actor(self, actor: Any) -> None:
        try:
            prop = actor.GetProperty()
            prop.LightingOn()
            if self.shader_mode_enabled:
                prop.SetAmbient(0.22)
                prop.SetDiffuse(0.78)
                prop.SetSpecular(0.25)
                prop.SetSpecularPower(24.0)
                if hasattr(prop, "SetInterpolationToPhong"):
                    prop.SetInterpolationToPhong()
            else:
                prop.SetAmbient(0.0)
                prop.SetDiffuse(1.0)
                prop.SetSpecular(0.0)
                if hasattr(prop, "SetInterpolationToFlat"):
                    prop.SetInterpolationToFlat()
            prop.Modified()
        except Exception:
            pass

    def _actor_has_temperature_scalars(self, actor: Any) -> bool:
        node_id = self._actor_node_ids.get(self._actor_key(actor))
        if node_id is None:
            return False
        mesh = self._node_meshes.get(int(node_id))
        if mesh is None:
            return False
        try:
            return "temperature_K" in mesh.cell_data
        except Exception:
            return False

    @staticmethod
    def _actor_key(actor: Any) -> str:
        vtk_pointer = getattr(actor, "__this__", None)
        if vtk_pointer is not None:
            return str(vtk_pointer)
        try:
            return str(actor.GetAddressAsString(""))
        except Exception:
            return str(id(actor))

    def _color_for_material(self, material: str) -> str:
        palette = [
            "#4e79a7",
            "#f28e2b",
            "#59a14f",
            "#e15759",
            "#76b7b2",
            "#edc948",
            "#b07aa1",
            "#9c755f",
            "#bab0ab",
        ]
        key = material or "unknown"
        if key not in self._material_colors:
            self._material_colors[key] = palette[len(self._material_colors) % len(palette)]
        return self._material_colors[key]

    def _label_color(self) -> str:
        return "#e5e7eb" if self.dark_mode else "black"

    def _edge_color(self) -> str:
        return "#9ca3af" if self.dark_mode else "#555555"

    def _node_edge_color(self) -> str:
        return "#d1d5db" if self.dark_mode else "#1f2937"

    def _node_edge_rgb(self) -> tuple[float, float, float]:
        return (0.8196078431, 0.8352941176, 0.8588235294) if self.dark_mode else (
            0.1215686275,
            0.1607843137,
            0.2156862745,
        )

    def _scalar_bar_args(self, title: str) -> dict[str, Any]:
        return {
            "title": title,
            "vertical": True,
            "position_x": 0.86,
            "position_y": 0.08,
            "width": 0.08,
            "height": 0.32,
            "title_font_size": 12,
            "label_font_size": 10,
            "color": self._label_color(),
        }

    @staticmethod
    def _mark_dataset_modified(mesh: Any) -> None:
        for method_name in ("Modified", "modified"):
            method = getattr(mesh, method_name, None)
            if method is None:
                continue
            try:
                method()
                return
            except Exception:
                pass

    def _update_actor_cell_scalars(
        self,
        actor: Any,
        source_mesh: Any,
        scalars: np.ndarray,
        scalar_clim: tuple[float, float] | None,
    ) -> bool:
        mapper = self._actor_mapper(actor)
        datasets: list[Any] = [source_mesh]
        if mapper is not None:
            try:
                mapper_input = mapper.GetInput()
                if mapper_input is not None:
                    wrapped = self.pv.wrap(mapper_input)
                    if wrapped is not source_mesh:
                        datasets.append(wrapped)
            except Exception:
                pass

        updated = False
        mapper_dataset = None
        for dataset in datasets:
            try:
                values = np.asarray(scalars, dtype=float)
                if int(getattr(dataset, "n_cells", len(values))) != len(values):
                    continue
                dataset.cell_data["temperature_K"] = values
                try:
                    dataset.set_active_scalars("temperature_K", preference="cell")
                except Exception:
                    pass
                self._mark_cell_data_modified(dataset, "temperature_K")
                self._mark_dataset_modified(dataset)
                mapper_dataset = dataset
                updated = True
            except Exception:
                continue

        if mapper is not None:
            try:
                if mapper_dataset is not None:
                    mapper.SetInputData(mapper_dataset)
                mapper.SetScalarModeToUseCellData()
                mapper.SelectColorArray("temperature_K")
                mapper.SetColorModeToMapScalars()
                self._set_mapper_scalar_range(mapper, scalar_clim)
                mapper.Update()
                mapper.Modified()
            except Exception:
                pass
        try:
            actor.Modified()
        except Exception:
            pass
        return updated

    def _update_actor_direct_colors(
        self,
        node_scalar_values: dict[int, float],
        scalar_clim: tuple[float, float] | None,
    ) -> bool:
        if not self._node_actors:
            return False
        cmin, cmax = scalar_clim or self._range_from_values(node_scalar_values)
        span = max(cmax - cmin, 1.0e-12)
        cmap = colormaps["jet"]
        updated = False
        for node_id, actor in self._node_actors.items():
            if node_id not in node_scalar_values or node_id == self.selected_node_id:
                continue
            try:
                t = max(0.0, min(1.0, (float(node_scalar_values[node_id]) - cmin) / span))
                rgba = cmap(t)
                actor.GetProperty().SetColor(float(rgba[0]), float(rgba[1]), float(rgba[2]))
                actor.GetProperty().Modified()
                actor.Modified()
                updated = True
            except Exception:
                continue
        if updated:
            self.safe_render()
        return updated

    @staticmethod
    def _range_from_values(values: dict[int, float]) -> tuple[float, float]:
        array = np.array(list(values.values()), dtype=float)
        if array.size == 0:
            return (0.0, 1.0)
        cmin = float(np.nanmin(array))
        cmax = float(np.nanmax(array))
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= cmin:
            cmax = cmin + 1.0
        return (cmin, cmax)

    @staticmethod
    def _actor_mapper(actor: Any) -> Any | None:
        if actor is None:
            return None
        for accessor in ("GetMapper",):
            method = getattr(actor, accessor, None)
            if method is None:
                continue
            try:
                mapper = method()
                if mapper is not None:
                    return mapper
            except Exception:
                pass
        mapper = getattr(actor, "mapper", None)
        return mapper

    @staticmethod
    def _mark_cell_data_modified(dataset: Any, array_name: str) -> None:
        try:
            array = dataset.GetCellData().GetArray(array_name)
            if array is not None:
                array.Modified()
            dataset.GetCellData().Modified()
            return
        except Exception:
            pass
        try:
            array = dataset.cell_data[array_name]
            if hasattr(array, "Modified"):
                array.Modified()
        except Exception:
            pass

    @staticmethod
    def _set_actor_scalar_range(actor: Any, scalar_clim: tuple[float, float] | None) -> None:
        if actor is None or scalar_clim is None:
            return
        try:
            mapper = actor.GetMapper()
            GraphPyVistaWidget._set_mapper_scalar_range(mapper, scalar_clim)
        except Exception:
            pass

    @staticmethod
    def _set_mapper_scalar_range(mapper: Any, scalar_clim: tuple[float, float] | None) -> None:
        if mapper is None or scalar_clim is None:
            return
        try:
            mapper.SetScalarRange(float(scalar_clim[0]), float(scalar_clim[1]))
            lookup_table = mapper.GetLookupTable()
            if lookup_table is not None:
                lookup_table.SetRange(float(scalar_clim[0]), float(scalar_clim[1]))
                lookup_table.Modified()
            mapper.Modified()
        except Exception:
            pass

    @staticmethod
    def _hex_to_uint8_rgb(color: str) -> list[int]:
        cleaned = str(color).strip().lstrip("#")
        if len(cleaned) != 6:
            return [88, 166, 255]
        try:
            return [int(cleaned[index : index + 2], 16) for index in (0, 2, 4)]
        except ValueError:
            return [88, 166, 255]


def _safe_node_cube_geometry(node: Any) -> tuple[np.ndarray, tuple[float, float, float]] | None:
    try:
        center = np.asarray(node.center, dtype=float)
    except Exception:
        return None
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        return None
    try:
        if getattr(node, "size_mm", None) is not None:
            lengths = np.asarray(node.size_mm, dtype=float)
        else:
            side = float(getattr(node, "side_length_m", 0.0))
            lengths = np.array([side, side, side], dtype=float)
    except Exception:
        return None
    if lengths.shape != (3,) or not np.all(np.isfinite(lengths)):
        return None
    lengths = np.maximum(lengths, 1.0e-6)
    return center, tuple(float(value) for value in lengths)
