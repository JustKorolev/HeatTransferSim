"""PyVista/PyVistaQt widget for sparse thermal graph visualization."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .models import ThermalGraphModel


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
        self.selected_node_id: int | None = None
        self.draw_mode_enabled = False
        self.show_labels = True
        self.show_edges = True
        self.show_heaters = True
        self.show_sensors = True
        self._node_actors: dict[int, Any] = {}
        self._actor_node_ids: dict[str, int] = {}
        self._label_actors: list[Any] = []
        self._marker_actors: list[Any] = []
        self._edge_actors: list[Any] = []
        self._preview_actors: list[Any] = []
        self._last_preview_coords: list[tuple[int, int, int]] = []
        self._last_preview_side = 1.0
        self._picking_enabled = False
        self._observers_enabled = False
        self._ignore_next_mesh_pick = False
        self._hover_node_id: int | None = None

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

    def set_toggles(
        self,
        show_labels: bool,
        show_edges: bool,
        show_heaters: bool,
        show_sensors: bool,
    ) -> None:
        self.show_labels = show_labels
        self.show_edges = show_edges
        self.show_heaters = show_heaters
        self.show_sensors = show_sensors

    def draw(self, model: ThermalGraphModel, reset_camera: bool = False) -> None:
        camera_position = self.plotter.camera_position if not reset_camera else None
        preview_coords = list(getattr(self, "_last_preview_coords", []))
        preview_side = float(getattr(self, "_last_preview_side", 1.0))
        committed_bounds = self._committed_model_bounds(model)
        self.plotter.clear()
        self._node_actors = {}
        self._actor_node_ids = {}
        self._label_actors = []
        self._marker_actors = []
        self._edge_actors = []
        for node_id in model.ordered_node_ids():
            node = model.nodes[node_id]
            center = np.array(node.center, dtype=float)
            side = max(1.0e-6, float(node.side_length_m))
            mesh = self.pv.Cube(center=center, x_length=side, y_length=side, z_length=side)
            mesh.field_data["node_id"] = np.array([node_id], dtype=int)
            selected = node_id == self.selected_node_id
            actor = self.plotter.add_mesh(
                mesh,
                color="#ffd166" if selected else "#58a6ff",
                opacity=0.34 if not selected else 0.62,
                show_edges=True,
                edge_color="#d00000" if selected else "#1f2937",
                line_width=3 if selected else 1,
                pickable=True,
            )
            self._node_actors[node_id] = actor
            self._actor_node_ids[self._actor_key(actor)] = node_id
            self._enable_actor_pick(actor)
            if self.show_labels:
                label_actor = self.plotter.add_point_labels(
                    np.array([center], dtype=float),
                    [str(node_id)],
                    font_size=13,
                    text_color="black",
                    shape_opacity=0.0,
                    always_visible=True,
                )
                self._label_actors.append(label_actor)
            if self.show_heaters and node.has_heater:
                self._add_marker(center + np.array([0.0, 0.0, 0.36 * side]), "#ff6b35", "H")
            if self.show_sensors and node.has_sensor:
                self._add_marker(center + np.array([0.0, 0.0, -0.36 * side]), "#2a9d8f", "S")

        if self.show_edges:
            for edge in model.edges.values():
                if edge.source not in model.nodes or edge.target not in model.nodes:
                    continue
                p0 = np.array(model.nodes[edge.source].center, dtype=float)
                p1 = np.array(model.nodes[edge.target].center, dtype=float)
                line = self.pv.Line(p0, p1)
                actor = self.plotter.add_mesh(line, color="#555555", line_width=3)
                self._edge_actors.append(actor)

        self.plotter.add_axes()
        self._remove_bounds_axes()
        show_bounds_kwargs = {
            "grid": "front",
            "location": "outer",
            "xlabel": "i",
            "ylabel": "j",
            "zlabel": "k",
        }
        if committed_bounds is not None:
            show_bounds_kwargs["bounds"] = committed_bounds
        self.plotter.show_bounds(**show_bounds_kwargs)
        if reset_camera:
            self.plotter.camera_position = "iso"
            self.plotter.reset_camera()
        elif camera_position is not None:
            self.plotter.camera_position = camera_position
        self._enable_mesh_picking_once()
        self._enable_interaction_observers_once()
        if preview_coords:
            self.show_preview(preview_coords, preview_side)
        self.plotter.render()

    def select_node(self, node_id: int | None, model: ThermalGraphModel | None = None) -> None:
        previous_node_id = self.selected_node_id
        self.selected_node_id = node_id
        if previous_node_id == node_id:
            return
        if previous_node_id in self._node_actors:
            self._style_node_actor(self._node_actors[previous_node_id], selected=False)
        if node_id in self._node_actors:
            self._style_node_actor(self._node_actors[node_id], selected=True)
        self.plotter.render()

    def set_draw_mode(self, enabled: bool) -> None:
        self.draw_mode_enabled = bool(enabled)

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
        self.plotter.render()

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
                self.plotter.render()
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

    def _add_marker(self, center: np.ndarray, color: str, label: str) -> None:
        actor = self.plotter.add_points(
            np.array([center], dtype=float),
            color=color,
            point_size=15,
            render_points_as_spheres=True,
        )
        self._marker_actors.append(actor)
        label_actor = self.plotter.add_point_labels(
            np.array([center], dtype=float),
            [label],
            font_size=10,
            text_color=color,
            shape_opacity=0.0,
            always_visible=True,
        )
        self._marker_actors.append(label_actor)

    @staticmethod
    def _style_node_actor(actor: Any, selected: bool) -> None:
        prop = actor.GetProperty()
        if selected:
            prop.SetColor(1.0, 0.8196078431, 0.4)
            prop.SetOpacity(0.62)
            prop.SetEdgeColor(0.8156862745, 0.0, 0.0)
            prop.SetLineWidth(3)
        else:
            prop.SetColor(0.3450980392, 0.6509803922, 1.0)
            prop.SetOpacity(0.34)
            prop.SetEdgeColor(0.1215686275, 0.1607843137, 0.2156862745)
            prop.SetLineWidth(1)
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
        if not self.draw_mode_enabled:
            return
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
            self.on_pick_node(node_id, picked_point, mouse_position)

    def _handle_mouse_move(self, *_: Any) -> None:
        if self.draw_mode_enabled and self.on_drag_update is not None:
            self.on_drag_update(self._mouse_position())
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
    def _committed_model_bounds(model: ThermalGraphModel) -> tuple[float, float, float, float, float, float] | None:
        """Bounds for real cells only; draw-mode preview cells must not affect the grid."""
        if not model.nodes:
            return None
        mins = np.array([np.inf, np.inf, np.inf], dtype=float)
        maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        for node in model.nodes.values():
            center = np.array(node.center, dtype=float)
            half = max(1.0e-6, float(node.side_length_m)) * 0.5
            mins = np.minimum(mins, center - half)
            maxs = np.maximum(maxs, center + half)
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

    @staticmethod
    def _actor_key(actor: Any) -> str:
        vtk_pointer = getattr(actor, "__this__", None)
        if vtk_pointer is not None:
            return str(vtk_pointer)
        try:
            return str(actor.GetAddressAsString(""))
        except Exception:
            return str(id(actor))
