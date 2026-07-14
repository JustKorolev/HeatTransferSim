"""Read-only Qt/matplotlib 2D adjacency graph view."""

from __future__ import annotations

from math import hypot
from typing import Any, Callable

import numpy as np

from .models import ThermalGraphModel
from .role_warnings import has_role_warning
from .tooltip_formatters import format_edge_tooltip, format_node_tooltip


class TwoDGraphWidget:
    """Read-only 2D graph widget backed by matplotlib and NetworkX."""

    def __init__(
        self,
        parent: Any,
        on_select_node: Callable[..., None] | None = None,
    ) -> None:
        self._load_dependencies()
        self.parent = parent
        self.on_select_node = on_select_node
        self.model: ThermalGraphModel | None = None
        self.visible_node_ids: set[int] | None = None
        self.selected_node_ids: set[int] = set()
        self.positions: dict[int, tuple[float, float]] = {}
        self.node_points: dict[int, tuple[float, float]] = {}
        self.edge_lines: list[tuple[int, int, Any]] = []
        self.dark_mode = False

        self.widget = self.QtWidgets.QWidget(parent)
        layout = self.QtWidgets.QVBoxLayout(self.widget)
        controls = self.QtWidgets.QHBoxLayout()
        controls.addWidget(self.QtWidgets.QLabel("Layout:"))
        self.layout_combo = self.QtWidgets.QComboBox()
        self.layout_combo.addItems(["XY projection", "XZ projection", "YZ projection", "Spring"])
        self.layout_combo.currentTextChanged.connect(self.refresh)
        controls.addWidget(self.layout_combo)
        refresh_button = self.QtWidgets.QPushButton("Refresh 2D Layout")
        refresh_button.clicked.connect(lambda: self.refresh(force_layout=True))
        controls.addWidget(refresh_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.selection_label = self.QtWidgets.QLabel("Selected node: none")
        self.selection_label.setWordWrap(True)
        layout.addWidget(self.selection_label)

        self.figure = self.Figure(figsize=(5, 4), tight_layout=True)
        self.canvas = self.FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)
        self.ax = self.figure.add_subplot(111)
        self.canvas.mpl_connect("motion_notify_event", self._handle_motion)
        self.canvas.mpl_connect("button_press_event", self._handle_click)

    def set_dark_mode(self, enabled: bool) -> None:
        self.dark_mode = bool(enabled)
        self._apply_theme()
        self.canvas.draw_idle()

    def _load_dependencies(self) -> None:
        try:
            from PySide6 import QtCore, QtGui, QtWidgets
        except ImportError:
            from qtpy import QtCore, QtGui, QtWidgets
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        from matplotlib.patches import FancyArrowPatch
        import networkx as nx

        self.QtCore = QtCore
        self.QtGui = QtGui
        self.QtWidgets = QtWidgets
        self.FigureCanvas = FigureCanvas
        self.Figure = Figure
        self.FancyArrowPatch = FancyArrowPatch
        self.nx = nx

    def set_model(
        self,
        model: ThermalGraphModel,
        force_layout: bool = False,
        visible_node_ids: set[int] | None = None,
        auto_refresh: bool = True,
    ) -> None:
        self.model = model
        self.visible_node_ids = visible_node_ids
        if auto_refresh:
            self.refresh(force_layout=force_layout)

    def refresh(self, *_: Any, force_layout: bool = False) -> None:
        self.ax.clear()
        self._apply_theme()
        self.node_points = {}
        self.edge_lines = []
        model = self.model
        visible = self.visible_node_ids if self.visible_node_ids is not None else set(model.nodes) if model is not None else set()
        self._refresh_selection_label(model, visible)
        if model is None or not visible:
            self.ax.set_title("No graph nodes", color=self._text_color())
            self.ax.axis("off")
            self.canvas.draw_idle()
            return
        if self.layout_combo.currentText() == "Spring" and len(visible) > 1000:
            self.ax.set_title(
                f"Spring layout skipped for {len(visible)} nodes; use projection or filters",
                color=self._text_color(),
            )
            self.ax.axis("off")
            self.canvas.draw_idle()
            return

        self.positions = self._compute_positions(model, force_layout=force_layout)
        if len(visible) > 1000:
            self._draw_large_projection(model, visible)
            return
        conductances = [max(0.0, float(edge.Gij_W_K)) for edge in model.edges.values()]
        max_g = max(conductances) if conductances else 1.0

        for edge in model.edges.values():
            if edge.source not in visible or edge.target not in visible:
                continue
            if edge.source not in self.positions or edge.target not in self.positions:
                continue
            x0, y0 = self.positions[edge.source]
            x1, y1 = self.positions[edge.target]
            width = 0.8 + 3.2 * safe_ratio(edge.Gij_W_K, max_g)
            curve = self._edge_curve(edge.source, edge.target)
            if abs(curve) > 1.0e-9:
                patch = self.FancyArrowPatch(
                    (x0, y0),
                    (x1, y1),
                    arrowstyle="-",
                    connectionstyle=f"arc3,rad={curve:.3f}",
                    color=self._edge_color(),
                    linewidth=width,
                    alpha=0.72,
                    zorder=1,
                )
                self.ax.add_patch(patch)
                self.edge_lines.append((edge.source, edge.target, patch))
            else:
                (line,) = self.ax.plot(
                    [x0, x1], [y0, y1], color=self._edge_color(), linewidth=width, alpha=0.72, zorder=1
                )
                self.edge_lines.append((edge.source, edge.target, line))

        for node_id in model.ordered_node_ids():
            if node_id not in visible:
                continue
            x, y = self.positions[node_id]
            node = model.nodes[node_id]
            warning = has_role_warning(node)
            color = _node_role_color(node)
            selected = node_id in self.selected_node_ids
            self.ax.scatter(
                [x],
                [y],
                s=360 if selected else 320,
                c=["#ffd166" if selected else "#ef4444" if warning else color],
                edgecolors="#ef4444" if warning else "#f87171" if selected else self._node_edge_color(),
                linewidths=2.8 if warning else 2.4 if selected else 1.0,
                zorder=3,
            )
            self.ax.text(
                x,
                y,
                str(node_id),
                ha="center",
                va="center",
                fontsize=9,
                color=self._node_text_color(),
                zorder=4,
            )
            self.node_points[node_id] = (x, y)

        self.ax.set_title("Read-only adjacency graph", color=self._text_color())
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.margins(0.18)
        self.ax.axis("off")
        self.canvas.draw_idle()

    def _draw_large_projection(self, model: ThermalGraphModel, visible: set[int]) -> None:
        xs: list[float] = []
        ys: list[float] = []
        colors: list[str] = []
        self.node_points = {}
        for node_id in model.ordered_node_ids():
            if node_id not in visible or node_id not in self.positions:
                continue
            x, y = self.positions[node_id]
            node = model.nodes[node_id]
            xs.append(x)
            ys.append(y)
            colors.append("#ef4444" if has_role_warning(node) else _node_role_color(node))
            self.node_points[node_id] = (x, y)
        self.ax.scatter(xs, ys, s=12, c=colors, alpha=0.72, linewidths=0, zorder=3)
        selected_xs: list[float] = []
        selected_ys: list[float] = []
        for node_id in self.selected_node_ids:
            if node_id in self.node_points:
                x, y = self.node_points[node_id]
                selected_xs.append(x)
                selected_ys.append(y)
        if selected_xs:
            self.ax.scatter(
                selected_xs,
                selected_ys,
                s=34,
                c=["#ffd166"],
                edgecolors="#f87171",
                linewidths=1.4,
                zorder=4,
            )
        edge_count = sum(
            1
            for edge in model.edges.values()
            if edge.source in visible and edge.target in visible
        )
        self.edge_lines = []
        self.ax.set_title(
            f"Projection view: {len(visible)} nodes, {edge_count} edges hidden for speed",
            color=self._text_color(),
        )
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.margins(0.08)
        self.ax.axis("off")
        self.canvas.draw_idle()

    def _apply_theme(self) -> None:
        background = "#111827" if self.dark_mode else "white"
        text = self._text_color()
        self.figure.patch.set_facecolor(background)
        self.ax.set_facecolor(background)
        self.ax.tick_params(colors=text)
        for spine in self.ax.spines.values():
            spine.set_color(text)

    def _text_color(self) -> str:
        return "#e5e7eb" if self.dark_mode else "#202124"

    def _node_text_color(self) -> str:
        return "#f9fafb" if self.dark_mode else "#202124"

    def _node_edge_color(self) -> str:
        return "#d1d5db" if self.dark_mode else "#202124"

    def _edge_color(self) -> str:
        return "#9ca3af" if self.dark_mode else "#5f6368"

    def _compute_positions(
        self, model: ThermalGraphModel, force_layout: bool = False
    ) -> dict[int, tuple[float, float]]:
        layout_name = self.layout_combo.currentText()
        visible = self.visible_node_ids if self.visible_node_ids is not None else set(model.nodes)
        if layout_name == "XY projection":
            return {node_id: (node.center[0], node.center[1]) for node_id, node in model.nodes.items() if node_id in visible}
        if layout_name == "XZ projection":
            return {node_id: (node.center[0], node.center[2]) for node_id, node in model.nodes.items() if node_id in visible}
        if layout_name == "YZ projection":
            return {node_id: (node.center[1], node.center[2]) for node_id, node in model.nodes.items() if node_id in visible}

        graph = self.nx.Graph()
        graph.add_nodes_from(node_id for node_id in model.ordered_node_ids() if node_id in visible)
        graph.add_edges_from(
            (edge.source, edge.target)
            for edge in model.edges.values()
            if edge.source in visible and edge.target in visible
        )
        if graph.number_of_edges() == 0:
            return {
                node_id: (float(index), 0.0)
                for index, node_id in enumerate(model.ordered_node_ids())
            }
        spacing = 8.0 / max(1.0, graph.number_of_nodes() ** 0.5)
        initial_positions = self._spring_initial_positions(model, graph, force_layout)
        raw = self.nx.spring_layout(
            graph,
            pos=initial_positions,
            seed=3,
            k=spacing,
            iterations=340,
            scale=8.0,
        )
        return expand_positions(
            {int(node_id): (float(pos[0]), float(pos[1])) for node_id, pos in raw.items()},
            minimum_distance=1.45,
            iterations=140,
        )

    def _spring_initial_positions(
        self, model: ThermalGraphModel, graph: Any, force_layout: bool
    ) -> dict[int, np.ndarray] | None:
        if force_layout and set(self.positions) == set(graph.nodes):
            return {
                node_id: np.array(self.positions[node_id], dtype=float)
                for node_id in graph.nodes
            }
        return None

    def _handle_motion(self, event: Any) -> None:
        if event.x is None or event.y is None or self.model is None:
            self.canvas.setToolTip("")
            return
        node_id = self._nearest_node(event)
        if node_id is not None:
            self._show_tooltip(event, format_node_tooltip(node_id, self.model.nodes[node_id]))
            return
        edge = self._nearest_edge(event)
        if edge is not None:
            source, target = edge
            attrs = self.model.edges.get((min(source, target), max(source, target)), {})
            self._show_tooltip(event, format_edge_tooltip(source, target, attrs))
            return
        self._hide_tooltip()

    def _handle_click(self, event: Any) -> None:
        if self.on_select_node is None:
            return
        node_id = self._nearest_node(event)
        if node_id is not None:
            modifiers = self.QtWidgets.QApplication.keyboardModifiers()
            additive = bool(modifiers & self.QtCore.Qt.ControlModifier)
            self.on_select_node(node_id, additive=additive)

    def _nearest_node(self, event: Any) -> int | None:
        if event.x is None or event.y is None:
            return None
        best: tuple[float, int] | None = None
        for node_id, point in self.node_points.items():
            px, py = self.ax.transData.transform(point)
            distance = hypot(float(event.x) - px, float(event.y) - py)
            if distance <= 24.0 and (best is None or distance < best[0]):
                best = (distance, node_id)
        return best[1] if best is not None else None

    def _nearest_edge(self, event: Any) -> tuple[int, int] | None:
        if event.x is None or event.y is None:
            return None
        best: tuple[float, tuple[int, int]] | None = None
        point = np.array([float(event.x), float(event.y)])
        for source, target, _line in self.edge_lines:
            distance = self._display_distance_to_edge(point, source, target)
            if distance <= 12.0 and (best is None or distance < best[0]):
                best = (distance, (source, target))
        return best[1] if best is not None else None

    def _edge_curve(self, source: int, target: int) -> float:
        return edge_curve_for_positions(source, target, self.positions, node_clearance=0.9)

    def _display_distance_to_edge(self, point: np.ndarray, source: int, target: int) -> float:
        p0 = np.array(self.ax.transData.transform(self.positions[source]), dtype=float)
        p1 = np.array(self.ax.transData.transform(self.positions[target]), dtype=float)
        return point_to_segment_distance(point, p0, p1)

    def _show_tooltip(self, event: Any, text: str) -> None:
        if not text:
            self._hide_tooltip()
            return
        self.canvas.setToolTip(text)
        try:
            global_pos = self.QtGui.QCursor.pos() + self.QtCore.QPoint(14, 18)
            self.QtWidgets.QToolTip.showText(global_pos, text, self.canvas)
        except Exception:
            pass

    def _hide_tooltip(self) -> None:
        self.canvas.setToolTip("")
        try:
            self.QtWidgets.QToolTip.hideText()
        except Exception:
            pass

    def _refresh_selection_label(self, model: ThermalGraphModel | None, visible: set[int]) -> None:
        if model is None or not self.selected_node_ids:
            self.selection_label.setText("Selected node: none")
            return
        active_node_id = next((node_id for node_id in sorted(self.selected_node_ids) if node_id in model.nodes), None)
        if active_node_id is None:
            self.selection_label.setText("Selected node: none")
            return
        total, visible_count = node_connection_counts(model, active_node_id, visible)
        visible_note = "" if visible_count == total else f", {visible_count} visible"
        self.selection_label.setText(
            f"Selected node {active_node_id}: {total} connection{'s' if total != 1 else ''}{visible_note}"
        )


def safe_ratio(value: Any, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if maximum <= 0.0:
        return 0.0
    return max(0.0, min(1.0, number / maximum))


def node_connection_counts(
    model: ThermalGraphModel,
    node_id: int,
    visible_node_ids: set[int] | None = None,
) -> tuple[int, int]:
    visible = visible_node_ids if visible_node_ids is not None else set(model.nodes)
    neighbors: set[int] = set()
    visible_neighbors: set[int] = set()
    for edge in model.edges.values():
        neighbor: int | None = None
        if edge.source == node_id:
            neighbor = edge.target
        elif edge.target == node_id:
            neighbor = edge.source
        if neighbor is None:
            continue
        neighbors.add(int(neighbor))
        if node_id in visible and neighbor in visible:
            visible_neighbors.add(int(neighbor))
    return len(neighbors), len(visible_neighbors)


def _node_role_color(node: Any) -> str:
    if getattr(node, "is_heater", False):
        return "#ffb703"
    if getattr(node, "has_cryocooler", False):
        return "#06b6d4"
    if getattr(node, "is_sensor", False):
        return "#2a9d8f"
    return "#58a6ff"


def point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1.0e-12:
        return float(np.linalg.norm(point - start))
    t = max(0.0, min(1.0, float(np.dot(point - start, segment) / length_sq)))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def expand_positions(
    positions: dict[int, tuple[float, float]],
    minimum_distance: float,
    iterations: int,
) -> dict[int, tuple[float, float]]:
    """Gently separate very close 2D nodes without changing graph topology."""
    if len(positions) < 2:
        return positions
    ids = list(positions)
    coords = {node_id: np.array(positions[node_id], dtype=float) for node_id in ids}
    for _ in range(iterations):
        moved = False
        for index, source in enumerate(ids):
            for target in ids[index + 1 :]:
                delta = coords[target] - coords[source]
                distance = float(np.linalg.norm(delta))
                if distance >= minimum_distance:
                    continue
                if distance <= 1.0e-9:
                    angle = (source * 37 + target * 17) % 360
                    radians = np.deg2rad(angle)
                    direction = np.array([np.cos(radians), np.sin(radians)])
                else:
                    direction = delta / distance
                push = 0.5 * (minimum_distance - max(distance, 0.0)) * direction
                coords[source] -= push
                coords[target] += push
                moved = True
        if not moved:
            break
    return {node_id: (float(coords[node_id][0]), float(coords[node_id][1])) for node_id in ids}


def edge_curve_for_positions(
    source: int,
    target: int,
    positions: dict[int, tuple[float, float]],
    node_clearance: float,
) -> float:
    """Return a curvature radius when an unrelated node lies under an edge."""
    if source not in positions or target not in positions:
        return 0.0
    blockers = 0
    start = np.array(positions[source], dtype=float)
    end = np.array(positions[target], dtype=float)
    for node_id, point in positions.items():
        if node_id in {source, target}:
            continue
        distance = point_to_segment_distance(np.array(point, dtype=float), start, end)
        if distance < node_clearance:
            blockers += 1
    if blockers == 0:
        return 0.0
    sign = 1 if (source + target) % 2 == 0 else -1
    return sign * min(0.55, 0.18 + 0.08 * blockers)
