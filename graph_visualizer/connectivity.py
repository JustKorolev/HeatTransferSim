"""Connected-component analysis helpers for loaded thermal graph models."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .models import ThermalGraphModel


CONNECTIVITY_COLORS = (
    "#3b82f6",
    "#ef4444",
    "#22c55e",
    "#f97316",
    "#a855f7",
    "#06b6d4",
    "#eab308",
    "#ec4899",
    "#14b8a6",
    "#8b5cf6",
    "#84cc16",
    "#f43f5e",
)


def analyze_model_connectivity(model: ThermalGraphModel) -> dict[str, Any]:
    nodes = [
        {
            "node_id": int(node.node_id),
            "center_mm": list(node.center_mm or node.center),
            "component_name": node.component_name,
            "material_name": node.material,
            "is_heater": bool(node.is_heater),
            "is_sensor": bool(node.is_sensor),
        }
        for node in model.nodes.values()
    ]
    edges = [
        {"node_i": int(edge.source), "node_j": int(edge.target)}
        for edge in model.edges.values()
    ]
    return graph_connectivity_analysis(nodes, edges)


def graph_connectivity_analysis(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    node_ids = sorted(_safe_node_id(node) for node in nodes if _safe_node_id(node) is not None)
    if not node_ids:
        return {
            "connected": True,
            "component_count": 0,
            "largest_component_id": None,
            "largest_component_size": 0,
            "disconnected_node_ids": [],
            "node_component_ids": {},
            "node_count": 0,
            "edge_count": 0,
            "components": [],
        }
    adjacency: dict[int, set[int]] = {node_id: set() for node_id in node_ids}
    valid_edge_count = 0
    for edge in edges:
        try:
            a = int(edge["node_i"])
            b = int(edge["node_j"])
        except (KeyError, TypeError, ValueError):
            continue
        if a not in adjacency or b not in adjacency:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
        valid_edge_count += 1
    nodes_by_id = {int(node["node_id"]): node for node in nodes if _safe_node_id(node) is not None}
    seen: set[int] = set()
    raw_components: list[list[int]] = []
    for node_id in node_ids:
        if node_id in seen:
            continue
        stack = [node_id]
        seen.add(node_id)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        raw_components.append(sorted(component))
    raw_components.sort(key=lambda ids: (-len(ids), ids[0] if ids else -1))
    largest = raw_components[0] if raw_components else []
    largest_id = 0 if raw_components else None
    disconnected_ids = [node_id for component in raw_components[1:] for node_id in component]
    node_component_ids = {
        str(node_id): component_id
        for component_id, component in enumerate(raw_components)
        for node_id in component
    }
    components = [
        _connectivity_component_summary(index, component, nodes_by_id)
        for index, component in enumerate(raw_components)
    ]
    return {
        "connected": len(raw_components) <= 1,
        "component_count": len(raw_components),
        "largest_component_id": largest_id,
        "largest_component_size": len(largest),
        "disconnected_node_ids": disconnected_ids,
        "node_component_ids": node_component_ids,
        "node_count": len(node_ids),
        "edge_count": valid_edge_count,
        "components": components,
    }


def connectivity_component_for_node(analysis: dict[str, Any] | None, node_id: int) -> int | None:
    if not isinstance(analysis, dict):
        return None
    mapping = analysis.get("node_component_ids")
    if isinstance(mapping, dict):
        value = mapping.get(str(int(node_id)), mapping.get(int(node_id)))
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None
    for component in analysis.get("components", []) or []:
        if not isinstance(component, dict):
            continue
        samples = component.get("node_ids_sample", []) or []
        if int(node_id) in {int(value) for value in samples if _can_int(value)} and not component.get("node_ids_truncated"):
            return int(component.get("component_id", -1))
    return None


def connectivity_component_color(component_id: int | None) -> str:
    if component_id is None:
        return "#94a3b8"
    if int(component_id) == 0:
        return "#64748b"
    return CONNECTIVITY_COLORS[(int(component_id) - 1) % len(CONNECTIVITY_COLORS)]


def _connectivity_component_summary(
    component_id: int,
    node_ids: list[int],
    nodes_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    component_nodes = [nodes_by_id[node_id] for node_id in node_ids if node_id in nodes_by_id]
    centers = [
        np.asarray(node.get("center_mm"), dtype=float)
        for node in component_nodes
        if isinstance(node.get("center_mm"), (list, tuple)) and len(node.get("center_mm")) == 3
    ]
    bounds: dict[str, Any] | None = None
    if centers:
        stacked = np.vstack(centers)
        bounds = {"min": np.min(stacked, axis=0).tolist(), "max": np.max(stacked, axis=0).tolist()}
    components = Counter(str(node.get("component_name", "") or "?") for node in component_nodes)
    materials = Counter(str(node.get("material_name", "") or "?") for node in component_nodes)
    return {
        "component_id": int(component_id),
        "node_count": len(node_ids),
        "node_ids_sample": node_ids[:50],
        "node_ids_truncated": len(node_ids) > 50,
        "heater_count": sum(1 for node in component_nodes if bool(node.get("is_heater"))),
        "sensor_count": sum(1 for node in component_nodes if bool(node.get("is_sensor"))),
        "bounds_center_mm": bounds,
        "top_components": components.most_common(10),
        "top_materials": materials.most_common(10),
    }


def _safe_node_id(node: dict[str, Any]) -> int | None:
    try:
        return int(node["node_id"])
    except (KeyError, TypeError, ValueError):
        return None


def _can_int(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False
