"""Heater/sensor connection and pairing helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import ThermalGraphModel


DEFAULT_MAX_HEATER_SENSOR_PAIR_DISTANCE_MM = 25.0


def refresh_sensor_connected_nodes(model: ThermalGraphModel) -> list[str]:
    """Refresh each sensor's external body-node readout set from graph edges."""
    warnings: list[str] = []
    for node in model.nodes.values():
        if not node.is_sensor:
            node.sensor_connected_node_ids = []
            node.sensor_valid = True
            node.sensor_monitor_only = False
            continue
        connected: set[int] = set()
        for edge in model.edges.values():
            other_id: int | None = None
            if int(edge.source) == int(node.node_id):
                other_id = int(edge.target)
            elif int(edge.target) == int(node.node_id):
                other_id = int(edge.source)
            if other_id is None:
                continue
            other = model.nodes.get(other_id)
            if other is None or other.is_heater or other.is_sensor:
                continue
            connected.add(other_id)
        node.sensor_connected_node_ids = sorted(connected)
        if not node.sensor_connected_node_ids and node.is_heater and str(getattr(getattr(node, "heater_control", None), "mode", "")) == "mimo":
            node.sensor_connected_node_ids = [int(node.node_id)]
            if node.assigned_heater_id is None:
                node.assigned_heater_id = int(node.node_id)
            if node.assigned_sensor_id is None:
                node.assigned_sensor_id = int(node.node_id)
        node.sensor_valid = bool(node.sensor_connected_node_ids)
        if not node.sensor_valid:
            node.sensor_monitor_only = True
            warnings.append(f"Sensor node {node.node_id} has no connected body nodes; marked monitor-only.")
    _refresh_sensor_assignment_summaries(model)
    return warnings


def recompute_heater_sensor_pairing(
    model: ThermalGraphModel,
    max_distance_mm: float = DEFAULT_MAX_HEATER_SENSOR_PAIR_DISTANCE_MM,
) -> list[str]:
    """Pair each heater to its closest valid sensor within an AABB gap."""
    warnings = refresh_sensor_connected_nodes(model)
    max_distance = max(0.0, float(max_distance_mm))
    heaters = [node for _, node in sorted(model.nodes.items()) if node.is_heater]
    sensors = [node for _, node in sorted(model.nodes.items()) if node.is_sensor]
    for heater in heaters:
        heater.assigned_sensor_id = None
        heater.sensor_pair_distance_mm = None
    for sensor in sensors:
        sensor.assigned_heater_id = None
        sensor.assigned_heater_ids = []
        sensor.sensor_pair_distance_mm = None
        sensor.sensor_monitor_only = not bool(sensor.sensor_valid)

    for heater in heaters:
        candidates: list[tuple[float, int, Any]] = []
        for sensor in sensors:
            if not bool(sensor.sensor_valid):
                continue
            distance = aabb_surface_gap_mm(heater, sensor)
            if distance <= max_distance:
                candidates.append((distance, int(sensor.node_id), sensor))
        if not candidates:
            warnings.append(
                f"Heater node {heater.node_id} has no available valid sensor within {max_distance:g} mm."
            )
            continue
        distance, _sensor_id, sensor = min(candidates, key=lambda item: (item[0], item[1]))
        _assign_pair(heater, sensor, distance)
        sensor.sensor_monitor_only = False

    _refresh_sensor_assignment_summaries(model)
    for sensor in sensors:
        if not sensor.assigned_heater_ids:
            sensor.sensor_monitor_only = True
            warnings.append(f"Sensor node {sensor.node_id} has no assigned heater; marked monitor-only.")
    return warnings


def assign_heater_to_sensor(
    model: ThermalGraphModel,
    heater_id: int,
    sensor_id: int | None,
) -> list[str]:
    """Manually assign one selected heater to one sensor; sensors may serve multiple heaters."""
    warnings = refresh_sensor_connected_nodes(model)
    heater = model.nodes.get(int(heater_id))
    if heater is None or not heater.is_heater:
        raise ValueError(f"Node {heater_id} is not a heater.")
    previous_sensor_id = heater.assigned_sensor_id
    if previous_sensor_id in model.nodes:
        previous = model.nodes[int(previous_sensor_id)]
        previous.assigned_heater_ids = [
            int(value)
            for value in getattr(previous, "assigned_heater_ids", []) or []
            if int(value) != int(heater.node_id)
        ]
    heater.assigned_sensor_id = None
    heater.sensor_pair_distance_mm = None
    if sensor_id is None:
        _refresh_sensor_assignment_summaries(model)
        return warnings
    sensor = model.nodes.get(int(sensor_id))
    if sensor is None or not sensor.is_sensor:
        raise ValueError(f"Node {sensor_id} is not a sensor.")
    distance = aabb_surface_gap_mm(heater, sensor)
    _assign_pair(heater, sensor, distance)
    _refresh_sensor_assignment_summaries(model)
    sensor.sensor_monitor_only = not bool(sensor.sensor_valid)
    if not sensor.sensor_valid:
        warnings.append(f"Sensor node {sensor.node_id} has no connected body nodes; manual pair will be monitor-only.")
    return warnings


def aabb_surface_gap_mm(left: Any, right: Any) -> float:
    left_min, left_max = node_bounds_mm(left)
    right_min, right_max = node_bounds_mm(right)
    gaps = np.maximum(np.maximum(left_min - right_max, right_min - left_max), 0.0)
    return float(np.linalg.norm(gaps))


def node_bounds_mm(node: Any) -> tuple[np.ndarray, np.ndarray]:
    if getattr(node, "source_bounds_mm", None):
        bounds = node.source_bounds_mm
        if isinstance(bounds, dict) and "min" in bounds and "max" in bounds:
            return np.asarray(bounds["min"], dtype=float), np.asarray(bounds["max"], dtype=float)
    center = np.asarray(getattr(node, "center_mm", None) or getattr(node, "center", (0.0, 0.0, 0.0)), dtype=float)
    size = np.asarray(getattr(node, "size_mm", None) or (float(getattr(node, "side_length_m", 0.0)) * 1000.0,) * 3, dtype=float)
    half = np.maximum(size, 0.0) * 0.5
    return center - half, center + half


def sensor_readout_temperature_K(
    model: ThermalGraphModel | None,
    node_index: dict[int, int],
    temperatures_K: np.ndarray,
    sensor_id: int,
) -> float:
    if model is None:
        return float("nan")
    sensor = model.nodes.get(int(sensor_id))
    if sensor is None or not sensor.is_sensor:
        return float("nan")
    ids = [int(node_id) for node_id in getattr(sensor, "sensor_connected_node_ids", []) or []]
    values = [
        float(temperatures_K[node_index[node_id]])
        for node_id in ids
        if node_id in node_index and np.isfinite(float(temperatures_K[node_index[node_id]]))
    ]
    return float(np.mean(values)) if values else float("nan")


def average_inverse_capacitance_for_sensor(
    model: ThermalGraphModel | None,
    node_index: dict[int, int],
    inv_C: np.ndarray,
    sensor_id: int,
) -> tuple[float, list[int]]:
    if model is None:
        return 0.0, []
    sensor = model.nodes.get(int(sensor_id))
    if sensor is None or not sensor.is_sensor:
        return 0.0, []
    valid_values: list[float] = []
    valid_node_ids: list[int] = []
    for node_id in getattr(sensor, "sensor_connected_node_ids", []) or []:
        node_id = int(node_id)
        row = node_index.get(node_id)
        if row is None:
            continue
        value = float(inv_C[int(row)])
        if np.isfinite(value) and value > 0.0:
            valid_values.append(value)
            valid_node_ids.append(node_id)
    return (float(np.mean(valid_values)), valid_node_ids) if valid_values else (0.0, valid_node_ids)


def _assign_pair(heater: Any, sensor: Any, distance_mm: float) -> None:
    heater.assigned_sensor_id = int(sensor.node_id)
    heater.sensor_pair_distance_mm = float(distance_mm)
    assigned = [int(value) for value in getattr(sensor, "assigned_heater_ids", []) or []]
    assigned.append(int(heater.node_id))
    sensor.assigned_heater_ids = sorted(set(assigned))
    sensor.assigned_heater_id = int(sensor.assigned_heater_ids[0])
    previous_distance = getattr(sensor, "sensor_pair_distance_mm", None)
    sensor.sensor_pair_distance_mm = (
        float(distance_mm)
        if previous_distance is None
        else min(float(previous_distance), float(distance_mm))
    )


def _refresh_sensor_assignment_summaries(model: ThermalGraphModel) -> None:
    assigned_by_sensor: dict[int, list[tuple[int, float | None]]] = {
        int(node.node_id): [] for node in model.nodes.values() if node.is_sensor
    }
    for heater in model.nodes.values():
        if not heater.is_heater or heater.assigned_sensor_id is None:
            continue
        sensor_id = int(heater.assigned_sensor_id)
        if sensor_id not in assigned_by_sensor:
            continue
        distance = getattr(heater, "sensor_pair_distance_mm", None)
        assigned_by_sensor[sensor_id].append(
            (int(heater.node_id), float(distance) if distance is not None else None)
        )
    for sensor_id, assignments in assigned_by_sensor.items():
        sensor = model.nodes[int(sensor_id)]
        heater_ids = sorted({int(heater_id) for heater_id, _distance in assignments})
        sensor.assigned_heater_ids = heater_ids
        sensor.assigned_heater_id = heater_ids[0] if heater_ids else None
        distances = [float(distance) for _heater_id, distance in assignments if distance is not None]
        sensor.sensor_pair_distance_mm = min(distances) if distances else None
        if sensor.sensor_valid:
            sensor.sensor_monitor_only = not bool(heater_ids)
