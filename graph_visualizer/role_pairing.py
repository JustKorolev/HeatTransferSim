"""Heater/sensor connection and pairing helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import ThermalGraphModel


DEFAULT_MAX_HEATER_SENSOR_PAIR_DISTANCE_MM = 25.0


def refresh_heater_power_deposition_nodes(model: ThermalGraphModel) -> list[str]:
    """Refresh heater body-node deposition sets, preserving explicit CAD contact metadata."""
    warnings: list[str] = []
    for node in model.nodes.values():
        if not node.is_heater:
            node.power_deposition_node_ids = []
            node.power_deposition_weights = []
            node.heater_attached = True
            node.heater_valid = True
            node.heater_warning = ""
            continue
        explicit = [
            int(node_id)
            for node_id in getattr(node, "power_deposition_node_ids", []) or []
            if _is_body_node(model, int(node_id))
        ]
        ids = explicit or sorted(_external_body_neighbors(model, int(node.node_id)))
        if not ids and not bool(getattr(node, "is_cad_role_node", False)):
            ids = [int(node.node_id)]
        if (
            not ids
            and node.is_sensor
            and str(getattr(getattr(node, "heater_control", None), "mode", "")) == "mimo"
        ):
            ids = [int(node.node_id)]
        weights = _normalized_weights(getattr(node, "power_deposition_weights", []) or [], len(ids))
        node.power_deposition_node_ids = ids
        node.power_deposition_weights = weights
        node.heater_attached = bool(ids)
        node.heater_valid = bool(ids)
        node.heater_warning = "" if ids else f"Heater node {node.node_id} has no body power deposition nodes."
        if not ids:
            warnings.append(f"Heater node {node.node_id} has no body power deposition nodes; excluded from MIMO.")
    return warnings


def refresh_sensor_connected_nodes(model: ThermalGraphModel) -> list[str]:
    """Refresh each sensor's external body-node readout set from graph edges."""
    warnings: list[str] = []
    for node in model.nodes.values():
        if not node.is_sensor:
            node.sensor_connected_node_ids = []
            node.readout_node_ids = []
            node.readout_weights = []
            node.sensor_valid = True
            node.sensor_monitor_only = False
            continue
        explicit = [
            int(node_id)
            for node_id in getattr(node, "readout_node_ids", []) or []
            if _is_body_node(model, int(node_id))
        ]
        connected: set[int] = set(explicit)
        connected_role_ids: set[int] = set()
        if not connected:
            for edge in model.edges.values():
                other_id: int | None = None
                if int(edge.source) == int(node.node_id):
                    other_id = int(edge.target)
                elif int(edge.target) == int(node.node_id):
                    other_id = int(edge.source)
                if other_id is None:
                    continue
                other = model.nodes.get(other_id)
                if other is None:
                    continue
                if other.is_heater or other.is_sensor:
                    connected_role_ids.add(other_id)
                    continue
                connected.add(other_id)
        inherited_from_heaters = False
        if not connected and connected_role_ids:
            for role_id in sorted(connected_role_ids):
                role_node = model.nodes.get(int(role_id))
                if role_node is None or not role_node.is_heater:
                    continue
                connected.update(
                    int(node_id)
                    for node_id in getattr(role_node, "power_deposition_node_ids", []) or []
                    if _is_body_node(model, int(node_id))
                )
                if not connected:
                    connected.update(_external_body_neighbors(model, int(role_id)))
            inherited_from_heaters = bool(connected)
        node.sensor_connected_node_ids = sorted(connected)
        node.readout_node_ids = sorted(connected)
        node.readout_weights = _normalized_weights(getattr(node, "readout_weights", []) or [], len(node.readout_node_ids))
        if not node.sensor_connected_node_ids and node.is_heater and str(getattr(getattr(node, "heater_control", None), "mode", "")) == "mimo":
            node.sensor_connected_node_ids = [int(node.node_id)]
            node.readout_node_ids = [int(node.node_id)]
            node.readout_weights = [1.0]
            if node.assigned_heater_id is None:
                node.assigned_heater_id = int(node.node_id)
            if node.assigned_sensor_id is None:
                node.assigned_sensor_id = int(node.node_id)
        node.sensor_valid = bool(node.sensor_connected_node_ids)
        if not node.sensor_valid:
            node.sensor_monitor_only = True
            warnings.append(f"Sensor node {node.node_id} has no connected body nodes; marked monitor-only.")
        elif inherited_from_heaters:
            warnings.append(
                f"Sensor node {node.node_id} has no direct body-node contacts but contacts heater node(s) "
                f"{sorted(connected_role_ids)}; using heater-adjacent body node(s) {sorted(connected)} for readout."
            )
    _refresh_sensor_assignment_summaries(model)
    return warnings


def recompute_heater_sensor_pairing(
    model: ThermalGraphModel,
    max_distance_mm: float = DEFAULT_MAX_HEATER_SENSOR_PAIR_DISTANCE_MM,
) -> list[str]:
    """Pair valid heaters to valid sensors using global one-to-one nearest AABB gaps."""
    warnings = refresh_heater_power_deposition_nodes(model)
    warnings.extend(refresh_sensor_connected_nodes(model))
    max_distance = max(0.0, float(max_distance_mm))
    heaters = [
        node
        for _, node in sorted(model.nodes.items())
        if node.is_heater and bool(getattr(node, "heater_valid", True))
    ]
    sensors = [node for _, node in sorted(model.nodes.items()) if node.is_sensor]
    for heater in heaters:
        heater.assigned_sensor_id = None
        heater.sensor_pair_distance_mm = None
    for sensor in sensors:
        sensor.assigned_heater_id = None
        sensor.assigned_heater_ids = []
        sensor.sensor_pair_distance_mm = None
        sensor.sensor_monitor_only = not bool(sensor.sensor_valid)

    candidate_pairs: list[tuple[float, int, int, Any, Any]] = []
    for heater in heaters:
        for sensor in sensors:
            if not bool(sensor.sensor_valid):
                continue
            distance = aabb_surface_gap_mm(heater, sensor)
            if distance <= max_distance:
                candidate_pairs.append((distance, int(heater.node_id), int(sensor.node_id), heater, sensor))
    assigned_heaters: set[int] = set()
    assigned_sensors: set[int] = set()
    for distance, heater_id, sensor_id, heater, sensor in sorted(candidate_pairs, key=lambda item: (item[0], item[1], item[2])):
        if heater_id in assigned_heaters or sensor_id in assigned_sensors:
            continue
        _assign_pair(heater, sensor, distance)
        assigned_heaters.add(heater_id)
        assigned_sensors.add(sensor_id)
        sensor.sensor_monitor_only = False

    _refresh_sensor_assignment_summaries(model)
    for heater in heaters:
        if heater.assigned_sensor_id is None:
            warnings.append(
                f"Heater node {heater.node_id} has no available valid unpaired sensor within {max_distance:g} mm."
            )
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
    """Manually assign one selected heater to one sensor, preserving one-to-one ownership."""
    warnings = refresh_heater_power_deposition_nodes(model)
    warnings.extend(refresh_sensor_connected_nodes(model))
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
    for other_heater in model.nodes.values():
        if not other_heater.is_heater or int(other_heater.node_id) == int(heater.node_id):
            continue
        if getattr(other_heater, "assigned_sensor_id", None) == int(sensor.node_id):
            other_heater.assigned_sensor_id = None
            other_heater.sensor_pair_distance_mm = None
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


def _external_body_neighbors(model: ThermalGraphModel, node_id: int) -> set[int]:
    connected: set[int] = set()
    for edge in model.edges.values():
        other_id: int | None = None
        if int(edge.source) == int(node_id):
            other_id = int(edge.target)
        elif int(edge.target) == int(node_id):
            other_id = int(edge.source)
        if other_id is None:
            continue
        other = model.nodes.get(int(other_id))
        if other is None or not _is_body_node(model, int(other_id)):
            continue
        connected.add(int(other_id))
    return connected


def _is_body_node(model: ThermalGraphModel, node_id: int) -> bool:
    node = model.nodes.get(int(node_id))
    return bool(node is not None and not node.is_heater and not node.is_sensor)


def _normalized_weights(weights: list[float], count: int) -> list[float]:
    if count <= 0:
        return []
    values = [float(value) for value in weights[:count] if np.isfinite(float(value)) and float(value) >= 0.0]
    if len(values) != count or sum(values) <= 0.0:
        return [1.0 / float(count)] * count
    total = float(sum(values))
    return [float(value) / total for value in values]


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
    ids = [int(node_id) for node_id in (getattr(sensor, "readout_node_ids", []) or getattr(sensor, "sensor_connected_node_ids", []) or [])]
    weights = _normalized_weights(getattr(sensor, "readout_weights", []) or [], len(ids))
    weighted_values: list[tuple[float, float]] = []
    for node_id, weight in zip(ids, weights):
        if node_id not in node_index:
            continue
        value = float(temperatures_K[node_index[node_id]])
        if np.isfinite(value):
            weighted_values.append((value, float(weight)))
    if not weighted_values:
        return float("nan")
    total_weight = sum(weight for _value, weight in weighted_values)
    if total_weight <= 0.0:
        return float(np.mean([value for value, _weight in weighted_values]))
    return float(sum(value * weight for value, weight in weighted_values) / total_weight)


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
    for node_id in (getattr(sensor, "readout_node_ids", []) or getattr(sensor, "sensor_connected_node_ids", []) or []):
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
