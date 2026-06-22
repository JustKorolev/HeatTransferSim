"""Matrix and conductance-edge construction for thermal graph models."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import EdgeMode, ThermalGraphModel


def refresh_auto_edges(model: ThermalGraphModel) -> None:
    """Replace edges with 6-neighbor face-adjacent estimated conductances."""
    model.clear_edges()
    coord_index = model.coord_index()
    seen: set[tuple[int, int]] = set()
    for node_id, node in model.nodes.items():
        i, j, k = node.coord
        for neighbor_coord in (
            (i + 1, j, k),
            (i - 1, j, k),
            (i, j + 1, k),
            (i, j - 1, k),
            (i, j, k + 1),
            (i, j, k - 1),
        ):
            neighbor_id = coord_index.get(neighbor_coord)
            if neighbor_id is None:
                continue
            key = (min(node_id, neighbor_id), max(node_id, neighbor_id))
            if key in seen:
                continue
            seen.add(key)
            conductance = estimate_conductance(model.nodes[key[0]], model.nodes[key[1]])
            if conductance > 0.0:
                model.set_edge(key[0], key[1], conductance, EdgeMode.AUTO.value)
    model.metadata.edge_mode = EdgeMode.AUTO.value


def estimate_conductance(node_i: Any, node_j: Any) -> float:
    """Estimate Gij for touching cubic cells using two half-cube resistances."""
    try:
        length_i = float(node_i.side_length_m)
        length_j = float(node_j.side_length_m)
        k_i = float(node_i.k_W_mK)
        k_j = float(node_j.k_W_mK)
    except (TypeError, ValueError):
        return 0.0
    if length_i <= 0.0 or length_j <= 0.0 or k_i <= 0.0 or k_j <= 0.0:
        return 0.0
    contact_area = min(length_i, length_j) ** 2
    if contact_area <= 0.0:
        return 0.0
    resistance = (length_i / 2.0) / (k_i * contact_area)
    resistance += (length_j / 2.0) / (k_j * contact_area)
    if resistance <= 0.0 or not np.isfinite(resistance):
        return 0.0
    return float(1.0 / resistance)


def apply_conductance_matrix(
    model: ThermalGraphModel, node_ids: np.ndarray, G: np.ndarray
) -> None:
    """Replace graph edges from a saved pairwise conductance matrix."""
    node_ids = np.asarray(node_ids, dtype=int)
    G = np.asarray(G, dtype=float)
    if G.shape != (len(node_ids), len(node_ids)):
        raise ValueError("G shape does not match node_ids length.")
    missing = sorted(set(int(v) for v in node_ids) - set(model.nodes))
    if missing:
        raise ValueError(f"Conductance matrix references missing node IDs: {missing}")
    model.clear_edges()
    for row, source in enumerate(node_ids):
        for col in range(row + 1, len(node_ids)):
            conductance = float(G[row, col])
            if conductance > 0.0:
                model.set_edge(int(source), int(node_ids[col]), conductance, EdgeMode.LOADED_G.value)
    model.metadata.edge_mode = EdgeMode.LOADED_G.value


def build_matrices(model: ThermalGraphModel) -> dict[str, np.ndarray]:
    """Build matrix arrays using sorted node IDs as row/column ordering."""
    node_ids = np.array(model.ordered_node_ids(), dtype=int)
    size = len(node_ids)
    index = {node_id: row for row, node_id in enumerate(node_ids)}
    coords = np.zeros((size, 3), dtype=int)
    C = np.zeros(size, dtype=float)
    Grad = np.zeros(size, dtype=float)
    G = np.zeros((size, size), dtype=float)
    has_heater = np.zeros(size, dtype=bool)
    heater_ids = np.full(size, -1, dtype=int)
    heater_min_power_W = np.zeros(size, dtype=float)
    heater_max_power_W = np.zeros(size, dtype=float)
    heater_efficiency = np.ones(size, dtype=float)
    has_sensor = np.zeros(size, dtype=bool)
    sensor_ids = np.full(size, -1, dtype=int)
    sensor_noise_std_K = np.zeros(size, dtype=float)
    sensor_bias_K = np.zeros(size, dtype=float)
    sensor_time_constant_s = np.zeros(size, dtype=float)

    for node_id in node_ids:
        row = index[int(node_id)]
        node = model.nodes[int(node_id)]
        coords[row, :] = np.array(node.coord, dtype=int)
        C[row] = float(node.C_J_K)
        Grad[row] = float(node.Grad_W_K)
        has_heater[row] = bool(node.has_heater)
        heater_ids[row] = int(node.heater.heater_id)
        heater_min_power_W[row] = float(node.heater.heater_min_power_W)
        heater_max_power_W[row] = float(node.heater.heater_max_power_W)
        heater_efficiency[row] = float(node.heater.heater_efficiency)
        has_sensor[row] = bool(node.has_sensor)
        sensor_ids[row] = int(node.sensor.sensor_id)
        sensor_noise_std_K[row] = float(node.sensor.sensor_noise_std_K)
        sensor_bias_K[row] = float(node.sensor.sensor_bias_K)
        sensor_time_constant_s[row] = float(node.sensor.sensor_time_constant_s)

    for edge in model.edges.values():
        if edge.source not in index or edge.target not in index:
            continue
        i = index[edge.source]
        j = index[edge.target]
        conductance = max(0.0, float(edge.Gij_W_K))
        G[i, j] = conductance
        G[j, i] = conductance

    L = np.diag(G.sum(axis=1)) - G
    return {
        "node_ids": node_ids,
        "coords": coords,
        "C": C,
        "Grad": Grad,
        "G": G,
        "L": L,
        "has_heater": has_heater,
        "heater_ids": heater_ids,
        "heater_min_power_W": heater_min_power_W,
        "heater_max_power_W": heater_max_power_W,
        "heater_efficiency": heater_efficiency,
        "has_sensor": has_sensor,
        "sensor_ids": sensor_ids,
        "sensor_noise_std_K": sensor_noise_std_K,
        "sensor_bias_K": sensor_bias_K,
        "sensor_time_constant_s": sensor_time_constant_s,
    }
