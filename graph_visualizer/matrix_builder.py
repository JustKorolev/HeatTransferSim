"""Matrix and conductance-edge construction for thermal graph models."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import coo_matrix

from .models import EdgeMode, ThermalGraphModel

STEFAN_BOLTZMANN_W_M2K4 = 5.670374419e-8
DENSE_MATRIX_NODE_LIMIT = 6000
DENSE_MATRIX_MAX_TOTAL_BYTES = 768 * 1024 * 1024


def refresh_auto_edges(model: ThermalGraphModel) -> None:
    """Replace edges with 6-neighbor face-adjacent estimated conductances."""
    model.clear_edges()
    coord_index = model.coord_index()
    seen: set[tuple[int, int]] = set()
    for node_id, node in model.nodes.items():
        if _is_cad_role_node(node):
            continue
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
            if _is_cad_role_node(model.nodes[neighbor_id]):
                continue
            key = (min(node_id, neighbor_id), max(node_id, neighbor_id))
            if key in seen:
                continue
            seen.add(key)
            conductance = estimate_conductance(model.nodes[key[0]], model.nodes[key[1]])
            if conductance > 0.0:
                model.set_edge(key[0], key[1], conductance, EdgeMode.AUTO.value)
    model.metadata.edge_mode = EdgeMode.AUTO.value


def refresh_geometry_edges(model: ThermalGraphModel, default_contact_G_W_K: float = 0.1) -> None:
    """Replace edges using physical face contacts from node centers and sizes."""
    model.clear_edges()
    node_ids = model.ordered_node_ids()
    tolerance_mm = 1.0e-7
    face_groups: dict[tuple[int, int], dict[str, list[tuple[int, tuple[float, float], tuple[float, float]]]]] = {}
    for node_id in node_ids:
        if _is_cad_role_node(model.nodes[node_id]):
            continue
        bounds = _node_bounds_mm(model.nodes[node_id])
        if bounds is None:
            continue
        min_corner, max_corner = bounds
        for axis in range(3):
            other_axes = tuple(index for index in range(3) if index != axis)
            intervals = (
                (min_corner[other_axes[0]], max_corner[other_axes[0]]),
                (min_corner[other_axes[1]], max_corner[other_axes[1]]),
            )
            low_key = (axis, round(float(min_corner[axis]) / tolerance_mm))
            high_key = (axis, round(float(max_corner[axis]) / tolerance_mm))
            face_groups.setdefault(low_key, {"low": [], "high": []})["low"].append((node_id, *intervals))
            face_groups.setdefault(high_key, {"low": [], "high": []})["high"].append((node_id, *intervals))

    seen: set[tuple[int, int]] = set()
    for (axis, _plane), groups in face_groups.items():
        low_faces = sorted(groups["low"], key=lambda item: item[1][0])
        high_faces = sorted(groups["high"], key=lambda item: item[1][0])
        for source_id, source_a, source_b in high_faces:
            for target_id, target_a, target_b in low_faces:
                if target_a[0] >= source_a[1] - tolerance_mm:
                    break
                if target_a[1] <= source_a[0] + tolerance_mm:
                    continue
                if source_id == target_id:
                    continue
                key = (min(source_id, target_id), max(source_id, target_id))
                if key in seen:
                    continue
                overlap_a = min(source_a[1], target_a[1]) - max(source_a[0], target_a[0])
                overlap_b = min(source_b[1], target_b[1]) - max(source_b[0], target_b[0])
                if overlap_a <= tolerance_mm or overlap_b <= tolerance_mm:
                    continue
                seen.add(key)
                source = model.nodes[source_id]
                target = model.nodes[target_id]
                area_m2 = float(overlap_a * overlap_b * 1.0e-6)
                distance_m = _center_distance_m(source, target)
                conductance = _geometry_conductance(source, target, area_m2, distance_m, default_contact_G_W_K)
                if conductance <= 0.0:
                    continue
                model.set_edge(
                    source_id,
                    target_id,
                    conductance,
                    EdgeMode.AUTO.value,
                    edge_type="geometry_contact",
                    shared_area_m2=area_m2,
                    distance_m=distance_m,
                    contact_confidence="medium",
                )
    model.metadata.edge_mode = EdgeMode.AUTO.value


def refresh_radiation_from_exposed_faces(
    model: ThermalGraphModel,
    reference_temperature_K: float | None = None,
) -> int:
    """Update node radiation from faces adjacent to empty space."""
    exposed_areas_m2 = exposed_areas_from_geometry_m2(model)
    reference_temperature = float(
        reference_temperature_K
        if reference_temperature_K is not None
        else getattr(model.metadata, "T_sur_K", 293.15)
    )
    updated = 0
    for node_id, area_m2 in exposed_areas_m2.items():
        node = model.nodes[node_id]
        previous = (
            bool(node.is_exposed),
            float(node.radiating_area_m2),
            float(node.G_rad_W_K),
        )
        emissivity = max(0.0, float(getattr(node, "emissivity", 0.0)))
        G_rad = (
            4.0
            * emissivity
            * STEFAN_BOLTZMANN_W_M2K4
            * max(0.0, float(area_m2))
            * reference_temperature**3
        )
        node.is_exposed = area_m2 > 0.0
        node.radiating_area_m2 = float(max(0.0, area_m2))
        node.G_rad_W_K = float(G_rad)
        node.Grad_W_K = float(G_rad)
        node.R_rad_K_W = float(1.0 / G_rad) if G_rad > 0.0 else None
        current = (node.is_exposed, node.radiating_area_m2, node.G_rad_W_K)
        if current != previous:
            updated += 1
    return updated


def exposed_areas_from_geometry_m2(model: ThermalGraphModel) -> dict[int, float]:
    """Return exterior area per node after subtracting solid face contacts."""
    tolerance_mm = 1.0e-7
    areas_mm2: dict[int, float] = {}
    face_groups: dict[tuple[int, int], dict[str, list[tuple[int, tuple[float, float], tuple[float, float]]]]] = {}
    for node_id in model.ordered_node_ids():
        if _is_cad_role_node(model.nodes[node_id]):
            continue
        bounds = _node_bounds_mm(model.nodes[node_id])
        if bounds is None:
            continue
        min_corner, max_corner = bounds
        size = max_corner - min_corner
        sx, sy, sz = (max(0.0, float(v)) for v in size)
        areas_mm2[node_id] = 2.0 * (sx * sy + sx * sz + sy * sz)
        for axis in range(3):
            other_axes = tuple(index for index in range(3) if index != axis)
            intervals = (
                (min_corner[other_axes[0]], max_corner[other_axes[0]]),
                (min_corner[other_axes[1]], max_corner[other_axes[1]]),
            )
            low_key = (axis, round(float(min_corner[axis]) / tolerance_mm))
            high_key = (axis, round(float(max_corner[axis]) / tolerance_mm))
            face_groups.setdefault(low_key, {"low": [], "high": []})["low"].append((node_id, *intervals))
            face_groups.setdefault(high_key, {"low": [], "high": []})["high"].append((node_id, *intervals))

    for groups in face_groups.values():
        low_faces = sorted(groups["low"], key=lambda item: item[1][0])
        high_faces = sorted(groups["high"], key=lambda item: item[1][0])
        for source_id, source_a, source_b in high_faces:
            for target_id, target_a, target_b in low_faces:
                if target_a[0] >= source_a[1] - tolerance_mm:
                    break
                if target_a[1] <= source_a[0] + tolerance_mm:
                    continue
                if source_id == target_id:
                    continue
                overlap_a = min(source_a[1], target_a[1]) - max(source_a[0], target_a[0])
                overlap_b = min(source_b[1], target_b[1]) - max(source_b[0], target_b[0])
                if overlap_a <= tolerance_mm or overlap_b <= tolerance_mm:
                    continue
                shared_area_mm2 = overlap_a * overlap_b
                areas_mm2[source_id] = max(0.0, areas_mm2[source_id] - shared_area_mm2)
                areas_mm2[target_id] = max(0.0, areas_mm2[target_id] - shared_area_mm2)
    return {node_id: area_mm2 * 1.0e-6 for node_id, area_mm2 in areas_mm2.items()}


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


def _geometry_conductance(
    node_i: Any,
    node_j: Any,
    area_m2: float,
    distance_m: float,
    default_contact_G_W_K: float,
) -> float:
    same_component = bool(getattr(node_i, "component_name", "")) and (
        getattr(node_i, "component_name", "") == getattr(node_j, "component_name", "")
    )
    same_material = getattr(node_i, "material", "") == getattr(node_j, "material", "")
    if same_component or same_material:
        k_i = max(0.0, float(getattr(node_i, "k_W_mK", 0.0)))
        k_j = max(0.0, float(getattr(node_j, "k_W_mK", 0.0)))
        if k_i <= 0.0 or k_j <= 0.0:
            return 0.0
        k_eff = 2.0 / (1.0 / k_i + 1.0 / k_j)
        return float(k_eff * area_m2 / max(distance_m, 1.0e-12))
    return float(max(0.0, default_contact_G_W_K))


def _shared_face_area_and_distance(node_i: Any, node_j: Any) -> tuple[float, float]:
    bounds_i = _node_bounds_mm(node_i)
    bounds_j = _node_bounds_mm(node_j)
    if bounds_i is None or bounds_j is None:
        return 0.0, 0.0
    center_i = np.asarray(node_i.center_mm, dtype=float)
    center_j = np.asarray(node_j.center_mm, dtype=float)
    min_i, max_i = bounds_i
    min_j, max_j = bounds_j
    touch_axes: list[int] = []
    overlaps_mm: list[float] = []
    tolerance_mm = 1.0e-7
    for axis in range(3):
        gap = max(min_j[axis] - max_i[axis], min_i[axis] - max_j[axis])
        if abs(gap) <= tolerance_mm:
            touch_axes.append(axis)
            overlaps_mm.append(0.0)
        elif gap > 0.0:
            return 0.0, 0.0
        else:
            overlaps_mm.append(min(max_i[axis], max_j[axis]) - max(min_i[axis], min_j[axis]))
    if len(touch_axes) != 1:
        return 0.0, 0.0
    face_axis = touch_axes[0]
    other_axes = [axis for axis in range(3) if axis != face_axis]
    area_mm2 = max(0.0, overlaps_mm[other_axes[0]]) * max(0.0, overlaps_mm[other_axes[1]])
    distance_mm = float(np.linalg.norm(center_i - center_j))
    return float(area_mm2 * 1.0e-6), float(distance_mm * 1.0e-3)


def _node_bounds_mm(node: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if getattr(node, "center_mm", None) is None or getattr(node, "size_mm", None) is None:
        return None
    center = np.asarray(node.center_mm, dtype=float)
    size = np.asarray(node.size_mm, dtype=float)
    return center - size * 0.5, center + size * 0.5


def _center_distance_m(node_i: Any, node_j: Any) -> float:
    center_i = np.asarray(node_i.center_mm, dtype=float)
    center_j = np.asarray(node_j.center_mm, dtype=float)
    return float(np.linalg.norm(center_i - center_j) * 1.0e-3)


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


def build_matrices(
    model: ThermalGraphModel,
    *,
    dense_node_limit: int = DENSE_MATRIX_NODE_LIMIT,
) -> dict[str, np.ndarray]:
    """Build matrix arrays using sorted node IDs as row/column ordering."""
    node_ids = np.array(model.ordered_node_ids(), dtype=int)
    size = len(node_ids)
    index = {node_id: row for row, node_id in enumerate(node_ids)}
    coords = np.zeros((size, 3), dtype=int)
    C = np.zeros(size, dtype=float)
    Grad = np.zeros(size, dtype=float)
    G_rad = np.zeros(size, dtype=float)
    initial_temperature_K = np.zeros(size, dtype=float)
    is_heater = np.zeros(size, dtype=bool)
    heater_ids = np.full(size, -1, dtype=int)
    heater_min_power_W = np.zeros(size, dtype=float)
    heater_max_power_W = np.zeros(size, dtype=float)
    heater_efficiency = np.ones(size, dtype=float)
    is_sensor = np.zeros(size, dtype=bool)
    sensor_ids = np.full(size, -1, dtype=int)
    sensor_noise_std_K = np.zeros(size, dtype=float)
    sensor_bias_K = np.zeros(size, dtype=float)
    sensor_time_constant_s = np.zeros(size, dtype=float)
    has_cryocooler = np.zeros(size, dtype=bool)

    for node_id in node_ids:
        row = index[int(node_id)]
        node = model.nodes[int(node_id)]
        coords[row, :] = np.array(node.coord, dtype=int)
        C[row] = float(node.C_J_K)
        if _is_cad_role_node(node):
            Grad[row] = 0.0
            G_rad[row] = 0.0
        else:
            Grad[row] = float(node.Grad_W_K)
            G_rad[row] = float(node.G_rad_W_K if node.G_rad_W_K > 0.0 else node.Grad_W_K)
        initial_temperature_K[row] = float(node.initial_temperature_K)
        is_heater[row] = bool(node.is_heater)
        heater_ids[row] = int(node.heater.heater_id)
        heater_min_power_W[row] = float(node.heater.heater_min_power_W)
        heater_max_power_W[row] = float(node.heater.heater_max_power_W)
        heater_efficiency[row] = float(node.heater.heater_efficiency)
        is_sensor[row] = bool(node.is_sensor)
        sensor_ids[row] = int(node.sensor.sensor_id)
        sensor_noise_std_K[row] = float(node.sensor.sensor_noise_std_K)
        sensor_bias_K[row] = float(node.sensor.sensor_bias_K)
        sensor_time_constant_s[row] = float(node.sensor.sensor_time_constant_s)
        has_cryocooler[row] = bool(node.has_cryocooler)

    matrices = {
        "node_ids": node_ids,
        "coords": coords,
        "C": C,
        "Grad": Grad,
        "G_rad": G_rad,
        "initial_temperature_K": initial_temperature_K,
        "is_heater": is_heater,
        "heater_ids": heater_ids,
        "heater_min_power_W": heater_min_power_W,
        "heater_max_power_W": heater_max_power_W,
        "heater_efficiency": heater_efficiency,
        "is_sensor": is_sensor,
        "sensor_ids": sensor_ids,
        "sensor_noise_std_K": sensor_noise_std_K,
        "sensor_bias_K": sensor_bias_K,
        "sensor_time_constant_s": sensor_time_constant_s,
        "has_cryocooler": has_cryocooler,
    }
    if not _should_build_dense_matrices(size, dense_node_limit):
        matrices["L"] = _sparse_laplacian_from_edges(model, index, size)
        return matrices

    G = np.zeros((size, size), dtype=float)
    for edge in model.edges.values():
        if _is_visual_role_contact_edge(edge):
            continue
        if edge.source not in index or edge.target not in index:
            continue
        i = index[edge.source]
        j = index[edge.target]
        conductance = max(0.0, float(edge.Gij_W_K))
        G[i, j] = conductance
        G[j, i] = conductance
    matrices["G"] = G
    matrices["L"] = np.diag(G.sum(axis=1)) - G
    return matrices


def _should_build_dense_matrices(size: int, dense_node_limit: int) -> bool:
    if size > int(dense_node_limit):
        return False
    dense_pair_bytes = int(size) * int(size) * np.dtype(float).itemsize * 2
    return dense_pair_bytes <= DENSE_MATRIX_MAX_TOTAL_BYTES


def _sparse_laplacian_from_edges(
    model: ThermalGraphModel,
    index: dict[int, int],
    size: int,
):
    diagonal = np.zeros(size, dtype=float)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for edge in model.edges.values():
        if _is_visual_role_contact_edge(edge):
            continue
        if edge.source not in index or edge.target not in index:
            continue
        i = index[edge.source]
        j = index[edge.target]
        conductance = max(0.0, float(edge.Gij_W_K))
        if conductance <= 0.0:
            continue
        diagonal[i] += conductance
        diagonal[j] += conductance
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([-conductance, -conductance])
    nonzero = np.nonzero(diagonal > 0.0)[0]
    rows.extend(nonzero.astype(int).tolist())
    cols.extend(nonzero.astype(int).tolist())
    data.extend(diagonal[nonzero].astype(float).tolist())
    return coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()


def _is_cad_role_node(node: Any) -> bool:
    return bool(getattr(node, "is_cad_role_node", False))


def _is_visual_role_contact_edge(edge: Any) -> bool:
    return (
        str(getattr(edge, "edge_type", "")) == "role_node_contact"
        or str(getattr(edge, "source_metadata", "")) == "cad_role_node_contact"
    )
