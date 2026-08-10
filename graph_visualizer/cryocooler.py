"""PT60 cryocooler capacity and physical-device distribution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .matrix_builder import _is_visual_role_contact_edge
from .models import ThermalGraphModel


PT60_MODEL_NAME = "pt60_lift_curve"


class PT60LiftCurve:
    """Inverse lookup for the measured PT60 cold-tip temperature curve."""

    MIN_POWER_W = 0.0
    DEFAULT_MAX_POWER_W = 150.0

    def __init__(
        self,
        max_power_w: float = DEFAULT_MAX_POWER_W,
        capacity_scale: float = 1.0,
        table_points: int = 4097,
    ) -> None:
        maximum = float(max_power_w)
        scale = float(capacity_scale)
        if not np.isfinite(maximum) or maximum < 0.0:
            raise ValueError("PT60 maximum cooling power must be finite and nonnegative.")
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("PT60 capacity scale must be finite and nonnegative.")
        points = max(2, int(table_points))
        self.max_power_w = maximum
        self.capacity_scale = scale
        self._power_grid_w = np.linspace(self.MIN_POWER_W, self.max_power_w, points)
        self._temperature_grid_k = self.temperature_for_power_w(self._power_grid_w)
        if self.max_power_w > self.MIN_POWER_W and not np.all(np.diff(self._temperature_grid_k) > 0.0):
            raise ValueError("PT60 lift curve must be monotonic over the selected power range.")

    @staticmethod
    def temperature_for_power_w(power_w: Any) -> np.ndarray:
        q = np.asarray(power_w, dtype=float)
        return -7.362e-6 * q**3 + 7.182e-3 * q**2 + 5.110e-1 * q + 27.669

    @property
    def minimum_temperature_k(self) -> float:
        return float(self._temperature_grid_k[0])

    @property
    def maximum_temperature_k(self) -> float:
        return float(self._temperature_grid_k[-1])

    def base_cooling_capacity_w(self, temperature_k: float) -> float:
        temperature = float(temperature_k)
        if not np.isfinite(temperature):
            return 0.0
        if temperature <= self.minimum_temperature_k:
            return 0.0
        if temperature >= self.maximum_temperature_k:
            return self.max_power_w
        return float(np.interp(temperature, self._temperature_grid_k, self._power_grid_w))

    def cooling_capacity_w(self, temperature_k: float) -> float:
        scaled = self.capacity_scale * self.base_cooling_capacity_w(temperature_k)
        return min(self.max_power_w, max(self.MIN_POWER_W, float(scaled)))


@dataclass(frozen=True)
class CryocoolerDevice:
    """One physical cooler and its normalized receiving-node weights."""

    identifier: str
    source_node_ids: tuple[int, ...]
    receiving_node_ids: tuple[int, ...]
    temperature_weights: tuple[float, ...]
    distribution_weights: tuple[float, ...]
    enabled: bool = True
    weighting_basis: str = "uniform"


def build_cryocooler_devices(
    model: ThermalGraphModel,
    simulation_node_ids: Sequence[int],
    capacitance_j_k: Sequence[float],
) -> tuple[tuple[CryocoolerDevice, ...], list[str]]:
    """Group tagged cells into physical coolers and resolve receiving nodes once."""
    ordered_ids = tuple(int(value) for value in simulation_node_ids)
    simulation_ids = set(ordered_ids)
    capacitance = {
        int(node_id): float(value)
        for node_id, value in zip(ordered_ids, np.asarray(capacitance_j_k, dtype=float).reshape(-1))
    }
    grouped: dict[str, list[int]] = {}
    for node_id in ordered_ids:
        node = model.nodes.get(int(node_id))
        if node is None or not bool(getattr(node, "has_cryocooler", False)):
            continue
        grouped.setdefault(_cryocooler_identifier(node), []).append(int(node_id))

    devices: list[CryocoolerDevice] = []
    warnings: list[str] = []
    for identifier, raw_source_ids in sorted(grouped.items()):
        source_ids = tuple(sorted(set(raw_source_ids)))
        source_nodes = [model.nodes[node_id] for node_id in source_ids]
        enabled_values = {bool(getattr(node, "cryocooler_enabled", True)) for node in source_nodes}
        if len(enabled_values) > 1:
            warnings.append(
                f"Cryocooler {identifier!r} has mixed per-cell enabled states; treating the physical unit as disabled."
            )
        device_enabled = len(enabled_values) == 1 and next(iter(enabled_values), True)
        receiving_ids, contact_area_by_node, explicit_receivers = _receiving_nodes_for_device(
            model,
            identifier,
            source_ids,
            simulation_ids,
        )
        if not receiving_ids and not explicit_receivers:
            component_names = {
                str(getattr(node, "component_name", "") or "").strip()
                for node in source_nodes
                if str(getattr(node, "component_name", "") or "").strip()
            }
            if len(component_names) == 1:
                component_name = next(iter(component_names))
                receiving_ids = tuple(
                    node_id
                    for node_id in ordered_ids
                    if str(getattr(model.nodes.get(node_id), "component_name", "") or "").strip() == component_name
                )
            if not receiving_ids:
                receiving_ids = tuple(node_id for node_id in source_ids if node_id in simulation_ids)
        if not receiving_ids:
            warnings.append(
                f"Cryocooler {identifier!r} has no valid receiving nodes; zero cooling will be applied."
            )
            devices.append(
                CryocoolerDevice(
                    identifier=identifier,
                    source_node_ids=source_ids,
                    receiving_node_ids=(),
                    temperature_weights=(),
                    distribution_weights=(),
                    enabled=device_enabled,
                )
            )
            continue
        weights, basis = _normalized_priority_weights(receiving_ids, contact_area_by_node, capacitance)
        devices.append(
            CryocoolerDevice(
                identifier=identifier,
                source_node_ids=source_ids,
                receiving_node_ids=receiving_ids,
                temperature_weights=weights,
                distribution_weights=weights,
                enabled=device_enabled,
                weighting_basis=basis,
            )
        )
    return tuple(devices), warnings


def normalized_weights(values: Sequence[float]) -> tuple[float, ...]:
    """Return finite nonnegative weights summing to one, or uniform weights."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return ()
    array = np.where(np.isfinite(array) & (array > 0.0), array, 0.0)
    total = float(np.sum(array))
    if total <= 0.0:
        array = np.full(array.size, 1.0 / float(array.size), dtype=float)
    else:
        array = array / total
    normalized = tuple(float(value) for value in array)
    if not np.isclose(sum(normalized), 1.0, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("Cryocooler weights must sum to one.")
    return normalized


def _cryocooler_identifier(node: Any) -> str:
    explicit = str(getattr(node, "cryocooler_id", "") or "").strip()
    if explicit:
        return explicit
    component = str(getattr(node, "component_name", "") or "").strip()
    if component:
        return component
    sources = tuple(
        sorted(
            str(value).strip()
            for value in (
                getattr(node, "role_source_components", [])
                or getattr(node, "source_components", [])
                or []
            )
            if str(value).strip()
        )
    )
    if sources:
        return " + ".join(sources)
    return f"cryocooler-node-{int(node.node_id)}"


def _receiving_nodes_for_device(
    model: ThermalGraphModel,
    identifier: str,
    source_ids: tuple[int, ...],
    simulation_ids: set[int],
) -> tuple[tuple[int, ...], dict[int, float], bool]:
    explicit_ids: list[int] = []
    explicit_areas: dict[int, float] = {}
    explicit_receivers = False
    for source_id in source_ids:
        source = model.nodes[source_id]
        raw_ids = list(getattr(source, "cryocooler_receiving_node_ids", []) or [])
        raw_areas = list(getattr(source, "cryocooler_contact_areas_m2", []) or [])
        if raw_ids:
            explicit_receivers = True
        for index, raw_id in enumerate(raw_ids):
            try:
                node_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if node_id not in simulation_ids or node_id not in model.nodes:
                continue
            explicit_ids.append(node_id)
            area = raw_areas[index] if index < len(raw_areas) else 0.0
            try:
                area_value = float(area)
            except (TypeError, ValueError):
                area_value = 0.0
            if np.isfinite(area_value) and area_value > 0.0:
                explicit_areas[node_id] = explicit_areas.get(node_id, 0.0) + area_value
    if explicit_receivers:
        return tuple(sorted(set(explicit_ids))), explicit_areas, True

    source_set = {int(value) for value in source_ids}
    contact_areas: dict[int, float] = {}
    interface_ids: set[int] = set()
    if not model.edges:
        return (), contact_areas, False
    # Find edges with exactly one endpoint in the (small) cryocooler source set.
    # Vectorised over the edge keys so this is fast on large graphs -- the old
    # per-edge Python scan over every edge (~millions) was the dominant cost of a
    # cryocooler assignment (it's a no-op until a cooler exists, hence the lag).
    edge_keys = list(model.edges.keys())
    key_array = np.asarray(edge_keys, dtype=np.int64)
    source_array = np.fromiter(source_set, dtype=np.int64, count=len(source_set))
    endpoint_a_in = np.isin(key_array[:, 0], source_array)
    endpoint_b_in = np.isin(key_array[:, 1], source_array)
    interface_rows = np.nonzero(endpoint_a_in ^ endpoint_b_in)[0]
    source_component = str(getattr(model.nodes[source_ids[0]], "component_name", "") or "").strip()
    for row in interface_rows:
        key = edge_keys[int(row)]
        low, high = int(key[0]), int(key[1])
        other_id = high if low in source_set else low
        if other_id not in simulation_ids or other_id not in model.nodes:
            continue
        edge = model.edges[key]
        # A CAD role marker is a VISUAL annotation: its edges carry G = 0 W/K and
        # exchange no heat. They must never count as a cooling interface. They used
        # to, because source_metadata is "cad_role_node_contact" and the check below
        # is a substring test for "contact" -- so markers were promoted to explicit
        # interfaces, took the majority of the contact-area weight, dragged the tip
        # temperature the lift curve is evaluated at far below the real cold tip,
        # and then had their share of the cooling applied to nodes that conduct
        # nowhere. On no_mli_high_res that delivered 2.9 W of an available 30.2 W.
        if _is_visual_role_contact_edge(edge):
            continue
        # Likewise a genuinely non-conducting edge: cooling routed through it is
        # discarded, so it is not an interface no matter what it is called.
        try:
            if not (float(getattr(edge, "Gij_W_K", 0.0)) > 0.0):
                continue
        except (TypeError, ValueError):
            continue
        edge_text = f"{getattr(edge, 'edge_type', '')} {getattr(edge, 'source_metadata', '')}".lower()
        other_component = str(getattr(model.nodes[other_id], "component_name", "") or "").strip()
        is_explicit_interface = "contact" in edge_text or "interface" in edge_text
        if source_component and other_component == source_component and not is_explicit_interface:
            continue
        interface_ids.add(other_id)
        try:
            area = float(getattr(edge, "shared_area_m2", 0.0))
        except (TypeError, ValueError):
            area = 0.0
        if np.isfinite(area) and area > 0.0:
            contact_areas[other_id] = contact_areas.get(other_id, 0.0) + area
    return tuple(sorted(interface_ids)), contact_areas, False


def _normalized_priority_weights(
    node_ids: tuple[int, ...],
    contact_area_by_node: dict[int, float],
    capacitance_by_node: dict[int, float],
) -> tuple[tuple[float, ...], str]:
    contact_values = [float(contact_area_by_node.get(node_id, 0.0)) for node_id in node_ids]
    if any(np.isfinite(value) and value > 0.0 for value in contact_values):
        return normalized_weights(contact_values), "contact_area"
    capacitance_values = [float(capacitance_by_node.get(node_id, 0.0)) for node_id in node_ids]
    if any(np.isfinite(value) and value > 0.0 for value in capacitance_values):
        return normalized_weights(capacitance_values), "thermal_capacitance"
    return normalized_weights([1.0] * len(node_ids)), "uniform"
