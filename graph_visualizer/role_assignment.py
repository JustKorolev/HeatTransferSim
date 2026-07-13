"""Post-generation heater/sensor role assignment helpers."""

from __future__ import annotations

from .models import NodeProperties, ThermalGraphModel


def assign_matching_nodes_to_role(
    model: ThermalGraphModel,
    substring: str,
    role: str,
) -> list[int]:
    normalized_substring = normalize_role_match_text(substring)
    if not normalized_substring:
        raise ValueError("Enter a non-empty substring before assigning heaters or sensors.")
    normalized_role = str(role).strip().lower()
    if normalized_role not in {"heater", "sensor"}:
        raise ValueError("Choose whether matching cells should become heaters or sensors.")
    matched_ids: list[int] = []
    for node_id, node in sorted(model.nodes.items()):
        if not node_matches_role_substring(node, normalized_substring):
            continue
        assign_node_role(node, normalized_role)
        matched_ids.append(int(node_id))
    return matched_ids


def node_matches_role_substring(node: NodeProperties, normalized_substring: str) -> bool:
    return any(
        normalized_substring in normalize_role_match_text(value)
        for value in node_role_match_values(node)
    )


def node_role_match_values(node: NodeProperties) -> list[str]:
    values = [
        str(node.node_id),
        str(node.cell_id or ""),
        str(node.component_name or ""),
        str(node.node_type or ""),
    ]
    values.extend(str(value) for value in getattr(node, "source_components", []) or [])
    values.extend(str(value) for value in getattr(node, "role_source_components", []) or [])
    return [value for value in values if value.strip()]


def normalize_role_match_text(value: str) -> str:
    return str(value).replace("\\", "/").replace("-", "_").replace(" ", "_").lower()


def assign_node_role(node: NodeProperties, role: str) -> None:
    if role == "heater":
        node.is_heater = True
        node.is_sensor = False
        node.heater.heater_id = node.node_id
        node.heater_control.reset_pid_state()
        node.sensor.sensor_id = node.node_id
        node.assigned_heater_id = None
        node.sensor_connected_node_ids = []
        node.sensor_valid = True
        node.sensor_monitor_only = False
    elif role == "sensor":
        node.is_heater = False
        node.is_sensor = True
        node.heater_control.reset_pid_state()
        node.heater.heater_id = node.node_id
        node.sensor.sensor_id = node.node_id
        node.assigned_sensor_id = None
        node.sensor_control_mode = "manual"
        node.sensor_monitor_only = True


def node_has_heater_sensor_role(node: NodeProperties) -> bool:
    return bool(getattr(node, "is_heater", False) or getattr(node, "is_sensor", False))


def node_matches_level_filter(node: NodeProperties, min_level: int, max_level: int) -> bool:
    if getattr(node, "is_cad_role_node", False):
        return True
    return int(min_level) <= int(getattr(node, "level", 0)) <= int(max_level)
