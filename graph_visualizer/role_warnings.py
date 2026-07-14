"""Heater/sensor warning state helpers used by visualizers."""

from __future__ import annotations

from typing import Any


def role_warning_reasons(node: Any) -> list[str]:
    """Return compact warning reasons for heater/sensor nodes."""
    reasons: list[str] = []
    if bool(getattr(node, "is_heater", False)):
        if not bool(getattr(node, "heater_valid", True)) or not bool(getattr(node, "heater_attached", True)):
            warning = str(getattr(node, "heater_warning", "") or "").strip()
            reasons.append(warning or "heater has no connected body deposition nodes")
        if getattr(node, "assigned_sensor_id", None) is None:
            reasons.append("heater has no assigned sensor")
    if bool(getattr(node, "is_sensor", False)):
        readout_nodes = list(getattr(node, "readout_node_ids", []) or getattr(node, "sensor_connected_node_ids", []) or [])
        if not bool(getattr(node, "sensor_valid", True)) or not readout_nodes:
            reasons.append("sensor has no connected body readout nodes")
    return reasons


def has_role_warning(node: Any) -> bool:
    return bool(role_warning_reasons(node))
