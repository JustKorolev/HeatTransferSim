"""Compact tooltip formatting for graph visualizer nodes and edges."""

from __future__ import annotations

from typing import Any


def format_node_tooltip(node_id: int, attrs: Any) -> str:
    """Return a compact, readable node tooltip."""
    lines = [
        f"Node {node_id}",
        f"coord: {getattr(attrs, 'coord', '?')}",
        f"material: {getattr(attrs, 'material', '?')}",
        f"mass: {_fmt(getattr(attrs, 'mass_kg', None))} kg",
        f"C: {_fmt(getattr(attrs, 'C_J_K', None))} J/K",
        f"Grad: {_fmt(getattr(attrs, 'Grad_W_K', None))} W/K",
        f"heater: {'yes' if getattr(attrs, 'has_heater', False) else 'no'}",
        f"sensor: {'yes' if getattr(attrs, 'has_sensor', False) else 'no'}",
    ]
    if getattr(attrs, "has_heater", False):
        heater = getattr(attrs, "heater", None)
        lines.extend(
            [
                f"heater_id: {getattr(heater, 'heater_id', '?')}",
                f"heater max: {_fmt(getattr(heater, 'heater_max_power_W', None))} W",
                f"efficiency: {_fmt(getattr(heater, 'heater_efficiency', None))}",
            ]
        )
    if getattr(attrs, "has_sensor", False):
        sensor = getattr(attrs, "sensor", None)
        lines.extend(
            [
                f"sensor_id: {getattr(sensor, 'sensor_id', '?')}",
                f"noise: {_fmt(getattr(sensor, 'sensor_noise_std_K', None))} K",
                f"bias: {_fmt(getattr(sensor, 'sensor_bias_K', None))} K",
                f"tau: {_fmt(getattr(sensor, 'sensor_time_constant_s', None))} s",
            ]
        )
    return "\n".join(lines)


def format_edge_tooltip(source: int, target: int, attrs: Any) -> str:
    """Return a compact edge tooltip."""
    getter = attrs.get if isinstance(attrs, dict) else lambda key, default=None: getattr(attrs, key, default)
    return "\n".join(
        [
            f"Edge {source} -- {target}",
            f"Gij: {_fmt(getter('Gij_W_K'))} W/K",
            f"mode: {getter('source_metadata', getter('source_type', '?'))}",
        ]
    )


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "?"
    if number == 0.0:
        return "0"
    if abs(number) < 1.0e-3 or abs(number) >= 1.0e4:
        return f"{number:.3e}"
    return f"{number:.3f}"
