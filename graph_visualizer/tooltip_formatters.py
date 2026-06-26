"""Compact tooltip formatting for graph visualizer nodes and edges."""

from __future__ import annotations

from typing import Any


def format_node_tooltip(node_id: int, attrs: Any) -> str:
    """Return a compact, readable node tooltip."""
    lines = [
        f"Node {node_id}",
        f"coord: {getattr(attrs, 'coord', '?')}",
        f"cell: {getattr(attrs, 'cell_id', None) or getattr(attrs, 'coord', '?')}",
        f"component: {getattr(attrs, 'component_name', '') or '?'}",
        f"material: {getattr(attrs, 'material', '?')}",
        "-- geometry --",
        f"center_mm: {getattr(attrs, 'center_mm', None) or getattr(attrs, 'coord', '?')}",
        f"size_mm: {getattr(attrs, 'size_mm', None) or getattr(attrs, 'side_length_m', '?')}",
        f"level: {getattr(attrs, 'level', '?')}",
        f"volume: {_fmt(getattr(attrs, 'volume_m3', None))} m^3",
        f"occupancy: {_fmt(getattr(attrs, 'occupancy_fraction', None))}",
        f"confidence: {getattr(attrs, 'confidence', '?')}",
        "-- material thermal props --",
        f"rho: {_fmt(getattr(attrs, 'rho_kg_m3', None))} kg/m^3",
        f"cp: {_fmt(getattr(attrs, 'cp_J_kgK', None))} J/kg/K",
        f"k: {_fmt(getattr(attrs, 'k_W_mK', None))} W/m/K",
        f"emissivity: {_fmt(getattr(attrs, 'emissivity', None))}",
        "-- lumped thermal props --",
        f"mass: {_fmt(getattr(attrs, 'mass_kg', None))} kg",
        f"C: {_fmt(getattr(attrs, 'C_J_K', None))} J/K",
        f"initial T: {_fmt(getattr(attrs, 'initial_temperature_K', None))} K",
        f"initial T: {_fmt(_kelvin_to_celsius(getattr(attrs, 'initial_temperature_K', None)))} C",
        "-- radiation --",
        f"Grad: {_fmt(getattr(attrs, 'Grad_W_K', None))} W/K",
        f"exposed: {'yes' if getattr(attrs, 'is_exposed', False) else 'no'}",
        f"radiating area: {_fmt(getattr(attrs, 'radiating_area_m2', None))} m^2",
        f"G_rad: {_fmt(getattr(attrs, 'G_rad_W_K', None))} W/K",
        f"R_rad: {_fmt(getattr(attrs, 'R_rad_K_W', None))} K/W",
        f"heater: {'yes' if getattr(attrs, 'has_heater', False) else 'no'}",
        f"sensor: {'yes' if getattr(attrs, 'has_sensor', False) else 'no'}",
        f"cryocooler: {'yes' if getattr(attrs, 'has_cryocooler', False) else 'no'}",
    ]
    if getattr(attrs, "has_heater", False):
        heater = getattr(attrs, "heater", None)
        control = getattr(attrs, "heater_control", None)
        pid = getattr(control, "pid", None)
        manual = getattr(control, "manual", None)
        state = getattr(control, "pid_state", None)
        lines.extend(
            [
                "-- heater --",
                f"heater_id: {getattr(heater, 'heater_id', '?')}",
                f"heater min: {_fmt(getattr(heater, 'heater_min_power_W', None))} W",
                f"heater max: {_fmt(getattr(heater, 'heater_max_power_W', None))} W",
                f"efficiency: {_fmt(getattr(heater, 'heater_efficiency', None))}",
                f"control mode: {getattr(control, 'mode', '?')}",
                f"manual power: {_fmt(getattr(manual, 'power', None))} W",
                f"PID kp: {_fmt(getattr(pid, 'kp', None))}",
                f"PID ki: {_fmt(getattr(pid, 'ki', None))}",
                f"PID kd: {_fmt(getattr(pid, 'kd', None))}",
                f"PID setpoint: {_fmt(getattr(pid, 'setpoint', None))} K",
                f"PID leak: {_fmt(getattr(pid, 'integral_leak_per_s', None))} 1/s",
                f"PID integral: {_fmt(getattr(state, 'integral', None))}",
                f"PID prev error: {_fmt(getattr(state, 'previous_error', None))} K",
            ]
        )
    if getattr(attrs, "has_sensor", False):
        sensor = getattr(attrs, "sensor", None)
        lines.extend(
            [
                "-- sensor --",
                f"sensor_id: {getattr(sensor, 'sensor_id', '?')}",
                f"noise: {_fmt(getattr(sensor, 'sensor_noise_std_K', None))} K",
                f"bias: {_fmt(getattr(sensor, 'sensor_bias_K', None))} K",
                f"tau: {_fmt(getattr(sensor, 'sensor_time_constant_s', None))} s",
                f"MIMO setpoint: {_fmt(getattr(attrs, 'controller_setpoint_K', None))} K",
                f"MIMO weight: {_fmt(getattr(attrs, 'controller_weight', None))}",
                f"MIMO settling: {_fmt(getattr(attrs, 'sensor_settling_time_s', None))} s",
                f"coarse kP: {_fmt(getattr(attrs, 'controller_kp_coarse', None))}",
                f"coarse kI: {_fmt(getattr(attrs, 'controller_ki_coarse', None))}",
                f"hold kP: {_fmt(getattr(attrs, 'controller_kp_hold', None))}",
                f"hold kI: {_fmt(getattr(attrs, 'controller_ki_hold', None))}",
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
            f"type: {getter('edge_type', '?')}",
            f"area: {_fmt(getter('shared_area_m2'))} m^2",
            f"distance: {_fmt(getter('distance_m'))} m",
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


def _kelvin_to_celsius(value: Any) -> float | None:
    try:
        return float(value) - 273.15
    except (TypeError, ValueError):
        return None
