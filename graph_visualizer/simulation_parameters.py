"""Load and save heat-transfer simulation parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any


@dataclass
class SimulationParameters:
    dt_s: float = 1.0
    t_final_s: float = 3600.0
    playback_speed: float = 1.0
    use_ambient_radiation: bool = True
    T_env_K: float = 293.15
    input_mode: str = "zero"
    Kp_cooler: float = 0.5
    P_cooler_max: float = 10.0
    T_cooler_setpoint: float = 270.0
    mimo_controller_enabled: bool = False
    mimo_hold_threshold_K: float = 1.0
    mimo_coarse_threshold_K: float = 3.0
    mimo_default_heater_max_power_W: float = 30.0
    mimo_lambda_u: float = 1.0e-3
    mimo_rho_du: float = 0.0
    mimo_heater_slew_rate_W_per_s: float = 0.0
    mimo_v_cmd_abs_max_K_per_s: float = 0.25
    mimo_integral_abs_max: float = 1.0e6
    mimo_freeze_integral_when_saturated: bool = True
    enabled_heater_node_ids: tuple[int, ...] | None = None
    enabled_sensor_node_ids: tuple[int, ...] | None = None
    autoscale_temperature: bool = True
    color_min_K: float = 0.0
    color_max_K: float = 400.0
    colormap: str = "thermal_jet"
    loop_playback: bool = False
    save_trajectory: bool = False
    browser_simulation_size_warning: int = 1000
    display_update_interval_ms: float = 100.0


def load_simulation_parameters(path: Path) -> tuple[SimulationParameters, dict[str, Any]]:
    raw: dict[str, Any] = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw = {}
    migrated = _migrate_legacy_fields(raw)
    known = {field.name for field in fields(SimulationParameters)}
    deprecated = {
        "mimo_Kp_coarse",
        "mimo_Ki_coarse",
        "mimo_Kp_hold",
        "mimo_Ki_hold",
        "mimo_decoupling_lambda",
        "mimo_lambda_regularization",
        "mimo_rho_smoothness",
        "mimo_coupling_cutoff_fraction",
        "mimo_control_deadband_K",
        "mimo_hold_control_deadband_K",
        "mimo_negative_error_bleed_per_s",
        "mimo_hold_negative_error_bleed_per_s",
    }
    values = {key: migrated[key] for key in known if key in migrated}
    extras = {key: value for key, value in raw.items() if key not in known and key not in deprecated}
    return SimulationParameters(**values), extras


def save_simulation_parameters(path: Path, params: SimulationParameters, extras: dict[str, Any] | None = None) -> None:
    payload = dict(extras or {})
    payload.update(asdict(params))
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def initial_temperature_parameter_payload(model: Any) -> dict[str, Any]:
    """Return a JSON-friendly snapshot of per-node initial temperatures."""
    nodes = getattr(model, "nodes", {}) or {}
    return {
        "initial_temperature_by_node_K": {
            str(node_id): float(getattr(node, "initial_temperature_K", 293.15))
            for node_id, node in sorted(nodes.items(), key=lambda item: int(item[0]))
        }
    }


def apply_initial_temperature_parameter_payload(model: Any, extras: dict[str, Any]) -> int:
    """Apply saved per-node initial temperatures to a graph model.

    Returns the number of nodes updated.
    """
    payload = extras.get("initial_temperature_by_node_K")
    if not isinstance(payload, dict):
        return 0
    nodes = getattr(model, "nodes", {}) or {}
    updated = 0
    for raw_node_id, raw_temperature in payload.items():
        try:
            node_id = int(raw_node_id)
            temperature = float(raw_temperature)
        except (TypeError, ValueError):
            continue
        node = nodes.get(node_id)
        if node is None:
            continue
        node.initial_temperature_K = temperature
        updated += 1
    return updated


def _migrate_legacy_fields(raw: dict[str, Any]) -> dict[str, Any]:
    data = dict(raw)
    if "dt_s" not in data and "simulated_seconds_per_update" in data:
        data["dt_s"] = data["simulated_seconds_per_update"]
    if "t_final_s" not in data and "simulation_duration" in data:
        data["t_final_s"] = data["simulation_duration"]
    if "display_update_interval_ms" not in data and "display_update_interval_ms" in raw:
        data["display_update_interval_ms"] = raw["display_update_interval_ms"]
    if "mimo_lambda_u" not in data:
        if "mimo_lambda_regularization" in data:
            data["mimo_lambda_u"] = data["mimo_lambda_regularization"]
        elif "mimo_decoupling_lambda" in data:
            data["mimo_lambda_u"] = data["mimo_decoupling_lambda"]
    if "mimo_rho_du" not in data and "mimo_rho_smoothness" in data:
        data["mimo_rho_du"] = data["mimo_rho_smoothness"]
    return data
