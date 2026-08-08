"""Load and save heat-transfer simulation parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cryocooler import PT60_MODEL_NAME, PT60LiftCurve


@dataclass
class SimulationParameters:
    dt_s: float = 1.0
    t_final_s: float = 3600.0
    playback_speed: float = 1.0
    use_ambient_radiation: bool = True
    # Radiative background temperatures. Exterior = the ambient surroundings the
    # outside of the assembly radiates to (room temperature). Interior = the
    # cryocooled vacuum enclosure the inner surfaces radiate to. Which surfaces
    # see which background is assigned by the view-factor classification; until
    # then every surface radiates to the exterior (T_env_K).
    T_env_K: float = 293.15
    interior_environment_temperature_K: float = 4.0
    # Surface-to-surface radiative coupling: ray-trace view factors over the
    # exposed faces and let parts exchange radiation with each other (not just a
    # background). One-time precompute at prepare; skipped for very large graphs.
    use_radiative_coupling: bool = False
    input_mode: str = "zero"
    cryocooler_model: str = PT60_MODEL_NAME
    cryocooler_max_power_W: float = PT60LiftCurve.DEFAULT_MAX_POWER_W
    cryocooler_capacity_scale: float = 1.0
    cryocooler_enabled: bool = True
    mimo_controller_enabled: bool = False
    mimo_hold_threshold_K: float = 1.0
    mimo_coarse_threshold_K: float = 3.0
    mimo_default_heater_max_power_W: float = 30.0
    mimo_lambda_u: float = 1.0e-3
    mimo_rho_du: float = 0.0
    mimo_heater_slew_rate_W_per_s: float = 0.0
    mimo_v_cmd_abs_max_K_per_s: float = 0.25
    heater_sensor_pair_alpha: float = 1.0
    role_contact_tolerance_mm: float = 1.0e-6
    role_contact_tolerance_max_mm: float = 1.0
    role_contact_tolerance_growth_factor: float = 2.0
    drift_lpf_tau_s: float = 2.0
    derivative_dt_floor_s: float = 1.0e-9
    mimo_integral_abs_max: float = 1.0e6
    mimo_freeze_integral_when_saturated: bool = True
    # Passive sensor-drift source for the MIMO feedforward. True (default): a
    # disturbance observer -- estimate drift from the MEASURED sensor rate minus
    # the commanded-heater effect (d = dT/dt_measured - B_s @ u_prev). This is what
    # a real controller can do (no plant model on the MCU) and is reactive (needs a
    # step of history). False: the model-based oracle (project the full thermal RHS
    # with MIMO heating excluded) -- only available in simulation, kept for A/B.
    mimo_passive_drift_from_measurement: bool = True
    # Which heater controller runs in "heater_inputs" mode:
    #   "pid_qp"    -> the standard PID + bounded QP allocator (default),
    #   "modal_lqr" -> the reduced-model LQR + regularized static state estimate
    #                  (needs a modal_controller.npz for the graph; see
    #                  tools/analyze_plant_modes.py). Falls back to pid_qp if the
    #                  artifact is missing or mismatched.
    mimo_controller_scheme: str = "pid_qp"
    # Path to the modal controller artifact; blank => look for
    # "<graph_folder>/modal_controller.npz".
    modal_controller_path: str = ""
    # Integral gain for the modal controller (offset-free + supplies the operating
    # holding power). Tunable; 0 disables integral action.
    modal_integral_gain: float = 0.02
    # Adaptive (learning) feedforward for the modal controller. Off by default.
    # When on, the controller uses recursive least squares to learn the model
    # error in the exact-DC-gain feedforward map: at steady state the integral's
    # accumulated output IS a measurement of (G_dc^-1_true - G_dc^-1_model) @ r,
    # so we regress it against the setpoint to estimate a correction matrix dM
    # and fold dM into the feedforward. Future (and revisited) setpoints then get
    # the corrected holding power immediately instead of waiting minutes for the
    # integral to re-accumulate. The transfer is bumpless (authority moves from
    # the reactive integral to the predictive feedforward without disturbing the
    # command). In-memory only: reset on controller reset / re-prepare, NOT
    # persisted to the graph artifact.
    modal_adaptive_ff_enabled: bool = False
    # RLS forgetting factor in (0, 1]. 1.0 = growing-window least squares (no
    # forgetting, exact but ever-more-confident); <1 lets a stale estimate fade
    # for a slowly time-varying plant. Keep close to 1 for this slow system.
    modal_adaptive_ff_forgetting: float = 0.999
    # Initial RLS covariance scale (P0 = this * I over the controlled-sensor
    # space). Larger => faster initial adaptation away from the zero-correction
    # prior, at the cost of more sensitivity to a noisy first sample.
    modal_adaptive_ff_p0: float = 1.0
    # Steady-state gate. A learning sample is only taken when every controlled
    # sensor's |dT/dt| is below rate_tol AND the max tracking error is below
    # error_tol -- so transient or saturated data never contaminates the
    # static-map regression.
    modal_adaptive_ff_rate_tol_K_per_s: float = 1.0e-3
    modal_adaptive_ff_error_tol_K: float = 0.05
    # Projection guardrail: the learned feedforward correction |dM @ r_sp| is
    # clamped, per heater, to this fraction of that heater's max power, so a bad
    # sample cannot drive the feedforward to an unphysical value. The effective
    # feedforward is clamped to [0, u_max] regardless. 0 disables the learned
    # correction's authority entirely.
    modal_adaptive_ff_max_correction_frac: float = 1.0
    enabled_heater_node_ids: tuple[int, ...] | None = None
    enabled_sensor_node_ids: tuple[int, ...] | None = None
    autoscale_temperature: bool = True
    color_min_K: float = 0.0
    color_max_K: float = 400.0
    colormap: str = "thermal_jet"
    loop_playback: bool = False
    save_trajectory: bool = False
    gpu_solver_enabled: bool = True
    implicit_sparse_simulation_method: str = "tr_bdf2"
    implicit_sparse_simulation_rtol: float = 1.0e-6
    implicit_sparse_simulation_maxiter: int = 300
    # Preconditioner for the implicit CG/BiCGSTAB solves. Plain Jacobi (diagonal)
    # is the default; block-Jacobi inverts small contiguous diagonal blocks and
    # can cut CG iterations on ill-conditioned cryogenic systems at some setup cost.
    implicit_sparse_block_jacobi_enabled: bool = False
    implicit_sparse_block_jacobi_size: int = 64
    implicit_sparse_adaptive_substeps_enabled: bool = True
    implicit_sparse_adaptive_target_delta_K: float = 1.0
    implicit_sparse_adaptive_max_substeps: int = 4
    implicit_sparse_residual_check_enabled: bool = True
    # Capacitance regularization: floor every cell's C so a degenerate near-zero
    # capacitance cannot blow up the stage-matrix condition number.
    #
    # OFF by default. It was introduced to stop tiny-C cells running away, but that
    # was a SYMPTOM of the zero-Laplacian bug: with temperature-dependent properties
    # the engine rebuilt L(T) from model.edges, and the low-memory loader supplied
    # none, so every cell was thermally isolated and any deposited power diverged.
    # With conduction restored those cells are coupled and the implicit solver
    # handles the stiffness, so a floor only adds heat capacity that does not exist.
    # Still available as an explicit knob for a genuinely pathological graph.
    implicit_capacitance_floor_J_K: float = 0.0
    # Auto floor: also floor capacitance at max(C)/this, i.e. cap the capacitance
    # ratio (a proxy for the stage-matrix condition number) at this value. Scales
    # to the graph, so it only bites on pathological spreads (5 mm cryo cells,
    # C~1e-3, next to bulk ~50) and leaves well-conditioned graphs untouched. 0
    # disables it; the effective floor is max(implicit_capacitance_floor_J_K,
    # max(C)/implicit_capacitance_condition_cap).
    implicit_capacitance_condition_cap: float = 0.0
    implicit_temperature_floor_K: float = 1.0e-3
    # Optional upper clamp (0 = disabled). KEEP IT DISABLED unless you know why you
    # are enabling it: the runaway "artifact cells" it was written for were cells
    # made isolated by the zero-Laplacian bug, not real geometry. It discards energy
    # wherever it binds, so on a correctly-conducting graph it hides physics instead
    # of fixing it. The step loop now reports how many cells each clamp touches.
    implicit_temperature_ceiling_K: float = 0.0
    use_temperature_dependent_properties: bool = False
    # Evaluate the lagged nonlinear terms (temperature-dependent C/L and
    # radiation) at a forward-Euler midpoint instead of the step-start
    # temperature. Second-order-in-dt operator splitting; only adds cost when
    # those terms are active (constant-property runs are unaffected).
    use_midpoint_property_coupling: bool = True
    copper_rrr: int = 100
    default_bolted_contact_conductance_W_m2K: float = 3000.0
    contact_conductance_temp_exponent: float = 1.0
    contact_conductance_reference_temperature_K: float = 293.15
    simulation_history_limit: int = 256
    # Ceiling on the replay history's memory, applied on top of
    # simulation_history_limit. An entry is 8 bytes/node, so a step-count limit
    # alone scales into gigabytes on multi-million-cell graphs. 0 disables the
    # clamp (step-count limit only).
    simulation_history_memory_budget_MB: float = 512.0
    live_step_profiling_enabled: bool = True
    live_step_profile_threshold_ms: float = 1000.0
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
        "cryocooler",
        "Kp_cooler",
        "kp_cooler",
        "kp_cooler_w_per_k",
        "T_cooler_setpoint",
        "setpoint_k",
        "cooler_setpoint_k",
        "P_cooler_max",
        "max_cooling",
        "max_cooling_w",
        "gpu_simulation_enabled",
        "gpu_simulation_max_substeps",
        "gpu_simulation_safety_factor",
        "fast_sparse_simulation_enabled",
        "fast_sparse_simulation_max_substeps",
        "fast_sparse_simulation_safety_factor",
        "implicit_sparse_simulation_enabled",
    }
    values = {key: migrated[key] for key in known if key in migrated}
    extras = {key: value for key, value in raw.items() if key not in known and key not in deprecated}
    return SimulationParameters(**values), extras


def save_simulation_parameters(path: Path, params: SimulationParameters, extras: dict[str, Any] | None = None) -> None:
    payload = dict(extras or {})
    parameter_payload = asdict(params)
    parameter_payload.pop("cryocooler_model", PT60_MODEL_NAME)
    max_power_w = parameter_payload.pop("cryocooler_max_power_W", PT60LiftCurve.DEFAULT_MAX_POWER_W)
    capacity_scale = parameter_payload.pop("cryocooler_capacity_scale", 1.0)
    enabled = parameter_payload.pop("cryocooler_enabled", True)
    payload.update(parameter_payload)
    payload["cryocooler"] = {
        "model": PT60_MODEL_NAME,
        "max_power_w": _finite_nonnegative_or_default(
            max_power_w,
            PT60LiftCurve.DEFAULT_MAX_POWER_W,
        ),
        "capacity_scale": _finite_nonnegative_or_default(capacity_scale, 1.0),
        "enabled": bool(enabled),
    }
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
    cryocooler = raw.get("cryocooler", {})
    if not isinstance(cryocooler, dict):
        cryocooler = {}
    data["cryocooler_model"] = PT60_MODEL_NAME
    maximum = _first_present(
        cryocooler,
        ("max_power_w", "max_cooling", "max_cooling_w"),
    )
    if maximum is None:
        maximum = data.get("cryocooler_max_power_W")
    if maximum is None:
        maximum = _first_present(data, ("P_cooler_max", "max_cooling", "max_cooling_w"))
    data["cryocooler_max_power_W"] = _finite_nonnegative_or_default(
        maximum,
        PT60LiftCurve.DEFAULT_MAX_POWER_W,
    )
    data["cryocooler_capacity_scale"] = _finite_nonnegative_or_default(
        cryocooler.get("capacity_scale", data.get("cryocooler_capacity_scale")),
        1.0,
    )
    data["cryocooler_enabled"] = bool(
        cryocooler.get("enabled", data.get("cryocooler_enabled", True))
    )
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


def _first_present(data: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return None


def _finite_nonnegative_or_default(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(number) or number < 0.0:
        return float(default)
    return number
