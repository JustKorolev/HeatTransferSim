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
    mimo_default_heater_max_power_W: float = 30.0
    mimo_lambda_u: float = 1.0e-3
    # Control-effort penalty as a singular-value RATIO of the gain matrix, rather
    # than the absolute number above. sigma below this fraction of sigma_max is a
    # direction the plant barely has, so inverting through it costs enormous power
    # for no delivered temperature and makes the allocation flip between
    # near-equivalent solutions step to step. The effective weight is
    # max(mimo_lambda_u, (mimo_lambda_u_relative * sigma_max)**2), so an explicit
    # absolute value is never overridden downward. 0 disables the scaling.
    mimo_lambda_u_relative: float = 1.0e-4
    mimo_rho_du: float = 0.0
    # Max rate of change of a heater's COMMANDED power (W/s), per heater: a step may
    # move the command by at most slew*dt. This models the DRIVER, not the thermal
    # response -- a resistive heater on a PWM/DAC stage reaches full power in ~1 ms
    # (~30 kW/s for a 30 W heater), and the thermal lag is already represented by C
    # and L, so rate-limiting for "thermal realism" would double-count it.
    #
    # 30 W/s = full range in 1 s: honest about the hardware, conservative by three
    # orders of magnitude, and non-binding at any sane dt (it exceeds the [0, u_max]
    # clamp whenever dt >= 1 s), so it adds no phase lag. That matters because a rate
    # limiter is a nonlinearity: when it binds it costs phase margin and can drive
    # rate-limiter-induced oscillation, and this controller's anti-windup keys off
    # the PRE-slew clamp, so the integral keeps winding toward a command the actuator
    # is not delivering. Lower it only to encode a REAL constraint (a documented
    # driver ramp, or a deliberate thermal-shock limit) -- not as damping.
    mimo_heater_slew_rate_W_per_s: float = 30.0
    # Read ONLY by the removed PID+QP allocator (it clamped that scheme's desired
    # sensor rate before allocation). Kept so existing simulation_parameters.json
    # files still load; nothing consumes it.
    role_contact_tolerance_mm: float = 1.0e-6
    role_contact_tolerance_max_mm: float = 1.0
    role_contact_tolerance_growth_factor: float = 2.0
    mimo_integral_abs_max: float = 1.0e6
    # Passive sensor-drift source for the MIMO feedforward. True (default): a
    # disturbance observer -- estimate drift from the MEASURED sensor rate minus
    # the commanded-heater effect (d = dT/dt_measured - B_s @ u_prev). This is what
    # a real controller can do (no plant model on the MCU) and is reactive (needs a
    # step of history). False: the model-based oracle (project the full thermal RHS
    # with MIMO heating excluded) -- only available in simulation, kept for A/B.
    # Which heater controller runs in "heater_inputs" mode:
    #   "mimo_pi"   -> static-decoupling PI over the plant's DC gain G (default),
    #   "none"      -> nothing regulates the heaters (cryocoolers/manual only),
    #   "modal_lqr" -> the reduced-model LQR + regularized static state estimate
    #                  (needs a modal_controller.npz for the graph; see
    #                  tools/analyze_plant_modes.py). Runs open-loop if the
    #                  artifact is missing or mismatched.
    mimo_controller_scheme: str = "mimo_pi"
    # --- MIMO PI ---------------------------------------------------------------
    # Static-decoupling PI. The plant's DC gain G (n_ctrl x n_heaters, K/W) is
    # inverted once so the loop from the virtual command v (in KELVIN) to the
    # sensors is the identity; the PI then runs as n_ctrl INDEPENDENT scalar loops
    # in that decoupled space:
    #
    #     e = r - y                       (K)
    #     v = r_dev + Kp e + Ki \int e dt (K)
    #     u = QP(G, v)  s.t. 0 <= u <= u_max        (W)
    #
    # Decoupling is mandatory here, not a refinement: the RGA diagonal of this
    # plant is negative on 26 of 27 pairings, i.e. a per-pair SISO loop drives the
    # WRONG WAY once its neighbours close. Only ~0.7% of a heater's steady
    # influence lands on its own sensor.
    #
    # Gains are PER CONTROLLED SENSOR, because after decoupling a channel owns a
    # sensor, not a heater -- G+ mixes all sensors into every heater.
    mimo_pi_gain_matrix_path: str = ""  # sys-id run folder holding G (and the gain preset)
    # Ki = 1/lambda for a desired closed-loop time constant lambda. Do not ask for
    # lambda faster than the plant's fastest RETAINED mode (1182 s on
    # no_mli_high_res) or the command excites truncated dynamics: Ki <~ 8.5e-4.
    mimo_pi_ki: float = 1.0e-3
    # After decoupling every channel has unit DC gain, so Kp = tau/lambda. Starting
    # at 0 gives feedforward + integral, which is what the modal scheme effectively
    # ran; raise it if the approach is too sluggish.
    mimo_pi_kp: float = 0.0
    # Back-calculation anti-windup gain (dimensionless). Each step the integral is
    # pulled back by kt*(G u - v_cmd)*dt, i.e. by however much the allocator fell
    # short of the command. 1.0 absorbs the whole shortfall; 0 disables the
    # correction and lets the integral wind against the actuator bounds.
    mimo_pi_antiwindup_gain: float = 1.0
    # First-order low-pass on the sensor readouts feeding the loop, in seconds.
    #
    # kp is the loop's only damping term, and on this plant it is capped near 0.1
    # by a FAST path: sensors share cells with their heaters, those cells settle
    # inside one control step, so the proportional term closes an algebraic loop
    # and goes bang-bang above unity gain. The mode that actually needs damping is
    # three orders slower -- a 34 h ring at zeta ~ 0.14. Filtering separates them:
    # at 900 s with dt = 30 s a step in the fast path contributes dt/tau = 3% on
    # the first step, so kp can rise ~30x, while the 34 h mode picks up
    # arctan(w*tau) = 2.6 degrees of phase. 0 disables it.
    mimo_pi_measurement_filter_s: float = 0.0
    # Integral gain multiplier applied only while a channel is ABOVE its setpoint.
    #
    # The plant's authority is one-sided. Too cold is corrected by adding heater
    # power: direct, immediate, and bounded only by the heaters. Too hot can only
    # be corrected by taking power away and waiting for the cryocooler, at a rate
    # nothing in the loop controls. Overshoot is therefore strictly more expensive
    # than undershoot, and a symmetric integrator prices them the same.
    #
    # Above 1 the loop cuts power faster than it adds it, so it approaches the
    # setpoint from below and resists crossing. This biases the TRANSIENT only:
    # the error is zero at steady state, so both branches agree there and no
    # offset is introduced. 1.0 is symmetric, i.e. the old behaviour.
    mimo_pi_overshoot_integral_scale: float = 1.0
    # Measured override for the passive equilibrium, in kelvin. 0 = derive it from
    # the gain matrix's operating point.
    #
    # The derivation reads the cryocooler's lift curve, and on no_mli_high_res_v3
    # the SIMULATED cooler does not follow that curve: at a reported tip of
    # 49.1 K it removes 18.2 W where the PT60 curve says 29.9 W, with a slope of
    # 0.72 W/K against the curve's 1.07. Every reference derived from the curve is
    # therefore wrong by an amount the curve cannot predict -- the tangent's
    # 21.3 K and the no-load floor's 27.7 K both commanded far too much power.
    #
    # Measuring it needs no settled run. The cooler's response to its own tip
    # temperature and the sensor-to-tip gradient both equilibrate in minutes, so
    # fitting P_out(T_tip) and R = (T_sensor - T_tip)/P_out from a transient, then
    # solving P = P_out(setpoint - P*R) for the holding power, gives the value the
    # feedforward should reproduce. That put this graph at 17.4 W and 33.2 K,
    # against 23.1 W from the curve.
    mimo_pi_passive_reference_K: float = 0.0
    # Quiescence threshold for latching the held passive reference (r - y_passive).
    # y_ss = y_passive + G u only holds at steady state, so the capture waits until
    # the sensors stop moving this fast; until then the feedforward is 0 and the
    # integral carries the load. Latching during a transient bakes an arbitrary
    # constant into every subsequent command.
    mimo_pi_passive_latch_rate_K_per_s: float = 1.0e-4
    # How much more the allocator dislikes leaving a sensor SHORT of its target than
    # pushing it past. 1.0 is symmetric (the old behaviour).
    #
    # Symmetric is wrong when the plant is strongly coupled and the demands are
    # unequal: heating the coldest sensor necessarily overshoots a neighbour that is
    # nearly right, and a symmetric objective scores that overshoot exactly as badly
    # as the cold it removes. The best non-negative fit then stops early. On
    # no_mli_high_res every sensor sat below setpoint while the controller used 10 W
    # of 840 W and left 25 of 28 heaters idle.
    #
    # Raising this makes "keep heating until the coldest arrive" optimal, at the
    # explicit cost of letting better-placed sensors run warm. Set it to 1.0 if
    # overshoot on any channel is worse than undershoot on the others.
    mimo_undershoot_weight: float = 4.0
    # Derivative is deliberately absent: the conduction operator is symmetric, so
    # its spectrum is real and has no resonance to damp -- a D term would only
    # amplify sensor noise.
    # Quarantine cells that can absorb heat but have no conduction path to a
    # cryocooler. Such a cell is a thermal dead end: deposited power can only
    # raise its temperature, forever. Quarantined cells receive no power and are
    # excluded from whole-graph metrics (see cell_quarantine.py for why this
    # matters far more than the cell count suggests).
    # Rebuild C(T)/L(T) only once the temperature has moved this far (max over
    # cells, K) since the last rebuild. 0.0 rebuilds every step (the old behaviour).
    #
    # Worth more than the rebuild's own cost: a rebuild invalidates the implicit
    # stepper's stage-matrix cache, so the CG solve restarts from scratch, and it
    # churns several 8.7M-nnz sparse matrices per step. On a plant whose fastest
    # RETAINED mode is ~20 minutes, 0.25 K of drift is far below the uncertainty in
    # the property curves themselves.
    tdep_rebuild_delta_K: float = 0.25
    quarantine_inert_cells: bool = True
    # Opt-in per-cell conductance floor for the quarantine, in W/K. 0.0 (the
    # default) quarantines only cells with literally no conduction edges. A
    # nonzero floor is NOT safe as a default -- cell conductance scales with both
    # material and cell size, so a floor generous for a 1 mm copper cell would
    # wrongly quarantine a legitimate 1 mm G10 cell. Set it only for a graph you
    # know.
    quarantine_min_conductance_W_per_K: float = 0.0
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
    #
    # ON by default, unlike the fixed floor above. The objection to that one -- it
    # adds heat capacity that does not exist -- does not apply here, because a
    # 100x cap only touches cells whose capacitance is already three orders below
    # the graph's largest, which no real mesh produces. It is also the term that
    # matters with temperature-dependent properties on: cp(T) is clamped to the
    # NIST curves' lower bound, so a cell sitting at the temperature floor still
    # loses ~30x of its capacitance versus 50 K, and that spread alone is enough
    # to make jacobi-CG return a converged-but-wrong solution.
    implicit_capacitance_condition_cap: float = 100.0
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
        "mimo_hold_threshold_K",
        "mimo_coarse_threshold_K",
        "mimo_v_cmd_abs_max_K_per_s",
        "heater_sensor_pair_alpha",
        "drift_lpf_tau_s",
        "derivative_dt_floor_s",
        "mimo_freeze_integral_when_saturated",
        "mimo_passive_drift_from_measurement",
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
