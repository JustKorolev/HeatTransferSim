"""Small math helpers for the MIMO thermal controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class AllocationResult:
    u: np.ndarray
    B_dyn: np.ndarray
    residual_norm: float
    command_norm: float
    bounds_active: bool
    solver_success: bool
    solver_message: str
    warnings: tuple[str, ...] = ()
    # Conditioning of this allocation, so a run can say WHICH part of the demand it
    # is declining to chase instead of thrashing between near-equivalent solutions.
    singular_values: tuple[float, ...] = ()
    lambda_effective: float = 0.0
    suppressed_directions: int = 0
    attenuated_command_fraction: float = 0.0


def weighted_rms_error(errors: np.ndarray, weights: np.ndarray) -> float:
    e = np.asarray(errors, dtype=float).reshape(-1)
    q = np.asarray(weights, dtype=float).reshape(-1)
    if e.size == 0:
        return 0.0
    q = np.where(np.isfinite(q), np.maximum(q, 0.0), 0.0)
    trace_q = float(np.sum(q))
    if trace_q <= 0.0:
        q = np.ones_like(e)
        trace_q = float(e.size)
    return float(np.sqrt(float(e.T @ (q * e)) / trace_q))


def allocate_thermal_rate_qp(
    B_dyn: np.ndarray,
    drift_dTdt: np.ndarray,
    v_cmd: np.ndarray,
    weights: np.ndarray,
    max_powers: np.ndarray,
    u_prev: np.ndarray,
    lambda_u: float,
    rho_du: float,
    max_delta_power: np.ndarray | None = None,
    u_ref: np.ndarray | None = None,
    *,
    # Keyword-only on purpose: existing callers pass max_delta_power and u_ref
    # POSITIONALLY, so a new positional parameter would silently land in u_ref's slot.
    absolute_target: bool = False,
    undershoot_weight: float = 1.0,
    lambda_u_relative: float = 1.0e-4,
) -> AllocationResult:
    raw_B = np.asarray(B_dyn, dtype=float)
    if raw_B.ndim != 2:
        raise ValueError("MIMO dynamic gain matrix B_dyn must be two-dimensional.")
    ns, nh = raw_B.shape
    drift = np.asarray(drift_dTdt, dtype=float).reshape(-1)
    command = np.asarray(v_cmd, dtype=float).reshape(-1)
    maxima = np.asarray(max_powers, dtype=float).reshape(-1)
    previous = np.asarray(u_prev, dtype=float).reshape(-1)
    reference = (
        np.zeros_like(previous, dtype=float)
        if u_ref is None
        else np.asarray(u_ref, dtype=float).reshape(-1)
    )
    delta_limit = (
        np.full(nh, np.inf, dtype=float)
        if max_delta_power is None
        else np.asarray(max_delta_power, dtype=float).reshape(-1)
    )
    if drift.shape != (ns,):
        raise ValueError(f"Drift dT/dt vector length {drift.shape} does not match B_dyn rows {ns}.")
    if command.shape != (ns,):
        raise ValueError(f"Rate command vector length {command.shape} does not match B_dyn rows {ns}.")
    if maxima.shape != (nh,):
        raise ValueError(f"Heater max-power vector length {maxima.shape} does not match B_dyn columns {nh}.")
    if previous.shape != (nh,):
        raise ValueError(f"Previous heater command vector length {previous.shape} does not match B_dyn columns {nh}.")
    if reference.shape != (nh,):
        raise ValueError(f"Heater reference vector length {reference.shape} does not match B_dyn columns {nh}.")
    if delta_limit.shape != (nh,):
        raise ValueError(f"Heater slew delta vector length {delta_limit.shape} does not match B_dyn columns {nh}.")
    if ns == 0 or nh == 0:
        return AllocationResult(
            np.zeros(nh, dtype=float),
            raw_B.copy(),
            0.0,
            0.0,
            False,
            True,
            "empty active MIMO allocation",
        )

    q = np.asarray(weights, dtype=float).reshape(-1)
    if q.shape != (ns,):
        raise ValueError(f"Sensor weight vector length {q.shape} does not match B_dyn rows {ns}.")
    q = np.where(np.isfinite(q), np.maximum(q, 0.0), 0.0)
    B = np.where(np.isfinite(raw_B), raw_B, 0.0)
    drift = np.where(np.isfinite(drift), drift, 0.0)
    command = np.where(np.isfinite(command), command, 0.0)
    maxima = np.where(np.isfinite(maxima), np.maximum(maxima, 0.0), 0.0)
    previous = np.clip(np.where(np.isfinite(previous), previous, 0.0), 0.0, maxima)
    reference = np.clip(np.where(np.isfinite(reference), reference, 0.0), 0.0, maxima)
    delta_limit = np.where(np.isfinite(delta_limit), np.maximum(delta_limit, 0.0), np.inf)
    warnings: list[str] = []
    positive_power = maxima > 0.0
    if not np.any(positive_power):
        return AllocationResult(
            np.zeros(nh, dtype=float),
            B,
            float(np.linalg.norm(drift - command)),
            0.0,
            True,
            False,
            "zero heater authority",
            ("All active MIMO heaters have zero max power.",),
        )
    if not np.all(positive_power):
        warnings.append(f"{int(np.sum(~positive_power))} MIMO heater(s) have zero max power.")
        sub_result = allocate_thermal_rate_qp(
            B[:, positive_power],
            drift,
            command,
            q,
            maxima[positive_power],
            previous[positive_power],
            lambda_u,
            rho_du,
            max_delta_power=delta_limit[positive_power],
            u_ref=reference[positive_power],
            absolute_target=absolute_target,
            # Forward these: dropping them silently disabled the asymmetric
            # undershoot penalty and the conditioning-scaled regularizer on any run
            # where a single heater was disabled or configured with 0 W.
            undershoot_weight=undershoot_weight,
            lambda_u_relative=lambda_u_relative,
        )
        u = np.zeros(nh, dtype=float)
        u[positive_power] = sub_result.u
        residual = drift + B @ (u - previous) - command
        return AllocationResult(
            u,
            B,
            float(np.linalg.norm(residual)),
            float(np.linalg.norm(u)),
            bool(sub_result.bounds_active),
            bool(sub_result.solver_success),
            str(sub_result.solver_message),
            tuple(warnings + list(sub_result.warnings)),
            sub_result.singular_values,
            sub_result.lambda_effective,
            sub_result.suppressed_directions,
            sub_result.attenuated_command_fraction,
        )

    zero_rows = [index for index, row in enumerate(B) if not np.any(np.abs(row) > 0.0)]
    if zero_rows:
        warnings.append(f"{len(zero_rows)} MIMO sensor row(s) have zero dynamic heater authority.")
    if not np.any(np.abs(B) > 0.0):
        if np.linalg.norm(drift - command) > 1.0e-12:
            warnings.append("MIMO allocation failed safely because the active dynamic gain matrix is zero.")
        return AllocationResult(
            np.zeros(nh, dtype=float),
            B,
            float(np.linalg.norm(drift - command)),
            0.0,
            False,
            False,
            "zero active dynamic gain matrix",
            tuple(warnings),
        )

    sqrt_weights = np.sqrt(q)
    A_parts = [sqrt_weights[:, None] * B]
    _ = A_parts  # rebuilt per reweighting pass below
    # Two different contracts, and they are not interchangeable.
    #
    # Incremental (the default, and what the rate-based PID+QP needed): command is
    # the desired CHANGE in sensor rate, so the target is B(u - u_prev) = cmd - drift
    # and the anchor B @ u_prev appears on the right.
    #
    # Absolute (absolute_target=True, what MIMO PI needs): command is the steady
    # deviation the plant must HOLD, so the target is simply B u = cmd. Passing the
    # anchor here would make each step add G^-1 cmd to the previous command -- an
    # unintended integrator that ramps to saturation and stays there, which is
    # exactly what MIMO PI did before this existed.
    #
    # u_prev still drives the rho_du smoothing term and the slew bounds in BOTH
    # modes; only the target anchor differs.
    anchor = np.zeros_like(command) if absolute_target else B @ previous
    b_parts = [sqrt_weights * (command - drift + anchor)]

    # Tikhonov weight, scaled to THIS gain matrix's spectrum.
    #
    # lambda_u is an absolute number, but it is compared against sigma^2 of B -- so
    # the same value damps completely differently on two graphs whose gains differ
    # by an order of magnitude, and "0.001" carries no meaning until you know
    # ||B||. On no_mli_high_res_v3 (sigma_1 = 33.0, sigma_27 = 0.090) the configured
    # 0.001 left the weakest direction 89% undamped, so the allocator inverted
    # through it: the setpoints' random +-0.5 K scatter projects almost entirely
    # onto those weak directions, and chasing it cost 11 W per K and flipped the
    # active heater set on 99% of steps -- 15 W pulses into single cells that
    # reached 250 K.
    #
    # lambda_u_relative fixes the scale: it is the singular-value RATIO below which
    # a direction is treated as unreachable, so lam >= (ratio * sigma_1)^2 damps
    # everything below ratio*sigma_1 while leaving the dominant mode untouched
    # (at the 1e-4 default, sigma_1 keeps 99.99% of its gain).
    #
    # It only ever RAISES an already-positive lambda_u. lambda_u = 0 means the
    # caller wants no regularization at all -- the exact-decoupling path, and what
    # the QP-is-the-decoupler tests assert -- and scaling must not quietly switch
    # it back on.
    left_singular, singular_values, _ = np.linalg.svd(B, full_matrices=False)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    lam_absolute = max(0.0, float(lambda_u))
    lam_floor = max(0.0, float(lambda_u_relative)) * sigma_max**2
    lam = max(lam_absolute, lam_floor) if lam_absolute > 0.0 else 0.0
    if lam > 0.0:
        scale = float(np.sqrt(lam))
        A_parts.append(scale * np.eye(nh))
        b_parts.append(scale * reference)
    rho = max(0.0, float(rho_du))
    if rho > 0.0:
        scale = float(np.sqrt(rho))
        A_parts.append(scale * np.eye(nh))
        b_parts.append(scale * previous)
    lower_bounds = np.maximum(np.zeros(nh, dtype=float), previous - delta_limit)
    upper_bounds = np.minimum(maxima, previous + delta_limit)
    lower_bounds = np.minimum(lower_bounds, upper_bounds)

    target = command - drift + anchor
    # Not clamped to >= 1. It was, which meant the knob could only ever ask for MORE
    # heat -- the one asymmetry this plant does not want. Heating is direct and
    # immediate; cooling is whatever the cryocooler happens to do, so overshoot
    # costs more than undershoot and a value below 1 is the physically motivated
    # setting here. Negative and zero are still meaningless, hence the floor.
    asym = max(1.0e-6, float(undershoot_weight))

    # How much of the demand this allocation is deliberately NOT chasing. A
    # direction with sigma^2 << lam is damped to nothing, and the component of the
    # target lying along it is simply not delivered -- which is the right answer,
    # but only if the run SAYS so rather than reporting a tracking error whose
    # cause looks like mistuning.
    if lam > 0.0 and singular_values.size:
        damping = singular_values**2 / (singular_values**2 + lam)
        projection = left_singular.T @ target
        total = float(np.linalg.norm(projection))
        attenuated_fraction = (
            float(np.linalg.norm((1.0 - damping) * projection) / total) if total > 1.0e-12 else 0.0
        )
        suppressed = int(np.count_nonzero(damping < 0.5))
    else:
        attenuated_fraction = 0.0
        suppressed = 0
    conditioning = (
        tuple(float(v) for v in singular_values),
        float(lam),
        suppressed,
        attenuated_fraction,
    )

    def _assemble(channel_weights):
        root = np.sqrt(channel_weights)
        parts_a = [root[:, None] * B]
        parts_b = [root * target]
        if lam > 0.0:
            scale = float(np.sqrt(lam))
            parts_a.append(scale * np.eye(nh))
            parts_b.append(scale * reference)
        if rho > 0.0:
            scale = float(np.sqrt(rho))
            parts_a.append(scale * np.eye(nh))
            parts_b.append(scale * previous)
        return np.vstack(parts_a), np.concatenate(parts_b)

    A, b = _assemble(q)
    try:
        result = lsq_linear(A, b, bounds=(lower_bounds, upper_bounds), method="trf", lsmr_tol="auto")
        # Asymmetric residual penalty, by iterative reweighting.
        #
        # A symmetric objective treats overshooting a sensor that is STILL too cold
        # as exactly as bad as leaving it cold, so with strong coupling and unequal
        # demands the best non-negative fit stops early and stays sparse: on
        # no_mli_high_res every sensor sat below setpoint while the controller used
        # 10 W of 840 W and left 25 of 28 heaters at zero.
        #
        # Weighting the UNDER-served channels more makes "keep heating until the
        # coldest ones arrive" the optimal answer. The weight depends on the sign of
        # the residual, which depends on the solution, so it is applied by
        # reweighting from the previous pass -- a couple of passes is plenty, and
        # each one is a bounded least-squares solve that is already cheap.
        #
        # Below 1 it reads the other way: under-delivering is cheaper than
        # over-delivering, so the allocator settles low and lets the integral walk
        # up. That is the right sign for a plant whose only fast actuator heats.
        if asym != 1.0:
            for _pass in range(2):
                residual = B @ np.asarray(result.x, dtype=float) - target
                reweighted = np.where(residual < 0.0, q * asym, q)
                A, b = _assemble(reweighted)
                result = lsq_linear(
                    A, b, bounds=(lower_bounds, upper_bounds), method="trf", lsmr_tol="auto"
                )
    except Exception as exc:
        warnings.append(f"MIMO bounded allocation failed safely: {exc}")
        return AllocationResult(
            np.zeros(nh, dtype=float),
            B,
            float(np.linalg.norm(drift - command)),
            0.0,
            False,
            False,
            str(exc),
            tuple(warnings),
            *conditioning,
        )
    if not result.success:
        warnings.append(f"MIMO bounded allocation solver did not converge: {result.message}")
        u = np.zeros(nh, dtype=float)
    else:
        u = np.asarray(result.x, dtype=float).reshape(-1)
    u = np.clip(np.where(np.isfinite(u), u, 0.0), 0.0, maxima)
    residual = drift + B @ (u - previous) - command
    bounds_active = bool(np.any(u <= lower_bounds + 1.0e-9) or np.any(u >= upper_bounds - 1.0e-9))
    return AllocationResult(
        u,
        B,
        float(np.linalg.norm(residual)),
        float(np.linalg.norm(u)),
        bounds_active,
        bool(result.success),
        str(result.message),
        tuple(warnings),
        *conditioning,
    )
