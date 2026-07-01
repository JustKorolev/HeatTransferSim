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
    natural_dTdt: np.ndarray,
    v_cmd: np.ndarray,
    weights: np.ndarray,
    max_powers: np.ndarray,
    u_prev: np.ndarray,
    lambda_u: float,
    rho_du: float,
    max_delta_power: np.ndarray | None = None,
) -> AllocationResult:
    raw_B = np.asarray(B_dyn, dtype=float)
    if raw_B.ndim != 2:
        raise ValueError("MIMO dynamic gain matrix B_dyn must be two-dimensional.")
    ns, nh = raw_B.shape
    natural = np.asarray(natural_dTdt, dtype=float).reshape(-1)
    command = np.asarray(v_cmd, dtype=float).reshape(-1)
    maxima = np.asarray(max_powers, dtype=float).reshape(-1)
    previous = np.asarray(u_prev, dtype=float).reshape(-1)
    delta_limit = (
        np.full(nh, np.inf, dtype=float)
        if max_delta_power is None
        else np.asarray(max_delta_power, dtype=float).reshape(-1)
    )
    if natural.shape != (ns,):
        raise ValueError(f"Natural dT/dt vector length {natural.shape} does not match B_dyn rows {ns}.")
    if command.shape != (ns,):
        raise ValueError(f"Rate command vector length {command.shape} does not match B_dyn rows {ns}.")
    if maxima.shape != (nh,):
        raise ValueError(f"Heater max-power vector length {maxima.shape} does not match B_dyn columns {nh}.")
    if previous.shape != (nh,):
        raise ValueError(f"Previous heater command vector length {previous.shape} does not match B_dyn columns {nh}.")
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
    natural = np.where(np.isfinite(natural), natural, 0.0)
    command = np.where(np.isfinite(command), command, 0.0)
    maxima = np.where(np.isfinite(maxima), np.maximum(maxima, 0.0), 0.0)
    previous = np.clip(np.where(np.isfinite(previous), previous, 0.0), 0.0, maxima)
    delta_limit = np.where(np.isfinite(delta_limit), np.maximum(delta_limit, 0.0), np.inf)
    warnings: list[str] = []
    positive_power = maxima > 0.0
    if not np.any(positive_power):
        return AllocationResult(
            np.zeros(nh, dtype=float),
            B,
            float(np.linalg.norm(natural - command)),
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
            natural,
            command,
            q,
            maxima[positive_power],
            previous[positive_power],
            lambda_u,
            rho_du,
            delta_limit[positive_power],
        )
        u = np.zeros(nh, dtype=float)
        u[positive_power] = sub_result.u
        residual = natural + B @ u - command
        return AllocationResult(
            u,
            B,
            float(np.linalg.norm(residual)),
            float(np.linalg.norm(u)),
            bool(sub_result.bounds_active),
            bool(sub_result.solver_success),
            str(sub_result.solver_message),
            tuple(warnings + list(sub_result.warnings)),
        )

    zero_rows = [index for index, row in enumerate(B) if not np.any(np.abs(row) > 0.0)]
    if zero_rows:
        warnings.append(f"{len(zero_rows)} MIMO sensor row(s) have zero dynamic heater authority.")
    if not np.any(np.abs(B) > 0.0):
        if np.linalg.norm(natural - command) > 1.0e-12:
            warnings.append("MIMO allocation failed safely because the active dynamic gain matrix is zero.")
        return AllocationResult(
            np.zeros(nh, dtype=float),
            B,
            float(np.linalg.norm(natural - command)),
            0.0,
            False,
            False,
            "zero active dynamic gain matrix",
            tuple(warnings),
        )

    sqrt_weights = np.sqrt(q)
    A_parts = [sqrt_weights[:, None] * B]
    b_parts = [sqrt_weights * (command - natural)]
    lam = max(0.0, float(lambda_u))
    if lam > 0.0:
        scale = float(np.sqrt(lam))
        A_parts.append(scale * np.eye(nh))
        b_parts.append(np.zeros(nh, dtype=float))
    rho = max(0.0, float(rho_du))
    if rho > 0.0:
        scale = float(np.sqrt(rho))
        A_parts.append(scale * np.eye(nh))
        b_parts.append(scale * previous)
    A = np.vstack(A_parts)
    b = np.concatenate(b_parts)
    lower_bounds = np.maximum(np.zeros(nh, dtype=float), previous - delta_limit)
    upper_bounds = np.minimum(maxima, previous + delta_limit)
    lower_bounds = np.minimum(lower_bounds, upper_bounds)
    try:
        result = lsq_linear(A, b, bounds=(lower_bounds, upper_bounds), method="trf", lsmr_tol="auto")
    except Exception as exc:
        warnings.append(f"MIMO bounded allocation failed safely: {exc}")
        return AllocationResult(
            np.zeros(nh, dtype=float),
            B,
            float(np.linalg.norm(natural - command)),
            0.0,
            False,
            False,
            str(exc),
            tuple(warnings),
        )
    if not result.success:
        warnings.append(f"MIMO bounded allocation solver did not converge: {result.message}")
        u = np.zeros(nh, dtype=float)
    else:
        u = np.asarray(result.x, dtype=float).reshape(-1)
    u = np.clip(np.where(np.isfinite(u), u, 0.0), 0.0, maxima)
    residual = natural + B @ u - command
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
    )
