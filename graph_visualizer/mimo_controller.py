"""Small math helpers for the MIMO thermal controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class AllocationResult:
    u: np.ndarray
    G_thresholded: np.ndarray
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


def apply_cumulative_coupling_cutoff(G: np.ndarray, fraction: float) -> np.ndarray:
    matrix = np.asarray(G, dtype=float).copy()
    if matrix.ndim != 2 or matrix.size == 0:
        return matrix
    keep_fraction = min(max(float(fraction), 0.0), 1.0)
    if keep_fraction >= 1.0:
        return matrix
    for row_index in range(matrix.shape[0]):
        row = matrix[row_index, :]
        magnitudes = np.abs(row)
        total = float(np.sum(magnitudes))
        if total <= 0.0:
            continue
        order = np.argsort(magnitudes)[::-1]
        cumulative = np.cumsum(magnitudes[order]) / total
        keep_count = int(np.searchsorted(cumulative, keep_fraction, side="left")) + 1
        keep = set(int(index) for index in order[:keep_count])
        for col_index in range(matrix.shape[1]):
            if col_index not in keep:
                matrix[row_index, col_index] = 0.0
    return matrix


def allocate_bounded_weighted_least_squares(
    G: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
    max_powers: np.ndarray,
    u_prev: np.ndarray,
    lambda_regularization: float,
    rho_smoothness: float,
    coupling_cutoff_fraction: float = 0.95,
    u_nominal: np.ndarray | None = None,
) -> AllocationResult:
    raw_G = np.asarray(G, dtype=float)
    if raw_G.ndim != 2:
        raise ValueError("MIMO gain matrix G must be two-dimensional.")
    ns, nh = raw_G.shape
    virtual = np.asarray(v, dtype=float).reshape(-1)
    maxima = np.asarray(max_powers, dtype=float).reshape(-1)
    previous = np.asarray(u_prev, dtype=float).reshape(-1)
    nominal = np.zeros(nh, dtype=float) if u_nominal is None else np.asarray(u_nominal, dtype=float).reshape(-1)
    if virtual.shape != (ns,):
        raise ValueError(f"Virtual command vector length {virtual.shape} does not match G rows {ns}.")
    if maxima.shape != (nh,):
        raise ValueError(f"Heater max-power vector length {maxima.shape} does not match G columns {nh}.")
    if previous.shape != (nh,):
        raise ValueError(f"Previous heater command vector length {previous.shape} does not match G columns {nh}.")
    if nominal.shape != (nh,):
        raise ValueError(f"Nominal heater command vector length {nominal.shape} does not match G columns {nh}.")
    if ns == 0 or nh == 0:
        return AllocationResult(
            np.zeros(nh, dtype=float),
            raw_G.copy(),
            0.0,
            0.0,
            False,
            True,
            "empty active MIMO allocation",
        )

    q = np.asarray(weights, dtype=float).reshape(-1)
    if q.shape != (ns,):
        raise ValueError(f"Sensor weight vector length {q.shape} does not match G rows {ns}.")
    q = np.where(np.isfinite(q), np.maximum(q, 0.0), 0.0)
    G_cut = apply_cumulative_coupling_cutoff(raw_G, coupling_cutoff_fraction)
    virtual = np.where(np.isfinite(virtual), virtual, 0.0)
    maxima = np.where(np.isfinite(maxima), np.maximum(maxima, 0.0), 0.0)
    previous = np.clip(np.where(np.isfinite(previous), previous, 0.0), 0.0, maxima)
    nominal = np.clip(np.where(np.isfinite(nominal), nominal, 0.0), 0.0, maxima)
    warnings: list[str] = []
    positive_power = maxima > 0.0
    if not np.any(positive_power):
        return AllocationResult(
            np.zeros(nh, dtype=float),
            G_cut,
            float(np.linalg.norm(virtual)),
            0.0,
            True,
            False,
            "zero heater authority",
            ("All active MIMO heaters have zero max power.",),
        )
    if not np.all(positive_power):
        warnings.append(f"{int(np.sum(~positive_power))} MIMO heater(s) have zero max power.")
        sub_result = allocate_bounded_weighted_least_squares(
            G_cut[:, positive_power],
            virtual,
            q,
            maxima[positive_power],
            previous[positive_power],
            lambda_regularization,
            rho_smoothness,
            1.0,
            nominal[positive_power],
        )
        u = np.zeros(nh, dtype=float)
        u[positive_power] = sub_result.u
        residual = G_cut @ (u - nominal) - virtual
        return AllocationResult(
            u,
            G_cut,
            float(np.linalg.norm(residual)),
            float(np.linalg.norm(u)),
            bool(sub_result.bounds_active),
            bool(sub_result.solver_success),
            str(sub_result.solver_message),
            tuple(warnings + list(sub_result.warnings)),
        )

    zero_rows = [index for index, row in enumerate(G_cut) if not np.any(np.abs(row) > 0.0)]
    if zero_rows:
        warnings.append(f"{len(zero_rows)} MIMO sensor row(s) have zero heater authority.")
    if not np.any(np.abs(G_cut) > 0.0):
        if np.linalg.norm(virtual) > 1.0e-12:
            warnings.append("MIMO allocation failed safely because the active gain matrix is zero.")
        return AllocationResult(
            np.zeros(nh, dtype=float),
            G_cut,
            float(np.linalg.norm(virtual)),
            0.0,
            False,
            False,
            "zero active gain matrix",
            tuple(warnings),
        )

    sqrt_weights = np.sqrt(q)
    A_parts = [sqrt_weights[:, None] * G_cut]
    b_parts = [sqrt_weights * (virtual + G_cut @ nominal)]
    lam = max(0.0, float(lambda_regularization))
    if lam > 0.0:
        scale = float(np.sqrt(lam))
        A_parts.append(scale * np.eye(nh))
        b_parts.append(scale * nominal)
    rho = max(0.0, float(rho_smoothness))
    if rho > 0.0:
        scale = float(np.sqrt(rho))
        A_parts.append(scale * np.eye(nh))
        b_parts.append(scale * previous)
    A = np.vstack(A_parts)
    b = np.concatenate(b_parts)
    try:
        result = lsq_linear(A, b, bounds=(np.zeros(nh, dtype=float), maxima), method="trf", lsmr_tol="auto")
    except Exception as exc:
        warnings.append(f"MIMO bounded allocation failed safely: {exc}")
        return AllocationResult(
            np.zeros(nh, dtype=float),
            G_cut,
            float(np.linalg.norm(virtual)),
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
    residual = G_cut @ (u - nominal) - virtual
    bounds_active = bool(np.any(u <= 1.0e-9) or np.any(u >= np.maximum(maxima - 1.0e-9, 0.0)))
    return AllocationResult(
        u,
        G_cut,
        float(np.linalg.norm(residual)),
        float(np.linalg.norm(u)),
        bounds_active,
        bool(result.success),
        str(result.message),
        tuple(warnings),
    )
