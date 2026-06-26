"""Small math helpers for the first-pass MIMO PI thermal controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecouplingResult:
    D: np.ndarray
    G_thresholded: np.ndarray
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


def update_nonnegative_integrator(
    integrator: np.ndarray,
    errors: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    eta = np.asarray(integrator, dtype=float).reshape(-1)
    e = np.asarray(errors, dtype=float).reshape(-1)
    dt = max(float(dt_s), 0.0)
    return np.maximum(0.0, eta + e * dt)


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


def compute_decoupling_matrix(
    G: np.ndarray,
    weights: np.ndarray,
    regularization_lambda: float,
    coupling_cutoff_fraction: float = 0.95,
) -> DecouplingResult:
    raw_G = np.asarray(G, dtype=float)
    if raw_G.ndim != 2:
        raise ValueError("MIMO gain matrix G must be two-dimensional.")
    ns, nh = raw_G.shape
    if ns == 0 or nh == 0:
        return DecouplingResult(np.zeros((nh, ns), dtype=float), raw_G.copy())

    q = np.asarray(weights, dtype=float).reshape(-1)
    if q.shape != (ns,):
        raise ValueError(f"Sensor weight vector length {q.shape} does not match G rows {ns}.")
    q = np.where(np.isfinite(q), np.maximum(q, 0.0), 0.0)
    G_cut = apply_cumulative_coupling_cutoff(raw_G, coupling_cutoff_fraction)

    warnings: list[str] = []
    zero_rows = [index for index, row in enumerate(G_cut) if not np.any(np.abs(row) > 0.0)]
    if zero_rows:
        warnings.append(f"{len(zero_rows)} MIMO sensor row(s) have zero heater authority.")

    weighted_G = G_cut * q[:, None]
    left = G_cut.T @ weighted_G
    lam = max(0.0, float(regularization_lambda))
    if lam > 0.0:
        left = left + lam * np.eye(nh)
    right = G_cut.T * q[None, :]
    try:
        D = np.linalg.solve(left, right)
    except np.linalg.LinAlgError:
        D = np.linalg.pinv(left) @ right
        warnings.append("MIMO decoupling solve was singular; used pseudo-inverse.")
    return DecouplingResult(np.asarray(D, dtype=float), G_cut, tuple(warnings))
