"""Everything a saved G matrix can be asked about the plant, as figures + a report.

G is the only model object the MIMO PI needs, and almost every claim anyone makes
about this plant is a claim about G: how coupled it is, how many directions it
really has, whether it could be paired one heater per sensor, which channels can
be held independently and which cannot. Those were previously answered one at a
time, by hand, in whatever notebook happened to be open -- so the answers drifted
and none of them were reproducible.

This computes them together from one artifact and writes:

    plant_analysis.md     the numbers, written out, meant to be read or pasted
    plant_analysis.json   the same numbers, for anything that wants to plot them
    *.csv                 per-channel / per-heater / spectrum tables
    *.png                 the figures

Every section is derived from G itself, so the report is reproducible from the
artifact alone. The two sections that need more than G -- an operating point and
a power budget -- say so and are skipped when the tab has not supplied them,
rather than inventing a reference.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .rga_report import RGA_PNG, has_pairing_verdict, render_rga_figure, rga_summary

ANALYSIS_DIRNAME = "analysis"
REPORT_MD = "plant_analysis.md"
REPORT_JSON = "plant_analysis.json"

# Directions weaker than this fraction of the strongest one are, for practical
# purposes, not directions the actuator set has: driving them costs 1/sigma and
# buys sigma. Two thresholds because "effective rank" means nothing without one.
RANK_TOLERANCES = (0.1, 0.01)
# How many singular directions get their own shape plot. Past the third the
# picture is noise on this plant, and the spectrum figure already carries the tail.
N_MODE_SHAPES = 3


@dataclass
class PlantAnalysis:
    """The computed numbers plus where each artifact landed."""

    out_dir: Path
    stats: dict[str, Any]
    report_path: Path
    json_path: Path
    figures: list[Path] = field(default_factory=list)
    tables: list[Path] = field(default_factory=list)


# --------------------------------------------------------------------- numbers
def compute_plant_analysis(
    G: np.ndarray,
    sensor_ids: Sequence[int],
    heater_ids: Sequence[int],
    *,
    setpoints_K: dict[int, float] | None = None,
    passive_reference_K: float | None = None,
    heater_max_power_W: dict[int, float] | float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Every statistic the report states, in one dict. No files, no matplotlib."""
    G = np.asarray(G, dtype=float)
    sensor_ids = [int(v) for v in sensor_ids]
    heater_ids = [int(v) for v in heater_ids]
    n_s, n_h = G.shape

    U, sigma, Vt = np.linalg.svd(G, full_matrices=False)
    # One pseudo-inverse, reused: the reachability costs and the least-squares
    # allocation are both G+ applied to something.
    G_pinv = np.linalg.pinv(G)

    stats: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_sensors": n_s,
        "n_heaters": n_h,
        "sensor_ids": sensor_ids,
        "heater_ids": heater_ids,
        "source_metadata": dict(metadata or {}),
    }
    stats["gain_structure"] = _gain_structure(G)
    stats["spectrum"] = _spectrum(sigma)
    stats["modes"] = _mode_shapes(U, Vt, sigma, sensor_ids, heater_ids)
    stats["pairing"] = _pairing(G, sensor_ids, heater_ids)
    stats["actuators"] = _actuator_structure(G, Vt, heater_ids)
    stats["reachability"] = _reachability(G, G_pinv, sensor_ids)
    stats["uniform_lift"] = _uniform_lift(G, G_pinv, sensor_ids, heater_ids)
    stats["operating_point"] = _operating_point(
        G, sensor_ids, heater_ids,
        setpoints_K=setpoints_K,
        passive_reference_K=passive_reference_K,
        heater_max_power_W=heater_max_power_W,
    )
    return stats


def _gain_structure(G: np.ndarray) -> dict[str, Any]:
    """How coupled G is, in plain magnitudes.

    A plant with one shared bottleneck to the sink has every entry the same order
    of magnitude; a plant that is genuinely 27 independent zones is nearly
    diagonal. The min/max entry pair separates those two cases in one line, and it
    does so without needing the inverse.
    """
    magnitude = np.abs(G)
    off_diagonal = magnitude.copy()
    square = G.shape[0] == G.shape[1]
    if square:
        np.fill_diagonal(off_diagonal, np.nan)
    return {
        "min_abs_K_per_W": float(magnitude.min()) if magnitude.size else 0.0,
        "max_abs_K_per_W": float(magnitude.max()) if magnitude.size else 0.0,
        "median_abs_K_per_W": float(np.median(magnitude)) if magnitude.size else 0.0,
        "dynamic_range": (
            float(magnitude.max() / magnitude.min()) if magnitude.size and magnitude.min() > 0 else None
        ),
        "fraction_positive": float((G > 0).mean()) if G.size else 0.0,
        "median_row_sum_K_per_W": float(np.median(magnitude.sum(axis=1))) if G.size else 0.0,
        "median_column_sum_K_per_W": float(np.median(magnitude.sum(axis=0))) if G.size else 0.0,
        # Below this, "every heater reaches every sensor" stops being a figure of
        # speech: it is the ratio of the weakest path to the strongest.
        "median_offdiagonal_to_diagonal": (
            float(np.nanmedian(np.nanmean(off_diagonal, axis=1) / np.maximum(np.abs(np.diag(G)), 1e-300)))
            if square
            else None
        ),
    }


def _spectrum(sigma: np.ndarray) -> dict[str, Any]:
    """Singular values, and how much of the plant each one actually carries.

    sigma_1's energy share is the number behind "there is really only one strong
    knob": it is the fraction of what the heaters can do that lies in a single
    direction, and it is what caps how differently two sensors can be held.
    """
    sigma = np.asarray(sigma, dtype=float)
    total = float((sigma ** 2).sum())
    energy = (sigma ** 2) / total if total > 0 else np.zeros_like(sigma)
    return {
        "singular_values": [float(v) for v in sigma],
        "energy_fraction": [float(v) for v in energy],
        "cumulative_energy_fraction": [float(v) for v in np.cumsum(energy)],
        "sigma_max": float(sigma[0]) if sigma.size else 0.0,
        "sigma_min": float(sigma[-1]) if sigma.size else 0.0,
        "condition_number": (
            float(sigma[0] / sigma[-1]) if sigma.size and sigma[-1] > 0 else float("inf")
        ),
        "top_energy_fraction": float(energy[0]) if energy.size else 0.0,
        "effective_rank": {
            f"tol_{tol:g}": int((sigma >= tol * sigma[0]).sum()) if sigma.size else 0
            for tol in RANK_TOLERANCES
        },
        # How many directions it takes to hold 90% / 99% of the plant.
        "directions_for_90pct": int(np.searchsorted(np.cumsum(energy), 0.90) + 1) if sigma.size else 0,
        "directions_for_99pct": int(np.searchsorted(np.cumsum(energy), 0.99) + 1) if sigma.size else 0,
    }


def _mode_shapes(
    U: np.ndarray, Vt: np.ndarray, sigma: np.ndarray,
    sensor_ids: Sequence[int], heater_ids: Sequence[int],
) -> dict[str, Any]:
    """The shape of the strongest directions, not just their size.

    A dominant direction whose sensor pattern is all-one-sign is the whole
    structure moving together -- which is what "one shared bottleneck" predicts
    and what makes differential shaping expensive. Sign-mixed patterns further
    down are the differential directions, and their sigma says what they cost.
    """
    count = min(N_MODE_SHAPES, U.shape[1], Vt.shape[0])
    modes = []
    for k in range(count):
        left = np.asarray(U[:, k], dtype=float)
        right = np.asarray(Vt[k, :], dtype=float)
        # Sign of a singular vector pair is arbitrary; fix it so the dominant
        # entry is positive, or "is it all one sign?" is not a stable question.
        if left.sum() < 0:
            left, right = -left, -right
        modes.append(
            {
                "index": k,
                "sigma": float(sigma[k]),
                "sensor_pattern": [float(v) for v in left],
                "heater_pattern": [float(v) for v in right],
                # 1.0 = every sensor moves the same way in this direction.
                "sensor_sign_agreement": float(np.abs(np.sign(left).sum()) / max(left.size, 1)),
                "heater_sign_agreement": float(np.abs(np.sign(right).sum()) / max(right.size, 1)),
                "dominant_sensor_id": int(sensor_ids[int(np.argmax(np.abs(left)))]),
                "dominant_heater_id": int(heater_ids[int(np.argmax(np.abs(right)))]),
            }
        )
    return {"count": count, "modes": modes}


def _pairing(G: np.ndarray, sensor_ids: Sequence[int], heater_ids: Sequence[int]) -> dict[str, Any]:
    """RGA plus the plain-magnitude companion, and the Niederlinski sign test.

    Three independent readings of the same question. The RGA says whether a
    pairing's gain survives the other loops closing; the paired-influence share
    says how little of the heater ever reached that sensor to begin with; the
    Niederlinski index says whether a diagonal controller with integral action can
    be stable at all. They are reported together because any one of them alone
    invites the objection that it is an artifact of how it was computed.
    """
    from .modal_reduction import relative_gain_array

    RGA = relative_gain_array(G)
    summary = rga_summary(G, RGA, sensor_ids, heater_ids)
    square = G.shape[0] == G.shape[1]
    result: dict[str, Any] = {"rga_summary": summary, "RGA": RGA}

    if square:
        row_sums = np.abs(G).sum(axis=1)
        diag = np.abs(np.diag(G))
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(row_sums > 0, diag / np.maximum(row_sums, 1e-300), np.nan)
        result["paired_influence_fraction"] = [float(v) for v in share]
        # Niederlinski: negative means a diagonal controller with integral action
        # CANNOT be stable, whatever its tuning -- a stronger statement than the
        # RGA's, and independent of it. Via slogdet because det of a 27x27 gain
        # matrix in K/W overflows a float outright.
        sign_det, logabs_det = np.linalg.slogdet(G)
        sign_prod = float(np.prod(np.sign(np.diag(G))))
        with np.errstate(divide="ignore", invalid="ignore"):
            log_prod = float(np.log(np.abs(np.diag(G))).sum())
        if sign_prod != 0.0 and np.isfinite(log_prod) and np.isfinite(logabs_det):
            result["niederlinski_index"] = float(sign_det * sign_prod * np.exp(logabs_det - log_prod))
        else:
            result["niederlinski_index"] = None
    else:
        result["paired_influence_fraction"] = None
        result["niederlinski_index"] = None
    return result


def _actuator_structure(G: np.ndarray, Vt: np.ndarray, heater_ids: Sequence[int]) -> dict[str, Any]:
    """Per-heater authority, and how much of it duplicates another heater's.

    Two heaters whose columns are collinear are one actuator with two names: no
    combination of them can produce anything a single one could not. That is what
    consolidating several heaters into one node does to the model, and it shows up
    here as a cosine near 1 -- a rank statement about the actuator set that no
    amount of controller tuning can undo.
    """
    norms = np.linalg.norm(G, axis=0)
    safe = np.maximum(norms, 1e-300)
    normalized = G / safe
    cosine = normalized.T @ normalized
    np.fill_diagonal(cosine, -np.inf)
    partner = np.argmax(cosine, axis=1) if cosine.size else np.array([], dtype=int)
    max_cosine = cosine[np.arange(cosine.shape[0]), partner] if cosine.size else np.array([])
    return {
        "column_norm_K_per_W": [float(v) for v in norms],
        "max_abs_K_per_W": [float(v) for v in np.abs(G).max(axis=0)] if G.size else [],
        "top_direction_participation": [float(abs(v)) for v in Vt[0, :]] if Vt.size else [],
        "max_cosine_with_another_heater": [float(v) for v in max_cosine],
        "most_similar_heater_id": [int(heater_ids[int(j)]) for j in partner],
        "median_max_cosine": float(np.median(max_cosine)) if max_cosine.size else None,
        "n_near_duplicate_pairs": int((max_cosine > 0.99).sum()) if max_cosine.size else 0,
    }


def _reachability(G: np.ndarray, G_pinv: np.ndarray, sensor_ids: Sequence[int]) -> dict[str, Any]:
    """What it costs to move one sensor 1 K while holding every other one still.

    ||G+ e_i|| is exactly that: the minimum-norm heater command that produces a
    unit deviation on sensor i alone. It is the concrete form of "this channel has
    no independent authority" -- a channel needing kilowatts to be shaped
    separately is not going to be shaped separately, and the number says so in
    watts rather than in adjectives.
    """
    if G_pinv.size == 0:
        return {"independent_control_cost_W_per_K": [], "worst_channels": []}
    # Column i of G+ IS G+ e_i, so the whole per-channel cost is one norm per column.
    cost = np.linalg.norm(G_pinv, axis=0)
    order = np.argsort(cost)[::-1]
    return {
        "independent_control_cost_W_per_K": [float(v) for v in cost],
        "median_cost_W_per_K": float(np.median(cost)),
        "worst_channels": [
            {"sensor_id": int(sensor_ids[int(i)]), "cost_W_per_K": float(cost[int(i)])}
            for i in order[:10]
        ],
        # How lopsided the set is. A ratio in the thousands means a handful of
        # channels are qualitatively different from the rest, not merely worse.
        "cost_ratio_worst_to_median": (
            float(cost.max() / np.median(cost)) if float(np.median(cost)) > 0 else None
        ),
    }


def _uniform_lift(
    G: np.ndarray, G_pinv: np.ndarray, sensor_ids: Sequence[int], heater_ids: Sequence[int]
) -> dict[str, Any]:
    """Hold every sensor 1 K above the passive equilibrium: what is left over?

    This is the controller's actual task in miniature, and it needs no operating
    point -- the answer is per kelvin of requested lift, so it is a property of the
    plant rather than of a setpoint. Solved twice: unconstrained, which is the
    algebraic floor, and with u >= 0, which is the real one, because heaters
    cannot cool and the difference between those two is exactly what the bounded
    allocator has to give up.
    """
    n_s, n_h = G.shape
    target = np.ones(n_s)
    result: dict[str, Any] = {"request": "uniform +1 K on every controlled sensor"}

    u_ls = G_pinv @ target
    residual_ls = target - G @ u_ls
    result["unconstrained"] = {
        "power_W_per_K": float(u_ls.sum()),
        "negative_power_heaters": int((u_ls < 0).sum()),
        "most_negative_W_per_K": float(u_ls.min()) if u_ls.size else 0.0,
        "residual_rms_K_per_K": float(np.sqrt(np.mean(residual_ls ** 2))),
        "residual_max_K_per_K": float(np.max(np.abs(residual_ls))) if residual_ls.size else 0.0,
        "residual_per_sensor": [float(v) for v in residual_ls],
    }

    try:
        from scipy.optimize import lsq_linear

        solution = lsq_linear(G, target, bounds=(0.0, np.inf), method="bvls")
        u_nn = np.asarray(solution.x, dtype=float)
        residual_nn = target - G @ u_nn
        result["nonnegative"] = {
            "power_W_per_K": float(u_nn.sum()),
            "active_heaters": int((u_nn > 1e-12).sum()),
            "residual_rms_K_per_K": float(np.sqrt(np.mean(residual_nn ** 2))),
            "residual_max_K_per_K": float(np.max(np.abs(residual_nn))) if residual_nn.size else 0.0,
            "residual_per_sensor": [float(v) for v in residual_nn],
            "power_per_heater_W_per_K": [float(v) for v in u_nn],
        }
        # The whole cost of one-sided actuation, in one number.
        result["nonnegativity_penalty_rms_K_per_K"] = float(
            result["nonnegative"]["residual_rms_K_per_K"]
            - result["unconstrained"]["residual_rms_K_per_K"]
        )
    except Exception as exc:  # noqa: BLE001 - the unconstrained answer still stands
        result["nonnegative"] = None
        result["nonnegative_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _operating_point(
    G: np.ndarray,
    sensor_ids: Sequence[int],
    heater_ids: Sequence[int],
    *,
    setpoints_K: dict[int, float] | None,
    passive_reference_K: float | None,
    heater_max_power_W: dict[int, float] | float | None,
) -> dict[str, Any]:
    """The real allocation, at the real setpoints, against the real power caps.

    Everything above is scale-free and true of the plant. This is the one section
    that needs an operating point, so it is also the one that can be missing:
    without a passive reference there is no deviation to ask for, and inventing
    one would produce a confident answer to a question nobody asked.
    """
    if setpoints_K is None or passive_reference_K is None:
        missing = []
        if setpoints_K is None:
            missing.append("per-sensor setpoints")
        if passive_reference_K is None:
            missing.append("a passive reference (metadata's passive_reference_K)")
        return {"available": False, "missing": missing}

    targets = np.array([float(setpoints_K.get(int(sid), np.nan)) for sid in sensor_ids])
    if not np.all(np.isfinite(targets)):
        return {
            "available": False,
            "missing": [
                f"setpoints for {int((~np.isfinite(targets)).sum())} of {len(sensor_ids)} sensors"
            ],
        }
    deviation = targets - float(passive_reference_K)

    if isinstance(heater_max_power_W, dict):
        caps = np.array(
            [float(heater_max_power_W.get(int(hid), np.inf)) for hid in heater_ids], dtype=float
        )
    elif heater_max_power_W is None:
        caps = np.full(len(heater_ids), np.inf)
    else:
        caps = np.full(len(heater_ids), float(heater_max_power_W))
    caps = np.where(caps > 0.0, caps, np.inf)

    try:
        from scipy.optimize import lsq_linear

        solution = lsq_linear(G, deviation, bounds=(np.zeros_like(caps), caps), method="bvls")
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "missing": [f"bounded solve failed: {type(exc).__name__}: {exc}"]}

    u = np.asarray(solution.x, dtype=float)
    achieved = G @ u
    error = achieved - deviation
    saturated = np.isfinite(caps) & (u >= caps - 1e-9)
    order = np.argsort(np.abs(error))[::-1]
    return {
        "available": True,
        "passive_reference_K": float(passive_reference_K),
        "setpoint_min_K": float(targets.min()),
        "setpoint_max_K": float(targets.max()),
        "requested_deviation_mean_K": float(deviation.mean()),
        "requested_deviation_spread_K": float(deviation.max() - deviation.min()),
        "total_power_W": float(u.sum()),
        "max_heater_power_W": float(u.max()) if u.size else 0.0,
        "active_heaters": int((u > 1e-9).sum()),
        "saturated_heaters": int(saturated.sum()),
        "saturated_heater_ids": [int(heater_ids[int(j)]) for j in np.nonzero(saturated)[0]],
        "error_rms_K": float(np.sqrt(np.mean(error ** 2))),
        "error_max_abs_K": float(np.max(np.abs(error))) if error.size else 0.0,
        "error_per_sensor_K": [float(v) for v in error],
        "power_per_heater_W": [float(v) for v in u],
        "worst_channels": [
            {"sensor_id": int(sensor_ids[int(i)]), "error_K": float(error[int(i)])}
            for i in order[:10]
        ],
    }
