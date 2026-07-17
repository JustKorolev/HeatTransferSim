"""Diagnostics for comparing heat-transfer simulation steppers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .graph_io import load_graph_folder
from .models import ThermalGraphModel
from .simulation_model import PreparedSimulation, prepare_simulation
from .simulation_parameters import (
    SimulationParameters,
    apply_initial_temperature_parameter_payload,
    load_simulation_parameters,
)


@dataclass
class StepperComparisonMetrics:
    steps: int
    node_count: int
    dt_s: float
    max_abs_error_K: float
    mean_abs_error_K: float
    rmse_K: float
    final_max_abs_error_K: float
    final_rmse_K: float
    relative_frobenius_error: float
    worst_step_index: int
    worst_time_s: float
    worst_node_id: int
    implicit_elapsed_s: float
    reference_elapsed_s: float
    implicit_stepper: str
    reference_stepper: str


@dataclass
class StepperComparisonResult:
    node_ids: np.ndarray
    time_s: np.ndarray
    implicit_temperature_K: np.ndarray
    reference_temperature_K: np.ndarray
    error_K: np.ndarray
    metrics: StepperComparisonMetrics
    implicit_profile_ms: dict[str, float]
    reference_profile_ms: dict[str, float]
    implicit_warnings: list[str]
    reference_warnings: list[str]


def compare_implicit_cpu_to_expm_multiply(
    model: ThermalGraphModel,
    matrices: dict[str, Any],
    params: SimulationParameters,
    *,
    steps: int,
) -> StepperComparisonResult:
    """Run implicit sparse CPU and expm_multiply reference trajectories and compare them."""
    steps = max(1, int(steps))
    implicit_model = deepcopy(model)
    reference_model = deepcopy(model)
    implicit = _prepare_for_solver(
        implicit_model,
        matrices,
        params,
        solver="implicit",
    )
    reference = _prepare_for_solver(
        reference_model,
        matrices,
        params,
        solver="expm_multiply",
    )
    if implicit.sparse_implicit_stepper is None:
        raise RuntimeError("Implicit sparse CPU stepper is unavailable for this model/parameter set.")

    implicit_temperature, implicit_elapsed = _run_temperature_matrix(implicit, steps)
    reference_temperature, reference_elapsed = _run_temperature_matrix(reference, steps)
    if not np.array_equal(implicit.node_ids, reference.node_ids):
        raise RuntimeError("Stepper comparison produced mismatched node ordering.")
    time_s = np.arange(steps + 1, dtype=float) * float(params.dt_s)
    error = implicit_temperature - reference_temperature
    metrics = _comparison_metrics(
        node_ids=np.asarray(implicit.node_ids, dtype=int),
        time_s=time_s,
        error=error,
        reference=reference_temperature,
        dt_s=float(params.dt_s),
        implicit_elapsed_s=implicit_elapsed,
        reference_elapsed_s=reference_elapsed,
        implicit_stepper=_last_solver_name(implicit),
        reference_stepper=_last_solver_name(reference),
    )
    return StepperComparisonResult(
        node_ids=np.asarray(implicit.node_ids, dtype=int),
        time_s=time_s,
        implicit_temperature_K=implicit_temperature,
        reference_temperature_K=reference_temperature,
        error_K=error,
        metrics=metrics,
        implicit_profile_ms=dict(implicit.last_step_profile_ms),
        reference_profile_ms=dict(reference.last_step_profile_ms),
        implicit_warnings=list(implicit.warnings),
        reference_warnings=list(reference.warnings),
    )


def compare_graph_folder_steppers(
    graph_folder: Path,
    *,
    steps: int | None = None,
    to_end: bool = False,
    params_path: Path | None = None,
) -> StepperComparisonResult:
    """Load a graph folder and compare implicit sparse CPU against expm_multiply."""
    folder = Path(graph_folder)
    model, matrices = load_graph_folder(folder)
    params_file = params_path or folder / "simulation_parameters.json"
    params, extras = load_simulation_parameters(params_file)
    apply_initial_temperature_parameter_payload(model, extras)
    if to_end:
        steps_to_run = max(1, int(np.ceil(float(params.t_final_s) / max(float(params.dt_s), 1.0e-12))))
    else:
        steps_to_run = max(1, int(steps if steps is not None else 10))
    return compare_implicit_cpu_to_expm_multiply(model, matrices, params, steps=steps_to_run)


def save_stepper_comparison(result: StepperComparisonResult, output_dir: Path) -> Path:
    """Persist comparison matrices and summary metrics."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "node_ids.npy", result.node_ids)
    np.save(target / "time_s.npy", result.time_s)
    np.save(target / "implicit_temperature_K.npy", result.implicit_temperature_K)
    np.save(target / "reference_temperature_K.npy", result.reference_temperature_K)
    np.save(target / "temperature_error_K.npy", result.error_K)
    summary = {
        "metrics": asdict(result.metrics),
        "implicit_profile_ms": result.implicit_profile_ms,
        "reference_profile_ms": result.reference_profile_ms,
        "implicit_warnings": result.implicit_warnings,
        "reference_warnings": result.reference_warnings,
        "matrix_files": {
            "node_ids": "node_ids.npy",
            "time_s": "time_s.npy",
            "implicit_temperature_K": "implicit_temperature_K.npy",
            "reference_temperature_K": "reference_temperature_K.npy",
            "temperature_error_K": "temperature_error_K.npy",
        },
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return target


def _prepare_for_solver(
    model: ThermalGraphModel,
    matrices: dict[str, Any],
    params: SimulationParameters,
    *,
    solver: str,
) -> PreparedSimulation:
    local_matrices = _copy_matrix_payload(matrices)
    local_params = deepcopy(params)
    if solver == "implicit":
        local_params.implicit_sparse_simulation_enabled = True
        local_params.fast_sparse_simulation_enabled = False
    elif solver == "expm_multiply":
        local_params.implicit_sparse_simulation_enabled = False
        local_params.fast_sparse_simulation_enabled = False
    else:
        raise ValueError(f"Unknown solver {solver!r}.")
    prepared = prepare_simulation(model, local_matrices, local_params)
    # Force this comparison onto CPU paths even if optional GPU acceleration exists.
    prepared.gpu_stepper = None
    if solver == "expm_multiply":
        prepared.sparse_implicit_stepper = None
        prepared.fast_sparse_substeps = None
    elif solver == "implicit":
        prepared.fast_sparse_substeps = None
    return prepared


def _copy_matrix_payload(matrices: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in matrices.items():
        if hasattr(value, "copy"):
            try:
                copied[key] = value.copy()
                continue
            except Exception:
                pass
        copied[key] = deepcopy(value)
    return copied


def _run_temperature_matrix(prepared: PreparedSimulation, steps: int) -> tuple[np.ndarray, float]:
    rows = [np.asarray(prepared.temperatures_K, dtype=float).copy()]
    start = time.perf_counter()
    for _ in range(max(1, int(steps))):
        prepared.step_forward()
        rows.append(np.asarray(prepared.temperatures_K, dtype=float).copy())
    return np.vstack(rows), time.perf_counter() - start


def _comparison_metrics(
    *,
    node_ids: np.ndarray,
    time_s: np.ndarray,
    error: np.ndarray,
    reference: np.ndarray,
    dt_s: float,
    implicit_elapsed_s: float,
    reference_elapsed_s: float,
    implicit_stepper: str,
    reference_stepper: str,
) -> StepperComparisonMetrics:
    abs_error = np.abs(np.asarray(error, dtype=float))
    finite = np.isfinite(abs_error)
    if not np.any(finite):
        raise RuntimeError("Stepper comparison produced no finite error values.")
    worst_flat = int(np.nanargmax(np.where(finite, abs_error, np.nan)))
    worst_step, worst_col = np.unravel_index(worst_flat, abs_error.shape)
    reference_norm = float(np.linalg.norm(np.asarray(reference, dtype=float)))
    error_norm = float(np.linalg.norm(np.asarray(error, dtype=float)))
    final_error = abs_error[-1, :]
    return StepperComparisonMetrics(
        steps=int(error.shape[0] - 1),
        node_count=int(error.shape[1]),
        dt_s=float(dt_s),
        max_abs_error_K=float(np.nanmax(abs_error)),
        mean_abs_error_K=float(np.nanmean(abs_error)),
        rmse_K=float(np.sqrt(np.nanmean(np.asarray(error, dtype=float) ** 2))),
        final_max_abs_error_K=float(np.nanmax(final_error)),
        final_rmse_K=float(np.sqrt(np.nanmean(np.asarray(error[-1, :], dtype=float) ** 2))),
        relative_frobenius_error=error_norm / reference_norm if reference_norm > 0.0 else float("nan"),
        worst_step_index=int(worst_step),
        worst_time_s=float(time_s[int(worst_step)]),
        worst_node_id=int(node_ids[int(worst_col)]),
        implicit_elapsed_s=float(implicit_elapsed_s),
        reference_elapsed_s=float(reference_elapsed_s),
        implicit_stepper=str(implicit_stepper),
        reference_stepper=str(reference_stepper),
    )


def _last_solver_name(prepared: PreparedSimulation) -> str:
    profile = prepared.last_step_profile_ms
    if "cpu_sparse_implicit_step_ms" in profile:
        return "implicit_sparse_cpu"
    if "cpu_expm_multiply_ms" in profile:
        return "expm_multiply"
    if "cpu_fast_sparse_step_ms" in profile:
        return "fast_sparse_cpu"
    if "gpu_step_ms" in profile:
        return "gpu_sparse"
    if "dense_phi_matvec_ms" in profile:
        return "dense_phi_matvec"
    return "unknown"
