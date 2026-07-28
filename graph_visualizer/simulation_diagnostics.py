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
from .matrix_builder import _is_cad_role_node
from .models import ThermalGraphModel
from .simulation_model import PreparedSimulation, prepare_simulation
from .simulation_parameters import (
    SimulationParameters,
    apply_initial_temperature_parameter_payload,
    load_simulation_parameters,
)


def inter_component_conduction_report(model: ThermalGraphModel) -> list[dict[str, Any]]:
    """List every pair of distinct components (parts) joined by a CONDUCTION edge.

    Parts that should only exchange heat by radiation must have NO conduction
    edges between them. An entry here for a pair you expect to be separated by a
    vacuum gap means the voxelization bridged that gap (cells from the two parts
    touch because the gap is narrower than ~2 cells), giving a spurious thermal
    short. Sorted by total inter-part conductance (worst first). Role nodes
    (heaters/sensors/cryocoolers) are excluded."""
    component_of: dict[int, str] = {}
    for node_id, node in model.nodes.items():
        if _is_cad_role_node(node):
            continue
        component_of[int(node_id)] = str(getattr(node, "component_name", "") or "").strip()
    pairs: dict[tuple[str, str], dict[str, float]] = {}
    for edge in model.edges.values():
        comp_a = component_of.get(int(edge.source))
        comp_b = component_of.get(int(edge.target))
        if not comp_a or not comp_b or comp_a == comp_b:
            continue
        key = tuple(sorted((comp_a, comp_b)))
        entry = pairs.setdefault(key, {"edges": 0.0, "total_conductance_W_K": 0.0})
        entry["edges"] += 1.0
        entry["total_conductance_W_K"] += float(getattr(edge, "Gij_W_K", 0.0) or 0.0)
    report = [
        {
            "component_a": key[0],
            "component_b": key[1],
            "edges": int(value["edges"]),
            "total_conductance_W_K": float(value["total_conductance_W_K"]),
        }
        for key, value in pairs.items()
    ]
    report.sort(key=lambda row: row["total_conductance_W_K"], reverse=True)
    return report


def inter_component_conduction_report_from_folder(folder: str | Path) -> list[dict[str, Any]]:
    """Load a saved octree graph folder and run inter_component_conduction_report."""
    model, _matrices = load_graph_folder(Path(folder))
    return inter_component_conduction_report(model)


def format_inter_component_conduction_report(report: list[dict[str, Any]]) -> str:
    """Human-readable table of inter-component conduction connections."""
    if not report:
        return "No conduction edges between distinct components (no bridged gaps)."
    lines = [f"{'part A':28s} {'part B':28s} {'edges':>6s} {'total G [W/K]':>14s}"]
    for row in report:
        lines.append(
            f"{row['component_a'][:28]:28s} {row['component_b'][:28]:28s} "
            f"{row['edges']:>6d} {row['total_conductance_W_K']:>14.4g}"
        )
    return "\n".join(lines)


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
    reference_temperature, reference_elapsed = _run_reference_expm_matrix(reference, steps)
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


def compare_current_state_to_expm_multiply(
    model: ThermalGraphModel,
    matrices: dict[str, Any],
    params: SimulationParameters,
    *,
    node_ids: np.ndarray,
    initial_temperatures_K: np.ndarray,
    current_temperatures_K: np.ndarray,
    current_time_s: float,
    current_stepper: str = "current",
    current_elapsed_s: float = 0.0,
) -> StepperComparisonResult:
    """Compare an already-computed simulation state against one expm_multiply solve to that time."""
    target_time_s = max(0.0, float(current_time_s))
    ordered_node_ids = np.asarray(node_ids, dtype=int).reshape(-1)
    current_temperature = np.asarray(current_temperatures_K, dtype=float).reshape(-1)
    initial_temperature = np.asarray(initial_temperatures_K, dtype=float).reshape(-1)
    if current_temperature.shape != ordered_node_ids.shape:
        raise ValueError(
            f"Current temperature length {current_temperature.shape[0]} does not match node count {ordered_node_ids.shape[0]}."
        )
    if initial_temperature.shape != ordered_node_ids.shape:
        raise ValueError(
            f"Initial temperature length {initial_temperature.shape[0]} does not match node count {ordered_node_ids.shape[0]}."
        )
    reference_model = deepcopy(model)
    reference = _prepare_current_state_reference(
        reference_model,
        matrices,
        params,
        initial_temperature,
        target_time_s,
    )
    reference_elapsed = 0.0
    if target_time_s > 0.0:
        _, reference_elapsed = _run_reference_expm_matrix(reference, 1)
    if not np.array_equal(ordered_node_ids, reference.node_ids):
        raise RuntimeError("Current-state comparison produced mismatched node ordering.")
    reference_temperature = np.asarray(reference.temperatures_K, dtype=float).reshape(-1)
    time_s = np.array([target_time_s], dtype=float)
    current_matrix = current_temperature.reshape(1, -1)
    reference_matrix = reference_temperature.reshape(1, -1)
    error = current_matrix - reference_matrix
    warnings = list(reference.warnings)
    if bool(getattr(reference, "dynamic_heater_inputs", False)):
        warnings.append(
            "Reference is a one-shot expm_multiply solve to the current time; dynamic heater/controller/radiation "
            "inputs are treated as one constant-input interval rather than replayed step-by-step."
        )
    metrics = _comparison_metrics(
        node_ids=ordered_node_ids,
        time_s=time_s,
        error=error,
        reference=reference_matrix,
        dt_s=float(params.dt_s),
        implicit_elapsed_s=float(current_elapsed_s),
        reference_elapsed_s=reference_elapsed,
        implicit_stepper=str(current_stepper),
        reference_stepper=_last_solver_name(reference),
        steps_override=_step_count_for_time(target_time_s, float(params.dt_s)),
    )
    return StepperComparisonResult(
        node_ids=ordered_node_ids,
        time_s=time_s,
        implicit_temperature_K=current_matrix,
        reference_temperature_K=reference_matrix,
        error_K=error,
        metrics=metrics,
        implicit_profile_ms={},
        reference_profile_ms=dict(reference.last_step_profile_ms),
        implicit_warnings=[],
        reference_warnings=warnings,
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


def save_current_state_comparison(result: StepperComparisonResult, output_dir: Path) -> Path:
    """Persist current-state comparison vectors and summary metrics."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "node_ids.npy", result.node_ids)
    np.save(target / "time_s.npy", result.time_s)
    np.save(target / "current_temperature_K.npy", result.implicit_temperature_K)
    np.save(target / "reference_temperature_K.npy", result.reference_temperature_K)
    np.save(target / "temperature_error_K.npy", result.error_K)
    summary = {
        "metrics": asdict(result.metrics),
        "current_profile_ms": result.implicit_profile_ms,
        "reference_profile_ms": result.reference_profile_ms,
        "current_warnings": result.implicit_warnings,
        "reference_warnings": result.reference_warnings,
        "matrix_files": {
            "node_ids": "node_ids.npy",
            "time_s": "time_s.npy",
            "current_temperature_K": "current_temperature_K.npy",
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
    if solver not in {"implicit", "expm_multiply"}:
        raise ValueError(f"Unknown solver {solver!r}.")
    local_matrices = _copy_matrix_payload(matrices)
    local_params = deepcopy(params)
    local_params.gpu_solver_enabled = False  # deterministic CPU comparison
    prepared = prepare_simulation(model, local_matrices, local_params)
    prepared.gpu_implicit_stepper = None
    return prepared


def _prepare_current_state_reference(
    model: ThermalGraphModel,
    matrices: dict[str, Any],
    params: SimulationParameters,
    initial_temperatures_K: np.ndarray,
    target_time_s: float,
) -> PreparedSimulation:
    local_params = deepcopy(params)
    local_params.dt_s = max(float(target_time_s), 1.0e-30)
    local_params.t_final_s = max(float(getattr(local_params, "t_final_s", 0.0)), float(target_time_s))
    local_params.gpu_solver_enabled = False
    prepared = prepare_simulation(model, dict(matrices), local_params)
    prepared.gpu_implicit_stepper = None
    prepared.initial_temperatures_K = np.asarray(initial_temperatures_K, dtype=float).reshape(-1).copy()
    prepared.reset()
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


def _run_reference_expm_matrix(prepared: PreparedSimulation, steps: int) -> tuple[np.ndarray, float]:
    """Exact matrix-exponential reference trajectory (zero-input / passive + radiation).

    This is the accuracy reference the implicit solver is validated against. It
    advances the affine linear system z(t+dt) = expm(A_aug*dt) z per step, with
    the radiation term frozen per step (the same operator splitting the implicit
    solver uses). Heater/controller inputs are not replayed here.
    """
    from scipy.sparse import bmat, csr_matrix
    from scipy.sparse.linalg import expm_multiply

    n = len(prepared.node_ids)
    base_b = (
        np.asarray(prepared.base_b, dtype=float).reshape(-1)
        if prepared.base_b is not None
        else np.zeros(n, dtype=float)
    )
    rows = [np.asarray(prepared.temperatures_K, dtype=float).copy()]
    start = time.perf_counter()
    for _ in range(max(1, int(steps))):
        b = base_b + prepared._radiation_source_vector()
        A_aug = bmat(
            [
                [csr_matrix(prepared.A), csr_matrix(b.reshape(-1, 1))],
                [csr_matrix((1, n)), csr_matrix((1, 1))],
            ],
            format="csr",
        )
        prepared.z = np.asarray(expm_multiply(A_aug * float(prepared.params.dt_s), prepared.z), dtype=float)
        prepared.z[-1] = 1.0
        rows.append(np.asarray(prepared.temperatures_K, dtype=float).copy())
    elapsed = time.perf_counter() - start
    prepared.last_step_profile_ms = {"cpu_expm_multiply_ms": elapsed * 1000.0}
    return np.vstack(rows), elapsed


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
    steps_override: int | None = None,
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
        steps=int(error.shape[0] - 1 if steps_override is None else steps_override),
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


def _step_count_for_time(time_s: float, dt_s: float) -> int:
    dt = abs(float(dt_s))
    if dt <= 0.0:
        return 0
    return max(0, int(round(max(0.0, float(time_s)) / dt)))


def _last_solver_name(prepared: PreparedSimulation) -> str:
    profile = prepared.last_step_profile_ms
    if "cpu_expm_multiply_ms" in profile:
        return "expm_multiply"
    if "implicit_step_ms" in profile:
        return "implicit_sparse_gpu" if profile.get("implicit_backend_gpu", 0.0) >= 1.0 else "implicit_sparse_cpu"
    return "unknown"
