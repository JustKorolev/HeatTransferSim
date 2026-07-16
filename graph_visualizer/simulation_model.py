"""Heat-transfer simulation model for octree thermal graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import time
from typing import Any, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.sparse import bmat, csr_matrix, diags, eye, issparse
from scipy.sparse.linalg import LinearOperator, bicgstab, cg, expm_multiply

from .mimo_controller import (
    allocate_thermal_rate_qp,
    weighted_rms_error,
)
from .models import ThermalGraphModel
from .role_pairing import (
    average_inverse_capacitance_for_sensor,
    refresh_heater_power_deposition_nodes,
    refresh_sensor_connected_nodes,
    sensor_readout_temperature_K,
)
from .simulation_parameters import SimulationParameters

STEFAN_BOLTZMANN_W_M2K4 = 5.670374419e-8


@dataclass
class GpuSparseStepper:
    cp: Any
    A_gpu: Any
    inv_C_gpu: Any
    base_b_gpu: Any
    radiation_coeff_gpu: Any
    use_ambient_radiation: bool
    ambient_K: float
    dt_s: float
    substeps: int
    temperatures_gpu: Any | None = None

    def set_state(self, temperatures_K: np.ndarray) -> None:
        self.temperatures_gpu = self.cp.asarray(np.asarray(temperatures_K, dtype=float).reshape(-1))

    def step(self, temperatures_K: np.ndarray, heater_power: np.ndarray) -> np.ndarray:
        cp = self.cp
        if self.temperatures_gpu is None:
            self.set_state(temperatures_K)
        temperatures = self.temperatures_gpu
        powers = cp.asarray(np.asarray(heater_power, dtype=float).reshape(-1))
        h = float(self.dt_s) / max(1, int(self.substeps))
        source = self.base_b_gpu + self.inv_C_gpu * powers
        for _ in range(max(1, int(self.substeps))):
            rhs = self.A_gpu @ temperatures + source
            if self.use_ambient_radiation:
                rhs = rhs + self.inv_C_gpu * self.radiation_coeff_gpu * (
                    float(self.ambient_K) ** 4 - temperatures**4
                )
            temperatures = temperatures + h * rhs
        self.temperatures_gpu = temperatures
        return cp.asnumpy(self.temperatures_gpu)


@dataclass
class SparseImplicitStepper:
    dt_s: float
    rtol: float
    maxiter: int
    solver: str
    state_operator: Any
    capacitance_J_K: np.ndarray | None = None
    method: str = "tr_bdf2"
    adaptive_substeps_enabled: bool = True
    adaptive_target_delta_K: float = 1.0
    adaptive_max_substeps: int = 4
    residual_check_enabled: bool = True
    gamma: float = 2.0 - float(np.sqrt(2.0))
    last_info: int = 0
    last_iterations: int = 0
    last_substeps: int = 1
    last_residual_norm: float = 0.0
    last_relative_residual_norm: float = 0.0
    last_predicted_delta_K: float = 0.0
    _stage_cache: dict[int, tuple[Any, Any | None, Any, Any | None]] = field(default_factory=dict)

    def step(self, temperatures_K: np.ndarray, source_K_s: np.ndarray) -> np.ndarray:
        temperatures = np.asarray(temperatures_K, dtype=float).reshape(-1)
        source = np.asarray(source_K_s, dtype=float).reshape(-1)
        if temperatures.shape != source.shape:
            raise ValueError(f"Source vector length {source.shape} does not match temperatures {temperatures.shape}.")
        if str(self.method).lower() != "tr_bdf2":
            return self._backward_euler_step(temperatures, source)
        substeps = self._adaptive_substep_count(temperatures, source)
        h = float(self.dt_s) / max(1, int(substeps))
        current = temperatures.copy()
        self.last_iterations = 0
        self.last_info = 0
        self.last_substeps = int(substeps)
        self.last_residual_norm = 0.0
        self.last_relative_residual_norm = 0.0
        for _ in range(max(1, int(substeps))):
            current = self._tr_bdf2_substep(current, source, h)
        if current.shape != temperatures.shape or not np.all(np.isfinite(current)):
            raise RuntimeError(f"{self.solver} returned an invalid temperature vector.")
        return current

    def _backward_euler_step(self, temperatures: np.ndarray, source: np.ndarray) -> np.ndarray:
        self.last_iterations = 0
        self.last_info = 0
        self.last_substeps = 1
        self.last_residual_norm = 0.0
        self.last_relative_residual_norm = 0.0
        self.last_predicted_delta_K = 0.0
        if self.capacitance_J_K is not None:
            capacitance = np.asarray(self.capacitance_J_K, dtype=float).reshape(-1)
            if capacitance.shape != temperatures.shape:
                raise ValueError(f"Capacitance vector length {capacitance.shape} does not match temperatures {temperatures.shape}.")
            rhs = float(self.dt_s) * (capacitance * source - np.asarray(self.state_operator @ temperatures, dtype=float).reshape(-1))
            system_matrix = (diags(capacitance, format="csr") + float(self.dt_s) * csr_matrix(self.state_operator)).tocsr()
        else:
            rhs = float(self.dt_s) * (
                np.asarray(self.state_operator @ temperatures, dtype=float).reshape(-1) + source
            )
            system_matrix = (eye(temperatures.shape[0], format="csr") - float(self.dt_s) * csr_matrix(self.state_operator)).tocsr()
        if not np.all(np.isfinite(rhs)):
            raise RuntimeError(f"{self.solver} received an invalid implicit update right-hand side.")
        if np.linalg.norm(rhs) <= 0.0:
            return temperatures.copy()
        preconditioner = _jacobi_preconditioner(system_matrix)
        result = self._solve_linear(system_matrix, np.asarray(rhs, dtype=float).reshape(-1), preconditioner, np.zeros_like(temperatures))
        if result.shape != temperatures.shape or not np.all(np.isfinite(result)):
            raise RuntimeError(f"{self.solver} returned an invalid temperature vector.")
        return temperatures + result

    def _tr_bdf2_substep(self, temperatures: np.ndarray, source: np.ndarray, h_s: float) -> np.ndarray:
        gamma = min(0.95, max(0.05, float(self.gamma)))
        h = max(float(h_s), 1.0e-30)
        alpha = 0.5 * gamma * h
        stage1_matrix, stage1_preconditioner, stage2_matrix, stage2_preconditioner = self._stage_matrices_for_h(h)
        if self.capacitance_J_K is not None:
            capacitance = np.asarray(self.capacitance_J_K, dtype=float).reshape(-1)
            operator = csr_matrix(self.state_operator)
            rhs1 = capacitance * temperatures - alpha * np.asarray(operator @ temperatures, dtype=float).reshape(-1)
            rhs1 = rhs1 + gamma * h * capacitance * source
            stage1 = self._solve_linear(stage1_matrix, rhs1, stage1_preconditioner, temperatures)
            rhs2 = (
                (1.0 / (gamma * (1.0 - gamma) * h)) * capacitance * stage1
                - ((1.0 - gamma) / (gamma * h)) * capacitance * temperatures
                + capacitance * source
            )
        else:
            operator = csr_matrix(self.state_operator)
            rhs1 = temperatures + alpha * np.asarray(operator @ temperatures, dtype=float).reshape(-1)
            rhs1 = rhs1 + gamma * h * source
            stage1 = self._solve_linear(stage1_matrix, rhs1, stage1_preconditioner, temperatures)
            rhs2 = (
                (1.0 / (gamma * (1.0 - gamma) * h)) * stage1
                - ((1.0 - gamma) / (gamma * h)) * temperatures
                + source
            )
        result = self._solve_linear(stage2_matrix, rhs2, stage2_preconditioner, stage1)
        if self.residual_check_enabled:
            residual = np.asarray(stage2_matrix @ result, dtype=float).reshape(-1) - np.asarray(rhs2, dtype=float).reshape(-1)
            residual_norm = float(np.linalg.norm(residual))
            rhs_norm = float(np.linalg.norm(rhs2))
            self.last_residual_norm = max(float(self.last_residual_norm), residual_norm)
            relative = residual_norm / max(rhs_norm, 1.0e-30)
            self.last_relative_residual_norm = max(float(self.last_relative_residual_norm), float(relative))
        return np.asarray(result, dtype=float).reshape(-1)

    def _stage_matrices_for_h(self, h_s: float) -> tuple[Any, Any | None, Any, Any | None]:
        key = int(round(float(h_s) / max(float(self.dt_s), 1.0e-30) * 1.0e9))
        cached = self._stage_cache.get(key)
        if cached is not None:
            return cached
        gamma = min(0.95, max(0.05, float(self.gamma)))
        h = max(float(h_s), 1.0e-30)
        alpha = 0.5 * gamma * h
        if self.capacitance_J_K is not None:
            capacitance = np.asarray(self.capacitance_J_K, dtype=float).reshape(-1)
            C_diag = diags(capacitance, format="csr")
            operator = csr_matrix(self.state_operator)
            stage1_matrix = (C_diag + alpha * operator).tocsr()
            stage2_scale = (2.0 - gamma) / ((1.0 - gamma) * h)
            stage2_matrix = (stage2_scale * C_diag + operator).tocsr()
        else:
            operator = csr_matrix(self.state_operator)
            identity = eye(operator.shape[0], format="csr")
            stage1_matrix = (identity - alpha * operator).tocsr()
            stage2_scale = (2.0 - gamma) / ((1.0 - gamma) * h)
            stage2_matrix = (stage2_scale * identity - operator).tocsr()
        cached = (
            stage1_matrix,
            _jacobi_preconditioner(stage1_matrix),
            stage2_matrix,
            _jacobi_preconditioner(stage2_matrix),
        )
        self._stage_cache[key] = cached
        return cached

    def _adaptive_substep_count(self, temperatures: np.ndarray, source: np.ndarray) -> int:
        max_substeps = max(1, int(self.adaptive_max_substeps))
        if not bool(self.adaptive_substeps_enabled) or max_substeps <= 1:
            self.last_predicted_delta_K = 0.0
            return 1
        rate = self._thermal_rate(temperatures, source)
        if rate.shape != temperatures.shape or not np.all(np.isfinite(rate)):
            self.last_predicted_delta_K = 0.0
            return 1
        predicted_delta = float(self.dt_s) * float(np.max(np.abs(rate))) if rate.size else 0.0
        self.last_predicted_delta_K = max(0.0, predicted_delta)
        target = max(float(self.adaptive_target_delta_K), 1.0e-12)
        return max(1, min(max_substeps, int(np.ceil(self.last_predicted_delta_K / target))))

    def _thermal_rate(self, temperatures: np.ndarray, source: np.ndarray) -> np.ndarray:
        if self.capacitance_J_K is not None:
            capacitance = np.asarray(self.capacitance_J_K, dtype=float).reshape(-1)
            if capacitance.shape != temperatures.shape:
                return np.zeros_like(temperatures)
            conduction = np.asarray(self.state_operator @ temperatures, dtype=float).reshape(-1)
            return source - conduction / capacitance
        return np.asarray(self.state_operator @ temperatures, dtype=float).reshape(-1) + source

    def _solve_linear(
        self,
        matrix: Any,
        rhs: np.ndarray,
        preconditioner: Any | None,
        x0: np.ndarray,
    ) -> np.ndarray:
        rhs = np.asarray(rhs, dtype=float).reshape(-1)
        if not np.all(np.isfinite(rhs)):
            raise RuntimeError(f"{self.solver} received an invalid right-hand side.")
        guess = np.asarray(x0, dtype=float).reshape(-1)
        if guess.shape != rhs.shape:
            raise RuntimeError(f"{self.solver} received an invalid initial guess.")
        correction_rhs = rhs - np.asarray(matrix @ guess, dtype=float).reshape(-1)
        if not np.all(np.isfinite(correction_rhs)):
            raise RuntimeError(f"{self.solver} received an invalid correction right-hand side.")
        if float(np.linalg.norm(correction_rhs)) <= 0.0:
            return guess.copy()
        iterations = 0

        def _count_iteration(_x: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        solve = cg if self.solver == "cg" else bicgstab
        result, info = solve(
            matrix,
            correction_rhs,
            x0=np.zeros_like(correction_rhs),
            rtol=max(0.0, float(self.rtol)),
            atol=0.0,
            maxiter=max(1, int(self.maxiter)),
            M=preconditioner,
            callback=_count_iteration,
        )
        self.last_info = int(info)
        self.last_iterations += int(iterations)
        if int(info) != 0:
            raise RuntimeError(f"{self.solver} did not converge; info={int(info)}, iterations={int(iterations)}")
        result = np.asarray(result, dtype=float).reshape(-1)
        if result.shape != rhs.shape or not np.all(np.isfinite(result)):
            raise RuntimeError(f"{self.solver} returned an invalid temperature vector.")
        return guess + result


@dataclass
class SimulationState:
    time_s: float
    temperatures_K: np.ndarray
    pid_states: dict[int, tuple[float, float | None, tuple[float, ...]]] = field(default_factory=dict)
    controller_integrators: dict[int, float] = field(default_factory=dict)
    controller_y_prev: dict[int, float] = field(default_factory=dict)
    controller_dTdt_hat_by_sensor: dict[int, float] = field(default_factory=dict)
    controller_error_prev: dict[int, float] = field(default_factory=dict)
    controller_error_history: dict[int, tuple[float, ...]] = field(default_factory=dict)
    controller_last_power_by_heater: dict[int, float] = field(default_factory=dict)
    controller_mode: str = "coarse"


@dataclass
class PreparedSimulationSnapshot:
    z: np.ndarray
    history: list[SimulationState]
    history_index: int
    pid_states: dict[int, tuple[float, float | None, tuple[float, ...]]]
    controller_integrators: dict[int, float]
    controller_y_prev: dict[int, float]
    controller_dTdt_hat_by_sensor: dict[int, float]
    controller_error_prev: dict[int, float]
    controller_error_history: dict[int, tuple[float, ...]]
    controller_mode: str
    controller_weighted_rms_error: float | None
    controller_warnings: list[str]
    controller_last_power_by_heater: dict[int, float]
    controller_allocator_diagnostics: dict[str, Any]


@dataclass
class PreparedSimulation:
    node_ids: np.ndarray
    A_aug: Any
    Phi_aug: np.ndarray | None
    z: np.ndarray
    initial_temperatures_K: np.ndarray
    params: SimulationParameters
    model: ThermalGraphModel | None = None
    inv_C: np.ndarray | None = None
    A: Any | None = None
    base_b: np.ndarray | None = None
    radiation_coeff_W_K4: np.ndarray | None = None
    gpu_stepper: Any | None = None
    sparse_implicit_stepper: SparseImplicitStepper | None = None
    fast_sparse_substeps: int | None = None
    node_index_by_id: dict[int, int] = field(default_factory=dict)
    heater_node_ids: tuple[int, ...] = ()
    cryocooler_node_ids: tuple[int, ...] = ()
    dynamic_heater_inputs: bool = False
    warnings: list[str] = field(default_factory=list)
    controller_integrators: dict[int, float] = field(default_factory=dict)
    controller_y_prev: dict[int, float] = field(default_factory=dict)
    controller_dTdt_hat_by_sensor: dict[int, float] = field(default_factory=dict)
    controller_error_prev: dict[int, float] = field(default_factory=dict)
    controller_error_history: dict[int, tuple[float, ...]] = field(default_factory=dict)
    controller_mode: str = "coarse"
    controller_weighted_rms_error: float | None = None
    controller_warnings: list[str] = field(default_factory=list)
    controller_last_power_by_heater: dict[int, float] = field(default_factory=dict)
    controller_allocator_diagnostics: dict[str, Any] = field(default_factory=dict)
    controller_dynamic_gain_cache: dict[str, Any] = field(default_factory=dict)
    history: list[SimulationState] = field(default_factory=list)
    history_index: int = 0
    last_step_profile_ms: dict[str, float] = field(default_factory=dict, init=False)

    @property
    def time_s(self) -> float:
        if not self.history:
            return 0.0
        return float(self.history[self.history_index].time_s)

    @property
    def temperatures_K(self) -> np.ndarray:
        return np.asarray(self.z[:-1], dtype=float)

    def reset(self) -> None:
        self._reset_pid_states()
        self.reset_controller_integrators()
        self.z = np.concatenate([self.initial_temperatures_K.astype(float), np.array([1.0])])
        self.history = [
            SimulationState(
                0.0,
                self.initial_temperatures_K.copy(),
                self._pid_state_snapshot(),
                dict(self.controller_integrators),
                dict(self.controller_y_prev),
                dict(self.controller_dTdt_hat_by_sensor),
                dict(self.controller_error_prev),
                dict(self.controller_error_history),
                dict(self.controller_last_power_by_heater),
                self.controller_mode,
            )
        ]
        self.history_index = 0
        self._sync_gpu_state()

    def set_uniform_temperature(self, temperature_K: float) -> None:
        uniform = np.full(len(self.node_ids), float(temperature_K), dtype=float)
        self.set_temperatures(uniform)

    def set_temperatures(self, temperatures_K: np.ndarray) -> None:
        temperatures = np.asarray(temperatures_K, dtype=float).reshape(-1)
        if temperatures.shape[0] != len(self.node_ids):
            raise ValueError(
                f"Expected {len(self.node_ids)} temperatures, got {temperatures.shape[0]}."
            )
        self._reset_pid_states()
        self.reset_controller_integrators()
        self.z = np.concatenate([temperatures, np.array([1.0])])
        self.history = [
            SimulationState(
                0.0,
                temperatures.copy(),
                self._pid_state_snapshot(),
                dict(self.controller_integrators),
                dict(self.controller_y_prev),
                dict(self.controller_dTdt_hat_by_sensor),
                dict(self.controller_error_prev),
                dict(self.controller_error_history),
                dict(self.controller_last_power_by_heater),
                self.controller_mode,
            )
        ]
        self.history_index = 0
        self._sync_gpu_state()

    def reset_controller_integrators(self) -> None:
        self.controller_integrators = {}
        self.controller_y_prev = {}
        self.controller_dTdt_hat_by_sensor = {}
        self.controller_error_prev = {}
        self.controller_error_history = {}
        self.controller_mode = "coarse"
        self.controller_weighted_rms_error = None
        self.controller_last_power_by_heater = {}
        self.controller_allocator_diagnostics = {}
        self.controller_dynamic_gain_cache = {}

    def mark_controller_stale(self) -> None:
        self.controller_dTdt_hat_by_sensor = {}
        self.controller_last_power_by_heater = {}
        self.controller_allocator_diagnostics = {}
        self.controller_dynamic_gain_cache = {}

    def snapshot_state(self) -> PreparedSimulationSnapshot:
        return PreparedSimulationSnapshot(
            z=np.asarray(self.z, dtype=float).copy(),
            history=[
                SimulationState(
                    float(state.time_s),
                    np.asarray(state.temperatures_K, dtype=float).copy(),
                    dict(state.pid_states),
                    dict(state.controller_integrators),
                    dict(state.controller_y_prev),
                    dict(state.controller_dTdt_hat_by_sensor),
                    dict(state.controller_error_prev),
                    dict(state.controller_error_history),
                    dict(state.controller_last_power_by_heater),
                    str(state.controller_mode),
                )
                for state in self.history
            ],
            history_index=int(self.history_index),
            pid_states=self._pid_state_snapshot(),
            controller_integrators=dict(self.controller_integrators),
            controller_y_prev=dict(self.controller_y_prev),
            controller_dTdt_hat_by_sensor=dict(self.controller_dTdt_hat_by_sensor),
            controller_error_prev=dict(self.controller_error_prev),
            controller_error_history=dict(self.controller_error_history),
            controller_mode=str(self.controller_mode),
            controller_weighted_rms_error=self.controller_weighted_rms_error,
            controller_warnings=list(self.controller_warnings),
            controller_last_power_by_heater=dict(self.controller_last_power_by_heater),
            controller_allocator_diagnostics=dict(self.controller_allocator_diagnostics),
        )

    def restore_state(self, snapshot: PreparedSimulationSnapshot) -> None:
        self.z = np.asarray(snapshot.z, dtype=float).copy()
        self.history = [
            SimulationState(
                float(state.time_s),
                np.asarray(state.temperatures_K, dtype=float).copy(),
                dict(state.pid_states),
                dict(state.controller_integrators),
                dict(state.controller_y_prev),
                dict(state.controller_dTdt_hat_by_sensor),
                dict(state.controller_error_prev),
                dict(state.controller_error_history),
                dict(state.controller_last_power_by_heater),
                str(state.controller_mode),
            )
            for state in snapshot.history
        ]
        self.history_index = int(snapshot.history_index)
        self._restore_pid_state_snapshot(snapshot.pid_states)
        self.controller_integrators = dict(snapshot.controller_integrators)
        self.controller_y_prev = dict(snapshot.controller_y_prev)
        self.controller_dTdt_hat_by_sensor = dict(snapshot.controller_dTdt_hat_by_sensor)
        self.controller_error_prev = dict(snapshot.controller_error_prev)
        self.controller_error_history = dict(snapshot.controller_error_history)
        self.controller_mode = str(snapshot.controller_mode)
        self.controller_weighted_rms_error = snapshot.controller_weighted_rms_error
        self.controller_warnings = list(snapshot.controller_warnings)
        self.controller_last_power_by_heater = dict(snapshot.controller_last_power_by_heater)
        self.controller_allocator_diagnostics = dict(snapshot.controller_allocator_diagnostics)
        self._sync_gpu_state()

    def step_forward(self) -> SimulationState:
        profile: dict[str, float] = {}
        total_start = time.perf_counter()
        if self.history_index < len(self.history) - 1:
            seek_start = time.perf_counter()
            state = self.seek(self.history_index + 1)
            profile["seek_ms"] = (time.perf_counter() - seek_start) * 1000.0
            profile["total_ms"] = (time.perf_counter() - total_start) * 1000.0
            self.last_step_profile_ms = profile
            return state
        solve_start = time.perf_counter()
        if self.dynamic_heater_inputs:
            self._step_dynamic_heater_inputs(profile)
        elif self.Phi_aug is None:
            zero_start = time.perf_counter()
            zero_power = np.zeros(len(self.node_ids), dtype=float)
            _record_profile_ms(profile, "zero_power_vector_ms", zero_start)
            if (
                not self._advance_with_gpu_power_vector(zero_power, profile)
                and not self._advance_with_sparse_implicit_power_vector(zero_power, profile)
                and not self._advance_with_fast_sparse_power_vector(
                zero_power,
                profile,
                )
            ):
                expm_start = time.perf_counter()
                self.z = np.asarray(
                    expm_multiply(self.A_aug * float(self.params.dt_s), self.z),
                    dtype=float,
                )
                self.z[-1] = 1.0
                _record_profile_ms(profile, "cpu_expm_multiply_ms", expm_start)
        else:
            matvec_start = time.perf_counter()
            self.z = self.Phi_aug @ self.z
            _record_profile_ms(profile, "dense_phi_matvec_ms", matvec_start)
        profile["model_solve_ms"] = (time.perf_counter() - solve_start) * 1000.0
        state_start = time.perf_counter()
        state = SimulationState(
            self.time_s + float(self.params.dt_s),
            self.temperatures_K.copy(),
            self._pid_state_snapshot(),
            dict(self.controller_integrators),
            dict(self.controller_y_prev),
            dict(self.controller_dTdt_hat_by_sensor),
            dict(self.controller_error_prev),
            dict(self.controller_error_history),
            dict(self.controller_last_power_by_heater),
            self.controller_mode,
        )
        profile["state_copy_ms"] = (time.perf_counter() - state_start) * 1000.0
        history_start = time.perf_counter()
        self._append_history_state(state)
        profile["history_append_ms"] = (time.perf_counter() - history_start) * 1000.0
        profile["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        self.last_step_profile_ms = profile
        return state

    def _append_history_state(self, state: SimulationState) -> None:
        self.history.append(state)
        limit = max(0, int(getattr(self.params, "simulation_history_limit", 0)))
        if limit > 0 and len(self.history) > limit:
            overflow = len(self.history) - limit
            del self.history[:overflow]
        self.history_index = len(self.history) - 1

    def step_with_forced_heater_powers(
        self,
        heater_power_by_node: dict[int, float],
        *,
        keep_cryocoolers_active: bool = True,
    ) -> None:
        if self.model is None:
            return
        powers = np.zeros(len(self.node_ids), dtype=float)
        node_index = self.node_index_by_id or {int(node_id): row for row, node_id in enumerate(self.node_ids)}
        for node_id in self.heater_node_ids:
            node = self.model.nodes[int(node_id)]
            if node.is_heater:
                _deposit_heater_command_power(
                    powers,
                    self.model,
                    node_index,
                    int(node_id),
                    max(0.0, float(heater_power_by_node.get(int(node_id), 0.0))),
                )
        for node_id in self.cryocooler_node_ids:
            row = node_index.get(int(node_id))
            if row is None:
                continue
            node = self.model.nodes[int(node_id)]
            if keep_cryocoolers_active and node.has_cryocooler:
                powers[int(row)] -= _cryocooler_power_for_temperature(float(self.temperatures_K[int(row)]), self.params)
        self._advance_with_power_vector(powers)

    def step_backward(self) -> SimulationState:
        if not self.history:
            self.reset()
        if self.history_index <= 0:
            return self.seek(0)
        return self.seek(self.history_index - 1)

    def seek(self, history_index: int) -> SimulationState:
        if not self.history:
            self.reset()
        self.history_index = max(0, min(int(history_index), len(self.history) - 1))
        state = self.history[self.history_index]
        self.z = np.concatenate([state.temperatures_K.copy(), np.array([1.0])])
        self._restore_pid_state_snapshot(state.pid_states)
        self.controller_integrators = dict(state.controller_integrators)
        self.controller_y_prev = dict(state.controller_y_prev)
        self.controller_dTdt_hat_by_sensor = dict(state.controller_dTdt_hat_by_sensor)
        self.controller_error_prev = dict(state.controller_error_prev)
        self.controller_error_history = dict(state.controller_error_history)
        self.controller_last_power_by_heater = dict(state.controller_last_power_by_heater)
        self.controller_mode = state.controller_mode
        self._sync_gpu_state()
        return state

    def _sync_gpu_state(self) -> None:
        if self.gpu_stepper is not None and hasattr(self.gpu_stepper, "set_state"):
            self.gpu_stepper.set_state(self.temperatures_K)

    def _step_dynamic_heater_inputs(self, profile: dict[str, float] | None = None) -> None:
        if self.model is None or self.inv_C is None or self.A is None or self.base_b is None:
            return
        mode_start = time.perf_counter()
        use_mimo = _mimo_controller_is_active(self.model, self.node_ids, self.params)
        _record_profile_ms(profile, "controller_mode_check_ms", mode_start)
        if use_mimo:
            controller_start = time.perf_counter()
            heater_power = self._mimo_controller_power_vector(update_state=True)
            _record_profile_ms(profile, "controller_mimo_ms", controller_start)
        else:
            controller_start = time.perf_counter()
            heater_power = _controlled_heater_power_vector(
                self.model,
                self.node_ids,
                self.temperatures_K,
                float(self.params.dt_s),
                self.params,
                include_heater_inputs=self.params.input_mode == "heater_inputs",
                update_pid_state=True,
                node_index_by_id=self.node_index_by_id,
                heater_node_ids=self.heater_node_ids,
                cryocooler_node_ids=self.cryocooler_node_ids,
            )
            _record_profile_ms(profile, "controller_heater_power_ms", controller_start)
        self._advance_with_power_vector(heater_power, profile)

    def _advance_with_power_vector(self, heater_power: np.ndarray, profile: dict[str, float] | None = None) -> None:
        if self.inv_C is None or self.A is None or self.base_b is None:
            return
        if self._advance_with_gpu_power_vector(heater_power, profile):
            return
        if self._advance_with_sparse_implicit_power_vector(heater_power, profile):
            return
        if self._advance_with_fast_sparse_power_vector(heater_power, profile):
            return
        source_start = time.perf_counter()
        radiation_start = time.perf_counter()
        radiation_source = self._radiation_source_vector()
        _record_profile_ms(profile, "radiation_source_ms", radiation_start)
        b = (
            np.asarray(self.base_b, dtype=float)
            + np.asarray(self.inv_C, dtype=float) * np.asarray(heater_power, dtype=float)
            + radiation_source
        )
        _record_profile_ms(profile, "source_vector_build_ms", source_start)
        if self.Phi_aug is None:
            build_start = time.perf_counter()
            A_aug = bmat(
                [
                    [self.A, csr_matrix(b.reshape(-1, 1))],
                    [csr_matrix((1, len(self.node_ids))), csr_matrix((1, 1))],
                ],
                format="csr",
            )
            _record_profile_ms(profile, "affine_matrix_build_ms", build_start)
            expm_start = time.perf_counter()
            self.z = np.asarray(expm_multiply(A_aug * float(self.params.dt_s), self.z), dtype=float)
            _record_profile_ms(profile, "cpu_expm_multiply_ms", expm_start)
        else:
            build_start = time.perf_counter()
            A_aug = np.zeros((len(self.node_ids) + 1, len(self.node_ids) + 1), dtype=float)
            A_aug[: len(self.node_ids), : len(self.node_ids)] = self.A
            A_aug[: len(self.node_ids), len(self.node_ids)] = b
            _record_profile_ms(profile, "dense_affine_matrix_build_ms", build_start)
            expm_start = time.perf_counter()
            self.z = expm(A_aug * float(self.params.dt_s)) @ self.z
            _record_profile_ms(profile, "dense_expm_matvec_ms", expm_start)
        self.z[-1] = 1.0

    def _advance_with_gpu_power_vector(
        self,
        heater_power: np.ndarray,
        profile: dict[str, float] | None = None,
    ) -> bool:
        if self.gpu_stepper is None:
            return False
        try:
            gpu_start = time.perf_counter()
            temperatures = self.gpu_stepper.step(self.temperatures_K, heater_power)
            _record_profile_ms(profile, "gpu_step_ms", gpu_start)
        except Exception as exc:
            self.warnings.append(f"GPU simulation step failed; falling back to CPU stepping: {exc}")
            self.gpu_stepper = None
            return False
        sync_start = time.perf_counter()
        self.z = np.concatenate([np.asarray(temperatures, dtype=float), np.array([1.0])])
        _record_profile_ms(profile, "state_vector_update_ms", sync_start)
        return True

    def _advance_with_sparse_implicit_power_vector(
        self,
        heater_power: np.ndarray,
        profile: dict[str, float] | None = None,
    ) -> bool:
        if (
            self.sparse_implicit_stepper is None
            or self.inv_C is None
            or self.base_b is None
        ):
            return False
        try:
            source_start = time.perf_counter()
            radiation_start = time.perf_counter()
            radiation_source = self._radiation_source_vector()
            _record_profile_ms(profile, "radiation_source_ms", radiation_start)
            temperatures = np.asarray(self.temperatures_K, dtype=float).reshape(-1)
            powers = np.asarray(heater_power, dtype=float).reshape(-1)
            if powers.shape != temperatures.shape:
                raise ValueError(f"Heater power vector length {powers.shape} does not match temperatures {temperatures.shape}.")
            source = (
                np.asarray(self.base_b, dtype=float).reshape(-1)
                + np.asarray(self.inv_C, dtype=float).reshape(-1) * powers
                + radiation_source
            )
            _record_profile_ms(profile, "source_vector_build_ms", source_start)
            step_start = time.perf_counter()
            temperatures_next = self.sparse_implicit_stepper.step(temperatures, source)
            _record_profile_ms(profile, "cpu_sparse_implicit_step_ms", step_start)
            if profile is not None:
                profile["cpu_sparse_implicit_iterations"] = float(self.sparse_implicit_stepper.last_iterations)
                profile["cpu_sparse_implicit_substeps"] = float(self.sparse_implicit_stepper.last_substeps)
                profile["cpu_sparse_implicit_residual_norm"] = float(self.sparse_implicit_stepper.last_residual_norm)
                profile["cpu_sparse_implicit_relative_residual"] = float(
                    self.sparse_implicit_stepper.last_relative_residual_norm
                )
                profile["cpu_sparse_implicit_predicted_delta_K"] = float(
                    self.sparse_implicit_stepper.last_predicted_delta_K
                )
            sync_start = time.perf_counter()
            self.z = np.concatenate([temperatures_next, np.array([1.0])])
            _record_profile_ms(profile, "state_vector_update_ms", sync_start)
        except Exception as exc:
            self.warnings.append(f"Sparse implicit CPU stepping failed; falling back to explicit/exponential stepping: {exc}")
            self.sparse_implicit_stepper = None
            return False
        return True

    def _advance_with_fast_sparse_power_vector(
        self,
        heater_power: np.ndarray,
        profile: dict[str, float] | None = None,
    ) -> bool:
        if (
            self.fast_sparse_substeps is None
            or self.A is None
            or self.inv_C is None
            or self.base_b is None
            or not issparse(self.A)
        ):
            return False
        try:
            step_start = time.perf_counter()
            temperatures = np.asarray(self.temperatures_K, dtype=float).reshape(-1).copy()
            powers = np.asarray(heater_power, dtype=float).reshape(-1)
            if powers.shape != temperatures.shape:
                raise ValueError(f"Heater power vector length {powers.shape} does not match temperatures {temperatures.shape}.")
            source = np.asarray(self.base_b, dtype=float).reshape(-1) + np.asarray(self.inv_C, dtype=float).reshape(-1) * powers
            inv_C = np.asarray(self.inv_C, dtype=float).reshape(-1)
            radiation_coeff = (
                np.asarray(self.radiation_coeff_W_K4, dtype=float).reshape(-1)
                if self.radiation_coeff_W_K4 is not None
                else np.zeros_like(temperatures)
            )
            use_radiation = bool(self.params.use_ambient_radiation and np.any(radiation_coeff > 0.0))
            ambient_K4 = float(self.params.T_env_K) ** 4
            substeps = max(1, int(self.fast_sparse_substeps))
            h = float(self.params.dt_s) / substeps
            for _ in range(substeps):
                rhs = np.asarray(self.A @ temperatures, dtype=float).reshape(-1) + source
                if use_radiation:
                    rhs = rhs + inv_C * radiation_coeff * (ambient_K4 - temperatures**4)
                temperatures = temperatures + h * rhs
            self.z = np.concatenate([temperatures, np.array([1.0])])
            _record_profile_ms(profile, "cpu_fast_sparse_step_ms", step_start)
        except Exception as exc:
            self.warnings.append(f"Fast sparse CPU stepping failed; falling back to CPU expm_multiply: {exc}")
            self.fast_sparse_substeps = None
            return False
        return True

    def _radiation_source_vector(self, temperatures_K: np.ndarray | None = None) -> np.ndarray:
        if (
            self.radiation_coeff_W_K4 is None
            or self.inv_C is None
            or not self.params.use_ambient_radiation
        ):
            return np.zeros(len(self.node_ids), dtype=float)
        coeff = np.asarray(self.radiation_coeff_W_K4, dtype=float).reshape(-1)
        if not np.any(coeff > 0.0):
            return np.zeros(len(self.node_ids), dtype=float)
        temperatures = (
            np.asarray(self.temperatures_K, dtype=float).reshape(-1)
            if temperatures_K is None
            else np.asarray(temperatures_K, dtype=float).reshape(-1)
        )
        ambient = float(self.params.T_env_K)
        radiation_power = coeff * (ambient**4 - temperatures**4)
        return np.asarray(self.inv_C, dtype=float) * radiation_power

    def _thermal_rhs(self, temperatures_K: np.ndarray, heater_power: np.ndarray) -> np.ndarray:
        if self.A is None or self.base_b is None or self.inv_C is None:
            return np.zeros(len(self.node_ids), dtype=float)
        temperatures = np.asarray(temperatures_K, dtype=float).reshape(-1)
        powers = np.asarray(heater_power, dtype=float).reshape(-1)
        if powers.shape != temperatures.shape:
            raise ValueError(f"Heater power vector length {powers.shape} does not match temperatures {temperatures.shape}.")
        return np.asarray(self.A @ temperatures, dtype=float).reshape(-1) + np.asarray(self.base_b, dtype=float) + (
            np.asarray(self.inv_C, dtype=float) * powers
        ) + self._radiation_source_vector(temperatures)

    def _reset_pid_states(self) -> None:
        return

    def _pid_state_snapshot(self) -> dict[int, tuple[float, float | None, tuple[float, ...]]]:
        return {}

    def _restore_pid_state_snapshot(self, snapshot: dict[int, tuple[float, float | None] | tuple[float, float | None, tuple[float, ...]]]) -> None:
        return

    def heater_power_by_node(self) -> dict[int, float]:
        if (
            self.model is None
            or not self.dynamic_heater_inputs
        ):
            return {
                int(node_id): 0.0
                for node_id in self.node_ids
                if self.model is not None
                and (
                    self.model.nodes[int(node_id)].is_heater
                    or self.model.nodes[int(node_id)].has_cryocooler
                )
            }
        powers = _controlled_heater_power_vector(
            self.model,
            self.node_ids,
            self.temperatures_K,
            float(self.params.dt_s),
            self.params,
            include_heater_inputs=self.params.input_mode == "heater_inputs",
            update_pid_state=False,
            node_index_by_id=self.node_index_by_id,
            heater_node_ids=self.heater_node_ids,
            cryocooler_node_ids=self.cryocooler_node_ids,
        ) if not _mimo_controller_is_active(self.model, self.node_ids, self.params) else self._mimo_controller_power_vector(update_state=False)
        node_index = self.node_index_by_id or {int(node_id): row for row, node_id in enumerate(self.node_ids)}
        role_node_ids = sorted(set(self.heater_node_ids).union(self.cryocooler_node_ids))
        result: dict[int, float] = {}
        for node_id in role_node_ids:
            row = node_index.get(int(node_id))
            if row is not None:
                result[int(node_id)] = float(powers[int(row)])
        return result

    def cryocooler_power_by_node(self) -> dict[int, float]:
        if (
            self.model is None
            or not self.dynamic_heater_inputs
        ):
            return {
                int(node_id): 0.0
                for node_id in self.cryocooler_node_ids
            }
        powers = _cryocooler_power_vector(
            self.model,
            self.node_ids,
            self.temperatures_K,
            self.params,
            node_index_by_id=self.node_index_by_id,
            cryocooler_node_ids=self.cryocooler_node_ids,
        )
        node_index = self.node_index_by_id or {int(node_id): row for row, node_id in enumerate(self.node_ids)}
        result: dict[int, float] = {}
        for node_id in self.cryocooler_node_ids:
            row = node_index.get(int(node_id))
            if row is not None:
                result[int(node_id)] = float(powers[int(row)])
        return result

    def heater_actuator_power_by_node(self, *, disable_mimo_controller: bool = False) -> dict[int, float]:
        if self.model is None:
            return {}
        if _mimo_controller_is_active(self.model, self.node_ids, self.params) and not disable_mimo_controller:
            self._mimo_controller_power_vector(update_state=False)
            diagnostics = self.controller_allocator_diagnostics or {}
            heater_ids = [int(value) for value in diagnostics.get("heater_ids", []) or []]
            commands = [float(value) for value in diagnostics.get("heater_commands_W", []) or []]
            return {heater_id: command for heater_id, command in zip(heater_ids, commands)}
        else:
            return _controlled_heater_command_by_node(
                self.model,
                self.node_ids,
                self.temperatures_K,
                float(self.params.dt_s),
                self.params,
                include_heater_inputs=self.params.input_mode == "heater_inputs",
                excluded_modes={"mimo"} if disable_mimo_controller else None,
                heater_node_ids=self.heater_node_ids,
            )

    def _mimo_controller_power_vector(self, update_state: bool) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(self.node_ids), dtype=float)
        powers = _controlled_heater_power_vector(
            self.model,
            self.node_ids,
            self.temperatures_K,
            float(self.params.dt_s),
            self.params,
            include_heater_inputs=True,
            update_pid_state=update_state,
            excluded_modes={"mimo"},
            node_index_by_id=self.node_index_by_id,
            heater_node_ids=self.heater_node_ids,
            cryocooler_node_ids=self.cryocooler_node_ids,
        )
        enabled_heater_ids = _enabled_node_id_set(self.params.enabled_heater_node_ids)
        enabled_sensor_ids = _enabled_node_id_set(self.params.enabled_sensor_node_ids)
        pair_warnings = refresh_heater_power_deposition_nodes(self.model)
        pair_warnings.extend(refresh_sensor_connected_nodes(self.model))
        active_sensor_ids: set[int] = set()
        heater_ids: list[int] = []
        for node_id in self.heater_node_ids:
            heater_id = int(node_id)
            heater = self.model.nodes[heater_id]
            if (
                not heater.is_heater
                or not _node_id_enabled(enabled_heater_ids, heater_id)
                or not bool(getattr(heater, "heater_valid", True))
                or not getattr(heater, "power_deposition_node_ids", [])
            ):
                continue
            sensor_id = getattr(heater, "assigned_sensor_id", None)
            if sensor_id is None and heater.is_sensor and str(getattr(getattr(heater, "heater_control", None), "mode", "")) == "mimo":
                sensor_id = heater_id
            if sensor_id is None:
                continue
            sensor_id = int(sensor_id)
            sensor = self.model.nodes.get(sensor_id)
            if sensor is None or not _node_is_mimo_sensor(sensor):
                continue
            if not _node_id_enabled(enabled_sensor_ids, sensor_id):
                continue
            active_sensor_ids.add(sensor_id)
            heater_ids.append(heater_id)
        sensor_ids = sorted(active_sensor_ids)
        if not sensor_ids or not heater_ids:
            self.controller_warnings = pair_warnings + [
                "MIMO controller enabled, but at least one paired valid MIMO sensor and heater are required."
            ]
            self.controller_allocator_diagnostics = {
                "active_sensor_count": len(sensor_ids),
                "active_heater_count": len(heater_ids),
                "rate_command_norm": 0.0,
                "heater_command_norm": 0.0,
                "measured_drift_dTdt_norm": 0.0,
                "predicted_dTdt_residual_norm": 0.0,
                "allocation_residual_norm": 0.0,
                "bounds_active": False,
                "solver_success": False,
                "solver_message": "empty active MIMO set",
            }
            if update_state:
                self.controller_last_power_by_heater = {heater_id: 0.0 for heater_id in heater_ids}
            return powers

        node_index = self.node_index_by_id or {int(node_id): row for row, node_id in enumerate(self.node_ids)}
        readouts = [
            sensor_readout_temperature_K(self.model, node_index, self.temperatures_K, sensor_id)
            for sensor_id in sensor_ids
        ]
        valid_pairs = [
            (sensor_id, readout)
            for sensor_id, readout in zip(sensor_ids, readouts)
            if np.isfinite(float(readout))
        ]
        if len(valid_pairs) != len(sensor_ids):
            pair_warnings.append("One or more MIMO sensors have invalid averaged readouts and were excluded.")
        sensor_ids = [int(sensor_id) for sensor_id, _readout in valid_pairs]
        valid_sensor_id_set = set(sensor_ids)
        filtered_heater_ids: list[int] = []
        for heater_id in heater_ids:
            heater = self.model.nodes[int(heater_id)]
            assigned_sensor_id = getattr(heater, "assigned_sensor_id", None)
            if assigned_sensor_id is not None and int(assigned_sensor_id) in valid_sensor_id_set:
                filtered_heater_ids.append(int(heater_id))
                continue
            if (
                int(heater_id) in valid_sensor_id_set
                and heater.is_sensor
                and str(getattr(getattr(heater, "heater_control", None), "mode", "")) == "mimo"
            ):
                filtered_heater_ids.append(int(heater_id))
        heater_ids = filtered_heater_ids
        if not sensor_ids or not heater_ids:
            self.controller_warnings = pair_warnings + ["No valid paired MIMO sensor readouts are available."]
            if update_state:
                self.controller_last_power_by_heater = {}
            return powers
        y = np.array([float(readout) for _sensor_id, readout in valid_pairs], dtype=float)
        estimates = []
        for sensor_id, measured in zip(sensor_ids, y):
            node = self.model.nodes[sensor_id]
            settling_time = max(0.0, float(getattr(node, "sensor_settling_time_s", 0.0)))
            tau = settling_time / 5.0 if settling_time > 0.0 else 0.0
            previous = self.controller_y_prev.get(sensor_id)
            if previous is None or tau <= 0.0:
                estimates.append(measured)
                continue
            alpha = float(np.exp(-max(float(self.params.dt_s), 0.0) / max(tau, 1.0e-12)))
            denominator = 1.0 - alpha
            if abs(denominator) <= 1.0e-9:
                estimates.append(measured)
            else:
                estimates.append((measured - alpha * float(previous)) / denominator)
        T_hat = np.array(estimates, dtype=float)
        setpoints = np.array(
            [float(getattr(self.model.nodes[sensor_id], "controller_setpoint_K", 293.15)) for sensor_id in sensor_ids],
            dtype=float,
        )
        errors = setpoints - T_hat
        weights = np.array(
            [
                _controller_sensor_weight(self.model.nodes[sensor_id])
                for sensor_id in sensor_ids
            ],
            dtype=float,
        )
        rms = weighted_rms_error(errors, weights)
        self.controller_weighted_rms_error = rms
        mode_changed = self._update_controller_mode(rms) if update_state else False

        dt = max(float(self.params.dt_s), 1.0e-12)
        lambda_orders = np.array(
            [
                max(0.0, float(getattr(self.model.nodes[sensor_id], "controller_lambda_order", 1.0)))
                for sensor_id in sensor_ids
            ],
            dtype=float,
        )
        mu_orders = np.array(
            [
                max(0.0, float(getattr(self.model.nodes[sensor_id], "controller_mu_order", 1.0)))
                for sensor_id in sensor_ids
            ],
            dtype=float,
        )
        previous_error_history = {} if mode_changed else self.controller_error_history
        candidate_error_history = {
            int(sensor_id): tuple(
                [float(value) for value in previous_error_history.get(int(sensor_id), ())]
                + [float(error)]
            )
            for sensor_id, error in zip(sensor_ids, errors)
        }
        eta = np.array(
            [
                _fractional_integral(
                    candidate_error_history[int(sensor_id)],
                    dt,
                    float(lambda_order),
                )
                for sensor_id, lambda_order in zip(sensor_ids, lambda_orders)
            ],
            dtype=float,
        )
        integral_abs_max = max(0.0, float(getattr(self.params, "mimo_integral_abs_max", 1.0e6)))
        if integral_abs_max > 0.0:
            eta = np.clip(eta, -integral_abs_max, integral_abs_max)
        error_derivative = np.array(
            [
                _fractional_derivative(
                    candidate_error_history[int(sensor_id)],
                    dt,
                    float(mu_order),
                    zero_initial_integer_order=True,
                )
                for sensor_id, mu_order in zip(sensor_ids, mu_orders)
            ],
            dtype=float,
        )
        if self.controller_mode == "hold":
            kp_key = "controller_kp_hold"
            ki_key = "controller_ki_hold"
            kd_key = "controller_kd_hold"
        else:
            kp_key = "controller_kp_coarse"
            ki_key = "controller_ki_coarse"
            kd_key = "controller_kd_coarse"
        Kp = np.array(
            [max(0.0, float(getattr(self.model.nodes[sensor_id], kp_key, 0.0))) for sensor_id in sensor_ids],
            dtype=float,
        )
        Ki = np.array(
            [max(0.0, float(getattr(self.model.nodes[sensor_id], ki_key, 0.0))) for sensor_id in sensor_ids],
            dtype=float,
        )
        Kd = np.array(
            [max(0.0, float(getattr(self.model.nodes[sensor_id], kd_key, 0.0))) for sensor_id in sensor_ids],
            dtype=float,
        )
        v_cmd = Kp * errors + Ki * eta + Kd * error_derivative
        v_abs_max = max(0.0, float(getattr(self.params, "mimo_v_cmd_abs_max_K_per_s", 0.25)))
        if v_abs_max > 0.0:
            v_cmd = np.clip(v_cmd, -v_abs_max, v_abs_max)

        maxima = np.array(
            [_controller_heater_max_power(self.model.nodes[heater_id], self.params) for heater_id in heater_ids],
            dtype=float,
        )
        u_prev = np.array(
            [float(self.controller_last_power_by_heater.get(int(heater_id), 0.0)) for heater_id in heater_ids],
            dtype=float,
        )
        slew_rate = max(0.0, float(getattr(self.params, "mimo_heater_slew_rate_W_per_s", 0.0)))
        max_delta_power = np.full(len(heater_ids), slew_rate * dt, dtype=float) if slew_rate > 0.0 else None
        raw_dTdt, dTdt_hat = self._mimo_sensor_drift_estimate(sensor_ids, y, dt)
        B_s = self._mimo_dynamic_gain_matrix(
            sensor_ids,
            heater_ids,
            node_index,
        )
        allocation = allocate_thermal_rate_qp(
            B_s,
            dTdt_hat,
            v_cmd,
            weights,
            maxima,
            u_prev,
            float(getattr(self.params, "mimo_lambda_u", 1.0e-3)),
            float(getattr(self.params, "mimo_rho_du", 0.0)),
            max_delta_power,
        )
        u = np.asarray(allocation.u, dtype=float).reshape(-1)
        u = np.clip(np.where(np.isfinite(u), u, 0.0), 0.0, maxima)
        for heater_id, command in zip(heater_ids, u):
            _deposit_heater_command_power(
                powers,
                self.model,
                node_index,
                int(heater_id),
                float(command),
            )
        heater_delta_dTdt = B_s @ (u - u_prev)
        predicted_dTdt = dTdt_hat + heater_delta_dTdt
        residual = predicted_dTdt - v_cmd
        gain_warnings = list((self.controller_dynamic_gain_cache or {}).get("warnings", ()))
        self.controller_warnings = pair_warnings + gain_warnings + list(allocation.warnings)
        self.controller_allocator_diagnostics = {
            "active_sensor_count": len(sensor_ids),
            "active_heater_count": len(heater_ids),
            "sensor_ids": [int(value) for value in sensor_ids],
            "heater_ids": [int(value) for value in heater_ids],
            "sensor_connected_node_ids": {
                str(sensor_id): [int(value) for value in getattr(self.model.nodes[int(sensor_id)], "readout_node_ids", []) or getattr(self.model.nodes[int(sensor_id)], "sensor_connected_node_ids", [])]
                for sensor_id in sensor_ids
            },
            "heater_power_deposition_node_ids": {
                str(heater_id): [int(value) for value in getattr(self.model.nodes[int(heater_id)], "power_deposition_node_ids", [])]
                for heater_id in heater_ids
            },
            "rate_command_norm": float(np.linalg.norm(v_cmd)),
            "heater_command_norm": float(np.linalg.norm(u)),
            "measured_drift_dTdt_norm": float(np.linalg.norm(dTdt_hat)),
            "predicted_dTdt_norm": float(np.linalg.norm(predicted_dTdt)),
            "predicted_dTdt_residual_norm": float(np.linalg.norm(residual)),
            "allocation_residual_norm": float(np.linalg.norm(residual)),
            "target_residual_norm": float(allocation.residual_norm),
            "bounds_active": bool(allocation.bounds_active),
            "solver_success": bool(allocation.solver_success),
            "solver_message": str(allocation.solver_message),
            "lambda_u": float(getattr(self.params, "mimo_lambda_u", 1.0e-3)),
            "rho_du": float(getattr(self.params, "mimo_rho_du", 0.0)),
            "slew_rate_limit_W_per_s": float(slew_rate),
            "slew_delta_limit_W": float(slew_rate * dt) if slew_rate > 0.0 else 0.0,
            "v_cmd_min_K_per_s": float(np.min(v_cmd)) if v_cmd.size else 0.0,
            "v_cmd_max_K_per_s": float(np.max(v_cmd)) if v_cmd.size else 0.0,
            "raw_dTdt_s": [float(value) for value in raw_dTdt],
            "filtered_dTdt_hat_s": [float(value) for value in dTdt_hat],
            "B_s": [[float(value) for value in row] for row in B_s],
            "average_inverse_C_s": list((self.controller_dynamic_gain_cache or {}).get("average_inverse_C_s", [])),
            "B_s_delta_u_dTdt_s": [float(value) for value in heater_delta_dTdt],
            "v_cmd_s": [float(value) for value in v_cmd],
            "predicted_dTdt_s": [float(value) for value in predicted_dTdt],
            "achieved_predicted_dTdt_s": [float(value) for value in predicted_dTdt],
            "residual_s": [float(value) for value in residual],
            "heater_commands_W": [float(value) for value in u],
            "u_prev_W": [float(value) for value in u_prev],
            "heater_at_lower_bound": [bool(value <= 1.0e-9) for value in u],
            "heater_at_upper_bound": [bool(value >= max(max_power - 1.0e-9, 0.0)) for value, max_power in zip(u, maxima)],
        }
        if update_state:
            committed_error_history = candidate_error_history
            if bool(getattr(self.params, "mimo_freeze_integral_when_saturated", True)):
                committed_error_history = {}
                for row, sensor_id in enumerate(sensor_ids):
                    relevant = np.abs(B_s[row, :]) > 1.0e-12
                    freeze = False
                    if np.any(relevant):
                        if errors[row] < 0.0 and bool(np.all(u[relevant] <= 1.0e-9)):
                            freeze = True
                        elif errors[row] > 0.0 and bool(np.all(u[relevant] >= np.maximum(maxima[relevant] - 1.0e-9, 0.0))):
                            freeze = True
                    committed_error_history[int(sensor_id)] = (
                        tuple(float(value) for value in previous_error_history.get(int(sensor_id), ()))
                        if freeze
                        else candidate_error_history[int(sensor_id)]
                    )
            committed_integrators = np.array(
                [
                    _fractional_integral(
                        committed_error_history[int(sensor_id)],
                        dt,
                        float(lambda_order),
                    )
                    for sensor_id, lambda_order in zip(sensor_ids, lambda_orders)
                ],
                dtype=float,
            )
            if integral_abs_max > 0.0:
                committed_integrators = np.clip(committed_integrators, -integral_abs_max, integral_abs_max)
            self.controller_last_power_by_heater = {
                int(heater_id): float(command)
                for heater_id, command in zip(heater_ids, u)
            }
            self.controller_integrators = {
                int(sensor_id): float(value)
                for sensor_id, value in zip(sensor_ids, committed_integrators)
            }
            self.controller_y_prev = {
                int(sensor_id): float(value)
                for sensor_id, value in zip(sensor_ids, y)
            }
            self.controller_dTdt_hat_by_sensor = {
                int(sensor_id): float(value)
                for sensor_id, value in zip(sensor_ids, dTdt_hat)
            }
            self.controller_error_prev = {
                int(sensor_id): float(value)
                for sensor_id, value in zip(sensor_ids, errors)
            }
            self.controller_error_history = committed_error_history
        return powers

    def _mimo_sensor_drift_estimate(
        self,
        sensor_ids: list[int],
        sensor_temperatures_K: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        temperatures = np.asarray(sensor_temperatures_K, dtype=float).reshape(-1)
        dt_floor = max(0.0, float(getattr(self.params, "derivative_dt_floor_s", 1.0e-9)))
        tau = max(0.0, float(getattr(self.params, "drift_lpf_tau_s", 0.0)))
        use_update = bool(np.isfinite(dt) and dt > dt_floor)
        raw = np.zeros(len(sensor_ids), dtype=float)
        filtered = np.zeros(len(sensor_ids), dtype=float)
        alpha = float(dt / (tau + dt)) if use_update else 0.0
        alpha = min(1.0, max(0.0, alpha)) if np.isfinite(alpha) else 0.0
        for index, (sensor_id, temperature) in enumerate(zip(sensor_ids, temperatures)):
            previous_hat = float(self.controller_dTdt_hat_by_sensor.get(int(sensor_id), 0.0))
            if not np.isfinite(previous_hat):
                previous_hat = 0.0
            previous_temperature = self.controller_y_prev.get(int(sensor_id))
            if not use_update or previous_temperature is None:
                raw_value = previous_hat
            else:
                raw_value = (float(temperature) - float(previous_temperature)) / float(dt)
            if not np.isfinite(raw_value):
                raw_value = previous_hat
            raw[index] = float(raw_value)
            filtered[index] = (1.0 - alpha) * previous_hat + alpha * float(raw_value)
        return raw, filtered

    def _mimo_dynamic_gain_matrix(
        self,
        sensor_ids: list[int],
        heater_ids: list[int],
        node_index: dict[int, int],
    ) -> np.ndarray:
        inv_C = np.asarray(self.inv_C, dtype=float).reshape(-1) if self.inv_C is not None else np.zeros(len(self.node_ids))
        connected_key: tuple[tuple[tuple[int, float], ...], ...]
        if self.model is None:
            connected_key = tuple()
        else:
            connected_rows: list[tuple[tuple[int, float], ...]] = []
            for sensor_id in sensor_ids:
                sensor = self.model.nodes[int(sensor_id)]
                readout_ids = (
                    getattr(sensor, "readout_node_ids", [])
                    or getattr(sensor, "sensor_connected_node_ids", [])
                    or []
                )
                rows_for_sensor: list[tuple[int, float]] = []
                for node_id in readout_ids:
                    node_id = int(node_id)
                    row = node_index.get(node_id)
                    if row is None:
                        continue
                    rows_for_sensor.append((node_id, round(float(inv_C[int(row)]), 12)))
                connected_rows.append(tuple(rows_for_sensor))
            connected_key = tuple(connected_rows)
        key = (
            tuple(int(value) for value in sensor_ids),
            tuple(int(value) for value in heater_ids),
            connected_key,
            float(getattr(self.params, "heater_sensor_pair_alpha", 1.0)),
        )
        cache = self.controller_dynamic_gain_cache
        if (
            isinstance(cache, dict)
            and cache.get("key") == key
            and "B_dyn" in cache
        ):
            return np.asarray(cache["B_dyn"], dtype=float).copy()
        B_dyn = np.zeros((len(sensor_ids), len(heater_ids)), dtype=float)
        heater_col_by_id = {int(heater_id): col for col, heater_id in enumerate(heater_ids)}
        warnings: list[str] = []
        average_inverse_C_s: list[float] = []
        alpha = max(0.0, float(getattr(self.params, "heater_sensor_pair_alpha", 1.0)))
        for sensor_index, sensor_id in enumerate(sensor_ids):
            if self.model is None:
                average_inverse_C_s.append(0.0)
                continue
            sensor = self.model.nodes[int(sensor_id)]
            average_inverse_C, valid_node_ids = average_inverse_capacitance_for_sensor(
                self.model,
                node_index,
                inv_C,
                int(sensor_id),
            )
            average_inverse_C_s.append(float(average_inverse_C))
            paired_heater_cols = [
                col
                for heater_id, col in heater_col_by_id.items()
                if int(getattr(self.model.nodes[int(heater_id)], "assigned_sensor_id", -1) or -1) == int(sensor_id)
                or (
                    int(heater_id) == int(sensor_id)
                    and sensor.is_heater
                    and str(getattr(getattr(sensor, "heater_control", None), "mode", "")) == "mimo"
                )
            ]
            if not paired_heater_cols:
                warnings.append(f"MIMO sensor {int(sensor_id)} has no active paired heater; B_s row set to zero.")
                continue
            if np.isfinite(average_inverse_C) and average_inverse_C > 0.0:
                for heater_col in paired_heater_cols:
                    B_dyn[sensor_index, heater_col] = alpha * float(average_inverse_C)
            else:
                warnings.append(
                    f"MIMO sensor {int(sensor_id)} has no valid connected-node capacitance; B_s row set to zero."
                )
            if not valid_node_ids:
                warnings.append(f"MIMO sensor {int(sensor_id)} has no valid connected nodes for average inverse C.")
        for sensor_index, sensor_id in enumerate(sensor_ids):
            sensor = self.model.nodes[int(sensor_id)] if self.model is not None else None
            if sensor is not None and _controller_sensor_weight(sensor) > 0.0 and not np.any(np.abs(B_dyn[sensor_index, :]) > 0.0):
                warnings.append(f"MIMO sensor {int(sensor_id)} has nonzero control weight but B_s row is all zero.")
        for heater_index, heater_id in enumerate(heater_ids):
            if not np.any(np.abs(B_dyn[:, heater_index]) > 0.0):
                warnings.append(f"MIMO heater {int(heater_id)} is active but B_s column is all zero.")
        self.controller_dynamic_gain_cache = {
            "key": key,
            "B_dyn": B_dyn.copy(),
            "warnings": tuple(warnings),
            "average_inverse_C_s": [float(value) for value in average_inverse_C_s],
        }
        return B_dyn

    def _update_controller_mode(self, weighted_rms: float) -> bool:
        previous_mode = self.controller_mode
        if self.controller_mode not in {"coarse", "hold"}:
            self.controller_mode = "coarse"
        if self.controller_mode == "coarse" and weighted_rms <= float(self.params.mimo_hold_threshold_K):
            self.controller_mode = "hold"
        elif self.controller_mode == "hold" and weighted_rms >= float(self.params.mimo_coarse_threshold_K):
            self.controller_mode = "coarse"
        return self.controller_mode != previous_mode

def prepare_simulation(
    model: ThermalGraphModel,
    matrices: dict[str, np.ndarray],
    params: SimulationParameters,
) -> PreparedSimulation:
    node_ids = np.asarray(matrices.get("node_ids", model.ordered_node_ids()), dtype=int)
    node_index_by_id = {int(node_id): row for row, node_id in enumerate(node_ids)}
    heater_node_ids = tuple(
        int(node_id)
        for node_id in sorted((int(value) for value in node_ids))
        if bool(getattr(model.nodes[int(node_id)], "is_heater", False))
    )
    cryocooler_node_ids = tuple(
        int(node_id)
        for node_id in sorted((int(value) for value in node_ids))
        if bool(getattr(model.nodes[int(node_id)], "has_cryocooler", False))
    )
    n = len(node_ids)
    C = np.asarray(matrices.get("C", [model.nodes[int(node_id)].C_J_K for node_id in node_ids]), dtype=float).reshape(-1)
    raw_L = matrices.get("L")
    if raw_L is None:
        raise ValueError("Cannot initialize simulation without L matrix.")
    L = raw_L if issparse(raw_L) else np.asarray(raw_L, dtype=float)
    G_rad = _radiation_vector(matrices, model, node_ids)
    initial = np.asarray(
        [model.nodes[int(node_id)].initial_temperature_K for node_id in node_ids],
        dtype=float,
    ).reshape(-1)
    warnings = validate_simulation_inputs(model, node_ids, C, L, G_rad, initial, params)
    if n > int(params.browser_simulation_size_warning):
        warnings.append(
            f"Graph has {n} nodes; dense matrix exponential playback may be slow above "
            f"{params.browser_simulation_size_warning} nodes."
        )
    if np.any(C <= 0.0):
        raise ValueError("Cannot initialize simulation with nonpositive thermal capacitance.")
    if L.shape != (n, n):
        raise ValueError(f"L shape {L.shape} does not match node count {n}.")

    radiation_coeff = _radiation_coefficient_vector(matrices, model, node_ids, G_rad, params)
    inv_C = 1.0 / C
    b = np.zeros(n, dtype=float)
    pairing_warnings = refresh_heater_power_deposition_nodes(model)
    pairing_warnings.extend(refresh_sensor_connected_nodes(model))
    warnings.extend(pairing_warnings)
    has_cryocooler = any(model.nodes[int(node_id)].has_cryocooler for node_id in node_ids)
    has_mimo_controller = _mimo_controller_is_active(model, node_ids, params)
    has_nonlinear_radiation = bool(params.use_ambient_radiation and np.any(radiation_coeff > 0.0))
    dynamic_heater_inputs = (
        params.input_mode == "heater_inputs"
        or has_cryocooler
        or has_mimo_controller
        or has_nonlinear_radiation
    )
    if params.input_mode == "heater_inputs":
        if not any(
            model.nodes[int(node_id)].is_heater
            or model.nodes[int(node_id)].has_cryocooler
            for node_id in node_ids
        ):
            warnings.append(
                "Input mode requested heater inputs, but no heater or cryocooler powers are defined; using zero input."
            )
    elif params.input_mode != "zero":
        warnings.append(f"Unknown input mode {params.input_mode!r}; using zero input.")
    if params.input_mode == "heater_inputs" and any(
        str(getattr(model.nodes[int(node_id)], "sensor_control_mode", "manual")) == "mimo"
        for node_id in node_ids
        if model.nodes[int(node_id)].is_sensor
    ) and not has_mimo_controller:
        warnings.append("MIMO sensor control is selected, but no valid paired MIMO sensor/heater set is available.")

    sparse_stepper = n > 512 or issparse(L)
    if sparse_stepper:
        L_sparse = csr_matrix(L)
        A = -(diags(inv_C, format="csr") @ L_sparse)
        if dynamic_heater_inputs:
            A_aug = A
        else:
            A_aug = bmat(
                [
                    [A, csr_matrix(b.reshape(-1, 1))],
                    [csr_matrix((1, n)), csr_matrix((1, 1))],
                ],
                format="csr",
            )
        Phi_aug = None
    else:
        if issparse(L):
            L = L.toarray()
        A = -(inv_C[:, None] * L)
        if dynamic_heater_inputs:
            A_aug = A
            Phi_aug = np.array([], dtype=float)
        else:
            A_aug = np.zeros((n + 1, n + 1), dtype=float)
            A_aug[:n, :n] = A
            A_aug[:n, n] = b
            Phi_aug = expm(A_aug * float(params.dt_s))
    gpu_stepper = _build_gpu_sparse_stepper(
        A,
        inv_C,
        b,
        radiation_coeff,
        C,
        L,
        G_rad,
        params,
        warnings,
    )
    sparse_implicit_stepper = _build_sparse_implicit_stepper(
        A,
        C,
        L,
        params,
        warnings,
    )
    fast_sparse_substeps = _build_fast_sparse_substeps(
        A,
        C,
        L,
        G_rad,
        params,
        warnings,
    )
    prepared = PreparedSimulation(
        node_ids=node_ids,
        A_aug=A_aug,
        Phi_aug=Phi_aug,
        z=np.concatenate([initial, np.array([1.0])]),
        initial_temperatures_K=initial,
        params=params,
        model=model,
        inv_C=inv_C,
        A=A,
        base_b=b,
        radiation_coeff_W_K4=radiation_coeff,
        gpu_stepper=gpu_stepper,
        sparse_implicit_stepper=sparse_implicit_stepper,
        fast_sparse_substeps=fast_sparse_substeps,
        node_index_by_id=node_index_by_id,
        heater_node_ids=heater_node_ids,
        cryocooler_node_ids=cryocooler_node_ids,
        dynamic_heater_inputs=dynamic_heater_inputs,
        warnings=warnings,
    )
    prepared.reset()
    return prepared


def _build_gpu_sparse_stepper(
    A: Any,
    inv_C: np.ndarray,
    base_b: np.ndarray,
    radiation_coeff: np.ndarray,
    C: np.ndarray,
    L: Any,
    G_rad: np.ndarray,
    params: SimulationParameters,
    warnings: list[str],
) -> GpuSparseStepper | None:
    if not issparse(A):
        return None
    cp, cupyx_sparse, reason = _optional_cupy_modules()
    if cp is None or cupyx_sparse is None:
        warnings.append(f"GPU sparse stepping unavailable; using CPU acceleration instead: {reason}")
        return None
    try:
        if int(cp.cuda.runtime.getDeviceCount()) <= 0:
            warnings.append("GPU sparse stepping unavailable; no CUDA device was reported by CuPy.")
            return None
    except Exception as exc:
        warnings.append(f"GPU sparse stepping unavailable; CUDA device detection failed: {exc}")
        return None
    substeps = _gpu_substep_count(C, L, G_rad, params)
    if substeps is None:
        warnings.append("GPU sparse stepping unavailable; no positive thermal time constant could be estimated.")
        return None
    max_substeps = max(1, int(getattr(params, "gpu_simulation_max_substeps", 128)))
    if substeps > max_substeps:
        warnings.append(
            "GPU sparse stepping unavailable because the timestep would require "
            f"{substeps} explicit substeps; CPU exponential stepping is safer. "
            f"Reduce dt_s or increase gpu_simulation_max_substeps above {max_substeps}."
        )
        return None
    try:
        stepper = GpuSparseStepper(
            cp=cp,
            A_gpu=cupyx_sparse.csr_matrix(A),
            inv_C_gpu=cp.asarray(np.asarray(inv_C, dtype=float).reshape(-1)),
            base_b_gpu=cp.asarray(np.asarray(base_b, dtype=float).reshape(-1)),
            radiation_coeff_gpu=cp.asarray(np.asarray(radiation_coeff, dtype=float).reshape(-1)),
            use_ambient_radiation=bool(params.use_ambient_radiation and np.any(radiation_coeff > 0.0)),
            ambient_K=float(params.T_env_K),
            dt_s=float(params.dt_s),
            substeps=int(substeps),
        )
    except Exception as exc:
        warnings.append(f"GPU sparse stepping unavailable; GPU array initialization failed: {exc}")
        return None
    warnings.append(f"GPU sparse stepping enabled with {int(substeps)} explicit substep(s) per simulation step.")
    return stepper


def _build_sparse_implicit_stepper(
    A: Any,
    C: np.ndarray,
    L: Any,
    params: SimulationParameters,
    warnings: list[str],
) -> SparseImplicitStepper | None:
    if not bool(getattr(params, "implicit_sparse_simulation_enabled", True)):
        return None
    if not issparse(A):
        return None
    dt = float(params.dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        warnings.append("Sparse implicit CPU stepping unavailable; dt_s must be positive.")
        return None
    try:
        C_values = np.asarray(C, dtype=float).reshape(-1)
        if C_values.size == 0 or np.any(C_values <= 0.0) or not np.all(np.isfinite(C_values)):
            warnings.append("Sparse implicit CPU stepping unavailable; capacitance vector is invalid.")
            return None
        L_sparse = csr_matrix(L)
        if _sparse_matrix_is_symmetric(L_sparse):
            solver = "cg"
            capacitance = C_values
            state_operator = L_sparse
        else:
            solver = "bicgstab"
            capacitance = None
            state_operator = csr_matrix(A)
        method = str(getattr(params, "implicit_sparse_simulation_method", "tr_bdf2") or "tr_bdf2").lower()
        if method not in {"tr_bdf2", "backward_euler"}:
            warnings.append(f"Unknown sparse implicit method {method!r}; using tr_bdf2.")
            method = "tr_bdf2"
        rtol = max(0.0, float(getattr(params, "implicit_sparse_simulation_rtol", 1.0e-6)))
        maxiter = max(1, int(getattr(params, "implicit_sparse_simulation_maxiter", 300)))
        stepper = SparseImplicitStepper(
            dt_s=dt,
            rtol=rtol,
            maxiter=maxiter,
            solver=solver,
            state_operator=state_operator,
            capacitance_J_K=capacitance,
            method=method,
            adaptive_substeps_enabled=bool(getattr(params, "implicit_sparse_adaptive_substeps_enabled", True)),
            adaptive_target_delta_K=max(
                1.0e-12,
                float(getattr(params, "implicit_sparse_adaptive_target_delta_K", 1.0)),
            ),
            adaptive_max_substeps=max(1, int(getattr(params, "implicit_sparse_adaptive_max_substeps", 4))),
            residual_check_enabled=bool(getattr(params, "implicit_sparse_residual_check_enabled", True)),
        )
    except Exception as exc:
        warnings.append(f"Sparse implicit CPU stepping unavailable; setup failed: {exc}")
        return None
    warnings.append(
        f"Sparse implicit CPU stepping enabled with {method} / {solver} solver "
        f"(rtol={rtol:g}, maxiter={maxiter})."
    )
    return stepper


def _build_fast_sparse_substeps(
    A: Any,
    C: np.ndarray,
    L: Any,
    G_rad: np.ndarray,
    params: SimulationParameters,
    warnings: list[str],
) -> int | None:
    if not bool(getattr(params, "fast_sparse_simulation_enabled", True)):
        return None
    if not issparse(A):
        return None
    substeps = _explicit_substep_count(
        C,
        L,
        G_rad,
        dt_s=float(params.dt_s),
        safety=float(getattr(params, "fast_sparse_simulation_safety_factor", 0.2)),
    )
    if substeps is None:
        warnings.append("Fast sparse CPU stepping unavailable; no positive thermal time constant could be estimated.")
        return None
    max_substeps = max(1, int(getattr(params, "fast_sparse_simulation_max_substeps", 128)))
    if substeps > max_substeps:
        warnings.append(
            "Fast sparse CPU stepping unavailable because the timestep would require "
            f"{substeps} explicit substeps; CPU expm_multiply is safer. "
            f"Reduce dt_s or increase fast_sparse_simulation_max_substeps above {max_substeps}."
        )
        return None
    warnings.append(f"Fast sparse CPU stepping enabled with {int(substeps)} explicit substep(s) per simulation step.")
    return int(substeps)


def _sparse_matrix_is_symmetric(matrix: Any, tolerance: float = 1.0e-12) -> bool:
    sparse = csr_matrix(matrix)
    if sparse.shape[0] != sparse.shape[1]:
        return False
    difference = (sparse - sparse.T).tocoo()
    if difference.nnz == 0:
        return True
    matrix_scale = float(np.max(np.abs(sparse.data))) if sparse.nnz else 1.0
    difference_scale = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
    return difference_scale <= max(1.0, matrix_scale) * float(tolerance)


def _optional_cupy_modules() -> tuple[Any | None, Any | None, str]:
    try:
        import cupy as cp
        from cupyx.scipy import sparse as cupyx_sparse
    except Exception as exc:
        return None, None, str(exc)
    return cp, cupyx_sparse, ""


def _record_profile_ms(profile: dict[str, float] | None, key: str, start: float) -> None:
    if profile is None:
        return
    profile[key] = profile.get(key, 0.0) + (time.perf_counter() - start) * 1000.0


def _jacobi_preconditioner(matrix: Any) -> LinearOperator:
    sparse = csr_matrix(matrix)
    diagonal = np.asarray(sparse.diagonal(), dtype=float).reshape(-1)
    inv_diagonal = np.zeros_like(diagonal)
    valid = np.isfinite(diagonal) & (np.abs(diagonal) > 1.0e-30)
    inv_diagonal[valid] = 1.0 / diagonal[valid]
    return LinearOperator(sparse.shape, matvec=lambda values: inv_diagonal * values)


def _gpu_substep_count(
    C: np.ndarray,
    L: Any,
    G_rad: np.ndarray,
    params: SimulationParameters,
) -> int | None:
    return _explicit_substep_count(
        C,
        L,
        G_rad,
        dt_s=float(params.dt_s),
        safety=float(getattr(params, "gpu_simulation_safety_factor", 0.2)),
    )


def _explicit_substep_count(
    C: np.ndarray,
    L: Any,
    G_rad: np.ndarray,
    *,
    dt_s: float,
    safety: float,
) -> int | None:
    tau_min = estimate_min_time_constant(C, L, G_rad)
    if tau_min is None or not np.isfinite(tau_min) or tau_min <= 0.0:
        return None
    safety = max(1.0e-6, min(1.0, float(safety)))
    max_substep_s = max(float(tau_min) * safety, 1.0e-12)
    return max(1, int(np.ceil(float(dt_s) / max_substep_s)))


def validate_simulation_inputs(
    model: ThermalGraphModel,
    node_ids: np.ndarray,
    C: np.ndarray,
    L: np.ndarray,
    G_rad: np.ndarray,
    initial: np.ndarray,
    params: SimulationParameters,
) -> list[str]:
    warnings: list[str] = []
    n = len(node_ids)
    if C.shape != (n,):
        warnings.append(f"C length {C.shape} does not match node count {n}.")
    if L.shape != (n, n):
        warnings.append(f"L shape {L.shape} does not match node count {n}.")
    if G_rad.shape != (n,):
        warnings.append(f"G_rad length {G_rad.shape} does not match node count {n}.")
    if initial.shape != (n,):
        warnings.append(f"Initial temperature length {initial.shape} does not match node count {n}.")
    if np.any(C <= 0.0):
        warnings.append("At least one node has nonpositive thermal capacitance.")
    if issparse(L):
        offdiag = L.tocoo()
        mask = offdiag.row != offdiag.col
        if np.any(offdiag.data[mask] > 1.0e-12):
            warnings.append("L has positive off-diagonal entries; expected graph Laplacian off-diagonals <= 0.")
    elif np.any(L - np.diag(np.diag(L)) > 1.0e-12):
        warnings.append("L has positive off-diagonal entries; expected graph Laplacian off-diagonals <= 0.")
    if np.any(G_rad < -1.0e-12):
        warnings.append("Radiation diagonal contains negative values.")
    if params.use_ambient_radiation and not np.isfinite(float(params.T_env_K)):
        warnings.append("Ambient temperature must be finite when radiation is enabled.")
    if not np.all(np.isfinite(initial)):
        warnings.append("At least one initial temperature is not finite.")
    tau = estimate_min_time_constant(C, L, G_rad if params.use_ambient_radiation else None)
    if tau is not None and float(params.dt_s) > 0.2 * tau:
        warnings.append(
            f"dt_s={params.dt_s:g} s is coarse relative to estimated fastest tau={tau:.4g} s."
        )
    missing_initial = [
        int(node_id)
        for node_id in node_ids
        if not np.isfinite(float(getattr(model.nodes[int(node_id)], "initial_temperature_K", 293.15)))
    ]
    if missing_initial:
        warnings.append(f"{len(missing_initial)} nodes have invalid initial temperatures.")
    return warnings


def estimate_min_time_constant(C: np.ndarray, L: np.ndarray, G_rad: np.ndarray | None = None) -> float | None:
    conductance_sum = np.asarray(L.diagonal() if issparse(L) else np.diag(L), dtype=float).copy()
    if G_rad is not None:
        conductance_sum += np.asarray(G_rad, dtype=float).reshape(-1)
    mask = conductance_sum > 0.0
    if not np.any(mask):
        return None
    tau = np.asarray(C, dtype=float).reshape(-1)[mask] / conductance_sum[mask]
    tau = tau[np.isfinite(tau) & (tau > 0.0)]
    return float(np.min(tau)) if tau.size else None


def _radiation_coefficient_vector(
    matrices: dict[str, np.ndarray],
    model: ThermalGraphModel,
    node_ids: np.ndarray,
    G_rad: np.ndarray,
    params: SimulationParameters,
) -> np.ndarray:
    coeff = np.zeros(len(node_ids), dtype=float)
    for row, node_id in enumerate(node_ids):
        node = model.nodes[int(node_id)]
        area_m2 = max(0.0, float(getattr(node, "radiating_area_m2", 0.0)))
        emissivity = max(0.0, float(getattr(node, "emissivity", 0.0)))
        if area_m2 > 0.0 and emissivity > 0.0:
            coeff[row] = emissivity * STEFAN_BOLTZMANN_W_M2K4 * area_m2
    missing = coeff <= 0.0
    if np.any(missing):
        reference_temperature = float(getattr(model.metadata, "T_sur_K", float(params.T_env_K)))
        if not np.isfinite(reference_temperature) or reference_temperature <= 0.0:
            reference_temperature = float(params.T_env_K)
        if np.isfinite(reference_temperature) and reference_temperature > 0.0:
            fallback = np.maximum(0.0, np.asarray(G_rad, dtype=float).reshape(-1)) / (
                4.0 * reference_temperature**3
            )
            coeff[missing] = fallback[missing]
    return coeff


def save_trajectory(folder: Path, simulation_name: str, prepared: PreparedSimulation, notes: str = "") -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in simulation_name).strip("_")
    target = folder / "simulations" / (safe_name or "simulation")
    target.mkdir(parents=True, exist_ok=True)
    times = np.array([state.time_s for state in prepared.history], dtype=float)
    trajectory = np.vstack([state.temperatures_K for state in prepared.history])
    np.save(target / "time.npy", times)
    np.save(target / "trajectory.npy", trajectory)
    with (target / "temperature_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", "min_K", "max_K", "mean_K"])
        writer.writeheader()
        for time_s, row in zip(times, trajectory):
            writer.writerow(
                {
                    "time_s": float(time_s),
                    "min_K": float(np.min(row)),
                    "max_K": float(np.max(row)),
                    "mean_K": float(np.mean(row)),
                }
            )
    (target / "notes.txt").write_text(notes, encoding="utf-8")
    return target


def _radiation_vector(
    matrices: dict[str, np.ndarray], model: ThermalGraphModel, node_ids: np.ndarray
) -> np.ndarray:
    if "G_rad" in matrices:
        raw = np.asarray(matrices["G_rad"], dtype=float)
        if raw.ndim == 2:
            return np.diag(raw).astype(float)
        return raw.reshape(-1).astype(float)
    return np.array(
        [
            model.nodes[int(node_id)].G_rad_W_K
            if model.nodes[int(node_id)].G_rad_W_K > 0.0
            else model.nodes[int(node_id)].Grad_W_K
            for node_id in node_ids
        ],
        dtype=float,
    )


def _heater_power_vector(model: ThermalGraphModel, node_ids: np.ndarray) -> np.ndarray:
    powers = np.zeros(len(node_ids), dtype=float)
    for row, node_id in enumerate(node_ids):
        node = model.nodes[int(node_id)]
        if node.is_heater:
            powers[row] = max(0.0, float(node.heater.heater_max_power_W) * float(node.heater.heater_efficiency))
    return powers


def _fractional_integral(error_history: list[float] | tuple[float, ...], dt_s: float, order: float) -> float:
    history = np.asarray(error_history, dtype=float).reshape(-1)
    if history.size == 0:
        return 0.0
    alpha = max(0.0, float(order))
    dt = max(float(dt_s), 1.0e-12)
    if alpha <= 1.0e-12:
        return float(history[-1])
    weights = np.empty(history.size, dtype=float)
    weights[0] = 1.0
    for index in range(1, history.size):
        weights[index] = weights[index - 1] * (float(index - 1) + alpha) / float(index)
    return float((dt**alpha) * np.dot(weights, history[::-1]))


def _fractional_derivative(
    error_history: list[float] | tuple[float, ...],
    dt_s: float,
    order: float,
    *,
    zero_initial_integer_order: bool = False,
) -> float:
    history = np.asarray(error_history, dtype=float).reshape(-1)
    if history.size == 0:
        return 0.0
    alpha = max(0.0, float(order))
    dt = max(float(dt_s), 1.0e-12)
    if zero_initial_integer_order and history.size == 1 and abs(alpha - 1.0) <= 1.0e-12:
        return 0.0
    if alpha <= 1.0e-12:
        return float(history[-1])
    weights = np.empty(history.size, dtype=float)
    weights[0] = 1.0
    for index in range(1, history.size):
        weights[index] = weights[index - 1] * (1.0 - (alpha + 1.0) / float(index))
    return float(np.dot(weights, history[::-1]) / (dt**alpha))


def _controlled_heater_power_vector(
    model: ThermalGraphModel,
    node_ids: np.ndarray,
    temperatures_K: np.ndarray,
    dt_s: float,
    params: SimulationParameters,
    include_heater_inputs: bool,
    update_pid_state: bool = True,
    excluded_modes: set[str] | None = None,
    include_cryocoolers: bool = True,
    node_index_by_id: dict[int, int] | None = None,
    heater_node_ids: Sequence[int] | None = None,
    cryocooler_node_ids: Sequence[int] | None = None,
) -> np.ndarray:
    powers = np.zeros(len(node_ids), dtype=float)
    skipped_modes = excluded_modes or set()
    node_index = node_index_by_id or {int(node_id): row for row, node_id in enumerate(node_ids)}
    cryocooler_ids = cryocooler_node_ids if cryocooler_node_ids is not None else node_ids
    for node_id in cryocooler_ids:
        row = node_index.get(int(node_id))
        if row is None:
            continue
        node = model.nodes[int(node_id)]
        if include_cryocoolers and node.has_cryocooler:
            powers[int(row)] -= _cryocooler_power_for_temperature(float(temperatures_K[int(row)]), params)
    if not include_heater_inputs:
        return powers
    enabled_heaters = _enabled_node_id_set(params.enabled_heater_node_ids)
    enabled_sensors = _enabled_node_id_set(params.enabled_sensor_node_ids)
    heater_items = (
        ((int(heater_id), model.nodes.get(int(heater_id))) for heater_id in heater_node_ids)
        if heater_node_ids is not None
        else sorted(model.nodes.items(), key=lambda item: int(item[0]))
    )
    for heater_id, heater in heater_items:
        if heater is None:
            continue
        if not heater.is_heater or not _node_id_enabled(enabled_heaters, int(heater_id)):
            continue
        sensor_id = getattr(heater, "assigned_sensor_id", None)
        if sensor_id is None:
            continue
        sensor_id = int(sensor_id)
        sensor = model.nodes.get(sensor_id)
        if sensor is None or not sensor.is_sensor:
            continue
        if not _node_id_enabled(enabled_sensors, sensor_id):
            continue
        if str(getattr(sensor, "sensor_control_mode", "manual")) in skipped_modes:
            continue
        if str(getattr(sensor, "sensor_control_mode", "manual")) != "manual":
            continue
        heater_row = node_index.get(int(heater_id))
        if heater is None or heater_row is None or not heater.is_heater:
            continue
        if not bool(getattr(heater, "heater_valid", True)) or not getattr(heater, "power_deposition_node_ids", []):
            continue
        max_power = max(0.0, float(heater.heater.heater_max_power_W) * float(heater.heater.heater_efficiency))
        command = min(max(float(getattr(sensor, "sensor_manual_power_W", 0.0)), 0.0), max_power)
        _deposit_heater_command_power(powers, model, node_index, int(heater_id), command)
    return powers


def _controlled_heater_command_by_node(
    model: ThermalGraphModel,
    node_ids: np.ndarray,
    temperatures_K: np.ndarray,
    dt_s: float,
    params: SimulationParameters,
    include_heater_inputs: bool,
    excluded_modes: set[str] | None = None,
    heater_node_ids: Sequence[int] | None = None,
) -> dict[int, float]:
    candidate_heater_ids = (
        tuple(int(node_id) for node_id in heater_node_ids)
        if heater_node_ids is not None
        else tuple(int(node_id) for node_id in node_ids if model.nodes[int(node_id)].is_heater)
    )
    if not include_heater_inputs:
        return {
            int(node_id): 0.0
            for node_id in candidate_heater_ids
        }
    skipped_modes = excluded_modes or set()
    enabled_heaters = _enabled_node_id_set(params.enabled_heater_node_ids)
    enabled_sensors = _enabled_node_id_set(params.enabled_sensor_node_ids)
    commands = {
        int(node_id): 0.0
        for node_id in candidate_heater_ids
    }
    heater_items = (
        ((int(heater_id), model.nodes.get(int(heater_id))) for heater_id in candidate_heater_ids)
        if heater_node_ids is not None
        else sorted(model.nodes.items(), key=lambda item: int(item[0]))
    )
    for heater_id, heater in heater_items:
        if heater is None:
            continue
        if int(heater_id) not in commands:
            continue
        if not heater.is_heater or not _node_id_enabled(enabled_heaters, int(heater_id)):
            continue
        sensor_id = getattr(heater, "assigned_sensor_id", None)
        if sensor_id is None:
            continue
        sensor_id = int(sensor_id)
        sensor = model.nodes.get(sensor_id)
        if sensor is None or not sensor.is_sensor:
            continue
        if not _node_id_enabled(enabled_sensors, sensor_id):
            continue
        if str(getattr(sensor, "sensor_control_mode", "manual")) in skipped_modes:
            continue
        if str(getattr(sensor, "sensor_control_mode", "manual")) != "manual":
            continue
        max_power = max(0.0, float(heater.heater.heater_max_power_W) * float(heater.heater.heater_efficiency))
        commands[int(heater_id)] = min(max(float(getattr(sensor, "sensor_manual_power_W", 0.0)), 0.0), max_power)
    return commands


def _deposit_heater_command_power(
    powers: np.ndarray,
    model: ThermalGraphModel,
    node_index: dict[int, int],
    heater_id: int,
    command_W: float,
) -> None:
    command = max(0.0, float(command_W))
    if command <= 0.0:
        return
    heater = model.nodes.get(int(heater_id))
    if heater is None:
        return
    deposition_ids = [
        int(node_id)
        for node_id in getattr(heater, "power_deposition_node_ids", []) or []
        if int(node_id) in node_index
    ]
    if not deposition_ids:
        row = node_index.get(int(heater_id))
        if row is not None:
            powers[int(row)] += command
        return
    weights = _normalized_power_weights(getattr(heater, "power_deposition_weights", []) or [], len(deposition_ids))
    for node_id, weight in zip(deposition_ids, weights):
        row = node_index.get(int(node_id))
        if row is not None:
            powers[int(row)] += command * float(weight)


def _normalized_power_weights(weights: list[float], count: int) -> list[float]:
    if count <= 0:
        return []
    values = [float(value) for value in list(weights)[:count] if np.isfinite(float(value)) and float(value) >= 0.0]
    if len(values) != count or sum(values) <= 0.0:
        return [1.0 / float(count)] * count
    total = float(sum(values))
    return [float(value) / total for value in values]


def _cryocooler_power_vector(
    model: ThermalGraphModel,
    node_ids: np.ndarray,
    temperatures_K: np.ndarray,
    params: SimulationParameters,
    node_index_by_id: dict[int, int] | None = None,
    cryocooler_node_ids: Sequence[int] | None = None,
) -> np.ndarray:
    powers = np.zeros(len(node_ids), dtype=float)
    node_index = node_index_by_id or {int(node_id): row for row, node_id in enumerate(node_ids)}
    candidate_node_ids = cryocooler_node_ids if cryocooler_node_ids is not None else node_ids
    for node_id in candidate_node_ids:
        row = node_index.get(int(node_id))
        if row is None:
            continue
        node = model.nodes[int(node_id)]
        if node.has_cryocooler:
            powers[int(row)] = _cryocooler_power_for_temperature(float(temperatures_K[int(row)]), params)
    return powers


def _cryocooler_power_for_temperature(temperature_K: float, params: SimulationParameters) -> float:
    error = float(temperature_K) - float(params.T_cooler_setpoint)
    raw_power = float(params.Kp_cooler) * error
    # Future improvement: enforce a shared budget such as sum(P_cooler_i) <= P_cooler_max_total.
    return min(max(raw_power, 0.0), max(0.0, float(params.P_cooler_max)))


def _controller_sensor_weight(node: Any) -> float:
    explicit = max(0.0, float(getattr(node, "controller_weight", 0.0)))
    if explicit > 0.0:
        return explicit
    return 0.5


def _controller_heater_max_power(node: Any, params: SimulationParameters) -> float:
    heater = getattr(node, "heater", None)
    max_power = (
        float(getattr(heater, "heater_max_power_W", 0.0))
        * float(getattr(heater, "heater_efficiency", 1.0))
    )
    if max_power <= 0.0:
        max_power = float(params.mimo_default_heater_max_power_W)
    return max(0.0, max_power)


def _enabled_node_id_set(raw_ids: tuple[int, ...] | list[int] | set[int] | None) -> set[int] | None:
    if raw_ids is None:
        return None
    enabled: set[int] = set()
    for raw_id in raw_ids:
        try:
            enabled.add(int(raw_id))
        except (TypeError, ValueError):
            continue
    return enabled


def _node_id_enabled(enabled_ids: set[int] | None, node_id: int) -> bool:
    return enabled_ids is None or int(node_id) in enabled_ids


def _node_has_mimo_controller_tags(node: Any) -> bool:
    return _node_is_mimo_sensor(node) or _node_is_mimo_heater(node)


def _node_is_mimo_heater(node: Any) -> bool:
    return (
        bool(getattr(node, "is_heater", False))
        and bool(getattr(node, "heater_valid", True))
        and bool(getattr(node, "power_deposition_node_ids", [1]))
    ) and (
        getattr(node, "assigned_sensor_id", None) is not None
        or str(getattr(getattr(node, "heater_control", None), "mode", "")) == "mimo"
    )


def _node_is_mimo_sensor(node: Any) -> bool:
    return (
        bool(getattr(node, "is_sensor", False))
        and bool(getattr(node, "sensor_valid", True))
        and not bool(getattr(node, "sensor_monitor_only", False))
        and bool(getattr(node, "readout_node_ids", None) or getattr(node, "sensor_connected_node_ids", None) or getattr(node, "is_heater", False))
        and (
            str(getattr(node, "sensor_control_mode", "manual")) == "mimo"
            or str(getattr(getattr(node, "heater_control", None), "mode", "")) == "mimo"
        )
        and (
            bool(getattr(node, "assigned_heater_ids", []) or [])
            or getattr(node, "assigned_heater_id", None) is not None
            or bool(getattr(node, "is_heater", False))
        )
    )


def _node_uses_mimo_controller(
    node: Any,
    *,
    heater_enabled: bool = True,
    sensor_enabled: bool = True,
) -> bool:
    return (_node_is_mimo_heater(node) and bool(heater_enabled)) or (
        _node_is_mimo_sensor(node) and bool(sensor_enabled)
    )


def _mimo_controller_is_active(
    model: ThermalGraphModel | None,
    node_ids: np.ndarray,
    params: SimulationParameters,
) -> bool:
    if model is None or str(params.input_mode) != "heater_inputs":
        return False
    enabled_heater_ids = _enabled_node_id_set(params.enabled_heater_node_ids)
    enabled_sensor_ids = _enabled_node_id_set(params.enabled_sensor_node_ids)
    is_sensor = any(
        _node_is_mimo_sensor(model.nodes[int(node_id)])
        and _node_id_enabled(enabled_sensor_ids, int(node_id))
        for node_id in node_ids
    )
    is_heater = any(
        bool(getattr(model.nodes[int(node_id)], "is_heater", False))
        and _node_id_enabled(enabled_heater_ids, int(node_id))
        and _heater_has_active_mimo_sensor(model, int(node_id), enabled_sensor_ids)
        for node_id in node_ids
    )
    return is_sensor and is_heater


def _heater_has_active_mimo_sensor(
    model: ThermalGraphModel,
    heater_id: int,
    enabled_sensor_ids: set[int] | None,
) -> bool:
    heater = model.nodes[int(heater_id)]
    sensor_id = getattr(heater, "assigned_sensor_id", None)
    if sensor_id is None and heater.is_sensor and str(getattr(getattr(heater, "heater_control", None), "mode", "")) == "mimo":
        sensor_id = int(heater_id)
    if sensor_id is None:
        return False
    sensor = model.nodes.get(int(sensor_id))
    return bool(
        sensor is not None
        and _node_is_mimo_sensor(sensor)
        and _node_id_enabled(enabled_sensor_ids, int(sensor_id))
    )
