"""Affine heat-transfer simulation model for octree thermal graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
from typing import Any

import numpy as np
from scipy.linalg import expm
from scipy.sparse import bmat, csr_matrix, diags
from scipy.sparse.linalg import expm_multiply

from .mimo_controller import (
    compute_decoupling_matrix,
    update_nonnegative_integrator,
    weighted_rms_error,
)
from .models import ThermalGraphModel
from .simulation_parameters import SimulationParameters


@dataclass
class SimulationState:
    time_s: float
    temperatures_K: np.ndarray
    pid_states: dict[int, tuple[float, float | None]] = field(default_factory=dict)
    controller_integrators: dict[int, float] = field(default_factory=dict)
    controller_y_prev: dict[int, float] = field(default_factory=dict)
    controller_mode: str = "coarse"


@dataclass
class PreparedSimulationSnapshot:
    z: np.ndarray
    history: list[SimulationState]
    history_index: int
    pid_states: dict[int, tuple[float, float | None]]
    controller_integrators: dict[int, float]
    controller_y_prev: dict[int, float]
    controller_mode: str
    controller_weighted_rms_error: float | None
    controller_warnings: list[str]
    controller_last_power_by_heater: dict[int, float]
    controller_D: np.ndarray | None
    controller_D_signature: tuple[Any, ...] | None


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
    dynamic_heater_inputs: bool = False
    warnings: list[str] = field(default_factory=list)
    controller_integrators: dict[int, float] = field(default_factory=dict)
    controller_y_prev: dict[int, float] = field(default_factory=dict)
    controller_mode: str = "coarse"
    controller_weighted_rms_error: float | None = None
    controller_warnings: list[str] = field(default_factory=list)
    controller_last_power_by_heater: dict[int, float] = field(default_factory=dict)
    controller_D: np.ndarray | None = None
    controller_D_signature: tuple[Any, ...] | None = None
    history: list[SimulationState] = field(default_factory=list)
    history_index: int = 0

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
                self.controller_mode,
            )
        ]
        self.history_index = 0

    def set_uniform_temperature(self, temperature_K: float) -> None:
        self._reset_pid_states()
        self.reset_controller_integrators()
        uniform = np.full(len(self.node_ids), float(temperature_K), dtype=float)
        self.z = np.concatenate([uniform, np.array([1.0])])
        self.history = [
            SimulationState(
                0.0,
                uniform.copy(),
                self._pid_state_snapshot(),
                dict(self.controller_integrators),
                dict(self.controller_y_prev),
                self.controller_mode,
            )
        ]
        self.history_index = 0

    def reset_controller_integrators(self) -> None:
        self.controller_integrators = {}
        self.controller_y_prev = {}
        self.controller_mode = "coarse"
        self.controller_weighted_rms_error = None
        self.controller_last_power_by_heater = {}

    def mark_controller_stale(self) -> None:
        self.controller_D = None
        self.controller_D_signature = None

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
                    str(state.controller_mode),
                )
                for state in self.history
            ],
            history_index=int(self.history_index),
            pid_states=self._pid_state_snapshot(),
            controller_integrators=dict(self.controller_integrators),
            controller_y_prev=dict(self.controller_y_prev),
            controller_mode=str(self.controller_mode),
            controller_weighted_rms_error=self.controller_weighted_rms_error,
            controller_warnings=list(self.controller_warnings),
            controller_last_power_by_heater=dict(self.controller_last_power_by_heater),
            controller_D=None if self.controller_D is None else np.asarray(self.controller_D, dtype=float).copy(),
            controller_D_signature=self.controller_D_signature,
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
                str(state.controller_mode),
            )
            for state in snapshot.history
        ]
        self.history_index = int(snapshot.history_index)
        self._restore_pid_state_snapshot(snapshot.pid_states)
        self.controller_integrators = dict(snapshot.controller_integrators)
        self.controller_y_prev = dict(snapshot.controller_y_prev)
        self.controller_mode = str(snapshot.controller_mode)
        self.controller_weighted_rms_error = snapshot.controller_weighted_rms_error
        self.controller_warnings = list(snapshot.controller_warnings)
        self.controller_last_power_by_heater = dict(snapshot.controller_last_power_by_heater)
        self.controller_D = None if snapshot.controller_D is None else np.asarray(snapshot.controller_D, dtype=float).copy()
        self.controller_D_signature = snapshot.controller_D_signature

    def step_forward(self) -> SimulationState:
        if self.history_index < len(self.history) - 1:
            return self.seek(self.history_index + 1)
        if self.dynamic_heater_inputs:
            self._step_dynamic_heater_inputs()
        elif self.Phi_aug is None:
            self.z = np.asarray(
                expm_multiply(self.A_aug * float(self.params.dt_s), self.z),
                dtype=float,
            )
            self.z[-1] = 1.0
        else:
            self.z = self.Phi_aug @ self.z
        state = SimulationState(
            self.time_s + float(self.params.dt_s),
            self.temperatures_K.copy(),
            self._pid_state_snapshot(),
            dict(self.controller_integrators),
            dict(self.controller_y_prev),
            self.controller_mode,
        )
        self.history.append(state)
        self.history_index = len(self.history) - 1
        return state

    def step_with_forced_heater_powers(
        self,
        heater_power_by_node: dict[int, float],
        *,
        keep_cryocoolers_active: bool = True,
    ) -> None:
        if self.model is None:
            return
        powers = np.zeros(len(self.node_ids), dtype=float)
        for row, node_id in enumerate(self.node_ids):
            node = self.model.nodes[int(node_id)]
            if node.has_heater:
                powers[row] += max(0.0, float(heater_power_by_node.get(int(node_id), 0.0)))
            if keep_cryocoolers_active and node.has_cryocooler:
                powers[row] -= _cryocooler_power_for_temperature(float(self.temperatures_K[row]), self.params)
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
        self.controller_mode = state.controller_mode
        return state

    def _step_dynamic_heater_inputs(self) -> None:
        if self.model is None or self.inv_C is None or self.A is None or self.base_b is None:
            return
        if _mimo_controller_is_active(self.model, self.node_ids, self.params):
            heater_power = self._mimo_controller_power_vector(update_state=True)
        else:
            heater_power = _controlled_heater_power_vector(
                self.model,
                self.node_ids,
                self.temperatures_K,
                float(self.params.dt_s),
                self.params,
                include_heater_inputs=self.params.input_mode == "heater_inputs",
                update_pid_state=True,
            )
        self._advance_with_power_vector(heater_power)

    def _advance_with_power_vector(self, heater_power: np.ndarray) -> None:
        if self.inv_C is None or self.A is None or self.base_b is None:
            return
        b = np.asarray(self.base_b, dtype=float) + np.asarray(self.inv_C, dtype=float) * np.asarray(heater_power, dtype=float)
        if self.Phi_aug is None:
            A_aug = bmat(
                [
                    [self.A, csr_matrix(b.reshape(-1, 1))],
                    [csr_matrix((1, len(self.node_ids))), csr_matrix((1, 1))],
                ],
                format="csr",
            )
            self.z = np.asarray(expm_multiply(A_aug * float(self.params.dt_s), self.z), dtype=float)
        else:
            A_aug = np.zeros((len(self.node_ids) + 1, len(self.node_ids) + 1), dtype=float)
            A_aug[: len(self.node_ids), : len(self.node_ids)] = self.A
            A_aug[: len(self.node_ids), len(self.node_ids)] = b
            self.z = expm(A_aug * float(self.params.dt_s)) @ self.z
        self.z[-1] = 1.0

    def _reset_pid_states(self) -> None:
        if self.model is None:
            return
        for node_id in self.node_ids:
            node = self.model.nodes.get(int(node_id))
            if node is not None and node.has_heater:
                node.heater_control.reset_pid_state()

    def _pid_state_snapshot(self) -> dict[int, tuple[float, float | None]]:
        if self.model is None:
            return {}
        snapshot: dict[int, tuple[float, float | None]] = {}
        for node_id in self.node_ids:
            node = self.model.nodes.get(int(node_id))
            if node is not None and node.has_heater:
                state = node.heater_control.pid_state
                snapshot[int(node_id)] = (float(state.integral), state.previous_error)
        return snapshot

    def _restore_pid_state_snapshot(self, snapshot: dict[int, tuple[float, float | None]]) -> None:
        if self.model is None:
            return
        for node_id, values in snapshot.items():
            node = self.model.nodes.get(int(node_id))
            if node is not None and node.has_heater:
                node.heater_control.pid_state.integral = float(values[0])
                node.heater_control.pid_state.previous_error = values[1]

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
                    self.model.nodes[int(node_id)].has_heater
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
        ) if not _mimo_controller_is_active(self.model, self.node_ids, self.params) else self._mimo_controller_power_vector(update_state=False)
        return {
            int(node_id): float(power)
            for node_id, power in zip(self.node_ids, powers)
            if self.model.nodes[int(node_id)].has_heater
            or self.model.nodes[int(node_id)].has_cryocooler
        }

    def cryocooler_power_by_node(self) -> dict[int, float]:
        if (
            self.model is None
            or not self.dynamic_heater_inputs
        ):
            return {
                int(node_id): 0.0
                for node_id in self.node_ids
                if self.model is not None and self.model.nodes[int(node_id)].has_cryocooler
            }
        powers = _cryocooler_power_vector(
            self.model,
            self.node_ids,
            self.temperatures_K,
            self.params,
        )
        return {
            int(node_id): float(power)
            for node_id, power in zip(self.node_ids, powers)
            if self.model.nodes[int(node_id)].has_cryocooler
        }

    def heater_actuator_power_by_node(self, *, disable_mimo_controller: bool = False) -> dict[int, float]:
        if self.model is None:
            return {}
        if _mimo_controller_is_active(self.model, self.node_ids, self.params) and not disable_mimo_controller:
            powers = self._mimo_controller_power_vector(update_state=False)
            cryocooler_powers = _cryocooler_power_vector(self.model, self.node_ids, self.temperatures_K, self.params)
            powers = np.asarray(powers, dtype=float) + cryocooler_powers
        else:
            powers = _controlled_heater_power_vector(
                self.model,
                self.node_ids,
                self.temperatures_K,
                float(self.params.dt_s),
                self.params,
                include_heater_inputs=self.params.input_mode == "heater_inputs",
                update_pid_state=False,
                include_cryocoolers=False,
                excluded_modes={"mimo"} if disable_mimo_controller else None,
            )
        return {
            int(node_id): float(power)
            for node_id, power in zip(self.node_ids, powers)
            if self.model.nodes[int(node_id)].has_heater
        }

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
        )
        sensor_ids = [
            int(node_id)
            for node_id in self.node_ids
            if _node_uses_mimo_controller(self.model.nodes[int(node_id)])
        ]
        heater_ids = [
            int(node_id)
            for node_id in self.node_ids
            if _node_uses_mimo_controller(self.model.nodes[int(node_id)])
        ]
        if not sensor_ids or not heater_ids:
            self.controller_warnings = ["MIMO controller enabled, but at least one sensor and one heater are required."]
            self.controller_last_power_by_heater = {heater_id: 0.0 for heater_id in heater_ids}
            return powers

        node_index = {int(node_id): row for row, node_id in enumerate(self.node_ids)}
        y = np.array([float(self.temperatures_K[node_index[sensor_id]]) for sensor_id in sensor_ids], dtype=float)
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
        self._update_controller_mode(rms)

        eta_old = np.array([float(self.controller_integrators.get(sensor_id, 0.0)) for sensor_id in sensor_ids])
        eta = update_nonnegative_integrator(eta_old, errors, float(self.params.dt_s)) if update_state else eta_old
        if self.controller_mode == "hold":
            kp_key = "controller_kp_hold"
            ki_key = "controller_ki_hold"
        else:
            kp_key = "controller_kp_coarse"
            ki_key = "controller_ki_coarse"
        Kp = np.array(
            [max(0.0, float(getattr(self.model.nodes[sensor_id], kp_key, 0.0))) for sensor_id in sensor_ids],
            dtype=float,
        )
        Ki = np.array(
            [max(0.0, float(getattr(self.model.nodes[sensor_id], ki_key, 0.0))) for sensor_id in sensor_ids],
            dtype=float,
        )
        virtual_power = Kp * errors + Ki * eta

        G = np.array(
            [
                [self.model.controller_gain(sensor_id, heater_id) for heater_id in heater_ids]
                for sensor_id in sensor_ids
            ],
            dtype=float,
        )
        D = self._controller_decoupling(sensor_ids, heater_ids, G, weights)
        if D is None:
            u = np.zeros(len(heater_ids), dtype=float)
        else:
            u = np.asarray(D @ virtual_power, dtype=float).reshape(-1)
        maxima = np.array(
            [_controller_heater_max_power(self.model.nodes[heater_id], self.params) for heater_id in heater_ids],
            dtype=float,
        )
        u = np.clip(u, 0.0, maxima)
        for heater_id, command in zip(heater_ids, u):
            powers[node_index[heater_id]] += float(command)
        self.controller_last_power_by_heater = {
            int(heater_id): float(command)
            for heater_id, command in zip(heater_ids, u)
        }
        if update_state:
            self.controller_integrators = {
                int(sensor_id): float(value)
                for sensor_id, value in zip(sensor_ids, eta)
            }
            self.controller_y_prev = {
                int(sensor_id): float(value)
                for sensor_id, value in zip(sensor_ids, y)
            }
        return powers

    def _update_controller_mode(self, weighted_rms: float) -> None:
        if self.controller_mode not in {"coarse", "hold"}:
            self.controller_mode = "coarse"
        if self.controller_mode == "coarse" and weighted_rms <= float(self.params.mimo_hold_threshold_K):
            self.controller_mode = "hold"
        elif self.controller_mode == "hold" and weighted_rms >= float(self.params.mimo_coarse_threshold_K):
            self.controller_mode = "coarse"

    def _controller_decoupling(
        self,
        sensor_ids: list[int],
        heater_ids: list[int],
        G: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray | None:
        signature = (
            tuple(sensor_ids),
            tuple(heater_ids),
            tuple(float(value) for value in np.asarray(G, dtype=float).reshape(-1)),
            tuple(float(value) for value in np.asarray(weights, dtype=float).reshape(-1)),
            float(self.params.mimo_decoupling_lambda),
            float(self.params.mimo_coupling_cutoff_fraction),
        )
        if self.controller_D is not None and self.controller_D_signature == signature:
            return self.controller_D
        try:
            result = compute_decoupling_matrix(
                G,
                weights,
                float(self.params.mimo_decoupling_lambda),
                float(self.params.mimo_coupling_cutoff_fraction),
            )
        except Exception as exc:
            self.controller_D = None
            self.controller_D_signature = None
            self.controller_warnings = [f"MIMO decoupling failed: {exc}"]
            return None
        self.controller_D = result.D
        self.controller_D_signature = signature
        self.controller_warnings = list(result.warnings)
        return self.controller_D


def prepare_simulation(
    model: ThermalGraphModel,
    matrices: dict[str, np.ndarray],
    params: SimulationParameters,
) -> PreparedSimulation:
    node_ids = np.asarray(matrices.get("node_ids", model.ordered_node_ids()), dtype=int)
    n = len(node_ids)
    C = np.asarray(matrices.get("C", [model.nodes[int(node_id)].C_J_K for node_id in node_ids]), dtype=float).reshape(-1)
    L = np.asarray(matrices.get("L"), dtype=float)
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

    radiation_diag = G_rad if params.use_ambient_radiation else np.zeros(n, dtype=float)
    inv_C = 1.0 / C
    b = inv_C * radiation_diag * float(params.T_env_K)
    has_cryocooler = any(model.nodes[int(node_id)].has_cryocooler for node_id in node_ids)
    has_mimo_controller = _mimo_controller_is_active(model, node_ids, params)
    dynamic_heater_inputs = params.input_mode == "heater_inputs" or has_cryocooler or has_mimo_controller
    if params.input_mode == "heater_inputs":
        if not any(
            model.nodes[int(node_id)].has_heater
            or model.nodes[int(node_id)].has_cryocooler
            for node_id in node_ids
        ):
            warnings.append(
                "Input mode requested heater inputs, but no heater or cryocooler powers are defined; using zero input."
            )
    elif params.input_mode != "zero":
        warnings.append(f"Unknown input mode {params.input_mode!r}; using zero input.")
    if params.input_mode == "heater_inputs" and any(
        getattr(model.nodes[int(node_id)].heater_control, "mode", "manual") == "mimo"
        for node_id in node_ids
        if model.nodes[int(node_id)].has_heater
    ) and not has_mimo_controller:
        warnings.append("MIMO heater control is selected, but no valid MIMO heater/sensor cells are tagged.")

    sparse_stepper = n > 512
    if sparse_stepper:
        L_sparse = csr_matrix(L)
        A = -(diags(inv_C, format="csr") @ (L_sparse + diags(radiation_diag, format="csr")))
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
        A = -(inv_C[:, None] * (L + np.diag(radiation_diag)))
        if dynamic_heater_inputs:
            A_aug = A
            Phi_aug = np.array([], dtype=float)
        else:
            A_aug = np.zeros((n + 1, n + 1), dtype=float)
            A_aug[:n, :n] = A
            A_aug[:n, n] = b
            Phi_aug = expm(A_aug * float(params.dt_s))
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
        dynamic_heater_inputs=dynamic_heater_inputs,
        warnings=warnings,
    )
    prepared.reset()
    return prepared


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
    if np.any(L - np.diag(np.diag(L)) > 1.0e-12):
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
    conductance_sum = np.asarray(np.diag(L), dtype=float).copy()
    if G_rad is not None:
        conductance_sum += np.asarray(G_rad, dtype=float).reshape(-1)
    mask = conductance_sum > 0.0
    if not np.any(mask):
        return None
    tau = np.asarray(C, dtype=float).reshape(-1)[mask] / conductance_sum[mask]
    tau = tau[np.isfinite(tau) & (tau > 0.0)]
    return float(np.min(tau)) if tau.size else None


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
        if node.has_heater:
            powers[row] = max(0.0, float(node.heater.heater_max_power_W) * float(node.heater.heater_efficiency))
    return powers


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
) -> np.ndarray:
    powers = np.zeros(len(node_ids), dtype=float)
    dt = max(float(dt_s), 1.0e-12)
    skipped_modes = excluded_modes or set()
    for row, node_id in enumerate(node_ids):
        node = model.nodes[int(node_id)]
        if include_cryocoolers and node.has_cryocooler:
            powers[row] -= _cryocooler_power_for_temperature(float(temperatures_K[row]), params)
        if not include_heater_inputs or not node.has_heater:
            continue
        control = node.heater_control
        if str(control.mode) in skipped_modes:
            continue
        max_power = max(0.0, float(node.heater.heater_max_power_W) * float(node.heater.heater_efficiency))
        if control.mode == "pid":
            current_temperature = float(temperatures_K[row])
            error = float(control.pid.setpoint) - current_temperature
            state = control.pid_state
            derivative = 0.0 if state.previous_error is None else (error - float(state.previous_error)) / dt
            integral = float(state.integral)
            leak_rate = max(0.0, float(getattr(control.pid, "integral_leak_per_s", 0.0)))
            leaked_integral = max(0.0, integral * float(np.exp(-leak_rate * dt)))
            if error <= 0.0:
                if update_pid_state:
                    state.integral = max(0.0, leaked_integral + error * dt)
                    state.previous_error = error
                continue
            candidate_integral = leaked_integral + error * dt if update_pid_state else integral
            raw_output = (
                float(control.pid.kp) * error
                + float(control.pid.ki) * candidate_integral
                + float(control.pid.kd) * derivative
            )
            powers[row] += min(max(raw_output, 0.0), max_power)
            if update_pid_state:
                if raw_output <= max_power:
                    state.integral = candidate_integral
                else:
                    state.integral = leaked_integral
                state.previous_error = error
        else:
            powers[row] += min(max(float(control.manual.power), 0.0), max_power)
    return powers


def _cryocooler_power_vector(
    model: ThermalGraphModel,
    node_ids: np.ndarray,
    temperatures_K: np.ndarray,
    params: SimulationParameters,
) -> np.ndarray:
    powers = np.zeros(len(node_ids), dtype=float)
    for row, node_id in enumerate(node_ids):
        node = model.nodes[int(node_id)]
        if node.has_cryocooler:
            powers[row] = _cryocooler_power_for_temperature(float(temperatures_K[row]), params)
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
    return 1.0 if bool(getattr(node, "has_heater", False)) else 0.5


def _controller_heater_max_power(node: Any, params: SimulationParameters) -> float:
    heater = getattr(node, "heater", None)
    max_power = (
        float(getattr(heater, "heater_max_power_W", 0.0))
        * float(getattr(heater, "heater_efficiency", 1.0))
    )
    if max_power <= 0.0:
        max_power = float(params.mimo_default_heater_max_power_W)
    return max(0.0, max_power)


def _node_uses_mimo_controller(node: Any) -> bool:
    return (
        bool(getattr(node, "has_heater", False))
        and bool(getattr(node, "has_sensor", False))
        and str(getattr(getattr(node, "heater_control", None), "mode", "manual")) == "mimo"
    )


def _mimo_controller_is_active(
    model: ThermalGraphModel | None,
    node_ids: np.ndarray,
    params: SimulationParameters,
) -> bool:
    if model is None or str(params.input_mode) != "heater_inputs":
        return False
    return any(_node_uses_mimo_controller(model.nodes[int(node_id)]) for node_id in node_ids)
