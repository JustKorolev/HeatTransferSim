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

from . import material_properties_cryo as _mp
from .cryocooler import CryocoolerDevice, PT60LiftCurve, build_cryocooler_devices
from .mimo_controller import (
    allocate_thermal_rate_qp,
    weighted_rms_error,
)
from .models import ThermalGraphModel
from .temperature_dependent_properties import (
    TemperatureDependentOperator,
    build_temperature_dependent_operator,
)
from .role_pairing import (
    average_inverse_capacitance_for_sensor,
    refresh_heater_power_deposition_nodes,
    refresh_sensor_connected_nodes,
    sensor_readout_temperature_K,
)
from .simulation_parameters import SimulationParameters

STEFAN_BOLTZMANN_W_M2K4 = 5.670374419e-8


def _resolve_solver_backend(backend: str) -> dict[str, Any]:
    """Return array/sparse/solver primitives for the CPU (numpy/scipy) or GPU (cupy) backend."""
    if str(backend).lower() == "gpu":
        import cupy as xp  # noqa: F401
        from cupyx.scipy import sparse as xsparse
        from cupyx.scipy.sparse.linalg import LinearOperator as _LinearOperator
        from cupyx.scipy.sparse.linalg import cg as _cg

        try:
            from cupyx.scipy.sparse.linalg import bicgstab as _bicgstab
        except Exception:
            _bicgstab = None
        return {
            "gpu": True,
            "xp": xp,
            "csr": xsparse.csr_matrix,
            "diags": lambda values: xsparse.diags(values, format="csr"),
            "eye": lambda size: xsparse.eye(size, format="csr"),
            "cg": _cg,
            "bicgstab": _bicgstab,
            "LinearOperator": _LinearOperator,
        }
    return {
        "gpu": False,
        "xp": np,
        "csr": csr_matrix,
        "diags": lambda values: diags(values, format="csr"),
        "eye": lambda size: eye(size, format="csr"),
        "cg": cg,
        "bicgstab": bicgstab,
        "LinearOperator": LinearOperator,
    }


@dataclass
class SparseImplicitStepper:
    """Implicit TR-BDF2 (or backward-Euler) thermal stepper.

    Runs on the CPU (numpy/scipy) or GPU (cupy/cupyx) depending on ``backend``.
    The numerics are identical across backends; only the array/sparse/CG modules
    differ. Host numpy arrays are accepted and returned at the ``step`` boundary.
    """

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
    backend: str = "cpu"
    block_jacobi_enabled: bool = False
    block_jacobi_size: int = 64
    gamma: float = 2.0 - float(np.sqrt(2.0))
    last_info: int = 0
    last_iterations: int = 0
    last_substeps: int = 1
    last_residual_norm: float = 0.0
    last_relative_residual_norm: float = 0.0
    last_predicted_delta_K: float = 0.0
    _stage_cache: dict[int, tuple[Any, Any | None, Any, Any | None]] = field(default_factory=dict)
    _block_partition_cache: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict, repr=False)
    _backend_modules: dict[str, Any] | None = field(default=None, repr=False)
    _operator_device: Any = field(default=None, repr=False)
    _capacitance_device: Any = field(default=None, repr=False)

    @property
    def is_gpu(self) -> bool:
        return str(self.backend).lower() == "gpu"

    @property
    def _be(self) -> dict[str, Any]:
        if self._backend_modules is None:
            self._backend_modules = _resolve_solver_backend(self.backend)
        return self._backend_modules

    def set_operator(self, state_operator: Any, capacitance_J_K: np.ndarray | None = None) -> None:
        """Swap in a rebuilt operator (temperature-dependent properties) and drop caches."""
        self.state_operator = state_operator
        if capacitance_J_K is not None:
            self.capacitance_J_K = capacitance_J_K
        self._operator_device = None
        self._capacitance_device = None
        self._stage_cache.clear()

    def _to_device(self, array: np.ndarray) -> Any:
        return self._be["xp"].asarray(np.asarray(array, dtype=float).reshape(-1))

    def _to_host(self, array: Any) -> np.ndarray:
        if self.is_gpu:
            import cupy as cp

            return cp.asnumpy(array).reshape(-1)
        return np.asarray(array, dtype=float).reshape(-1)

    def _operator(self) -> Any:
        if self._operator_device is None:
            self._operator_device = self._be["csr"](self.state_operator)
        return self._operator_device

    def _capacitance(self) -> Any | None:
        if self.capacitance_J_K is None:
            return None
        if self._capacitance_device is None:
            self._capacitance_device = self._to_device(self.capacitance_J_K)
        return self._capacitance_device

    def step(self, temperatures_K: np.ndarray, source_K_s: np.ndarray, min_substeps: int = 1) -> np.ndarray:
        temps_host = np.asarray(temperatures_K, dtype=float).reshape(-1)
        source_host = np.asarray(source_K_s, dtype=float).reshape(-1)
        if temps_host.shape != source_host.shape:
            raise ValueError(f"Source vector length {source_host.shape} does not match temperatures {temps_host.shape}.")
        temperatures = self._to_device(temps_host)
        source = self._to_device(source_host)
        if str(self.method).lower() != "tr_bdf2":
            return self._to_host(self._backward_euler_step(temperatures, source))
        # min_substeps lets the caller force a finer subdivision than the adaptive
        # estimate -- used to recover from stiff cryogenic steps where the coarse
        # stage matrix diag(C)+alpha*L is too ill-conditioned for the linear solver.
        substeps = max(self._adaptive_substep_count(temperatures, source), max(1, int(min_substeps)))
        h = float(self.dt_s) / max(1, int(substeps))
        current = temperatures.copy()
        self.last_iterations = 0
        self.last_info = 0
        self.last_substeps = int(substeps)
        self.last_residual_norm = 0.0
        self.last_relative_residual_norm = 0.0
        for _ in range(max(1, int(substeps))):
            current = self._tr_bdf2_substep(current, source, h)
        result = self._to_host(current)
        if result.shape != temps_host.shape or not np.all(np.isfinite(result)):
            raise RuntimeError(f"{self.solver} returned an invalid temperature vector.")
        return result

    def _backward_euler_step(self, temperatures: Any, source: Any) -> Any:
        xp = self._be["xp"]
        self.last_iterations = 0
        self.last_info = 0
        self.last_substeps = 1
        self.last_residual_norm = 0.0
        self.last_relative_residual_norm = 0.0
        self.last_predicted_delta_K = 0.0
        operator = self._operator()
        capacitance = self._capacitance()
        if capacitance is not None:
            if capacitance.shape != temperatures.shape:
                raise ValueError(f"Capacitance vector length {capacitance.shape} does not match temperatures {temperatures.shape}.")
            rhs = float(self.dt_s) * (capacitance * source - (operator @ temperatures))
            system_matrix = (self._be["diags"](capacitance) + float(self.dt_s) * operator).tocsr()
        else:
            rhs = float(self.dt_s) * ((operator @ temperatures) + source)
            system_matrix = (self._be["eye"](temperatures.shape[0]) - float(self.dt_s) * operator).tocsr()
        if not bool(xp.all(xp.isfinite(rhs))):
            raise RuntimeError(f"{self.solver} received an invalid implicit update right-hand side.")
        if float(xp.linalg.norm(rhs)) <= 0.0:
            return temperatures.copy()
        preconditioner = self._make_preconditioner(system_matrix)
        result = self._solve_linear(system_matrix, rhs, preconditioner, xp.zeros_like(temperatures))
        if result.shape != temperatures.shape or not bool(xp.all(xp.isfinite(result))):
            raise RuntimeError(f"{self.solver} returned an invalid temperature vector.")
        return temperatures + result

    def _tr_bdf2_substep(self, temperatures: Any, source: Any, h_s: float) -> Any:
        xp = self._be["xp"]
        gamma = min(0.95, max(0.05, float(self.gamma)))
        h = max(float(h_s), 1.0e-30)
        alpha = 0.5 * gamma * h
        stage1_matrix, stage1_preconditioner, stage2_matrix, stage2_preconditioner = self._stage_matrices_for_h(h)
        operator = self._operator()
        capacitance = self._capacitance()
        if capacitance is not None:
            rhs1 = capacitance * temperatures - alpha * (operator @ temperatures)
            rhs1 = rhs1 + gamma * h * capacitance * source
            stage1 = self._solve_linear(stage1_matrix, rhs1, stage1_preconditioner, temperatures)
            rhs2 = (
                (1.0 / (gamma * (1.0 - gamma) * h)) * capacitance * stage1
                - ((1.0 - gamma) / (gamma * h)) * capacitance * temperatures
                + capacitance * source
            )
        else:
            rhs1 = temperatures + alpha * (operator @ temperatures)
            rhs1 = rhs1 + gamma * h * source
            stage1 = self._solve_linear(stage1_matrix, rhs1, stage1_preconditioner, temperatures)
            rhs2 = (
                (1.0 / (gamma * (1.0 - gamma) * h)) * stage1
                - ((1.0 - gamma) / (gamma * h)) * temperatures
                + source
            )
        result = self._solve_linear(stage2_matrix, rhs2, stage2_preconditioner, stage1)
        if self.residual_check_enabled:
            residual = (stage2_matrix @ result) - rhs2
            residual_norm = float(xp.linalg.norm(residual))
            rhs_norm = float(xp.linalg.norm(rhs2))
            self.last_residual_norm = max(float(self.last_residual_norm), residual_norm)
            relative = residual_norm / max(rhs_norm, 1.0e-30)
            self.last_relative_residual_norm = max(float(self.last_relative_residual_norm), float(relative))
        return result

    def _stage_matrices_for_h(self, h_s: float) -> tuple[Any, Any | None, Any, Any | None]:
        key = int(round(float(h_s) / max(float(self.dt_s), 1.0e-30) * 1.0e9))
        cached = self._stage_cache.get(key)
        if cached is not None:
            return cached
        gamma = min(0.95, max(0.05, float(self.gamma)))
        h = max(float(h_s), 1.0e-30)
        alpha = 0.5 * gamma * h
        operator = self._operator()
        capacitance = self._capacitance()
        stage2_scale = (2.0 - gamma) / ((1.0 - gamma) * h)
        if capacitance is not None:
            C_diag = self._be["diags"](capacitance)
            stage1_matrix = (C_diag + alpha * operator).tocsr()
            stage2_matrix = (stage2_scale * C_diag + operator).tocsr()
        else:
            identity = self._be["eye"](operator.shape[0])
            stage1_matrix = (identity - alpha * operator).tocsr()
            stage2_matrix = (stage2_scale * identity - operator).tocsr()
        cached = (
            stage1_matrix,
            self._make_preconditioner(stage1_matrix),
            stage2_matrix,
            self._make_preconditioner(stage2_matrix),
        )
        self._stage_cache[key] = cached
        return cached

    def _make_preconditioner(self, matrix: Any) -> Any:
        """Pick the preconditioner for ``matrix`` based on the configured strategy."""
        if bool(self.block_jacobi_enabled) and int(self.block_jacobi_size) > 1:
            return self._block_jacobi_preconditioner(matrix)
        return self._jacobi_preconditioner(matrix)

    def _jacobi_preconditioner(self, matrix: Any) -> Any:
        xp = self._be["xp"]
        diagonal = xp.asarray(matrix.diagonal()).reshape(-1)
        valid = xp.isfinite(diagonal) & (xp.abs(diagonal) > 1.0e-30)
        inv_diagonal = xp.where(valid, 1.0 / xp.where(valid, diagonal, 1.0), 0.0)
        return self._be["LinearOperator"](matrix.shape, matvec=lambda values: inv_diagonal * values)

    def _block_partition(self, n: int, block_size: int) -> dict[str, Any]:
        """Contiguous block layout for ``n`` nodes; cached because it depends only
        on (n, block_size), not on the matrix values (which change per stage)."""
        key = (int(n), int(block_size))
        cached = self._block_partition_cache.get(key)
        if cached is not None:
            return cached
        xp = self._be["xp"]
        num_blocks = (int(n) + int(block_size) - 1) // int(block_size)
        layout = {
            "num_blocks": int(num_blocks),
            "block_size": int(block_size),
            "padded_n": int(num_blocks * int(block_size)),
            "diag_index": xp.arange(int(block_size)),
        }
        self._block_partition_cache[key] = layout
        return layout

    def _block_jacobi_preconditioner(self, matrix: Any) -> Any:
        """Block-Jacobi preconditioner: invert small contiguous diagonal blocks.

        The N nodes are split into contiguous blocks of ``block_jacobi_size``; the
        dense diagonal sub-block of each is inverted and applied as a batched
        matmul, which works identically on scipy (CPU) and cupyx (GPU) arrays.
        Falls back to plain Jacobi if the blocks cannot be inverted.
        """
        xp = self._be["xp"]
        n = int(matrix.shape[0])
        block_size = max(1, int(self.block_jacobi_size))
        if n == 0 or block_size <= 1:
            return self._jacobi_preconditioner(matrix)
        layout = self._block_partition(n, block_size)
        num_blocks = layout["num_blocks"]
        padded_n = layout["padded_n"]
        diag_index = layout["diag_index"]

        # Dense (num_blocks, block_size, block_size) stack of the contiguous
        # diagonal blocks. Initialise to identity so the trailing partial block's
        # padding (local index >= remainder) inverts to identity and acts as a
        # no-op on the zero-padded tail of the vector.
        blocks = xp.zeros((num_blocks, block_size, block_size), dtype=float)
        blocks[:, diag_index, diag_index] = 1.0
        coo = matrix.tocoo()
        rows = xp.asarray(coo.row)
        cols = xp.asarray(coo.col)
        data = xp.asarray(coo.data, dtype=float)
        same_block = (rows // block_size) == (cols // block_size)
        block_of = rows[same_block] // block_size
        local_i = rows[same_block] % block_size
        local_j = cols[same_block] % block_size
        blocks[block_of, local_i, local_j] = data[same_block]
        try:
            inv_blocks = xp.linalg.inv(blocks)
        except Exception:  # noqa: BLE001 - singular block; degrade gracefully
            return self._jacobi_preconditioner(matrix)

        def _matvec(values: Any) -> Any:
            vec = values.reshape(-1)
            if padded_n != n:
                buffer = xp.zeros(padded_n, dtype=vec.dtype)
                buffer[:n] = vec
            else:
                buffer = vec
            reshaped = buffer.reshape(num_blocks, block_size)
            applied = xp.einsum("kij,kj->ki", inv_blocks, reshaped).reshape(-1)
            return applied[:n]

        return self._be["LinearOperator"](matrix.shape, matvec=_matvec)

    def _adaptive_substep_count(self, temperatures: Any, source: Any) -> int:
        xp = self._be["xp"]
        max_substeps = max(1, int(self.adaptive_max_substeps))
        if not bool(self.adaptive_substeps_enabled) or max_substeps <= 1:
            self.last_predicted_delta_K = 0.0
            return 1
        rate = self._thermal_rate(temperatures, source)
        if rate.shape != temperatures.shape or not bool(xp.all(xp.isfinite(rate))):
            self.last_predicted_delta_K = 0.0
            return 1
        predicted_delta = float(self.dt_s) * float(xp.max(xp.abs(rate))) if rate.size else 0.0
        self.last_predicted_delta_K = max(0.0, predicted_delta)
        target = max(float(self.adaptive_target_delta_K), 1.0e-12)
        return max(1, min(max_substeps, int(np.ceil(self.last_predicted_delta_K / target))))

    def _thermal_rate(self, temperatures: Any, source: Any) -> Any:
        xp = self._be["xp"]
        operator = self._operator()
        capacitance = self._capacitance()
        if capacitance is not None:
            if capacitance.shape != temperatures.shape:
                return xp.zeros_like(temperatures)
            return source - (operator @ temperatures) / capacitance
        return (operator @ temperatures) + source

    def _solve_linear(
        self,
        matrix: Any,
        rhs: Any,
        preconditioner: Any | None,
        x0: Any,
    ) -> Any:
        xp = self._be["xp"]
        if not bool(xp.all(xp.isfinite(rhs))):
            raise RuntimeError(f"{self.solver} received an invalid right-hand side.")
        guess = x0
        if guess.shape != rhs.shape:
            raise RuntimeError(f"{self.solver} received an invalid initial guess.")
        correction_rhs = rhs - (matrix @ guess)
        if not bool(xp.all(xp.isfinite(correction_rhs))):
            raise RuntimeError(f"{self.solver} received an invalid correction right-hand side.")
        if float(xp.linalg.norm(correction_rhs)) <= 0.0:
            return guess.copy()
        iterations = 0

        def _count_iteration(_x: Any) -> None:
            nonlocal iterations
            iterations += 1

        solve = self._be["cg"] if self.solver == "cg" else self._be["bicgstab"]
        if solve is None:
            raise RuntimeError(f"Solver {self.solver!r} is unavailable on backend {self.backend!r}.")
        result, info = solve(
            matrix,
            correction_rhs,
            x0=xp.zeros_like(correction_rhs),
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
        if result.shape != rhs.shape or not bool(xp.all(xp.isfinite(result))):
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
    controller_mode: str
    controller_weighted_rms_error: float | None
    controller_warnings: list[str]
    controller_last_power_by_heater: dict[int, float]
    controller_allocator_diagnostics: dict[str, Any]


@dataclass
class PreparedSimulation:
    node_ids: np.ndarray
    z: np.ndarray
    initial_temperatures_K: np.ndarray
    params: SimulationParameters
    model: ThermalGraphModel | None = None
    inv_C: np.ndarray | None = None
    A: Any | None = None
    base_b: np.ndarray | None = None
    radiation_coeff_W_K4: np.ndarray | None = None
    # Per-node radiative background temperature [K]: the fixed-temperature
    # environment each surface radiates to (exterior/ambient by default; interior
    # cryo enclosure for inward-facing surfaces once classified). None => scalar
    # params.T_env_K for every node.
    environment_temperature_K: np.ndarray | None = None
    # Surface-to-surface radiative exchange (gray-diffuse). W is a sparse
    # symmetric matrix of exchange areas G_ij = A_i * script-F_ij [m^2]; the net
    # exchange power is sigma*(W @ T^4 - degree * T^4), a Laplacian on T^4.
    radiation_exchange_W: Any | None = None
    radiation_exchange_degree: np.ndarray | None = None
    # Factored (grouped) surface-to-surface exchange for large graphs: aggregate
    # node T^4 to super-surfaces (S), do the small super-level Laplacian
    # (W_super), distribute power back (S^T). Net power = sigma * S^T (W_super -
    # D_super) S T^4 -- three sparse matvecs, O(n_node) per step.
    radiation_super_S: Any | None = None
    radiation_super_W: Any | None = None
    radiation_super_degree: np.ndarray | None = None
    sparse_implicit_stepper: SparseImplicitStepper | None = None
    gpu_implicit_stepper: SparseImplicitStepper | None = None
    temperature_dependent_operator: TemperatureDependentOperator | None = None
    node_index_by_id: dict[int, int] = field(default_factory=dict)
    heater_node_ids: tuple[int, ...] = ()
    cryocooler_node_ids: tuple[int, ...] = ()
    cryocooler_devices: tuple[CryocoolerDevice, ...] = ()
    cryocooler_lift_curve: PT60LiftCurve | None = None
    last_cryocooler_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    dynamic_heater_inputs: bool = False
    warnings: list[str] = field(default_factory=list)
    controller_integrators: dict[int, float] = field(default_factory=dict)
    controller_y_prev: dict[int, float] = field(default_factory=dict)
    controller_dTdt_hat_by_sensor: dict[int, float] = field(default_factory=dict)
    controller_error_prev: dict[int, float] = field(default_factory=dict)
    controller_mode: str = "coarse"
    controller_weighted_rms_error: float | None = None
    controller_warnings: list[str] = field(default_factory=list)
    controller_last_power_by_heater: dict[int, float] = field(default_factory=dict)
    controller_allocator_diagnostics: dict[str, Any] = field(default_factory=dict)
    controller_dynamic_gain_cache: dict[str, Any] = field(default_factory=dict)
    history: list[SimulationState] = field(default_factory=list)
    history_index: int = 0
    last_step_profile_ms: dict[str, float] = field(default_factory=dict, init=False)
    _history_limit_warned: bool = field(default=False, init=False, repr=False)

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
        self.controller_mode = "coarse"
        self.controller_weighted_rms_error = None
        self.controller_last_power_by_heater = {}
        self.controller_allocator_diagnostics = {}
        self.controller_modal_integral = None
        self.controller_modal_ff_correction = None
        self._modal_ff_rls_P = None
        self._modal_ff_prev_y_dev = None
        self.controller_dynamic_gain_cache = {}

    def mark_controller_stale(self) -> None:
        self.controller_dTdt_hat_by_sensor = {}
        self.controller_last_power_by_heater = {}
        self.controller_allocator_diagnostics = {}
        self.controller_modal_integral = None
        self.controller_modal_ff_correction = None
        self._modal_ff_rls_P = None
        self._modal_ff_prev_y_dev = None
        self.controller_dynamic_gain_cache = {}

    def refresh_cryocoolers(self) -> None:
        """Rebuild the cryocooler devices from the current model WITHOUT re-preparing
        the whole simulation. A cryocooler is a runtime heat-sink source -- it does
        not change C/L/G_rad or the radiation coupling -- so matrices, operators,
        radiation and solver state are all preserved. This is much cheaper than a
        full ``prepare_simulation`` on a large graph, so a cryocooler (re)assignment
        no longer has to trigger a costly reinitialisation."""
        if self.model is None or self.inv_C is None:
            return
        capacitance = 1.0 / np.asarray(self.inv_C, dtype=float)
        devices, _warnings = build_cryocooler_devices(self.model, self.node_ids, capacitance)
        self.cryocooler_devices = tuple(devices)
        # The per-step cooling filters devices by cryocooler_node_ids, so refresh
        # that source list too (it was captured at prepare time).
        self.cryocooler_node_ids = tuple(
            int(node_id)
            for node_id in sorted(int(value) for value in self.node_ids)
            if bool(getattr(self.model.nodes.get(int(node_id)), "has_cryocooler", False))
        )
        # A cooler makes the per-step source non-trivial, so ensure the dynamic
        # per-step RHS path runs (it may have been off when no cooler existed).
        if self.cryocooler_node_ids:
            self.dynamic_heater_inputs = True

    def _live_capacitance(self) -> np.ndarray | None:
        """Current per-node heat capacity C=1/inv_C [J/K] (reflects tdep cp(T) when
        active), so the cryocooler over-cool cap uses the genuine cryogenic capacity
        rather than re-deriving it per node."""
        inv_C = getattr(self, "inv_C", None)
        if inv_C is None:
            return None
        inv = np.asarray(inv_C, dtype=float).reshape(-1)
        return np.where(inv > 0.0, 1.0 / np.where(inv > 0.0, inv, 1.0), 0.0)

    def snapshot_state(self) -> PreparedSimulationSnapshot:
        # The history list is copied by REFERENCE, not deep-copied. A
        # SimulationState is write-once: step_forward builds a fresh one per step
        # and seek/restore always copy out of it, so no stored entry is ever
        # mutated in place. Only the list itself changes (append, and the
        # front-trim in _append_history_state), and a shallow list copy undoes
        # both. Deep-copying here made every snapshot O(len(history) * n_nodes):
        # the runner takes one snapshot per step for the adaptive-dt rollback, so
        # a 3M-node run was copying gigabytes per step and eventually died with a
        # MemoryError, having spent most of its wall time in memcpy.
        return PreparedSimulationSnapshot(
            z=np.asarray(self.z, dtype=float).copy(),
            history=list(self.history),
            history_index=int(self.history_index),
            pid_states=self._pid_state_snapshot(),
            controller_integrators=dict(self.controller_integrators),
            controller_y_prev=dict(self.controller_y_prev),
            controller_dTdt_hat_by_sensor=dict(self.controller_dTdt_hat_by_sensor),
            controller_error_prev=dict(self.controller_error_prev),
            controller_mode=str(self.controller_mode),
            controller_weighted_rms_error=self.controller_weighted_rms_error,
            controller_warnings=list(self.controller_warnings),
            controller_last_power_by_heater=dict(self.controller_last_power_by_heater),
            controller_allocator_diagnostics=dict(self.controller_allocator_diagnostics),
        )

    def restore_state(self, snapshot: PreparedSimulationSnapshot) -> None:
        self.z = np.asarray(snapshot.z, dtype=float).copy()
        # Shallow, for the same write-once reason as snapshot_state. Entries the
        # step trimmed off the front are still referenced by the snapshot list,
        # so a rollback restores them intact.
        self.history = list(snapshot.history)
        self.history_index = int(snapshot.history_index)
        self._restore_pid_state_snapshot(snapshot.pid_states)
        self.controller_integrators = dict(snapshot.controller_integrators)
        self.controller_y_prev = dict(snapshot.controller_y_prev)
        self.controller_dTdt_hat_by_sensor = dict(snapshot.controller_dTdt_hat_by_sensor)
        self.controller_error_prev = dict(snapshot.controller_error_prev)
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
        refresh_start = time.perf_counter()
        self._refresh_temperature_dependent_operator()
        _record_profile_ms(profile, "property_rebuild_ms", refresh_start)
        solve_start = time.perf_counter()
        if self.dynamic_heater_inputs:
            self._step_dynamic_heater_inputs(profile)
        else:
            self._advance_with_power_vector(np.zeros(len(self.node_ids), dtype=float), profile)
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
        limit = self._effective_history_limit()
        if limit > 0 and len(self.history) > limit:
            overflow = len(self.history) - limit
            del self.history[:overflow]
        self.history_index = len(self.history) - 1

    def _effective_history_limit(self) -> int:
        """``simulation_history_limit`` clamped to a memory budget.

        The configured limit counts STEPS, but an entry costs 8 bytes per node,
        so the same setting is a few MB on a lab-scale graph and gigabytes on a
        multi-million-cell one (360 steps x 3M nodes = 8.2 GB). Scrub-back depth
        is a convenience; running out of memory mid-run is not, so the node count
        gets a vote."""
        limit = max(0, int(getattr(self.params, "simulation_history_limit", 0)))
        budget_MB = float(getattr(self.params, "simulation_history_memory_budget_MB", 0.0) or 0.0)
        if budget_MB <= 0.0:
            return limit
        bytes_per_state = max(1, len(self.node_ids)) * 8
        budget_states = max(2, int((budget_MB * 1024.0 * 1024.0) // bytes_per_state))
        if limit <= 0:
            capped = budget_states
        elif limit <= budget_states:
            return limit
        else:
            capped = budget_states
        if not self._history_limit_warned:
            self._history_limit_warned = True
            self.warnings.append(
                f"Replay history capped at {capped} steps (~{budget_MB:g} MB) instead of "
                f"{limit if limit > 0 else 'unlimited'}: {len(self.node_ids)} nodes cost "
                f"{bytes_per_state / 1024.0 / 1024.0:.1f} MB per stored step. Raise "
                f"simulation_history_memory_budget_MB to scrub back further."
            )
        return capped

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
        if keep_cryocoolers_active:
            powers -= _cryocooler_power_vector(
                self.model,
                self.node_ids,
                self.temperatures_K,
                self.params,
                node_index_by_id=node_index,
                cryocooler_devices=self.cryocooler_devices,
                lift_curve=self.cryocooler_lift_curve,
                diagnostics_out=self.last_cryocooler_diagnostics,
                capacitance=self._live_capacitance(),
            )
        self._refresh_temperature_dependent_operator()
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
        self.controller_last_power_by_heater = dict(state.controller_last_power_by_heater)
        self.controller_mode = state.controller_mode
        self._sync_gpu_state()
        return state

    def _sync_gpu_state(self) -> None:
        # The implicit steppers are stateless between steps; nothing to sync.
        return None

    def _step_dynamic_heater_inputs(self, profile: dict[str, float] | None = None) -> None:
        if self.model is None or self.inv_C is None or self.A is None or self.base_b is None:
            return
        mode_start = time.perf_counter()
        use_mimo = _mimo_controller_is_active(self.model, self.heater_node_ids, self.params) or self._modal_scheme_active()
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
                cryocooler_devices=self.cryocooler_devices,
                cryocooler_lift_curve=self.cryocooler_lift_curve,
                cryocooler_diagnostics=self.last_cryocooler_diagnostics,
                capacitance=self._live_capacitance(),
            )
            _record_profile_ms(profile, "controller_heater_power_ms", controller_start)
        self._advance_with_power_vector(heater_power, profile)

    def _refresh_temperature_dependent_operator(
        self, evaluation_temperatures: np.ndarray | None = None
    ) -> None:
        """Rebuild C(T)/L(T)/A from a temperature (semi-implicit lag).

        No-op unless temperature-dependent properties are enabled. Updates the
        shared operator (inv_C, A) and every active stepper in place, clearing
        the implicit stepper's stage-matrix cache since the operator changed.

        By default the properties are evaluated at the current (step-start)
        temperature; pass ``evaluation_temperatures`` to lag them at a predicted
        midpoint instead (see midpoint property coupling in the step advance).
        """
        operator = self.temperature_dependent_operator
        if operator is None:
            return
        temps = (
            self.temperatures_K
            if evaluation_temperatures is None
            else np.asarray(evaluation_temperatures, dtype=float).reshape(-1)
        )
        C, inv_C, L = operator.rebuild(temps)
        C = _regularize_capacitance(np.asarray(C, dtype=float).reshape(-1), self.params)
        inv_C = np.where(C > 0.0, 1.0 / np.where(C > 0.0, C, 1.0), 0.0)
        self.inv_C = inv_C
        L_sparse = csr_matrix(L)
        # Preserve the operator's density: small graphs use the dense expm path.
        if issparse(self.A):
            A = -(diags(inv_C, format="csr") @ L_sparse)
        else:
            A = -(np.asarray(inv_C, dtype=float)[:, None] * L_sparse.toarray())
        self.A = A
        L_csr = csr_matrix(L)
        A_csr = csr_matrix(A)
        for stepper in (self.sparse_implicit_stepper, self.gpu_implicit_stepper):
            if stepper is None:
                continue
            if stepper.solver == "cg":
                stepper.set_operator(L_csr, C)
            else:
                stepper.set_operator(A_csr, None)

    def _advance_with_power_vector(self, heater_power: np.ndarray, profile: dict[str, float] | None = None) -> None:
        """Advance one step with the implicit solver (GPU if available, else CPU)."""
        if self.inv_C is None or self.base_b is None or self.sparse_implicit_stepper is None:
            return
        source_start = time.perf_counter()
        temperatures = np.asarray(self.temperatures_K, dtype=float).reshape(-1)
        powers = np.asarray(heater_power, dtype=float).reshape(-1)
        if powers.shape != temperatures.shape:
            raise ValueError(f"Heater power vector length {powers.shape} does not match temperatures {temperatures.shape}.")
        # Midpoint property/radiation coupling: evaluate the lagged nonlinear
        # terms (temperature-dependent C/L and radiation) at a forward-Euler
        # midpoint T + 0.5*dt*f(T) rather than the step-start temperature. This
        # makes the operator splitting second-order in dt for those terms. The
        # operator entering _refresh here is still at the step-start temperature
        # (refreshed by the caller), so _thermal_rhs gives the f(T) predictor.
        if self._midpoint_property_coupling_active():
            rhs = self._thermal_rhs(temperatures, powers)
            midpoint = temperatures + 0.5 * float(self.params.dt_s) * rhs
            np.clip(midpoint, 1.0e-3, None, out=midpoint)
            self._refresh_temperature_dependent_operator(evaluation_temperatures=midpoint)
            radiation_source = self._radiation_source_vector(midpoint)
        else:
            radiation_source = self._radiation_source_vector()
        _record_profile_ms(profile, "radiation_source_ms", source_start)
        source = (
            np.asarray(self.base_b, dtype=float).reshape(-1)
            + np.asarray(self.inv_C, dtype=float).reshape(-1) * powers
            + radiation_source
        )
        _record_profile_ms(profile, "source_vector_build_ms", source_start)
        step_start = time.perf_counter()
        temperatures_next = self._run_implicit_step(temperatures, source, profile)
        _record_profile_ms(profile, "implicit_step_ms", step_start)
        self.z = np.concatenate([temperatures_next, np.array([1.0])])

    # Max timestep subdivision when recovering from a stiff/ill-conditioned step:
    # substeps are doubled each retry, so this caps the finest subdivision at
    # 2**N x the adaptive count (N=8 -> up to 256x) before giving up.
    _MAX_STEP_SUBDIVISIONS = 8

    def _physical_step_check_mask(self) -> np.ndarray | None:
        """Boolean mask (over node_ids) of REAL body cells for the step-sanity check.
        Heater/marker nodes are excluded: they carry artificial tiny capacitance and
        physically meaningless temperatures (they deposit into real cells), so their
        explicit-source overshoot must not trigger the conduction-stiffness recovery.
        None when there is no model (check every entry)."""
        cached = getattr(self, "_physical_step_mask_cache", None)
        if cached is not None:
            return cached
        if self.model is None:
            return None
        mask = np.array(
            [not bool(getattr(self.model.nodes.get(int(node_id)), "is_heater", False)) for node_id in self.node_ids],
            dtype=bool,
        )
        self._physical_step_mask_cache = mask
        return mask

    @staticmethod
    def _implicit_step_is_physical(previous: np.ndarray, result: Any, mask: np.ndarray | None = None) -> bool:
        """Reject a step whose result is non-finite, drives an (absolute) temperature
        negative, or changes any cell by an implausibly large amount. Catches the
        case where the linear solver reports convergence on the residual but the
        ill-conditioned stage matrix yields a solution with enormous error. Only the
        masked (real body) cells are policed -- heater/marker nodes are ignored."""
        if result is None:
            return False
        candidate = np.asarray(result, dtype=float).reshape(-1)
        reference = np.asarray(previous, dtype=float).reshape(-1)
        if candidate.shape != reference.shape:
            return False
        if mask is not None and mask.shape == candidate.shape:
            candidate = candidate[mask]
            reference = reference[mask]
        if candidate.size == 0:
            return bool(np.all(np.isfinite(candidate)))
        if not bool(np.all(np.isfinite(candidate))):
            return False
        if float(np.min(candidate)) < -1.0:  # absolute temperature cannot go negative
            return False
        # Nothing physical moves by 10,000 K in a single step; larger => solver garbage.
        return float(np.max(np.abs(candidate - reference))) < 1.0e4

    def _run_implicit_step(
        self,
        temperatures: np.ndarray,
        source: np.ndarray,
        profile: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Run one implicit TR-BDF2 step (GPU backend first, CPU fallback), robustly.

        Stiff deep-cryo cells (tiny C(T), large k(T)) make the stage matrix
        diag(C)+alpha*L badly ill-conditioned, so the Jacobi-preconditioned CG can
        fail to converge -- or worse, meet its residual tolerance while returning a
        solution with huge error that detonates the run. On a solver failure OR a
        non-physical result we subdivide the timestep (smaller h shrinks alpha, so
        the stage matrix approaches the well-conditioned diag(C)) and retry with
        more substeps, up to _MAX_STEP_SUBDIVISIONS doublings."""
        mask = self._physical_step_check_mask()
        temp_floor = float(getattr(self.params, "implicit_temperature_floor_K", 0.0) or 0.0)
        temp_ceiling = float(getattr(self.params, "implicit_temperature_ceiling_K", 0.0) or 0.0)
        min_substeps = 1
        last_error: Exception | None = None
        best_finite: np.ndarray | None = None  # finest finite attempt, for best-effort fallback
        best_backend: tuple[SparseImplicitStepper, str] | None = None
        for attempt in range(int(self._MAX_STEP_SUBDIVISIONS) + 1):
            result: np.ndarray | None = None
            backend_stepper: SparseImplicitStepper | None = None
            backend_name = "cpu"
            if self.gpu_implicit_stepper is not None:
                try:
                    result = self.gpu_implicit_stepper.step(temperatures, source, min_substeps=min_substeps)
                    backend_stepper = self.gpu_implicit_stepper
                    backend_name = "gpu"
                except Exception as exc:  # noqa: BLE001 - fall back to CPU on any GPU error
                    self.warnings.append(f"GPU implicit step failed; falling back to CPU implicit solver: {exc}")
                    self.gpu_implicit_stepper = None
                    result = None
            if result is None and self.sparse_implicit_stepper is not None:
                try:
                    result = self.sparse_implicit_stepper.step(temperatures, source, min_substeps=min_substeps)
                    backend_stepper = self.sparse_implicit_stepper
                    backend_name = "cpu"
                except Exception as exc:  # noqa: BLE001 - subdivide and retry below
                    last_error = exc
                    result = None
            # Positivity clamp: a residual solver error on the ill-conditioned stiff
            # system can drive cells below zero (unphysical). Clamp up to the floor
            # BEFORE the physical check so the common cryo over-cool overshoot is
            # neutralized cheaply here, rather than doom-looping the substep retry
            # (which cannot help a constant cooling source anyway). A genuine
            # positive runaway (huge upward jump) still fails the check and subdivides.
            if result is not None and temp_floor > 0.0:
                result = np.maximum(np.asarray(result, dtype=float), temp_floor)
            if result is not None and temp_ceiling > 0.0:
                result = np.minimum(np.asarray(result, dtype=float), temp_ceiling)
            if result is not None and bool(np.all(np.isfinite(np.asarray(result, dtype=float)))):
                best_finite = np.asarray(result, dtype=float)
                best_backend = (backend_stepper, backend_name) if backend_stepper is not None else None
            if result is not None and self._implicit_step_is_physical(temperatures, result, mask):
                if backend_stepper is not None:
                    self._record_implicit_profile(backend_stepper, backend_name, profile)
                if attempt > 0:
                    self.warnings.append(
                        f"Implicit step subdivided {min_substeps}x to stay stable through stiff cryogenic cells."
                    )
                return result
            min_substeps = max(2, min_substeps * 2)  # subdivide and retry
        # Subdivision did not recover a fully physical step. Never raise (that would
        # regress vs. the old behaviour): apply the best finite result we found, or
        # hold the previous temperatures if nothing was even finite.
        if best_finite is not None:
            if best_backend is not None:
                self._record_implicit_profile(best_backend[0], best_backend[1], profile)
            self.warnings.append(
                f"Implicit step did not fully stabilise after {min_substeps}x subdivision "
                f"(stiff cryogenic cells); applying best-effort result. last solver error: {last_error}"
            )
            return best_finite
        self.warnings.append(
            "Implicit step produced no finite solution even after subdivision; holding previous temperatures. "
            f"last solver error: {last_error}"
        )
        return np.asarray(temperatures, dtype=float).copy()

    @staticmethod
    def _record_implicit_profile(
        stepper: "SparseImplicitStepper",
        backend: str,
        profile: dict[str, float] | None,
    ) -> None:
        if profile is None:
            return
        profile["implicit_backend_gpu"] = 1.0 if backend == "gpu" else 0.0
        profile["implicit_iterations"] = float(stepper.last_iterations)
        profile["implicit_substeps"] = float(stepper.last_substeps)
        profile["implicit_residual_norm"] = float(stepper.last_residual_norm)
        profile["implicit_relative_residual"] = float(stepper.last_relative_residual_norm)
        profile["implicit_predicted_delta_K"] = float(stepper.last_predicted_delta_K)

    def _radiation_is_active(self) -> bool:
        if self.inv_C is None or not self.params.use_ambient_radiation:
            return False
        coeff = self.radiation_coeff_W_K4
        if coeff is not None and bool(np.any(np.asarray(coeff, dtype=float) > 0.0)):
            return True
        return self.radiation_exchange_W is not None or self.radiation_super_S is not None

    def _midpoint_property_coupling_active(self) -> bool:
        """Midpoint coupling only helps (and only costs) when a lagged nonlinear
        term is present: temperature-dependent properties or active radiation."""
        if not bool(getattr(self.params, "use_midpoint_property_coupling", True)):
            return False
        if self.A is None or self.base_b is None or self.inv_C is None:
            return False
        return self.temperature_dependent_operator is not None or self._radiation_is_active()

    def _environment_temperature_K(self) -> np.ndarray:
        env = self.environment_temperature_K
        if env is None:
            return np.full(len(self.node_ids), float(self.params.T_env_K), dtype=float)
        return np.asarray(env, dtype=float).reshape(-1)

    def _radiation_source_vector(self, temperatures_K: np.ndarray | None = None) -> np.ndarray:
        if not self._radiation_is_active():
            return np.zeros(len(self.node_ids), dtype=float)
        temperatures = (
            np.asarray(self.temperatures_K, dtype=float).reshape(-1)
            if temperatures_K is None
            else np.asarray(temperatures_K, dtype=float).reshape(-1)
        )
        radiation_power = np.zeros_like(temperatures)
        # Exchange to the (per-node) background sink; coeff already includes sigma.
        coeff = self.radiation_coeff_W_K4
        if coeff is not None:
            coeff = np.asarray(coeff, dtype=float).reshape(-1)
            if np.any(coeff > 0.0):
                env4 = self._environment_temperature_K()**4
                radiation_power = radiation_power + coeff * (env4 - temperatures**4)
        # Surface-to-surface exchange: sigma*(W @ T^4 - degree * T^4), a Laplacian
        # on U = T^4. Conserves energy across the body (rows sum to zero).
        if self.radiation_exchange_W is not None:
            u = temperatures**4
            coupled = np.asarray(self.radiation_exchange_W @ u, dtype=float).reshape(-1)
            degree = np.asarray(self.radiation_exchange_degree, dtype=float).reshape(-1)
            radiation_power = radiation_power + STEFAN_BOLTZMANN_W_M2K4 * (coupled - degree * u)
        # Factored grouped exchange: aggregate T^4 to super-surfaces, small super
        # Laplacian, distribute back (S^T). Energy-conserving (rows of S sum to 1).
        if self.radiation_super_S is not None:
            u = temperatures**4
            aggregated = np.asarray(self.radiation_super_S @ u, dtype=float).reshape(-1)
            super_degree = np.asarray(self.radiation_super_degree, dtype=float).reshape(-1)
            super_power = np.asarray(self.radiation_super_W @ aggregated, dtype=float).reshape(-1) - super_degree * aggregated
            radiation_power = radiation_power + STEFAN_BOLTZMANN_W_M2K4 * np.asarray(
                self.radiation_super_S.T @ super_power, dtype=float
            ).reshape(-1)
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
            cryocooler_devices=self.cryocooler_devices,
            cryocooler_lift_curve=self.cryocooler_lift_curve,
            cryocooler_diagnostics=self.last_cryocooler_diagnostics,
        ) if not (_mimo_controller_is_active(self.model, self.heater_node_ids, self.params) or self._modal_scheme_active()) else self._mimo_controller_power_vector(update_state=False)
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
            cryocooler_devices=self.cryocooler_devices,
            lift_curve=self.cryocooler_lift_curve,
            diagnostics_out=self.last_cryocooler_diagnostics,
        )
        node_index = self.node_index_by_id or {int(node_id): row for row, node_id in enumerate(self.node_ids)}
        result: dict[int, float] = {}
        receiving_node_ids = {
            int(node_id)
            for device in self.cryocooler_devices
            for node_id in device.receiving_node_ids
        }
        for node_id in sorted(receiving_node_ids):
            row = node_index.get(int(node_id))
            if row is not None:
                result[int(node_id)] = float(powers[int(row)])
        return result

    def power_balance_W(self) -> dict[str, float]:
        """Instantaneous power flows at the current state, in watts.

        heater_W: total heater injection; cryocooler_W: total cryocooler removal;
        radiation_W: net radiative power INTO the system (positive when the body
        is colder than ambient, i.e. a parasitic load); net_W ~ dU/dt. In a
        cryocooled+heater regime these are not expected to balance.
        """
        heater_W = (
            float(sum(self.heater_actuator_power_by_node().values())) if self.heater_node_ids else 0.0
        )
        cryocooler_W = float(sum(self.cryocooler_power_by_node().values())) if self.cryocooler_devices else 0.0
        radiation_W = 0.0
        if self.radiation_coeff_W_K4 is not None and self.params.use_ambient_radiation:
            coeff = np.asarray(self.radiation_coeff_W_K4, dtype=float).reshape(-1)
            if np.any(coeff > 0.0):
                temperatures = np.asarray(self.temperatures_K, dtype=float).reshape(-1)
                env4 = self._environment_temperature_K() ** 4
                radiation_W = float(np.sum(coeff * (env4 - temperatures**4)))
        return {
            "heater_W": heater_W,
            "cryocooler_W": cryocooler_W,
            "radiation_W": radiation_W,
            "net_W": heater_W + radiation_W - cryocooler_W,
        }

    def cryocooler_diagnostics(self) -> list[dict[str, Any]]:
        if self.model is None:
            return []
        _cryocooler_power_vector(
            self.model,
            self.node_ids,
            self.temperatures_K,
            self.params,
            node_index_by_id=self.node_index_by_id,
            cryocooler_devices=self.cryocooler_devices,
            lift_curve=self.cryocooler_lift_curve,
            diagnostics_out=self.last_cryocooler_diagnostics,
        )
        return [dict(item) for item in self.last_cryocooler_diagnostics]

    def heater_actuator_power_by_node(self, *, disable_mimo_controller: bool = False) -> dict[int, float]:
        if self.model is None:
            return {}
        if (_mimo_controller_is_active(self.model, self.heater_node_ids, self.params) or self._modal_scheme_active()) and not disable_mimo_controller:
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

    def _dc_gain_feedforward(self, sensor_ids, heater_ids, errors) -> np.ndarray | None:
        """Steady-state setpoint-reach feedforward from the EXACT full-plant DC gain
        (stored in the modal controller artifact as dc_gain_pinv): u_ff = G_pinv ·
        (setpoint - measured) over the controlled sensors, mapped onto the active
        heaters. Returns None if no artifact is loaded (then the caller keeps its
        existing behavior). Well-conditioned, so commands stay physical."""
        mc = self._load_modal_controller()
        if mc is None or mc.get("dc_pinv") is None:
            return None
        dc_pinv = mc["dc_pinv"]
        ctrl_sensor_ids = [int(mc["sensor_ids"][i]) for i in mc["ctrl_idx"]]
        error_by_sensor = {int(s): float(e) for s, e in zip(sensor_ids, errors)}
        error_ctrl = np.array([error_by_sensor.get(sid, 0.0) for sid in ctrl_sensor_ids], dtype=float)
        command_by_heater = {
            int(h): float(c) for h, c in zip(mc["heater_ids"], dc_pinv @ error_ctrl)
        }
        return np.array([command_by_heater.get(int(h), 0.0) for h in heater_ids], dtype=float)

    def _modal_scheme_active(self) -> bool:
        """True when the reduced-model LQR controller should run instead of the
        PID+QP allocator: scheme selected, heater-input mode, and a usable modal
        controller artifact is loaded for this graph."""
        if str(getattr(self.params, "mimo_controller_scheme", "pid_qp")) != "modal_lqr":
            return False
        if self.params.input_mode != "heater_inputs":
            return False
        return self._load_modal_controller() is not None

    def _load_modal_controller(self):
        """Load + cache the modal controller artifact (K, E_reg, Nx, Nu, node ids,
        operating point) and validate it maps onto this graph. Returns a dict or
        None (falls back to PID+QP). Cached per prepared simulation."""
        path = str(getattr(self.params, "modal_controller_path", "") or "")
        cache = getattr(self, "_modal_controller_cache", None)
        if cache is not None and getattr(self, "_modal_controller_cache_path", None) == path:
            return cache or None  # {} means "tried this path and it was unavailable"
        self._modal_controller_cache_path = path
        if not path or self.model is None:
            self._modal_controller_cache = {}
            return None
        try:
            data = np.load(path, allow_pickle=False)
            heater_ids = [int(v) for v in data["heater_ids"]]
            sensor_ids = [int(v) for v in data["sensor_ids"]]
            monitor = np.asarray(data["monitor"], dtype=bool)
            missing = [h for h in heater_ids if h not in self.model.nodes] + [
                s for s in sensor_ids if s not in self.model.nodes
            ]
            if missing:
                raise ValueError(
                    f"modal controller references {len(missing)} node id(s) not in this graph "
                    "(built for a different graph?)"
                )
            cache = {
                "K": np.asarray(data["K"], dtype=float),
                "E": np.asarray(data["E_reg"], dtype=float),
                "Nx": np.asarray(data["Nx"], dtype=float),
                "Nu": np.asarray(data["Nu"], dtype=float),
                # Feedforward from the EXACT full-plant DC gain (well-conditioned);
                # the reduced-model feedforward (Nx/Nu) is ill-conditioned at the
                # steady state and only used as a fallback for old artifacts.
                "dc_pinv": np.asarray(data["dc_gain_pinv"], dtype=float) if "dc_gain_pinv" in data else None,
                "heater_ids": heater_ids,
                "sensor_ids": sensor_ids,
                "ctrl_idx": np.where(~monitor)[0],
                "T_op": float(data["T_op_K"]),
            }
            self._modal_controller_cache = cache
            return cache
        except Exception as exc:  # noqa: BLE001 - controller is optional
            self.warnings.append(f"Modal controller unavailable ({exc}); using PID+QP allocator.")
            self._modal_controller_cache = {}
            return None

    def _modal_controller_power_vector(self, update_state: bool) -> np.ndarray:
        """Reduced-model output-feedback controller: regularized static state
        estimate x_hat = E_reg (y - T_op), then u = u_ff - K (x_hat - x_ff) + u_int,
        clamped to [0, u_max] and slew-rate limited. Both the estimate x_hat and
        the target x_ff are formed with the well-conditioned estimator E_reg (x_ff
        from the setpoint deviations), so the feedback is a pure tracking-error
        regulator; u_ff comes from the EXACT plant DC gain. The integral term is
        offset-free and supplies the operating holding power (which the linearized
        model excludes). Cryocoolers and manual (non-MIMO) heaters are applied
        first as the baseline."""
        model = self.model
        node_index = self.node_index_by_id or {int(v): i for i, v in enumerate(self.node_ids)}
        # Baseline: cryocoolers + any manual/non-MIMO heaters.
        powers = _controlled_heater_power_vector(
            model, self.node_ids, self.temperatures_K, float(self.params.dt_s), self.params,
            include_heater_inputs=True, update_pid_state=False, excluded_modes={"mimo"},
            node_index_by_id=node_index, heater_node_ids=self.heater_node_ids,
            cryocooler_node_ids=self.cryocooler_node_ids, cryocooler_devices=self.cryocooler_devices,
            cryocooler_lift_curve=self.cryocooler_lift_curve, cryocooler_diagnostics=self.last_cryocooler_diagnostics,
            capacitance=self._live_capacitance(),
        )
        mc = self._load_modal_controller()
        if mc is None:
            return powers
        temps = np.asarray(self.temperatures_K, dtype=float)
        T_op = mc["T_op"]
        heater_ids = mc["heater_ids"]
        sensor_ids = mc["sensor_ids"]
        ctrl = mc["ctrl_idx"]
        # Measured sensor deviations from the operating point.
        y = np.array(
            [sensor_readout_temperature_K(model, node_index, temps, int(sid)) for sid in sensor_ids],
            dtype=float,
        )
        y_dev = np.where(np.isfinite(y), y, T_op) - T_op
        x_hat = mc["E"] @ y_dev
        setpoints = np.array(
            [float(getattr(model.nodes[int(sensor_ids[i])], "controller_setpoint_K", T_op)) for i in ctrl],
            dtype=float,
        )
        r_sp = setpoints - T_op
        # Target reduced state from the SAME regularized estimator E, NOT the
        # reduced-model Nx DC map. Nx is ill-conditioned at DC (truncated fast
        # modes carry the near-field conduction), so Nx @ r extrapolates wildly
        # for setpoints far from T_op and made K @ x_ff explode (~365 W/heater at
        # a +10 K setpoint -> solver divergence). Building x_ff = E @ r_full uses
        # the well-conditioned estimator, so K @ (x_hat - x_ff) becomes a clean
        # tracking-error regulator: controlled sensors track their setpoint and
        # uncontrolled sensors default to their current measurement (zero error),
        # which vanishes at steady state leaving u = u_ff (exact DC) + integral.
        r_full = y_dev.copy()
        r_full[ctrl] = r_sp
        x_ff = mc["E"] @ r_full
        # Steady-state feedforward and integral direction from the EXACT full-plant
        # DC gain (dc_pinv) when available -- the reduced-model map (Nu) is
        # ill-conditioned at DC and drives huge, non-transferable commands.
        ff_map = mc["dc_pinv"] if mc.get("dc_pinv") is not None else mc["Nu"]
        u_max = np.array(
            [max(0.0, _controller_heater_max_power(model.nodes[int(h)], self.params)) for h in heater_ids],
            dtype=float,
        )
        # Adaptive feedforward: fold the learned correction dM (accumulated from
        # prior steady-state samples) into the feedforward. dM starts at zero over
        # the model prior, so with learning off -- or before the first sample --
        # this is identical to the fixed exact-DC-gain feedforward. The correction
        # is projected to +/- frac * u_max per heater so a bad sample can never
        # drive the feedforward unphysical; the effective command is clamped to
        # [0, u_max] downstream regardless.
        adaptive_ff = bool(getattr(self.params, "modal_adaptive_ff_enabled", False))
        dM = getattr(self, "controller_modal_ff_correction", None)
        if dM is None or np.asarray(dM).shape != ff_map.shape:
            dM = np.zeros_like(ff_map)
        if adaptive_ff:
            frac = max(0.0, float(getattr(self.params, "modal_adaptive_ff_max_correction_frac", 1.0)))
            ff_correction = np.clip(dM @ r_sp, -frac * u_max, frac * u_max)
        else:
            ff_correction = np.zeros(len(heater_ids), dtype=float)
        u_ff = ff_map @ r_sp + ff_correction
        dt = float(self.params.dt_s)
        ki = max(0.0, float(getattr(self.params, "modal_integral_gain", 0.0)))
        u_int = getattr(self, "controller_modal_integral", None)
        if u_int is None or np.asarray(u_int).shape[0] != len(heater_ids):
            u_int = np.zeros(len(heater_ids), dtype=float)
        error = r_sp - y_dev[ctrl]
        increment = ki * (ff_map @ error) * dt
        candidate_int = u_int + increment
        base = u_ff - mc["K"] @ (x_hat - x_ff)
        # Global controller limits (shared with the PID+QP scheme): absolute
        # per-heater power clamp, then a hard per-step slew-rate limit. These
        # cap the transient a large-signal setpoint step can inject into the
        # low-heat-capacity deposition nodes.
        u_cmd = np.clip(base + candidate_int, 0.0, u_max)
        u_prev = np.array(
            [float(self.controller_last_power_by_heater.get(int(h), 0.0)) for h in heater_ids],
            dtype=float,
        )
        slew = max(0.0, float(getattr(self.params, "mimo_heater_slew_rate_W_per_s", 0.0)))
        if slew > 0.0:
            max_delta = slew * dt
            u = np.clip(u_cmd, u_prev - max_delta, u_prev + max_delta)
            u = np.clip(u, 0.0, u_max)
        else:
            u = u_cmd
        self.controller_allocator_diagnostics = {
            "controller_scheme": "modal_lqr",
            "heater_ids": [int(h) for h in heater_ids],
            "heater_commands_W": [float(c) for c in u],
            "reduced_order": int(mc["K"].shape[1]),
            "active_sensor_count": int(len(ctrl)),
            "rate_command_norm": 0.0,
            "slew_rate_limit_W_per_s": float(slew),
            "heater_max_power_W": [float(v) for v in u_max],
            "adaptive_ff_enabled": bool(adaptive_ff),
            "adaptive_ff_correction_norm": float(np.linalg.norm(dM)),
            "adaptive_ff_command_W": [float(v) for v in ff_correction],
        }
        if update_state:
            # Anti-windup keys off the absolute saturation clamp (pre-slew), so a
            # transient slew limit does not permanently freeze the integrator.
            at_upper = u_cmd >= u_max - 1.0e-9
            at_lower = u_cmd <= 1.0e-9
            freeze = (at_upper & (increment > 0.0)) | (at_lower & (increment < 0.0))
            committed = np.clip(np.where(freeze, u_int, candidate_int), -u_max, u_max)
            # Adaptive feedforward learning: if this operating point is steady, take
            # one RLS sample -- transfer the integral's holding authority into dM
            # (bumpless), so the correction is available as feedforward next time.
            if adaptive_ff:
                committed, dM, learned, alpha = self._adaptive_ff_learn(
                    committed=committed, dM=dM, r_sp=r_sp, y_dev=y_dev, ctrl=ctrl,
                    error=error, u_max=u_max, u_cmd=u_cmd, dt=dt,
                )
                self.controller_modal_ff_correction = dM
                self.controller_allocator_diagnostics["adaptive_ff_updated"] = bool(learned)
                self.controller_allocator_diagnostics["adaptive_ff_transfer_alpha"] = float(alpha)
                self.controller_allocator_diagnostics["adaptive_ff_correction_norm"] = float(np.linalg.norm(dM))
            self._modal_ff_prev_y_dev = y_dev.copy()
            self.controller_modal_integral = committed
            self.controller_weighted_rms_error = float(np.sqrt(np.mean(error**2))) if error.size else 0.0
            self.controller_warnings = []
            self.controller_last_power_by_heater = {
                int(h): float(c) for h, c in zip(heater_ids, u)
            }
        for heater_id, command in zip(heater_ids, u):
            _deposit_heater_command_power(powers, model, node_index, int(heater_id), float(command))
        return powers

    def _adaptive_ff_learn(
        self, *, committed, dM, r_sp, y_dev, ctrl, error, u_max, u_cmd, dt,
    ):
        """Take one adaptive-feedforward RLS sample IF this operating point is a
        valid steady-state hold, otherwise a no-op.

        Gates (all must pass): a nonzero setpoint deviation (a ~zero setpoint
        carries no information); every controlled sensor settled in both tracking
        error and rate (so we never regress transient data into a static map -- a
        settled hold also implies no meaningful saturation, since a saturated
        heater could not be holding the setpoint); and the integral within a sane
        multiple of actuator range (outlier guard).

        Returns ``(committed, dM, learned, alpha)``. On an accepted sample the
        returned ``committed`` has the transferred authority removed (bumpless) and
        ``dM`` is the updated correction matrix."""
        prev = getattr(self, "_modal_ff_prev_y_dev", None)
        if prev is None or np.asarray(prev).shape != y_dev.shape:
            return committed, dM, False, 0.0
        if error.size == 0 or float(np.linalg.norm(r_sp)) <= 1.0e-9 or dt <= 0.0:
            return committed, dM, False, 0.0
        rate_tol = max(0.0, float(getattr(self.params, "modal_adaptive_ff_rate_tol_K_per_s", 1.0e-3)))
        error_tol = max(0.0, float(getattr(self.params, "modal_adaptive_ff_error_tol_K", 0.05)))
        rate = (y_dev[ctrl] - np.asarray(prev)[ctrl]) / dt
        settled = (
            float(np.max(np.abs(error))) <= error_tol
            and float(np.max(np.abs(rate))) <= rate_tol
        )
        if not settled:
            return committed, dM, False, 0.0
        if np.any(np.abs(committed) > 5.0 * np.maximum(u_max, 1.0e-12)):
            return committed, dM, False, 0.0
        forgetting = min(1.0, max(1.0e-3, float(getattr(self.params, "modal_adaptive_ff_forgetting", 0.999))))
        n = int(np.asarray(r_sp).shape[0])
        P = getattr(self, "_modal_ff_rls_P", None)
        if P is None or np.asarray(P).shape != (n, n):
            p0 = max(1.0e-9, float(getattr(self.params, "modal_adaptive_ff_p0", 1.0)))
            P = p0 * np.eye(n)
        P, dM, committed, alpha = _rls_ff_update(P, dM, np.asarray(r_sp, dtype=float), committed, forgetting)
        self._modal_ff_rls_P = P
        return committed, dM, True, float(alpha)

    def _mimo_controller_power_vector(self, update_state: bool) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(self.node_ids), dtype=float)
        if self._modal_scheme_active():
            return self._modal_controller_power_vector(update_state)
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
            cryocooler_devices=self.cryocooler_devices,
            cryocooler_lift_curve=self.cryocooler_lift_curve,
            cryocooler_diagnostics=self.last_cryocooler_diagnostics,
            capacitance=self._live_capacitance(),
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
            if _heater_controller_mode(heater, sensor) != "mimo":
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
        heaters_by_sensor: dict[int, list[int]] = {int(sensor_id): [] for sensor_id in sensor_ids}
        for heater_id in heater_ids:
            heater = self.model.nodes[int(heater_id)]
            sensor_id = getattr(heater, "assigned_sensor_id", None)
            if sensor_id is None and heater.is_sensor:
                sensor_id = int(heater_id)
            if sensor_id is not None and int(sensor_id) in heaters_by_sensor:
                heaters_by_sensor[int(sensor_id)].append(int(heater_id))
        y = np.array([float(readout) for _sensor_id, readout in valid_pairs], dtype=float)
        estimates = []
        for sensor_id, measured in zip(sensor_ids, y):
            settling_time = _heater_controller_average(
                self.model,
                heaters_by_sensor.get(int(sensor_id), []),
                "sensor_settling_time_s",
                0.0,
                sensor_id=int(sensor_id),
                clamp_min=0.0,
            )
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
                _heater_controller_average(
                    self.model,
                    heaters_by_sensor.get(int(sensor_id), []),
                    "controller_weight",
                    0.0,
                    sensor_id=int(sensor_id),
                    clamp_min=0.0,
                )
                for sensor_id in sensor_ids
            ],
            dtype=float,
        )
        rms = weighted_rms_error(errors, weights)
        self.controller_weighted_rms_error = rms
        mode_changed = self._update_controller_mode(rms) if update_state else False

        dt = max(float(self.params.dt_s), 1.0e-12)
        # Standard (integer-order) PID with recursive O(1) state -- no growing
        # error history, so it maps directly onto the microcontroller firmware.
        # On a mode change the integrator and derivative memory reset so gains
        # tuned for one mode do not carry into the other.
        integral_abs_max = max(0.0, float(getattr(self.params, "mimo_integral_abs_max", 1.0e6)))
        previous_integrators = {} if mode_changed else self.controller_integrators
        previous_error = {} if mode_changed else self.controller_error_prev
        integrator_prev = np.array(
            [float(previous_integrators.get(int(sensor_id), 0.0)) for sensor_id in sensor_ids],
            dtype=float,
        )
        # Recursive integral, rectangular rule: I_k = I_{k-1} + e_k * dt.
        eta = integrator_prev + errors * dt
        if integral_abs_max > 0.0:
            eta = np.clip(eta, -integral_abs_max, integral_abs_max)
        # Backward-difference derivative; zero on the first sample after a reset.
        has_prev_error = np.array(
            [int(sensor_id) in previous_error for sensor_id in sensor_ids], dtype=bool
        )
        error_prev = np.array(
            [float(previous_error.get(int(sensor_id), 0.0)) for sensor_id in sensor_ids],
            dtype=float,
        )
        error_derivative = np.where(has_prev_error, (errors - error_prev) / dt, 0.0)
        if self.controller_mode == "hold":
            kp_key = "controller_kp_hold"
            ki_key = "controller_ki_hold"
            kd_key = "controller_kd_hold"
        else:
            kp_key = "controller_kp_coarse"
            ki_key = "controller_ki_coarse"
            kd_key = "controller_kd_coarse"
        sensor_row_by_id = {int(sensor_id): row for row, sensor_id in enumerate(sensor_ids)}
        heater_sensor_rows = np.full(len(heater_ids), -1, dtype=int)
        per_heater_rate_cmd = np.zeros(len(heater_ids), dtype=float)
        for heater_col, heater_id in enumerate(heater_ids):
            heater = self.model.nodes[int(heater_id)]
            assigned_sensor_id = getattr(heater, "assigned_sensor_id", None)
            if assigned_sensor_id is None and heater.is_sensor:
                assigned_sensor_id = int(heater_id)
            row = sensor_row_by_id.get(int(assigned_sensor_id)) if assigned_sensor_id is not None else None
            if row is None:
                continue
            sensor = self.model.nodes[int(sensor_ids[int(row)])]
            try:
                kp = max(0.0, float(_heater_controller_value(heater, sensor, kp_key, 0.0)))
            except (TypeError, ValueError):
                kp = 0.0
            try:
                ki = max(0.0, float(_heater_controller_value(heater, sensor, ki_key, 0.0)))
            except (TypeError, ValueError):
                ki = 0.0
            try:
                kd = max(0.0, float(_heater_controller_value(heater, sensor, kd_key, 0.0)))
            except (TypeError, ValueError):
                kd = 0.0
            if _heater_controller_is_default(heater):
                inherited_count = max(1, len(heaters_by_sensor.get(int(sensor_ids[int(row)]), [])))
                kp /= float(inherited_count)
                ki /= float(inherited_count)
                kd /= float(inherited_count)
            heater_sensor_rows[int(heater_col)] = int(row)
            per_heater_rate_cmd[int(heater_col)] = (
                kp * float(errors[int(row)])
                + ki * float(eta[int(row)])
                + kd * float(error_derivative[int(row)])
            )
        v_cmd_unclipped = np.zeros(len(sensor_ids), dtype=float)
        for heater_col, row in enumerate(heater_sensor_rows):
            if int(row) >= 0:
                v_cmd_unclipped[int(row)] += float(per_heater_rate_cmd[int(heater_col)])
        v_cmd = v_cmd_unclipped.copy()
        v_abs_max = max(0.0, float(getattr(self.params, "mimo_v_cmd_abs_max_K_per_s", 0.25)))
        if v_abs_max > 0.0:
            v_cmd = np.clip(v_cmd, -v_abs_max, v_abs_max)
            for row, (raw, clipped) in enumerate(zip(v_cmd_unclipped, v_cmd)):
                if abs(float(raw)) <= float(v_abs_max) or abs(float(raw)) <= 1.0e-15:
                    continue
                per_heater_rate_cmd[heater_sensor_rows == int(row)] *= float(clipped) / float(raw)

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
        B_s = self._mimo_dynamic_gain_matrix(
            sensor_ids,
            heater_ids,
            node_index,
        )
        raw_dTdt, dTdt_hat = self._mimo_sensor_drift_estimate(sensor_ids, y, dt)
        # Steady-state setpoint feedforward from the exact plant DC gain (if the
        # graph has a modal controller artifact). When present it carries the reach,
        # so we DROP the passive-drift observer and let the PID handle the hold.
        dc_feedforward = self._dc_gain_feedforward(sensor_ids, heater_ids, errors)
        if dc_feedforward is not None:
            passive_dTdt = np.zeros(len(sensor_ids), dtype=float)
        elif bool(getattr(self.params, "mimo_passive_drift_from_measurement", True)):
            # Disturbance observer: the passive (non-MIMO) sensor drift is the
            # MEASURED sensor rate with the commanded-heater contribution removed.
            # Deployable on the MCU without a plant model; reactive by one step.
            passive_dTdt = dTdt_hat - B_s @ u_prev
        else:
            # Model-based oracle (simulation only): project the full thermal RHS
            # evaluated with MIMO heating excluded.
            passive_dTdt = self._mimo_baseline_sensor_drift(sensor_ids, node_index, powers)
        hold_result = allocate_thermal_rate_qp(
            B_s,
            passive_dTdt,
            np.zeros(len(sensor_ids), dtype=float),
            weights,
            maxima,
            np.zeros(len(heater_ids), dtype=float),
            0.0,
            0.0,
            None,
        )
        u_ff = np.asarray(hold_result.u, dtype=float).reshape(-1)
        u_ff = np.clip(np.where(np.isfinite(u_ff), u_ff, 0.0), 0.0, maxima)
        u_ref = u_ff.copy()
        if dc_feedforward is not None:
            # Bias the allocator toward the DC-gain reach power (clamped to limits).
            u_ref = np.clip(u_ref + dc_feedforward, 0.0, maxima)
        for heater_col, row in enumerate(heater_sensor_rows):
            if int(row) < 0:
                continue
            authority = float(B_s[int(row), int(heater_col)])
            if not np.isfinite(authority) or abs(authority) <= 1.0e-15:
                continue
            u_ref[int(heater_col)] += float(per_heater_rate_cmd[int(heater_col)]) / authority
        u_ref = np.clip(np.where(np.isfinite(u_ref), u_ref, 0.0), 0.0, maxima)
        drift_for_qp = passive_dTdt + B_s @ u_prev
        allocation = allocate_thermal_rate_qp(
            B_s,
            drift_for_qp,
            v_cmd,
            weights,
            maxima,
            u_prev,
            float(getattr(self.params, "mimo_lambda_u", 1.0e-3)),
            float(getattr(self.params, "mimo_rho_du", 0.0)),
            max_delta_power,
            u_ref,
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
        heater_total_dTdt = B_s @ u
        predicted_dTdt = passive_dTdt + heater_total_dTdt
        residual = predicted_dTdt - v_cmd
        gain_warnings = list((self.controller_dynamic_gain_cache or {}).get("warnings", ()))
        hold_warnings = [f"MIMO feedforward hold solve: {warning}" for warning in hold_result.warnings]
        self.controller_warnings = pair_warnings + gain_warnings + hold_warnings + list(allocation.warnings)
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
            "rate_command_unclipped_norm": float(np.linalg.norm(v_cmd_unclipped)),
            "heater_command_norm": float(np.linalg.norm(u)),
            "measured_drift_dTdt_norm": float(np.linalg.norm(dTdt_hat)),
            "passive_drift_dTdt_norm": float(np.linalg.norm(passive_dTdt)),
            "feedforward_hold_power_norm": float(np.linalg.norm(u_ff)),
            "predicted_dTdt_norm": float(np.linalg.norm(predicted_dTdt)),
            "predicted_dTdt_residual_norm": float(np.linalg.norm(residual)),
            "allocation_residual_norm": float(np.linalg.norm(residual)),
            "target_residual_norm": float(allocation.residual_norm),
            "bounds_active": bool(allocation.bounds_active),
            "solver_success": bool(allocation.solver_success),
            "solver_message": str(allocation.solver_message),
            "feedforward_solver_success": bool(hold_result.solver_success),
            "feedforward_solver_message": str(hold_result.solver_message),
            "feedforward_residual_norm": float(hold_result.residual_norm),
            "lambda_u": float(getattr(self.params, "mimo_lambda_u", 1.0e-3)),
            "rho_du": float(getattr(self.params, "mimo_rho_du", 0.0)),
            "slew_rate_limit_W_per_s": float(slew_rate),
            "slew_delta_limit_W": float(slew_rate * dt) if slew_rate > 0.0 else 0.0,
            "v_cmd_min_K_per_s": float(np.min(v_cmd)) if v_cmd.size else 0.0,
            "v_cmd_max_K_per_s": float(np.max(v_cmd)) if v_cmd.size else 0.0,
            "v_cmd_clipped": bool(v_abs_max > 0.0 and np.any(np.abs(v_cmd_unclipped - v_cmd) > 1.0e-12)),
            "raw_dTdt_s": [float(value) for value in raw_dTdt],
            "filtered_dTdt_hat_s": [float(value) for value in dTdt_hat],
            "passive_dTdt_s": [float(value) for value in passive_dTdt],
            "drift_for_qp_dTdt_s": [float(value) for value in drift_for_qp],
            "B_s": [[float(value) for value in row] for row in B_s],
            "B_s_source": list((self.controller_dynamic_gain_cache or {}).get("sources", [])),
            "average_inverse_C_s": list((self.controller_dynamic_gain_cache or {}).get("average_inverse_C_s", [])),
            "B_s_delta_u_dTdt_s": [float(value) for value in heater_delta_dTdt],
            "B_s_u_dTdt_s": [float(value) for value in heater_total_dTdt],
            "v_cmd_s": [float(value) for value in v_cmd],
            "v_cmd_unclipped_s": [float(value) for value in v_cmd_unclipped],
            "per_heater_rate_cmd_s": [float(value) for value in per_heater_rate_cmd],
            "predicted_dTdt_s": [float(value) for value in predicted_dTdt],
            "achieved_predicted_dTdt_s": [float(value) for value in predicted_dTdt],
            "residual_s": [float(value) for value in residual],
            "feedforward_hold_power_W": [float(value) for value in u_ff],
            "heater_reference_power_W": [float(value) for value in u_ref],
            "heater_commands_W": [float(value) for value in u],
            "u_prev_W": [float(value) for value in u_prev],
            "heater_at_lower_bound": [bool(value <= 1.0e-9) for value in u],
            "heater_at_upper_bound": [bool(value >= max(max_power - 1.0e-9, 0.0)) for value, max_power in zip(u, maxima)],
        }
        if update_state:
            # Conditional-integration anti-windup: if a sensor is saturated in the
            # direction it needs (all its heaters at 0 while too warm, or all at max
            # while too cold), discard this step's integral increment (keep I_{k-1}).
            committed_integrators = eta.copy()
            if bool(getattr(self.params, "mimo_freeze_integral_when_saturated", True)):
                for row in range(len(sensor_ids)):
                    relevant = np.abs(B_s[row, :]) > 1.0e-12
                    if not np.any(relevant):
                        continue
                    if errors[row] < 0.0 and bool(np.all(u[relevant] <= 1.0e-9)):
                        committed_integrators[row] = integrator_prev[row]
                    elif errors[row] > 0.0 and bool(np.all(u[relevant] >= np.maximum(maxima[relevant] - 1.0e-9, 0.0))):
                        committed_integrators[row] = integrator_prev[row]
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
        return powers

    def _mimo_baseline_sensor_drift(
        self,
        sensor_ids: list[int],
        node_index: dict[int, int],
        baseline_powers: np.ndarray,
    ) -> np.ndarray:
        if self.model is None or self.A is None or self.base_b is None or self.inv_C is None:
            return np.zeros(len(sensor_ids), dtype=float)
        try:
            rhs = self._thermal_rhs(self.temperatures_K, baseline_powers)
        except Exception as exc:
            self.controller_warnings.append(f"MIMO feedforward passive drift estimate failed: {exc}")
            return np.zeros(len(sensor_ids), dtype=float)
        values = [
            sensor_readout_temperature_K(self.model, node_index, rhs, int(sensor_id))
            for sensor_id in sensor_ids
        ]
        return np.where(np.isfinite(values), np.asarray(values, dtype=float), 0.0)

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
        gain_key: tuple[tuple[float, ...], ...]
        if self.model is None:
            gain_key = tuple()
        else:
            gain_key = tuple(
                tuple(float(self.model.controller_gain(int(sensor_id), int(heater_id))) for heater_id in heater_ids)
                for sensor_id in sensor_ids
            )
        key = (
            tuple(int(value) for value in sensor_ids),
            tuple(int(value) for value in heater_ids),
            connected_key,
            gain_key,
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
        sources: list[str] = []
        alpha = max(0.0, float(getattr(self.params, "heater_sensor_pair_alpha", 1.0)))
        for sensor_index, sensor_id in enumerate(sensor_ids):
            if self.model is None:
                average_inverse_C_s.append(0.0)
                sources.append("none")
                continue
            sensor = self.model.nodes[int(sensor_id)]
            average_inverse_C, valid_node_ids = average_inverse_capacitance_for_sensor(
                self.model,
                node_index,
                inv_C,
                int(sensor_id),
            )
            average_inverse_C_s.append(float(average_inverse_C))
            source = "none"
            for heater_id, heater_col in heater_col_by_id.items():
                explicit_gain = float(self.model.controller_gain(int(sensor_id), int(heater_id)))
                if np.isfinite(explicit_gain) and abs(explicit_gain) > 0.0:
                    B_dyn[sensor_index, heater_col] = explicit_gain
                    source = "G_ctrl"
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
                if not np.any(np.abs(B_dyn[sensor_index, :]) > 0.0):
                    warnings.append(f"MIMO sensor {int(sensor_id)} has no active paired heater; B_s row set to zero.")
                sources.append(source)
                continue
            if np.isfinite(average_inverse_C) and average_inverse_C > 0.0:
                for heater_col in paired_heater_cols:
                    if abs(float(B_dyn[sensor_index, heater_col])) <= 0.0:
                        B_dyn[sensor_index, heater_col] = alpha * float(average_inverse_C)
                        if source == "none":
                            source = "capacitance"
            elif not np.any(np.abs(B_dyn[sensor_index, :]) > 0.0):
                warnings.append(
                    f"MIMO sensor {int(sensor_id)} has no valid connected-node capacitance; B_s row set to zero."
                )
            if not valid_node_ids and not np.any(np.abs(B_dyn[sensor_index, :]) > 0.0):
                warnings.append(f"MIMO sensor {int(sensor_id)} has no valid connected nodes for average inverse C.")
            sources.append(source)
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
            "sources": tuple(sources),
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
    C = _regularize_capacitance(C, params)
    cryocooler_lift_curve = PT60LiftCurve(
        max_power_w=float(params.cryocooler_max_power_W),
        capacity_scale=float(params.cryocooler_capacity_scale),
    )
    cryocooler_devices, cryocooler_warnings = build_cryocooler_devices(model, node_ids, C)
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
    warnings.extend(cryocooler_warnings)
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
    if (
        bool(getattr(params, "use_radiative_coupling", False))
        and model is not None
        and not getattr(model, "radiation_super_members", None)
        and not getattr(model, "radiation_exchange_links", None)
    ):
        try:
            from .radiation_coupling import apply_radiation_coupling

            diagnostics = apply_radiation_coupling(model)
            if diagnostics.get("skipped_too_many_patches"):
                warnings.append(
                    "Radiative coupling skipped: too many exposed faces for the ray tracer "
                    f"({int(diagnostics.get('patches', 0))}); raise target super-surfaces or coarsen the graph."
                )
            elif diagnostics.get("skipped"):
                warnings.append("Radiative coupling: nothing to couple (fewer than two exposed surfaces).")
            else:
                warnings.append(
                    f"Radiative coupling: {int(diagnostics.get('super_links', 0))} exchange links across "
                    f"{int(diagnostics.get('super_surfaces', 0))} super-surfaces (from "
                    f"{int(diagnostics.get('patches', 0))} exposed faces)."
                )
        except Exception as exc:  # noqa: BLE001 - coupling is optional
            warnings.append(f"Radiative coupling unavailable; continuing without it: {exc}")
    radiation_coeff = _scale_environment_by_coupling(radiation_coeff, model, node_ids)
    gap_links = _gap_radiation_links(model, node_ids, params)
    if gap_links:
        warnings.append(
            f"Contact-gap radiation: {len(gap_links)} suppressed inter-part interface(s) couple by direct "
            "A<->B radiation across the gap."
        )
    radiation_exchange_W, radiation_exchange_degree = _build_radiation_exchange(model, node_ids, gap_links)
    radiation_super_S, radiation_super_W, radiation_super_degree = _build_radiation_super(model, node_ids)
    environment_temperature_K = _environment_temperature_vector(model, node_ids, params)
    inv_C = 1.0 / C
    b = np.zeros(n, dtype=float)
    pairing_warnings = refresh_heater_power_deposition_nodes(model)
    pairing_warnings.extend(refresh_sensor_connected_nodes(model))
    warnings.extend(pairing_warnings)
    has_cryocooler = any(model.nodes[int(node_id)].has_cryocooler for node_id in node_ids)
    has_mimo_controller = _mimo_controller_is_active(model, heater_node_ids, params)
    # Any nonlinear (T^4) radiation term forces per-step RHS evaluation: the
    # ambient sink, surface-to-surface node exchange (incl. contact-gap links),
    # or the factored super-surface exchange.
    has_nonlinear_radiation = bool(
        (params.use_ambient_radiation and np.any(radiation_coeff > 0.0))
        or radiation_exchange_W is not None
        or radiation_super_S is not None
    )
    use_temperature_dependent_properties = bool(
        getattr(params, "use_temperature_dependent_properties", False)
    )
    dynamic_heater_inputs = (
        params.input_mode == "heater_inputs"
        or has_cryocooler
        or has_mimo_controller
        or has_nonlinear_radiation
        or use_temperature_dependent_properties
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
        _heater_controller_mode(model.nodes[int(node_id)]) == "mimo"
        for node_id in node_ids
        if model.nodes[int(node_id)].is_heater
    ) and not has_mimo_controller:
        warnings.append("MIMO heater control is selected, but no valid paired MIMO sensor/heater set is available.")

    # Single integration scheme: implicit TR-BDF2 on a sparse conduction operator,
    # run on the GPU when available and on the CPU as the fallback.
    L_sparse = csr_matrix(L)
    A = -(diags(inv_C, format="csr") @ L_sparse)
    sparse_implicit_stepper = _build_implicit_stepper(A, C, L_sparse, params, warnings, backend="cpu")
    gpu_implicit_stepper = _build_implicit_stepper(A, C, L_sparse, params, warnings, backend="gpu")
    temperature_dependent_operator = None
    if use_temperature_dependent_properties:
        try:
            temperature_dependent_operator = build_temperature_dependent_operator(
                model,
                node_ids,
                copper_rrr=int(getattr(params, "copper_rrr", 100)),
                default_bolted_conductance_W_m2K=float(
                    getattr(params, "default_bolted_contact_conductance_W_m2K", 3000.0)
                ),
                contact_temp_exponent=float(getattr(params, "contact_conductance_temp_exponent", 1.0)),
                contact_reference_temperature_K=float(
                    getattr(params, "contact_conductance_reference_temperature_K", 293.15)
                ),
            )
            warnings.append(
                "Temperature-dependent material properties enabled: C(T)/L(T) rebuilt each step "
                f"(copper RRR={int(getattr(params, 'copper_rrr', 100))}, "
                f"bolted h={float(getattr(params, 'default_bolted_contact_conductance_W_m2K', 3000.0)):g} W/m2K, "
                f"h(T) exponent n={float(getattr(params, 'contact_conductance_temp_exponent', 1.0)):g})."
            )
        except Exception as exc:  # noqa: BLE001 - fall back to constant properties
            warnings.append(f"Temperature-dependent properties unavailable; using constant properties: {exc}")
            temperature_dependent_operator = None
    prepared = PreparedSimulation(
        node_ids=node_ids,
        z=np.concatenate([initial, np.array([1.0])]),
        initial_temperatures_K=initial,
        params=params,
        model=model,
        inv_C=inv_C,
        A=A,
        base_b=b,
        radiation_coeff_W_K4=radiation_coeff,
        environment_temperature_K=environment_temperature_K,
        radiation_exchange_W=radiation_exchange_W,
        radiation_exchange_degree=radiation_exchange_degree,
        radiation_super_S=radiation_super_S,
        radiation_super_W=radiation_super_W,
        radiation_super_degree=radiation_super_degree,
        sparse_implicit_stepper=sparse_implicit_stepper,
        gpu_implicit_stepper=gpu_implicit_stepper,
        temperature_dependent_operator=temperature_dependent_operator,
        node_index_by_id=node_index_by_id,
        heater_node_ids=heater_node_ids,
        cryocooler_node_ids=cryocooler_node_ids,
        cryocooler_devices=cryocooler_devices,
        cryocooler_lift_curve=cryocooler_lift_curve,
        dynamic_heater_inputs=dynamic_heater_inputs,
        warnings=warnings,
    )
    prepared.reset()
    return prepared


def _build_implicit_stepper(
    A: Any,
    C: np.ndarray,
    L_sparse: Any,
    params: SimulationParameters,
    warnings: list[str],
    *,
    backend: str,
) -> SparseImplicitStepper | None:
    """Build the implicit TR-BDF2 stepper for the CPU or GPU backend.

    The CPU stepper is always built (it is the fallback). The GPU stepper is
    built only when a CUDA device and CuPy are available and the operator is
    symmetric (CG); otherwise this returns None and the CPU stepper is used.
    """
    is_gpu = str(backend).lower() == "gpu"
    if is_gpu:
        if not bool(getattr(params, "gpu_solver_enabled", True)):
            return None
        cp, cupyx_sparse, reason = _optional_cupy_modules()
        if cp is None or cupyx_sparse is None:
            warnings.append(f"GPU implicit solver unavailable; using CPU implicit solver: {reason}")
            return None
        try:
            if int(cp.cuda.runtime.getDeviceCount()) <= 0:
                warnings.append("GPU implicit solver unavailable; no CUDA device reported by CuPy.")
                return None
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"GPU implicit solver unavailable; CUDA detection failed: {exc}")
            return None

    dt = float(params.dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        if not is_gpu:
            warnings.append("Implicit solver unavailable; dt_s must be positive.")
        return None
    C_values = np.asarray(C, dtype=float).reshape(-1)
    if C_values.size == 0 or np.any(C_values <= 0.0) or not np.all(np.isfinite(C_values)):
        if not is_gpu:
            warnings.append("Implicit solver unavailable; capacitance vector is invalid.")
        return None
    if _sparse_matrix_is_symmetric(L_sparse):
        solver = "cg"
        capacitance = C_values
        state_operator = csr_matrix(L_sparse)
    else:
        solver = "bicgstab"
        capacitance = None
        state_operator = csr_matrix(A)
    if is_gpu and solver != "cg":
        warnings.append("GPU implicit solver unavailable for an asymmetric operator; using CPU implicit solver.")
        return None
    method = str(getattr(params, "implicit_sparse_simulation_method", "tr_bdf2") or "tr_bdf2").lower()
    if method not in {"tr_bdf2", "backward_euler"}:
        if not is_gpu:
            warnings.append(f"Unknown implicit method {method!r}; using tr_bdf2.")
        method = "tr_bdf2"
    rtol = max(0.0, float(getattr(params, "implicit_sparse_simulation_rtol", 1.0e-6)))
    maxiter = max(1, int(getattr(params, "implicit_sparse_simulation_maxiter", 300)))
    try:
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
            backend="gpu" if is_gpu else "cpu",
            block_jacobi_enabled=bool(getattr(params, "implicit_sparse_block_jacobi_enabled", False)),
            block_jacobi_size=max(1, int(getattr(params, "implicit_sparse_block_jacobi_size", 64))),
        )
        if is_gpu:
            # Resolve the CuPy modules now so any import/JIT error is caught here.
            _ = stepper._be
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"{'GPU' if is_gpu else 'CPU'} implicit solver unavailable; setup failed: {exc}")
        return None
    precond = (
        f"block-jacobi({int(getattr(params, 'implicit_sparse_block_jacobi_size', 64))})"
        if bool(getattr(params, "implicit_sparse_block_jacobi_enabled", False))
        else "jacobi"
    )
    warnings.append(
        f"{'GPU' if is_gpu else 'CPU'} implicit {method} solver enabled "
        f"(solver={solver}, rtol={rtol:g}, maxiter={maxiter}, precond={precond})."
    )
    return stepper


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


def _node_emissivity(node) -> float:
    """Emissivity clamped to the physical open interval (0, 1] for use in the
    two-surface exchange formula (which divides by it)."""
    value = float(getattr(node, "emissivity", 0.0) or 0.0) if node is not None else 0.0
    if value <= 0.0:
        return 0.9  # sensible default when a node carries no emissivity
    return min(value, 1.0)


def _gap_radiation_links(
    model: ThermalGraphModel | None, node_ids: np.ndarray, params
) -> list[tuple[int, int, float]]:
    """Direct A<->B radiative exchange links across contact-gap-suppressed
    interfaces. The octree builder records ``gap_radiation_links`` -- (i, j,
    shared_face_area_m2) for each inter-part voxel interface whose conduction was
    suppressed because the CAD parts are separated by a sub-voxel gap. The two
    faces are coincident, so they see each other with view factor ~1 and the gray
    two-surface exchange area is A / (1/eps_i + 1/eps_j - 1). Emissivities are read
    live from the nodes (so GUI edits take effect). Returns [] when radiation is
    disabled or no gap links exist."""
    if model is None:
        return []
    if not (
        bool(getattr(params, "use_ambient_radiation", False))
        or bool(getattr(params, "use_radiative_coupling", False))
    ):
        return []  # no radiation modeled at all -> gaps stay uncoupled
    data = getattr(model, "octree_graph_data", None)
    raw = data.get("gap_radiation_links") if isinstance(data, dict) else None
    if not raw:
        return []
    present = {int(v) for v in node_ids}
    links: list[tuple[int, int, float]] = []
    for entry in raw:
        i_id, j_id, area = int(entry[0]), int(entry[1]), float(entry[2])
        if i_id not in present or j_id not in present or i_id == j_id or not (area > 0.0):
            continue
        eps_i = _node_emissivity(model.nodes.get(i_id))
        eps_j = _node_emissivity(model.nodes.get(j_id))
        denom = 1.0 / eps_i + 1.0 / eps_j - 1.0
        if denom <= 0.0:
            continue
        links.append((i_id, j_id, area / denom))
    return links


def _build_radiation_exchange(
    model: ThermalGraphModel | None,
    node_ids: np.ndarray,
    extra_links: list[tuple[int, int, float]] | None = None,
) -> tuple[Any | None, np.ndarray | None]:
    """Build the sparse symmetric radiative exchange-area matrix W [m^2] from the
    model's ``radiation_exchange_links`` (list of (node_i, node_j, G_ij)) combined
    with ``extra_links`` (e.g. contact-gap direct links), plus its row-sum degree
    vector. Returns (None, None) when no links are present. Populated by ray-traced
    view factors (or, in validation, analytic factors) and the gap-coupling links;
    the net exchange power is sigma*(W @ T^4 - degree * T^4)."""
    base = getattr(model, "radiation_exchange_links", None) if model is not None else None
    links = list(base or []) + list(extra_links or [])
    if not links:
        return None, None
    index = {int(node_id): row for row, node_id in enumerate(node_ids)}
    n = len(node_ids)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for entry in links:
        i_id, j_id, g_ij = int(entry[0]), int(entry[1]), float(entry[2])
        ri = index.get(i_id)
        rj = index.get(j_id)
        if ri is None or rj is None or ri == rj or not (g_ij > 0.0):
            continue
        rows.extend((ri, rj))
        cols.extend((rj, ri))
        data.extend((g_ij, g_ij))
    if not data:
        return None, None
    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    degree = np.asarray(W.sum(axis=1), dtype=float).reshape(-1)
    return W, degree


def _build_radiation_super(
    model: ThermalGraphModel | None, node_ids: np.ndarray
) -> tuple[Any | None, Any | None, np.ndarray | None]:
    """Build the factored grouped-exchange operators from the model's
    ``radiation_super_members`` / ``radiation_super_links``: S (n_super x n_node,
    area-weighted node->super aggregation, rows sum to 1), the super-surface
    exchange-area matrix W_super (symmetric), and its degree (row sums).
    Returns (None, None, None) when no grouped coupling is present."""
    members = getattr(model, "radiation_super_members", None) if model is not None else None
    links = getattr(model, "radiation_super_links", None) if model is not None else None
    if not members or not links:
        return None, None, None
    index = {int(node_id): row for row, node_id in enumerate(node_ids)}
    n_node = len(node_ids)
    n_super = len(members)
    s_rows: list[int] = []
    s_cols: list[int] = []
    s_data: list[float] = []
    for super_index, member_area in enumerate(members):
        total = float(sum(member_area.values()))
        if total <= 0.0:
            continue
        for node_id, area in member_area.items():
            row = index.get(int(node_id))
            if row is None:
                continue
            s_rows.append(super_index)
            s_cols.append(row)
            s_data.append(float(area) / total)
    if not s_data:
        return None, None, None
    S = csr_matrix((s_data, (s_rows, s_cols)), shape=(n_super, n_node))
    w_rows: list[int] = []
    w_cols: list[int] = []
    w_data: list[float] = []
    for entry in links:
        i, j, g = int(entry[0]), int(entry[1]), float(entry[2])
        if i == j or not (g > 0.0) or i >= n_super or j >= n_super:
            continue
        w_rows.extend((i, j))
        w_cols.extend((j, i))
        w_data.extend((g, g))
    if not w_data:
        return None, None, None
    W = csr_matrix((w_data, (w_rows, w_cols)), shape=(n_super, n_super))
    degree = np.asarray(W.sum(axis=1), dtype=float).reshape(-1)
    return S, W, degree


def _scale_environment_by_coupling(
    radiation_coeff: np.ndarray, model: ThermalGraphModel | None, node_ids: np.ndarray
) -> np.ndarray:
    """Scale each node's ambient-radiation coefficient by the fraction of its view
    that escapes to the background (``radiation_env_fraction_by_node``, set by the
    coupling driver). Without this the coupled area would radiate both to other
    surfaces and fully to the background -- double counting. No-op when coupling
    is absent (fraction defaults to 1)."""
    fractions = getattr(model, "radiation_env_fraction_by_node", None) if model is not None else None
    if not fractions:
        return radiation_coeff
    coeff = np.asarray(radiation_coeff, dtype=float).reshape(-1).copy()
    for row, node_id in enumerate(node_ids):
        fraction = fractions.get(int(node_id))
        if fraction is not None:
            coeff[row] *= float(fraction)
    return coeff


def _environment_temperature_vector(
    model: ThermalGraphModel | None, node_ids: np.ndarray, params: SimulationParameters
) -> np.ndarray:
    """Per-node radiative background temperature [K]. Every node radiates to the
    exterior/ambient ``T_env_K`` unless it is listed in the model's
    ``radiation_interior_node_ids`` (assigned by the view-factor classification),
    in which case it radiates to the interior cryo enclosure temperature."""
    exterior = float(params.T_env_K)
    env = np.full(len(node_ids), exterior, dtype=float)
    interior_ids = getattr(model, "radiation_interior_node_ids", None) if model is not None else None
    if interior_ids:
        interior = float(getattr(params, "interior_environment_temperature_K", exterior))
        interior_set = {int(v) for v in interior_ids}
        for row, node_id in enumerate(node_ids):
            if int(node_id) in interior_set:
                env[row] = interior
    return env


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
    cryocooler_devices: Sequence[CryocoolerDevice] | None = None,
    cryocooler_lift_curve: PT60LiftCurve | None = None,
    cryocooler_diagnostics: list[dict[str, Any]] | None = None,
    capacitance: np.ndarray | None = None,
) -> np.ndarray:
    powers = np.zeros(len(node_ids), dtype=float)
    skipped_modes = excluded_modes or set()
    node_index = node_index_by_id or {int(node_id): row for row, node_id in enumerate(node_ids)}
    if include_cryocoolers:
        powers -= _cryocooler_power_vector(
            model,
            node_ids,
            temperatures_K,
            params,
            node_index_by_id=node_index,
            cryocooler_node_ids=cryocooler_node_ids,
            cryocooler_devices=cryocooler_devices,
            lift_curve=cryocooler_lift_curve,
            diagnostics_out=cryocooler_diagnostics,
            capacitance=capacitance,
        )
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
        heater_mode = _heater_controller_mode(heater, sensor)
        if heater_mode in skipped_modes:
            continue
        if heater_mode != "manual":
            continue
        heater_row = node_index.get(int(heater_id))
        if heater is None or heater_row is None or not heater.is_heater:
            continue
        if not bool(getattr(heater, "heater_valid", True)) or not getattr(heater, "power_deposition_node_ids", []):
            continue
        max_power = max(0.0, float(heater.heater.heater_max_power_W) * float(heater.heater.heater_efficiency))
        command = min(max(_heater_controller_value(heater, sensor, "sensor_manual_power_W", 0.0), 0.0), max_power)
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
        heater_mode = _heater_controller_mode(heater, sensor)
        if heater_mode in skipped_modes:
            continue
        if heater_mode != "manual":
            continue
        max_power = max(0.0, float(heater.heater.heater_max_power_W) * float(heater.heater.heater_efficiency))
        commands[int(heater_id)] = min(max(_heater_controller_value(heater, sensor, "sensor_manual_power_W", 0.0), 0.0), max_power)
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


def _regularize_capacitance(C: np.ndarray, params: SimulationParameters) -> np.ndarray:
    """Floor per-node heat capacity at ``implicit_capacitance_floor_J_K``.

    Degenerate near-zero-capacitance cells (thin-shell / oversized-marker mesh
    artifacts, ~1e-12 J/K) blow up the implicit stage-matrix condition number, so
    jacobi-CG returns an inaccurate solution that overshoots temperatures negative.
    Flooring shrinks the spread so the solve stays accurate. No-op when the floor
    is <= 0."""
    C = np.asarray(C, dtype=float).reshape(-1)
    floor = float(getattr(params, "implicit_capacitance_floor_J_K", 0.0) or 0.0)
    if floor > 0.0:
        return np.maximum(C, floor)
    return C


def _node_capacitance_at(node: Any, temperature_K: float, tdep: bool) -> float:
    """Heat capacity [J/K] of a node at a temperature. Uses the cp(T) curve when
    temperature-dependent properties are active (so the cryocooler cap reflects the
    genuinely small cryogenic capacity), else the node's stored constant C."""
    constant_C = float(getattr(node, "C_J_K", 0.0) or 0.0)
    if not tdep:
        return constant_C if constant_C > 0.0 else 1.0e-12
    mass = float(getattr(node, "mass_kg", 0.0) or 0.0)
    cp0 = float(getattr(node, "cp_J_kgK", 0.0) or 0.0)
    material = str(getattr(node, "material", "") or "")
    try:
        cp = float(_mp.specific_heat_J_kgK(material, np.array([float(temperature_K)]), fallback_cp=cp0)[0])
    except Exception:  # noqa: BLE001 - fall back to the stored constant capacity
        cp = cp0
    capacity = mass * cp
    if not (np.isfinite(capacity) and capacity > 0.0):
        capacity = constant_C
    return capacity if capacity > 0.0 else 1.0e-12


def _cryocooler_power_vector(
    model: ThermalGraphModel,
    node_ids: np.ndarray,
    temperatures_K: np.ndarray,
    params: SimulationParameters,
    node_index_by_id: dict[int, int] | None = None,
    cryocooler_node_ids: Sequence[int] | None = None,
    cryocooler_devices: Sequence[CryocoolerDevice] | None = None,
    lift_curve: PT60LiftCurve | None = None,
    diagnostics_out: list[dict[str, Any]] | None = None,
    capacitance: np.ndarray | None = None,
) -> np.ndarray:
    powers = np.zeros(len(node_ids), dtype=float)
    node_index = node_index_by_id or {int(node_id): row for row, node_id in enumerate(node_ids)}
    # Current per-node heat capacity, used to cap cooling so a receiving cell cannot
    # be driven below the cooler's floor temperature in one (explicit) step. Falls
    # back to the node's stored (constant) C when a live tdep vector isn't supplied.
    capacitance_by_row = np.asarray(capacitance, dtype=float).reshape(-1) if capacitance is not None else None
    devices = tuple(cryocooler_devices or ())
    if not devices:
        capacitance = [float(model.nodes[int(node_id)].C_J_K) for node_id in node_ids]
        devices, _warnings = build_cryocooler_devices(model, node_ids, capacitance)
    if cryocooler_node_ids is not None:
        allowed_source_ids = {int(value) for value in cryocooler_node_ids}
        devices = tuple(
            device
            for device in devices
            if any(int(node_id) in allowed_source_ids for node_id in device.source_node_ids)
        )
    curve = lift_curve or PT60LiftCurve(
        max_power_w=float(params.cryocooler_max_power_W),
        capacity_scale=float(params.cryocooler_capacity_scale),
    )
    diagnostics: list[dict[str, Any]] = []
    temperatures = np.asarray(temperatures_K, dtype=float).reshape(-1)
    for device in devices:
        rows = [node_index[int(node_id)] for node_id in device.receiving_node_ids if int(node_id) in node_index]
        temperature_weights = np.asarray(device.temperature_weights, dtype=float)
        distribution_weights = np.asarray(device.distribution_weights, dtype=float)
        warning = ""
        if len(rows) != len(device.receiving_node_ids) or len(rows) == 0:
            tip_temperature = float("nan")
            base_capacity_w = 0.0
            applied_cooling_w = 0.0
            distributed = np.zeros(len(rows), dtype=float)
            warning = "no valid receiving nodes; zero cooling applied"
        else:
            if temperature_weights.size != len(rows) or distribution_weights.size != len(rows):
                raise ValueError(f"Cryocooler {device.identifier!r} weight count does not match receiving nodes.")
            if not np.isclose(float(np.sum(temperature_weights)), 1.0, rtol=1.0e-12, atol=1.0e-12):
                raise ValueError(f"Cryocooler {device.identifier!r} temperature weights do not sum to one.")
            if not np.isclose(float(np.sum(distribution_weights)), 1.0, rtol=1.0e-12, atol=1.0e-12):
                raise ValueError(f"Cryocooler {device.identifier!r} distribution weights do not sum to one.")
            tip_temperature = float(np.dot(temperature_weights, temperatures[np.asarray(rows, dtype=int)]))
            base_capacity_w = curve.base_cooling_capacity_w(tip_temperature)
            enabled = bool(params.cryocooler_enabled) and bool(device.enabled)
            applied_cooling_w = curve.cooling_capacity_w(tip_temperature) if enabled else 0.0
            distributed = distribution_weights * float(applied_cooling_w)
            total_distributed = float(np.sum(distributed))
            if not np.isclose(
                total_distributed,
                applied_cooling_w,
                rtol=1.0e-12,
                atol=max(1.0e-12, 1.0e-12 * abs(float(applied_cooling_w))),
            ):
                raise AssertionError(
                    f"Cryocooler {device.identifier!r} distributed {total_distributed} W instead of {applied_cooling_w} W."
                )
            # Cap each receiving cell's cooling to the energy that just reaches the
            # cooler's floor temperature this step: P <= C_i*(T_i - T_floor)/dt.
            # cooling_capacity_w is already zero below the floor, so this only bites
            # on the (explicit) overshoot that would otherwise drive a tiny-C cell
            # below the floor -- and negative -- detonating the T^4 radiation term.
            rows_arr = np.asarray(rows, dtype=int)
            if capacitance_by_row is not None and capacitance_by_row.shape[0] == temperatures.shape[0]:
                capacity_rows = capacitance_by_row[rows_arr]
            else:
                tdep_active = bool(getattr(params, "use_temperature_dependent_properties", False))
                capacity_rows = np.array(
                    [
                        _node_capacitance_at(model.nodes[int(node_ids[int(r)])], float(temperatures[int(r)]), tdep_active)
                        for r in rows_arr
                    ],
                    dtype=float,
                )
            floor_K = float(curve.minimum_temperature_k)
            dt_s = max(float(params.dt_s), 1.0e-12)
            max_removable_w = np.maximum(0.0, capacity_rows * (temperatures[rows_arr] - floor_K) / dt_s)
            distributed = np.minimum(np.asarray(distributed, dtype=float), max_removable_w)
            for row, cooling_w in zip(rows, distributed):
                powers[int(row)] += float(cooling_w)
        enabled = bool(params.cryocooler_enabled) and bool(device.enabled)
        diagnostics.append(
            {
                "cryocooler_id": str(device.identifier),
                "source_node_ids": [int(value) for value in device.source_node_ids],
                "receiving_node_ids": [int(value) for value in device.receiving_node_ids],
                "representative_temperature_K": tip_temperature if np.isfinite(tip_temperature) else None,
                "base_curve_capacity_W": float(base_capacity_w),
                "capacity_scale": float(curve.capacity_scale),
                "maximum_cooling_cap_W": float(curve.max_power_w),
                "applied_cooling_W": float(applied_cooling_w),
                "enabled": enabled,
                "receiving_node_count": len(rows),
                "temperature_weight_sum": float(np.sum(temperature_weights)) if temperature_weights.size else 0.0,
                "distribution_weight_sum": float(np.sum(distribution_weights)) if distribution_weights.size else 0.0,
                "total_distributed_cooling_W": float(np.sum(distributed)),
                "weighting_basis": str(device.weighting_basis),
                "warning": warning,
            }
        )
    # Global per-node cap. The per-device cap above limits EACH device's cooling
    # of a cell to the energy that brings it to the floor -- but a cell served by
    # several devices gets that removed once PER device, so the SUM over-cools it
    # far below the floor and negative. That is what detonated stiff shell graphs
    # (a cell dropping 293 K -> -58 K in one step). Cap the TOTAL cooling at each
    # cell to C_i*(T_i - floor)/dt so no cell is driven below the cooler floor,
    # regardless of how many devices target it.
    cooled_rows = np.nonzero(powers > 0.0)[0]
    if cooled_rows.size and devices:
        floor_K = float(curve.minimum_temperature_k)
        dt_s = max(float(params.dt_s), 1.0e-12)
        if capacitance_by_row is not None and capacitance_by_row.shape[0] == temperatures.shape[0]:
            cap_rows = capacitance_by_row[cooled_rows]
        else:
            tdep_active = bool(getattr(params, "use_temperature_dependent_properties", False))
            cap_rows = np.array(
                [
                    _node_capacitance_at(
                        model.nodes[int(node_ids[int(r)])], float(temperatures[int(r)]), tdep_active
                    )
                    for r in cooled_rows
                ],
                dtype=float,
            )
        max_removable = np.maximum(0.0, cap_rows * (temperatures[cooled_rows] - floor_K) / dt_s)
        powers[cooled_rows] = np.minimum(powers[cooled_rows], max_removable)
    if diagnostics_out is not None:
        diagnostics_out.clear()
        diagnostics_out.extend(diagnostics)
    return powers


def _controller_sensor_weight(node: Any) -> float:
    explicit = max(0.0, float(getattr(node, "controller_weight", 0.0)))
    if explicit > 0.0:
        return explicit
    return 0.5


def _heater_controller_average(
    model: ThermalGraphModel,
    heater_ids: Sequence[int],
    field: str,
    default: float,
    *,
    sensor_id: int | None = None,
    clamp_min: float | None = None,
) -> float:
    sensor = model.nodes.get(int(sensor_id)) if sensor_id is not None else None
    values: list[float] = []
    for heater_id in heater_ids:
        heater = model.nodes.get(int(heater_id))
        if heater is None:
            continue
        try:
            value = float(_heater_controller_value(heater, sensor, field, default))
        except (TypeError, ValueError):
            continue
        if clamp_min is not None:
            value = max(float(clamp_min), value)
        values.append(value)
    if not values:
        return max(float(clamp_min), float(default)) if clamp_min is not None else float(default)
    average = float(np.mean(values))
    if field == "controller_weight" and average <= 0.0:
        return 0.5
    return average


def _rls_ff_update(
    P: np.ndarray,
    dM: np.ndarray,
    r_sp: np.ndarray,
    integral: np.ndarray,
    forgetting: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """One recursive-least-squares step for the adaptive feedforward.

    At a steady operating point the integral holds exactly the part of the true
    holding power that the model feedforward fails to supply -- i.e. the integral
    is a direct measurement of ``(dM_true - dM) @ r_sp`` for the correction matrix
    ``dM`` we are learning (``dM_true = G_dc^-1_true - G_dc^-1_model``). So the RLS
    innovation for regressing the correction against the setpoint IS the current
    integral; we never have to form the target explicitly.

    The transfer is bumpless: bumping ``dM`` by ``outer(integral, K)`` raises the
    feedforward at THIS setpoint by exactly ``alpha * integral`` (``alpha = phi^T K``
    in ``[0, 1)``), so we hand back the integral with that same amount removed. The
    delivered command is therefore unchanged this step; over repeated steady
    samples the deficit decays geometrically and authority migrates from the
    reactive integral into the predictive feedforward.

    Returns ``(P_new, dM_new, integral_new, alpha)``. On a degenerate regressor
    (``phi ~ 0`` or a non-finite denominator) it is a no-op that returns the
    inputs unchanged with ``alpha = 0``.
    """
    phi = np.asarray(r_sp, dtype=float)
    g = P @ phi
    info = float(phi @ g)  # phi^T P phi -- the sample's information content
    denom = float(forgetting) + info
    # No-op on a degenerate sample: a ~zero regressor (info ~ 0) carries no
    # information, so skip it entirely rather than inflating P by 1/forgetting.
    if not np.isfinite(denom) or denom <= 0.0 or info <= 1.0e-12:
        return P, dM, integral, 0.0
    K = g / denom
    alpha = float(phi @ K)  # in [0, 1)
    dM_new = dM + np.outer(integral, K)
    P_new = (P - np.outer(K, g)) / float(forgetting)
    # Keep the covariance symmetric against round-off accumulation.
    P_new = 0.5 * (P_new + P_new.T)
    integral_new = integral - alpha * integral  # = (1 - alpha) * integral
    return P_new, dM_new, integral_new, alpha


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


_HEATER_CONTROLLER_DEFAULTS = {
    "sensor_control_mode": "manual",
    "sensor_manual_power_W": 0.0,
    "controller_weight": 0.0,
    "sensor_settling_time_s": 0.0,
    "controller_kp_coarse": 0.0,
    "controller_ki_coarse": 0.0,
    "controller_kd_coarse": 0.0,
    "controller_kp_hold": 0.0,
    "controller_ki_hold": 0.0,
    "controller_kd_hold": 0.0,
}


def _heater_controller_mode(heater: Any, sensor: Any | None = None) -> str:
    mode = str(getattr(heater, "sensor_control_mode", "manual")).strip().lower()
    legacy_heater_mode = str(getattr(getattr(heater, "heater_control", None), "mode", "")).strip().lower()
    if mode == "mimo" or legacy_heater_mode == "mimo":
        return "mimo"
    if sensor is not None and _heater_controller_is_default(heater):
        legacy_mode = str(getattr(sensor, "sensor_control_mode", "manual")).strip().lower()
        if legacy_mode == "mimo":
            return "mimo"
    return "manual"


def _heater_controller_value(heater: Any, sensor: Any | None, field: str, default: float) -> float:
    if sensor is not None and _heater_controller_is_default(heater):
        return float(getattr(sensor, field, getattr(heater, field, default)))
    return float(getattr(heater, field, default))


def _heater_controller_is_default(heater: Any) -> bool:
    for field, default in _HEATER_CONTROLLER_DEFAULTS.items():
        value = getattr(heater, field, default)
        if field == "sensor_control_mode":
            if str(value or "manual").strip().lower() == "mimo":
                return False
            continue
        try:
            if abs(float(value) - float(default)) > 1.0e-12:
                return False
        except (TypeError, ValueError):
            continue
    return True


def _node_has_mimo_controller_tags(node: Any) -> bool:
    return _node_is_mimo_sensor(node) or _node_is_mimo_heater(node)


def _node_is_mimo_heater(node: Any) -> bool:
    return (
        bool(getattr(node, "is_heater", False))
        and bool(getattr(node, "heater_valid", True))
        and bool(getattr(node, "power_deposition_node_ids", [1]))
        and _heater_controller_mode(node) == "mimo"
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
    heater_node_ids: Sequence[int] | np.ndarray,
    params: SimulationParameters,
) -> bool:
    if model is None or str(params.input_mode) != "heater_inputs":
        return False
    enabled_sensor_ids = _enabled_node_id_set(params.enabled_sensor_node_ids)
    enabled_heater_ids = _enabled_node_id_set(params.enabled_heater_node_ids)
    for node_id in heater_node_ids:
        heater = model.nodes.get(int(node_id))
        if (
            heater is not None
            and bool(getattr(heater, "is_heater", False))
            and _node_id_enabled(enabled_heater_ids, int(node_id))
            and _heater_has_active_mimo_sensor(model, int(node_id), enabled_sensor_ids)
        ):
            return True
    return False


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
        and _heater_controller_mode(heater, sensor) == "mimo"
        and _node_is_mimo_sensor(sensor)
        and _node_id_enabled(enabled_sensor_ids, int(sensor_id))
    )
