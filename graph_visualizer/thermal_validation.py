"""Built-in analytical validation experiments for the thermal solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import csv
import json
import math
import shutil
from typing import Any

import numpy as np

from .graph_io import load_graph_folder, save_graph_folder
from .material_library import default_material_library, material_defaults
from .matrix_builder import (
    DEFAULT_CONTACT_INTERFACE_CONDUCTANCE_W_M2K,
    build_matrices,
    refresh_geometry_edges,
)
from .models import EdgeMode, GraphMetadata, HeaterProperties, NodeProperties, ThermalGraphModel
from .simulation_model import PreparedSimulation, SimulationState, prepare_simulation
from .simulation_parameters import SimulationParameters


VALIDATION_EXPERIMENTS = (
    "Insulated Block with Constant Heating",
    "Two-Block Thermal Exchange",
    "One-Dimensional Prism Conduction",
    "Two-Node Lumped Conductance",
    "Geometry-Derived Contact Pair",
    "One-Dimensional Distributed Rod",
    "Radiation Cooling (Lumped)",
    "Temperature-Dependent Heating",
    "Cryo Regime (heater + radiation + cp(T))",
    "Sandia Thermal Challenge (experimental)",
    "Radiative Coupling (two-plate enclosure)",
    "Temperature-Dependent Conduction (k(T))",
    "Cryocooler Lift Curve (PT60)",
    "Global Energy Conservation",
)
INSULATED_BLOCK = VALIDATION_EXPERIMENTS[0]
TWO_BLOCK_EXCHANGE = VALIDATION_EXPERIMENTS[1]
ONE_D_PRISM = VALIDATION_EXPERIMENTS[2]
TWO_NODE_LUMPED = VALIDATION_EXPERIMENTS[3]
GEOMETRY_CONTACT_PAIR = VALIDATION_EXPERIMENTS[4]
DISTRIBUTED_ROD = VALIDATION_EXPERIMENTS[5]
RADIATION_COOLING = VALIDATION_EXPERIMENTS[6]
TDEP_HEATING = VALIDATION_EXPERIMENTS[7]
CRYO_REGIME = VALIDATION_EXPERIMENTS[8]
SANDIA_THERMAL_CHALLENGE = VALIDATION_EXPERIMENTS[9]
RADIATIVE_COUPLING = VALIDATION_EXPERIMENTS[10]
TDEP_CONDUCTION = VALIDATION_EXPERIMENTS[11]
CRYOCOOLER_LIFT = VALIDATION_EXPERIMENTS[12]
ENERGY_CONSERVATION = VALIDATION_EXPERIMENTS[13]

STEFAN_BOLTZMANN_W_M2K4 = 5.670374419e-8


def _solve_ivp_reference(
    prepared: "PreparedSimulation",
    source_power_by_node: dict[int, float] | None,
    sample_times_s: np.ndarray,
) -> np.ndarray:
    """Independent high-accuracy trajectory for the SAME nonlinear ODE the solver
    integrates: C(T) dT/dt = -L(T) T + P + eps*sigma*A*(T_env^4 - T^4).

    Integrated with scipy's adaptive RK (LSODA) from the same initial state and
    operators the sim uses, so comparing it to the sim validates correctness
    (time-integration + operator-splitting error), not just self-convergence.
    Returns temperatures at ``sample_times_s`` with shape (len(times), n_nodes).
    """
    from scipy.integrate import solve_ivp

    node_ids = np.asarray(prepared.node_ids, dtype=int)
    n = int(node_ids.size)
    index = prepared.node_index_by_id or {int(v): i for i, v in enumerate(node_ids)}
    source = np.zeros(n, dtype=float)
    for node_id, power in (source_power_by_node or {}).items():
        row = index.get(int(node_id))
        if row is not None:
            source[row] += float(power)
    params = prepared.params
    env_temperature = (
        np.asarray(prepared.environment_temperature_K, dtype=float).reshape(-1)
        if prepared.environment_temperature_K is not None
        else np.full(n, float(params.T_env_K), dtype=float)
    )
    ambient4 = env_temperature**4
    use_radiation = bool(params.use_ambient_radiation) and prepared.radiation_coeff_W_K4 is not None
    radiation_coeff = (
        np.asarray(prepared.radiation_coeff_W_K4, dtype=float).reshape(-1) if use_radiation else None
    )
    operator = prepared.temperature_dependent_operator
    constant_inv_C = np.asarray(prepared.inv_C, dtype=float).reshape(-1) if prepared.inv_C is not None else None
    exchange_W = prepared.radiation_exchange_W if bool(params.use_ambient_radiation) else None
    exchange_degree = (
        np.asarray(prepared.radiation_exchange_degree, dtype=float).reshape(-1)
        if exchange_W is not None
        else None
    )
    super_S = prepared.radiation_super_S if bool(params.use_ambient_radiation) else None
    super_W = prepared.radiation_super_W if super_S is not None else None
    super_degree = (
        np.asarray(prepared.radiation_super_degree, dtype=float).reshape(-1) if super_S is not None else None
    )

    def rhs(_t: float, temperatures: np.ndarray) -> np.ndarray:
        temperatures = np.asarray(temperatures, dtype=float).reshape(-1)
        if operator is not None:
            _C, inv_C, laplacian = operator.rebuild(temperatures)
            derivative = -(inv_C * np.asarray(laplacian @ temperatures, dtype=float).reshape(-1))
        else:
            inv_C = constant_inv_C
            derivative = np.asarray(prepared.A @ temperatures, dtype=float).reshape(-1)
        derivative = derivative + inv_C * source
        if use_radiation:
            derivative = derivative + inv_C * radiation_coeff * (ambient4 - temperatures**4)
        if exchange_W is not None:
            u = temperatures**4
            coupled = np.asarray(exchange_W @ u, dtype=float).reshape(-1)
            derivative = derivative + inv_C * STEFAN_BOLTZMANN_W_M2K4 * (coupled - exchange_degree * u)
        if super_S is not None:
            u = temperatures**4
            aggregated = np.asarray(super_S @ u, dtype=float).reshape(-1)
            super_power = np.asarray(super_W @ aggregated, dtype=float).reshape(-1) - super_degree * aggregated
            derivative = derivative + inv_C * STEFAN_BOLTZMANN_W_M2K4 * np.asarray(
                super_S.T @ super_power, dtype=float
            ).reshape(-1)
        return derivative

    initial = np.asarray(prepared.initial_temperatures_K, dtype=float).reshape(-1)
    times = np.asarray(sample_times_s, dtype=float)
    t_final = float(times[-1]) if times.size else 0.0
    if t_final <= 0.0:
        return np.tile(initial, (max(times.size, 1), 1))
    solution = solve_ivp(
        rhs,
        (0.0, t_final),
        initial,
        t_eval=times,
        method="LSODA",
        rtol=1.0e-9,
        atol=1.0e-9,
    )
    return np.asarray(solution.y, dtype=float).T


@dataclass
class MetricResult:
    name: str
    value: float
    tolerance: float | None = None
    status: str = "PASS"
    units: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThermalValidationParameters:
    experiment_name: str = INSULATED_BLOCK
    material: str = "Copper"
    length_mm: float = 100.0
    width_mm: float = 20.0
    height_mm: float = 20.0
    initial_temperature_K: float = 293.15
    reference_temperature_K: float = 293.15
    duration_s: float = 100.0
    dt_s: float = 0.1
    output_sample_interval_s: float = 1.0
    voxel_min_size_mm: float = 5.0
    voxel_max_size_mm: float = 10.0
    max_octree_depth: int = 8
    samples_per_cell: int = 9
    absolute_tolerance_K: float = 0.05
    relative_tolerance: float = 1.0e-3
    heater_power_W: float = 10.0
    hot_initial_temperature_K: float = 300.0
    cold_initial_temperature_K: float = 200.0
    interface_conductance_W_K: float = 0.1
    interface_model: str = "explicit_total_conductance"
    surface_temperature_K: float = 200.0
    analytical_series_terms: int = 100
    probe_positions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    use_octree_pipeline: bool = True
    # Solver knobs, exposed so the validation runs "just like the actual
    # simulator". Defaults match SimulationParameters so a default run reveals
    # the real solver's accuracy; tighten them to check convergence.
    solver_adaptive_max_substeps: int = 4
    solver_adaptive_target_delta_K: float = 1.0
    solver_rtol: float = 1.0e-6
    gpu_solver_enabled: bool = True
    use_ambient_radiation: bool = False
    use_temperature_dependent_properties: bool = False
    use_midpoint_property_coupling: bool = True
    copper_rrr: int = 100

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["probe_positions"] = list(self.probe_positions)
        return payload


@dataclass
class ValidationBuildResult:
    experiment: "ThermalValidationExperiment"
    model: ThermalGraphModel
    matrices: dict[str, Any]
    graph_folder: Path
    glb_path: Path | None
    requested_volume_m3: float
    imported_volume_m3: float
    warnings: list[str] = field(default_factory=list)

    @property
    def volume_error_fraction(self) -> float:
        if self.requested_volume_m3 <= 0.0:
            return 0.0
        return (self.imported_volume_m3 - self.requested_volume_m3) / self.requested_volume_m3


@dataclass
class ValidationRunResult:
    experiment_name: str
    parameters: ThermalValidationParameters
    material_properties: dict[str, float]
    times_s: list[float]
    simulated: dict[str, list[float]]
    analytical: dict[str, list[float]]
    errors: dict[str, list[float]]
    metrics: list[MetricResult]
    status: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = [metric.to_dict() for metric in self.metrics]
        return payload


class ThermalValidationExperiment:
    name = ""
    description = ""
    equation = ""

    def default_parameters(self) -> ThermalValidationParameters:
        return ThermalValidationParameters(experiment_name=self.name)

    def create_geometry(self, params: ThermalValidationParameters, assets_dir: Path) -> Path | None:
        components = self.geometry_components(params)
        if not components:
            return None
        return _write_glb_scene(components, assets_dir / "mesh" / "validation.glb", params.material)

    def geometry_components(
        self, params: ThermalValidationParameters
    ) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
        raise NotImplementedError

    def build(
        self,
        params: ThermalValidationParameters,
        assets_root: Path,
    ) -> ValidationBuildResult:
        assets_dir = assets_root / _safe_asset_name(self.name)
        if assets_dir.exists():
            shutil.rmtree(assets_dir)
        assets_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        glb_path: Path | None = None
        if params.use_octree_pipeline:
            try:
                glb_path = self.create_geometry(params, assets_dir)
            except Exception as exc:
                warnings.append(f"Generated GLB unavailable; using validation graph fallback: {exc}")
        model: ThermalGraphModel | None = None
        matrices: dict[str, Any] | None = None
        graph_folder = assets_dir / "graphs" / _safe_asset_name(self.name)
        if params.use_octree_pipeline and glb_path is not None:
            try:
                graph_folder = self._build_octree_graph(params, assets_dir)
                model, matrices = load_graph_folder(graph_folder)
                warnings.append("Built via generated GLB and octree graph importer.")
            except Exception as exc:
                warnings.append(
                    "Generated GLB was written, but octree import failed; "
                    f"using deterministic validation graph fallback: {exc}"
                )
        if model is None or matrices is None:
            graph_folder = assets_dir / "fallback_graph"
            model = self.build_fallback_model(params)
            matrices = save_graph_folder(model, graph_folder)
            model, matrices = load_graph_folder(graph_folder)
        self.configure_model(model, matrices, params)
        # Heater/marker nodes must never radiate; the graph save/load round-trip
        # can otherwise assign them exposed-face radiation from their geometry.
        for node in model.nodes.values():
            if bool(getattr(node, "is_heater", False)) or "HEATER" in str(getattr(node, "component_name", "")):
                node.is_exposed = False
                node.G_rad_W_K = 0.0
                node.Grad_W_K = 0.0
        matrices = build_matrices(model)
        requested_volume = self.requested_volume_m3(params)
        imported_volume = self.imported_volume_m3(model)
        warnings.extend(self.integrity_warnings(model, matrices, params, requested_volume, imported_volume))
        return ValidationBuildResult(
            experiment=self,
            model=model,
            matrices=matrices,
            graph_folder=graph_folder,
            glb_path=glb_path,
            requested_volume_m3=requested_volume,
            imported_volume_m3=imported_volume,
            warnings=warnings,
        )

    def _build_octree_graph(self, params: ThermalValidationParameters, assets_dir: Path) -> Path:
        from octree_graph.cli import main as octree_main

        output_root = assets_dir / "graphs"
        graph_name = _safe_asset_name(self.name)
        argv = [
            "--mesh-dir",
            str(assets_dir / "mesh"),
            "--materials",
            str(Path(__file__).resolve().parents[1] / "materials.json"),
            "--graph-name",
            graph_name,
            "--output-root",
            str(output_root),
            "--min-cell-size-mm",
            str(params.voxel_min_size_mm),
            "--max-cell-size-mm",
            str(params.voxel_max_size_mm),
            "--max-depth",
            str(params.max_octree_depth),
            "--samples-per-cell",
            str(params.samples_per_cell),
            "--dense-matrix-node-limit",
            "6000",
            "--voxel-workers",
            "1",
            "--contains-backend",
            "ray",
            "--no-progress",
            "--no-checkpoint-build",
            "--heater-name-pattern",
            "VALIDATION_HEATER",
        ]
        if isinstance(self, TwoBlockExchangeExperiment) and params.interface_model == "explicit_total_conductance":
            area = max(1.0e-12, params.width_mm * params.height_mm * 1.0e-6)
            argv.extend(
                [
                    "--contact-interface-conductance-W-m2K",
                    str(max(params.interface_conductance_W_K / area, 1.0e-12)),
                ]
            )
        octree_main(argv)
        return output_root / graph_name

    def build_fallback_model(self, params: ThermalValidationParameters) -> ThermalGraphModel:
        raise NotImplementedError

    def configure_model(
        self,
        model: ThermalGraphModel,
        matrices: dict[str, Any],
        params: ThermalValidationParameters,
    ) -> None:
        library = model.material_library or default_material_library()
        for node in model.nodes.values():
            if _is_validation_body_node(node):
                _apply_material(node, params.material, library)
                node.initial_temperature_K = float(params.initial_temperature_K)
                node.Grad_W_K = 0.0
                node.G_rad_W_K = 0.0
                node.is_exposed = False
        model.metadata.T_sur_K = float(params.initial_temperature_K)
        model.metadata.edge_mode = EdgeMode.AUTO.value

    def requested_volume_m3(self, params: ThermalValidationParameters) -> float:
        return params.length_mm * params.width_mm * params.height_mm * 1.0e-9

    def imported_volume_m3(self, model: ThermalGraphModel) -> float:
        return _volume_m3_for_nodes(model.nodes.values())

    def simulation_parameters(self, params: ThermalValidationParameters) -> SimulationParameters:
        # Mirror the actual simulator: the solver settings come straight from the
        # experiment parameters (which default to the real-sim defaults), so a
        # default validation run exposes the same accuracy the live sim has.
        # Tighten solver_* to confirm convergence to the analytical reference.
        return SimulationParameters(
            dt_s=float(params.dt_s),
            t_final_s=float(params.duration_s),
            input_mode="zero",
            use_ambient_radiation=bool(params.use_ambient_radiation),
            cryocooler_enabled=False,
            gpu_solver_enabled=bool(params.gpu_solver_enabled),
            use_temperature_dependent_properties=bool(params.use_temperature_dependent_properties),
            use_midpoint_property_coupling=bool(params.use_midpoint_property_coupling),
            copper_rrr=int(params.copper_rrr),
            implicit_sparse_simulation_rtol=float(params.solver_rtol),
            implicit_sparse_adaptive_target_delta_K=float(params.solver_adaptive_target_delta_K),
            implicit_sparse_adaptive_max_substeps=int(params.solver_adaptive_max_substeps),
            simulation_history_limit=0,
        )

    def run(self, build: ValidationBuildResult, params: ThermalValidationParameters) -> ValidationRunResult:
        prepared = prepare_simulation(build.model, build.matrices, self.simulation_parameters(params))
        prepared.reset()
        return self._run_prepared(build, prepared, params)

    def _run_prepared(
        self,
        build: ValidationBuildResult,
        prepared: PreparedSimulation,
        params: ThermalValidationParameters,
    ) -> ValidationRunResult:
        raise NotImplementedError

    def analytical_solution(
        self,
        times_s: np.ndarray,
        params: ThermalValidationParameters,
        build: ValidationBuildResult,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def integrity_warnings(
        self,
        model: ThermalGraphModel,
        matrices: dict[str, Any],
        params: ThermalValidationParameters,
        requested_volume_m3: float,
        imported_volume_m3: float,
    ) -> list[str]:
        warnings: list[str] = []
        if not model.nodes:
            warnings.append("No validation graph nodes were generated.")
        if imported_volume_m3 <= 0.0:
            warnings.append("Imported validation volume is zero.")
        if requested_volume_m3 > 0.0:
            error = abs(imported_volume_m3 - requested_volume_m3) / requested_volume_m3
            if error > 0.05:
                warnings.append(f"Imported volume differs from requested volume by {error:.3%}.")
        C = np.asarray(matrices.get("C", []), dtype=float)
        if C.size == 0 or np.any(~np.isfinite(C)) or np.any(C <= 0.0):
            warnings.append("Graph has missing or non-finite heat capacities.")
        if not bool(getattr(params, "use_ambient_radiation", False)) and np.any(
            np.asarray(matrices.get("G_rad", np.zeros_like(C)), dtype=float) > 0.0
        ):
            warnings.append("Radiation conductance is nonzero but ambient radiation is disabled; this experiment expects an adiabatic exterior.")
        return warnings


class InsulatedHeatedBlockExperiment(ThermalValidationExperiment):
    name = INSULATED_BLOCK
    description = "Energy-conservation check for an adiabatic block with a fixed total heater power."
    equation = "T_avg(t) = T0 + P t / (rho cp V)"

    def geometry_components(
        self, params: ThermalValidationParameters
    ) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
        heater_thickness = max(1.0, min(2.0, params.length_mm * 0.05))
        return [
            (
                "VALIDATION_BLOCK",
                (0.0, 0.0, 0.0),
                (params.length_mm, params.width_mm, params.height_mm),
            ),
            (
                "VALIDATION_HEATER",
                (-(params.length_mm + heater_thickness) * 0.5, 0.0, 0.0),
                (heater_thickness, params.width_mm, params.height_mm),
            ),
        ]

    def build_fallback_model(self, params: ThermalValidationParameters) -> ThermalGraphModel:
        model = _new_validation_model(self.name)
        body_ids = _add_box_grid(
            model,
            "VALIDATION_BLOCK",
            center_mm=(0.0, 0.0, 0.0),
            size_mm=(params.length_mm, params.width_mm, params.height_mm),
            material=params.material,
            initial_temperature_K=params.initial_temperature_K,
            max_cell_size_mm=params.voxel_max_size_mm,
        )
        refresh_geometry_edges(model)
        heater = NodeProperties.with_material(next(_node_id_counter(model)), (-1_000_000, 0, 0), params.material, model.material_library)
        heater.component_name = "VALIDATION_HEATER"
        heater.center_mm = (-(params.length_mm * 0.5 + 1.0), 0.0, 0.0)
        heater.size_mm = (1.0, params.width_mm, params.height_mm)
        heater.side_length_m = 0.001
        heater.mass_kg = 1.0e-9
        heater.C_J_K = 1.0e-6
        heater.is_heater = True
        heater.heater = HeaterProperties(
            heater_id=int(heater.node_id),
            heater_min_power_W=0.0,
            heater_max_power_W=max(0.0, float(params.heater_power_W)),
            heater_efficiency=1.0,
        )
        face_ids = _nodes_on_min_x_face(model, body_ids)
        heater.power_deposition_node_ids = face_ids
        heater.power_deposition_weights = _capacitance_weights(model, face_ids)
        model.add_node(heater)
        return model

    def configure_model(self, model: ThermalGraphModel, matrices: dict[str, Any], params: ThermalValidationParameters) -> None:
        super().configure_model(model, matrices, params)
        body_ids = _component_node_ids(model, "VALIDATION_BLOCK")
        heater_ids = [node_id for node_id, node in model.nodes.items() if "VALIDATION_HEATER" in node.component_name or node.is_heater]
        if not heater_ids:
            return
        face_ids = _nodes_on_min_x_face(model, body_ids) or body_ids
        for heater_id in heater_ids:
            heater = model.nodes[int(heater_id)]
            heater.is_heater = True
            heater.heater.heater_id = int(heater_id)
            heater.heater.heater_max_power_W = max(0.0, float(params.heater_power_W))
            heater.heater.heater_efficiency = 1.0
            heater.power_deposition_node_ids = list(face_ids)
            heater.power_deposition_weights = _capacitance_weights(model, face_ids)
            heater.Grad_W_K = 0.0
            heater.G_rad_W_K = 0.0
            heater.mass_kg = 1.0e-9
            heater.C_J_K = 1.0e-6
        _remove_edges_touching(model, heater_ids)

    def _run_prepared(
        self,
        build: ValidationBuildResult,
        prepared: PreparedSimulation,
        params: ThermalValidationParameters,
    ) -> ValidationRunResult:
        body_ids = _component_node_ids(build.model, "VALIDATION_BLOCK")
        heater_ids = [int(node_id) for node_id, node in build.model.nodes.items() if node.is_heater]
        heater_id = heater_ids[0] if heater_ids else None
        node_index = prepared.node_index_by_id
        C = np.asarray(build.matrices["C"], dtype=float)
        body_rows = np.asarray([node_index[node_id] for node_id in body_ids if node_id in node_index], dtype=int)
        times, simulated, min_values, max_values, energy_values = [], [], [], [], []
        initial = prepared.temperatures_K.copy()
        sample = _SampleClock(params.output_sample_interval_s)
        steps = _step_count(params.duration_s, params.dt_s)
        _record_block_sample(prepared, body_rows, C, initial, times, simulated, min_values, max_values, energy_values)
        for _ in range(steps):
            if heater_id is None:
                prepared.step_forward()
            else:
                _advance_forced_power(prepared, {heater_id: params.heater_power_W})
            if sample.should_sample(prepared.time_s):
                _record_block_sample(prepared, body_rows, C, initial, times, simulated, min_values, max_values, energy_values)
        time_array = np.asarray(times, dtype=float)
        analytical = self.analytical_solution(time_array, params, build)
        simulated_map = {
            "average_temperature_K": simulated,
            "minimum_temperature_K": min_values,
            "maximum_temperature_K": max_values,
            "stored_energy_J": energy_values,
        }
        errors = {
            "average_temperature_K": (
                np.asarray(simulated, dtype=float) - analytical["average_temperature_K"]
            ).astype(float).tolist()
        }
        metrics = _temperature_metrics(
            "average",
            np.asarray(simulated, dtype=float),
            analytical["average_temperature_K"],
            params,
        )
        final_time = float(time_array[-1]) if time_array.size else 0.0
        applied = float(params.heater_power_W) * final_time
        stored = float(energy_values[-1]) if energy_values else 0.0
        metrics.extend(
            [
                _metric("Applied energy", applied, None, "PASS", "J"),
                _metric("Stored thermal energy", stored, None, "PASS", "J"),
                _metric(
                    "Energy conservation error",
                    stored - applied,
                    max(params.absolute_tolerance_K * _capacitance_sum(build.model, body_ids), 1.0e-9),
                    _status_abs(stored - applied, max(params.absolute_tolerance_K * _capacitance_sum(build.model, body_ids), 1.0e-9)),
                    "J",
                ),
                _metric("Imported volume error", build.volume_error_fraction, 0.05, _status_abs(build.volume_error_fraction, 0.05), "fraction"),
                _metric("Total simulated thermal capacitance", _capacitance_sum(build.model, body_ids), None, "PASS", "J/K"),
                _metric("Maximum-to-minimum temperature spread", max(np.asarray(max_values) - np.asarray(min_values)), params.absolute_tolerance_K * 10.0, "PASS", "K"),
            ]
        )
        status = _overall_status(metrics, build.warnings)
        return ValidationRunResult(
            self.name,
            params,
            _material_properties(params.material, build.model.material_library),
            times,
            simulated_map,
            {key: value.astype(float).tolist() for key, value in analytical.items()},
            errors,
            metrics,
            status,
            list(build.warnings),
        )

    def analytical_solution(
        self,
        times_s: np.ndarray,
        params: ThermalValidationParameters,
        build: ValidationBuildResult,
    ) -> dict[str, np.ndarray]:
        body_ids = _component_node_ids(build.model, "VALIDATION_BLOCK")
        C_total = _capacitance_sum(build.model, body_ids)
        return {
            "average_temperature_K": float(params.initial_temperature_K)
            + float(params.heater_power_W) * np.asarray(times_s, dtype=float) / max(C_total, 1.0e-30)
        }


class TwoBlockExchangeExperiment(ThermalValidationExperiment):
    name = TWO_BLOCK_EXCHANGE
    description = "Two adiabatic bodies exchanging heat through a controlled total interface conductance."
    equation = "Delta(t) = Delta0 exp[-G(1/C1 + 1/C2)t]"

    def default_parameters(self) -> ThermalValidationParameters:
        params = super().default_parameters()
        params.length_mm = 20.0
        params.width_mm = 20.0
        params.height_mm = 20.0
        params.initial_temperature_K = 293.15
        params.duration_s = 400.0
        params.dt_s = 0.2
        params.output_sample_interval_s = 2.0
        params.voxel_max_size_mm = 10.0
        return params

    def geometry_components(
        self, params: ThermalValidationParameters
    ) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
        return [
            ("VALIDATION_HOT_BLOCK", (-params.length_mm * 0.5, 0.0, 0.0), (params.length_mm, params.width_mm, params.height_mm)),
            ("VALIDATION_COLD_BLOCK", (params.length_mm * 0.5, 0.0, 0.0), (params.length_mm, params.width_mm, params.height_mm)),
        ]

    def build_fallback_model(self, params: ThermalValidationParameters) -> ThermalGraphModel:
        model = _new_validation_model(self.name)
        _add_box_grid(
            model,
            "VALIDATION_HOT_BLOCK",
            center_mm=(-params.length_mm * 0.5, 0.0, 0.0),
            size_mm=(params.length_mm, params.width_mm, params.height_mm),
            material=params.material,
            initial_temperature_K=params.hot_initial_temperature_K,
            max_cell_size_mm=params.voxel_max_size_mm,
        )
        _add_box_grid(
            model,
            "VALIDATION_COLD_BLOCK",
            center_mm=(params.length_mm * 0.5, 0.0, 0.0),
            size_mm=(params.length_mm, params.width_mm, params.height_mm),
            material=params.material,
            initial_temperature_K=params.cold_initial_temperature_K,
            max_cell_size_mm=params.voxel_max_size_mm,
        )
        refresh_geometry_edges(model)
        return model

    def configure_model(self, model: ThermalGraphModel, matrices: dict[str, Any], params: ThermalValidationParameters) -> None:
        super().configure_model(model, matrices, params)
        for node_id in _component_node_ids(model, "VALIDATION_HOT_BLOCK"):
            model.nodes[node_id].initial_temperature_K = float(params.hot_initial_temperature_K)
        for node_id in _component_node_ids(model, "VALIDATION_COLD_BLOCK"):
            model.nodes[node_id].initial_temperature_K = float(params.cold_initial_temperature_K)
        if params.interface_model == "explicit_total_conductance":
            edge_keys = _interface_edge_keys(model, "VALIDATION_HOT_BLOCK", "VALIDATION_COLD_BLOCK")
            if edge_keys:
                per_edge = float(params.interface_conductance_W_K) / float(len(edge_keys))
                for key in edge_keys:
                    model.edges[key].Gij_W_K = per_edge
                    model.edges[key].edge_type = "validation_explicit_interface"
                    model.edges[key].source_metadata = EdgeMode.LOADED_G.value

    def requested_volume_m3(self, params: ThermalValidationParameters) -> float:
        return 2.0 * params.length_mm * params.width_mm * params.height_mm * 1.0e-9

    def _run_prepared(
        self,
        build: ValidationBuildResult,
        prepared: PreparedSimulation,
        params: ThermalValidationParameters,
    ) -> ValidationRunResult:
        hot_ids = _component_node_ids(build.model, "VALIDATION_HOT_BLOCK")
        cold_ids = _component_node_ids(build.model, "VALIDATION_COLD_BLOCK")
        node_index = prepared.node_index_by_id
        C = np.asarray(build.matrices["C"], dtype=float)
        hot_rows = np.asarray([node_index[node_id] for node_id in hot_ids if node_id in node_index], dtype=int)
        cold_rows = np.asarray([node_index[node_id] for node_id in cold_ids if node_id in node_index], dtype=int)
        times, hot, cold, delta, energy = [], [], [], [], []
        initial = prepared.temperatures_K.copy()
        sample = _SampleClock(params.output_sample_interval_s)
        steps = _step_count(params.duration_s, params.dt_s)
        _record_two_block_sample(prepared, hot_rows, cold_rows, C, initial, times, hot, cold, delta, energy)
        for _ in range(steps):
            prepared.step_forward()
            if sample.should_sample(prepared.time_s):
                _record_two_block_sample(prepared, hot_rows, cold_rows, C, initial, times, hot, cold, delta, energy)
        time_array = np.asarray(times, dtype=float)
        analytical = self.analytical_solution(time_array, params, build)
        errors = {
            "hot_average_temperature_K": (np.asarray(hot) - analytical["hot_average_temperature_K"]).astype(float).tolist(),
            "cold_average_temperature_K": (np.asarray(cold) - analytical["cold_average_temperature_K"]).astype(float).tolist(),
            "temperature_difference_K": (np.asarray(delta) - analytical["temperature_difference_K"]).astype(float).tolist(),
        }
        metrics = []
        metrics.extend(_temperature_metrics("hot block", np.asarray(hot), analytical["hot_average_temperature_K"], params))
        metrics.extend(_temperature_metrics("cold block", np.asarray(cold), analytical["cold_average_temperature_K"], params))
        T_eq = float(analytical["equilibrium_temperature_K"][0])
        hot_C = float(sum(C[hot_rows]))
        cold_C = float(sum(C[cold_rows]))
        total_C = max(hot_C + cold_C, 1.0e-30)
        final_equilibrium = (hot_C * float(hot[-1]) + cold_C * float(cold[-1])) / total_C
        metrics.append(_metric("Final equilibrium temperature error", final_equilibrium - T_eq, params.absolute_tolerance_K, _status_abs(final_equilibrium - T_eq, params.absolute_tolerance_K), "K"))
        metrics.append(_metric("Total energy conservation error", energy[-1], max(params.absolute_tolerance_K * (hot_C + cold_C), 1.0e-9), _status_abs(energy[-1], max(params.absolute_tolerance_K * (hot_C + cold_C), 1.0e-9)), "J"))
        actual_g = _interface_conductance(build.model, "VALIDATION_HOT_BLOCK", "VALIDATION_COLD_BLOCK")
        if params.interface_model == "explicit_total_conductance":
            conductance_tolerance = max(abs(params.interface_conductance_W_K) * 1.0e-9, 1.0e-12)
            metrics.append(
                _metric(
                    "Actual interface conductance",
                    actual_g,
                    conductance_tolerance,
                    _status_abs(actual_g - params.interface_conductance_W_K, conductance_tolerance),
                    "W/K",
                )
            )
        else:
            metrics.append(_metric("Geometry-derived interface conductance", actual_g, None, "PASS", "W/K"))
        metrics.append(_metric("Hot-block internal spread", _component_spread(prepared, hot_rows), params.absolute_tolerance_K * 10.0, "PASS", "K"))
        metrics.append(_metric("Cold-block internal spread", _component_spread(prepared, cold_rows), params.absolute_tolerance_K * 10.0, "PASS", "K"))
        if max(hot) > max(params.hot_initial_temperature_K, params.cold_initial_temperature_K) + params.absolute_tolerance_K:
            metrics.append(_metric("Passive maximum bound", max(hot), None, "FAIL", "K"))
        if min(cold) < min(params.hot_initial_temperature_K, params.cold_initial_temperature_K) - params.absolute_tolerance_K:
            metrics.append(_metric("Passive minimum bound", min(cold), None, "FAIL", "K"))
        return ValidationRunResult(
            self.name,
            params,
            _material_properties(params.material, build.model.material_library),
            times,
            {
                "hot_average_temperature_K": hot,
                "cold_average_temperature_K": cold,
                "temperature_difference_K": delta,
                "stored_energy_change_J": energy,
            },
            {key: value.astype(float).tolist() for key, value in analytical.items()},
            errors,
            metrics,
            _overall_status(metrics, build.warnings),
            list(build.warnings),
        )

    def analytical_solution(
        self,
        times_s: np.ndarray,
        params: ThermalValidationParameters,
        build: ValidationBuildResult,
    ) -> dict[str, np.ndarray]:
        hot_ids = _component_node_ids(build.model, "VALIDATION_HOT_BLOCK")
        cold_ids = _component_node_ids(build.model, "VALIDATION_COLD_BLOCK")
        C1 = _capacitance_sum(build.model, hot_ids)
        C2 = _capacitance_sum(build.model, cold_ids)
        T1_0 = float(params.hot_initial_temperature_K)
        T2_0 = float(params.cold_initial_temperature_K)
        total = max(C1 + C2, 1.0e-30)
        T_eq = (C1 * T1_0 + C2 * T2_0) / total
        delta_0 = T1_0 - T2_0
        G = _interface_conductance(build.model, "VALIDATION_HOT_BLOCK", "VALIDATION_COLD_BLOCK")
        lam = G * (1.0 / max(C1, 1.0e-30) + 1.0 / max(C2, 1.0e-30))
        delta = delta_0 * np.exp(-lam * np.asarray(times_s, dtype=float))
        return {
            "hot_average_temperature_K": T_eq + (C2 / total) * delta,
            "cold_average_temperature_K": T_eq - (C1 / total) * delta,
            "temperature_difference_K": delta,
            "equilibrium_temperature_K": np.full_like(np.asarray(times_s, dtype=float), T_eq),
        }

    def integrity_warnings(self, model, matrices, params, requested_volume_m3, imported_volume_m3) -> list[str]:
        warnings = super().integrity_warnings(model, matrices, params, requested_volume_m3, imported_volume_m3)
        edge_count = len(_interface_edge_keys(model, "VALIDATION_HOT_BLOCK", "VALIDATION_COLD_BLOCK"))
        if edge_count == 0:
            warnings.append("No hot/cold interface contact edges were detected.")
        actual_g = _interface_conductance(model, "VALIDATION_HOT_BLOCK", "VALIDATION_COLD_BLOCK")
        if params.interface_model == "explicit_total_conductance" and not math.isclose(actual_g, params.interface_conductance_W_K, rel_tol=1.0e-9, abs_tol=1.0e-12):
            warnings.append(f"Actual interface conductance {actual_g:.6g} W/K does not match requested {params.interface_conductance_W_K:.6g} W/K.")
        return warnings


class TwoNodeLumpedConductanceExperiment(TwoBlockExchangeExperiment):
    name = TWO_NODE_LUMPED
    description = "Strict two-node ODE check with known capacitances and an explicit total conductance."
    equation = "Delta(t) = Delta0 exp[-G(1/C1 + 1/C2)t]"

    def default_parameters(self) -> ThermalValidationParameters:
        params = super().default_parameters()
        params.use_octree_pipeline = False
        params.interface_model = "explicit_total_conductance"
        params.interface_conductance_W_K = 0.8
        params.duration_s = 10.0
        params.dt_s = 0.05
        params.output_sample_interval_s = 0.25
        params.absolute_tolerance_K = 1.0e-7
        return params

    def geometry_components(
        self, params: ThermalValidationParameters
    ) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
        return []

    def build_fallback_model(self, params: ThermalValidationParameters) -> ThermalGraphModel:
        model = _new_validation_model(self.name)
        hot = NodeProperties.with_material(1, (0, 0, 0), params.material, model.material_library)
        hot.component_name = "VALIDATION_HOT_BLOCK"
        hot.center_mm = (-5.0, 0.0, 0.0)
        hot.size_mm = (1.0, 1.0, 1.0)
        hot.C_J_K = 10.0
        hot.mass_kg = hot.C_J_K / max(float(hot.cp_J_kgK), 1.0e-30)
        hot.initial_temperature_K = float(params.hot_initial_temperature_K)
        hot.Grad_W_K = 0.0
        hot.G_rad_W_K = 0.0
        hot.is_exposed = False
        cold = NodeProperties.with_material(2, (1, 0, 0), params.material, model.material_library)
        cold.component_name = "VALIDATION_COLD_BLOCK"
        cold.center_mm = (5.0, 0.0, 0.0)
        cold.size_mm = (1.0, 1.0, 1.0)
        cold.C_J_K = 30.0
        cold.mass_kg = cold.C_J_K / max(float(cold.cp_J_kgK), 1.0e-30)
        cold.initial_temperature_K = float(params.cold_initial_temperature_K)
        cold.Grad_W_K = 0.0
        cold.G_rad_W_K = 0.0
        cold.is_exposed = False
        model.add_node(hot)
        model.add_node(cold)
        model.set_edge(1, 2, float(params.interface_conductance_W_K), EdgeMode.LOADED_G.value, edge_type="validation_explicit_interface")
        return model

    def configure_model(self, model: ThermalGraphModel, matrices: dict[str, Any], params: ThermalValidationParameters) -> None:
        hot_ids = _component_node_ids(model, "VALIDATION_HOT_BLOCK")
        cold_ids = _component_node_ids(model, "VALIDATION_COLD_BLOCK")
        for node in model.nodes.values():
            node.material = params.material
            node.initial_temperature_K = (
                float(params.hot_initial_temperature_K)
                if str(node.component_name).startswith("VALIDATION_HOT_BLOCK")
                else float(params.cold_initial_temperature_K)
            )
            node.Grad_W_K = 0.0
            node.G_rad_W_K = 0.0
            node.is_exposed = False
        if hot_ids and cold_ids:
            model.set_edge(
                int(hot_ids[0]),
                int(cold_ids[0]),
                float(params.interface_conductance_W_K),
                EdgeMode.LOADED_G.value,
                edge_type="validation_explicit_interface",
            )
        for edge in model.edges.values():
            edge.Gij_W_K = float(params.interface_conductance_W_K)
            edge.edge_type = "validation_explicit_interface"
            edge.source_metadata = EdgeMode.LOADED_G.value

    def requested_volume_m3(self, params: ThermalValidationParameters) -> float:
        return 0.0

    def imported_volume_m3(self, model: ThermalGraphModel) -> float:
        return 0.0

    def integrity_warnings(self, model, matrices, params, requested_volume_m3, imported_volume_m3) -> list[str]:
        warnings: list[str] = []
        if len(_component_node_ids(model, "VALIDATION_HOT_BLOCK")) != 1:
            warnings.append("Two-node lumped experiment requires exactly one hot node.")
        if len(_component_node_ids(model, "VALIDATION_COLD_BLOCK")) != 1:
            warnings.append("Two-node lumped experiment requires exactly one cold node.")
        if len(_interface_edge_keys(model, "VALIDATION_HOT_BLOCK", "VALIDATION_COLD_BLOCK")) != 1:
            warnings.append("Two-node lumped experiment requires exactly one interface edge.")
        C = np.asarray(matrices.get("C", []), dtype=float)
        if C.size != 2 or np.any(~np.isfinite(C)) or np.any(C <= 0.0):
            warnings.append("Two-node lumped experiment has invalid capacitance values.")
        return warnings


class GeometryDerivedContactPairExperiment(TwoBlockExchangeExperiment):
    name = GEOMETRY_CONTACT_PAIR
    description = "Two touching boxes with conductance derived from face area, center distance, material k, and interface conductance."
    equation = "G = 1 / [(d/2)/(kA) + (d/2)/(kA) + 1/(hA)]"

    def default_parameters(self) -> ThermalValidationParameters:
        params = super().default_parameters()
        params.use_octree_pipeline = False
        params.interface_model = "geometry_derived_conductance"
        params.length_mm = 10.0
        params.width_mm = 10.0
        params.height_mm = 10.0
        params.voxel_max_size_mm = 10.0
        params.duration_s = 20.0
        params.dt_s = 0.05
        params.output_sample_interval_s = 0.25
        params.absolute_tolerance_K = 1.0e-6
        params.relative_tolerance = 1.0e-9
        return params

    def _run_prepared(
        self,
        build: ValidationBuildResult,
        prepared: PreparedSimulation,
        params: ThermalValidationParameters,
    ) -> ValidationRunResult:
        result = super()._run_prepared(build, prepared, params)
        if params.interface_model == "geometry_derived_conductance":
            expected = _expected_contact_pair_conductance(params, build.model.material_library)
            actual = _interface_conductance(build.model, "VALIDATION_HOT_BLOCK", "VALIDATION_COLD_BLOCK")
            tolerance = max(abs(expected) * max(float(params.relative_tolerance), 0.0), 1.0e-12)
            result.metrics.append(
                _metric(
                    "Expected geometry conductance error",
                    actual - expected,
                    tolerance,
                    _status_abs(actual - expected, tolerance),
                    "W/K",
                )
            )
            result.status = _overall_status(result.metrics, result.warnings)
        return result


class OneDimensionalPrismExperiment(ThermalValidationExperiment):
    name = ONE_D_PRISM
    description = "Transient one-dimensional diffusion in a prism with a fixed-temperature face."
    equation = "theta = (4/pi) sum((-1)^n/(2n+1) cos((2n+1) pi x / 2L) exp(-alpha ((2n+1) pi / 2L)^2 t))"

    def default_parameters(self) -> ThermalValidationParameters:
        params = super().default_parameters()
        params.material = "Copper"
        params.length_mm = 100.0
        params.width_mm = 10.0
        params.height_mm = 10.0
        params.initial_temperature_K = 300.0
        params.reference_temperature_K = 300.0
        params.surface_temperature_K = 200.0
        params.duration_s = 2.0
        params.dt_s = 0.002
        params.output_sample_interval_s = 0.02
        params.voxel_max_size_mm = 5.0
        params.absolute_tolerance_K = 3.0
        return params

    def geometry_components(
        self, params: ThermalValidationParameters
    ) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
        return [("VALIDATION_PRISM", (0.0, 0.0, 0.0), (params.length_mm, params.width_mm, params.height_mm))]

    def build_fallback_model(self, params: ThermalValidationParameters) -> ThermalGraphModel:
        model = _new_validation_model(self.name)
        _add_box_grid(
            model,
            "VALIDATION_PRISM",
            center_mm=(0.0, 0.0, 0.0),
            size_mm=(params.length_mm, params.width_mm, params.height_mm),
            material=params.material,
            initial_temperature_K=params.initial_temperature_K,
            max_cell_size_mm=params.voxel_max_size_mm,
        )
        refresh_geometry_edges(model)
        return model

    def _run_prepared(
        self,
        build: ValidationBuildResult,
        prepared: PreparedSimulation,
        params: ThermalValidationParameters,
    ) -> ValidationRunResult:
        prism_ids = _component_node_ids(build.model, "VALIDATION_PRISM")
        fixed_ids = _nodes_on_min_x_face(build.model, prism_ids)
        if not fixed_ids:
            fixed_ids = prism_ids[:1]
        probe_rows = _probe_row_groups(build.model, prepared.node_index_by_id, prism_ids, params.probe_positions)
        probe_x_m = {label: x_mm * 1.0e-3 for label, (_rows, x_mm) in probe_rows.items()}
        times: list[float] = []
        simulated: dict[str, list[float]] = {label: [] for label in probe_rows}
        boundary_errors: list[float] = []
        energy_removed: list[float] = []
        sample = _SampleClock(params.output_sample_interval_s)
        steps = _step_count(params.duration_s, params.dt_s)
        _apply_fixed_temperature(prepared, fixed_ids, params.surface_temperature_K)
        _record_prism_sample(prepared, probe_rows, params.surface_temperature_K, times, simulated, boundary_errors, energy_removed, 0.0)
        cumulative_removed = 0.0
        for _ in range(steps):
            prepared.step_forward()
            cumulative_removed += _apply_fixed_temperature(prepared, fixed_ids, params.surface_temperature_K)
            if sample.should_sample(prepared.time_s):
                _record_prism_sample(prepared, probe_rows, params.surface_temperature_K, times, simulated, boundary_errors, energy_removed, cumulative_removed)
        time_array = np.asarray(times, dtype=float)
        analytical = self.analytical_solution(time_array, params, build)
        errors = {
            label: (np.asarray(values, dtype=float) - analytical[label]).astype(float).tolist()
            for label, values in simulated.items()
        }
        metrics: list[MetricResult] = []
        for label, values in simulated.items():
            metrics.extend(_temperature_metrics(label, np.asarray(values, dtype=float), analytical[label], params))
        if errors:
            all_errors = np.concatenate([np.asarray(values, dtype=float) for values in errors.values()])
            metrics.append(_metric("Global weighted RMSE", _rmse(all_errors, np.zeros_like(all_errors)), params.absolute_tolerance_K, _status_abs(_rmse(all_errors, np.zeros_like(all_errors)), params.absolute_tolerance_K), "K"))
        boundary = max(abs(value) for value in boundary_errors) if boundary_errors else 0.0
        metrics.append(_metric("Boundary temperature error", boundary, params.absolute_tolerance_K, _status_abs(boundary, params.absolute_tolerance_K), "K"))
        metrics.append(_metric("Insulated-end heat-flow error", _insulated_end_heat_flow(build.model, prepared, prism_ids), 1.0e-6, "PASS", "W"))
        metrics.append(_metric("Error versus voxel size", float(params.voxel_max_size_mm), None, "PASS", "mm"))
        analytical_with_positions = {key: value.astype(float).tolist() for key, value in analytical.items()}
        for label, x_m in probe_x_m.items():
            analytical_with_positions[f"{label}_actual_x_m"] = [float(x_m)]
        return ValidationRunResult(
            self.name,
            params,
            _material_properties(params.material, build.model.material_library),
            times,
            {**simulated, "boundary_temperature_error_K": boundary_errors, "fixed_boundary_removed_energy_J": energy_removed},
            analytical_with_positions,
            errors,
            metrics,
            _overall_status(metrics, build.warnings),
            list(build.warnings),
        )

    def analytical_solution(
        self,
        times_s: np.ndarray,
        params: ThermalValidationParameters,
        build: ValidationBuildResult,
    ) -> dict[str, np.ndarray]:
        prism_ids = _component_node_ids(build.model, "VALIDATION_PRISM")
        probe_rows = _probe_row_groups(build.model, {node_id: index for index, node_id in enumerate(build.model.ordered_node_ids())}, prism_ids, params.probe_positions)
        material = _material_properties(params.material, build.model.material_library)
        alpha = material["alpha_m2_s"]
        length_m = params.length_mm * 1.0e-3
        # The fixed face is imposed by clamping the min-x cells to the surface
        # temperature. Those cells are held at their CENTERS, which sit half a
        # cell in from the geometric face, so the effective Dirichlet boundary
        # is at x0 (not x=0). Shift the continuum solution onto [x0, length] to
        # remove the systematic cell-center-vs-continuum half-cell offset.
        min_x = min(_node_min_max_x(build.model.nodes[node_id])[0] for node_id in prism_ids)
        fixed_ids = _nodes_on_min_x_face(build.model, prism_ids) or prism_ids[:1]
        boundary_offset_m = (
            float(np.mean([
                float((build.model.nodes[node_id].center_mm or build.model.nodes[node_id].center)[0])
                for node_id in fixed_ids
            ])) - min_x
        ) * 1.0e-3
        length_eff = max(length_m - boundary_offset_m, 1.0e-9)
        return {
            label: prism_dirichlet_insulated_solution(
                max(x_mm * 1.0e-3 - boundary_offset_m, 0.0),
                np.asarray(times_s, dtype=float),
                length_eff,
                alpha,
                params.initial_temperature_K,
                params.surface_temperature_K,
                params.analytical_series_terms,
            )
            for label, (_rows, x_mm) in probe_rows.items()
        }


class OneDimensionalDistributedRodExperiment(ThermalValidationExperiment):
    name = DISTRIBUTED_ROD
    description = "Uniform 1D cell chain initialized to one discrete diffusion mode with a known exponential decay."
    equation = "A(t) = A0 exp[-(G/C) 2(1 - cos(pi/N)) t]"

    def default_parameters(self) -> ThermalValidationParameters:
        params = super().default_parameters()
        params.material = "Copper"
        params.length_mm = 100.0
        params.width_mm = 10.0
        params.height_mm = 10.0
        params.hot_initial_temperature_K = 300.0
        params.cold_initial_temperature_K = 200.0
        params.initial_temperature_K = 250.0
        params.duration_s = 20.0
        params.dt_s = 0.05
        params.output_sample_interval_s = 0.25
        params.voxel_max_size_mm = 10.0
        params.use_octree_pipeline = False
        params.absolute_tolerance_K = 1.0e-7
        return params

    def geometry_components(
        self, params: ThermalValidationParameters
    ) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
        return []

    def build_fallback_model(self, params: ThermalValidationParameters) -> ThermalGraphModel:
        model = _new_validation_model(self.name)
        material = _material_properties(params.material, model.material_library)
        count = max(3, int(math.ceil(float(params.length_mm) / max(float(params.voxel_max_size_mm), 1.0e-9))))
        dx_mm = float(params.length_mm) / float(count)
        area_m2 = max(0.0, float(params.width_mm) * float(params.height_mm) * 1.0e-6)
        volume_m3 = dx_mm * float(params.width_mm) * float(params.height_mm) * 1.0e-9
        capacitance = material["rho_kg_m3"] * material["cp_J_kgK"] * volume_m3
        conductance = material["k_W_mK"] * area_m2 / max(dx_mm * 1.0e-3, 1.0e-30)
        mean = 0.5 * (float(params.hot_initial_temperature_K) + float(params.cold_initial_temperature_K))
        amplitude = 0.5 * (float(params.hot_initial_temperature_K) - float(params.cold_initial_temperature_K))
        mode = _rod_first_mode(count)
        min_x = -0.5 * float(params.length_mm)
        for index in range(count):
            node_id = index + 1
            node = NodeProperties.with_material(node_id, (index, 0, 0), params.material, model.material_library)
            node.component_name = "VALIDATION_DISTRIBUTED_ROD"
            node.center_mm = (min_x + (index + 0.5) * dx_mm, 0.0, 0.0)
            node.size_mm = (dx_mm, float(params.width_mm), float(params.height_mm))
            node.side_length_m = max(node.size_mm) * 1.0e-3
            node.mass_kg = material["rho_kg_m3"] * volume_m3
            node.C_J_K = capacitance
            node.k_W_mK = material["k_W_mK"]
            node.initial_temperature_K = mean + amplitude * float(mode[index])
            node.Grad_W_K = 0.0
            node.G_rad_W_K = 0.0
            node.is_exposed = False
            model.add_node(node)
        for node_id in range(1, count):
            model.set_edge(node_id, node_id + 1, conductance, EdgeMode.LOADED_G.value, edge_type="validation_distributed_rod")
        return model

    def _run_prepared(
        self,
        build: ValidationBuildResult,
        prepared: PreparedSimulation,
        params: ThermalValidationParameters,
    ) -> ValidationRunResult:
        rod_ids = _component_node_ids(build.model, "VALIDATION_DISTRIBUTED_ROD")
        node_index = prepared.node_index_by_id
        rows = np.asarray([node_index[node_id] for node_id in rod_ids if node_id in node_index], dtype=int)
        times: list[float] = []
        simulated: dict[str, list[float]] = {"mode_amplitude_K": []}
        for node_id in rod_ids:
            simulated[f"node_{node_id}_temperature_K"] = []
        sample = _SampleClock(params.output_sample_interval_s)
        steps = _step_count(params.duration_s, params.dt_s)
        _record_distributed_rod_sample(prepared, build.model, rod_ids, rows, times, simulated)
        for _ in range(steps):
            prepared.step_forward()
            if sample.should_sample(prepared.time_s):
                _record_distributed_rod_sample(prepared, build.model, rod_ids, rows, times, simulated)
        time_array = np.asarray(times, dtype=float)
        analytical = self.analytical_solution(time_array, params, build)
        errors = {
            key: (np.asarray(values, dtype=float) - analytical[key]).astype(float).tolist()
            for key, values in simulated.items()
            if key in analytical
        }
        metrics: list[MetricResult] = []
        metrics.extend(
            _temperature_metrics(
                "rod mode amplitude",
                np.asarray(simulated["mode_amplitude_K"], dtype=float),
                analytical["mode_amplitude_K"],
                params,
            )
        )
        node_error = np.concatenate([np.asarray(values, dtype=float) for key, values in errors.items() if key.startswith("node_")])
        metrics.append(
            _metric(
                "node temperature global RMSE",
                _rmse(node_error, np.zeros_like(node_error)),
                params.absolute_tolerance_K,
                _status_abs(_rmse(node_error, np.zeros_like(node_error)), params.absolute_tolerance_K),
                "K",
            )
        )
        C = np.asarray(build.matrices["C"], dtype=float)
        initial = np.asarray([build.model.nodes[node_id].initial_temperature_K for node_id in prepared.node_ids], dtype=float)
        energy_error = float(np.dot(C, prepared.temperatures_K - initial))
        metrics.append(_metric("Total energy conservation error", energy_error, max(params.absolute_tolerance_K * float(np.sum(C)), 1.0e-12), _status_abs(energy_error, max(params.absolute_tolerance_K * float(np.sum(C)), 1.0e-12)), "J"))
        return ValidationRunResult(
            self.name,
            params,
            _material_properties(params.material, build.model.material_library),
            times,
            simulated,
            {key: value.astype(float).tolist() for key, value in analytical.items()},
            errors,
            metrics,
            _overall_status(metrics, build.warnings),
            list(build.warnings),
        )

    def analytical_solution(
        self,
        times_s: np.ndarray,
        params: ThermalValidationParameters,
        build: ValidationBuildResult,
    ) -> dict[str, np.ndarray]:
        rod_ids = _component_node_ids(build.model, "VALIDATION_DISTRIBUTED_ROD")
        count = len(rod_ids)
        mode = _rod_first_mode(count)
        C = _capacitance_sum(build.model, rod_ids) / max(float(count), 1.0)
        G = _uniform_rod_conductance(build.model, rod_ids)
        mean = _weighted_average(
            np.asarray([build.model.nodes[node_id].initial_temperature_K for node_id in rod_ids], dtype=float),
            np.asarray([build.model.nodes[node_id].C_J_K for node_id in rod_ids], dtype=float),
        )
        initial_values = np.asarray([build.model.nodes[node_id].initial_temperature_K for node_id in rod_ids], dtype=float)
        amplitude_0 = _project_rod_mode(initial_values, mode, mean)
        lam = (G / max(C, 1.0e-30)) * 2.0 * (1.0 - math.cos(math.pi / max(count, 1)))
        amplitude = amplitude_0 * np.exp(-lam * np.asarray(times_s, dtype=float))
        result: dict[str, np.ndarray] = {"mode_amplitude_K": amplitude}
        for index, node_id in enumerate(rod_ids):
            result[f"node_{node_id}_temperature_K"] = mean + amplitude * float(mode[index])
        return result


def _validation_emissivity(material_emissivity: float) -> float:
    # Floor emissivity for radiation validation so cooling is a clear signal;
    # the reference uses the same coefficient, so correctness holds either way.
    return max(0.5, float(material_emissivity))


def _enable_cell_radiation(node: NodeProperties, params: ThermalValidationParameters) -> None:
    """Set a node's radiation properties authoritatively (call after material is
    applied and after any graph save/load round-trip, which recomputes them)."""
    size = np.asarray(node.size_mm, dtype=float)
    area = 2.0 * (size[0] * size[1] + size[0] * size[2] + size[1] * size[2]) * 1.0e-6
    emissivity = _validation_emissivity(float(getattr(node, "emissivity", 0.0)))
    node.is_exposed = True
    node.emissivity = emissivity
    node.radiating_area_m2 = area
    node.G_rad_W_K = 4.0 * emissivity * STEFAN_BOLTZMANN_W_M2K4 * area * max(float(params.reference_temperature_K), 1.0) ** 3


def _add_lumped_cell(
    model: ThermalGraphModel,
    component: str,
    params: ThermalValidationParameters,
    temperature_K: float,
    *,
    node_id: int,
    coord_index: int,
    center_mm: tuple[float, float, float],
    size_mm: tuple[float, float, float],
    exposed: bool,
) -> int:
    mat = _material_properties(params.material, model.material_library)
    volume = float(size_mm[0] * size_mm[1] * size_mm[2]) * 1.0e-9
    node = NodeProperties.with_material(node_id, (coord_index, 0, 0), params.material, model.material_library)
    node.component_name = component
    node.center_mm = center_mm
    node.size_mm = size_mm
    node.side_length_m = max(size_mm) * 1.0e-3
    node.mass_kg = mat["rho_kg_m3"] * volume
    node.C_J_K = mat["rho_kg_m3"] * mat["cp_J_kgK"] * volume
    node.initial_temperature_K = float(temperature_K)
    node.Grad_W_K = 0.0
    node.G_rad_W_K = 0.0
    node.is_exposed = False
    if exposed:
        area = 2.0 * (
            size_mm[0] * size_mm[1] + size_mm[0] * size_mm[2] + size_mm[1] * size_mm[2]
        ) * 1.0e-6
        node.is_exposed = True
        node.emissivity = _validation_emissivity(mat["emissivity"])
        node.radiating_area_m2 = area
        node.G_rad_W_K = 4.0 * node.emissivity * STEFAN_BOLTZMANN_W_M2K4 * area * max(float(params.reference_temperature_K), 1.0) ** 3
    model.add_node(node)
    return node_id


def _add_forcing_heater(model: ThermalGraphModel, heater_id: int, deposit_node_id: int) -> int:
    heater = NodeProperties.with_material(heater_id, (-1_000_000, 0, 0), "Not assigned", model.material_library)
    heater.component_name = "VALIDATION_HEATER"
    heater.center_mm = (-1.0e6, 0.0, 0.0)
    heater.size_mm = (1.0, 1.0, 1.0)
    heater.side_length_m = 0.001
    heater.mass_kg = 1.0e-9
    heater.C_J_K = 1.0e-6
    heater.is_heater = True
    heater.heater = HeaterProperties(heater_id=heater_id, heater_min_power_W=0.0, heater_max_power_W=1.0e9, heater_efficiency=1.0)
    heater.power_deposition_node_ids = [int(deposit_node_id)]
    heater.power_deposition_weights = [1.0]
    heater.Grad_W_K = 0.0
    heater.G_rad_W_K = 0.0
    heater.is_exposed = False
    model.add_node(heater)
    return heater_id


def _numeric_reference_result(
    name: str,
    params: ThermalValidationParameters,
    build: ValidationBuildResult,
    times: np.ndarray,
    simulated_map: dict[str, np.ndarray],
    reference_map: dict[str, np.ndarray],
    extra_metrics: list[MetricResult] | None = None,
) -> ValidationRunResult:
    errors: dict[str, list[float]] = {}
    metrics: list[MetricResult] = []
    for label, simulated in simulated_map.items():
        simulated = np.asarray(simulated, dtype=float)
        reference = np.asarray(reference_map[label], dtype=float)
        errors[label] = (simulated - reference).astype(float).tolist()
        metrics.extend(_temperature_metrics(label, simulated, reference, params))
    metrics.extend(extra_metrics or [])
    return ValidationRunResult(
        name,
        params,
        _material_properties(params.material, build.model.material_library),
        list(np.asarray(times, dtype=float)),
        {key: np.asarray(value, dtype=float).tolist() for key, value in simulated_map.items()},
        {key: np.asarray(value, dtype=float).tolist() for key, value in reference_map.items()},
        errors,
        metrics,
        _overall_status(metrics, build.warnings),
        list(build.warnings),
    )


class RadiationCoolingExperiment(ThermalValidationExperiment):
    name = RADIATION_COOLING
    description = "Isolated radiation test: a lumped body cools to ambient by Stefan-Boltzmann radiation (no conduction)."
    equation = "C dT/dt = -eps*sigma*A*(T^4 - T_env^4); reference: independent scipy solve_ivp"

    def default_parameters(self) -> ThermalValidationParameters:
        params = ThermalValidationParameters(experiment_name=self.name)
        params.use_octree_pipeline = False
        params.material = "Copper"
        params.length_mm = params.width_mm = params.height_mm = 20.0
        params.initial_temperature_K = 300.0
        params.reference_temperature_K = 80.0
        params.duration_s = 400.0
        params.dt_s = 0.5
        params.output_sample_interval_s = 4.0
        params.use_ambient_radiation = True
        params.absolute_tolerance_K = 0.5
        return params

    def geometry_components(self, params: ThermalValidationParameters) -> list:
        return []

    def build_fallback_model(self, params: ThermalValidationParameters) -> ThermalGraphModel:
        model = _new_validation_model(self.name)
        _add_lumped_cell(
            model, "VALIDATION_RADIATOR", params, params.initial_temperature_K,
            node_id=1, coord_index=0, center_mm=(0.0, 0.0, 0.0),
            size_mm=(params.length_mm, params.width_mm, params.height_mm), exposed=True,
        )
        return model

    def configure_model(self, model, matrices, params) -> None:
        library = model.material_library or default_material_library()
        for node_id in _component_node_ids(model, "VALIDATION_RADIATOR"):
            node = model.nodes[node_id]
            _apply_material(node, params.material, library)
            node.initial_temperature_K = float(params.initial_temperature_K)
            _enable_cell_radiation(node, params)
        model.metadata.T_sur_K = float(params.reference_temperature_K)
        model.metadata.edge_mode = EdgeMode.AUTO.value

    def requested_volume_m3(self, params) -> float:
        return params.length_mm * params.width_mm * params.height_mm * 1.0e-9

    def simulation_parameters(self, params) -> SimulationParameters:
        sim = super().simulation_parameters(params)
        sim.use_ambient_radiation = True
        sim.T_env_K = float(params.reference_temperature_K)
        return sim

    def _run_prepared(self, build, prepared, params) -> ValidationRunResult:
        rows = np.asarray([prepared.node_index_by_id[nid] for nid in _component_node_ids(build.model, "VALIDATION_RADIATOR")], dtype=int)
        C = np.asarray(build.matrices["C"], dtype=float)
        times: list[float] = []
        sim_avg: list[float] = []
        sample = _SampleClock(params.output_sample_interval_s)
        steps = _step_count(params.duration_s, params.dt_s)

        def record() -> None:
            times.append(float(prepared.time_s))
            sim_avg.append(float(np.average(prepared.temperatures_K[rows], weights=C[rows])))

        record()
        for _ in range(steps):
            prepared.step_forward()
            if sample.should_sample(prepared.time_s):
                record()
        time_array = np.asarray(times, dtype=float)
        reference = _solve_ivp_reference(prepared, None, time_array)
        reference_avg = np.average(reference[:, rows], axis=1, weights=C[rows])
        return _numeric_reference_result(
            self.name, params, build, time_array,
            {"average_temperature_K": np.asarray(sim_avg, dtype=float)},
            {"average_temperature_K": reference_avg},
        )


class TemperatureDependentHeatingExperiment(ThermalValidationExperiment):
    name = TDEP_HEATING
    description = "Isolated cp(T) test: an adiabatic lumped body is heated by a constant power with temperature-dependent heat capacity."
    equation = "integral rho*V*cp(T) dT = P*t; reference: independent scipy solve_ivp with cp(T)"

    def default_parameters(self) -> ThermalValidationParameters:
        params = ThermalValidationParameters(experiment_name=self.name)
        params.use_octree_pipeline = False
        params.material = "Copper"
        params.length_mm = params.width_mm = params.height_mm = 20.0
        params.initial_temperature_K = 20.0
        params.duration_s = 60.0
        params.dt_s = 0.2
        params.output_sample_interval_s = 1.0
        params.heater_power_W = 5.0
        params.use_temperature_dependent_properties = True
        params.absolute_tolerance_K = 0.5
        return params

    def geometry_components(self, params) -> list:
        return []

    def build_fallback_model(self, params) -> ThermalGraphModel:
        model = _new_validation_model(self.name)
        _add_lumped_cell(
            model, "VALIDATION_BLOCK", params, params.initial_temperature_K,
            node_id=1, coord_index=0, center_mm=(0.0, 0.0, 0.0),
            size_mm=(params.length_mm, params.width_mm, params.height_mm), exposed=False,
        )
        _add_forcing_heater(model, heater_id=2, deposit_node_id=1)
        return model

    def requested_volume_m3(self, params) -> float:
        return params.length_mm * params.width_mm * params.height_mm * 1.0e-9

    def _run_prepared(self, build, prepared, params) -> ValidationRunResult:
        body_ids = _component_node_ids(build.model, "VALIDATION_BLOCK")
        heater_ids = [int(nid) for nid, node in build.model.nodes.items() if node.is_heater]
        rows = np.asarray([prepared.node_index_by_id[nid] for nid in body_ids], dtype=int)
        C = np.asarray(build.matrices["C"], dtype=float)
        times: list[float] = []
        sim_avg: list[float] = []
        sample = _SampleClock(params.output_sample_interval_s)
        steps = _step_count(params.duration_s, params.dt_s)

        def record() -> None:
            times.append(float(prepared.time_s))
            sim_avg.append(float(np.average(prepared.temperatures_K[rows], weights=C[rows])))

        record()
        for _ in range(steps):
            _advance_forced_power(prepared, {heater_ids[0]: params.heater_power_W})
            if sample.should_sample(prepared.time_s):
                record()
        time_array = np.asarray(times, dtype=float)
        reference = _solve_ivp_reference(prepared, {int(body_ids[0]): float(params.heater_power_W)}, time_array)
        reference_avg = np.average(reference[:, rows], axis=1, weights=C[rows])
        return _numeric_reference_result(
            self.name, params, build, time_array,
            {"average_temperature_K": np.asarray(sim_avg, dtype=float)},
            {"average_temperature_K": reference_avg},
        )


class CryoRegimeExperiment(ThermalValidationExperiment):
    name = CRYO_REGIME
    description = "Overarching cryo test: a conductive rod with a localized heater, ambient radiation, and temperature-dependent cp(T)/k(T)."
    equation = "C(T) dT/dt = -L(T) T + P_heater + eps*sigma*A*(T_env^4 - T^4); reference: independent scipy solve_ivp"

    def default_parameters(self) -> ThermalValidationParameters:
        params = ThermalValidationParameters(experiment_name=self.name)
        params.use_octree_pipeline = False
        params.material = "Copper"
        params.length_mm = 100.0
        params.width_mm = params.height_mm = 10.0
        params.initial_temperature_K = 40.0
        params.reference_temperature_K = 40.0
        params.duration_s = 60.0
        params.dt_s = 0.2
        params.output_sample_interval_s = 1.0
        params.heater_power_W = 2.0
        params.voxel_max_size_mm = 20.0
        params.use_ambient_radiation = True
        params.use_temperature_dependent_properties = True
        params.absolute_tolerance_K = 0.5
        return params

    def geometry_components(self, params) -> list:
        return []

    def build_fallback_model(self, params) -> ThermalGraphModel:
        model = _new_validation_model(self.name)
        mat = _material_properties(params.material, model.material_library)
        count = max(3, int(math.ceil(float(params.length_mm) / max(float(params.voxel_max_size_mm), 1.0e-9))))
        dx_mm = float(params.length_mm) / float(count)
        area_m2 = float(params.width_mm * params.height_mm) * 1.0e-6
        conductance = mat["k_W_mK"] * area_m2 / max(dx_mm * 1.0e-3, 1.0e-30)
        min_x = -0.5 * float(params.length_mm)
        for index in range(count):
            _add_lumped_cell(
                model, "VALIDATION_ROD", params, params.initial_temperature_K,
                node_id=index + 1, coord_index=index,
                center_mm=(min_x + (index + 0.5) * dx_mm, 0.0, 0.0),
                size_mm=(dx_mm, float(params.width_mm), float(params.height_mm)), exposed=True,
            )
        for node_id in range(1, count):
            model.set_edge(node_id, node_id + 1, conductance, EdgeMode.LOADED_G.value, edge_type="validation_rod")
        _add_forcing_heater(model, heater_id=count + 1, deposit_node_id=1)
        return model

    def configure_model(self, model, matrices, params) -> None:
        library = model.material_library or default_material_library()
        for node_id in _component_node_ids(model, "VALIDATION_ROD"):
            node = model.nodes[node_id]
            _apply_material(node, params.material, library)
            node.initial_temperature_K = float(params.initial_temperature_K)
            _enable_cell_radiation(node, params)
        model.metadata.T_sur_K = float(params.reference_temperature_K)
        model.metadata.edge_mode = EdgeMode.LOADED_G.value

    def requested_volume_m3(self, params) -> float:
        return params.length_mm * params.width_mm * params.height_mm * 1.0e-9

    def simulation_parameters(self, params) -> SimulationParameters:
        sim = super().simulation_parameters(params)
        sim.use_ambient_radiation = True
        sim.T_env_K = float(params.reference_temperature_K)
        return sim

    def _run_prepared(self, build, prepared, params) -> ValidationRunResult:
        rod_ids = _component_node_ids(build.model, "VALIDATION_ROD")
        heater_ids = [int(nid) for nid, node in build.model.nodes.items() if node.is_heater]
        rows = np.asarray([prepared.node_index_by_id[nid] for nid in rod_ids], dtype=int)
        C = np.asarray(build.matrices["C"], dtype=float)
        times: list[float] = []
        sim_avg: list[float] = []
        sim_max: list[float] = []
        sample = _SampleClock(params.output_sample_interval_s)
        steps = _step_count(params.duration_s, params.dt_s)

        def record() -> None:
            times.append(float(prepared.time_s))
            sim_avg.append(float(np.average(prepared.temperatures_K[rows], weights=C[rows])))
            sim_max.append(float(np.max(prepared.temperatures_K[rows])))

        record()
        for _ in range(steps):
            _advance_forced_power(prepared, {heater_ids[0]: params.heater_power_W})
            if sample.should_sample(prepared.time_s):
                record()
        time_array = np.asarray(times, dtype=float)
        reference = _solve_ivp_reference(prepared, {int(rod_ids[0]): float(params.heater_power_W)}, time_array)
        reference_avg = np.average(reference[:, rows], axis=1, weights=C[rows])
        reference_max = np.max(reference[:, rows], axis=1)
        return _numeric_reference_result(
            self.name, params, build, time_array,
            {"average_temperature_K": np.asarray(sim_avg, dtype=float), "hot_end_temperature_K": np.asarray(sim_max, dtype=float)},
            {"average_temperature_K": reference_avg, "hot_end_temperature_K": reference_max},
        )


# --- Sandia thermal validation challenge (experimental comparison) ----------
# 1-D slab, constant heat flux q at x=0, insulated back face at x=L, constant
# properties. Problem: Dowding, Pilch & Hills, CMAME 197 (2008). The accreditation
# configuration and its experimental temperature profile at t=1000 s were digitized
# from Chantrasmi et al., CTR Annual Research Briefs 2006, Fig. 3 (top); the raw
# ensembles are not openly available. See validation-assets/sandia_thermal_challenge.json.
SANDIA_CHALLENGE_FLUX_W_M2 = 3000.0
SANDIA_CHALLENGE_THICKNESS_M = 0.019
SANDIA_CHALLENGE_EVAL_TIME_S = 1000.0
SANDIA_CHALLENGE_MATERIAL = "SandiaChallengeSlab"
SANDIA_CHALLENGE_K_W_MK = 0.06        # mean of the measured k(T) (CTR Fig. 4)
SANDIA_CHALLENGE_RHO_KG_M3 = 2100.0   # rho*cp = 4.2e5 J/(m^3 K) reproduces the
SANDIA_CHALLENGE_CP_J_KGK = 200.0     # mean profile / experimental points at 1000 s
# Experimental profile at t=1000 s (deg C), digitized; two runs at x=0, L/2, L.
SANDIA_CHALLENGE_EXP_POSITIONS_M = (0.0, 0.0095, 0.019)
SANDIA_CHALLENGE_EXP_LABELS = ("x=0", "x=L/2", "x=L")
SANDIA_CHALLENGE_EXP_RUN1_C = (718.0, 370.0, 247.0)
SANDIA_CHALLENGE_EXP_RUN2_C = (675.0, 348.0, 225.0)
SANDIA_CHALLENGE_DIGITIZE_UNCERTAINTY_K = 20.0  # reading + surface-probe offset margin


def sandia_challenge_flux_solution(
    x_m: float,
    times_s: np.ndarray,
    k_W_mK: float,
    rho_cp_J_m3K: float,
    q_W_m2: float,
    thickness_m: float,
    initial_temperature_K: float,
    series_terms: int = 200,
) -> np.ndarray:
    """Closed-form challenge model: constant-flux slab insulated at the back face.

    T(x,t) = T0 + (qL/k)*[ Fo + 1/3 - x/L + x^2/(2L^2)
                           - (2/pi^2) sum_{n>=1} (1/n^2) e^{-n^2 pi^2 Fo} cos(n pi x/L) ],
    with Fo = k t /(rho_cp L^2). Reduces to T0 at t=0 and to a linear-in-time rise
    plus a fixed spatial profile once the transient decays.

    The series converges slowly near the flux turn-on (Fo -> 0): a truncated sum
    Gibbs-oscillates there, and with few terms and/or a low (cryogenic) T0 the
    ripple can dip below T0 -- even below 0 K, which is non-physical. For this
    problem (constant flux IN, insulated back, no losses) the exact temperature
    is monotonic and everywhere >= T0, so we clamp to that exact lower bound.
    This removes the truncation undershoot without affecting the converged
    late-time solution the experiment actually validates against.
    """
    x = float(x_m)
    length = float(thickness_m)
    t = np.asarray(times_s, dtype=float)
    fourier = float(k_W_mK) * t / (float(rho_cp_J_m3K) * length * length)
    series = np.zeros_like(t)
    for n in range(1, int(series_terms) + 1):
        series += (1.0 / n ** 2) * np.exp(-(n ** 2) * (np.pi ** 2) * fourier) * math.cos(n * math.pi * x / length)
    theta = fourier + 1.0 / 3.0 - x / length + x * x / (2.0 * length * length) - (2.0 / np.pi ** 2) * series
    T0 = float(initial_temperature_K)
    return np.maximum(T0 + (float(q_W_m2) * length / float(k_W_mK)) * theta, T0)


class SandiaThermalChallengeExperiment(ThermalValidationExperiment):
    name = SANDIA_THERMAL_CHALLENGE
    description = (
        "Experimental comparison: the Sandia thermal validation challenge (1-D slab, constant "
        "surface heat flux, insulated back). Compares the simulator against BOTH the challenge "
        "closed-form model and digitized experimental temperatures at t=1000 s."
    )
    equation = "constant-flux slab: T(x,t)=T0+(qL/k)[Fo+1/3-x/L+x^2/2L^2-(2/pi^2)Sum(1/n^2)e^{-n^2 pi^2 Fo}cos(n pi x/L)]"

    def default_parameters(self) -> ThermalValidationParameters:
        params = ThermalValidationParameters(experiment_name=self.name)
        params.use_octree_pipeline = False
        params.material = SANDIA_CHALLENGE_MATERIAL
        params.length_mm = SANDIA_CHALLENGE_THICKNESS_M * 1.0e3
        params.width_mm = params.height_mm = 10.0
        params.initial_temperature_K = 298.15  # 25 C
        params.duration_s = SANDIA_CHALLENGE_EVAL_TIME_S
        params.dt_s = 1.0
        params.output_sample_interval_s = 20.0
        params.voxel_max_size_mm = 0.25  # ~76 cells across the 19 mm slab
        params.probe_positions = (0.0, 0.5, 1.0)
        params.use_ambient_radiation = False
        params.use_temperature_dependent_properties = False
        params.absolute_tolerance_K = 8.0  # sim-vs-closed-form (spatial discretization)
        return params

    def geometry_components(self, params: ThermalValidationParameters) -> list:
        return []

    def _register_material(self, model: ThermalGraphModel, params: ThermalValidationParameters) -> None:
        """Register the intrinsic challenge material AND force the experiment to
        use it. The material (k, rho*cp) is part of the problem definition -- it
        is tied to the closed-form model and the digitized experimental data --
        so it must NOT be overridden by the tab's material selector (which
        defaults to Copper; using it makes the slab isothermal and cold, giving
        large negative errors vs the experiment)."""
        library = model.material_library
        if library is None:
            library = default_material_library()
            model.material_library = library
        library[SANDIA_CHALLENGE_MATERIAL] = {
            "rho_kg_m3": SANDIA_CHALLENGE_RHO_KG_M3,
            "cp_J_kgK": SANDIA_CHALLENGE_CP_J_KGK,
            "k_W_mK": SANDIA_CHALLENGE_K_W_MK,
            "emissivity": 0.0,
        }
        params.material = SANDIA_CHALLENGE_MATERIAL

    def build_fallback_model(self, params: ThermalValidationParameters) -> ThermalGraphModel:
        model = _new_validation_model(self.name)
        self._register_material(model, params)
        mat = _material_properties(params.material, model.material_library)
        count = max(4, int(math.ceil(float(params.length_mm) / max(float(params.voxel_max_size_mm), 1.0e-9))))
        dx_mm = float(params.length_mm) / float(count)
        area_m2 = float(params.width_mm * params.height_mm) * 1.0e-6
        conductance = mat["k_W_mK"] * area_m2 / max(dx_mm * 1.0e-3, 1.0e-30)
        min_x = -0.5 * float(params.length_mm)
        for index in range(count):
            _add_lumped_cell(
                model, "VALIDATION_SLAB", params, params.initial_temperature_K,
                node_id=index + 1, coord_index=index,
                center_mm=(min_x + (index + 0.5) * dx_mm, 0.0, 0.0),
                size_mm=(dx_mm, float(params.width_mm), float(params.height_mm)), exposed=False,
            )
        for node_id in range(1, count):
            model.set_edge(node_id, node_id + 1, conductance, EdgeMode.LOADED_G.value, edge_type="validation_slab")
        # Surface flux enters the x=0 (min-x) cell, which is node 1.
        _add_forcing_heater(model, heater_id=count + 1, deposit_node_id=1)
        return model

    def configure_model(self, model, matrices, params) -> None:
        self._register_material(model, params)
        library = model.material_library or default_material_library()
        for node_id in _component_node_ids(model, "VALIDATION_SLAB"):
            node = model.nodes[node_id]
            _apply_material(node, params.material, library)
            node.initial_temperature_K = float(params.initial_temperature_K)
            node.is_exposed = False
            node.emissivity = 0.0
            node.G_rad_W_K = 0.0
        model.metadata.edge_mode = EdgeMode.LOADED_G.value

    def requested_volume_m3(self, params) -> float:
        return params.length_mm * params.width_mm * params.height_mm * 1.0e-9

    def _run_prepared(self, build, prepared, params) -> ValidationRunResult:
        self._register_material(build.model, params)  # reference must use the challenge material
        slab_ids = _component_node_ids(build.model, "VALIDATION_SLAB")
        heater_ids = [int(nid) for nid, node in build.model.nodes.items() if node.is_heater]
        centers = {nid: float((build.model.nodes[nid].center_mm or build.model.nodes[nid].center)[0]) for nid in slab_ids}
        faces = [_node_min_max_x(build.model.nodes[nid]) for nid in slab_ids]
        min_face = min(face[0] for face in faces)
        max_face = max(face[1] for face in faces)
        length_mm = max(max_face - min_face, 1.0e-9)

        def nearest(fraction: float) -> int:
            target = min_face + float(fraction) * length_mm
            return min(slab_ids, key=lambda nid: abs(centers[nid] - target))

        probe_ids = {label: nearest(frac) for label, frac in zip(SANDIA_CHALLENGE_EXP_LABELS, (0.0, 0.5, 1.0))}
        probe_x_m = {label: (centers[nid] - min_face) * 1.0e-3 for label, nid in probe_ids.items()}
        rows = {label: prepared.node_index_by_id[nid] for label, nid in probe_ids.items()}
        power_W = SANDIA_CHALLENGE_FLUX_W_M2 * float(params.width_mm * params.height_mm) * 1.0e-6

        times: list[float] = []
        simulated: dict[str, list[float]] = {label: [] for label in probe_ids}
        sample = _SampleClock(params.output_sample_interval_s)
        steps = _step_count(params.duration_s, params.dt_s)

        def record() -> None:
            times.append(float(prepared.time_s))
            temps = np.asarray(prepared.temperatures_K, dtype=float)
            for label, row in rows.items():
                simulated[label].append(float(temps[row]))

        record()
        for _ in range(steps):
            _advance_forced_power(prepared, {heater_ids[0]: power_W})
            if sample.should_sample(prepared.time_s):
                record()
        time_array = np.asarray(times, dtype=float)

        mat = _material_properties(params.material, build.model.material_library)
        k = mat["k_W_mK"]
        rho_cp = mat["rho_kg_m3"] * mat["cp_J_kgK"]
        T0 = float(params.initial_temperature_K)
        # Rigorous reference: closed-form model at each probe's actual cell-center x.
        reference = {
            label: sandia_challenge_flux_solution(
                probe_x_m[label], time_array, k, rho_cp,
                SANDIA_CHALLENGE_FLUX_W_M2, SANDIA_CHALLENGE_THICKNESS_M, T0, params.analytical_series_terms,
            )
            for label in probe_ids
        }
        simulated_map = {label: np.asarray(simulated[label], dtype=float) for label in probe_ids}

        # Physics check: simulator vs the digitized experimental profile at t=1000 s.
        eval_index = int(np.argmin(np.abs(time_array - SANDIA_CHALLENGE_EVAL_TIME_S))) if time_array.size else 0
        extra_metrics: list[MetricResult] = []
        for j, label in enumerate(SANDIA_CHALLENGE_EXP_LABELS):
            run1_K = SANDIA_CHALLENGE_EXP_RUN1_C[j] + 273.15
            run2_K = SANDIA_CHALLENGE_EXP_RUN2_C[j] + 273.15
            mean_K = 0.5 * (run1_K + run2_K)
            sim_K = float(simulated_map[label][eval_index]) if simulated_map[label].size else float("nan")
            err = sim_K - mean_K
            # Tolerance = experiment-to-experiment scatter + digitization/offset margin.
            tolerance = abs(run1_K - run2_K) + SANDIA_CHALLENGE_DIGITIZE_UNCERTAINTY_K
            extra_metrics.append(_metric(
                f"{label} vs experiment at t=1000 s (mean of 2 runs)", err, tolerance, _status_abs(err, tolerance), "K",
            ))
        return _numeric_reference_result(
            self.name, params, build, time_array, simulated_map, reference, extra_metrics=extra_metrics,
        )


class RadiativeCouplingExperiment(ThermalValidationExperiment):
    name = RADIATIVE_COUPLING
    description = (
        "Surface-to-surface radiative coupling: two facing plates exchange heat only by "
        "gray-diffuse radiation (no conduction, no ambient) and relax to a common temperature. "
        "Validates the coupled sigma*(W T^4 - D T^4) exchange term against an independent solve_ivp."
    )
    equation = "C_i dT_i/dt = sigma * A * scriptF * (T_j^4 - T_i^4); scriptF = 1/(1/e_i + 1/e_j - 1)"
    EMISSIVITY = 0.8

    def default_parameters(self) -> ThermalValidationParameters:
        params = ThermalValidationParameters(experiment_name=self.name)
        params.use_octree_pipeline = False
        params.material = "Copper"
        params.length_mm = 2.0            # plate thickness (x)
        params.width_mm = params.height_mm = 100.0  # facing area = width x height
        params.hot_initial_temperature_K = 400.0
        params.cold_initial_temperature_K = 200.0
        params.duration_s = 2000.0
        params.dt_s = 2.0
        params.output_sample_interval_s = 40.0
        params.use_ambient_radiation = True
        params.absolute_tolerance_K = 0.5
        return params

    def geometry_components(self, params: ThermalValidationParameters) -> list:
        return []

    def build_fallback_model(self, params: ThermalValidationParameters) -> ThermalGraphModel:
        model = _new_validation_model(self.name)
        gap_mm = 10.0
        offset = 0.5 * (float(params.length_mm) + gap_mm)
        size = (float(params.length_mm), float(params.width_mm), float(params.height_mm))
        _add_lumped_cell(
            model, "VALIDATION_PLATE_HOT", params, params.hot_initial_temperature_K,
            node_id=1, coord_index=0, center_mm=(-offset, 0.0, 0.0), size_mm=size, exposed=False,
        )
        _add_lumped_cell(
            model, "VALIDATION_PLATE_COLD", params, params.cold_initial_temperature_K,
            node_id=2, coord_index=1, center_mm=(offset, 0.0, 0.0), size_mm=size, exposed=False,
        )
        return model  # no conduction edges: the plates are coupled only by radiation

    def _exchange_area_m2(self, params: ThermalValidationParameters) -> float:
        # Infinite-parallel-plate gray exchange factor (view factor F=1):
        # scriptF = 1/(1/e1 + 1/e2 - 1); exchange area G = A * scriptF.
        facing_area_m2 = float(params.width_mm * params.height_mm) * 1.0e-6
        script_f = 1.0 / (2.0 / self.EMISSIVITY - 1.0)
        return facing_area_m2 * script_f

    def configure_model(self, model, matrices, params) -> None:
        library = model.material_library or default_material_library()
        hot = _component_node_ids(model, "VALIDATION_PLATE_HOT")
        cold = _component_node_ids(model, "VALIDATION_PLATE_COLD")
        for node_id in hot + cold:
            node = model.nodes[node_id]
            _apply_material(node, params.material, library)
            # No ambient radiation: the plates exchange ONLY with each other via
            # the explicit exchange link. Zero the env-radiation inputs (the
            # save/load round-trip can otherwise re-derive a radiating area).
            node.is_exposed = False
            node.emissivity = self.EMISSIVITY  # used only to document the surface
            node.radiating_area_m2 = 0.0
            node.G_rad_W_K = 0.0
            node.Grad_W_K = 0.0
        for node_id in hot:
            model.nodes[node_id].initial_temperature_K = float(params.hot_initial_temperature_K)
        for node_id in cold:
            model.nodes[node_id].initial_temperature_K = float(params.cold_initial_temperature_K)
        # Radiative exchange links (set here, after the build save/load round-trip,
        # so they survive on the reloaded model). Split evenly if a plate is
        # discretized into several cells (single lumped cells here).
        total_G = self._exchange_area_m2(params)
        pairs = [(i, j) for i in hot for j in cold]
        per_pair = total_G / max(len(pairs), 1)
        model.radiation_exchange_links = [(int(i), int(j), per_pair) for i, j in pairs]
        model.metadata.edge_mode = EdgeMode.AUTO.value

    def requested_volume_m3(self, params) -> float:
        return 2.0 * params.length_mm * params.width_mm * params.height_mm * 1.0e-9

    def simulation_parameters(self, params) -> SimulationParameters:
        sim = super().simulation_parameters(params)
        sim.use_ambient_radiation = True
        return sim

    def _run_prepared(self, build, prepared, params) -> ValidationRunResult:
        hot = _component_node_ids(build.model, "VALIDATION_PLATE_HOT")
        cold = _component_node_ids(build.model, "VALIDATION_PLATE_COLD")
        rows_hot = np.asarray([prepared.node_index_by_id[i] for i in hot], dtype=int)
        rows_cold = np.asarray([prepared.node_index_by_id[i] for i in cold], dtype=int)
        C = np.asarray(build.matrices["C"], dtype=float)
        times: list[float] = []
        sim_hot: list[float] = []
        sim_cold: list[float] = []
        sample = _SampleClock(params.output_sample_interval_s)
        steps = _step_count(params.duration_s, params.dt_s)

        def record() -> None:
            times.append(float(prepared.time_s))
            temps = np.asarray(prepared.temperatures_K, dtype=float)
            sim_hot.append(float(np.average(temps[rows_hot], weights=C[rows_hot])))
            sim_cold.append(float(np.average(temps[rows_cold], weights=C[rows_cold])))

        record()
        for _ in range(steps):
            prepared.step_forward()
            if sample.should_sample(prepared.time_s):
                record()
        time_array = np.asarray(times, dtype=float)
        reference = _solve_ivp_reference(prepared, None, time_array)
        ref_hot = np.average(reference[:, rows_hot], axis=1, weights=C[rows_hot])
        ref_cold = np.average(reference[:, rows_cold], axis=1, weights=C[rows_cold])
        equilibrium = float((C[rows_hot].sum() * params.hot_initial_temperature_K
                             + C[rows_cold].sum() * params.cold_initial_temperature_K)
                            / (C[rows_hot].sum() + C[rows_cold].sum()))
        extra = [_metric("Energy-conserving equilibrium temperature", equilibrium, None, "PASS", "K")]
        return _numeric_reference_result(
            self.name, params, build, time_array,
            {"hot_plate_temperature_K": np.asarray(sim_hot, dtype=float),
             "cold_plate_temperature_K": np.asarray(sim_cold, dtype=float)},
            {"hot_plate_temperature_K": ref_hot, "cold_plate_temperature_K": ref_cold},
            extra_metrics=extra,
        )


def _copper_discrete_steady_profile(
    hot_K: float, cold_K: float, n_cells: int, cell_length_m: float, area_m2: float, rrr: int
) -> np.ndarray:
    """Independent steady-state profile for a uniform 1-D chain with k(T).

    Reference for the k(T) conduction test. Reimplements the finite-volume face
    conductance ``G = A / (dx/2/k_i + dx/2/k_j)`` (harmonic mean of the two cell
    conductivities) directly from the material's k(T) curve -- deliberately NOT
    reusing the solver's operator code -- and relaxes the interior nodes to the
    steady state (each interior node is the conductance-weighted mean of its
    neighbours) with the two ends pinned. Comparing the solver to THIS validates
    that the operator applies k(T) correctly in a gradient, with zero spatial-
    discretisation error relative to the sim (same node chain). A mis-applied k(T)
    (wrong RRR, arithmetic instead of harmonic mean, wrong evaluation temperature)
    bends the profile far outside the tolerance."""
    from . import material_properties_cryo as mp

    n = max(2, int(n_cells))
    temperatures = np.linspace(float(hot_K), float(cold_K), n)
    for _ in range(20000):
        k = np.asarray(mp.thermal_conductivity_W_mK("Copper", temperatures, rrr=int(rrr)), dtype=float)
        face = float(area_m2) / (0.5 * float(cell_length_m) / k[:-1] + 0.5 * float(cell_length_m) / k[1:])
        updated = temperatures.copy()
        updated[1:-1] = (face[:-1] * temperatures[:-2] + face[1:] * temperatures[2:]) / (face[:-1] + face[1:])
        updated[0], updated[-1] = float(hot_K), float(cold_K)
        if float(np.max(np.abs(updated - temperatures))) < 1.0e-10:
            temperatures = updated
            break
        temperatures = 0.3 * temperatures + 0.7 * updated
    return temperatures


class TemperatureDependentConductionExperiment(ThermalValidationExperiment):
    name = TDEP_CONDUCTION
    description = (
        "Isolated k(T) test: steady conduction along a rod with fixed hot/cold ends "
        "and temperature-dependent conductivity, compared to the analytical Kirchhoff-"
        "transform profile of the solver's own k(T). X axis is position along the rod."
    )
    equation = "d/dx(k(T) dT/dx) = 0  ->  psi(T) = integral k dT is linear in x"

    def default_parameters(self) -> ThermalValidationParameters:
        params = ThermalValidationParameters(experiment_name=self.name)
        params.use_octree_pipeline = False
        params.material = "Copper"
        params.length_mm = 100.0
        params.width_mm = params.height_mm = 2.0  # single-cell cross-section -> a 1-D chain
        params.voxel_max_size_mm = 2.0  # 50 cells along the rod
        params.initial_temperature_K = 170.0
        params.hot_initial_temperature_K = 300.0  # hot fixed end
        params.surface_temperature_K = 40.0       # cold fixed end (operating cryo range, tractable cp)
        # dt is fine because deep-cryo cp collapse makes the conduction stiff; the
        # Dirichlet ends are held by overwrite, whose splitting leaves an O(dt) bias
        # (~5 K here) -- far inside the ~29 K a mis-applied k(T) would produce.
        params.duration_s = 120.0
        params.dt_s = 0.01
        params.output_sample_interval_s = 120.0
        params.use_temperature_dependent_properties = True
        params.copper_rrr = 100
        params.gpu_solver_enabled = False  # 50-node problem: CPU beats GPU launch overhead
        params.absolute_tolerance_K = 12.0
        return params

    def geometry_components(self, params) -> list:
        return []

    def build_fallback_model(self, params) -> ThermalGraphModel:
        model = _new_validation_model(self.name)
        _add_box_grid(
            model, "VALIDATION_KT_ROD", center_mm=(0.0, 0.0, 0.0),
            size_mm=(params.length_mm, params.width_mm, params.height_mm),
            material=params.material, initial_temperature_K=params.initial_temperature_K,
            max_cell_size_mm=params.voxel_max_size_mm,
        )
        refresh_geometry_edges(model)
        return model

    def requested_volume_m3(self, params) -> float:
        return params.length_mm * params.width_mm * params.height_mm * 1.0e-9

    def simulation_parameters(self, params) -> SimulationParameters:
        sim = super().simulation_parameters(params)
        sim.use_temperature_dependent_properties = True
        sim.copper_rrr = int(params.copper_rrr)
        return sim

    def _run_prepared(self, build, prepared, params) -> ValidationRunResult:
        rod_ids = _component_node_ids(build.model, "VALIDATION_KT_ROD")
        centers = {nid: float(build.model.nodes[nid].center_mm[0]) for nid in rod_ids}
        ordered = sorted(rod_ids, key=lambda nid: centers[nid])
        hot_id, cold_id = ordered[0], ordered[-1]
        hot_K = float(params.hot_initial_temperature_K)
        cold_K = float(params.surface_temperature_K)
        steps = _step_count(params.duration_s, params.dt_s)
        _apply_fixed_temperature(prepared, [hot_id], hot_K)
        _apply_fixed_temperature(prepared, [cold_id], cold_K)
        for _ in range(steps):
            prepared.step_forward()
            _apply_fixed_temperature(prepared, [hot_id], hot_K)
            _apply_fixed_temperature(prepared, [cold_id], cold_K)
        interior = ordered[1:-1]
        rows = np.asarray([prepared.node_index_by_id[nid] for nid in interior], dtype=int)
        simulated = np.asarray(prepared.temperatures_K, dtype=float)[rows]
        cell_length_m = float(params.length_mm) / len(ordered) * 1.0e-3
        area_m2 = float(params.width_mm) * float(params.height_mm) * 1.0e-6
        reference_full = _copper_discrete_steady_profile(
            hot_K, cold_K, len(ordered), cell_length_m, area_m2, int(params.copper_rrr)
        )
        analytical = reference_full[1:-1]
        position_m = np.asarray([centers[nid] * 1.0e-3 for nid in interior], dtype=float)
        return _numeric_reference_result(
            self.name, params, build, position_m,
            {"steady_state_temperature_K": simulated},
            {"steady_state_temperature_K": analytical},
        )


class CryocoolerLiftCurveExperiment(ThermalValidationExperiment):
    name = CRYOCOOLER_LIFT
    description = (
        "Isolated cryocooler test: a cold head under a constant heat load settles where "
        "the PT60 capacity equals the load. Sweeping the load reproduces the manufacturer "
        "lift curve. X axis is applied heat load (W)."
    )
    equation = "steady state: cooling_capacity_W(T*) = P  <=>  T* = T_lift(P)"

    def default_parameters(self) -> ThermalValidationParameters:
        params = ThermalValidationParameters(experiment_name=self.name)
        params.use_octree_pipeline = False
        params.material = "Copper"
        params.length_mm = params.width_mm = params.height_mm = 30.0
        params.initial_temperature_K = 50.0
        params.duration_s = 300.0
        params.dt_s = 1.0
        params.gpu_solver_enabled = False  # single-node problem: CPU is faster
        params.absolute_tolerance_K = 1.0
        return params

    def geometry_components(self, params) -> list:
        return []

    def build_fallback_model(self, params) -> ThermalGraphModel:
        model = _new_validation_model(self.name)
        _add_lumped_cell(
            model, "VALIDATION_COLDHEAD", params, params.initial_temperature_K,
            node_id=1, coord_index=0, center_mm=(0.0, 0.0, 0.0),
            size_mm=(params.length_mm, params.width_mm, params.height_mm), exposed=False,
        )
        _add_forcing_heater(model, heater_id=2, deposit_node_id=1)
        return model

    def configure_model(self, model, matrices, params) -> None:
        library = model.material_library or default_material_library()
        for node_id in _component_node_ids(model, "VALIDATION_COLDHEAD"):
            node = model.nodes[node_id]
            _apply_material(node, params.material, library)
            node.initial_temperature_K = float(params.initial_temperature_K)
            node.is_exposed = False
            node.G_rad_W_K = 0.0
            node.Grad_W_K = 0.0
            node.has_cryocooler = True
            node.cryocooler_enabled = True
            node.cryocooler_id = "VALIDATION_PT60"
        model.metadata.edge_mode = EdgeMode.AUTO.value

    def requested_volume_m3(self, params) -> float:
        return params.length_mm * params.width_mm * params.height_mm * 1.0e-9

    def simulation_parameters(self, params) -> SimulationParameters:
        sim = super().simulation_parameters(params)
        sim.cryocooler_enabled = True
        sim.cryocooler_capacity_scale = 1.0
        sim.cryocooler_max_power_W = 150.0
        return sim

    def _run_prepared(self, build, prepared, params) -> ValidationRunResult:
        from .cryocooler import PT60LiftCurve

        head_id = _component_node_ids(build.model, "VALIDATION_COLDHEAD")[0]
        heater_id = next(int(nid) for nid, node in build.model.nodes.items() if node.is_heater)
        row = prepared.node_index_by_id[head_id]
        loads_W = [5.0, 10.0, 20.0, 40.0, 80.0, 120.0]
        simulated: list[float] = []
        reference: list[float] = []
        prepared.reset()
        chunk = max(1, _step_count(params.duration_s, params.dt_s) // 40)
        max_chunks = 200  # generous cap; the lift curve flattens near max power (slow tau)
        # Loads sweep upward, so the cold head warms monotonically -- warm-starting
        # each load from the previous steady state keeps every transient short.
        for load in loads_W:
            previous = float("inf")
            for _ in range(max_chunks):
                for _ in range(chunk):
                    prepared.step_with_forced_heater_powers({heater_id: load}, keep_cryocoolers_active=True)
                current = float(prepared.temperatures_K[row])
                if abs(current - previous) < 1.0e-4:  # settled to steady state
                    break
                previous = current
            simulated.append(float(prepared.temperatures_K[row]))
            reference.append(float(PT60LiftCurve.temperature_for_power_w(load)))
        return _numeric_reference_result(
            self.name, params, build, np.asarray(loads_W, dtype=float),
            {"cold_head_temperature_K": np.asarray(simulated, dtype=float)},
            {"cold_head_temperature_K": np.asarray(reference, dtype=float)},
        )


class EnergyConservationExperiment(ThermalValidationExperiment):
    name = ENERGY_CONSERVATION
    description = (
        "Global energy-conservation audit with conduction + ambient radiation + a heater "
        "all active at once: the change in stored internal energy must equal the time "
        "integral of net power in (heater minus radiated). Conduction is internal, so it "
        "must net to zero -- any sign/scale error in the coupled terms shows up here."
    )
    equation = "dU/dt = P_heater + sum_i coeff_i*(T_env^4 - T_i^4);  U = sum_i C_i (T_i - T_i0)"

    def default_parameters(self) -> ThermalValidationParameters:
        params = ThermalValidationParameters(experiment_name=self.name)
        params.use_octree_pipeline = False
        params.material = "Copper"
        params.length_mm = 60.0
        params.width_mm = params.height_mm = 10.0
        params.voxel_max_size_mm = 12.0  # ~5 cells
        params.initial_temperature_K = 300.0
        params.reference_temperature_K = 300.0
        params.heater_power_W = 20.0
        params.duration_s = 200.0
        params.dt_s = 0.2
        params.output_sample_interval_s = 1.0
        params.use_ambient_radiation = True
        params.gpu_solver_enabled = False  # ~5-node problem: CPU is faster
        params.absolute_tolerance_K = 1.0  # unused; energy metrics carry their own tolerances
        return params

    def geometry_components(self, params) -> list:
        return []

    def build_fallback_model(self, params) -> ThermalGraphModel:
        model = _new_validation_model(self.name)
        _add_box_grid(
            model, "VALIDATION_ENERGY_ROD", center_mm=(0.0, 0.0, 0.0),
            size_mm=(params.length_mm, params.width_mm, params.height_mm),
            material=params.material, initial_temperature_K=params.initial_temperature_K,
            max_cell_size_mm=params.voxel_max_size_mm,
        )
        refresh_geometry_edges(model)
        rod_ids = _component_node_ids(model, "VALIDATION_ENERGY_ROD")
        _add_forcing_heater(model, heater_id=next(_node_id_counter(model)), deposit_node_id=rod_ids[0])
        return model

    def configure_model(self, model, matrices, params) -> None:
        library = model.material_library or default_material_library()
        for node_id in _component_node_ids(model, "VALIDATION_ENERGY_ROD"):
            node = model.nodes[node_id]
            _apply_material(node, params.material, library)
            node.initial_temperature_K = float(params.initial_temperature_K)
            _enable_cell_radiation(node, params)
        model.metadata.T_sur_K = float(params.reference_temperature_K)
        model.metadata.edge_mode = EdgeMode.AUTO.value

    def requested_volume_m3(self, params) -> float:
        return params.length_mm * params.width_mm * params.height_mm * 1.0e-9

    def simulation_parameters(self, params) -> SimulationParameters:
        sim = super().simulation_parameters(params)
        sim.use_ambient_radiation = True
        sim.T_env_K = float(params.reference_temperature_K)
        return sim

    def _run_prepared(self, build, prepared, params) -> ValidationRunResult:
        rod_ids = _component_node_ids(build.model, "VALIDATION_ENERGY_ROD")
        heater_id = next(int(nid) for nid, node in build.model.nodes.items() if node.is_heater)
        rows = np.asarray([prepared.node_index_by_id[nid] for nid in rod_ids], dtype=int)
        C = np.asarray(build.matrices["C"], dtype=float)[rows]
        initial = np.asarray(prepared.temperatures_K, dtype=float)[rows].copy()
        coeff = (
            np.asarray(prepared.radiation_coeff_W_K4, dtype=float).reshape(-1)[rows]
            if prepared.radiation_coeff_W_K4 is not None
            else np.zeros(rows.size)
        )
        env4 = (
            np.asarray(prepared.environment_temperature_K, dtype=float).reshape(-1)[rows]
            if prepared.environment_temperature_K is not None
            else np.full(rows.size, float(params.reference_temperature_K))
        ) ** 4
        heater_W = float(params.heater_power_W)
        steps = _step_count(params.duration_s, params.dt_s)
        sample = _SampleClock(params.output_sample_interval_s)

        def net_power_in_W() -> float:
            temps = np.asarray(prepared.temperatures_K, dtype=float)[rows]
            return heater_W + float(np.sum(coeff * (env4 - temps ** 4)))

        times = [float(prepared.time_s)]
        stored = [0.0]
        power_series = [net_power_in_W()]
        for _ in range(steps):
            _advance_forced_power(prepared, {heater_id: heater_W})
            if sample.should_sample(prepared.time_s):
                temps = np.asarray(prepared.temperatures_K, dtype=float)[rows]
                times.append(float(prepared.time_s))
                stored.append(float(np.sum(C * (temps - initial))))
                power_series.append(net_power_in_W())
        time_array = np.asarray(times, dtype=float)
        stored_array = np.asarray(stored, dtype=float)
        # Cumulative net energy in = trapezoidal integral of the net power.
        supplied = np.concatenate([[0.0], np.cumsum(
            0.5 * (np.asarray(power_series[1:]) + np.asarray(power_series[:-1])) * np.diff(time_array)
        )]) if time_array.size > 1 else np.zeros_like(time_array)
        residual = stored_array - supplied
        max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
        scale = max(float(np.max(np.abs(supplied))) if supplied.size else 0.0, 1.0e-12)
        rel = max_abs / scale
        energy_tol_J = 0.01 * scale
        metrics = [
            _metric("maximum energy imbalance", max_abs, energy_tol_J, _status_abs(max_abs, energy_tol_J), "J"),
            _metric("relative energy imbalance", rel, 0.01, _status_abs(rel, 0.01), ""),
            _metric("net energy supplied", float(supplied[-1]) if supplied.size else 0.0, None, "PASS", "J"),
        ]
        return ValidationRunResult(
            self.name, params,
            _material_properties(params.material, build.model.material_library),
            list(time_array),
            {"stored_internal_energy_J": stored_array.tolist()},
            {"stored_internal_energy_J": supplied.tolist()},
            {"stored_internal_energy_J": residual.tolist()},
            metrics,
            _overall_status(metrics, build.warnings),
            list(build.warnings),
        )


def experiments_by_name() -> dict[str, ThermalValidationExperiment]:
    return {
        INSULATED_BLOCK: InsulatedHeatedBlockExperiment(),
        TWO_BLOCK_EXCHANGE: TwoBlockExchangeExperiment(),
        ONE_D_PRISM: OneDimensionalPrismExperiment(),
        TWO_NODE_LUMPED: TwoNodeLumpedConductanceExperiment(),
        GEOMETRY_CONTACT_PAIR: GeometryDerivedContactPairExperiment(),
        DISTRIBUTED_ROD: OneDimensionalDistributedRodExperiment(),
        RADIATION_COOLING: RadiationCoolingExperiment(),
        TDEP_HEATING: TemperatureDependentHeatingExperiment(),
        CRYO_REGIME: CryoRegimeExperiment(),
        SANDIA_THERMAL_CHALLENGE: SandiaThermalChallengeExperiment(),
        RADIATIVE_COUPLING: RadiativeCouplingExperiment(),
        TDEP_CONDUCTION: TemperatureDependentConductionExperiment(),
        CRYOCOOLER_LIFT: CryocoolerLiftCurveExperiment(),
        ENERGY_CONSERVATION: EnergyConservationExperiment(),
    }


def prism_dirichlet_insulated_solution(
    x_m: float,
    times_s: np.ndarray,
    length_m: float,
    alpha_m2_s: float,
    initial_temperature_K: float,
    surface_temperature_K: float,
    terms: int = 100,
) -> np.ndarray:
    times = np.asarray(times_s, dtype=float)
    result = np.empty_like(times, dtype=float)
    if abs(initial_temperature_K - surface_temperature_K) <= 0.0:
        result.fill(float(surface_temperature_K))
        return result
    theta = np.zeros_like(times, dtype=float)
    L = max(float(length_m), 1.0e-30)
    alpha = max(float(alpha_m2_s), 0.0)
    # The cosine series is naturally measured from the insulated end. Validation
    # probe coordinates are measured from the fixed-temperature face.
    x = L - min(max(float(x_m), 0.0), L)
    for n in range(max(1, int(terms))):
        m = 2 * n + 1
        beta = m * math.pi / (2.0 * L)
        theta += ((-1.0) ** n / float(m)) * math.cos(beta * x) * np.exp(-alpha * beta * beta * times)
    theta *= 4.0 / math.pi
    result = float(surface_temperature_K) + theta * (float(initial_temperature_K) - float(surface_temperature_K))
    result[np.isclose(times, 0.0)] = float(initial_temperature_K)
    low = min(float(surface_temperature_K), float(initial_temperature_K))
    high = max(float(surface_temperature_K), float(initial_temperature_K))
    result = np.clip(result, low, high)
    return result


def export_validation_result(result: ValidationRunResult, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        _export_csv(result, output_path)
    else:
        output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return output_path


def _export_csv(result: ValidationRunResult, path: Path) -> None:
    columns = ["time_s"]
    for prefix, values in (("simulated", result.simulated), ("analytical", result.analytical), ("error", result.errors)):
        for key, series in values.items():
            if isinstance(series, list) and len(series) == len(result.times_s):
                columns.append(f"{prefix}_{key}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, time_s in enumerate(result.times_s):
            row: dict[str, Any] = {"time_s": time_s}
            for prefix, values in (("simulated", result.simulated), ("analytical", result.analytical), ("error", result.errors)):
                for key, series in values.items():
                    if isinstance(series, list) and len(series) == len(result.times_s):
                        row[f"{prefix}_{key}"] = series[index]
            writer.writerow(row)


def _write_glb_scene(
    components: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]],
    path: Path,
    material_name: str,
) -> Path:
    try:
        import trimesh
        from trimesh.visual.material import SimpleMaterial
    except ImportError as exc:
        raise RuntimeError("trimesh is required to generate validation GLB geometry.") from exc
    scene = trimesh.Scene()
    material = SimpleMaterial(name=str(material_name), diffuse=[180, 80, 40, 255])
    for name, center_mm, size_mm in components:
        mesh = trimesh.creation.box(extents=np.asarray(size_mm, dtype=float) * 1.0e-3)
        mesh.apply_translation(np.asarray(center_mm, dtype=float) * 1.0e-3)
        mesh.visual.material = material
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(path)
    return path


def _new_validation_model(name: str) -> ThermalGraphModel:
    library = default_material_library()
    return ThermalGraphModel(
        metadata=GraphMetadata(graph_name=_safe_asset_name(name), edge_mode=EdgeMode.AUTO.value),
        material_library=library,
    )


def _add_box_grid(
    model: ThermalGraphModel,
    component: str,
    *,
    center_mm: tuple[float, float, float],
    size_mm: tuple[float, float, float],
    material: str,
    initial_temperature_K: float,
    max_cell_size_mm: float,
) -> list[int]:
    counts = [max(1, int(math.ceil(float(length) / max(float(max_cell_size_mm), 1.0e-9)))) for length in size_mm]
    cell_size = [float(size_mm[index]) / counts[index] for index in range(3)]
    minimum = [float(center_mm[index]) - float(size_mm[index]) * 0.5 for index in range(3)]
    node_ids: list[int] = []
    for ix in range(counts[0]):
        for iy in range(counts[1]):
            for iz in range(counts[2]):
                node_id = next(_node_id_counter(model))
                node = NodeProperties.with_material(node_id, (node_id, 0, 0), material, model.material_library)
                node.component_name = component
                node.center_mm = (
                    minimum[0] + (ix + 0.5) * cell_size[0],
                    minimum[1] + (iy + 0.5) * cell_size[1],
                    minimum[2] + (iz + 0.5) * cell_size[2],
                )
                node.size_mm = tuple(cell_size)
                node.side_length_m = max(cell_size) * 1.0e-3
                volume_m3 = cell_size[0] * cell_size[1] * cell_size[2] * 1.0e-9
                node.mass_kg = float(node.rho_kg_m3) * volume_m3
                node.C_J_K = node.mass_kg * float(node.cp_J_kgK)
                node.initial_temperature_K = float(initial_temperature_K)
                node.Grad_W_K = 0.0
                node.G_rad_W_K = 0.0
                node.is_exposed = False
                model.add_node(node)
                node_ids.append(node_id)
    return node_ids


def _node_id_counter(model: ThermalGraphModel):
    value = max(model.nodes, default=-1) + 1
    while True:
        while value in model.nodes:
            value += 1
        yield value
        value += 1


def _apply_material(node: NodeProperties, material: str, library: dict[str, dict[str, float]]) -> None:
    defaults = material_defaults(material, library)
    node.material = material
    node.rho_kg_m3 = float(defaults["rho_kg_m3"])
    node.cp_J_kgK = float(defaults["cp_J_kgK"])
    node.k_W_mK = float(defaults["k_W_mK"])
    node.emissivity = float(defaults["emissivity"])
    volume = _node_volume_m3(node)
    if volume > 0.0:
        node.mass_kg = node.rho_kg_m3 * volume
        node.C_J_K = node.mass_kg * node.cp_J_kgK


def _material_properties(material: str, library: dict[str, dict[str, float]]) -> dict[str, float]:
    defaults = material_defaults(material, library)
    rho = float(defaults["rho_kg_m3"])
    cp = float(defaults["cp_J_kgK"])
    k = float(defaults["k_W_mK"])
    return {
        "k_W_mK": k,
        "rho_kg_m3": rho,
        "cp_J_kgK": cp,
        "emissivity": float(defaults["emissivity"]),
        "alpha_m2_s": k / max(rho * cp, 1.0e-30),
    }


def _expected_contact_pair_conductance(
    params: ThermalValidationParameters,
    library: dict[str, dict[str, float]],
) -> float:
    material = _material_properties(params.material, library)
    area_m2 = max(float(params.width_mm) * float(params.height_mm) * 1.0e-6, 1.0e-30)
    distance_m = max(float(params.length_mm) * 1.0e-3, 1.0e-30)
    k = max(float(material["k_W_mK"]), 1.0e-30)
    resistance = 0.5 * distance_m / (k * area_m2)
    resistance += 0.5 * distance_m / (k * area_m2)
    resistance += 1.0 / (DEFAULT_CONTACT_INTERFACE_CONDUCTANCE_W_M2K * area_m2)
    return float(1.0 / resistance)


def _component_node_ids(model: ThermalGraphModel, component: str) -> list[int]:
    return [
        int(node_id)
        for node_id, node in sorted(model.nodes.items())
        if str(getattr(node, "component_name", "") or "").startswith(component)
    ]


def _is_validation_body_node(node: NodeProperties) -> bool:
    name = str(getattr(node, "component_name", "") or "")
    return name.startswith("VALIDATION_") and "HEATER" not in name


def _volume_m3_for_nodes(nodes: Any) -> float:
    return float(sum(_node_volume_m3(node) for node in nodes if _is_validation_body_node(node)))


def _node_volume_m3(node: NodeProperties) -> float:
    # Occupied (physical) volume: full cell volume scaled by the octree
    # occupancy fraction. Without this, partial boundary cells (occupancy < 1)
    # are counted as full, which over-states volume and inflates capacitance.
    occupancy = float(getattr(node, "occupancy_fraction", 1.0) or 1.0)
    if not (occupancy > 0.0):
        occupancy = 1.0
    if node.size_mm is not None:
        sx, sy, sz = (max(0.0, float(value)) for value in node.size_mm)
        return sx * sy * sz * 1.0e-9 * occupancy
    return max(0.0, float(getattr(node, "side_length_m", 0.0)) ** 3) * occupancy


def _nodes_on_min_x_face(model: ThermalGraphModel, node_ids: list[int]) -> list[int]:
    if not node_ids:
        return []
    bounds = [(node_id, _node_min_max_x(model.nodes[node_id])) for node_id in node_ids]
    min_x = min(pair[1][0] for pair in bounds)
    return [node_id for node_id, (node_min, _node_max) in bounds if abs(node_min - min_x) <= 1.0e-6]


def _nodes_on_max_x_face(model: ThermalGraphModel, node_ids: list[int]) -> list[int]:
    if not node_ids:
        return []
    bounds = [(node_id, _node_min_max_x(model.nodes[node_id])) for node_id in node_ids]
    max_x = max(pair[1][1] for pair in bounds)
    return [node_id for node_id, (_node_min, node_max) in bounds if abs(node_max - max_x) <= 1.0e-6]


def _node_min_max_x(node: NodeProperties) -> tuple[float, float]:
    center = float((node.center_mm or node.center)[0])
    size = float((node.size_mm or (node.side_length_m * 1000.0,) * 3)[0])
    return center - 0.5 * size, center + 0.5 * size


def _capacitance_weights(model: ThermalGraphModel, node_ids: list[int]) -> list[float]:
    values = [max(0.0, float(model.nodes[node_id].C_J_K)) for node_id in node_ids]
    total = sum(values)
    if total <= 0.0:
        return [1.0 / len(node_ids)] * len(node_ids) if node_ids else []
    return [value / total for value in values]


def _capacitance_sum(model: ThermalGraphModel, node_ids: list[int]) -> float:
    return float(sum(max(0.0, float(model.nodes[int(node_id)].C_J_K)) for node_id in node_ids))


def _interface_edge_keys(model: ThermalGraphModel, a: str, b: str) -> list[tuple[int, int]]:
    keys: list[tuple[int, int]] = []
    for key, edge in model.edges.items():
        source = model.nodes.get(edge.source)
        target = model.nodes.get(edge.target)
        if source is None or target is None:
            continue
        source_component = str(source.component_name)
        target_component = str(target.component_name)
        if (source_component.startswith(a) and target_component.startswith(b)) or (
            source_component.startswith(b) and target_component.startswith(a)
        ):
            keys.append(key)
    return keys


def _interface_conductance(model: ThermalGraphModel, a: str, b: str) -> float:
    return float(sum(max(0.0, float(model.edges[key].Gij_W_K)) for key in _interface_edge_keys(model, a, b)))


def _remove_edges_touching(model: ThermalGraphModel, node_ids: list[int]) -> None:
    endpoints = {int(node_id) for node_id in node_ids}
    if not endpoints:
        return
    original_count = len(model.edges)
    model.edges = {
        key: edge
        for key, edge in model.edges.items()
        if int(edge.source) not in endpoints and int(edge.target) not in endpoints
    }
    if len(model.edges) != original_count:
        model.touch()


def _advance_forced_power(prepared: PreparedSimulation, power_by_heater: dict[int, float]) -> None:
    old_time = float(prepared.time_s)
    prepared.step_with_forced_heater_powers(power_by_heater, keep_cryocoolers_active=False)
    state = SimulationState(old_time + float(prepared.params.dt_s), prepared.temperatures_K.copy())
    prepared._append_history_state(state)


def _apply_fixed_temperature(prepared: PreparedSimulation, node_ids: list[int], temperature_K: float) -> float:
    if prepared.inv_C is None:
        return 0.0
    removed = 0.0
    for node_id in node_ids:
        row = prepared.node_index_by_id.get(int(node_id))
        if row is None:
            continue
        previous = float(prepared.z[row])
        removed += float(prepared.params.dt_s) * 0.0
        if prepared.inv_C[row] > 0.0:
            removed += (previous - float(temperature_K)) / prepared.inv_C[row]
        prepared.z[row] = float(temperature_K)
    prepared._sync_gpu_state()
    return removed


def _record_block_sample(prepared, rows, C, initial, times, avg, minimum, maximum, energy) -> None:
    temps = np.asarray(prepared.temperatures_K, dtype=float)
    values = temps[rows] if rows.size else temps
    weights = C[rows] if rows.size else C
    times.append(float(prepared.time_s))
    avg.append(_weighted_average(values, weights))
    minimum.append(float(np.min(values)) if values.size else float("nan"))
    maximum.append(float(np.max(values)) if values.size else float("nan"))
    energy.append(float(np.dot(C[rows], temps[rows] - initial[rows])) if rows.size else 0.0)


def _record_two_block_sample(prepared, hot_rows, cold_rows, C, initial, times, hot, cold, delta, energy) -> None:
    temps = np.asarray(prepared.temperatures_K, dtype=float)
    hot_value = _weighted_average(temps[hot_rows], C[hot_rows])
    cold_value = _weighted_average(temps[cold_rows], C[cold_rows])
    rows = np.concatenate([hot_rows, cold_rows])
    times.append(float(prepared.time_s))
    hot.append(hot_value)
    cold.append(cold_value)
    delta.append(hot_value - cold_value)
    energy.append(float(np.dot(C[rows], temps[rows] - initial[rows])) if rows.size else 0.0)


def _probe_row_groups(
    model: ThermalGraphModel,
    node_index_by_id: dict[int, int],
    node_ids: list[int],
    positions: tuple[float, ...],
) -> dict[str, tuple[np.ndarray, float]]:
    if not node_ids:
        return {}
    min_x = min(_node_min_max_x(model.nodes[node_id])[0] for node_id in node_ids)
    max_x = max(_node_min_max_x(model.nodes[node_id])[1] for node_id in node_ids)
    length = max(max_x - min_x, 1.0e-12)
    centers = {node_id: float((model.nodes[node_id].center_mm or model.nodes[node_id].center)[0]) for node_id in node_ids}
    size_x = np.median([float((model.nodes[node_id].size_mm or (1.0, 1.0, 1.0))[0]) for node_id in node_ids])
    groups: dict[str, tuple[np.ndarray, float]] = {}
    for position in positions:
        target = min_x + min(max(float(position), 0.0), 1.0) * length
        close = [node_id for node_id, center in centers.items() if abs(center - target) <= max(size_x * 0.55, 1.0e-9)]
        if not close:
            closest = min(node_ids, key=lambda node_id: abs(centers[node_id] - target))
            close = [closest]
        rows = np.asarray([node_index_by_id[node_id] for node_id in close if node_id in node_index_by_id], dtype=int)
        actual_x = float(np.mean([centers[node_id] for node_id in close])) - min_x
        groups[f"x_over_L_{float(position):.2f}"] = (rows, actual_x)
    return groups


def _record_prism_sample(prepared, probe_rows, surface_temperature, times, simulated, boundary_errors, energy_removed, cumulative_removed) -> None:
    temps = np.asarray(prepared.temperatures_K, dtype=float)
    times.append(float(prepared.time_s))
    for label, (rows, _x_mm) in probe_rows.items():
        simulated[label].append(float(np.mean(temps[rows])) if rows.size else float("nan"))
    boundary_errors.append(0.0)
    energy_removed.append(float(cumulative_removed))


def _record_distributed_rod_sample(
    prepared: PreparedSimulation,
    model: ThermalGraphModel,
    rod_ids: list[int],
    rows: np.ndarray,
    times: list[float],
    simulated: dict[str, list[float]],
) -> None:
    temps = np.asarray(prepared.temperatures_K, dtype=float)
    values = temps[rows]
    weights = np.asarray([model.nodes[node_id].C_J_K for node_id in rod_ids], dtype=float)
    mean = _weighted_average(values, weights)
    mode = _rod_first_mode(len(rod_ids))
    times.append(float(prepared.time_s))
    simulated["mode_amplitude_K"].append(_project_rod_mode(values, mode, mean))
    for node_id, value in zip(rod_ids, values):
        simulated[f"node_{node_id}_temperature_K"].append(float(value))


def _insulated_end_heat_flow(model: ThermalGraphModel, prepared: PreparedSimulation, prism_ids: list[int]) -> float:
    end_ids = set(_nodes_on_max_x_face(model, prism_ids))
    if not end_ids:
        return 0.0
    flow = 0.0
    temps = prepared.temperatures_K
    index = prepared.node_index_by_id
    for edge in model.edges.values():
        source_in = edge.source in end_ids
        target_in = edge.target in end_ids
        if source_in == target_in:
            continue
        if edge.source not in index or edge.target not in index:
            continue
        flow += abs(float(edge.Gij_W_K) * (float(temps[index[edge.source]]) - float(temps[index[edge.target]])))
    return float(flow)


def _component_spread(prepared: PreparedSimulation, rows: np.ndarray) -> float:
    values = np.asarray(prepared.temperatures_K, dtype=float)[rows]
    return float(np.max(values) - np.min(values)) if values.size else 0.0


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.mean(values))
    return float(np.dot(values, weights) / total)


def _rod_first_mode(count: int) -> np.ndarray:
    n = max(1, int(count))
    return np.asarray([math.cos(math.pi * (index + 0.5) / float(n)) for index in range(n)], dtype=float)


def _project_rod_mode(values: np.ndarray, mode: np.ndarray, mean: float) -> float:
    centered = np.asarray(values, dtype=float).reshape(-1) - float(mean)
    shape = np.asarray(mode, dtype=float).reshape(-1)
    denominator = float(np.dot(shape, shape))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(centered, shape) / denominator)


def _uniform_rod_conductance(model: ThermalGraphModel, rod_ids: list[int]) -> float:
    rod_set = {int(node_id) for node_id in rod_ids}
    values = [
        float(edge.Gij_W_K)
        for edge in model.edges.values()
        if int(edge.source) in rod_set and int(edge.target) in rod_set and float(edge.Gij_W_K) > 0.0
    ]
    if not values:
        return 0.0
    return float(np.mean(values))


def _temperature_metrics(prefix: str, simulated: np.ndarray, analytical: np.ndarray, params: ThermalValidationParameters) -> list[MetricResult]:
    error = np.asarray(simulated, dtype=float) - np.asarray(analytical, dtype=float)
    max_abs = float(np.max(np.abs(error))) if error.size else 0.0
    rmse = _rmse(simulated, analytical)
    final = float(error[-1]) if error.size else 0.0
    tolerance = float(params.absolute_tolerance_K)
    return [
        _metric(f"{prefix} maximum absolute temperature error", max_abs, tolerance, _status_abs(max_abs, tolerance), "K"),
        _metric(f"{prefix} RMSE", rmse, tolerance, _status_abs(rmse, tolerance), "K"),
        _metric(f"{prefix} final temperature error", final, tolerance, _status_abs(final, tolerance), "K"),
    ]


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    error = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(math.sqrt(float(np.mean(error * error)))) if error.size else 0.0


def _metric(name: str, value: float, tolerance: float | None, status: str, units: str) -> MetricResult:
    if not np.isfinite(float(value)):
        status = "FAIL"
    return MetricResult(name=name, value=float(value), tolerance=None if tolerance is None else float(tolerance), status=status, units=units)


def _status_abs(value: float, tolerance: float) -> str:
    if not np.isfinite(float(value)):
        return "FAIL"
    return "PASS" if abs(float(value)) <= abs(float(tolerance)) else "WARNING"


def _overall_status(metrics: list[MetricResult], warnings: list[str]) -> str:
    if any(metric.status == "FAIL" for metric in metrics):
        return "FAIL"
    if warnings or any(metric.status == "WARNING" for metric in metrics):
        return "WARNING"
    return "PASS"


def _step_count(duration_s: float, dt_s: float) -> int:
    return max(0, int(math.ceil(max(0.0, float(duration_s)) / max(float(dt_s), 1.0e-12))))


def _safe_asset_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_") or "validation"


class _SampleClock:
    def __init__(self, interval_s: float) -> None:
        self.interval_s = max(float(interval_s), 1.0e-12)
        self.next_sample_s = self.interval_s

    def should_sample(self, time_s: float) -> bool:
        if float(time_s) + 1.0e-12 < self.next_sample_s:
            return False
        while self.next_sample_s <= float(time_s) + 1.0e-12:
            self.next_sample_s += self.interval_s
        return True
