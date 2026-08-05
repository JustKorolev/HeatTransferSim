"""Focused mechanics tests for the heat-transfer simulator."""

from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix

from graph_visualizer.cryocooler import PT60LiftCurve
from graph_visualizer.graph_io import save_graph_folder
from graph_visualizer.matrix_builder import (
    STEFAN_BOLTZMANN_W_M2K4,
    build_matrices,
    refresh_geometry_edges,
)
from graph_visualizer.models import GraphMetadata, HeaterProperties, NodeProperties, ThermalGraphModel
from graph_visualizer.simulation_model import prepare_simulation
from graph_visualizer.simulation_parameters import SimulationParameters
from graph_visualizer.thermal_validation import (
    INSULATED_BLOCK,
    TWO_BLOCK_EXCHANGE,
    experiments_by_name,
)


class SimulatorMechanicsTests(unittest.TestCase):
    def test_dense_sparse_expm_and_sparse_implicit_paths_agree_for_passive_graph(self) -> None:
        model = _line_model(
            capacitances=(12.0, 20.0, 16.0),
            conductances=(1.5, 0.75),
            temperatures=(330.0, 280.0, 250.0),
        )
        dense = build_matrices(model)
        sparse = dict(dense)
        sparse["L"] = csr_matrix(dense["L"])

        dense_prepared = prepare_simulation(model, dense, _params(dt_s=0.05))
        sparse_expm = prepare_simulation(
            model,
            sparse,
            _params(dt_s=0.05, implicit=False, fast=False),
        )
        sparse_implicit = prepare_simulation(
            model,
            sparse,
            _params(
                dt_s=0.05,
                implicit=True,
                fast=False,
                implicit_sparse_simulation_rtol=1.0e-11,
                implicit_sparse_simulation_maxiter=1000,
                implicit_sparse_adaptive_substeps_enabled=False,
            ),
        )

        for prepared in (dense_prepared, sparse_expm, sparse_implicit):
            for _ in range(8):
                prepared.step_forward()

        np.testing.assert_allclose(sparse_expm.temperatures_K, dense_prepared.temperatures_K, rtol=1.0e-10, atol=1.0e-10)
        np.testing.assert_allclose(sparse_implicit.temperatures_K, dense_prepared.temperatures_K, rtol=2.0e-4, atol=2.0e-3)

    def test_passive_closed_system_conserves_total_thermal_energy(self) -> None:
        model = _line_model(
            capacitances=(8.0, 13.0, 21.0, 34.0),
            conductances=(0.5, 2.0, 1.25),
            temperatures=(310.0, 270.0, 295.0, 240.0),
        )
        matrices = build_matrices(model)
        # Total heat sum(C*T) is conserved exactly by the scheme (1^T L = 0); the
        # only error is the CG residual, so use a tight linear tolerance.
        prepared = prepare_simulation(model, matrices, _params(dt_s=0.2, implicit_sparse_simulation_rtol=1.0e-12))
        C = np.asarray(matrices["C"], dtype=float)
        initial_energy = float(np.dot(C, prepared.temperatures_K))

        for _ in range(40):
            prepared.step_forward()

        self.assertAlmostEqual(float(np.dot(C, prepared.temperatures_K)), initial_energy, places=5)

    def test_two_node_explicit_conductance_matches_closed_form_exponential(self) -> None:
        C1, C2, G = 10.0, 30.0, 0.8
        T1_0, T2_0 = 320.0, 260.0
        model = _line_model(
            capacitances=(C1, C2),
            conductances=(G,),
            temperatures=(T1_0, T2_0),
        )
        prepared = prepare_simulation(model, build_matrices(model), _params(dt_s=0.25))

        for _ in range(12):
            prepared.step_forward()

        time_s = 3.0
        equilibrium = (C1 * T1_0 + C2 * T2_0) / (C1 + C2)
        delta = (T1_0 - T2_0) * math.exp(-G * (1.0 / C1 + 1.0 / C2) * time_s)
        expected = np.array(
            [
                equilibrium + (C2 / (C1 + C2)) * delta,
                equilibrium - (C1 / (C1 + C2)) * delta,
            ]
        )
        # Implicit TR-BDF2 is 2nd-order accurate in time (not exact like the old
        # matrix-exponential path); match the closed form to solver accuracy.
        np.testing.assert_allclose(prepared.temperatures_K, expected, rtol=1.0e-4, atol=1.0e-2)

    def test_three_cell_distributed_slab_matches_known_finite_difference_mode(self) -> None:
        C, G = 10.0, 2.0
        model = _line_model(
            capacitances=(C, C, C),
            conductances=(G, G),
            temperatures=(300.0, 200.0, 100.0),
        )
        prepared = prepare_simulation(model, build_matrices(model), _params(dt_s=0.1))

        for _ in range(10):
            prepared.step_forward()

        amplitude = 100.0 * math.exp(-(G / C) * 1.0)
        expected = np.array([200.0 + amplitude, 200.0, 200.0 - amplitude])
        # Implicit TR-BDF2 matches the analytic mode to 2nd-order time accuracy.
        np.testing.assert_allclose(prepared.temperatures_K, expected, rtol=1.0e-4, atol=1.0e-2)

    def test_geometry_derived_conductance_uses_area_distance_materials_and_interface_term(self) -> None:
        same = ThermalGraphModel(metadata=GraphMetadata(graph_name="same_material_geometry"))
        same.add_node(_box_node(1, "body", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0), k=400.0))
        same.add_node(_box_node(2, "body", (10.0, 0.0, 0.0), (10.0, 10.0, 10.0), k=400.0))
        refresh_geometry_edges(same)
        self.assertEqual(len(same.edges), 1)
        self.assertAlmostEqual(next(iter(same.edges.values())).Gij_W_K, 4.0)

        interface = ThermalGraphModel(metadata=GraphMetadata(graph_name="interface_geometry"))
        interface.add_node(_box_node(1, "left", (0.0, 0.0, 0.0), (10.0, 10.0, 10.0), k=100.0))
        interface.add_node(_box_node(2, "right", (10.0, 0.0, 0.0), (10.0, 10.0, 10.0), k=200.0))
        refresh_geometry_edges(interface, contact_interface_conductance_W_m2K=1.0e4)

        area = 10.0 * 10.0 * 1.0e-6
        distance = 10.0e-3
        resistance = (0.5 * distance) / (100.0 * area)
        resistance += (0.5 * distance) / (200.0 * area)
        resistance += 1.0 / (1.0e4 * area)
        self.assertAlmostEqual(next(iter(interface.edges.values())).Gij_W_K, 1.0 / resistance)

    def test_interface_modes_distinguish_explicit_total_from_geometry_derived_reporting(self) -> None:
        experiment = experiments_by_name()[TWO_BLOCK_EXCHANGE]
        with TemporaryDirectory() as tmp:
            explicit = experiment.default_parameters()
            explicit.use_octree_pipeline = False
            explicit.interface_model = "explicit_total_conductance"
            explicit.interface_conductance_W_K = 0.37
            explicit.duration_s = explicit.dt_s
            explicit.output_sample_interval_s = explicit.dt_s
            explicit_result = experiment.run(experiment.build(explicit, Path(tmp)), explicit)

            derived = experiment.default_parameters()
            derived.use_octree_pipeline = False
            derived.interface_model = "geometry_derived_conductance"
            derived.interface_conductance_W_K = 0.37
            derived.duration_s = derived.dt_s
            derived.output_sample_interval_s = derived.dt_s
            derived_result = experiment.run(experiment.build(derived, Path(tmp)), derived)

        explicit_metric = next(metric for metric in explicit_result.metrics if metric.name == "Actual interface conductance")
        derived_metric = next(metric for metric in derived_result.metrics if metric.name == "Geometry-derived interface conductance")
        self.assertAlmostEqual(explicit_metric.value, explicit.interface_conductance_W_K)
        self.assertEqual(explicit_metric.status, "PASS")
        self.assertIsNone(derived_metric.tolerance)
        self.assertEqual(derived_metric.status, "PASS")
        self.assertNotAlmostEqual(derived_metric.value, explicit.interface_conductance_W_K)

    def test_validation_graph_volume_and_capacitance_match_requested_body(self) -> None:
        experiment = experiments_by_name()[INSULATED_BLOCK]
        params = experiment.default_parameters()
        params.use_octree_pipeline = False
        params.material = "Copper"
        with TemporaryDirectory() as tmp:
            build = experiment.build(params, Path(tmp))

        expected_volume = params.length_mm * params.width_mm * params.height_mm * 1.0e-9
        material = build.model.material_library[params.material]
        expected_capacitance = expected_volume * material["rho_kg_m3"] * material["cp_J_kgK"]
        body_nodes = [
            node
            for node in build.model.nodes.values()
            if node.component_name.startswith("VALIDATION_BLOCK")
        ]

        self.assertAlmostEqual(build.imported_volume_m3, expected_volume)
        self.assertAlmostEqual(sum(node.C_J_K for node in body_nodes), expected_capacitance)

    def test_heater_power_deposition_conserves_energy_and_respects_weights(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="heater_deposition"))
        body_a = _node(1, C=10.0, temperature=300.0)
        body_b = _node(2, C=10.0, temperature=300.0)
        heater = _node(3, C=1.0e-6, temperature=300.0)
        heater.is_heater = True
        heater.heater = HeaterProperties(heater_id=3, heater_max_power_W=100.0, heater_efficiency=1.0)
        heater.power_deposition_node_ids = [1, 2]
        heater.power_deposition_weights = [1.0, 3.0]
        for node in (body_a, body_b, heater):
            model.add_node(node)
        prepared = prepare_simulation(model, build_matrices(model), _params(dt_s=2.0))

        prepared.step_with_forced_heater_powers({3: 20.0}, keep_cryocoolers_active=False)

        self.assertAlmostEqual(prepared.temperatures_K[0], 301.0)
        self.assertAlmostEqual(prepared.temperatures_K[1], 303.0)
        stored = 10.0 * 1.0 + 10.0 * 3.0
        self.assertAlmostEqual(stored, 20.0 * 2.0)

    def test_cryocooler_groups_component_cells_and_distributes_one_physical_capacity(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="cryocooler_distribution"))
        for node_id, C in ((1, 10.0), (2, 30.0)):
            node = _node(node_id, C=C, temperature=50.0)
            node.component_name = "PT60-A"
            node.has_cryocooler = True
            model.add_node(node)
        matrices = build_matrices(model)
        prepared = prepare_simulation(
            model,
            matrices,
            _params(dt_s=1.0, cryocooler=True, max_cooling=150.0),
        )

        powers = prepared.cryocooler_power_by_node()
        expected_total = PT60LiftCurve().cooling_capacity_w(50.0)

        self.assertEqual(len(prepared.cryocooler_devices), 1)
        self.assertEqual(prepared.cryocooler_devices[0].source_node_ids, (1, 2))
        self.assertAlmostEqual(sum(powers.values()), expected_total)
        self.assertAlmostEqual(powers[1], expected_total * 0.25)
        self.assertAlmostEqual(powers[2], expected_total * 0.75)

    def test_interior_and_exterior_background_temperatures_apply_per_node(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="two_env"))
        for node_id in (1, 2):
            node = _node(node_id, C=50.0, temperature=300.0)
            node.center_mm = (float(node_id), 0.0, 0.0)
            node.size_mm = (1.0, 1.0, 1.0)
            node.is_exposed = True
            node.radiating_area_m2 = 0.25
            node.emissivity = 0.8
            node.G_rad_W_K = 4.0 * node.emissivity * STEFAN_BOLTZMANN_W_M2K4 * node.radiating_area_m2 * 300.0**3
            model.add_node(node)
        model.radiation_interior_node_ids = [1]  # node 1 faces the cryo interior
        matrices = build_matrices(model)
        prepared = prepare_simulation(
            model,
            matrices,
            _params(dt_s=0.5, radiation=True, ambient=293.15, implicit=False,
                    interior_environment_temperature_K=4.0),
        )
        env = prepared.environment_temperature_K
        self.assertAlmostEqual(float(env[0]), 4.0)      # interior
        self.assertAlmostEqual(float(env[1]), 293.15)   # exterior
        for _ in range(10):
            prepared.step_forward()
        interior_drop = 300.0 - float(prepared.temperatures_K[0])
        exterior_drop = 300.0 - float(prepared.temperatures_K[1])
        # Radiating to 4 K cools far faster than radiating to 293 K.
        self.assertGreater(interior_drop, exterior_drop)
        self.assertGreater(interior_drop, 5.0 * exterior_drop)

    def test_radiation_cools_with_stefan_boltzmann_source_and_can_be_disabled(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="radiation"))
        node = _node(1, C=50.0, temperature=320.0)
        node.is_exposed = True
        node.radiating_area_m2 = 0.25
        node.emissivity = 0.8
        node.G_rad_W_K = 4.0 * node.emissivity * STEFAN_BOLTZMANN_W_M2K4 * node.radiating_area_m2 * 300.0**3
        model.add_node(node)
        matrices = build_matrices(model)

        # Disable midpoint coupling so the radiation source is evaluated at the
        # step-start temperature, matching this exact single-step closed form.
        cooled = prepare_simulation(
            model,
            matrices,
            _params(
                dt_s=0.01,
                radiation=True,
                ambient=300.0,
                implicit=False,
                fast=False,
                use_midpoint_property_coupling=False,
            ),
        )
        cooled.step_forward()
        expected_power = node.emissivity * STEFAN_BOLTZMANN_W_M2K4 * node.radiating_area_m2 * (300.0**4 - 320.0**4)
        self.assertAlmostEqual(cooled.temperatures_K[0], 320.0 + 0.01 * expected_power / 50.0)

        disabled = prepare_simulation(
            model,
            matrices,
            _params(dt_s=0.01, radiation=False, implicit=False, fast=False),
        )
        disabled.step_forward()
        self.assertAlmostEqual(disabled.temperatures_K[0], 320.0)

    def test_snapshot_shares_history_entries_and_still_rolls_back(self) -> None:
        # snapshot_state runs once per step for the adaptive-dt retry. Deep-copying
        # the history made it O(steps * nodes) per call -- gigabytes per step on a
        # multi-million-cell graph. Entries are write-once, so the list is shared.
        model = _line_model(
            capacitances=(10.0, 10.0, 10.0),
            conductances=(1.0, 1.0),
            temperatures=(300.0, 290.0, 280.0),
        )
        matrices = build_matrices(model)
        prepared = prepare_simulation(model, matrices, _params(dt_s=0.5, simulation_history_limit=0))
        for _ in range(4):
            prepared.step_forward()

        snapshot = prepared.snapshot_state()
        self.assertIs(snapshot.history[0], prepared.history[0])  # shared, not copied
        depth = len(prepared.history)
        temps_before = prepared.temperatures_K.copy()

        prepared.step_forward()
        self.assertEqual(len(prepared.history), depth + 1)
        prepared.restore_state(snapshot)

        self.assertEqual(len(prepared.history), depth)
        self.assertEqual(prepared.history_index, snapshot.history_index)
        np.testing.assert_allclose(prepared.temperatures_K, temps_before)

    def test_rollback_restores_a_history_entry_trimmed_by_the_limit(self) -> None:
        model = _line_model(
            capacitances=(10.0, 10.0),
            conductances=(1.0,),
            temperatures=(300.0, 280.0),
        )
        matrices = build_matrices(model)
        prepared = prepare_simulation(model, matrices, _params(dt_s=0.5, simulation_history_limit=3))
        for _ in range(3):
            prepared.step_forward()
        self.assertEqual(len(prepared.history), 3)
        oldest = prepared.history[0]

        snapshot = prepared.snapshot_state()
        prepared.step_forward()  # pushes the oldest entry off the front
        self.assertIsNot(prepared.history[0], oldest)

        prepared.restore_state(snapshot)
        self.assertIs(prepared.history[0], oldest)
        self.assertEqual(len(prepared.history), 3)

    def test_history_limit_is_clamped_by_the_memory_budget(self) -> None:
        model = _line_model(
            capacitances=(10.0, 10.0),
            conductances=(1.0,),
            temperatures=(300.0, 280.0),
        )
        matrices = build_matrices(model)
        # 2 nodes = 16 bytes per stored step; a 1 MB budget cannot bind.
        roomy = prepare_simulation(
            model,
            matrices,
            _params(dt_s=0.5, simulation_history_limit=64, simulation_history_memory_budget_MB=1.0),
        )
        self.assertEqual(roomy._effective_history_limit(), 64)
        self.assertFalse([w for w in roomy.warnings if "Replay history capped" in w])

        # A budget below one step's cost floors at 2 entries and warns once.
        tight = prepare_simulation(
            model,
            matrices,
            _params(dt_s=0.5, simulation_history_limit=64, simulation_history_memory_budget_MB=1.0e-6),
        )
        self.assertEqual(tight._effective_history_limit(), 2)
        self.assertEqual(tight._effective_history_limit(), 2)  # idempotent
        capped = [w for w in tight.warnings if "Replay history capped" in w]
        self.assertEqual(len(capped), 1)  # warned once, not once per step
        self.assertIn("Replay history capped at 2 steps", capped[0])
        for _ in range(5):
            tight.step_forward()
        self.assertEqual(len(tight.history), 2)

        # Budget disabled -> the configured step count is honored verbatim.
        unbounded = prepare_simulation(
            model,
            matrices,
            _params(dt_s=0.5, simulation_history_limit=64, simulation_history_memory_budget_MB=0.0),
        )
        self.assertEqual(unbounded._effective_history_limit(), 64)


def _params(
    *,
    dt_s: float,
    radiation: bool = False,
    ambient: float = 293.15,
    cryocooler: bool = False,
    max_cooling: float = 150.0,
    implicit: bool = True,  # retained for call-site compatibility; the solver is always implicit now
    fast: bool = False,
    **overrides: object,
) -> SimulationParameters:
    values = dict(
        dt_s=float(dt_s),
        t_final_s=10.0,
        input_mode="zero",
        use_ambient_radiation=bool(radiation),
        T_env_K=float(ambient),
        cryocooler_enabled=bool(cryocooler),
        cryocooler_max_power_W=float(max_cooling),
        gpu_solver_enabled=False,  # deterministic CPU implicit solver for tests
        simulation_history_limit=0,
        browser_simulation_size_warning=1_000_000,
    )
    values.update(overrides)
    return SimulationParameters(**values)


def _line_model(
    *,
    capacitances: tuple[float, ...],
    conductances: tuple[float, ...],
    temperatures: tuple[float, ...],
) -> ThermalGraphModel:
    model = ThermalGraphModel(metadata=GraphMetadata(graph_name="line"))
    for index, (capacitance, temperature) in enumerate(zip(capacitances, temperatures), start=1):
        node = _node(index, C=capacitance, temperature=temperature)
        node.center_mm = (float(index - 1), 0.0, 0.0)
        node.size_mm = (1.0, 1.0, 1.0)
        model.add_node(node)
    for index, conductance in enumerate(conductances, start=1):
        model.set_edge(index, index + 1, float(conductance))
    return model


def _node(node_id: int, *, C: float, temperature: float) -> NodeProperties:
    node = NodeProperties.with_material(node_id, (node_id, 0, 0), material="Copper")
    node.C_J_K = float(C)
    node.mass_kg = max(1.0e-12, float(C) / max(float(node.cp_J_kgK), 1.0e-12))
    node.initial_temperature_K = float(temperature)
    node.Grad_W_K = 0.0
    node.G_rad_W_K = 0.0
    node.is_exposed = False
    return node


def _box_node(
    node_id: int,
    component: str,
    center_mm: tuple[float, float, float],
    size_mm: tuple[float, float, float],
    *,
    k: float,
) -> NodeProperties:
    node = _node(node_id, C=1.0, temperature=293.15)
    node.component_name = component
    node.center_mm = center_mm
    node.size_mm = size_mm
    node.side_length_m = max(size_mm) * 1.0e-3
    node.k_W_mK = float(k)
    return node


if __name__ == "__main__":
    unittest.main()
