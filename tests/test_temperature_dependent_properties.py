"""Tests for temperature-dependent C(T)/L(T) operator rebuilding."""

import unittest

import numpy as np

from graph_visualizer import material_properties_cryo as mp
from graph_visualizer.matrix_builder import build_matrices
from graph_visualizer.models import GraphMetadata, NodeProperties, ThermalGraphModel
from graph_visualizer.simulation_model import prepare_simulation
from graph_visualizer.simulation_parameters import SimulationParameters
from graph_visualizer.temperature_dependent_properties import build_temperature_dependent_operator


def _copper_node(node_id: int, *, C: float, temperature: float) -> NodeProperties:
    node = NodeProperties.with_material(node_id, (node_id, 0, 0), material="Copper")
    # The in-memory default library has no real copper; set the materials.json
    # room-temperature constants a loaded octree graph would carry, so the
    # constant k0/cp0 match the values that built the stored conductances.
    node.cp_J_kgK = 385.0
    node.k_W_mK = 401.0
    node.C_J_K = float(C)
    node.mass_kg = float(C) / max(float(node.cp_J_kgK), 1e-12)
    node.initial_temperature_K = float(temperature)
    node.Grad_W_K = 0.0
    node.G_rad_W_K = 0.0
    node.is_exposed = False
    node.center_mm = (float(node_id), 0.0, 0.0)
    node.size_mm = (1.0, 1.0, 1.0)
    return node


def _two_copper_model(temps=(300.0, 300.0), conductance=2.0) -> ThermalGraphModel:
    model = ThermalGraphModel(metadata=GraphMetadata(graph_name="tdep"))
    for index, temperature in enumerate(temps, start=1):
        model.add_node(_copper_node(index, C=100.0, temperature=temperature))
    model.set_edge(1, 2, float(conductance))
    return model


def _node_ids(model: ThermalGraphModel) -> np.ndarray:
    return np.array(model.ordered_node_ids(), dtype=int)


def _params(**overrides) -> SimulationParameters:
    values = dict(
        dt_s=0.5,
        t_final_s=10.0,
        input_mode="zero",
        use_ambient_radiation=False,
        gpu_solver_enabled=False,
        simulation_history_limit=0,
        browser_simulation_size_warning=1_000_000,
    )
    values.update(overrides)
    return SimulationParameters(**values)


class OperatorRebuildTests(unittest.TestCase):
    def test_capacitance_collapses_when_cold(self):
        model = _two_copper_model()
        op = build_temperature_dependent_operator(model, _node_ids(model))
        C_warm = op.capacitance(np.array([300.0, 300.0]))
        C_cold = op.capacitance(np.array([4.0, 4.0]))
        self.assertTrue(np.all(C_warm / C_cold > 100.0))

    def test_capacitance_matches_mass_times_cp(self):
        model = _two_copper_model()
        op = build_temperature_dependent_operator(model, _node_ids(model))
        T = np.array([77.0, 20.0])
        expected = op.thermal_mass_kg * mp.specific_heat_J_kgK("Copper", T)
        np.testing.assert_allclose(op.capacitance(T), expected, rtol=1e-9)

    def test_capacitance_near_build_value_at_300K(self):
        # At 300 K the NIST cp is within a few % of the materials.json constant.
        model = _two_copper_model()
        node = model.nodes[1]
        op = build_temperature_dependent_operator(model, _node_ids(model))
        C_300 = float(op.capacitance(np.array([300.0, 300.0]))[0])
        self.assertAlmostEqual(C_300 / float(node.C_J_K), 1.0, delta=0.05)

    def test_conductance_increases_for_cold_copper(self):
        # OFHC copper k rises steeply below ~40 K, so the edge conductance grows.
        model = _two_copper_model()
        op = build_temperature_dependent_operator(model, _node_ids(model))
        L_warm = op.laplacian(np.array([300.0, 300.0]))
        L_cold = op.laplacian(np.array([20.0, 20.0]))
        self.assertGreater(abs(L_cold[0, 1]), abs(L_warm[0, 1]))

    def test_laplacian_symmetric_and_zero_row_sums(self):
        model = _two_copper_model()
        op = build_temperature_dependent_operator(model, _node_ids(model))
        L = op.laplacian(np.array([50.0, 90.0])).toarray()
        np.testing.assert_allclose(L, L.T, atol=1e-12)
        np.testing.assert_allclose(L.sum(axis=1), np.zeros(2), atol=1e-9)

    def test_rebuild_returns_consistent_shapes(self):
        model = _two_copper_model()
        op = build_temperature_dependent_operator(model, _node_ids(model))
        C, inv_C, L = op.rebuild(np.array([100.0, 200.0]))
        self.assertEqual(C.shape, (2,))
        np.testing.assert_allclose(inv_C, 1.0 / C, rtol=1e-12)
        self.assertEqual(L.shape, (2, 2))


def _two_node_model(*, components, material="Copper", k0=None, temps=(300.0, 300.0)) -> ThermalGraphModel:
    model = ThermalGraphModel(metadata=GraphMetadata(graph_name="contact"))
    for idx, (comp, temperature) in enumerate(zip(components, temps), start=1):
        node = _copper_node(idx, C=100.0, temperature=temperature)
        node.material = material
        if k0 is not None:
            node.k_W_mK = float(k0)
        node.component_name = comp
        model.add_node(node)
    model.set_edge(1, 2, 2.0)
    return model


class ContactModelTests(unittest.TestCase):
    def test_bonded_edge_conducts_more_than_bolted(self):
        # Same component name => bonded (no interface term); different => bolted.
        bonded = build_temperature_dependent_operator(_two_node_model(components=("partA", "partA")), np.array([1, 2]))
        bolted = build_temperature_dependent_operator(
            _two_node_model(components=("partA", "partB")),
            np.array([1, 2]),
            default_bolted_conductance_W_m2K=3000.0,
        )
        T = np.array([300.0, 300.0])
        g_bonded = bonded.edge_conductance(T)[0]
        g_bolted = bolted.edge_conductance(T)[0]
        self.assertGreater(g_bonded, g_bolted)

    def test_bolted_interface_weakens_when_cold(self):
        # Use a high, curve-less conductivity so the edge conductance is
        # interface-dominated; then h(T)=h_ref*(T/Tref)^n makes it drop cold.
        op = build_temperature_dependent_operator(
            _two_node_model(components=("A", "B"), material="SteelX", k0=1.0e6),
            np.array([1, 2]),
            default_bolted_conductance_W_m2K=3000.0,
            contact_temp_exponent=1.0,
            contact_reference_temperature_K=300.0,
        )
        g_warm = float(op.edge_conductance(np.array([300.0, 300.0]))[0])
        g_cold = float(op.edge_conductance(np.array([30.0, 30.0]))[0])
        self.assertLess(g_cold, g_warm)
        # Interface-dominated => ~linear in T (n=1): ~10x drop from 300 K to 30 K.
        self.assertAlmostEqual(g_cold / g_warm, 30.0 / 300.0, delta=0.03)

    def test_temp_exponent_zero_disables_interface_temperature_dependence(self):
        op = build_temperature_dependent_operator(
            _two_node_model(components=("A", "B"), material="SteelX", k0=1.0e6),
            np.array([1, 2]),
            contact_temp_exponent=0.0,
        )
        g_warm = float(op.edge_conductance(np.array([300.0, 300.0]))[0])
        g_cold = float(op.edge_conductance(np.array([30.0, 30.0]))[0])
        self.assertAlmostEqual(g_cold, g_warm, delta=abs(g_warm) * 1e-9)


class IntegrationTests(unittest.TestCase):
    def test_toggle_off_leaves_operator_none(self):
        model = _two_copper_model(temps=(330.0, 280.0))
        prepared = prepare_simulation(model, build_matrices(model), _params())
        self.assertIsNone(prepared.temperature_dependent_operator)

    def test_toggle_on_builds_operator_and_steps(self):
        model = _two_copper_model(temps=(330.0, 280.0))
        prepared = prepare_simulation(
            model, build_matrices(model), _params(use_temperature_dependent_properties=True)
        )
        self.assertIsNotNone(prepared.temperature_dependent_operator)
        for _ in range(5):
            prepared.step_forward()
        self.assertTrue(np.all(np.isfinite(prepared.temperatures_K)))
        # Two coupled bodies should move toward each other (hot cools, cold warms).
        self.assertLess(prepared.temperatures_K[0], 330.0)
        self.assertGreater(prepared.temperatures_K[1], 280.0)

    def test_cold_bodies_equilibrate_faster_than_constant_properties(self):
        # With cp(T) collapse, cold copper has far less thermal mass, so a cold
        # pair equilibrates much faster than the constant-property model predicts.
        def final_gap(use_tdep: bool) -> float:
            model = _two_copper_model(temps=(30.0, 10.0))
            prepared = prepare_simulation(
                model,
                build_matrices(model),
                _params(dt_s=0.2, use_temperature_dependent_properties=use_tdep),
            )
            for _ in range(10):
                prepared.step_forward()
            return abs(float(prepared.temperatures_K[0] - prepared.temperatures_K[1]))

        self.assertLess(final_gap(True), final_gap(False))


if __name__ == "__main__":
    unittest.main()
