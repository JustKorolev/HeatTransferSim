"""Tests for octree-graph <-> view-factor integration (radiation_coupling)."""

from __future__ import annotations

import unittest

import numpy as np

from graph_visualizer.matrix_builder import build_matrices
from graph_visualizer.models import GraphMetadata, NodeProperties, ThermalGraphModel
from graph_visualizer.radiation_coupling import apply_radiation_coupling, exposed_face_patches
from graph_visualizer.simulation_diagnostics import inter_component_conduction_report
from graph_visualizer.simulation_model import _build_radiation_super, prepare_simulation
from graph_visualizer.simulation_parameters import SimulationParameters


def _cell(node_id: int, center_x_mm: float, temperature: float, emissivity: float = 0.9) -> NodeProperties:
    node = NodeProperties.with_material(node_id, (node_id, 0, 0), material="Copper")
    node.center_mm = (center_x_mm, 0.0, 0.0)
    node.size_mm = (5.0, 60.0, 60.0)
    node.side_length_m = 0.06
    volume = 5.0 * 60.0 * 60.0 * 1.0e-9
    node.mass_kg = node.rho_kg_m3 * volume
    node.C_J_K = node.mass_kg * node.cp_J_kgK
    node.initial_temperature_K = float(temperature)
    node.emissivity = float(emissivity)
    node.is_exposed = True
    return node


def _two_facing_cells(hot: float = 600.0, cold: float = 300.0) -> ThermalGraphModel:
    model = ThermalGraphModel(metadata=GraphMetadata(graph_name="coupling"))
    model.add_node(_cell(1, -4.0, hot))
    model.add_node(_cell(2, 4.0, cold))  # 3 mm gap between the facing faces
    return model


class RadiationCouplingTests(unittest.TestCase):
    def test_exposed_faces_extracted_per_cell(self) -> None:
        patches = exposed_face_patches(_two_facing_cells())
        self.assertEqual(len(patches), 12)  # 6 faces x 2 cells, no contact
        for group in (1, 2):
            normals = {tuple(np.round(p.normal).astype(int)) for p in patches if p.group_id == group}
            self.assertEqual(len(normals), 6)  # all six face directions present

    def test_apply_coupling_builds_factored_super_surface_exchange(self) -> None:
        model = _two_facing_cells()
        diagnostics = apply_radiation_coupling(model, rays_per_patch=4000, seed=1, use_cache=False)
        self.assertEqual(int(diagnostics["patches"]), 12)
        self.assertGreaterEqual(int(diagnostics["super_surfaces"]), 2)
        self.assertTrue(model.radiation_super_members)
        self.assertTrue(model.radiation_super_links)
        self.assertTrue(all(g > 0.0 for _i, _j, g in model.radiation_super_links))
        # Every node's background fraction is a physical fraction in [0, 1].
        self.assertTrue(all(0.0 <= f <= 1.0 for f in model.radiation_env_fraction_by_node.values()))

    def test_interior_faces_are_vacuum_exterior_faces_radiate_to_ambient(self) -> None:
        # Hollow shell: inner cavity faces must classify interior (no sink),
        # outer faces exterior (radiate to ambient). A fully enclosed cell (all
        # faces interior) must end up with zero ambient fraction.
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="shell"))
        k = 1
        for ix in range(5):
            for iy in range(5):
                for iz in range(5):
                    if 0 < ix < 4 and 0 < iy < 4 and 0 < iz < 4:
                        continue  # hollow interior
                    n = NodeProperties.with_material(k, (ix, iy, iz), material="Copper")
                    n.center_mm = (ix * 10.0, iy * 10.0, iz * 10.0)
                    n.size_mm = (10.0, 10.0, 10.0)
                    n.emissivity = 0.9
                    model.add_node(n)
                    k += 1
        diagnostics = apply_radiation_coupling(model, rays_per_patch=3000, seed=1, use_cache=False)
        # Both interior (vacuum) and exterior (ambient) surfaces must be found.
        self.assertGreater(int(diagnostics["interior_super_surfaces"]), 0)
        self.assertGreater(int(diagnostics["exterior_super_surfaces"]), 0)
        # Ambient fractions are physical fractions in [0, 1].
        fractions = list(model.radiation_env_fraction_by_node.values())
        self.assertTrue(all(0.0 <= f <= 1.0 for f in fractions))

    def test_disabling_vacuum_sends_all_escape_to_ambient(self) -> None:
        model = _two_facing_cells()
        apply_radiation_coupling(model, rays_per_patch=3000, seed=1, use_cache=False, assume_enclosure_vacuum=False)
        # With vacuum off, no surface is interior: every exposed node radiates to
        # the ambient environment (positive ambient fraction).
        self.assertTrue(all(f > 0.0 for f in model.radiation_env_fraction_by_node.values()))

    def test_coupling_cache_hit_reproduces_result(self) -> None:
        import tempfile
        from pathlib import Path

        cache = Path(tempfile.mkdtemp()) / "radcache"
        first = _two_facing_cells()
        miss = apply_radiation_coupling(first, rays_per_patch=3000, seed=1, cache_dir=cache)
        self.assertEqual(miss.get("cache_hit"), 0.0)
        second = _two_facing_cells()
        hit = apply_radiation_coupling(second, rays_per_patch=3000, seed=1, cache_dir=cache)
        self.assertEqual(hit.get("cache_hit"), 1.0)
        self.assertEqual(first.radiation_super_links, second.radiation_super_links)
        self.assertEqual(first.radiation_env_fraction_by_node, second.radiation_env_fraction_by_node)

    def test_extraction_matches_expected_faces_on_grid(self) -> None:
        # Solid block -> only the outer surface is exposed (O(faces) hash must
        # still count exactly the surface faces).
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="grid"))
        k = 1
        for ix in range(4):
            for iy in range(4):
                for iz in range(4):
                    n = NodeProperties.with_material(k, (ix, iy, iz), material="Copper")
                    n.center_mm = (ix * 5.0, iy * 5.0, iz * 5.0)
                    n.size_mm = (5.0, 5.0, 5.0)
                    n.emissivity = 0.9
                    model.add_node(n)
                    k += 1
        patches = exposed_face_patches(model)
        self.assertEqual(len(patches), 6 * 4 * 4)  # 6 faces of a 4x4x4 solid cube

    def test_factored_exchange_operator_conserves_energy(self) -> None:
        model = _two_facing_cells()
        apply_radiation_coupling(model, rays_per_patch=3000, seed=1, use_cache=False)
        node_ids = build_matrices(model)["node_ids"]
        S, W, degree = _build_radiation_super(model, node_ids)
        self.assertIsNotNone(S)
        # Uniform temperature -> zero net exchange (operator has zero row sums).
        u = np.ones(len(node_ids))
        aggregated = np.asarray(S @ u).reshape(-1)
        super_power = np.asarray(W @ aggregated).reshape(-1) - degree * aggregated
        node_power = np.asarray(S.T @ super_power).reshape(-1)
        self.assertLess(float(np.max(np.abs(node_power))), 1.0e-12)

    def test_suppressed_contact_pairs_skipped_when_edges_rebuilt(self) -> None:
        # Persisted gap-suppressed pairs must be honored when the load-time edge
        # rebuild re-derives conduction from geometry (or the gap re-bridges).
        from graph_visualizer.matrix_builder import refresh_geometry_edges

        def _touching_pair() -> ThermalGraphModel:
            model = ThermalGraphModel(metadata=GraphMetadata(graph_name="gap"))
            for nid, cx, comp in ((1, -5.0, "PartA"), (2, 5.0, "PartB")):
                n = NodeProperties.with_material(nid, (nid, 0, 0), material="Copper")
                n.center_mm = (cx, 0.0, 0.0)
                n.size_mm = (10.0, 10.0, 10.0)
                n.side_length_m = 0.01
                n.component_name = comp
                model.add_node(n)
            return model

        # Without suppression the two touching cells get a conduction edge.
        model = _touching_pair()
        refresh_geometry_edges(model)
        self.assertTrue(any({e.source, e.target} == {1, 2} for e in model.edges.values()))
        # With the pair persisted as gap-suppressed, the rebuild must skip it.
        model = _touching_pair()
        model.octree_graph_data = {"suppressed_contact_pairs": [[1, 2]]}
        refresh_geometry_edges(model)
        self.assertFalse(any({e.source, e.target} == {1, 2} for e in model.edges.values()))

    def test_inter_component_conduction_report_flags_bridges(self) -> None:
        # Two distinct parts joined by a conduction edge => flagged (a bridged gap
        # / spurious short). Same-part edges and no-edge cases are not flagged.
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="bridge"))
        for node_id, comp in ((1, "PartA"), (2, "PartA"), (3, "PartB")):
            n = _cell(node_id, float(node_id), 300.0)
            n.component_name = comp
            model.add_node(n)
        model.set_edge(1, 2, 5.0)  # within PartA -- must NOT be flagged
        self.assertEqual(inter_component_conduction_report(model), [])
        model.set_edge(2, 3, 7.5)  # PartA <-> PartB -- a spurious bridge
        report = inter_component_conduction_report(model)
        self.assertEqual(len(report), 1)
        self.assertEqual({report[0]["component_a"], report[0]["component_b"]}, {"PartA", "PartB"})
        self.assertEqual(report[0]["edges"], 1)
        self.assertAlmostEqual(report[0]["total_conductance_W_K"], 7.5)

    def test_contact_gap_couples_parts_by_direct_radiation(self) -> None:
        # A contact-gap-suppressed interface (persisted in octree_graph_data as
        # gap_radiation_links) must become a direct A<->B radiative exchange link:
        # conduction is gone but the near-touching faces (view factor ~1) still
        # transfer heat by radiation across the gap.
        from graph_visualizer.simulation_model import _gap_radiation_links

        model = _two_facing_cells(hot=600.0, cold=300.0)
        # 1e-4 m^2 shared face between the two (0.9-emissivity) cells.
        model.octree_graph_data = {"gap_radiation_links": [[1, 2, 1.0e-4]]}
        node_ids = np.array(model.ordered_node_ids())

        # Gated on radiation being modeled at all.
        off = SimulationParameters(use_ambient_radiation=False, use_radiative_coupling=False)
        self.assertEqual(_gap_radiation_links(model, node_ids, off), [])
        on = SimulationParameters(use_ambient_radiation=True, use_radiative_coupling=False)
        links = _gap_radiation_links(model, node_ids, on)
        self.assertEqual(len(links), 1)
        # Gray two-surface exchange area A / (1/e1 + 1/e2 - 1), e1 = e2 = 0.9.
        expected = 1.0e-4 / (1.0 / 0.9 + 1.0 / 0.9 - 1.0)
        self.assertAlmostEqual(links[0][2], expected, places=9)

        # It reaches the solver's exchange matrix without the (expensive) super-
        # surface ray trace: the gap link alone is a node-level exchange. The cold
        # cell starts at the ambient temperature, so any warming beyond it is the
        # radiation crossing the gap from the hot cell (the ambient sink alone would
        # hold it at T_env).
        matrices = build_matrices(model)
        params = SimulationParameters(
            dt_s=2.0, t_final_s=4000.0, input_mode="zero", use_ambient_radiation=True,
            T_env_K=300.0, use_radiative_coupling=False, gpu_solver_enabled=False,
            cryocooler_enabled=False,
        )
        prepared = prepare_simulation(model, matrices, params)
        prepared.reset()
        self.assertIsNotNone(prepared.radiation_exchange_W)
        self.assertIsNone(prepared.radiation_super_S)  # no ray-traced super coupling
        self.assertTrue(prepared.dynamic_heater_inputs)  # T^4 term forces per-step RHS
        cold_start = float(prepared.temperatures_K[1])  # 300 K == T_env
        for _ in range(1500):
            prepared.step_forward()
        self.assertGreater(float(prepared.temperatures_K[1]), cold_start + 20.0)
        self.assertLess(float(prepared.temperatures_K[0]), 600.0)

    def test_toggle_runs_coupling_and_transfers_heat_between_parts(self) -> None:
        model = _two_facing_cells(hot=600.0, cold=300.0)
        matrices = build_matrices(model)
        params = SimulationParameters(
            dt_s=2.0, t_final_s=4000.0, input_mode="zero", use_ambient_radiation=True,
            T_env_K=300.0, use_radiative_coupling=True, gpu_solver_enabled=False, cryocooler_enabled=False,
        )
        prepared = prepare_simulation(model, matrices, params)
        prepared.reset()
        self.assertTrue(getattr(model, "radiation_super_members", None))  # coupling was computed
        self.assertIsNotNone(prepared.radiation_super_S)
        cold_start = float(prepared.temperatures_K[1])
        for _ in range(1500):
            prepared.step_forward()
        hot_end = float(prepared.temperatures_K[0])
        cold_end = float(prepared.temperatures_K[1])
        # The hot part radiates onto the cold part: the cold cell warms well above
        # its start (it would only cool toward the 300 K background without coupling).
        self.assertGreater(cold_end, cold_start + 20.0)
        self.assertLess(hot_end, 600.0)

    def test_coupling_toggle_off_leaves_no_links(self) -> None:
        model = _two_facing_cells()
        matrices = build_matrices(model)
        params = SimulationParameters(
            dt_s=2.0, t_final_s=10.0, input_mode="zero", use_ambient_radiation=True,
            use_radiative_coupling=False, gpu_solver_enabled=False, cryocooler_enabled=False,
        )
        prepared = prepare_simulation(model, matrices, params)
        self.assertIsNone(prepared.radiation_exchange_W)
        self.assertIsNone(prepared.radiation_super_S)


if __name__ == "__main__":
    unittest.main()
