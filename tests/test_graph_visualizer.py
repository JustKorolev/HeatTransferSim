"""Tests for sparse thermal graph data and matrix construction."""

from __future__ import annotations

from concurrent.futures import Future
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix, issparse

import graph_visualizer.graph_io as graph_io
import graph_visualizer.matrix_builder as gv_matrix_builder
import graph_visualizer.validation as graph_validation
from graph_visualizer.connectivity import (
    analyze_model_connectivity,
    connectivity_component_color,
    connectivity_component_for_node,
)
from graph_visualizer.draw_tools import (
    clone_node_for_extrusion,
    compute_face_normal,
    extrusion_count_from_projected_pixel_drag,
    next_node_id,
    preview_coords,
)
from graph_visualizer.graph_io import (
    _atomic_write_json,
    load_conductance_matrix_from_folder,
    load_graph_folder,
    save_graph_folder,
)
from graph_visualizer.material_library import default_material_library
from graph_visualizer.mimo_controller import (
    allocate_thermal_rate_qp,
    weighted_rms_error,
)
from graph_visualizer.matrix_builder import (
    DEFAULT_CONTACT_INTERFACE_CONDUCTANCE_W_M2K,
    build_matrices,
    estimate_conductance,
    exposed_areas_from_geometry_m2,
    refresh_auto_edges,
    refresh_radiation_from_exposed_faces,
)
from graph_visualizer.models import EdgeMode, GraphMetadata, NodeProperties, ThermalGraphModel
from graph_visualizer.role_assignment import (
    assign_matching_nodes_to_role,
    node_has_heater_sensor_role,
    node_matches_heater_sensor_filters,
    node_matches_level_filter,
    node_matches_role_substring,
    normalize_role_match_text,
)
from graph_visualizer.role_pairing import assign_heater_to_sensor, recompute_heater_sensor_pairing
from graph_visualizer.role_warnings import has_role_warning, role_warning_reasons
from graph_visualizer.simulation_model import prepare_simulation
from graph_visualizer.simulation_parameters import (
    SimulationParameters,
    apply_initial_temperature_parameter_payload,
    initial_temperature_parameter_payload,
    load_simulation_parameters,
    save_simulation_parameters,
)
from graph_visualizer.sys_id_artifacts import (
    compare_sys_id_gain_matrices,
    list_sys_id_gain_matrices,
    load_sys_id_gain_matrix,
    save_sys_id_gain_matrix,
    update_sys_id_gain_matrix,
)
from graph_visualizer.tooltip_formatters import format_edge_tooltip, format_node_tooltip
from graph_visualizer.two_d_graph_widget import (
    edge_curve_for_positions,
    expand_positions,
    node_connection_counts,
)
from graph_visualizer.validation import validate_conductance_matrix, validate_matrices


def _expected_contact_g(k_i: float, k_j: float, area_m2: float, distance_m: float) -> float:
    half_distance = float(distance_m) * 0.5
    resistance = half_distance / (float(k_i) * float(area_m2))
    resistance += 1.0 / (DEFAULT_CONTACT_INTERFACE_CONDUCTANCE_W_M2K * float(area_m2))
    resistance += half_distance / (float(k_j) * float(area_m2))
    return 1.0 / resistance


class GraphVisualizerModelTests(unittest.TestCase):
    def make_model(self) -> ThermalGraphModel:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="test_graph"))
        node_10 = NodeProperties.with_material(10, (0, 0, 0), material="copper")
        node_2 = NodeProperties.with_material(2, (1, 0, 0), material="aluminum")
        node_7 = NodeProperties.with_material(7, (1, 1, 0), material="FR4 / PCB")
        model.add_node(node_10)
        model.add_node(node_2)
        model.add_node(node_7)
        refresh_auto_edges(model)
        return model

    def test_model_connectivity_analysis_identifies_disconnected_groups(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="connectivity_groups"))
        for node_id, coord in ((1, (0, 0, 0)), (2, (1, 0, 0)), (3, (10, 0, 0)), (4, (11, 0, 0))):
            model.add_node(NodeProperties.with_material(node_id, coord, material="aluminum"))
        model.set_edge(1, 2, 1.0)
        model.set_edge(3, 4, 1.0)

        analysis = analyze_model_connectivity(model)

        self.assertFalse(analysis["connected"])
        self.assertEqual(analysis["component_count"], 2)
        self.assertEqual(analysis["largest_component_size"], 2)
        self.assertEqual(connectivity_component_for_node(analysis, 1), 0)
        self.assertEqual(connectivity_component_for_node(analysis, 3), 1)
        self.assertEqual(set(analysis["disconnected_node_ids"]), {3, 4})
        self.assertEqual(connectivity_component_color(0), "#64748b")
        self.assertNotEqual(connectivity_component_color(1), "#64748b")

    def test_validation_skips_oversized_dense_symmetry_check(self) -> None:
        matrices = {
            "node_ids": np.array([1, 2], dtype=int),
            "coords": np.array([[0, 0, 0], [1, 0, 0]], dtype=int),
            "C": np.ones(2, dtype=float),
            "Grad": np.zeros(2, dtype=float),
            "G": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            "L": np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float),
        }
        old_limit = graph_validation._DENSE_SYMMETRY_CHECK_MAX_BYTES
        graph_validation._DENSE_SYMMETRY_CHECK_MAX_BYTES = 1
        try:
            with patch("graph_visualizer.validation.np.allclose", side_effect=AssertionError):
                self.assertEqual(validate_matrices(matrices, [1, 2]), [])
                self.assertEqual(validate_conductance_matrix(matrices, [1, 2]), [])
        finally:
            graph_validation._DENSE_SYMMETRY_CHECK_MAX_BYTES = old_limit

    def test_auto_edges_use_six_neighbor_adjacency(self) -> None:
        model = self.make_model()
        self.assertEqual(set(model.edges), {(2, 7), (2, 10)})
        self.assertTrue(all(edge.Gij_W_K > 0.0 for edge in model.edges.values()))

    def test_higher_thermal_conductivity_gives_higher_contact_conductance(self) -> None:
        copper_a = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        copper_b = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        aluminum_a = NodeProperties.with_material(3, (0, 0, 0), material="aluminum")
        aluminum_b = NodeProperties.with_material(4, (1, 0, 0), material="aluminum")
        for node in (copper_a, copper_b, aluminum_a, aluminum_b):
            node.side_length_m = 0.01

        copper_G = estimate_conductance(copper_a, copper_b)
        aluminum_G = estimate_conductance(aluminum_a, aluminum_b)

        self.assertGreater(copper_G, aluminum_G)
        self.assertAlmostEqual(copper_G / aluminum_G, copper_a.k_W_mK / aluminum_a.k_W_mK)

    def test_radiating_area_scales_with_exposed_faces(self) -> None:
        isolated = ThermalGraphModel(metadata=GraphMetadata(graph_name="isolated_radiation"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.center_mm = (0.0, 0.0, 0.0)
        node.size_mm = (1.0, 1.0, 1.0)
        isolated.add_node(node)

        refresh_radiation_from_exposed_faces(isolated, reference_temperature_K=300.0)

        self.assertAlmostEqual(isolated.nodes[1].radiating_area_m2, 6.0e-6)
        self.assertTrue(isolated.nodes[1].is_exposed)
        self.assertGreater(isolated.nodes[1].G_rad_W_K, 0.0)

        pair = ThermalGraphModel(metadata=GraphMetadata(graph_name="paired_radiation"))
        left = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        right = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        left.center_mm = (0.0, 0.0, 0.0)
        right.center_mm = (1.0, 0.0, 0.0)
        left.size_mm = right.size_mm = (1.0, 1.0, 1.0)
        pair.add_node(left)
        pair.add_node(right)

        exposed_areas = exposed_areas_from_geometry_m2(pair)

        self.assertAlmostEqual(exposed_areas[1], 5.0e-6)
        self.assertAlmostEqual(exposed_areas[2], 5.0e-6)

    def test_duplicate_node_id_is_rejected(self) -> None:
        model = ThermalGraphModel()
        model.add_node(NodeProperties.with_material(1, (0, 0, 0)))
        with self.assertRaises(ValueError):
            model.add_node(NodeProperties.with_material(1, (1, 0, 0)))

    def test_duplicate_coord_is_rejected(self) -> None:
        model = ThermalGraphModel()
        model.add_node(NodeProperties.with_material(1, (0, 0, 0)))
        with self.assertRaises(ValueError):
            model.add_node(NodeProperties.with_material(2, (0, 0, 0)))

    def test_save_load_round_trip_preserves_matrix_ordering(self) -> None:
        model = self.make_model()
        with TemporaryDirectory() as directory:
            model.nodes[2].initial_temperature_K = 310.0
            matrices = save_graph_folder(model, directory)
            self.assertFalse((Path(directory) / "A.npy").exists())
            self.assertTrue((Path(directory) / "G_rad.npy").exists())
            self.assertEqual(matrices["node_ids"].tolist(), [2, 7, 10])
            loaded_model, loaded_matrices = load_graph_folder(Path(directory))
        self.assertEqual(loaded_model.ordered_node_ids(), [2, 7, 10])
        self.assertEqual(loaded_model.nodes[2].initial_temperature_K, 310.0)
        self.assertIn("G_rad", loaded_matrices)
        self.assertTrue(np.allclose(loaded_matrices["G"], loaded_matrices["G"].T))
        self.assertTrue(np.allclose(np.diag(loaded_matrices["L"]), loaded_matrices["G"].sum(axis=1)))

    def test_atomic_json_write_preserves_existing_file_on_failure(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            path.write_text('{"ok": true}\n', encoding="utf-8")

            with self.assertRaises(TypeError):
                _atomic_write_json(path, {"bad": object()}, indent=2)

            self.assertEqual(path.read_text(encoding="utf-8"), '{"ok": true}\n')
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_atomic_json_write_normalizes_numpy_scalars(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            _atomic_write_json(
                path,
                {
                    "flag": np.bool_(True),
                    "count": np.int64(3),
                    "value": np.float64(1.25),
                    "items": np.array([1, 2, 3]),
                },
                indent=2,
            )

            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded, {"flag": True, "count": 3, "value": 1.25, "items": [1, 2, 3]})

    def test_empty_legacy_octree_matrices_rebuild_geometry_edges_on_load(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="stale_octree"))
        left = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        right = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        left.material = right.material = "Copper"
        left.center_mm = (0.0, 0.0, 0.0)
        right.center_mm = (1.0, 0.0, 0.0)
        left.size_mm = right.size_mm = (1.0, 1.0, 1.0)
        left.C_J_K = right.C_J_K = 10.0
        left.initial_temperature_K = 310.0
        right.initial_temperature_K = 290.0
        model.add_node(left)
        model.add_node(right)
        model.octree_graph_data = {"graph_edges": []}

        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "graph.json").write_text(
                json.dumps(model.to_octree_graph_dict()),
                encoding="utf-8",
            )
            node_count = len(model.nodes)
            np.save(folder / "C.npy", np.array([10.0, 10.0], dtype=float))
            np.save(folder / "G.npy", np.zeros((node_count, node_count), dtype=float))
            np.save(folder / "L.npy", np.zeros((node_count, node_count), dtype=float))
            (folder / "materials_used.json").write_text(
                json.dumps(default_material_library()),
                encoding="utf-8",
            )

            loaded_model, loaded_matrices = load_graph_folder(folder)

        self.assertEqual(set(loaded_model.edges), {(1, 2)})
        self.assertAlmostEqual(loaded_model.nodes[1].k_W_mK, 401.0)
        expected_g = _expected_contact_g(401.0, 401.0, area_m2=1.0e-6, distance_m=1.0e-3)
        self.assertAlmostEqual(loaded_matrices["G"][0, 1], expected_g)
        self.assertTrue(np.any(np.abs(loaded_matrices["L"]) > 0.0))

    def test_octree_load_preserves_consolidated_role_edges(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="consolidated_heater"))
        body = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        body.material = "Copper"
        body.center_mm = (0.0, 0.0, 0.0)
        body.size_mm = (1.0, 1.0, 1.0)
        body.C_J_K = 10.0
        heater = NodeProperties.with_material(20, (20, 0, 0), material="copper")
        heater.material = "Copper"
        heater.center_mm = (20.0, 0.0, 0.0)
        heater.size_mm = (2.0, 1.0, 1.0)
        heater.C_J_K = 5.0
        heater.is_heater = True
        heater.node_type = "heater"
        heater.source_components = ["safe_heater_1"]
        model.add_node(body)
        model.add_node(heater)
        model.set_edge(
            1,
            20,
            0.123,
            "voxel_role_consolidation",
            edge_type="consolidated_role_contact",
        )
        model.octree_graph_data = {"graph_edges": []}

        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "graph.json").write_text(
                json.dumps(model.to_octree_graph_dict()),
                encoding="utf-8",
            )
            np.save(folder / "C.npy", np.array([10.0, 5.0], dtype=float))
            np.save(folder / "G.npy", np.array([[0.0, 0.123], [0.123, 0.0]], dtype=float))
            np.save(folder / "L.npy", np.array([[0.123, -0.123], [-0.123, 0.123]], dtype=float))
            (folder / "materials_used.json").write_text(
                json.dumps(default_material_library()),
                encoding="utf-8",
            )

            loaded_model, loaded_matrices = load_graph_folder(folder)

        self.assertEqual(set(loaded_model.edges), {(1, 20)})
        self.assertEqual(loaded_model.edges[(1, 20)].edge_type, "consolidated_role_contact")
        self.assertAlmostEqual(loaded_model.edges[(1, 20)].Gij_W_K, 0.123)
        self.assertAlmostEqual(loaded_matrices["G"][0, 1], 0.123)

    def test_octree_load_preserves_dedicated_role_node_contact_edges(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="dedicated_heater"))
        body = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        body.material = "Copper"
        body.center_mm = (0.0, 0.0, 0.0)
        body.size_mm = (10.0, 10.0, 10.0)
        body.C_J_K = 10.0
        heater = NodeProperties.with_material(20, (20, 0, 0), material="copper")
        heater.material = "Copper"
        heater.center_mm = (6.0, 0.0, 0.0)
        heater.size_mm = (2.0, 8.0, 8.0)
        heater.C_J_K = 5.0
        heater.is_heater = True
        heater.node_type = "heater"
        heater.source_components = ["safe_heater_1"]
        model.add_node(body)
        model.add_node(heater)
        model.set_edge(
            1,
            20,
            0.123,
            "cad_role_node_contact",
            edge_type="role_node_contact",
        )
        model.octree_graph_data = {"graph_edges": []}

        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "graph.json").write_text(
                json.dumps(model.to_octree_graph_dict()),
                encoding="utf-8",
            )
            np.save(folder / "C.npy", np.array([10.0, 5.0], dtype=float))
            np.save(folder / "G.npy", np.array([[0.0, 0.123], [0.123, 0.0]], dtype=float))
            np.save(folder / "L.npy", np.array([[0.123, -0.123], [-0.123, 0.123]], dtype=float))
            (folder / "materials_used.json").write_text(
                json.dumps(default_material_library()),
                encoding="utf-8",
            )

            loaded_model, loaded_matrices = load_graph_folder(folder)

        self.assertEqual(set(loaded_model.edges), {(1, 20)})
        self.assertEqual(loaded_model.edges[(1, 20)].edge_type, "role_node_contact")
        self.assertAlmostEqual(loaded_model.edges[(1, 20)].Gij_W_K, 0.123)
        self.assertAlmostEqual(loaded_matrices["G"][0, 1], 0.0)

    def test_large_octree_load_uses_sparse_laplacian_without_dense_g(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="large_sparse_octree"))
        left = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        right = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        left.material = right.material = "Copper"
        left.center_mm = (0.0, 0.0, 0.0)
        right.center_mm = (1.0, 0.0, 0.0)
        left.size_mm = right.size_mm = (1.0, 1.0, 1.0)
        left.C_J_K = right.C_J_K = 10.0
        model.add_node(left)
        model.add_node(right)
        model.set_edge(1, 2, 0.5, EdgeMode.AUTO.value)
        model.octree_graph_data = {"graph_edges": []}

        old_limit = graph_io._DENSE_OCTREE_MATRIX_NODE_LIMIT
        graph_io._DENSE_OCTREE_MATRIX_NODE_LIMIT = 1
        try:
            with TemporaryDirectory() as directory:
                folder = Path(directory)
                (folder / "graph.json").write_text(
                    json.dumps(model.to_octree_graph_dict()),
                    encoding="utf-8",
                )

                loaded_model, loaded_matrices = load_graph_folder(folder)
        finally:
            graph_io._DENSE_OCTREE_MATRIX_NODE_LIMIT = old_limit

        self.assertEqual(set(loaded_model.edges), {(1, 2)})
        self.assertNotIn("G", loaded_matrices)
        self.assertEqual(loaded_matrices["L"].shape, (2, 2))
        self.assertTrue(hasattr(loaded_matrices["L"], "tocsr"))
        self.assertGreater(float(loaded_matrices["L"][0, 0]), 0.0)
        self.assertLess(float(loaded_matrices["L"][0, 1]), 0.0)

    def test_visualizer_matrix_builder_uses_sparse_laplacian_above_limit(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="large_visualizer_matrix"))
        left = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        right = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        for node in (left, right):
            node.C_J_K = 10.0
            model.add_node(node)
        model.set_edge(1, 2, 0.75, EdgeMode.AUTO.value)

        matrices = build_matrices(model, dense_node_limit=1)

        self.assertNotIn("G", matrices)
        self.assertTrue(issparse(matrices["L"]))
        self.assertAlmostEqual(float(matrices["L"][0, 0]), 0.75)
        self.assertAlmostEqual(float(matrices["L"][0, 1]), -0.75)

    def test_visualizer_matrix_builder_forces_sparse_when_dense_byte_budget_exceeded(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="byte_guard_visualizer_matrix"))
        left = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        right = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        for node in (left, right):
            node.C_J_K = 10.0
            model.add_node(node)
        model.set_edge(1, 2, 0.25, EdgeMode.AUTO.value)
        old_limit = gv_matrix_builder.DENSE_MATRIX_MAX_TOTAL_BYTES
        gv_matrix_builder.DENSE_MATRIX_MAX_TOTAL_BYTES = 1
        try:
            matrices = build_matrices(model, dense_node_limit=100)
        finally:
            gv_matrix_builder.DENSE_MATRIX_MAX_TOTAL_BYTES = old_limit

        self.assertNotIn("G", matrices)
        self.assertTrue(issparse(matrices["L"]))
        self.assertAlmostEqual(float(matrices["L"][0, 0]), 0.25)

    def test_large_octree_load_preserves_existing_sparse_laplacian(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="preserve_sparse_octree"))
        left = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        right = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        for node in (left, right):
            node.material = "Copper"
            node.size_mm = (1.0, 1.0, 1.0)
            node.C_J_K = 10.0
            model.add_node(node)
        model.set_edge(1, 2, 99.0, EdgeMode.AUTO.value)
        model.octree_graph_data = {"graph_edges": []}

        old_limit = graph_io._DENSE_OCTREE_MATRIX_NODE_LIMIT
        graph_io._DENSE_OCTREE_MATRIX_NODE_LIMIT = 1
        try:
            with TemporaryDirectory() as directory:
                folder = Path(directory)
                (folder / "graph.json").write_text(
                    json.dumps(model.to_octree_graph_dict()),
                    encoding="utf-8",
                )
                (folder / "L_sparse.json").write_text(
                    json.dumps(
                        {
                            "shape": [2, 2],
                            "format": "coo",
                            "row": [0, 0, 1, 1],
                            "col": [0, 1, 0, 1],
                            "data": [0.25, -0.25, -0.25, 0.25],
                        }
                    ),
                    encoding="utf-8",
                )

                loaded_model, loaded_matrices = load_graph_folder(folder)
        finally:
            graph_io._DENSE_OCTREE_MATRIX_NODE_LIMIT = old_limit

        self.assertEqual(set(loaded_model.edges), {(1, 2)})
        self.assertTrue(issparse(loaded_matrices["L"]))
        self.assertAlmostEqual(float(loaded_matrices["L"][0, 0]), 0.25)

    def test_octree_load_preserves_complete_saved_radiation_without_recomputing(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="saved_radiation"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.material = "Copper"
        node.center_mm = (0.0, 0.0, 0.0)
        node.size_mm = (1.0, 1.0, 1.0)
        node.C_J_K = 10.0
        node.is_exposed = True
        node.radiating_area_m2 = 1.25e-6
        node.G_rad_W_K = 4.5e-8
        node.Grad_W_K = 4.5e-8
        model.add_node(node)
        model.octree_graph_data = {"graph_edges": []}

        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "graph.json").write_text(
                json.dumps(model.to_octree_graph_dict()),
                encoding="utf-8",
            )

            with patch("graph_visualizer.graph_io.refresh_radiation_from_exposed_faces") as refresh:
                loaded_model, loaded_matrices = load_graph_folder(folder)

        refresh.assert_not_called()
        self.assertTrue(loaded_model.nodes[1].is_exposed)
        self.assertAlmostEqual(loaded_model.nodes[1].radiating_area_m2, 1.25e-6)
        self.assertAlmostEqual(loaded_model.nodes[1].G_rad_W_K, 4.5e-8)
        self.assertAlmostEqual(float(loaded_matrices["G_rad"][0]), 4.5e-8)

    def test_octree_load_recomputes_radiation_for_incomplete_legacy_payload(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="legacy_radiation"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.material = "Copper"
        node.center_mm = (0.0, 0.0, 0.0)
        node.size_mm = (1.0, 1.0, 1.0)
        node.C_J_K = 10.0
        model.add_node(node)
        model.octree_graph_data = {"graph_edges": []}
        payload = model.to_octree_graph_dict()
        payload["graph_nodes"][0].pop("radiation")

        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "graph.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            with patch("graph_visualizer.graph_io.refresh_radiation_from_exposed_faces", return_value=0) as refresh:
                load_graph_folder(folder)

        refresh.assert_called_once()

    def test_octree_load_recomputes_radiation_for_all_zero_placeholder_payload(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="placeholder_radiation"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.material = "Copper"
        node.center_mm = (0.0, 0.0, 0.0)
        node.size_mm = (1.0, 1.0, 1.0)
        node.C_J_K = 10.0
        model.add_node(node)
        model.octree_graph_data = {"graph_edges": []}

        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "graph.json").write_text(
                json.dumps(model.to_octree_graph_dict()),
                encoding="utf-8",
            )

            with patch("graph_visualizer.graph_io.refresh_radiation_from_exposed_faces", return_value=0) as refresh:
                load_graph_folder(folder)

        refresh.assert_called_once()

    def test_large_octree_save_preserves_existing_matrix_payload(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="metadata_only_large_save"))
        left = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        right = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        for node in (left, right):
            node.material = "Copper"
            node.size_mm = (1.0, 1.0, 1.0)
            node.C_J_K = 10.0
            model.add_node(node)
        model.set_edge(1, 2, 0.5, EdgeMode.AUTO.value)
        model.octree_graph_data = {"graph_edges": []}

        old_limit = graph_io._DENSE_OCTREE_MATRIX_NODE_LIMIT
        graph_io._DENSE_OCTREE_MATRIX_NODE_LIMIT = 1
        try:
            with TemporaryDirectory() as directory:
                folder = Path(directory)
                (folder / "L_sparse.json").write_text(
                    json.dumps(
                        {
                            "shape": [2, 2],
                            "format": "coo",
                            "row": [0, 0, 1, 1],
                            "col": [0, 1, 0, 1],
                            "data": [0.5, -0.5, -0.5, 0.5],
                        }
                    ),
                    encoding="utf-8",
                )
                with patch("graph_visualizer.graph_io._save_octree_outputs") as save_outputs:
                    matrices = save_graph_folder(model, folder)
                save_outputs.assert_not_called()
                self.assertTrue((folder / "graph.json").exists())
                self.assertTrue(issparse(matrices["L"]))
        finally:
            graph_io._DENSE_OCTREE_MATRIX_NODE_LIMIT = old_limit

    def test_stale_octree_conductance_rebuilds_after_material_load(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="stale_conductance"))
        left = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        right = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        left.material = right.material = "Copper"
        left.center_mm = (0.0, 0.0, 0.0)
        right.center_mm = (1.0, 0.0, 0.0)
        left.size_mm = right.size_mm = (1.0, 1.0, 1.0)
        left.C_J_K = right.C_J_K = 10.0
        model.add_node(left)
        model.add_node(right)
        model.set_edge(1, 2, 0.002, EdgeMode.AUTO.value)
        model.octree_graph_data = {"graph_edges": []}

        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "graph.json").write_text(
                json.dumps(model.to_octree_graph_dict()),
                encoding="utf-8",
            )
            np.save(folder / "C.npy", np.array([10.0, 10.0], dtype=float))
            np.save(folder / "G.npy", np.array([[0.0, 0.002], [0.002, 0.0]], dtype=float))
            np.save(folder / "L.npy", np.array([[0.002, -0.002], [-0.002, 0.002]], dtype=float))
            (folder / "materials_used.json").write_text(
                json.dumps(default_material_library()),
                encoding="utf-8",
            )

            loaded_model, loaded_matrices = load_graph_folder(folder)

        self.assertAlmostEqual(loaded_model.nodes[1].k_W_mK, 401.0)
        expected_g = _expected_contact_g(401.0, 401.0, area_m2=1.0e-6, distance_m=1.0e-3)
        self.assertAlmostEqual(loaded_model.edges[(1, 2)].Gij_W_K, expected_g)
        self.assertAlmostEqual(loaded_matrices["G"][0, 1], expected_g)

    def test_simulation_uses_actual_radiation_term_and_toggle(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="radiation_graph"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.emissivity = 0.5
        node.radiating_area_m2 = 2.0
        model.add_node(node)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
            "initial_temperature_K": np.array([300.0]),
        }
        params = SimulationParameters(dt_s=1.0, T_env_K=290.0, use_ambient_radiation=True)
        prepared = prepare_simulation(model, matrices, params)
        prepared.step_forward()
        sigma = 5.670374419e-8
        expected = 300.0 + (0.5 * sigma * 2.0 / 10.0) * (290.0**4 - 300.0**4)
        self.assertAlmostEqual(prepared.temperatures_K[0], expected)

        params.use_ambient_radiation = False
        prepared = prepare_simulation(model, matrices, params)
        prepared.step_forward()
        self.assertAlmostEqual(prepared.temperatures_K[0], 300.0)

    def test_simulation_initial_condition_uses_edited_node_temperature_over_stale_matrix(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="stale_initial"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 325.0
        model.add_node(node)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
            "initial_temperature_K": np.array([293.15]),
        }

        prepared = prepare_simulation(model, matrices, SimulationParameters(dt_s=1.0))

        self.assertAlmostEqual(prepared.temperatures_K[0], 325.0)

    def test_heater_control_manual_power_is_used_for_heater_inputs(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="manual_heater"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 20.0
        node.assigned_sensor_id = 1
        node.assigned_heater_id = 1
        node.sensor_control_mode = "manual"
        node.sensor_manual_power_W = 5.0
        model.add_node(node)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }

        prepared = prepare_simulation(model, matrices, SimulationParameters(dt_s=1.0, input_mode="heater_inputs"))
        self.assertEqual(prepared.heater_power_by_node(), {1: 5.0})
        prepared.step_forward()

        self.assertGreater(float(prepared.temperatures_K[0]), 300.0)

    def test_paired_manual_heater_power_is_read_from_heater_not_sensor(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="manual_heater_per_actuator"))
        sensor = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        sensor.C_J_K = 10.0
        sensor.initial_temperature_K = 300.0
        sensor.is_sensor = True
        sensor.assigned_heater_ids = [2]
        sensor.sensor_manual_power_W = 100.0
        heater = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        heater.C_J_K = 10.0
        heater.initial_temperature_K = 300.0
        heater.is_heater = True
        heater.assigned_sensor_id = 1
        heater.heater.heater_max_power_W = 20.0
        heater.sensor_control_mode = "manual"
        heater.sensor_manual_power_W = 7.0
        model.add_node(sensor)
        model.add_node(heater)
        matrices = {
            "node_ids": np.array([1, 2], dtype=int),
            "C": np.array([10.0, 10.0]),
            "L": np.zeros((2, 2)),
            "G_rad": np.array([0.0, 0.0]),
        }

        prepared = prepare_simulation(model, matrices, SimulationParameters(dt_s=1.0, input_mode="heater_inputs"))

        self.assertEqual(prepared.heater_power_by_node(), {2: 7.0})

    def test_cryocooler_removes_heat_above_setpoint_only(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="cryocooler"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.has_cryocooler = True
        model.add_node(node)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }
        params = SimulationParameters(
            dt_s=1.0,
            input_mode="zero",
            Kp_cooler=0.5,
            P_cooler_max=10.0,
            T_cooler_setpoint=270.0,
        )

        prepared = prepare_simulation(model, matrices, params)

        self.assertEqual(prepared.cryocooler_power_by_node(), {1: 10.0})
        self.assertEqual(prepared.heater_power_by_node(), {1: -10.0})
        prepared.step_forward()
        self.assertLess(float(prepared.temperatures_K[0]), 300.0)

        prepared.z[0] = 260.0

        self.assertEqual(prepared.cryocooler_power_by_node(), {1: 0.0})
        self.assertEqual(prepared.heater_power_by_node(), {1: 0.0})

    def test_mimo_controller_math_helpers(self) -> None:
        self.assertAlmostEqual(weighted_rms_error(np.array([2.0, 4.0]), np.array([1.0, 3.0])), np.sqrt(13.0))

        result = allocate_thermal_rate_qp(
            np.array([[2.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([10.0], dtype=float),
            np.array([1.0], dtype=float),
            np.array([20.0], dtype=float),
            np.array([0.0], dtype=float),
            0.0,
            0.0,
        )

        self.assertEqual(result.u.shape, (1,))
        self.assertAlmostEqual(float(result.u[0]), 5.0)
        self.assertTrue(result.solver_success)

    def test_mimo_rate_allocator_penalizes_power_changes(self) -> None:
        result = allocate_thermal_rate_qp(
            np.array([[1.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([10.0], dtype=float),
            np.array([1.0], dtype=float),
            np.array([20.0], dtype=float),
            np.array([2.0], dtype=float),
            0.0,
            1.0,
        )

        self.assertTrue(result.solver_success)
        self.assertAlmostEqual(float(result.u[0]), 7.0)
        self.assertFalse(result.bounds_active)

    def test_mimo_rate_allocator_penalizes_distance_from_reference_power(self) -> None:
        result = allocate_thermal_rate_qp(
            np.array([[1.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([10.0], dtype=float),
            np.array([1.0], dtype=float),
            np.array([20.0], dtype=float),
            np.array([0.0], dtype=float),
            1.0,
            0.0,
            None,
            np.array([20.0], dtype=float),
        )

        self.assertTrue(result.solver_success)
        self.assertAlmostEqual(float(result.u[0]), 15.0)
        self.assertFalse(result.bounds_active)

    def test_mimo_rate_allocator_enforces_hard_slew_bound(self) -> None:
        result = allocate_thermal_rate_qp(
            np.array([[1.0]], dtype=float),
            np.array([0.0], dtype=float),
            np.array([10.0], dtype=float),
            np.array([1.0], dtype=float),
            np.array([20.0], dtype=float),
            np.array([2.0], dtype=float),
            0.0,
            0.0,
            np.array([3.0], dtype=float),
        )

        self.assertTrue(result.solver_success)
        self.assertAlmostEqual(float(result.u[0]), 5.0)
        self.assertTrue(result.bounds_active)

    def test_mimo_controller_drives_heater_from_dynamic_rate_gain(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_controller"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 20.0
        node.heater_control.mode = "mimo"
        node.controller_setpoint_K = 310.0
        node.controller_kp_coarse = 1.0
        node.controller_ki_coarse = 0.0
        model.add_node(node)
        model.set_controller_gain(1, 1, 2.0)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }
        params = SimulationParameters(
            dt_s=1.0,
            input_mode="heater_inputs",
            mimo_lambda_u=0.0,
            use_ambient_radiation=False,
        )

        prepared = prepare_simulation(model, matrices, params)

        self.assertAlmostEqual(prepared.heater_power_by_node()[1], 2.5)
        prepared.step_forward()
        self.assertGreater(float(prepared.temperatures_K[0]), 300.0)
        self.assertEqual(prepared.controller_mode, "coarse")

    def test_mimo_controller_uses_separate_sensor_and_heater_nodes(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_separate_roles"))
        sensor = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        sensor.C_J_K = 10.0
        sensor.initial_temperature_K = 300.0
        sensor.is_sensor = True
        sensor.sensor_control_mode = "mimo"
        sensor.assigned_heater_id = 2
        sensor.controller_setpoint_K = 310.0
        sensor.controller_kp_coarse = 1.0
        sensor.controller_ki_coarse = 0.0
        heater = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        heater.C_J_K = 10.0
        heater.initial_temperature_K = 300.0
        heater.is_heater = True
        heater.assigned_sensor_id = 1
        heater.heater.heater_max_power_W = 20.0
        body = NodeProperties.with_material(3, (2, 0, 0), material="copper")
        body.C_J_K = 10.0
        body.initial_temperature_K = 300.0
        model.add_node(sensor)
        model.add_node(heater)
        model.add_node(body)
        model.set_edge(1, 3, 0.1)
        matrices = {
            "node_ids": np.array([1, 2, 3], dtype=int),
            "C": np.array([10.0, 10.0, 10.0]),
            "L": np.zeros((3, 3)),
            "G_rad": np.array([0.0, 0.0, 0.0]),
        }
        params = SimulationParameters(
            dt_s=1.0,
            input_mode="heater_inputs",
            mimo_lambda_u=0.0,
            use_ambient_radiation=False,
        )

        prepared = prepare_simulation(model, matrices, params)

        self.assertAlmostEqual(prepared.heater_power_by_node()[2], 2.5)
        self.assertEqual(prepared.controller_mode, "coarse")

    def test_mimo_qp_only_includes_heaters_with_mimo_enabled(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_only_enabled_heaters"))
        sensor = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        sensor.C_J_K = 10.0
        sensor.initial_temperature_K = 300.0
        sensor.is_sensor = True
        sensor.assigned_heater_ids = [2, 3]
        sensor.readout_node_ids = [1]
        sensor.controller_setpoint_K = 310.0
        mimo_heater = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        mimo_heater.C_J_K = 10.0
        mimo_heater.initial_temperature_K = 300.0
        mimo_heater.is_heater = True
        mimo_heater.assigned_sensor_id = 1
        mimo_heater.heater.heater_max_power_W = 20.0
        mimo_heater.sensor_control_mode = "mimo"
        mimo_heater.controller_kp_coarse = 1.0
        mimo_heater.power_deposition_node_ids = [2]
        mimo_heater.power_deposition_weights = [1.0]
        manual_heater = NodeProperties.with_material(3, (2, 0, 0), material="copper")
        manual_heater.C_J_K = 10.0
        manual_heater.initial_temperature_K = 300.0
        manual_heater.is_heater = True
        manual_heater.assigned_sensor_id = 1
        manual_heater.heater.heater_max_power_W = 20.0
        manual_heater.sensor_control_mode = "manual"
        manual_heater.sensor_manual_power_W = 4.0
        manual_heater.power_deposition_node_ids = [3]
        manual_heater.power_deposition_weights = [1.0]
        body = NodeProperties.with_material(4, (3, 0, 0), material="copper")
        body.C_J_K = 10.0
        body.initial_temperature_K = 300.0
        model.add_node(sensor)
        model.add_node(mimo_heater)
        model.add_node(manual_heater)
        model.add_node(body)
        model.set_edge(1, 4, 0.1)
        model.set_controller_gain(1, 2, 2.0)
        model.set_controller_gain(1, 3, 999.0)
        matrices = {
            "node_ids": np.array([1, 2, 3, 4], dtype=int),
            "C": np.array([10.0, 10.0, 10.0, 10.0]),
            "L": np.zeros((4, 4)),
            "G_rad": np.array([0.0, 0.0, 0.0, 0.0]),
        }
        params = SimulationParameters(
            dt_s=1.0,
            input_mode="heater_inputs",
            mimo_controller_enabled=True,
            mimo_lambda_u=0.0,
            use_ambient_radiation=False,
        )

        prepared = prepare_simulation(model, matrices, params)
        powers = prepared.heater_power_by_node()
        diagnostics = prepared.controller_allocator_diagnostics

        self.assertEqual(diagnostics["heater_ids"], [2])
        self.assertEqual(diagnostics["active_heater_count"], 1)
        self.assertEqual(len(diagnostics["B_s"][0]), 1)
        self.assertEqual(len(diagnostics["heater_commands_W"]), 1)
        self.assertAlmostEqual(powers[2], 2.5)
        self.assertAlmostEqual(powers[3], 4.0)

    def test_bulk_role_assignment_matches_normalized_component_substring_per_node(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="bulk_roles"))
        left = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        left.component_name = "V_GUUTZ_SAFE-HEATER_LEFT"
        right = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        right.component_name = "V_GUUTZ_SAFE_HEATER_RIGHT"
        body = NodeProperties.with_material(3, (2, 0, 0), material="copper")
        body.component_name = "V_GUUTZ_SENSOR_HEATER_CABLE"
        for node in (left, right, body):
            model.add_node(node)

        matched = assign_matching_nodes_to_role(model, "safe-heater", "heater")

        self.assertEqual(matched, [1, 2])
        self.assertTrue(model.nodes[1].is_heater)
        self.assertTrue(model.nodes[2].is_heater)
        self.assertFalse(model.nodes[3].is_heater)
        self.assertEqual(model.nodes[1].heater.heater_id, 1)
        self.assertEqual(model.nodes[2].heater.heater_id, 2)
        self.assertEqual(len(model.nodes), 3)

    def test_bulk_role_assignment_can_match_source_components_as_sensor(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="bulk_source_roles"))
        sensor = NodeProperties.with_material(4, (0, 0, 0), material="copper")
        sensor.component_name = "cell_body"
        sensor.source_components = ["assembly/TEMP-PROBE-A"]
        sensor.is_heater = True
        model.add_node(sensor)

        matched = assign_matching_nodes_to_role(model, "temp probe", "sensor")

        self.assertEqual(matched, [4])
        self.assertFalse(model.nodes[4].is_heater)
        self.assertTrue(model.nodes[4].is_sensor)
        self.assertEqual(model.nodes[4].sensor.sensor_id, 4)

    def test_cad_role_nodes_bypass_octree_level_filter_and_match_source_search(self) -> None:
        heater = NodeProperties.with_material(8, (0, 0, 0), material="copper")
        heater.node_type = "heater"
        heater.level = -1
        heater.source_components = ["assembly/V_GUUTZ_SAFE-HEATER"]
        body = NodeProperties.with_material(9, (1, 0, 0), material="copper")
        body.level = 1

        self.assertTrue(node_matches_level_filter(heater, 0, 99))
        self.assertFalse(node_matches_level_filter(body, 2, 99))
        self.assertTrue(
            node_matches_role_substring(
                heater,
                normalize_role_match_text("safe heater"),
            )
        )

    def test_heater_sensor_filter_excludes_cryocooler_only_nodes(self) -> None:
        heater = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        heater.is_heater = True
        sensor = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        sensor.is_sensor = True
        cooler = NodeProperties.with_material(3, (2, 0, 0), material="copper")
        cooler.has_cryocooler = True
        body = NodeProperties.with_material(4, (3, 0, 0), material="copper")

        self.assertTrue(node_has_heater_sensor_role(heater))
        self.assertTrue(node_has_heater_sensor_role(sensor))
        self.assertFalse(node_has_heater_sensor_role(cooler))
        self.assertFalse(node_has_heater_sensor_role(body))

        self.assertTrue(node_matches_heater_sensor_filters(body, False, False))
        self.assertTrue(node_matches_heater_sensor_filters(heater, True, False))
        self.assertFalse(node_matches_heater_sensor_filters(sensor, True, False))
        self.assertTrue(node_matches_heater_sensor_filters(sensor, False, True))
        self.assertFalse(node_matches_heater_sensor_filters(heater, False, True))
        self.assertTrue(node_matches_heater_sensor_filters(heater, True, True))
        self.assertTrue(node_matches_heater_sensor_filters(sensor, True, True))
        self.assertFalse(node_matches_heater_sensor_filters(cooler, True, True))

    def test_node_connection_counts_reports_total_and_visible_neighbors(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="connection_counts"))
        for node_id in (1, 2, 3, 4):
            model.add_node(NodeProperties.with_material(node_id, (node_id, 0, 0), material="copper"))
        model.set_edge(1, 2, 0.5)
        model.set_edge(1, 3, 0.25)
        model.set_edge(2, 4, 0.1)

        self.assertEqual(node_connection_counts(model, 1), (2, 2))
        self.assertEqual(node_connection_counts(model, 1, {1, 2}), (2, 1))

    def test_role_pairing_uses_sensor_body_connections_and_aabb_gap(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="role_pairing"))
        heater = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        heater.is_heater = True
        heater.center_mm = (0.0, 0.0, 0.0)
        heater.size_mm = (2.0, 2.0, 2.0)
        sensor = NodeProperties.with_material(2, (4, 0, 0), material="copper")
        sensor.is_sensor = True
        sensor.center_mm = (4.0, 0.0, 0.0)
        sensor.size_mm = (2.0, 2.0, 2.0)
        body = NodeProperties.with_material(3, (8, 0, 0), material="copper")
        for node in (heater, sensor, body):
            model.add_node(node)
        model.set_edge(2, 3, 0.1)

        warnings = recompute_heater_sensor_pairing(model, max_distance_mm=2.1)

        self.assertFalse(warnings)
        self.assertEqual(model.nodes[1].assigned_sensor_id, 2)
        self.assertEqual(model.nodes[2].assigned_heater_id, 1)
        self.assertEqual(model.nodes[2].assigned_heater_ids, [1])
        self.assertAlmostEqual(float(model.nodes[1].sensor_pair_distance_mm), 2.0)
        self.assertEqual(model.nodes[2].sensor_connected_node_ids, [3])
        self.assertFalse(model.nodes[2].sensor_monitor_only)

    def test_role_pairing_assigns_each_sensor_to_one_nearest_heater(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="one_to_one_pairing"))
        heater_a = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        heater_a.is_heater = True
        heater_a.center_mm = (0.0, 0.0, 0.0)
        heater_a.size_mm = (2.0, 2.0, 2.0)
        heater_b = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        heater_b.is_heater = True
        heater_b.center_mm = (1.0, 0.0, 0.0)
        heater_b.size_mm = (2.0, 2.0, 2.0)
        sensor = NodeProperties.with_material(3, (3, 0, 0), material="copper")
        sensor.is_sensor = True
        sensor.center_mm = (3.0, 0.0, 0.0)
        sensor.size_mm = (2.0, 2.0, 2.0)
        body = NodeProperties.with_material(4, (4, 0, 0), material="copper")
        for node in (heater_a, heater_b, sensor, body):
            model.add_node(node)
        model.set_edge(3, 4, 0.1)

        warnings = recompute_heater_sensor_pairing(model, max_distance_mm=3.0)

        self.assertIn("Heater node 1 has no available valid unpaired sensor", " ".join(warnings))
        self.assertIsNone(model.nodes[1].assigned_sensor_id)
        self.assertEqual(model.nodes[2].assigned_sensor_id, 3)
        self.assertEqual(model.nodes[3].assigned_heater_ids, [2])
        self.assertEqual(model.nodes[3].assigned_heater_id, 2)
        self.assertFalse(model.nodes[3].sensor_monitor_only)

    def test_role_pairing_allows_configured_multiple_heaters_per_sensor(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="many_to_one_pairing"))
        heater_a = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        heater_a.is_heater = True
        heater_a.center_mm = (0.0, 0.0, 0.0)
        heater_a.size_mm = (2.0, 2.0, 2.0)
        heater_b = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        heater_b.is_heater = True
        heater_b.center_mm = (1.0, 0.0, 0.0)
        heater_b.size_mm = (2.0, 2.0, 2.0)
        sensor = NodeProperties.with_material(3, (3, 0, 0), material="copper")
        sensor.is_sensor = True
        sensor.center_mm = (3.0, 0.0, 0.0)
        sensor.size_mm = (2.0, 2.0, 2.0)
        body = NodeProperties.with_material(4, (4, 0, 0), material="copper")
        for node in (heater_a, heater_b, sensor, body):
            model.add_node(node)
        model.set_edge(3, 4, 0.1)

        warnings = recompute_heater_sensor_pairing(model, max_distance_mm=3.0, max_heaters_per_sensor=2)

        self.assertNotIn("no available valid unpaired sensor", " ".join(warnings))
        self.assertEqual(model.nodes[1].assigned_sensor_id, 3)
        self.assertEqual(model.nodes[2].assigned_sensor_id, 3)
        self.assertEqual(model.nodes[3].assigned_heater_ids, [1, 2])
        self.assertEqual(model.nodes[3].assigned_heater_id, 1)
        self.assertFalse(model.nodes[3].sensor_monitor_only)

    def test_sensor_touching_heater_inherits_heater_body_readout_nodes(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="embedded_sensor"))
        heater = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        heater.is_heater = True
        heater.center_mm = (0.0, 0.0, 0.0)
        heater.size_mm = (2.0, 2.0, 2.0)
        sensor = NodeProperties.with_material(2, (0, 1, 0), material="copper")
        sensor.is_sensor = True
        sensor.center_mm = (0.0, 0.0, 0.0)
        sensor.size_mm = (1.0, 1.0, 1.0)
        body = NodeProperties.with_material(3, (1, 0, 0), material="copper")
        for node in (heater, sensor, body):
            model.add_node(node)
        model.set_edge(1, 2, 0.1)
        model.set_edge(1, 3, 0.1)

        warnings = recompute_heater_sensor_pairing(model, max_distance_mm=1.0)

        self.assertEqual(model.nodes[2].sensor_connected_node_ids, [3])
        self.assertTrue(model.nodes[2].sensor_valid)
        self.assertEqual(model.nodes[1].assigned_sensor_id, 2)
        self.assertIn("heater-adjacent body node", " ".join(warnings))

    def test_manual_pairing_reassigns_sensor_one_to_one(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="manual_one_to_one"))
        heater_a = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        heater_a.is_heater = True
        heater_b = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        heater_b.is_heater = True
        sensor = NodeProperties.with_material(3, (2, 0, 0), material="copper")
        sensor.is_sensor = True
        body = NodeProperties.with_material(4, (3, 0, 0), material="copper")
        for node in (heater_a, heater_b, sensor, body):
            model.add_node(node)
        model.set_edge(3, 4, 0.1)

        assign_heater_to_sensor(model, 1, 3)
        assign_heater_to_sensor(model, 2, 3)

        self.assertIsNone(model.nodes[1].assigned_sensor_id)
        self.assertEqual(model.nodes[2].assigned_sensor_id, 3)
        self.assertEqual(model.nodes[3].assigned_heater_ids, [2])

    def test_paired_mimo_uses_average_sensor_readout_and_inverse_connected_capacitance(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="paired_mimo"))
        sensor = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        sensor.is_sensor = True
        sensor.sensor_control_mode = "mimo"
        sensor.assigned_heater_id = 2
        sensor.controller_setpoint_K = 315.0
        sensor.controller_kp_coarse = 0.1
        heater = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        heater.is_heater = True
        heater.assigned_sensor_id = 1
        heater.heater.heater_max_power_W = 100.0
        body_a = NodeProperties.with_material(3, (2, 0, 0), material="copper")
        body_a.initial_temperature_K = 300.0
        body_a.C_J_K = 10.0
        body_b = NodeProperties.with_material(4, (3, 0, 0), material="copper")
        body_b.initial_temperature_K = 310.0
        body_b.C_J_K = 20.0
        for node in (sensor, heater, body_a, body_b):
            model.add_node(node)
        model.set_edge(1, 3, 0.1)
        model.set_edge(1, 4, 0.1)
        matrices = {
            "node_ids": np.array([1, 2, 3, 4], dtype=int),
            "C": np.array([5.0, 5.0, 10.0, 20.0]),
            "L": np.zeros((4, 4)),
            "G_rad": np.zeros(4),
        }

        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(
                dt_s=1.0,
                input_mode="heater_inputs",
                mimo_lambda_u=0.0,
                mimo_v_cmd_abs_max_K_per_s=100.0,
                heater_sensor_pair_alpha=2.0,
                use_ambient_radiation=False,
            ),
        )

        power = prepared.heater_power_by_node()[2]
        self.assertAlmostEqual(power, 6.666666666666667)
        diagnostics = prepared.controller_allocator_diagnostics
        self.assertEqual(diagnostics["sensor_connected_node_ids"], {"1": [3, 4]})
        self.assertEqual(diagnostics["B_s"], [[0.15000000000000002]])
        self.assertAlmostEqual(diagnostics["average_inverse_C_s"][0], 0.075)

    def test_mimo_dynamic_rate_gain_is_static_direct_capacitance(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_static_bdyn"))
        for node_id, capacitance in ((1, 10.0), (2, 20.0)):
            node = NodeProperties.with_material(node_id, (node_id, 0, 0), material="copper")
            node.C_J_K = capacitance
            node.initial_temperature_K = 300.0
            node.is_heater = True
            node.is_sensor = True
            node.heater.heater_max_power_W = 20.0
            node.heater_control.mode = "mimo"
            node.controller_setpoint_K = 310.0
            node.controller_kp_coarse = 1.0
            model.add_node(node)
        matrices = {
            "node_ids": np.array([1, 2], dtype=int),
            "C": np.array([10.0, 20.0]),
            "L": np.array([[1.0, -1.0], [-1.0, 1.0]]),
            "G_rad": np.array([0.0, 0.0]),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(dt_s=1.0, input_mode="heater_inputs", mimo_lambda_u=0.0, use_ambient_radiation=False),
        )

        B_dyn = prepared._mimo_dynamic_gain_matrix(
            [1, 2],
            [1, 2],
            {1: 0, 2: 1},
        )

        np.testing.assert_allclose(B_dyn, np.diag([0.1, 0.05]))

    def test_mimo_controller_uses_measured_drift_and_capacitance_not_steady_gain(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_measured_drift"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 20.0
        node.heater_control.mode = "mimo"
        node.controller_setpoint_K = 310.0
        node.controller_kp_coarse = 1.0
        model.add_node(node)
        model.set_controller_gain(1, 1, 999.0)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(
                dt_s=1.0,
                input_mode="heater_inputs",
                mimo_lambda_u=0.0,
                drift_lpf_tau_s=0.0,
                use_ambient_radiation=False,
            ),
        )

        prepared.step_forward()
        preview = prepared.heater_power_by_node()

        self.assertAlmostEqual(prepared.controller_last_power_by_heater[1], 2.5)
        self.assertAlmostEqual(preview[1], 2.5)
        diagnostics = prepared.controller_allocator_diagnostics
        self.assertEqual(diagnostics["filtered_dTdt_hat_s"], [0.25])
        self.assertEqual(diagnostics["B_s"], [[0.1]])

    def test_mimo_zero_gains_feedforward_holds_against_model_cooling_load(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_feedforward_hold"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.is_heater = True
        node.is_sensor = True
        node.has_cryocooler = True
        node.sensor_connected_node_ids = [1]
        node.power_deposition_node_ids = [1]
        node.power_deposition_weights = [1.0]
        node.heater.heater_max_power_W = 20.0
        node.heater_control.mode = "mimo"
        node.controller_setpoint_K = 300.0
        node.controller_kp_coarse = 0.0
        node.controller_ki_coarse = 0.0
        node.controller_kd_coarse = 0.0
        model.add_node(node)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(
                dt_s=1.0,
                input_mode="heater_inputs",
                Kp_cooler=1.0,
                P_cooler_max=5.0,
                T_cooler_setpoint=290.0,
                use_ambient_radiation=False,
            ),
        )

        actuator_power = prepared.heater_actuator_power_by_node()[1]
        net_power = prepared.heater_power_by_node()[1]

        self.assertAlmostEqual(actuator_power, 5.0)
        self.assertAlmostEqual(net_power, 0.0)
        diagnostics = prepared.controller_allocator_diagnostics
        self.assertEqual(diagnostics["v_cmd_s"], [0.0])
        self.assertEqual(diagnostics["passive_dTdt_s"], [-0.5])
        self.assertAlmostEqual(diagnostics["feedforward_hold_power_W"][0], 5.0)
        prepared.step_forward()
        self.assertAlmostEqual(float(prepared.temperatures_K[0]), 300.0)

    def test_disabled_manual_heater_outputs_zero_power(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="disabled_manual_heater"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 20.0
        node.heater_control.mode = "manual"
        node.heater_control.manual.power = 12.0
        model.add_node(node)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(
                dt_s=1.0,
                input_mode="heater_inputs",
                enabled_heater_node_ids=(),
                enabled_sensor_node_ids=(1,),
            ),
        )

        self.assertEqual(prepared.heater_power_by_node(), {1: 0.0})
        prepared.step_forward()
        self.assertAlmostEqual(float(prepared.temperatures_K[0]), 300.0)

    def test_disabled_mimo_heater_is_excluded_from_controller(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="disabled_mimo_heater"))
        for node_id in (1, 2):
            node = NodeProperties.with_material(node_id, (node_id, 0, 0), material="copper")
            node.C_J_K = 10.0
            node.initial_temperature_K = 300.0
            node.is_heater = True
            node.is_sensor = True
            node.heater.heater_max_power_W = 20.0
            node.heater_control.mode = "mimo"
            node.controller_setpoint_K = 310.0
            node.controller_kp_coarse = 1.0
            model.add_node(node)
        model.set_controller_gain(1, 1, 1.0)
        model.set_controller_gain(2, 2, 1.0)
        matrices = {
            "node_ids": np.array([1, 2], dtype=int),
            "C": np.array([10.0, 10.0]),
            "L": np.zeros((2, 2)),
            "G_rad": np.array([0.0, 0.0]),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(
                dt_s=1.0,
                input_mode="heater_inputs",
                mimo_lambda_u=0.0,
                use_ambient_radiation=False,
                enabled_heater_node_ids=(1,),
                enabled_sensor_node_ids=(1, 2),
            ),
        )

        powers = prepared.heater_power_by_node()

        self.assertGreater(powers[1], 0.0)
        self.assertEqual(powers[2], 0.0)
        self.assertEqual(prepared.controller_last_power_by_heater, {})
        prepared.step_forward()
        self.assertEqual(set(prepared.controller_last_power_by_heater), {1})

    def test_disabled_mimo_sensor_does_not_drive_controller(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="disabled_mimo_sensor"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 20.0
        node.heater_control.mode = "mimo"
        node.controller_setpoint_K = 310.0
        node.controller_kp_coarse = 1.0
        model.add_node(node)
        model.set_controller_gain(1, 1, 1.0)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(
                dt_s=1.0,
                input_mode="heater_inputs",
                mimo_lambda_u=0.0,
                use_ambient_radiation=False,
                enabled_heater_node_ids=(1,),
                enabled_sensor_node_ids=(),
            ),
        )

        self.assertEqual(prepared.heater_power_by_node(), {1: 0.0})
        self.assertEqual(prepared.controller_last_power_by_heater, {})

    def test_mimo_active_check_uses_heater_role_ids(self) -> None:
        from graph_visualizer.simulation_model import _mimo_controller_is_active

        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_role_id_check"))
        sensor = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        sensor.is_sensor = True
        sensor.sensor_control_mode = "mimo"
        sensor.controller_setpoint_K = 310.0
        sensor.sensor_connected_node_ids = [1]
        sensor.assigned_heater_ids = [2]
        heater = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        heater.is_heater = True
        heater.heater_valid = True
        heater.heater_control.mode = "mimo"
        heater.assigned_sensor_id = 1
        heater.power_deposition_node_ids = [2]
        model.add_node(sensor)
        model.add_node(heater)

        active = _mimo_controller_is_active(
            model,
            np.array([2], dtype=int),
            SimulationParameters(input_mode="heater_inputs"),
        )

        self.assertTrue(active)

    def test_mimo_controller_integrator_updates_per_sensor(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_integrator"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 20.0
        node.heater_control.mode = "mimo"
        node.controller_setpoint_K = 310.0
        node.controller_kp_coarse = 0.0
        node.controller_ki_coarse = 1.0
        model.add_node(node)
        model.set_controller_gain(1, 1, 1.0)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }
        params = SimulationParameters(
            dt_s=1.0,
            input_mode="heater_inputs",
            mimo_lambda_u=0.0,
            use_ambient_radiation=False,
        )

        prepared = prepare_simulation(model, matrices, params)
        prepared.step_forward()

        self.assertAlmostEqual(prepared.controller_integrators[1], 10.0)
        self.assertAlmostEqual(prepared.heater_power_by_node()[1], 2.5)

    def test_mimo_fractional_integral_order_scales_rate_command(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_fractional_integrator"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 290.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 100.0
        node.heater_control.mode = "mimo"
        node.controller_setpoint_K = 300.0
        node.controller_ki_coarse = 1.0
        node.controller_lambda_order = 0.5
        model.add_node(node)
        model.set_controller_gain(1, 1, 1.0)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(
                dt_s=4.0,
                input_mode="heater_inputs",
                mimo_lambda_u=0.0,
                use_ambient_radiation=False,
            ),
        )

        self.assertAlmostEqual(prepared.heater_power_by_node()[1], 2.5)
        prepared.step_forward()

        self.assertAlmostEqual(prepared.controller_integrators[1], 20.0)
        self.assertEqual(prepared.controller_error_history[1], (10.0,))
        self.assertAlmostEqual(prepared.controller_last_power_by_heater[1], 2.5)

    def test_mimo_integral_can_go_negative_above_setpoint(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_signed_integral"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 310.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 200.0
        node.heater_control.mode = "mimo"
        node.controller_setpoint_K = 300.0
        node.controller_kp_coarse = 0.0
        node.controller_ki_coarse = 1.0
        model.add_node(node)
        model.set_controller_gain(1, 1, 1.0)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(
                dt_s=1.0,
                input_mode="heater_inputs",
                mimo_lambda_u=0.0,
                mimo_freeze_integral_when_saturated=False,
                use_ambient_radiation=False,
            ),
        )
        prepared.controller_integrators[1] = 100.0

        prepared.step_forward()

        self.assertAlmostEqual(prepared.controller_integrators[1], -10.0)
        self.assertAlmostEqual(prepared.controller_last_power_by_heater[1], 0.0, places=8)

    def test_mimo_allocator_respects_heater_bounds_during_solve(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_bounded_allocator"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 4.0
        node.heater_control.mode = "mimo"
        node.controller_setpoint_K = 320.0
        node.controller_kp_coarse = 1.0
        model.add_node(node)
        model.set_controller_gain(1, 1, 1.0)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(
                dt_s=1.0,
                input_mode="heater_inputs",
                mimo_lambda_u=0.0,
                mimo_v_cmd_abs_max_K_per_s=100.0,
                use_ambient_radiation=False,
            ),
        )

        prepared.step_forward()

        self.assertAlmostEqual(prepared.controller_last_power_by_heater[1], 4.0)
        self.assertTrue(prepared.controller_allocator_diagnostics["bounds_active"])

    def test_mimo_controller_derivative_damps_approach_to_setpoint(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_derivative"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 305.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 20.0
        node.heater_control.mode = "mimo"
        node.controller_setpoint_K = 310.0
        node.controller_kp_coarse = 1.0
        node.controller_ki_coarse = 0.0
        node.controller_kd_coarse = 1.0
        model.add_node(node)
        model.set_controller_gain(1, 1, 1.0)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(
                dt_s=1.0,
                input_mode="heater_inputs",
                mimo_lambda_u=0.0,
                use_ambient_radiation=False,
            ),
        )
        prepared.controller_error_history[1] = (10.0,)

        self.assertEqual(prepared.heater_power_by_node(), {1: 0.0})

    def test_mimo_controller_clears_fractional_memory_on_mode_change(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_mode_memory"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 200.0
        node.heater_control.mode = "mimo"
        node.controller_setpoint_K = 310.0
        node.controller_kp_hold = 0.0
        node.controller_ki_hold = 1.0
        node.controller_kd_hold = 1.0
        model.add_node(node)
        model.set_controller_gain(1, 1, 1.0)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(
                dt_s=1.0,
                input_mode="heater_inputs",
                mimo_hold_threshold_K=20.0,
                mimo_lambda_u=0.0,
                use_ambient_radiation=False,
            ),
        )
        prepared.controller_mode = "coarse"
        prepared.controller_error_history[1] = (2.0,)

        prepared.step_forward()

        self.assertEqual(prepared.controller_mode, "hold")
        self.assertEqual(prepared.controller_error_history[1], (10.0,))
        self.assertAlmostEqual(prepared.controller_integrators[1], 10.0)
        self.assertAlmostEqual(prepared.controller_last_power_by_heater[1], 2.5)

    def test_mimo_controller_penalizes_power_change_on_mode_change(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_mode_last_power"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 200.0
        node.heater_control.mode = "mimo"
        node.controller_setpoint_K = 300.0
        model.add_node(node)
        model.set_controller_gain(1, 1, 1.0)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(
                dt_s=1.0,
                input_mode="heater_inputs",
                mimo_hold_threshold_K=1.0,
                mimo_lambda_u=0.0,
                mimo_rho_du=1.0,
                use_ambient_radiation=False,
            ),
        )
        prepared.controller_mode = "coarse"
        prepared.controller_last_power_by_heater[1] = 4.0

        prepared.step_forward()

        self.assertEqual(prepared.controller_mode, "hold")
        self.assertGreater(prepared.controller_last_power_by_heater[1], 3.9)
        self.assertLess(prepared.controller_last_power_by_heater[1], 4.0)

    def test_disabled_mimo_controller_does_not_use_manual_fallback_power_for_sys_id(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_sys_id_baseline"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 20.0
        node.heater_control.mode = "mimo"
        node.heater_control.manual.power = 20.0
        node.controller_setpoint_K = 310.0
        node.controller_kp_coarse = 1.0
        model.add_node(node)
        model.set_controller_gain(1, 1, 1.0)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(
                dt_s=1.0,
                input_mode="heater_inputs",
                mimo_lambda_u=0.0,
                use_ambient_radiation=False,
            ),
        )

        self.assertAlmostEqual(prepared.heater_power_by_node()[1], 2.5)
        self.assertEqual(prepared.heater_actuator_power_by_node(disable_mimo_controller=True), {1: 0.0})

    def test_legacy_heat_sink_fields_are_ignored_and_cryocooler_round_trips(self) -> None:
        node = NodeProperties.from_dict(
            {
                "node_id": 4,
                "coord": [0, 0, 0],
                "has_heat_sink": True,
                "heat_sink": {"heat_sink_id": 44, "heat_sink_power_W": 7.5},
                "has_cryocooler": True,
            }
        )
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="cryocooler_round_trip"))
        model.add_node(node)

        restored = ThermalGraphModel.from_octree_graph_dict(model.to_octree_graph_dict())
        matrices = build_matrices(restored)

        self.assertFalse(hasattr(restored.nodes[4], "has_heat_sink"))
        self.assertNotIn("has_heat_sink", matrices)
        self.assertTrue(restored.nodes[4].has_cryocooler)
        self.assertTrue(bool(matrices["has_cryocooler"][0]))

    def test_heater_node_metadata_defines_role_without_tags(self) -> None:
        node = NodeProperties.from_dict(
            {
                "node_id": 12,
                "cell_id": "heater_12",
                "coord": [12, 0, 0],
                "center_mm": [5.0, 0.0, 0.0],
                "size_mm": [2.0, 4.0, 4.0],
                "level": -1,
                "node_type": "heater",
                "component_name": "heater_strip",
                "material_name": "Copper",
                "mass_kg": 0.01,
                "C_J_K": 3.85,
                "source_components": ["heater_strip_1"],
                "source_node_ids": [2, 8],
                "source_cell_ids": ["cell_heater_left", "cell_heater_right"],
                "role_source_components": ["heater_strip_1", "heater_strip_2"],
            }
        )

        self.assertEqual(node.node_id, 12)
        self.assertEqual(node.cell_id, "heater_12")
        self.assertEqual(node.node_type, "heater")
        self.assertEqual(node.component_name, "heater_strip")
        self.assertEqual(node.source_node_ids, [2, 8])
        self.assertEqual(node.source_cell_ids, ["cell_heater_left", "cell_heater_right"])
        self.assertEqual(node.role_source_components, ["heater_strip_1", "heater_strip_2"])
        self.assertTrue(node.is_heater)
        self.assertFalse(node.is_sensor)

        saved = node.to_octree_node_dict()

        self.assertEqual(saved["node_type"], "heater")
        self.assertTrue(saved["is_heater"])
        self.assertFalse(saved["is_sensor"])
        self.assertEqual(saved["source_node_ids"], [2, 8])
        self.assertEqual(saved["source_cell_ids"], ["cell_heater_left", "cell_heater_right"])
        self.assertEqual(saved["role_source_components"], ["heater_strip_1", "heater_strip_2"])
        self.assertNotIn("heater", saved["tags"])
        self.assertNotIn("sensor", saved["tags"])

    def test_octree_node_load_accepts_is_heater_is_sensor_schema(self) -> None:
        heater = NodeProperties.from_dict(
            {
                "node_id": 21,
                "coord": [21, 0, 0],
                "is_heater": True,
                "is_sensor": False,
            }
        )
        sensor = NodeProperties.from_dict(
            {
                "node_id": 22,
                "coord": [22, 0, 0],
                "is_heater": False,
                "is_sensor": True,
            }
        )

        self.assertTrue(heater.is_heater)
        self.assertFalse(heater.is_sensor)
        self.assertFalse(sensor.is_heater)
        self.assertTrue(sensor.is_sensor)

    def test_octree_node_load_migrates_legacy_physical_device_role(self) -> None:
        node = NodeProperties.from_dict(
            {
                "node_id": 23,
                "coord": [23, 0, 0],
                "node_type": "physical_heater_sensor",
                "component_name": "legacy_role",
                "physical_device": {
                    "kind": "heater_sensor",
                    "source_components": ["legacy_heater", "legacy_sensor"],
                    "bounds_mm": {
                        "min": [1.0, 2.0, 3.0],
                        "max": [4.0, 5.0, 6.0],
                    },
                },
            }
        )

        self.assertTrue(node.is_heater)
        self.assertTrue(node.is_sensor)
        self.assertEqual(node.source_components, ["legacy_heater", "legacy_sensor"])
        self.assertEqual(node.role_source_components, ["legacy_heater", "legacy_sensor"])
        self.assertEqual(node.source_bounds_mm["min"], [1.0, 2.0, 3.0])
        self.assertEqual(node.source_bounds_mm["max"], [4.0, 5.0, 6.0])

    def test_octree_load_adjusts_duplicate_loaded_coords(self) -> None:
        model = ThermalGraphModel.from_octree_graph_dict(
            {
                "metadata": {"graph_name": "duplicate_coords"},
                "graph_nodes": [
                    {
                        "node_id": 1,
                        "coord": [1, 0, 0],
                        "component_name": "body",
                        "material_name": "Copper",
                        "mass_kg": 0.01,
                        "C_J_K": 1.0,
                    },
                    {
                        "node_id": 2,
                        "coord": [1, 0, 0],
                        "component_name": "heater",
                        "material_name": "Copper",
                        "mass_kg": 0.02,
                        "C_J_K": 2.0,
                        "is_heater": True,
                    },
                ],
                "graph_edges": [
                    {"edge_id": "edge_0", "node_i": 1, "node_j": 2, "G_W_K": 0.5}
                ],
            }
        )

        self.assertEqual(set(model.nodes), {1, 2})
        self.assertNotEqual(model.nodes[1].coord, model.nodes[2].coord)
        self.assertIn("Adjusted duplicate loaded coordinate", " ".join(model.octree_graph_data["warnings"]))

    def test_octree_load_uses_incremental_coord_index(self) -> None:
        payload = {
            "metadata": {"graph_name": "large_coord_index"},
            "graph_nodes": [
                {
                    "node_id": node_id,
                    "coord": [node_id, 0, 0],
                    "component_name": "body",
                    "material_name": "Copper",
                    "mass_kg": 0.01,
                    "C_J_K": 1.0,
                }
                for node_id in range(1, 50)
            ],
            "graph_edges": [],
        }

        with patch.object(ThermalGraphModel, "coord_index", side_effect=AssertionError("coord_index rebuilt")):
            model = ThermalGraphModel.from_octree_graph_dict(payload)

        self.assertEqual(len(model.nodes), 49)

    def test_octree_node_load_sanitizes_invalid_coord(self) -> None:
        model = ThermalGraphModel.from_octree_graph_dict(
            {
                "metadata": {"graph_name": "bad_coord"},
                "graph_nodes": [
                    {
                        "node_id": 8,
                        "coord": [0, float("nan"), 0],
                        "component_name": "bad_coord",
                        "material_name": "Copper",
                    }
                ],
                "graph_edges": [],
            }
        )

        node = model.nodes[8]
        self.assertEqual(node.coord, (8, 0, 0))
        self.assertEqual(model.coord_index(), {(8, 0, 0): 8})
        self.assertIn("non-finite coord", " ".join(node.warnings))

    def test_depth_focus_classifies_nodes_by_normalized_axis_layer(self) -> None:
        try:
            from graph_visualizer.pyvista_widget import GraphPyVistaWidget
        except ModuleNotFoundError as exc:
            self.skipTest(f"PyVista widget dependency unavailable: {exc}")
        widget = object.__new__(GraphPyVistaWidget)
        widget.depth_focus_enabled = True
        widget.depth_focus_axis = "x"
        widget.depth_focus_fraction = 0.5
        widget.depth_focus_width = 0.2

        bounds = (0.0, 10.0, 0.0, 10.0, 0.0, 100.0)

        self.assertTrue(widget._node_in_depth_focus(np.array([5.0, 0.0, 0.0]), bounds))
        self.assertFalse(widget._node_in_depth_focus(np.array([1.0, 0.0, 50.0]), bounds))

        widget.depth_focus_axis = "z"
        self.assertTrue(widget._node_in_depth_focus(np.array([1.0, 0.0, 50.0]), bounds))

    def test_depth_focus_dims_unfocused_opacity_and_color(self) -> None:
        try:
            from graph_visualizer.pyvista_widget import GraphPyVistaWidget
        except ModuleNotFoundError as exc:
            self.skipTest(f"PyVista widget dependency unavailable: {exc}")
        widget = object.__new__(GraphPyVistaWidget)
        widget.shader_mode_enabled = False
        widget.cell_opacity = 0.5
        widget.depth_focus_enabled = True
        widget.depth_focus_fraction = 0.5
        widget.depth_focus_width = 0.2
        widget.dark_mode = False

        self.assertAlmostEqual(widget._cell_opacity(False, False, False), 0.11)
        self.assertAlmostEqual(widget._cell_opacity(False, True, False), 0.78)
        self.assertEqual(widget._depth_adjust_rgb([100, 150, 200], False), [199, 213, 227])

    def test_depth_focus_setter_updates_axis_and_width(self) -> None:
        try:
            from graph_visualizer.pyvista_widget import GraphPyVistaWidget
        except ModuleNotFoundError as exc:
            self.skipTest(f"PyVista widget dependency unavailable: {exc}")
        widget = object.__new__(GraphPyVistaWidget)
        widget.depth_focus_enabled = False
        widget.depth_focus_axis = "z"
        widget.depth_focus_fraction = 0.5
        widget.depth_focus_width = 0.12
        widget._apply_visual_controls_to_scene = lambda: None
        widget.safe_render = lambda: True

        widget.set_depth_focus(True, 0.25, axis="Y", width=0.4, render=False)

        self.assertTrue(widget.depth_focus_enabled)
        self.assertEqual(widget.depth_focus_axis, "y")
        self.assertAlmostEqual(widget.depth_focus_fraction, 0.25)
        self.assertAlmostEqual(widget.depth_focus_width, 0.4)

    def test_role_interface_overlays_batch_outlines_by_style(self) -> None:
        import sys
        import types

        matplotlib_stubbed = "matplotlib" not in sys.modules
        if matplotlib_stubbed:
            sys.modules["matplotlib"] = types.SimpleNamespace(colormaps={})
        try:
            from graph_visualizer.pyvista_widget import GraphPyVistaWidget
        finally:
            if matplotlib_stubbed:
                sys.modules.pop("matplotlib", None)

        class FakePV:
            @staticmethod
            def PolyData(points, faces=None, lines=None):
                return {"points": points, "faces": faces, "lines": lines}

        class FakePlotter:
            def __init__(self):
                self.calls = []

            def add_mesh(self, mesh, **kwargs):
                self.calls.append((mesh, kwargs))
                return object()

        model = ThermalGraphModel()
        for node_id, coord in ((1, (0, 0, 0)), (2, (1, 0, 0))):
            node = NodeProperties.with_material(node_id, coord, material="copper")
            node.center_mm = tuple(float(value) for value in coord)
            node.size_mm = (1.0, 1.0, 1.0)
            model.add_node(node)
        heater = NodeProperties.with_material(10, (2, 0, 0), material="copper")
        heater.center_mm = (2.0, 0.0, 0.0)
        heater.size_mm = (1.0, 1.0, 1.0)
        heater.is_heater = True
        heater.power_deposition_node_ids = [1, 2]
        heater.assigned_sensor_id = 20
        model.add_node(heater)
        sensor = NodeProperties.with_material(20, (3, 0, 0), material="copper")
        sensor.center_mm = (3.0, 0.0, 0.0)
        sensor.size_mm = (1.0, 1.0, 1.0)
        sensor.is_sensor = True
        sensor.readout_node_ids = [2]
        model.add_node(sensor)

        widget = object.__new__(GraphPyVistaWidget)
        widget.show_heaters = True
        widget.show_sensors = True
        widget.pv = FakePV()
        widget.plotter = FakePlotter()
        widget._role_overlay_actors = []

        widget._draw_role_interface_overlays(model, set(model.nodes))

        self.assertEqual(len(widget.plotter.calls), 3)
        wireframe_calls = [kwargs for _mesh, kwargs in widget.plotter.calls if kwargs.get("style") == "wireframe"]
        self.assertEqual(len(wireframe_calls), 2)
        line_calls = [mesh for mesh, kwargs in widget.plotter.calls if kwargs.get("line_width") == 4]
        self.assertEqual(len(line_calls), 1)

    def test_editor_view_controls_do_not_redraw_voxel_geometry(self) -> None:
        try:
            from graph_visualizer.app import GraphVisualizerApp
        except ModuleNotFoundError as exc:
            self.skipTest(f"Graph visualizer dependency unavailable: {exc}")

        class Viewer:
            def __init__(self) -> None:
                self.render_count = 0

            def safe_render(self) -> bool:
                self.render_count += 1
                return True

        app = object.__new__(GraphVisualizerApp)
        app.viewer = Viewer()
        sync_calls = []
        app._sync_view_controls_to_viewer = lambda: sync_calls.append(True)
        app._refresh_all = lambda reset_camera=False: (_ for _ in ()).throw(AssertionError("redrew voxels"))

        app._handle_view_control_changed()

        self.assertEqual(sync_calls, [True])
        self.assertEqual(app.viewer.render_count, 1)

    def test_simulation_view_controls_do_not_redraw_voxel_geometry(self) -> None:
        try:
            from graph_visualizer.heat_transfer_simulation_tab import HeatTransferSimulationTab
        except ModuleNotFoundError as exc:
            self.skipTest(f"Graph visualizer dependency unavailable: {exc}")

        class Viewer:
            def __init__(self) -> None:
                self.render_count = 0

            def safe_render(self) -> bool:
                self.render_count += 1
                return True

        tab = object.__new__(HeatTransferSimulationTab)
        tab.viewer = Viewer()
        sync_calls = []
        tab._sync_view_controls_to_viewer = lambda: sync_calls.append(True)
        tab._draw_current = lambda reset_camera=False: (_ for _ in ()).throw(AssertionError("redrew voxels"))

        tab._handle_visual_control_changed()

        self.assertEqual(sync_calls, [True])
        self.assertEqual(tab.viewer.render_count, 1)

    def test_simulation_readout_groups_heaters_under_sensors(self) -> None:
        import sys
        import types

        heat_tab_module_name = "graph_visualizer.heat_transfer_simulation_tab"
        pyvista_module_name = "graph_visualizer.pyvista_widget"
        qtpy_module_name = "qtpy"
        previous_heat_tab = sys.modules.pop(heat_tab_module_name, None)
        previous_pyvista = sys.modules.get(pyvista_module_name)
        previous_qtpy = sys.modules.get(qtpy_module_name)
        pyvista_stub = types.ModuleType(pyvista_module_name)
        pyvista_stub.GraphPyVistaWidget = object
        sys.modules[pyvista_module_name] = pyvista_stub
        qtpy_stub = types.ModuleType(qtpy_module_name)
        qtpy_stub.QtGui = types.SimpleNamespace()
        sys.modules[qtpy_module_name] = qtpy_stub
        try:
            from graph_visualizer.heat_transfer_simulation_tab import HeatTransferSimulationTab
        finally:
            sys.modules.pop(heat_tab_module_name, None)
            if previous_heat_tab is not None:
                sys.modules[heat_tab_module_name] = previous_heat_tab
            if previous_pyvista is not None:
                sys.modules[pyvista_module_name] = previous_pyvista
            else:
                sys.modules.pop(pyvista_module_name, None)
            if previous_qtpy is not None:
                sys.modules[qtpy_module_name] = previous_qtpy
            else:
                sys.modules.pop(qtpy_module_name, None)

        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="readout_groups"))
        sensor = NodeProperties.with_material(10, (0, 0, 0), material="copper")
        sensor.is_sensor = True
        sensor.assigned_heater_ids = [20]
        sensor.sensor_control_mode = "manual"
        sensor.sensor_manual_power_W = 100.0
        heater = NodeProperties.with_material(20, (1, 0, 0), material="copper")
        heater.is_heater = True
        heater.assigned_sensor_id = 10
        heater.heater.heater_max_power_W = 120.0
        second_heater = NodeProperties.with_material(30, (2, 0, 0), material="copper")
        second_heater.is_heater = True
        second_heater.assigned_sensor_id = 10
        model.add_node(sensor)
        model.add_node(heater)
        model.add_node(second_heater)
        tab = object.__new__(HeatTransferSimulationTab)
        tab.model = model
        tab.enabled_sensor_node_ids = {10}
        tab.enabled_heater_node_ids = {20, 30}
        tab._readout_editor_syncing = False
        tab._readout_editor_sensor_id = 10
        tab._readout_editor_node_id = 20
        tab._simulation_reinitialize_pending = False

        class Spin:
            def __init__(self, value: float) -> None:
                self._value = float(value)

            def value(self) -> float:
                return self._value

        class Prepared:
            def __init__(self) -> None:
                self.marked = False
                self.reset = False

            def mark_controller_stale(self) -> None:
                self.marked = True

            def reset_controller_integrators(self) -> None:
                self.reset = True

        tab.readout_editor_inputs = {
            "sensor_manual_power_W": Spin(77.0),
            "heater_id": Spin(44.0),
            "heater_min_power_W": Spin(3.0),
            "heater_max_power_W": Spin(55.0),
            "heater_efficiency": Spin(0.8),
        }
        tab.prepared = Prepared()
        tab._refresh_stats = lambda: None
        tab._refresh_sensor_readouts = lambda: None
        tab._show_readout_heater_editor = lambda heater_id: None
        tab._status = lambda message, error=False: None

        sensors = tab._heating_sensor_nodes()
        heater_ids = tab._associated_heater_ids_for_sensor(10)
        manual_power = tab._heater_readout_power_for_sensor_heater(10, 20, {})
        tab._apply_readout_heater_editor_change("sensor_manual_power_W")
        tab._apply_readout_heater_editor_change("heater_id")
        tab._apply_readout_heater_editor_change("heater_min_power_W")
        tab._apply_readout_heater_editor_change("heater_max_power_W")
        tab._apply_readout_heater_editor_change("heater_efficiency")

        self.assertEqual([node.node_id for node in sensors], [10])
        self.assertEqual(heater_ids, [20, 30])
        self.assertAlmostEqual(manual_power, 100.0)
        self.assertAlmostEqual(model.nodes[10].sensor_manual_power_W, 100.0)
        self.assertAlmostEqual(model.nodes[20].sensor_manual_power_W, 77.0)
        self.assertEqual(model.nodes[20].heater.heater_id, 44)
        self.assertAlmostEqual(model.nodes[20].heater.heater_min_power_W, 3.0)
        self.assertAlmostEqual(model.nodes[20].heater.heater_max_power_W, 55.0)
        self.assertAlmostEqual(model.nodes[20].heater.heater_efficiency, 0.8)
        self.assertTrue(tab.prepared.marked)
        self.assertTrue(tab.prepared.reset)

    def test_simulation_parameter_change_defers_reinitialize_without_redraw(self) -> None:
        try:
            from graph_visualizer.heat_transfer_simulation_tab import HeatTransferSimulationTab
        except ModuleNotFoundError as exc:
            self.skipTest(f"Graph visualizer dependency unavailable: {exc}")

        class Timer:
            def __init__(self) -> None:
                self.active = True

            def isActive(self) -> bool:
                return self.active

            def stop(self) -> None:
                self.active = False

        class Prepared:
            def __init__(self) -> None:
                self.params = None
                self.marked = False
                self.reset = False

            def mark_controller_stale(self) -> None:
                self.marked = True

            def reset_controller_integrators(self) -> None:
                self.reset = True

        tab = object.__new__(HeatTransferSimulationTab)
        tab.sys_id_state = None
        tab.params = SimulationParameters(dt_s=1.0)
        tab.prepared = Prepared()
        tab.timer = Timer()
        tab._simulation_reinitialize_pending = False
        tab._read_params = lambda: SimulationParameters(dt_s=2.0)
        tab._save_params_to_folder = lambda: None
        tab.initialize_simulation = lambda: (_ for _ in ()).throw(AssertionError("initialized immediately"))
        tab._draw_current = lambda reset_camera=False: (_ for _ in ()).throw(AssertionError("redrew voxels"))
        tab._update_colors = lambda: (_ for _ in ()).throw(AssertionError("updated colors"))
        statuses = []
        tab._status = lambda message, error=False: statuses.append(message)
        tab.pause = lambda: tab.timer.stop()

        tab._handle_parameter_change()

        self.assertTrue(tab._simulation_reinitialize_pending)
        self.assertFalse(tab.timer.isActive())
        self.assertEqual(tab.prepared.params.dt_s, 2.0)
        self.assertTrue(any("Reinitialize" in message for message in statuses))

    def test_simulation_display_parameter_change_updates_colors_only(self) -> None:
        try:
            from graph_visualizer.heat_transfer_simulation_tab import HeatTransferSimulationTab
        except ModuleNotFoundError as exc:
            self.skipTest(f"Graph visualizer dependency unavailable: {exc}")

        class Timer:
            def isActive(self) -> bool:
                return False

        class Prepared:
            def __init__(self) -> None:
                self.params = None

            def mark_controller_stale(self) -> None:
                raise AssertionError("controller marked stale")

            def reset_controller_integrators(self) -> None:
                raise AssertionError("controller reset")

        tab = object.__new__(HeatTransferSimulationTab)
        tab.sys_id_state = None
        tab.params = SimulationParameters(color_min_K=0.0)
        tab.prepared = Prepared()
        tab.timer = Timer()
        tab._simulation_reinitialize_pending = False
        tab._read_params = lambda: SimulationParameters(color_min_K=10.0)
        tab._save_params_to_folder = lambda: None
        tab.initialize_simulation = lambda: (_ for _ in ()).throw(AssertionError("initialized immediately"))
        tab._draw_current = lambda reset_camera=False: (_ for _ in ()).throw(AssertionError("redrew voxels"))
        color_updates = []
        tab._update_colors = lambda: color_updates.append(True)
        tab._refresh_stats = lambda: (_ for _ in ()).throw(AssertionError("refreshed stats"))
        tab._refresh_sensor_readouts = lambda: (_ for _ in ()).throw(AssertionError("refreshed readouts"))

        tab._handle_parameter_change()

        self.assertFalse(tab._simulation_reinitialize_pending)
        self.assertEqual(color_updates, [True])
        self.assertEqual(tab.prepared.params.color_min_K, 10.0)

    def test_simulation_run_reuses_loaded_octree_matrices(self) -> None:
        import sys
        import types

        heat_tab_module_name = "graph_visualizer.heat_transfer_simulation_tab"
        pyvista_module_name = "graph_visualizer.pyvista_widget"
        qtpy_module_name = "qtpy"
        previous_heat_tab = sys.modules.pop(heat_tab_module_name, None)
        previous_pyvista = sys.modules.get(pyvista_module_name)
        previous_qtpy = sys.modules.get(qtpy_module_name)
        pyvista_stub = types.ModuleType(pyvista_module_name)
        pyvista_stub.GraphPyVistaWidget = object
        sys.modules[pyvista_module_name] = pyvista_stub
        qtpy_stub = types.ModuleType(qtpy_module_name)
        qtpy_stub.QtGui = types.SimpleNamespace()
        sys.modules[qtpy_module_name] = qtpy_stub
        try:
            from graph_visualizer import heat_transfer_simulation_tab as heat_tab_module
            HeatTransferSimulationTab = heat_tab_module.HeatTransferSimulationTab
        finally:
            sys.modules.pop(heat_tab_module_name, None)
            if previous_heat_tab is not None:
                sys.modules[heat_tab_module_name] = previous_heat_tab
            if previous_pyvista is not None:
                sys.modules[pyvista_module_name] = previous_pyvista
            else:
                sys.modules.pop(pyvista_module_name, None)
            if previous_qtpy is not None:
                sys.modules[qtpy_module_name] = previous_qtpy
            else:
                sys.modules.pop(qtpy_module_name, None)

        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="loaded_octree_runtime"))
        left = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        right = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        for node in (left, right):
            node.center_mm = tuple(float(value) for value in node.coord)
            node.size_mm = (1.0, 1.0, 1.0)
            node.C_J_K = 20.0
            node.G_rad_W_K = 1.0e-8
            model.add_node(node)
        model.octree_graph_data = {"graph_edges": []}
        L = csr_matrix(np.array([[0.25, -0.25], [-0.25, 0.25]], dtype=float))

        tab = object.__new__(HeatTransferSimulationTab)
        tab.model = model
        tab.matrices = {
            "node_ids": np.array([1, 2], dtype=int),
            "coords": np.array([[0, 0, 0], [1, 0, 0]], dtype=int),
            "C": np.array([1.0, 1.0], dtype=float),
            "Grad": np.array([0.0, 0.0], dtype=float),
            "G_rad": np.array([0.0, 0.0], dtype=float),
            "initial_temperature_K": np.array([293.15, 293.15], dtype=float),
            "L": L,
        }

        with patch.object(heat_tab_module, "refresh_geometry_edges") as refresh_edges, patch.object(
            heat_tab_module, "refresh_radiation_from_exposed_faces"
        ) as refresh_radiation:
            tab._refresh_matrices_for_run()

        refresh_edges.assert_not_called()
        refresh_radiation.assert_not_called()
        self.assertIs(tab.matrices["L"], L)
        self.assertTrue(np.array_equal(tab.matrices["node_ids"], np.array([1, 2], dtype=int)))
        self.assertTrue(np.allclose(tab.matrices["C"], np.array([20.0, 20.0], dtype=float)))
        self.assertTrue(np.allclose(tab.matrices["G_rad"], np.array([1.0e-8, 1.0e-8], dtype=float)))

    def test_simulation_initialize_updates_existing_view_without_redraw(self) -> None:
        import sys
        import types

        heat_tab_module_name = "graph_visualizer.heat_transfer_simulation_tab"
        pyvista_module_name = "graph_visualizer.pyvista_widget"
        qtpy_module_name = "qtpy"
        previous_heat_tab = sys.modules.pop(heat_tab_module_name, None)
        previous_pyvista = sys.modules.get(pyvista_module_name)
        previous_qtpy = sys.modules.get(qtpy_module_name)
        pyvista_stub = types.ModuleType(pyvista_module_name)
        pyvista_stub.GraphPyVistaWidget = object
        sys.modules[pyvista_module_name] = pyvista_stub
        qtpy_stub = types.ModuleType(qtpy_module_name)
        qtpy_stub.QtGui = types.SimpleNamespace()
        sys.modules[qtpy_module_name] = qtpy_stub
        try:
            from graph_visualizer import heat_transfer_simulation_tab as heat_tab_module
            HeatTransferSimulationTab = heat_tab_module.HeatTransferSimulationTab
        finally:
            sys.modules.pop(heat_tab_module_name, None)
            if previous_heat_tab is not None:
                sys.modules[heat_tab_module_name] = previous_heat_tab
            if previous_pyvista is not None:
                sys.modules[pyvista_module_name] = previous_pyvista
            else:
                sys.modules.pop(pyvista_module_name, None)
            if previous_qtpy is not None:
                sys.modules[qtpy_module_name] = previous_qtpy
            else:
                sys.modules.pop(qtpy_module_name, None)

        class Viewer:
            def __init__(self) -> None:
                self.calls = 0

            def update_node_scalars(self, values, scalar_clim=None) -> bool:
                self.calls += 1
                self.values = values
                self.scalar_clim = scalar_clim
                return True

        model = ThermalGraphModel()
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.initial_temperature_K = 300.0
        model.add_node(node)
        tab = object.__new__(HeatTransferSimulationTab)
        tab.model = model
        tab.params = SimulationParameters(autoscale_temperature=True)
        tab.temperature_by_node = {1: 301.0}
        tab.viewer = Viewer()
        tab._draw_current = lambda reset_camera=False: (_ for _ in ()).throw(AssertionError("redrew voxels"))

        tab._refresh_initialized_view()

        self.assertEqual(tab.viewer.calls, 1)
        self.assertEqual(tab.viewer.values, {1: 301.0})

    def test_playback_speed_change_does_not_stop_active_simulation_worker(self) -> None:
        try:
            from graph_visualizer.heat_transfer_simulation_tab import HeatTransferSimulationTab
        except ModuleNotFoundError as exc:
            self.skipTest(f"Graph visualizer dependency unavailable: {exc}")

        class Spin:
            def __init__(self, value: float) -> None:
                self._value = value

            def value(self) -> float:
                return self._value

        class Timer:
            def __init__(self, active: bool = True) -> None:
                self.active = active
                self.started_with: int | None = None

            def isActive(self) -> bool:
                return self.active

            def start(self, interval: int) -> None:
                self.started_with = int(interval)

            def stop(self) -> None:
                self.active = False

        class Future:
            def done(self) -> bool:
                return False

            def cancel(self) -> bool:
                raise AssertionError("worker cancelled")

        class Prepared:
            def __init__(self) -> None:
                self.params = None

        tab = object.__new__(HeatTransferSimulationTab)
        tab.sys_id_state = None
        tab.params = SimulationParameters(playback_speed=1.0)
        tab.inputs = {"playback_speed": Spin(20.0)}
        tab.prepared = Prepared()
        tab.timer = Timer(active=True)
        tab.simulation_future = Future()
        tab.simulation_cancel_event = None
        tab._read_params = lambda: (_ for _ in ()).throw(AssertionError("read all params"))
        tab._save_params_to_folder = lambda: (_ for _ in ()).throw(AssertionError("saved immediately"))
        tab._refresh_stats = lambda: (_ for _ in ()).throw(AssertionError("refreshed stats"))
        tab._refresh_sensor_readouts = lambda: (_ for _ in ()).throw(AssertionError("refreshed readouts"))
        tab._status = lambda message, error=False: (_ for _ in ()).throw(AssertionError(message))
        saves = []
        tab._schedule_parameter_save = lambda: saves.append(True)

        tab._handle_parameter_change("playback_speed")

        self.assertTrue(tab.timer.isActive())
        self.assertEqual(tab.timer.started_with, 10)
        self.assertAlmostEqual(tab.params.playback_speed, 20.0)
        self.assertIs(tab.prepared.params, tab.params)
        self.assertEqual(saves, [True])

    def test_controller_parameter_change_applies_without_reinitialize_when_paused(self) -> None:
        try:
            from graph_visualizer.heat_transfer_simulation_tab import HeatTransferSimulationTab
        except ModuleNotFoundError as exc:
            self.skipTest(f"Graph visualizer dependency unavailable: {exc}")

        class Spin:
            def __init__(self, value: float) -> None:
                self._value = value

            def value(self) -> float:
                return self._value

        class Prepared:
            def __init__(self) -> None:
                self.params = None
                self.marked = False
                self.reset = False

            def mark_controller_stale(self) -> None:
                self.marked = True

            def reset_controller_integrators(self) -> None:
                self.reset = True

        tab = object.__new__(HeatTransferSimulationTab)
        tab.sys_id_state = None
        tab.params = SimulationParameters(mimo_lambda_u=1.0e-3)
        tab.inputs = {"mimo_lambda_u": Spin(0.25)}
        tab.prepared = Prepared()
        tab.simulation_future = None
        tab._simulation_reinitialize_pending = False
        tab._read_params = lambda: (_ for _ in ()).throw(AssertionError("read all params"))
        saves = []
        tab._save_params_to_folder = lambda: saves.append(True)
        tab._refresh_stats = lambda: None
        tab._refresh_sensor_readouts = lambda: None

        tab._handle_parameter_change("mimo_lambda_u")

        self.assertFalse(tab._simulation_reinitialize_pending)
        self.assertAlmostEqual(tab.params.mimo_lambda_u, 0.25)
        self.assertIs(tab.prepared.params, tab.params)
        self.assertTrue(tab.prepared.marked)
        self.assertTrue(tab.prepared.reset)
        self.assertEqual(saves, [True])

    def test_controller_parameter_change_pauses_active_worker_and_defers_apply(self) -> None:
        try:
            from graph_visualizer.heat_transfer_simulation_tab import HeatTransferSimulationTab
        except ModuleNotFoundError as exc:
            self.skipTest(f"Graph visualizer dependency unavailable: {exc}")

        class Spin:
            def __init__(self, value: float) -> None:
                self._value = value

            def value(self) -> float:
                return self._value

        class Timer:
            def __init__(self) -> None:
                self.active = True

            def isActive(self) -> bool:
                return self.active

            def stop(self) -> None:
                self.active = False

        class Event:
            def __init__(self) -> None:
                self.set_called = False

            def set(self) -> None:
                self.set_called = True

        class Future:
            def __init__(self) -> None:
                self.cancelled = False

            def done(self) -> bool:
                return False

            def cancel(self) -> bool:
                self.cancelled = True
                return True

        class Prepared:
            def __init__(self) -> None:
                self.params = None
                self.marked = False
                self.reset = False

            def mark_controller_stale(self) -> None:
                self.marked = True

            def reset_controller_integrators(self) -> None:
                self.reset = True

        tab = object.__new__(HeatTransferSimulationTab)
        tab.sys_id_state = None
        tab.params = SimulationParameters(mimo_lambda_u=1.0e-3)
        tab.inputs = {"mimo_lambda_u": Spin(0.5)}
        tab.prepared = Prepared()
        tab.timer = Timer()
        tab.simulation_future = Future()
        tab.simulation_cancel_event = Event()
        tab._simulation_reinitialize_pending = False
        tab._read_params = lambda: (_ for _ in ()).throw(AssertionError("read all params"))
        scheduled_saves = []
        final_saves = []
        statuses = []
        tab._schedule_parameter_save = lambda: scheduled_saves.append(True)
        tab._save_params_to_folder = lambda: final_saves.append(True)
        tab._refresh_stats = lambda: None
        tab._refresh_sensor_readouts = lambda: None
        tab._status = lambda message, error=False: statuses.append(message)

        tab._handle_parameter_change("mimo_lambda_u")

        self.assertFalse(tab.timer.isActive())
        self.assertTrue(tab.simulation_cancel_event.set_called)
        self.assertTrue(tab.simulation_future.cancelled)
        self.assertAlmostEqual(tab.params.mimo_lambda_u, 0.5)
        self.assertIsNone(tab.prepared.params)
        self.assertFalse(tab.prepared.marked)
        self.assertEqual(scheduled_saves, [True])

        self.assertTrue(tab._apply_pending_runtime_changes())

        self.assertFalse(tab._simulation_reinitialize_pending)
        self.assertIs(tab.prepared.params, tab.params)
        self.assertTrue(tab.prepared.marked)
        self.assertTrue(tab.prepared.reset)
        self.assertEqual(final_saves, [True])
        self.assertTrue(any("Simulation paused" in message for message in statuses))

    def test_editor_controller_refresh_pauses_active_worker_and_defers_apply(self) -> None:
        try:
            from graph_visualizer.heat_transfer_simulation_tab import HeatTransferSimulationTab
        except ModuleNotFoundError as exc:
            self.skipTest(f"Graph visualizer dependency unavailable: {exc}")

        class Timer:
            def __init__(self) -> None:
                self.active = True

            def isActive(self) -> bool:
                return self.active

            def stop(self) -> None:
                self.active = False

        class Event:
            def __init__(self) -> None:
                self.set_called = False

            def set(self) -> None:
                self.set_called = True

        class Future:
            def __init__(self) -> None:
                self.cancelled = False

            def done(self) -> bool:
                return False

            def cancel(self) -> bool:
                self.cancelled = True
                return True

        class Prepared:
            def __init__(self) -> None:
                self.marked = False
                self.reset = False

            def mark_controller_stale(self) -> None:
                self.marked = True

            def reset_controller_integrators(self) -> None:
                self.reset = True

        tab = object.__new__(HeatTransferSimulationTab)
        model = ThermalGraphModel()
        folder = Path("graphs") / "controller-test"
        tab.sys_id_state = None
        tab.model = model
        tab.folder = None
        tab.prepared = Prepared()
        tab.timer = Timer()
        tab.simulation_future = Future()
        tab.simulation_cancel_event = Event()
        tab._simulation_reinitialize_pending = False
        tab._sync_enabled_io_table = lambda: None
        tab._refresh_sensor_readouts = lambda: None
        statuses = []
        tab._status = lambda message, error=False: statuses.append(message)

        tab.refresh_controller_settings_from_editor(model, folder)

        self.assertFalse(tab.timer.isActive())
        self.assertTrue(tab.simulation_cancel_event.set_called)
        self.assertTrue(tab.simulation_future.cancelled)
        self.assertFalse(tab.prepared.marked)
        self.assertFalse(tab.prepared.reset)

        self.assertTrue(tab._apply_pending_runtime_changes())

        self.assertIs(tab.folder, folder)
        self.assertFalse(tab._simulation_reinitialize_pending)
        self.assertTrue(tab.prepared.marked)
        self.assertTrue(tab.prepared.reset)
        self.assertTrue(any("Simulation paused" in message for message in statuses))

    def test_display_parameter_change_does_not_stop_active_simulation_worker(self) -> None:
        try:
            from graph_visualizer.heat_transfer_simulation_tab import HeatTransferSimulationTab
        except ModuleNotFoundError as exc:
            self.skipTest(f"Graph visualizer dependency unavailable: {exc}")

        class Spin:
            def __init__(self, value: float) -> None:
                self._value = value

            def value(self) -> float:
                return self._value

        class Future:
            def done(self) -> bool:
                return False

            def cancel(self) -> bool:
                raise AssertionError("worker cancelled")

        class Prepared:
            def __init__(self) -> None:
                self.params = None

        tab = object.__new__(HeatTransferSimulationTab)
        tab.sys_id_state = None
        tab.params = SimulationParameters(color_min_K=0.0)
        tab.inputs = {"color_min_K": Spin(12.0)}
        tab.prepared = Prepared()
        tab.simulation_future = Future()
        tab.simulation_cancel_event = None
        tab._read_params = lambda: (_ for _ in ()).throw(AssertionError("read all params"))
        tab._save_params_to_folder = lambda: (_ for _ in ()).throw(AssertionError("saved immediately"))
        tab._refresh_stats = lambda: (_ for _ in ()).throw(AssertionError("refreshed stats"))
        tab._refresh_sensor_readouts = lambda: (_ for _ in ()).throw(AssertionError("refreshed readouts"))
        tab._status = lambda message, error=False: (_ for _ in ()).throw(AssertionError(message))
        color_updates = []
        saves = []
        tab._update_colors = lambda: color_updates.append(True)
        tab._schedule_parameter_save = lambda: saves.append(True)

        tab._handle_parameter_change("color_min_K")

        self.assertAlmostEqual(tab.params.color_min_K, 12.0)
        self.assertIs(tab.prepared.params, tab.params)
        self.assertEqual(color_updates, [True])
        self.assertEqual(saves, [True])

    def test_simulation_playback_batches_physics_steps_per_visual_update(self) -> None:
        try:
            from graph_visualizer.heat_transfer_simulation_tab import HeatTransferSimulationTab
        except ModuleNotFoundError as exc:
            self.skipTest(f"Graph visualizer dependency unavailable: {exc}")

        class Timer:
            def isActive(self) -> bool:
                return True

        class Prepared:
            def __init__(self) -> None:
                self.time_s = 0.0
                self.values = np.array([300.0], dtype=float)
                self.calls = 0

            @property
            def temperatures_K(self) -> np.ndarray:
                return self.values

            def step_forward(self) -> None:
                self.calls += 1
                self.time_s += 1.0
                self.values = self.values + 1.0

            def reset(self) -> None:
                self.time_s = 0.0

        tab = object.__new__(HeatTransferSimulationTab)
        tab.prepared = Prepared()
        tab.params = SimulationParameters(
            dt_s=1.0,
            t_final_s=100.0,
            playback_speed=10.0,
            display_update_interval_ms=100.0,
        )
        tab.timer = Timer()
        tab._simulation_reinitialize_pending = False
        tab._after_state_change = lambda: None
        statuses = []
        tab._status = lambda message, error=False: statuses.append(message)
        tab.pause = lambda: None

        self.assertEqual(tab._playback_timer_interval_ms(), 100)
        self.assertEqual(tab._playback_steps_per_tick(), 10)

        tab.step_forward()

        self.assertEqual(tab.prepared.calls, 10)
        self.assertAlmostEqual(tab.prepared.time_s, 10.0)
        self.assertAlmostEqual(float(tab.prepared.temperatures_K[0]), 310.0)
        self.assertTrue(any("steps/update = 10" in message for message in statuses))

    def test_simulation_step_forward_submits_background_worker_when_executor_exists(self) -> None:
        try:
            from graph_visualizer.heat_transfer_simulation_tab import HeatTransferSimulationTab
        except ModuleNotFoundError as exc:
            self.skipTest(f"Graph visualizer dependency unavailable: {exc}")

        class Timer:
            def __init__(self, active: bool = False) -> None:
                self.active = active
                self.started = False
                self.stopped = False

            def isActive(self) -> bool:
                return self.active

            def start(self, *_: object) -> None:
                self.started = True

            def stop(self) -> None:
                self.stopped = True

        class Executor:
            def __init__(self) -> None:
                self.future: Future = Future()
                self.submitted: tuple[object, ...] | None = None

            def submit(self, fn, *args):
                self.submitted = (fn, *args)
                return self.future

        class Prepared:
            def __init__(self) -> None:
                self.node_ids = np.array([1], dtype=int)
                self.time_s = 0.0
                self.values = np.array([300.0], dtype=float)
                self.calls = 0
                self.last_step_profile_ms: dict[str, float] = {}

            @property
            def temperatures_K(self) -> np.ndarray:
                return self.values

            def step_forward(self) -> None:
                self.calls += 1
                self.time_s += 1.0
                self.values = self.values + 2.0
                self.last_step_profile_ms = {"model_solve_ms": 1.0}

        executor = Executor()
        tab = object.__new__(HeatTransferSimulationTab)
        tab.prepared = Prepared()
        tab.params = SimulationParameters(t_final_s=10.0, live_step_profile_threshold_ms=1.0e9)
        tab.timer = Timer(active=False)
        tab.simulation_worker_timer = Timer(active=False)
        tab.simulation_executor = executor
        tab.simulation_future = None
        tab.simulation_cancel_event = None
        tab._simulation_worker_mode = None
        tab._simulation_reinitialize_pending = False
        statuses = []
        tab._status = lambda message, error=False: statuses.append(message)
        after_calls = []
        tab._after_state_change = lambda profile=None: after_calls.append(profile)
        tab._report_live_step_profile = lambda profile, steps, max_delta: None

        tab.step_forward()

        self.assertIs(tab.simulation_future, executor.future)
        self.assertTrue(tab.simulation_worker_timer.started)
        self.assertEqual(tab.prepared.calls, 0)
        self.assertTrue(any("background" in message for message in statuses))

        fn, *args = executor.submitted
        executor.future.set_result(fn(*args))
        tab._poll_simulation_worker()

        self.assertIsNone(tab.simulation_future)
        self.assertEqual(tab.prepared.calls, 1)
        self.assertAlmostEqual(float(tab.prepared.temperatures_K[0]), 302.0)
        self.assertEqual(len(after_calls), 1)

    def test_slow_playback_keeps_single_step_interval(self) -> None:
        try:
            from graph_visualizer.heat_transfer_simulation_tab import HeatTransferSimulationTab
        except ModuleNotFoundError as exc:
            self.skipTest(f"Graph visualizer dependency unavailable: {exc}")

        tab = object.__new__(HeatTransferSimulationTab)
        tab.params = SimulationParameters(playback_speed=0.25, display_update_interval_ms=100.0)

        self.assertEqual(tab._playback_timer_interval_ms(), 400)
        self.assertEqual(tab._playback_steps_per_tick(), 1)

    def test_live_step_profile_reports_slow_update_breakdown(self) -> None:
        try:
            from graph_visualizer.heat_transfer_simulation_tab import HeatTransferSimulationTab
        except ModuleNotFoundError as exc:
            self.skipTest(f"Graph visualizer dependency unavailable: {exc}")

        class Timer:
            def isActive(self) -> bool:
                return False

        class Prepared:
            def __init__(self) -> None:
                self.node_ids = np.array([10, 20], dtype=int)
                self.time_s = 0.0
                self.values = np.array([300.0, 301.0], dtype=float)
                self.last_step_profile_ms: dict[str, float] = {}

            @property
            def temperatures_K(self) -> np.ndarray:
                return self.values

            def step_forward(self) -> None:
                self.time_s += 1.0
                self.values = self.values + np.array([1.0, 0.5])
                self.last_step_profile_ms = {
                    "model_solve_ms": 120.0,
                    "state_copy_ms": 10.0,
                    "history_append_ms": 5.0,
                }

        tab = object.__new__(HeatTransferSimulationTab)
        tab.prepared = Prepared()
        tab.params = SimulationParameters(
            t_final_s=10.0,
            live_step_profiling_enabled=True,
            live_step_profile_threshold_ms=0.0,
        )
        tab.timer = Timer()
        tab._simulation_reinitialize_pending = False
        tab.pause = lambda: None
        tab.temperature_by_node = {}
        tab._update_colors = lambda: None
        tab._refresh_stats = lambda: None
        tab._refresh_sensor_readouts = lambda: None
        tab._sync_time_slider_to_history = lambda: None
        statuses = []
        tab._status = lambda message, error=False: statuses.append(message)

        tab.step_forward()

        self.assertEqual(tab.prepared.time_s, 1.0)
        self.assertEqual(tab.temperature_by_node, {10: 301.0, 20: 301.5})
        self.assertTrue(any("Live step profile" in message for message in statuses))
        self.assertTrue(any("solve/controller=120.0 ms" in message for message in statuses))

    def test_simulation_parameter_round_trips_live_step_profile_settings(self) -> None:
        params = SimulationParameters(
            live_step_profiling_enabled=True,
            live_step_profile_threshold_ms=12.5,
            fast_sparse_simulation_enabled=True,
            fast_sparse_simulation_max_substeps=64,
            fast_sparse_simulation_safety_factor=0.5,
            implicit_sparse_simulation_enabled=True,
            implicit_sparse_simulation_method="tr_bdf2",
            implicit_sparse_simulation_rtol=1.0e-4,
            implicit_sparse_simulation_maxiter=321,
            implicit_sparse_adaptive_substeps_enabled=True,
            implicit_sparse_adaptive_target_delta_K=0.25,
            implicit_sparse_adaptive_max_substeps=3,
            implicit_sparse_residual_check_enabled=True,
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "simulation_parameters.json"
            save_simulation_parameters(path, params, {"custom": "kept"})

            loaded, extras = load_simulation_parameters(path)

        self.assertTrue(loaded.live_step_profiling_enabled)
        self.assertAlmostEqual(loaded.live_step_profile_threshold_ms, 12.5)
        self.assertTrue(loaded.fast_sparse_simulation_enabled)
        self.assertEqual(loaded.fast_sparse_simulation_max_substeps, 64)
        self.assertAlmostEqual(loaded.fast_sparse_simulation_safety_factor, 0.5)
        self.assertTrue(loaded.implicit_sparse_simulation_enabled)
        self.assertEqual(loaded.implicit_sparse_simulation_method, "tr_bdf2")
        self.assertAlmostEqual(loaded.implicit_sparse_simulation_rtol, 1.0e-4)
        self.assertEqual(loaded.implicit_sparse_simulation_maxiter, 321)
        self.assertTrue(loaded.implicit_sparse_adaptive_substeps_enabled)
        self.assertAlmostEqual(loaded.implicit_sparse_adaptive_target_delta_K, 0.25)
        self.assertEqual(loaded.implicit_sparse_adaptive_max_substeps, 3)
        self.assertTrue(loaded.implicit_sparse_residual_check_enabled)
        self.assertEqual(extras["custom"], "kept")

    def test_octree_node_load_sanitizes_nonfinite_geometry(self) -> None:
        model = ThermalGraphModel.from_octree_graph_dict(
            {
                "metadata": {"graph_name": "bad_geometry"},
                "graph_nodes": [
                    {
                        "node_id": 5,
                        "coord": [5, 0, 0],
                        "center_mm": [0.0, float("nan"), 0.0],
                        "size_mm": [1.0, float("inf"), 1.0],
                        "source_bounds_mm": {
                            "min": [0.0, 0.0, 0.0],
                            "max": [1.0, float("nan"), 1.0],
                        },
                        "component_name": "bad_sensor",
                        "material_name": "Copper",
                        "is_sensor": True,
                    }
                ],
                "graph_edges": [],
            }
        )

        node = model.nodes[5]
        self.assertIsNone(node.center_mm)
        self.assertIsNone(node.size_mm)
        self.assertEqual(node.source_bounds_mm, {})
        self.assertIn("non-finite center_mm", " ".join(node.warnings))
        self.assertIn("non-finite size_mm", " ".join(node.warnings))

    def test_octree_node_load_promotes_legacy_role_tags(self) -> None:
        node = NodeProperties.from_dict(
            {
                "node_id": 31,
                "coord": [31, 0, 0],
                "tags": {
                    "heater": True,
                    "sensor": True,
                    "heater_id": 310,
                    "sensor_id": 311,
                },
            }
        )

        self.assertTrue(node.is_heater)
        self.assertTrue(node.is_sensor)
        self.assertEqual(node.heater.heater_id, 310)
        self.assertEqual(node.sensor.sensor_id, 311)

    def test_octree_node_load_ignores_warning_tags_metadata(self) -> None:
        node = NodeProperties.from_dict(
            {
                "node_id": 32,
                "coord": [32, 0, 0],
                "material_name": "Copper",
                "warning_tags": ["oversized_cell"],
                "tags": {"warning_tags": ["low_confidence"]},
            }
        )

        self.assertEqual(node.node_id, 32)
        self.assertEqual(node.material, "Copper")

    def test_octree_graph_load_applies_top_level_heater_sensor_tags(self) -> None:
        model = ThermalGraphModel.from_octree_graph_dict(
            {
                "metadata": {"graph_name": "legacy_tags"},
                "graph_nodes": [
                    {
                        "node_id": 41,
                        "coord": [41, 0, 0],
                        "center_mm": [1.0, 2.0, 3.0],
                        "size_mm": [1.0, 1.0, 1.0],
                        "component_name": "legacy_heater_cell",
                        "material_name": "Copper",
                        "is_heater": False,
                        "is_sensor": False,
                    }
                ],
                "graph_edges": [],
                "heater_sensor_tags": {
                    "41": {
                        "heater": True,
                        "sensor": False,
                        "heater_id": 41,
                        "sensor_id": None,
                        "notes": "legacy assignment",
                    }
                },
            }
        )

        node = model.nodes[41]
        self.assertTrue(node.is_heater)
        self.assertFalse(node.is_sensor)
        self.assertEqual(node.heater.heater_id, 41)
        self.assertEqual(node.notes, "legacy assignment")

    def test_mimo_controller_metadata_and_gain_matrix_round_trip(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_round_trip"))
        sensor = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        sensor.is_sensor = True
        sensor.controller_setpoint_K = 315.0
        sensor.controller_weight = 2.0
        sensor.sensor_settling_time_s = 8.0
        sensor.controller_kp_coarse = 0.75
        sensor.controller_ki_coarse = 0.25
        sensor.controller_kp_hold = 0.5
        sensor.controller_ki_hold = 0.125
        sensor.controller_kd_coarse = 0.4
        sensor.controller_kd_hold = 0.2
        sensor.controller_lambda_order = 0.8
        sensor.controller_mu_order = 0.6
        heater = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        heater.is_heater = True
        heater.is_sensor = True
        heater.heater_control.mode = "mimo"
        model.add_node(sensor)
        model.add_node(heater)
        model.set_controller_gain(1, 2, 0.125)

        restored = ThermalGraphModel.from_octree_graph_dict(model.to_octree_graph_dict())

        self.assertAlmostEqual(restored.nodes[1].controller_setpoint_K, 315.0)
        self.assertAlmostEqual(restored.nodes[1].controller_weight, 2.0)
        self.assertAlmostEqual(restored.nodes[1].sensor_settling_time_s, 8.0)
        self.assertAlmostEqual(restored.nodes[1].controller_kp_coarse, 0.75)
        self.assertAlmostEqual(restored.nodes[1].controller_ki_coarse, 0.25)
        self.assertAlmostEqual(restored.nodes[1].controller_kp_hold, 0.5)
        self.assertAlmostEqual(restored.nodes[1].controller_ki_hold, 0.125)
        self.assertAlmostEqual(restored.nodes[1].controller_kd_coarse, 0.4)
        self.assertAlmostEqual(restored.nodes[1].controller_kd_hold, 0.2)
        self.assertAlmostEqual(restored.nodes[1].controller_lambda_order, 0.8)
        self.assertAlmostEqual(restored.nodes[1].controller_mu_order, 0.6)
        self.assertFalse(hasattr(restored.nodes[1], "controller_integral_negative_error_leak_per_s"))
        self.assertAlmostEqual(restored.controller_gain(1, 2), 0.125)

    def test_legacy_mimo_gain_scales_load_as_sensor_specific_gains(self) -> None:
        node = NodeProperties.from_dict(
            {
                "node_id": 1,
                "coord": [0, 0, 0],
                "controller": {
                    "setpoint_K": 312.0,
                    "kp_scale": 0.75,
                    "ki_scale": 0.25,
                },
            }
        )

        self.assertAlmostEqual(node.controller_kp_coarse, 0.75)
        self.assertAlmostEqual(node.controller_ki_coarse, 0.25)
        self.assertAlmostEqual(node.controller_kp_hold, 0.75)
        self.assertAlmostEqual(node.controller_ki_hold, 0.25)
        serialized = node.to_octree_node_dict()["controller"]
        self.assertNotIn("kp_scale", serialized)
        self.assertNotIn("ki_scale", serialized)
        self.assertAlmostEqual(serialized["kp_coarse"], 0.75)

    def test_heater_inputs_still_conducts_with_zero_heater_power(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="zero_power_conduction"))
        hot = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        cold = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        hot.C_J_K = cold.C_J_K = 10.0
        hot.initial_temperature_K = 310.0
        cold.initial_temperature_K = 290.0
        cold.is_heater = True
        cold.is_sensor = True
        cold.heater_control.manual.power = 0.0
        model.add_node(hot)
        model.add_node(cold)
        matrices = {
            "node_ids": np.array([1, 2], dtype=int),
            "C": np.array([10.0, 10.0]),
            "L": np.array([[2.0, -2.0], [-2.0, 2.0]]),
            "G_rad": np.array([0.0, 0.0]),
        }

        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(dt_s=1.0, input_mode="heater_inputs", use_ambient_radiation=False),
        )
        prepared.step_forward()

        self.assertLess(float(prepared.temperatures_K[0]), 310.0)
        self.assertGreater(float(prepared.temperatures_K[1]), 290.0)
        self.assertEqual(prepared.heater_power_by_node(), {2: 0.0})

    def test_forced_heater_power_step_and_snapshot_restore(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="sys_id_forced_power"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.is_heater = True
        node.is_sensor = True
        model.add_node(node)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(dt_s=1.0, input_mode="zero", use_ambient_radiation=False),
        )
        snapshot = prepared.snapshot_state()

        prepared.step_with_forced_heater_powers({1: 10.0}, keep_cryocoolers_active=False)

        self.assertAlmostEqual(float(prepared.temperatures_K[0]), 301.0)

        prepared.restore_state(snapshot)

        self.assertAlmostEqual(float(prepared.temperatures_K[0]), 300.0)
        self.assertEqual(prepared.history_index, snapshot.history_index)

        prepared.set_uniform_temperature(275.0)

        self.assertAlmostEqual(float(prepared.temperatures_K[0]), 275.0)
        self.assertEqual(prepared.time_s, 0.0)

    def test_prepared_simulation_can_reset_to_node_temperature_vector(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="vector_temperature_reset"))
        for node_id in (1, 2):
            node = NodeProperties.with_material(node_id, (node_id, 0, 0), material="copper")
            node.C_J_K = 10.0
            node.initial_temperature_K = 300.0
            model.add_node(node)
        matrices = {
            "node_ids": np.array([1, 2], dtype=int),
            "C": np.array([10.0, 10.0]),
            "L": np.zeros((2, 2)),
            "G_rad": np.zeros(2),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(dt_s=1.0, input_mode="zero", use_ambient_radiation=False),
        )

        prepared.set_temperatures(np.array([275.0, 315.0]))

        np.testing.assert_allclose(prepared.temperatures_K, np.array([275.0, 315.0]))
        np.testing.assert_allclose(prepared.history[prepared.history_index].temperatures_K, np.array([275.0, 315.0]))
        self.assertEqual(prepared.time_s, 0.0)
        with self.assertRaises(ValueError):
            prepared.set_temperatures(np.array([275.0]))

    def test_forced_heater_power_step_only_applies_requested_heater(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="sys_id_single_forced_heater"))
        for node_id in (1, 2):
            node = NodeProperties.with_material(node_id, (node_id, 0, 0), material="copper")
            node.C_J_K = 10.0
            node.initial_temperature_K = 300.0
            node.is_heater = True
            node.is_sensor = True
            node.heater_control.mode = "pid"
            node.heater_control.pid.kp = 100.0
            node.heater_control.pid.setpoint = 350.0
            node.heater_control.manual.power = 9.0
            model.add_node(node)
        matrices = {
            "node_ids": np.array([1, 2], dtype=int),
            "C": np.array([10.0, 10.0]),
            "L": np.zeros((2, 2)),
            "G_rad": np.array([0.0, 0.0]),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(dt_s=1.0, input_mode="heater_inputs", use_ambient_radiation=False),
        )

        prepared.step_with_forced_heater_powers({2: 10.0}, keep_cryocoolers_active=False)

        self.assertAlmostEqual(float(prepared.temperatures_K[0]), 300.0)
        self.assertAlmostEqual(float(prepared.temperatures_K[1]), 301.0)

    def test_sys_id_gain_matrix_artifact_round_trips_and_updates(self) -> None:
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            run_path = save_sys_id_gain_matrix(
                folder,
                "run one",
                [10, 11],
                [20, 21],
                np.array([[0.0, 0.2], [0.3, 0.0]], dtype=float),
                {"global_temperature_K": 280.0, "requested_delta_power_W": 2.0},
            )

            self.assertEqual(run_path, folder / "simulations" / "sys_id" / "run_one")
            infos = list_sys_id_gain_matrices(folder)
            self.assertEqual([info.name for info in infos], ["run_one"])
            self.assertEqual(
                load_sys_id_gain_matrix(run_path),
                {10: {21: 0.2}, 11: {20: 0.3}},
            )
            metadata = json.loads((run_path / "metadata.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(metadata["global_temperature_K"], 280.0)

            update_sys_id_gain_matrix(run_path, {10: {20: 1.5}, 11: {21: 2.5}})

            self.assertEqual(
                load_sys_id_gain_matrix(run_path),
                {10: {20: 1.5}, 11: {21: 2.5}},
            )
            csv_text = (run_path / "gain_matrix.csv").read_text(encoding="utf-8")
            self.assertIn("10,20,1.5", csv_text)

    def test_sys_id_gain_matrix_compare_aligns_axes_and_reports_metrics(self) -> None:
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            baseline = save_sys_id_gain_matrix(
                folder,
                "baseline",
                [1, 2],
                [10],
                np.array([[1.0], [2.0]], dtype=float),
            )
            comparison = save_sys_id_gain_matrix(
                folder,
                "comparison",
                [2, 3],
                [10, 11],
                np.array([[3.0, 4.0], [5.0, 6.0]], dtype=float),
            )

            result = compare_sys_id_gain_matrices(baseline, comparison)

            self.assertEqual(result["sensor_ids"], [1, 2, 3])
            self.assertEqual(result["heater_ids"], [10, 11])
            np.testing.assert_allclose(
                result["baseline_G"],
                np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]], dtype=float),
            )
            np.testing.assert_allclose(
                result["comparison_G"],
                np.array([[0.0, 0.0], [3.0, 4.0], [5.0, 6.0]], dtype=float),
            )
            np.testing.assert_allclose(
                result["delta_G"],
                np.array([[-1.0, 0.0], [1.0, 4.0], [5.0, 6.0]], dtype=float),
            )
            self.assertAlmostEqual(result["metrics"]["max_abs_delta"], 6.0)
            self.assertGreater(result["metrics"]["relative_frobenius_error"], 0.0)

    def test_heater_actuator_power_excludes_cryocooler_power(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="actuator_power_only"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.is_heater = True
        node.is_sensor = True
        node.has_cryocooler = True
        node.assigned_sensor_id = 1
        node.assigned_heater_id = 1
        node.sensor_control_mode = "manual"
        node.sensor_manual_power_W = 3.0
        model.add_node(node)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(
                dt_s=1.0,
                input_mode="heater_inputs",
                Kp_cooler=1.0,
                P_cooler_max=20.0,
                T_cooler_setpoint=290.0,
            ),
        )

        self.assertEqual(prepared.heater_actuator_power_by_node(), {1: 3.0})
        self.assertEqual(prepared.heater_power_by_node(), {1: -7.0})

    def test_pid_heater_output_is_clamped_and_pid_state_resets(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="pid_heater"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 4.0
        node.heater_control.mode = "pid"
        node.heater_control.pid.kp = 100.0
        node.heater_control.pid.setpoint = 350.0
        model.add_node(node)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }

        prepared = prepare_simulation(model, matrices, SimulationParameters(dt_s=1.0, input_mode="heater_inputs"))
        preview_power = prepared.heater_power_by_node()

        self.assertEqual(preview_power, {1: 0.0})
        self.assertEqual(model.nodes[1].heater_control.pid_state.integral, 0.0)
        self.assertIsNone(model.nodes[1].heater_control.pid_state.previous_error)

        prepared.step_forward()
        temperature_after_step = float(prepared.temperatures_K[0])
        self.assertAlmostEqual(temperature_after_step, 300.0)
        self.assertIsNone(model.nodes[1].heater_control.pid_state.previous_error)

        prepared.reset()

        self.assertEqual(model.nodes[1].heater_control.pid_state.integral, 0.0)
        self.assertIsNone(model.nodes[1].heater_control.pid_state.previous_error)

    def test_pid_integral_can_go_negative_above_setpoint(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="pid_signed_integral"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 310.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 20.0
        node.heater_control.mode = "pid"
        node.heater_control.pid.ki = 1.0
        node.heater_control.pid.setpoint = 300.0
        model.add_node(node)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }

        prepared = prepare_simulation(model, matrices, SimulationParameters(dt_s=1.0, input_mode="heater_inputs"))
        prepared.step_forward()

        self.assertEqual(model.nodes[1].heater_control.pid_state.integral, 0.0)
        self.assertAlmostEqual(float(prepared.temperatures_K[0]), 310.0)

    def test_pid_negative_integral_keeps_output_at_zero_above_setpoint(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="pid_above_setpoint"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 310.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 20.0
        node.heater_control.mode = "pid"
        node.heater_control.pid.ki = 1.0
        node.heater_control.pid.setpoint = 300.0
        node.heater_control.pid_state.integral = 100.0
        model.add_node(node)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }

        prepared = prepare_simulation(model, matrices, SimulationParameters(dt_s=1.0, input_mode="heater_inputs"))
        model.nodes[1].heater_control.pid_state.integral = 100.0

        self.assertEqual(prepared.heater_power_by_node(), {1: 0.0})

        prepared.step_forward()

        self.assertAlmostEqual(float(prepared.temperatures_K[0]), 310.0)
        self.assertEqual(model.nodes[1].heater_control.pid_state.integral, 100.0)
        self.assertEqual(model.nodes[1].heater_control.pid_state.error_history, [])

        model.nodes[1].heater_control.pid_state.integral = 5.0
        prepared.step_forward()

        self.assertEqual(model.nodes[1].heater_control.pid_state.integral, 5.0)

    def test_pid_integral_uses_fractional_history_without_leak(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="pid_integral_history"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 290.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 20.0
        node.heater_control.mode = "pid"
        node.heater_control.pid.ki = 0.0
        node.heater_control.pid.setpoint = 300.0
        model.add_node(node)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }

        prepared = prepare_simulation(model, matrices, SimulationParameters(dt_s=1.0, input_mode="heater_inputs"))
        model.nodes[1].heater_control.pid_state.integral = 100.0

        prepared.step_forward()

        self.assertAlmostEqual(model.nodes[1].heater_control.pid_state.integral, 100.0)

        prepared.z[0] = 310.0
        model.nodes[1].heater_control.pid_state.integral = 100.0
        prepared.step_forward()

        self.assertAlmostEqual(model.nodes[1].heater_control.pid_state.integral, 100.0)

    def test_fractional_pid_integral_order_scales_error_memory(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="fractional_pid_integral"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 290.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 100.0
        node.heater_control.mode = "pid"
        node.heater_control.pid.ki = 1.0
        node.heater_control.pid.lambda_order = 0.5
        node.heater_control.pid.setpoint = 300.0
        model.add_node(node)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }

        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(dt_s=4.0, input_mode="heater_inputs", use_ambient_radiation=False),
        )

        self.assertEqual(prepared.heater_power_by_node(), {1: 0.0})
        self.assertEqual(model.nodes[1].heater_control.pid_state.error_history, [])
        prepared.step_forward()

        self.assertAlmostEqual(model.nodes[1].heater_control.pid_state.integral, 0.0)
        self.assertEqual(model.nodes[1].heater_control.pid_state.error_history, [])
        self.assertAlmostEqual(float(prepared.temperatures_K[0]), 290.0)

    def test_fractional_pid_derivative_order_uses_error_history(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="fractional_pid_derivative"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 291.0
        node.is_heater = True
        node.is_sensor = True
        node.heater.heater_max_power_W = 100.0
        node.heater_control.mode = "pid"
        node.heater_control.pid.kd = 1.0
        node.heater_control.pid.mu_order = 0.5
        node.heater_control.pid.setpoint = 300.0
        node.heater_control.pid_state.previous_error = 4.0
        node.heater_control.pid_state.error_history = [4.0]
        model.add_node(node)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([0.0]),
        }

        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(dt_s=4.0, input_mode="heater_inputs", use_ambient_radiation=False),
        )
        model.nodes[1].heater_control.pid_state.previous_error = 4.0
        model.nodes[1].heater_control.pid_state.error_history = [4.0]

        self.assertAlmostEqual(prepared.heater_power_by_node()[1], 0.0)

    def test_loaded_heater_keeps_sensor_role_separate_and_default_control(self) -> None:
        node = NodeProperties.from_dict(
            {
                "node_id": 3,
                "coord": [0, 0, 0],
                "is_heater": True,
                "heater": {"heater_id": 3, "heater_max_power_W": 12.0, "heater_efficiency": 0.5},
                "heater_control": {"pid": {"integral_leak_per_s": 0.25}},
            }
        )

        self.assertTrue(node.is_heater)
        self.assertFalse(node.is_sensor)
        self.assertEqual(node.heater_control.mode, "manual")
        self.assertAlmostEqual(node.heater_control.manual.power, 6.0)
        self.assertAlmostEqual(node.heater_control.pid.integral_leak_per_s, 0.25)
        self.assertAlmostEqual(node.heater_control.pid.lambda_order, 1.0)
        self.assertAlmostEqual(node.heater_control.pid.mu_order, 1.0)

    def test_large_simulation_uses_sparse_implicit_cpu_stepper_by_default(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="large_sparse_stepper"))
        node_count = 513
        for node_id in range(node_count):
            node = NodeProperties.with_material(node_id, (node_id, 0, 0), material="copper")
            node.C_J_K = 10.0
            node.G_rad_W_K = 1.0
            node.initial_temperature_K = 310.0
            model.add_node(node)
        matrices = {
            "node_ids": np.arange(node_count, dtype=int),
            "C": np.full(node_count, 10.0),
            "L": np.zeros((node_count, node_count)),
            "G_rad": np.ones(node_count),
        }

        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(dt_s=1.0, T_env_K=290.0, use_ambient_radiation=True),
        )
        before = prepared.temperatures_K.copy()
        prepared.step_forward()

        self.assertIsNone(prepared.Phi_aug)
        self.assertLess(float(prepared.temperatures_K[0]), float(before[0]))
        self.assertIsNotNone(prepared.sparse_implicit_stepper)
        self.assertEqual(prepared.sparse_implicit_stepper.method, "tr_bdf2")
        self.assertIn("cpu_sparse_implicit_step_ms", prepared.last_step_profile_ms)
        self.assertIn("cpu_sparse_implicit_residual_norm", prepared.last_step_profile_ms)
        self.assertNotIn("cpu_expm_multiply_ms", prepared.last_step_profile_ms)
        self.assertIn("model_solve_ms", prepared.last_step_profile_ms)

    def test_sparse_implicit_cpu_stepper_bypasses_expm_multiply_automatically(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="implicit_sparse_cpu"))
        node_count = 513
        for node_id in range(node_count):
            node = NodeProperties.with_material(node_id, (node_id, 0, 0), material="copper")
            node.C_J_K = 10.0
            node.initial_temperature_K = 300.0
            model.add_node(node)
        diagonal = np.ones(node_count, dtype=float)
        matrices = {
            "node_ids": np.arange(node_count, dtype=int),
            "C": np.full(node_count, 10.0),
            "L": csr_matrix((diagonal, (np.arange(node_count), np.arange(node_count))), shape=(node_count, node_count)),
            "G_rad": np.zeros(node_count),
        }

        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(dt_s=1.0, use_ambient_radiation=False),
        )
        prepared.step_forward()

        self.assertIsNotNone(prepared.sparse_implicit_stepper)
        self.assertIn("cpu_sparse_implicit_step_ms", prepared.last_step_profile_ms)
        self.assertNotIn("cpu_expm_multiply_ms", prepared.last_step_profile_ms)
        exact = 300.0 * np.exp(-0.1)
        backward_euler = 10.0 * 300.0 / 11.0
        self.assertLess(abs(float(prepared.temperatures_K[0]) - exact), abs(backward_euler - exact))

    def test_sparse_implicit_cpu_stepper_keeps_diffusing_small_gradients(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="implicit_small_gradient"))
        node_count = 513
        for node_id in range(node_count):
            node = NodeProperties.with_material(node_id, (node_id, 0, 0), material="copper")
            node.C_J_K = 1.0
            node.initial_temperature_K = 300.0
            model.add_node(node)
        model.nodes[0].initial_temperature_K = 300.001
        conductance = 1.0e-3
        row = np.array([0, 0, 1, 1], dtype=int)
        col = np.array([0, 1, 0, 1], dtype=int)
        data = np.array([conductance, -conductance, -conductance, conductance], dtype=float)
        matrices = {
            "node_ids": np.arange(node_count, dtype=int),
            "C": np.ones(node_count, dtype=float),
            "L": csr_matrix((data, (row, col)), shape=(node_count, node_count)),
            "G_rad": np.zeros(node_count),
        }

        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(dt_s=1.0, use_ambient_radiation=False),
        )
        before = prepared.temperatures_K.copy()
        prepared.step_forward()

        self.assertIsNotNone(prepared.sparse_implicit_stepper)
        self.assertIn("cpu_sparse_implicit_step_ms", prepared.last_step_profile_ms)
        self.assertGreaterEqual(prepared.last_step_profile_ms["cpu_sparse_implicit_iterations"], 0.0)
        self.assertLess(float(prepared.temperatures_K[0]), float(before[0]))
        self.assertGreater(float(prepared.temperatures_K[1]), float(before[1]))
        self.assertGreater(float(np.max(np.abs(prepared.temperatures_K - before))), 1.0e-9)

    def test_sparse_implicit_tr_bdf2_adapts_substeps_and_reports_residual(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="implicit_tr_bdf2_adaptive"))
        node_count = 513
        for node_id in range(node_count):
            node = NodeProperties.with_material(node_id, (node_id, 0, 0), material="copper")
            node.C_J_K = 1.0
            node.initial_temperature_K = 10.0
            model.add_node(node)
        diagonal = np.full(node_count, 10.0, dtype=float)
        matrices = {
            "node_ids": np.arange(node_count, dtype=int),
            "C": np.ones(node_count, dtype=float),
            "L": csr_matrix((diagonal, (np.arange(node_count), np.arange(node_count))), shape=(node_count, node_count)),
            "G_rad": np.zeros(node_count),
        }

        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(
                dt_s=1.0,
                use_ambient_radiation=False,
                implicit_sparse_adaptive_target_delta_K=1.0,
                implicit_sparse_adaptive_max_substeps=4,
            ),
        )
        prepared.step_forward()

        self.assertIsNotNone(prepared.sparse_implicit_stepper)
        self.assertEqual(prepared.sparse_implicit_stepper.last_substeps, 4)
        self.assertEqual(prepared.last_step_profile_ms["cpu_sparse_implicit_substeps"], 4.0)
        self.assertGreater(prepared.last_step_profile_ms["cpu_sparse_implicit_predicted_delta_K"], 1.0)
        self.assertLess(prepared.last_step_profile_ms["cpu_sparse_implicit_relative_residual"], 1.0e-5)

    def test_fast_sparse_cpu_stepper_bypasses_expm_multiply_automatically(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="fast_sparse_cpu"))
        node_count = 513
        for node_id in range(node_count):
            node = NodeProperties.with_material(node_id, (node_id, 0, 0), material="copper")
            node.C_J_K = 10.0
            node.initial_temperature_K = 300.0
            model.add_node(node)
        diagonal = np.ones(node_count, dtype=float)
        matrices = {
            "node_ids": np.arange(node_count, dtype=int),
            "C": np.full(node_count, 10.0),
            "L": csr_matrix((diagonal, (np.arange(node_count), np.arange(node_count))), shape=(node_count, node_count)),
            "G_rad": np.zeros(node_count),
        }

        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(
                dt_s=1.0,
                use_ambient_radiation=False,
                implicit_sparse_simulation_enabled=False,
                fast_sparse_simulation_safety_factor=1.0,
            ),
        )
        prepared.step_forward()

        self.assertEqual(prepared.fast_sparse_substeps, 1)
        self.assertIn("cpu_fast_sparse_step_ms", prepared.last_step_profile_ms)
        self.assertNotIn("cpu_expm_multiply_ms", prepared.last_step_profile_ms)

    def test_fast_sparse_cpu_stepper_falls_back_when_substep_limit_is_too_low(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="fast_sparse_cpu_fallback"))
        node_count = 513
        for node_id in range(node_count):
            node = NodeProperties.with_material(node_id, (node_id, 0, 0), material="copper")
            node.C_J_K = 10.0
            node.initial_temperature_K = 300.0
            model.add_node(node)
        diagonal = np.ones(node_count, dtype=float)
        matrices = {
            "node_ids": np.arange(node_count, dtype=int),
            "C": np.full(node_count, 10.0),
            "L": csr_matrix((diagonal, (np.arange(node_count), np.arange(node_count))), shape=(node_count, node_count)),
            "G_rad": np.zeros(node_count),
        }

        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(
                dt_s=100.0,
                use_ambient_radiation=False,
                implicit_sparse_simulation_enabled=False,
                fast_sparse_simulation_enabled=True,
                fast_sparse_simulation_max_substeps=1,
            ),
        )
        prepared.step_forward()

        self.assertIsNone(prepared.fast_sparse_substeps)
        self.assertIn("cpu_expm_multiply_ms", prepared.last_step_profile_ms)
        self.assertTrue(any("Fast sparse CPU stepping" in warning for warning in prepared.warnings))

    def test_gpu_simulation_request_falls_back_when_cupy_unavailable(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="gpu_unavailable"))
        node_count = 513
        for node_id in range(node_count):
            node = NodeProperties.with_material(node_id, (node_id, 0, 0), material="copper")
            node.C_J_K = 10.0
            node.G_rad_W_K = 1.0
            node.initial_temperature_K = 310.0
            model.add_node(node)
        matrices = {
            "node_ids": np.arange(node_count, dtype=int),
            "C": np.full(node_count, 10.0),
            "L": csr_matrix((node_count, node_count), dtype=float),
            "G_rad": np.ones(node_count),
        }

        with patch(
            "graph_visualizer.simulation_model._optional_cupy_modules",
            return_value=(None, None, "not installed"),
        ):
            prepared = prepare_simulation(
                model,
                matrices,
                SimulationParameters(dt_s=1.0, gpu_simulation_enabled=True),
            )

        self.assertIsNone(prepared.gpu_stepper)
        self.assertTrue(any("GPU sparse stepping unavailable" in warning for warning in prepared.warnings))

    def test_gpu_stepper_is_used_when_available(self) -> None:
        class FakeGpuStepper:
            def __init__(self) -> None:
                self.calls = 0

            def step(self, temperatures_K: np.ndarray, heater_power: np.ndarray) -> np.ndarray:
                self.calls += 1
                return np.asarray(temperatures_K, dtype=float) - 2.0

        fake_stepper = FakeGpuStepper()
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="gpu_hook"))
        node_count = 513
        for node_id in range(node_count):
            node = NodeProperties.with_material(node_id, (node_id, 0, 0), material="copper")
            node.C_J_K = 10.0
            node.initial_temperature_K = 310.0
            model.add_node(node)
        matrices = {
            "node_ids": np.arange(node_count, dtype=int),
            "C": np.full(node_count, 10.0),
            "L": csr_matrix((node_count, node_count), dtype=float),
            "G_rad": np.zeros(node_count),
        }

        with patch("graph_visualizer.simulation_model._build_gpu_sparse_stepper", return_value=fake_stepper):
            prepared = prepare_simulation(
                model,
                matrices,
                SimulationParameters(dt_s=1.0, gpu_simulation_enabled=True),
            )
        prepared.step_forward()

        self.assertEqual(fake_stepper.calls, 1)
        self.assertAlmostEqual(float(prepared.temperatures_K[0]), 308.0)
        self.assertIn("gpu_step_ms", prepared.last_step_profile_ms)
        self.assertIn("state_vector_update_ms", prepared.last_step_profile_ms)

    def test_gpu_sparse_stepper_keeps_temperature_state_on_device(self) -> None:
        from graph_visualizer.simulation_model import GpuSparseStepper

        class FakeCp:
            def __init__(self) -> None:
                self.asarray_calls = 0

            def asarray(self, value):
                self.asarray_calls += 1
                return np.asarray(value, dtype=float)

            def asnumpy(self, value):
                return np.asarray(value, dtype=float)

        cp = FakeCp()
        stepper = GpuSparseStepper(
            cp=cp,
            A_gpu=np.zeros((2, 2), dtype=float),
            inv_C_gpu=np.ones(2, dtype=float),
            base_b_gpu=np.zeros(2, dtype=float),
            radiation_coeff_gpu=np.zeros(2, dtype=float),
            use_ambient_radiation=False,
            ambient_K=293.15,
            dt_s=1.0,
            substeps=1,
        )

        stepper.step(np.array([300.0, 301.0]), np.zeros(2))
        stepper.step(np.array([999.0, 999.0]), np.zeros(2))

        self.assertEqual(cp.asarray_calls, 3)
        np.testing.assert_allclose(stepper.step(np.array([999.0, 999.0]), np.zeros(2)), np.array([300.0, 301.0]))

    def test_simulation_history_seek_preserves_computed_future(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="history_cursor"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.G_rad_W_K = 1.0
        node.initial_temperature_K = 310.0
        model.add_node(node)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([1.0]),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(dt_s=1.0, T_env_K=290.0, use_ambient_radiation=True),
        )
        prepared.step_forward()
        prepared.step_forward()
        future_temperature = float(prepared.temperatures_K[0])

        prepared.seek(0)

        self.assertEqual(prepared.history_index, 0)
        self.assertEqual(len(prepared.history), 3)
        self.assertAlmostEqual(prepared.time_s, 0.0)

        prepared.step_forward()
        prepared.step_forward()

        self.assertEqual(prepared.history_index, 2)
        self.assertEqual(len(prepared.history), 3)
        self.assertAlmostEqual(float(prepared.temperatures_K[0]), future_temperature)

        prepared.step_forward()

        self.assertEqual(prepared.history_index, 3)
        self.assertEqual(len(prepared.history), 4)

    def test_simulation_history_limit_retains_recent_states_only(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="bounded_history"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.G_rad_W_K = 1.0
        node.initial_temperature_K = 310.0
        model.add_node(node)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([1.0]),
        }
        prepared = prepare_simulation(
            model,
            matrices,
            SimulationParameters(
                dt_s=1.0,
                T_env_K=290.0,
                use_ambient_radiation=True,
                simulation_history_limit=3,
            ),
        )

        for _ in range(5):
            prepared.step_forward()

        self.assertEqual(len(prepared.history), 3)
        self.assertEqual(prepared.history_index, 2)
        self.assertAlmostEqual(prepared.history[0].time_s, 3.0)
        self.assertAlmostEqual(prepared.time_s, 5.0)

    def test_initial_temperature_parameter_payload_is_keyed_by_node_id(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="initials"))
        node_2 = NodeProperties.with_material(2, (0, 0, 0), material="copper")
        node_2.initial_temperature_K = 301.5
        node_9 = NodeProperties.with_material(9, (1, 0, 0), material="copper")
        node_9.initial_temperature_K = 315.25
        model.add_node(node_9)
        model.add_node(node_2)

        payload = initial_temperature_parameter_payload(model)

        self.assertEqual(
            payload["initial_temperature_by_node_K"],
            {"2": 301.5, "9": 315.25},
        )

    def test_initial_temperature_parameter_payload_applies_to_model(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="apply_initials"))
        node = NodeProperties.with_material(4, (0, 0, 0), material="copper")
        model.add_node(node)

        updated = apply_initial_temperature_parameter_payload(
            model,
            {"initial_temperature_by_node_K": {"4": 333.0, "999": 111.0}},
        )

        self.assertEqual(updated, 1)
        self.assertAlmostEqual(model.nodes[4].initial_temperature_K, 333.0)

    def test_loaded_g_mode_accepts_minimal_conductance_matrix(self) -> None:
        model = self.make_model()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "matrices.npz"
            node_ids = np.array([2, 7, 10], dtype=int)
            G = np.array(
                [
                    [0.0, 1.5, 0.0],
                    [1.5, 0.0, 2.5],
                    [0.0, 2.5, 0.0],
                ]
            )
            np.savez(path, node_ids=node_ids, G=G)
            load_conductance_matrix_from_folder(model, directory)
        self.assertEqual(model.metadata.edge_mode, EdgeMode.LOADED_G.value)
        self.assertEqual(model.edges[(2, 7)].Gij_W_K, 1.5)
        self.assertEqual(model.edges[(7, 10)].Gij_W_K, 2.5)

    def test_draw_face_normal_uses_largest_offset_axis(self) -> None:
        self.assertEqual(compute_face_normal((0, 0, 0), (0.51, 0.1, -0.05)), (1, 0, 0))
        self.assertEqual(compute_face_normal((0, 0, 0), (-0.52, 0.05, 0.02)), (-1, 0, 0))
        self.assertEqual(compute_face_normal((0, 0, 0), (0.01, 0.49, 0.08)), (0, 1, 0))
        self.assertEqual(compute_face_normal((0, 0, 0), (0.02, 0.05, -0.50)), (0, 0, -1))

    def test_draw_preview_stops_at_collision(self) -> None:
        coords = preview_coords((0, 0, 0), (1, 0, 0), 5, {(3, 0, 0)})
        self.assertEqual(coords, [(1, 0, 0), (2, 0, 0)])
        self.assertEqual(preview_coords((0, 0, 0), (1, 0, 0), 5, {(1, 0, 0)}), [])

    def test_draw_count_uses_projected_drag_not_raw_distance(self) -> None:
        self.assertEqual(
            extrusion_count_from_projected_pixel_drag((0, 0), (0, 500), (1.0, 0.0)),
            0,
        )
        self.assertEqual(
            extrusion_count_from_projected_pixel_drag((0, 0), (170, 500), (1.0, 0.0)),
            2,
        )
        self.assertEqual(
            extrusion_count_from_projected_pixel_drag((100, 0), (0, 0), (1.0, 0.0)),
            0,
        )

    def test_extruded_node_copies_thermal_mass_but_resets_io_metadata(self) -> None:
        source = NodeProperties.with_material(10, (0, 0, 0), material="copper")
        source.is_heater = True
        source.heater.heater_id = 99
        source.is_sensor = True
        source.sensor.sensor_id = 42
        source.has_cryocooler = True
        source.Grad_W_K = 3.0
        cloned = clone_node_for_extrusion(source, 11, (1, 0, 0))
        self.assertEqual(cloned.node_id, 11)
        self.assertEqual(cloned.coord, (1, 0, 0))
        self.assertEqual(cloned.material, source.material)
        self.assertEqual(cloned.side_length_m, source.side_length_m)
        self.assertEqual(cloned.Grad_W_K, 0.0)
        self.assertFalse(cloned.is_heater)
        self.assertEqual(cloned.heater.heater_id, 11)
        self.assertFalse(cloned.is_sensor)
        self.assertEqual(cloned.sensor.sensor_id, 11)
        self.assertFalse(cloned.has_cryocooler)

    def test_draw_node_ids_start_after_max_existing_id(self) -> None:
        self.assertEqual(next_node_id([0, 1, 2, 10]), 11)
        self.assertEqual(next_node_id([]), 0)

    def test_tooltip_formatters_include_core_node_and_edge_fields(self) -> None:
        node = NodeProperties.with_material(7, (2, 0, 1), material="copper")
        node.is_exposed = True
        node.radiating_area_m2 = 3.0e-6
        node.G_rad_W_K = 1.2e-4
        node.R_rad_K_W = 8333.3
        node.is_heater = True
        node.heater.heater_id = 7
        node.heater.heater_min_power_W = 0.0
        node.heater.heater_max_power_W = 12.0
        node.assigned_sensor_id = 7
        node.is_sensor = True
        node.sensor.sensor_id = 7
        node.assigned_heater_id = 7
        node.sensor_control_mode = "mimo"
        node.controller_setpoint_K = 310.0
        node.controller_lambda_order = 0.7
        node.controller_mu_order = 0.4
        node.has_cryocooler = True
        node_text = format_node_tooltip(7, node)
        self.assertIn("Node 7", node_text)
        self.assertIn("coord: (2, 0, 1)", node_text)
        self.assertIn("material: copper", node_text)
        self.assertIn("rho:", node_text)
        self.assertIn("cp:", node_text)
        self.assertIn("k: 401.000 W/m/K", node_text)
        self.assertIn("radiating area:", node_text)
        self.assertIn("G_rad:", node_text)
        self.assertIn("R_rad:", node_text)
        self.assertIn("heater min:", node_text)
        self.assertIn("assigned sensor: 7", node_text)
        self.assertIn("control mode: mimo", node_text)
        self.assertIn("MIMO lambda: 0.700", node_text)
        self.assertIn("MIMO mu: 0.400", node_text)
        self.assertIn("cryocooler: yes", node_text)
        self.assertIn("sensor_id: 7", node_text)

        edge_text = format_edge_tooltip(
            3, 7, {"Gij_W_K": 1.42e-2, "source_metadata": "auto"}
        )
        self.assertIn("Edge 3 -- 7", edge_text)
        self.assertIn("Gij:", edge_text)
        self.assertIn("mode: auto", edge_text)

    def test_role_warning_reasons_flag_unconnected_and_unpaired_roles(self) -> None:
        heater = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        heater.is_heater = True
        heater.heater_valid = False
        heater.heater_warning = "No body power deposition nodes found."

        sensor = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        sensor.is_sensor = True
        sensor.sensor_valid = False
        sensor.sensor_connected_node_ids = []

        paired_heater = NodeProperties.with_material(3, (2, 0, 0), material="copper")
        paired_heater.is_heater = True
        paired_heater.assigned_sensor_id = 2
        paired_heater.power_deposition_node_ids = [4]

        self.assertTrue(has_role_warning(heater))
        self.assertIn("No body power deposition nodes found.", role_warning_reasons(heater))
        self.assertIn("heater has no assigned sensor", role_warning_reasons(heater))
        self.assertTrue(has_role_warning(sensor))
        self.assertIn("sensor has no connected body readout nodes", role_warning_reasons(sensor))
        self.assertFalse(has_role_warning(paired_heater))

    def test_tooltip_includes_role_warning_reasons(self) -> None:
        heater = NodeProperties.with_material(9, (0, 0, 0), material="copper")
        heater.is_heater = True
        heater.heater_valid = False
        heater.heater_warning = "No body power deposition nodes found."

        text = format_node_tooltip(9, heater)

        self.assertIn("-- role warnings --", text)
        self.assertIn("No body power deposition nodes found.", text)
        self.assertIn("heater has no assigned sensor", text)

    def test_stepper_comparison_builds_elementwise_error_matrix(self) -> None:
        from graph_visualizer.simulation_diagnostics import compare_implicit_cpu_to_expm_multiply

        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="stepper_compare"))
        left = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        right = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        left.C_J_K = right.C_J_K = 10.0
        left.initial_temperature_K = 310.0
        right.initial_temperature_K = 290.0
        model.add_node(left)
        model.add_node(right)
        L = csr_matrix(np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=float))
        matrices = {
            "node_ids": np.array([1, 2], dtype=int),
            "C": np.array([10.0, 10.0], dtype=float),
            "L": L,
            "G_rad": np.zeros(2, dtype=float),
        }

        result = compare_implicit_cpu_to_expm_multiply(
            model,
            matrices,
            SimulationParameters(dt_s=1.0, use_ambient_radiation=False),
            steps=3,
        )

        self.assertEqual(result.implicit_temperature_K.shape, (4, 2))
        self.assertEqual(result.reference_temperature_K.shape, (4, 2))
        self.assertEqual(result.error_K.shape, (4, 2))
        self.assertEqual(result.metrics.steps, 3)
        self.assertEqual(result.metrics.node_count, 2)
        self.assertEqual(result.metrics.implicit_stepper, "implicit_sparse_cpu")
        self.assertEqual(result.metrics.reference_stepper, "expm_multiply")
        self.assertGreaterEqual(result.metrics.max_abs_error_K, 0.0)
        self.assertTrue(np.allclose(result.error_K, result.implicit_temperature_K - result.reference_temperature_K))

    def test_stepper_comparison_saves_artifacts(self) -> None:
        from graph_visualizer.simulation_diagnostics import (
            compare_implicit_cpu_to_expm_multiply,
            save_stepper_comparison,
        )

        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="stepper_compare_save"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        model.add_node(node)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0], dtype=float),
            "L": csr_matrix(np.array([[1.0]], dtype=float)),
            "G_rad": np.zeros(1, dtype=float),
        }
        result = compare_implicit_cpu_to_expm_multiply(
            model,
            matrices,
            SimulationParameters(dt_s=1.0, use_ambient_radiation=False),
            steps=2,
        )

        with TemporaryDirectory() as directory:
            output_dir = save_stepper_comparison(result, Path(directory))

            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "implicit_temperature_K.npy").exists())
            self.assertTrue((output_dir / "reference_temperature_K.npy").exists())
            self.assertTrue((output_dir / "temperature_error_K.npy").exists())
            saved_error = np.load(output_dir / "temperature_error_K.npy")

        self.assertTrue(np.allclose(saved_error, result.error_K))

    def test_stepper_diagnostic_worker_compares_current_state_to_reference(self) -> None:
        import sys
        import types

        heat_tab_module_name = "graph_visualizer.heat_transfer_simulation_tab"
        pyvista_module_name = "graph_visualizer.pyvista_widget"
        qtpy_module_name = "qtpy"
        previous_heat_tab = sys.modules.pop(heat_tab_module_name, None)
        previous_pyvista = sys.modules.get(pyvista_module_name)
        previous_qtpy = sys.modules.get(qtpy_module_name)
        pyvista_stub = types.ModuleType(pyvista_module_name)
        pyvista_stub.GraphPyVistaWidget = object
        sys.modules[pyvista_module_name] = pyvista_stub
        qtpy_stub = types.ModuleType(qtpy_module_name)
        qtpy_stub.QtGui = types.SimpleNamespace()
        sys.modules[qtpy_module_name] = qtpy_stub
        try:
            from graph_visualizer.heat_transfer_simulation_tab import (
                _format_stepper_diagnostic_summary,
                _run_stepper_diagnostic_worker,
            )
        finally:
            sys.modules.pop(heat_tab_module_name, None)
            if previous_heat_tab is not None:
                sys.modules[heat_tab_module_name] = previous_heat_tab
            if previous_pyvista is not None:
                sys.modules[pyvista_module_name] = previous_pyvista
            else:
                sys.modules.pop(pyvista_module_name, None)
            if previous_qtpy is not None:
                sys.modules[qtpy_module_name] = previous_qtpy
            else:
                sys.modules.pop(qtpy_module_name, None)

        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="stepper_compare_worker"))
        left = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        right = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        left.C_J_K = right.C_J_K = 10.0
        left.initial_temperature_K = 310.0
        right.initial_temperature_K = 290.0
        model.add_node(left)
        model.add_node(right)
        matrices = {
            "node_ids": np.array([1, 2], dtype=int),
            "C": np.array([10.0, 10.0], dtype=float),
            "L": csr_matrix(np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=float)),
            "G_rad": np.zeros(2, dtype=float),
        }
        params = SimulationParameters(dt_s=1.0, use_ambient_radiation=False)
        prepared = prepare_simulation(model, matrices, params)
        prepared.gpu_stepper = None
        prepared.fast_sparse_substeps = None
        prepared.step_forward()

        with TemporaryDirectory() as directory:
            payload = _run_stepper_diagnostic_worker(
                model,
                matrices,
                params,
                prepared.node_ids.copy(),
                prepared.initial_temperatures_K.copy(),
                prepared.temperatures_K.copy(),
                prepared.time_s,
                "implicit_sparse_cpu",
                float(prepared.last_step_profile_ms.get("total_ms", 0.0)) / 1000.0,
                dict(prepared.last_step_profile_ms),
                Path(directory) / "diagnostic",
            )
            summary = _format_stepper_diagnostic_summary(payload)

            self.assertTrue((Path(directory) / "diagnostic" / "summary.json").exists())
            self.assertTrue((Path(directory) / "diagnostic" / "current_temperature_K.npy").exists())

        self.assertEqual(payload["mode"], "current_state")
        self.assertEqual(payload["metrics"]["steps"], 1)
        self.assertEqual(payload["metrics"]["implicit_stepper"], "implicit_sparse_cpu")
        self.assertEqual(payload["metrics"]["reference_stepper"], "expm_multiply")
        self.assertIn("max abs error", summary)
        self.assertIn("current time", summary)
        self.assertIn("saved:", summary)

    def test_2d_position_expansion_separates_close_nodes(self) -> None:
        positions = {1: (0.0, 0.0), 2: (0.01, 0.0), 3: (2.0, 0.0)}
        expanded = expand_positions(positions, minimum_distance=0.4, iterations=20)
        first = np.array(expanded[1])
        second = np.array(expanded[2])
        self.assertGreaterEqual(np.linalg.norm(first - second), 0.39)

    def test_2d_edge_curve_detects_unrelated_node_under_edge(self) -> None:
        positions = {1: (0.0, 0.0), 2: (2.0, 0.0), 3: (1.0, 0.05)}
        self.assertNotEqual(edge_curve_for_positions(1, 2, positions, 0.2), 0.0)
        clear_positions = {1: (0.0, 0.0), 2: (2.0, 0.0), 3: (1.0, 1.0)}
        self.assertEqual(edge_curve_for_positions(1, 2, clear_positions, 0.2), 0.0)



if __name__ == "__main__":
    unittest.main()
