"""Tests for sparse thermal graph data and matrix construction."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from graph_visualizer.draw_tools import (
    clone_node_for_extrusion,
    compute_face_normal,
    extrusion_count_from_projected_pixel_drag,
    next_node_id,
    preview_coords,
)
from graph_visualizer.graph_io import (
    load_conductance_matrix_from_folder,
    load_graph_folder,
    save_graph_folder,
)
from graph_visualizer.matrix_builder import refresh_auto_edges
from graph_visualizer.models import EdgeMode, GraphMetadata, NodeProperties, ThermalGraphModel
from graph_visualizer.tooltip_formatters import format_edge_tooltip, format_node_tooltip
from graph_visualizer.two_d_graph_widget import (
    edge_curve_for_positions,
    expand_positions,
)


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

    def test_auto_edges_use_six_neighbor_adjacency(self) -> None:
        model = self.make_model()
        self.assertEqual(set(model.edges), {(2, 7), (2, 10)})
        self.assertTrue(all(edge.Gij_W_K > 0.0 for edge in model.edges.values()))

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
            matrices = save_graph_folder(model, directory)
            self.assertEqual(matrices["node_ids"].tolist(), [2, 7, 10])
            loaded_model, loaded_matrices = load_graph_folder(Path(directory))
        self.assertEqual(loaded_model.ordered_node_ids(), [2, 7, 10])
        self.assertTrue(np.allclose(loaded_matrices["G"], loaded_matrices["G"].T))
        self.assertTrue(np.allclose(np.diag(loaded_matrices["L"]), loaded_matrices["G"].sum(axis=1)))

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
        source.has_heater = True
        source.heater.heater_id = 99
        source.has_sensor = True
        source.sensor.sensor_id = 42
        source.Grad_W_K = 3.0
        cloned = clone_node_for_extrusion(source, 11, (1, 0, 0))
        self.assertEqual(cloned.node_id, 11)
        self.assertEqual(cloned.coord, (1, 0, 0))
        self.assertEqual(cloned.material, source.material)
        self.assertEqual(cloned.side_length_m, source.side_length_m)
        self.assertEqual(cloned.Grad_W_K, 0.0)
        self.assertFalse(cloned.has_heater)
        self.assertEqual(cloned.heater.heater_id, 11)
        self.assertFalse(cloned.has_sensor)
        self.assertEqual(cloned.sensor.sensor_id, 11)

    def test_draw_node_ids_start_after_max_existing_id(self) -> None:
        self.assertEqual(next_node_id([0, 1, 2, 10]), 11)
        self.assertEqual(next_node_id([]), 0)

    def test_tooltip_formatters_include_core_node_and_edge_fields(self) -> None:
        node = NodeProperties.with_material(7, (2, 0, 1), material="copper")
        node.has_sensor = True
        node.sensor.sensor_id = 7
        node_text = format_node_tooltip(7, node)
        self.assertIn("Node 7", node_text)
        self.assertIn("coord: (2, 0, 1)", node_text)
        self.assertIn("material: copper", node_text)
        self.assertIn("sensor_id: 7", node_text)

        edge_text = format_edge_tooltip(
            3, 7, {"Gij_W_K": 1.42e-2, "source_metadata": "auto"}
        )
        self.assertIn("Edge 3 -- 7", edge_text)
        self.assertIn("Gij:", edge_text)
        self.assertIn("mode: auto", edge_text)

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
