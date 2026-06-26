"""Tests for sparse thermal graph data and matrix construction."""

from __future__ import annotations

import json
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
    _atomic_write_json,
    load_conductance_matrix_from_folder,
    load_graph_folder,
    save_graph_folder,
)
from graph_visualizer.material_library import default_material_library
from graph_visualizer.mimo_controller import (
    apply_cumulative_coupling_cutoff,
    compute_decoupling_matrix,
    update_nonnegative_integrator,
    weighted_rms_error,
)
from graph_visualizer.matrix_builder import (
    build_matrices,
    estimate_conductance,
    exposed_areas_from_geometry_m2,
    refresh_auto_edges,
    refresh_radiation_from_exposed_faces,
)
from graph_visualizer.models import EdgeMode, GraphMetadata, NodeProperties, ThermalGraphModel
from graph_visualizer.simulation_model import prepare_simulation
from graph_visualizer.simulation_parameters import (
    SimulationParameters,
    apply_initial_temperature_parameter_payload,
    initial_temperature_parameter_payload,
)
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
        self.assertAlmostEqual(loaded_matrices["G"][0, 1], 0.401)
        self.assertTrue(np.any(np.abs(loaded_matrices["L"]) > 0.0))

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
        self.assertAlmostEqual(loaded_model.edges[(1, 2)].Gij_W_K, 0.401)
        self.assertAlmostEqual(loaded_matrices["G"][0, 1], 0.401)

    def test_affine_simulation_uses_radiation_toggle(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="radiation_graph"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.G_rad_W_K = 2.0
        model.add_node(node)
        matrices = {
            "node_ids": np.array([1], dtype=int),
            "C": np.array([10.0]),
            "L": np.zeros((1, 1)),
            "G_rad": np.array([2.0]),
            "initial_temperature_K": np.array([300.0]),
        }
        params = SimulationParameters(dt_s=1.0, T_env_K=290.0, use_ambient_radiation=True)
        prepared = prepare_simulation(model, matrices, params)
        prepared.step_forward()
        self.assertLess(prepared.temperatures_K[0], 300.0)

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
        node.has_heater = True
        node.has_sensor = True
        node.heater.heater_max_power_W = 20.0
        node.heater_control.manual.power = 5.0
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
        G = np.array([[4.0, 1.0], [0.1, 2.0]], dtype=float)
        cutoff = apply_cumulative_coupling_cutoff(G, 0.8)

        self.assertEqual(cutoff[0, 1], 0.0)
        self.assertEqual(cutoff[1, 0], 0.0)
        self.assertAlmostEqual(weighted_rms_error(np.array([2.0, 4.0]), np.array([1.0, 3.0])), np.sqrt(13.0))
        self.assertTrue(
            np.allclose(
                update_nonnegative_integrator(np.array([1.0, 1.0]), np.array([2.0, -5.0]), 1.0),
                np.array([3.0, 0.0]),
            )
        )

        result = compute_decoupling_matrix(np.array([[2.0]], dtype=float), np.array([1.0]), 0.0, 1.0)

        self.assertEqual(result.D.shape, (1, 1))
        self.assertAlmostEqual(float(result.D[0, 0]), 0.5)

    def test_mimo_controller_drives_heater_from_gain_matrix(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_controller"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.has_heater = True
        node.has_sensor = True
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
            mimo_decoupling_lambda=0.0,
            use_ambient_radiation=False,
        )

        prepared = prepare_simulation(model, matrices, params)

        self.assertEqual(prepared.heater_power_by_node(), {1: 5.0})
        prepared.step_forward()
        self.assertGreater(float(prepared.temperatures_K[0]), 300.0)
        self.assertEqual(prepared.controller_mode, "coarse")

    def test_mimo_controller_integrator_updates_per_sensor(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_integrator"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.has_heater = True
        node.has_sensor = True
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
            mimo_decoupling_lambda=0.0,
            use_ambient_radiation=False,
        )

        prepared = prepare_simulation(model, matrices, params)
        prepared.step_forward()

        self.assertAlmostEqual(prepared.controller_integrators[1], 10.0)
        self.assertEqual(prepared.heater_power_by_node(), {1: 10.0})

    def test_disabled_mimo_controller_does_not_use_manual_fallback_power_for_sys_id(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_sys_id_baseline"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.has_heater = True
        node.has_sensor = True
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
                mimo_decoupling_lambda=0.0,
                use_ambient_radiation=False,
            ),
        )

        self.assertEqual(prepared.heater_power_by_node(), {1: 10.0})
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

    def test_mimo_controller_metadata_and_gain_matrix_round_trip(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="mimo_round_trip"))
        sensor = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        sensor.has_sensor = True
        sensor.controller_setpoint_K = 315.0
        sensor.controller_weight = 2.0
        sensor.sensor_settling_time_s = 8.0
        sensor.controller_kp_coarse = 0.75
        sensor.controller_ki_coarse = 0.25
        sensor.controller_kp_hold = 0.5
        sensor.controller_ki_hold = 0.125
        heater = NodeProperties.with_material(2, (1, 0, 0), material="copper")
        heater.has_heater = True
        heater.has_sensor = True
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
        cold.has_heater = True
        cold.has_sensor = True
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
        node.has_heater = True
        node.has_sensor = True
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

    def test_heater_actuator_power_excludes_cryocooler_power(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="actuator_power_only"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 300.0
        node.has_heater = True
        node.has_sensor = True
        node.has_cryocooler = True
        node.heater_control.manual.power = 3.0
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
        node.has_heater = True
        node.has_sensor = True
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

        self.assertEqual(preview_power, {1: 4.0})
        self.assertEqual(model.nodes[1].heater_control.pid_state.integral, 0.0)
        self.assertIsNone(model.nodes[1].heater_control.pid_state.previous_error)

        prepared.step_forward()
        temperature_after_step = float(prepared.temperatures_K[0])
        self.assertGreater(temperature_after_step, 300.0)
        self.assertLess(temperature_after_step, 301.0)
        self.assertIsNotNone(model.nodes[1].heater_control.pid_state.previous_error)

        prepared.reset()

        self.assertEqual(model.nodes[1].heater_control.pid_state.integral, 0.0)
        self.assertIsNone(model.nodes[1].heater_control.pid_state.previous_error)

    def test_pid_integral_only_accumulates_below_setpoint(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="pid_integral_desaturation"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 310.0
        node.has_heater = True
        node.has_sensor = True
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

    def test_pid_integral_deintegrates_above_setpoint_without_heating(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="pid_above_setpoint"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 310.0
        node.has_heater = True
        node.has_sensor = True
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
        self.assertEqual(model.nodes[1].heater_control.pid_state.integral, 90.0)

        model.nodes[1].heater_control.pid_state.integral = 5.0
        prepared.step_forward()

        self.assertEqual(model.nodes[1].heater_control.pid_state.integral, 0.0)

    def test_pid_integral_leak_decays_stored_integral(self) -> None:
        model = ThermalGraphModel(metadata=GraphMetadata(graph_name="pid_leaky_integrator"))
        node = NodeProperties.with_material(1, (0, 0, 0), material="copper")
        node.C_J_K = 10.0
        node.initial_temperature_K = 290.0
        node.has_heater = True
        node.has_sensor = True
        node.heater.heater_max_power_W = 20.0
        node.heater_control.mode = "pid"
        node.heater_control.pid.ki = 0.0
        node.heater_control.pid.integral_leak_per_s = float(np.log(2.0))
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

        self.assertAlmostEqual(model.nodes[1].heater_control.pid_state.integral, 60.0)

        prepared.z[0] = 310.0
        model.nodes[1].heater_control.pid_state.integral = 100.0
        prepared.step_forward()

        self.assertAlmostEqual(model.nodes[1].heater_control.pid_state.integral, 40.0)

    def test_loaded_legacy_heater_implies_sensor_and_default_control(self) -> None:
        node = NodeProperties.from_dict(
            {
                "node_id": 3,
                "coord": [0, 0, 0],
                "has_heater": True,
                "heater": {"heater_id": 3, "heater_max_power_W": 12.0, "heater_efficiency": 0.5},
                "heater_control": {"pid": {"integral_leak_per_s": 0.25}},
            }
        )

        self.assertTrue(node.has_sensor)
        self.assertEqual(node.heater_control.mode, "manual")
        self.assertAlmostEqual(node.heater_control.manual.power, 6.0)
        self.assertAlmostEqual(node.heater_control.pid.integral_leak_per_s, 0.25)

    def test_large_simulation_uses_incremental_expm_multiply_stepper(self) -> None:
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
        source.has_heater = True
        source.heater.heater_id = 99
        source.has_sensor = True
        source.sensor.sensor_id = 42
        source.has_cryocooler = True
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
        node.has_heater = True
        node.heater.heater_id = 7
        node.heater.heater_min_power_W = 0.0
        node.heater.heater_max_power_W = 12.0
        node.heater_control.mode = "pid"
        node.heater_control.pid.kp = 1.0
        node.heater_control.pid.ki = 2.0
        node.heater_control.pid.kd = 3.0
        node.heater_control.pid.setpoint = 310.0
        node.heater_control.pid.integral_leak_per_s = 0.25
        node.has_sensor = True
        node.sensor.sensor_id = 7
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
        self.assertIn("control mode: pid", node_text)
        self.assertIn("PID leak: 0.250 1/s", node_text)
        self.assertIn("cryocooler: yes", node_text)
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
