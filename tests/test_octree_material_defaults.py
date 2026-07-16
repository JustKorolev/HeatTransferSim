"""Tests for octree material defaulting behavior."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import issparse

import octree_graph.validation as octree_validation
import octree_graph.matrix_builder as octree_matrix_builder
from octree_graph.connectivity_cli import main as connectivity_main
from octree_graph.cli import (
    BuildCheckpointer,
    _annotate_graph_warning_tags,
    _build_quality_report,
    _graph_connectivity_analysis,
    build_parser,
    _build_graph_with_optional_fallback,
    _filter_ignored_components,
    _log_scene_memory_risk,
    _raise_if_empty_graph,
    _resolve_material_lookup_path,
    _split_role_components,
    _write_outputs,
)
from octree_graph.load_contact_report import ContactReport
from octree_graph.graph_builder import (
    DEFAULT_CONTACT_INTERFACE_CONDUCTANCE_W_M2K,
    DEFAULT_ROLE_EXCLUDE_NAME_PATTERNS,
    RoleComponent,
    _candidate_cell_pairs,
    contact_conductance_W_K,
    _exposed_areas_m2,
    build_graph,
    collapse_role_components,
)
from octree_graph.load_gltf import GltfScene, MeshObject
from octree_graph.load_contact_report import load_contact_report
from octree_graph.matrix_builder import build_matrices as build_octree_matrices
from octree_graph.materials import (
    DEFAULT_ASSIGNED_MATERIAL_NAME,
    Material,
    infer_material_name_from_text,
    resolve_material,
)
from octree_graph.octree import (
    OctreeCell,
    CellClassification,
    OctreeDiagnostics,
    OctreeParams,
    _adjacent_balance_refinement_targets,
    _mesh_contains_point,
    _mesh_triangles,
    _needs_gap_preservation_refinement,
    _objects_intersecting_bounds,
    _physical_material_name,
    _refinement_priority,
    _sample_points,
    _triangle_intersects_aabb,
    TriangleSpatialIndex,
    build_octree,
)
from octree_graph.validation import validate_graph


def _mesh_object(
    name: str,
    material_name: str | None,
    bounds_min: list[float],
    bounds_max: list[float],
) -> MeshObject:
    bounds_min_array = np.asarray(bounds_min, dtype=float)
    bounds_max_array = np.asarray(bounds_max, dtype=float)
    size = np.maximum(bounds_max_array - bounds_min_array, 0.0)
    mesh = SimpleNamespace(
        vertices=np.empty((0, 3)),
        faces=np.empty((0, 3), dtype=int),
        triangles=np.empty((0, 3, 3)),
        is_watertight=False,
        volume=float(np.prod(np.maximum(size, 1.0))),
    )
    return MeshObject(
        name=name,
        material_name=material_name,
        mesh=mesh,
        vertices_mm=np.empty((0, 3)),
        bounds_mm=(bounds_min_array, bounds_max_array),
        watertight=False,
        scene_path=name,
    )


def _pair_ids(a: OctreeCell, b: OctreeCell) -> tuple[str, str]:
    return tuple(sorted((a.cell_id, b.cell_id)))


class OctreeMaterialDefaultTests(unittest.TestCase):
    def make_materials(self) -> dict[str, Material]:
        return {
            DEFAULT_ASSIGNED_MATERIAL_NAME: Material(
                name=DEFAULT_ASSIGNED_MATERIAL_NAME,
                density_kg_m3=2700.0,
                cp_J_kgK=896.0,
                k_W_mK=167.0,
                emissivity=0.09,
            ),
            "Copper": Material(
                name="Copper",
                density_kg_m3=8960.0,
                cp_J_kgK=385.0,
                k_W_mK=401.0,
                emissivity=0.35,
            ),
            "Not assigned": Material(
                name="Not assigned",
                density_kg_m3=2200.0,
                cp_J_kgK=800.0,
                k_W_mK=2.0,
                emissivity=0.85,
            ),
        }

    def test_resolve_material_maps_unassigned_to_aluminum(self) -> None:
        warnings: list[str] = []
        material = resolve_material("Not assigned", self.make_materials(), warnings)
        self.assertEqual(material.name, DEFAULT_ASSIGNED_MATERIAL_NAME)

    def test_inter_component_contact_uses_bulk_plus_interface_resistance(self) -> None:
        materials = self.make_materials()
        copper = materials["Copper"]
        aluminum = materials[DEFAULT_ASSIGNED_MATERIAL_NAME]
        conductance = contact_conductance_W_K(
            copper,
            aluminum,
            area_mm2=100.0,
            distance_mm=10.0,
            interface_conductance_W_m2K=1.0e4,
        )

        area_m2 = 100.0e-6
        half_distance_m = 0.010 * 0.5
        expected = 1.0 / (
            half_distance_m / (copper.k_W_mK * area_m2)
            + 1.0 / (1.0e4 * area_m2)
            + half_distance_m / (aluminum.k_W_mK * area_m2)
        )
        self.assertAlmostEqual(conductance, expected)

    def test_graph_build_uncertain_contact_is_not_fixed_fallback_conductance(self) -> None:
        materials = self.make_materials()
        leaves = [
            OctreeCell(
                cell_id="cell_1",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(0.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"copper_part": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="copper_part",
                dominant_material="Copper",
                confidence="high",
            ),
            OctreeCell(
                cell_id="cell_2",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(10.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"aluminum_part": 1.0},
                material_fractions={DEFAULT_ASSIGNED_MATERIAL_NAME: 1.0},
                dominant_component="aluminum_part",
                dominant_material=DEFAULT_ASSIGNED_MATERIAL_NAME,
                confidence="high",
            ),
        ]

        result = build_graph(
            leaves,
            ContactReport(),
            materials,
            warnings=[],
            contact_interface_conductance_W_m2K=1.0e4,
        )

        self.assertEqual(len(result.edges), 1)
        edge = result.edges[0]
        self.assertEqual(edge["edge_type"], "uncertain_contact")
        self.assertNotEqual(edge["G_W_K"], 0.1)
        expected = contact_conductance_W_K(
            materials["Copper"],
            materials[DEFAULT_ASSIGNED_MATERIAL_NAME],
            area_mm2=100.0,
            distance_mm=10.0,
            interface_conductance_W_m2K=1.0e4,
        )
        self.assertAlmostEqual(edge["G_W_K"], expected)

    def test_validate_graph_skips_oversized_dense_symmetry_check(self) -> None:
        nodes = [
            {
                "node_id": 1,
                "component_name": "A",
                "material_name": "Copper",
                "C_J_K": 1.0,
            },
            {
                "node_id": 2,
                "component_name": "B",
                "material_name": "Copper",
                "C_J_K": 1.0,
            },
        ]
        matrices = {
            "C": np.ones(2, dtype=float),
            "G": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
            "L": np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float),
        }
        old_limit = octree_validation._DENSE_SYMMETRY_CHECK_MAX_BYTES
        octree_validation._DENSE_SYMMETRY_CHECK_MAX_BYTES = 1
        try:
            with patch("octree_graph.validation.np.allclose", side_effect=AssertionError):
                errors, warnings = validate_graph(
                    {"graph_nodes": nodes, "graph_edges": [], "warnings": []},
                    matrices,
                )
        finally:
            octree_validation._DENSE_SYMMETRY_CHECK_MAX_BYTES = old_limit

        self.assertEqual(errors, [])
        self.assertTrue(any("Skipped dense G symmetry validation" in warning for warning in warnings))
        self.assertTrue(any("Skipped dense L symmetry validation" in warning for warning in warnings))

    def test_octree_matrix_builder_forces_sparse_when_dense_byte_budget_exceeded(self) -> None:
        nodes = [
            {"node_id": 1, "C_J_K": 1.0},
            {"node_id": 2, "C_J_K": 1.0},
        ]
        edges = [{"node_i": 1, "node_j": 2, "G_W_K": 0.5}]
        old_limit = octree_matrix_builder.DENSE_MATRIX_MAX_TOTAL_BYTES
        octree_matrix_builder.DENSE_MATRIX_MAX_TOTAL_BYTES = 1
        try:
            matrices = build_octree_matrices(nodes, edges, dense_node_limit=100)
        finally:
            octree_matrix_builder.DENSE_MATRIX_MAX_TOTAL_BYTES = old_limit

        self.assertNotIn("G", matrices)
        self.assertTrue(issparse(matrices["L"]))
        self.assertAlmostEqual(float(matrices["L"][0, 0]), 0.5)

    def test_resolve_material_maps_unknown_material_placeholder_to_aluminum(self) -> None:
        warnings: list[str] = []
        material = resolve_material("unknown material", self.make_materials(), warnings)
        self.assertEqual(material.name, DEFAULT_ASSIGNED_MATERIAL_NAME)

    def test_physical_material_uses_aluminum_for_unknown_mesh_and_report_materials(self) -> None:
        materials = self.make_materials()
        known = set(materials)
        report = ContactReport(component_materials={"part_1": "Not assigned"})
        obj = SimpleNamespace(name="part_1", material_name="missing_material")

        self.assertEqual(
            _physical_material_name(obj, report, known),
            DEFAULT_ASSIGNED_MATERIAL_NAME,
        )

    def test_physical_material_preserves_known_assigned_material(self) -> None:
        materials = self.make_materials()
        report = ContactReport()
        obj = SimpleNamespace(name="part_1", material_name="Copper")

        self.assertEqual(_physical_material_name(obj, report, set(materials)), "Copper")

    def test_infers_material_from_solidworks_component_tokens(self) -> None:
        materials = self.make_materials()
        materials["18-8 Stainless Steel"] = Material(
            name="18-8 Stainless Steel",
            density_kg_m3=8000.0,
            cp_J_kgK=500.0,
            k_W_mK=16.2,
            emissivity=0.35,
        )
        materials["AISI 304 Stainless Steel"] = Material(
            name="AISI 304 Stainless Steel",
            density_kg_m3=8000.0,
            cp_J_kgK=500.0,
            k_W_mK=16.2,
            emissivity=0.35,
        )
        materials["17-7PH Stainless Steel"] = Material(
            name="17-7PH Stainless Steel",
            density_kg_m3=7800.0,
            cp_J_kgK=460.0,
            k_W_mK=16.0,
            emissivity=0.35,
        )

        self.assertEqual(
            infer_material_name_from_text("V_MMC_METRIC_SHCS_18-8SS_1530", materials),
            "18-8 Stainless Steel",
        )
        self.assertEqual(
            infer_material_name_from_text("V_MISUMI_METRIC_DISC SPRING_304SS_483", materials),
            "AISI 304 Stainless Steel",
        )
        self.assertEqual(
            infer_material_name_from_text("SOME_PART_17-7PH_22", materials),
            "17-7PH Stainless Steel",
        )

    def test_physical_material_infers_from_component_name_before_aluminum_fallback(self) -> None:
        materials = self.make_materials()
        materials["18-8 Stainless Steel"] = Material(
            name="18-8 Stainless Steel",
            density_kg_m3=8000.0,
            cp_J_kgK=500.0,
            k_W_mK=16.2,
            emissivity=0.35,
        )
        obj = SimpleNamespace(name="V_MMC_METRIC_SHCS_18-8SS_1530", material_name=None)

        self.assertEqual(
            _physical_material_name(obj, ContactReport(), set(materials)),
            "18-8 Stainless Steel",
        )

    def test_missing_contact_report_loads_empty_report_without_excel_dependency(self) -> None:
        report = load_contact_report(None)

        self.assertEqual(report.component_materials, {})

    def test_macro_style_material_lookup_loads_part_and_material_names_only(self) -> None:
        try:
            from openpyxl import Workbook
        except ImportError:
            self.skipTest("openpyxl is required for Excel material-lookup tests")

        with TemporaryDirectory() as directory:
            path = Path(directory) / "macro_material_lookup.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Materials"
            sheet.append(["Part Name", "Material Name"])
            sheet.append(["PartA-1", "Copper"])
            workbook.save(path)
            workbook.close()

            report = load_contact_report(path)

        self.assertEqual(report.component_materials["PartA-1"], "Copper")
        self.assertEqual(report.component_materials["PartA"], "Copper")
        self.assertEqual(report.warnings, [])

    def test_cli_uses_materials_xlsx_in_mesh_directory_when_present(self) -> None:
        with TemporaryDirectory() as directory:
            mesh_dir = Path(directory)
            lookup = mesh_dir / "materials.xlsx"
            lookup.write_bytes(b"placeholder")

            self.assertEqual(_resolve_material_lookup_path(mesh_dir), lookup)

    def test_cli_material_lookup_is_optional_when_materials_xlsx_is_absent(self) -> None:
        with TemporaryDirectory() as directory:
            self.assertIsNone(_resolve_material_lookup_path(directory))

    def test_cli_memory_guard_disables_parallel_voxel_workers_when_payload_is_too_large(self) -> None:
        triangle = np.zeros((100, 3, 3), dtype=float)
        obj = _mesh_object("body_panel", "Copper", [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])
        obj.mesh.triangles = triangle
        scene = GltfScene(
            path=SimpleNamespace(),
            objects=[obj],
            bounds_mm=(np.array([-5.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0])),
            warnings=[],
        )
        args = SimpleNamespace(voxel_workers=2)
        warnings: list[str] = []
        logger = SimpleNamespace(messages=[], log=lambda message: logger.messages.append(message))

        with patch("octree_graph.cli._available_memory_bytes", return_value=1024):
            _log_scene_memory_risk(scene, args, logger, warnings)

        self.assertEqual(args.voxel_workers, 1)
        self.assertIn("Disabled multiprocessing", warnings[0])
        self.assertTrue(any("Scene memory estimate" in message for message in logger.messages))

    def test_build_quality_counts_warning_tags(self) -> None:
        args = SimpleNamespace(max_cell_size_mm=4.0)
        nodes = [
            {
                "node_id": 1,
                "size_mm": [8.0, 4.0, 4.0],
                "confidence": "low",
                "warnings": ["Accepted occupied cell above max_cell_size_mm."],
                "is_heater": True,
                "heater_valid": True,
                "assigned_sensor_id": None,
                "component_name": "heater",
                "material_name": "Copper",
            },
            {
                "node_id": 2,
                "size_mm": [2.0, 2.0, 2.0],
                "confidence": "high",
                "warnings": [],
                "is_sensor": True,
                "sensor_valid": False,
                "assigned_heater_ids": [],
                "component_name": "sensor",
                "material_name": "Copper",
            },
        ]
        edges = [{"node_i": 1, "node_j": 2}]

        _annotate_graph_warning_tags(nodes, edges, args)
        graph = {
            "graph_nodes": nodes,
            "graph_edges": edges,
            "warnings": ["top-level warning"],
            "validation_results": {"errors": [], "warnings": []},
        }
        quality = _build_quality_report(graph, args)

        self.assertIn("oversized_cell", nodes[0]["tags"]["warning_tags"])
        self.assertIn("unpaired_heater", nodes[0]["tags"]["warning_tags"])
        self.assertIn("invalid_sensor", nodes[1]["tags"]["warning_tags"])
        self.assertNotIn("warning_tags", nodes[0])
        self.assertEqual(quality["node_warning_tag_counts"]["oversized_cell"], 1)
        self.assertLess(quality["quality_score"], 100)

    def test_connectivity_analysis_tags_nodes_outside_largest_component(self) -> None:
        args = SimpleNamespace(max_cell_size_mm=10.0)
        nodes = [
            {"node_id": 1, "center_mm": [0.0, 0.0, 0.0], "size_mm": [1.0, 1.0, 1.0], "component_name": "main", "material_name": "Copper"},
            {"node_id": 2, "center_mm": [1.0, 0.0, 0.0], "size_mm": [1.0, 1.0, 1.0], "component_name": "main", "material_name": "Copper"},
            {"node_id": 3, "center_mm": [100.0, 0.0, 0.0], "size_mm": [1.0, 1.0, 1.0], "component_name": "island", "material_name": "Copper"},
        ]
        edges = [{"node_i": 1, "node_j": 2}]

        analysis = _graph_connectivity_analysis(nodes, edges)
        _annotate_graph_warning_tags(nodes, edges, args, analysis)
        graph = {
            "graph_nodes": nodes,
            "graph_edges": edges,
            "warnings": [],
            "validation_results": {"errors": [], "warnings": []},
            "connectivity_analysis": analysis,
        }
        quality = _build_quality_report(graph, args)

        self.assertFalse(analysis["connected"])
        self.assertEqual(analysis["component_count"], 2)
        self.assertEqual(analysis["disconnected_node_ids"], [3])
        self.assertIn("disconnected_component", nodes[2]["tags"]["warning_tags"])
        self.assertIn("isolated_node", nodes[2]["tags"]["warning_tags"])
        self.assertEqual(quality["node_warning_tag_counts"]["disconnected_component"], 1)

    def test_standalone_connectivity_cli_updates_existing_graph_folder(self) -> None:
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            graph = {
                "parameters": {"max_cell_size_mm": 10.0},
                "graph_nodes": [
                    {"node_id": 1, "center_mm": [0.0, 0.0, 0.0], "size_mm": [1.0, 1.0, 1.0], "component_name": "main", "material_name": "Copper"},
                    {"node_id": 2, "center_mm": [1.0, 0.0, 0.0], "size_mm": [1.0, 1.0, 1.0], "component_name": "main", "material_name": "Copper"},
                    {"node_id": 3, "center_mm": [100.0, 0.0, 0.0], "size_mm": [1.0, 1.0, 1.0], "component_name": "island", "material_name": "Copper"},
                ],
                "graph_edges": [{"node_i": 1, "node_j": 2}],
                "warnings": [],
                "validation_results": {"errors": [], "warnings": []},
            }
            (folder / "graph.json").write_text(json.dumps(graph), encoding="utf-8")

            connectivity_main(["--graph-folder", str(folder)])

            analysis = json.loads((folder / "connectivity_analysis.json").read_text(encoding="utf-8"))
            updated = json.loads((folder / "graph.json").read_text(encoding="utf-8"))
            self.assertFalse(analysis["connected"])
            self.assertEqual(analysis["disconnected_node_ids"], [3])
            self.assertIn("disconnected_component", updated["graph_nodes"][2]["tags"]["warning_tags"])
            self.assertTrue((folder / "build_quality.json").is_file())

    def test_build_checkpointer_writes_phase_and_completed_graph_files(self) -> None:
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(checkpoint_build=True, checkpoint_interval_s=1.0)
            checkpointer = BuildCheckpointer(Path(directory), args)
            diagnostics = OctreeDiagnostics()
            leaves = [
                OctreeCell(
                    cell_id="cell_1",
                    parent_id=None,
                    children_ids=[],
                    level=0,
                    center_mm=(0.0, 0.0, 0.0),
                    size_mm=(1.0, 1.0, 1.0),
                    occupancy={"body": 1.0},
                    material_fractions={"Copper": 1.0},
                    dominant_component="body",
                    dominant_material="Copper",
                    confidence="high",
                )
            ]

            checkpointer.phase("started", {"graph_name": "demo"})
            checkpointer.octree_complete(leaves, diagnostics)
            checkpointer.graph_complete(SimpleNamespace(nodes=[{"node_id": 1}], edges=[], warnings=[]))

            folder = Path(directory) / "build_checkpoints"
            self.assertTrue((folder / "latest.json").is_file())
            self.assertTrue((folder / "octree_leaves_complete.json").is_file())
            self.assertTrue((folder / "graph_complete.json").is_file())

    def test_graph_build_accepts_no_contact_report(self) -> None:
        materials = self.make_materials()
        leaves = [
            OctreeCell(
                cell_id="cell_1",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(0.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"part_1": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="part_1",
                dominant_material="Copper",
                confidence="high",
            )
        ]

        result = build_graph(leaves, None, materials, warnings=[])

        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(result.nodes[0]["component_name"], "part_1")
        self.assertFalse(result.nodes[0]["is_heater"])
        self.assertFalse(result.nodes[0]["is_sensor"])
        self.assertNotIn("heater", result.nodes[0]["tags"])
        self.assertNotIn("sensor", result.nodes[0]["tags"])
        self.assertGreater(result.nodes[0]["C_J_K"], 0.0)

    def test_role_component_detection_removes_heaters_and_sensors_from_body_objects(self) -> None:
        body = _mesh_object("body_panel", "Copper", [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])
        heater = _mesh_object("kapton_heater_1", "Copper", [5.0, -2.0, -2.0], [6.0, 2.0, 2.0])
        sensor = _mesh_object("assembly/temperature_probe_A", "Copper", [-6.0, -1.0, -1.0], [-5.0, 1.0, 1.0])

        body_objects, role_components = collapse_role_components(
            [body, heater, sensor],
            [r"kapton_heater"],
            [r"temperature_probe"],
        )

        self.assertEqual([obj.name for obj in body_objects], ["body_panel"])
        self.assertEqual([component.kind for component in role_components], ["heater", "sensor"])
        self.assertEqual({component.name for component in role_components}, {"assembly/temperature_probe_A", "kapton_heater_1"})

    def test_role_component_detection_excludes_cables_and_breakout_boards_by_default(self) -> None:
        flex = _mesh_object(
            "V_GUUTZ_SENSOR-HEATER-FLEX-CABLE_HISPEC_36",
            "Copper",
            [-20.0, -2.0, -1.0],
            [20.0, 2.0, 1.0],
        )
        breakout = _mesh_object(
            "Copy_of_FX23_100P_Male_Component_2_1^V_GUUTZ_EXTERNAL-SENSOR-HEATER-BREAKOUT-PCB_HISPEC",
            "Copper",
            [-10.0, -10.0, -1.0],
            [10.0, 10.0, 1.0],
        )
        heater = _mesh_object("kapton_heater_1", "Copper", [5.0, -2.0, -2.0], [6.0, 2.0, 2.0])

        body_objects, role_components = collapse_role_components(
            [flex, breakout, heater],
            [r"kapton_heater"],
            [r"sensor"],
            exclude_patterns=DEFAULT_ROLE_EXCLUDE_NAME_PATTERNS,
        )

        self.assertEqual([obj.name for obj in body_objects], [flex.name, breakout.name])
        self.assertEqual(len(role_components), 1)
        self.assertEqual(role_components[0].name, "kapton_heater_1")

    def test_role_component_detection_preserves_numeric_instance_identity(self) -> None:
        left = _mesh_object("safe_heater_1", "Copper", [0.0, 0.0, 0.0], [5.0, 5.0, 1.0])
        right = _mesh_object("safe_heater_2", "Copper", [100.0, 0.0, 0.0], [105.0, 5.0, 1.0])
        bridge = _mesh_object("safe_heater_3", "Copper", [108.0, 0.0, 0.0], [112.0, 5.0, 1.0])

        body_objects, role_components = collapse_role_components(
            [left, right, bridge],
            [r"safe_heater"],
            [],
            exclude_patterns=DEFAULT_ROLE_EXCLUDE_NAME_PATTERNS,
            group_gap_mm=10.0,
        )

        self.assertEqual(body_objects, [])
        self.assertEqual(
            [component.name for component in role_components],
            ["safe_heater_1", "safe_heater_2", "safe_heater_3"],
        )
        self.assertEqual([len(component.objects) for component in role_components], [1, 1, 1])

    def test_role_component_detection_merges_same_instance_fragment_suffixes(self) -> None:
        body = _mesh_object("body_panel", "Copper", [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])
        heater_mesh = _mesh_object("safe_heater_1_mesh", "Copper", [10.0, 0.0, 0.0], [15.0, 5.0, 1.0])
        heater_body = _mesh_object("safe_heater_1_body", "Copper", [15.001, 0.0, 0.0], [20.0, 5.0, 1.0])
        other_heater = _mesh_object("safe_heater_2", "Copper", [20.0, 0.0, 0.0], [25.0, 5.0, 1.0])

        body_objects, role_components = collapse_role_components(
            [body, heater_mesh, heater_body, other_heater],
            [r"safe_heater"],
            [],
            exclude_patterns=DEFAULT_ROLE_EXCLUDE_NAME_PATTERNS,
            group_gap_mm=0.01,
        )

        self.assertEqual([obj.name for obj in body_objects], ["body_panel"])
        self.assertEqual([component.name for component in role_components], ["safe_heater_1", "safe_heater_2"])
        self.assertEqual([len(component.objects) for component in role_components], [2, 1])

    def test_role_component_detection_uses_hierarchy_role_roots(self) -> None:
        body = _mesh_object("body_panel", "Copper", [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])
        heater_a_1 = _mesh_object("V_GUUTZ_SAFE-HEATER_HISPEC", "Copper", [10.0, 0.0, 0.0], [11.0, 1.0, 1.0])
        heater_a_2 = _mesh_object("V_GUUTZ_SAFE-HEATER_HISPEC", "Copper", [11.0, 0.0, 0.0], [12.0, 1.0, 1.0])
        heater_b = _mesh_object("V_GUUTZ_SAFE-HEATER_HISPEC", "Copper", [12.0, 0.0, 0.0], [13.0, 1.0, 1.0])
        heater_a_1.hierarchy_path = (
            "Default",
            "HISPEC-0030-A0005",
            "V_GUUTZ_SAFE-HEATER_HISPEC_1522",
            "leaf_1",
        )
        heater_a_2.hierarchy_path = (
            "Default",
            "HISPEC-0030-A0005",
            "V_GUUTZ_SAFE-HEATER_HISPEC_1522",
            "leaf_2",
        )
        heater_b.hierarchy_path = (
            "Default",
            "HISPEC-0030-A0005",
            "V_GUUTZ_SAFE-HEATER_HISPEC_1622",
            "leaf_1",
        )

        body_objects, role_components = collapse_role_components(
            [body, heater_a_1, heater_a_2, heater_b],
            [r"safe_heater"],
            [],
            exclude_patterns=DEFAULT_ROLE_EXCLUDE_NAME_PATTERNS,
            group_gap_mm=10.0,
        )

        self.assertEqual([obj.name for obj in body_objects], ["body_panel"])
        self.assertEqual(len(role_components), 2)
        self.assertEqual([len(component.objects) for component in role_components], [2, 1])
        self.assertTrue(all("V_GUUTZ_SAFE_HEATER_HISPEC" in component.name for component in role_components))

    def test_role_component_detection_uses_hierarchy_root_for_generic_leaf_names(self) -> None:
        heater_leaf = _mesh_object("solid_body", "Copper", [10.0, 0.0, 0.0], [11.0, 1.0, 1.0])
        heater_leaf.hierarchy_path = (
            "Default",
            "HISPEC-0030-A0005",
            "V_GUUTZ_SAFE-HEATER_HISPEC#1522",
            "solid_body",
        )

        body_objects, role_components = collapse_role_components(
            [heater_leaf],
            [r"safe_heater"],
            [r"temp_sensor"],
            group_gap_mm=10.0,
        )

        self.assertEqual(body_objects, [])
        self.assertEqual([(component.kind, component.name) for component in role_components], [
            ("heater", "V_GUUTZ_SAFE_HEATER_HISPEC#1522")
        ])

    def test_role_component_detection_does_not_spatially_split_hierarchy_groups(self) -> None:
        heater_leaf_a = _mesh_object("solid_body_a", "Copper", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        heater_leaf_b = _mesh_object("solid_body_b", "Copper", [1000.0, 0.0, 0.0], [1001.0, 1.0, 1.0])
        heater_leaf_a.hierarchy_path = (
            "Default",
            "HISPEC-0030-A0005",
            "V_GUUTZ_SAFE-HEATER_HISPEC#1522",
            "solid_body_a",
        )
        heater_leaf_b.hierarchy_path = (
            "Default",
            "HISPEC-0030-A0005",
            "V_GUUTZ_SAFE-HEATER_HISPEC#1522",
            "solid_body_b",
        )

        body_objects, role_components = collapse_role_components(
            [heater_leaf_a, heater_leaf_b],
            [r"safe_heater"],
            [],
            group_gap_mm=0.01,
        )

        self.assertEqual(body_objects, [])
        self.assertEqual(len(role_components), 1)
        self.assertEqual(role_components[0].name, "V_GUUTZ_SAFE_HEATER_HISPEC#1522")
        self.assertEqual(len(role_components[0].objects), 2)

    def test_role_component_detection_uses_scene_path_when_hierarchy_path_is_leaf_only(self) -> None:
        heater_leaf_a = _mesh_object("V_GUUTZ_SAFE-HEATER_HISPEC#1422", "Copper", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        heater_leaf_b = _mesh_object("V_GUUTZ_SAFE-HEATER_HISPEC#1424", "Copper", [1000.0, 0.0, 0.0], [1001.0, 1.0, 1.0])
        heater_leaf_a.hierarchy_path = ("V_GUUTZ_SAFE-HEATER_HISPEC#1422",)
        heater_leaf_b.hierarchy_path = ("V_GUUTZ_SAFE-HEATER_HISPEC#1424",)
        heater_leaf_a.scene_path = (
            "Default/HISPEC-0030-A0005/V_GUUTZ_SAFE-HEATER_HISPEC#1522/"
            "V_GUUTZ_SAFE-HEATER_HISPEC#1422"
        )
        heater_leaf_b.scene_path = (
            "Default/HISPEC-0030-A0005/V_GUUTZ_SAFE-HEATER_HISPEC#1522/"
            "V_GUUTZ_SAFE-HEATER_HISPEC#1424"
        )

        body_objects, role_components = collapse_role_components(
            [heater_leaf_a, heater_leaf_b],
            [r"safe_heater"],
            [],
            group_gap_mm=0.01,
        )

        self.assertEqual(body_objects, [])
        self.assertEqual(len(role_components), 1)
        self.assertEqual(role_components[0].name, "V_GUUTZ_SAFE_HEATER_HISPEC#1522")
        self.assertEqual(len(role_components[0].objects), 2)

    def test_role_component_detection_ignores_appended_geometry_name_when_hierarchy_exists(self) -> None:
        body_mesh = _mesh_object("V_GUUTZ_SAFE-HEATER_HISPEC_1209", "Copper", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        body_mesh.hierarchy_path = (
            "Default",
            "V_GUUTZ_EXTERNAL-SENSOR-HEATER-BREAKOUT-PCB_HISPEC#15414",
            "Copy of connector#15324",
        )
        body_mesh.scene_path = (
            "Default/V_GUUTZ_EXTERNAL-SENSOR-HEATER-BREAKOUT-PCB_HISPEC#15414/"
            "Copy of connector#15324 V_GUUTZ_SAFE-HEATER_HISPEC_599"
        )

        body_objects, role_components = collapse_role_components(
            [body_mesh],
            [r"safe_heater"],
            [r"coo_0001_p0003"],
            group_gap_mm=0.01,
        )

        self.assertEqual(body_objects, [body_mesh])
        self.assertEqual(role_components, [])

    def test_role_component_detection_prefers_scene_path_parent_over_partial_leaf_path(self) -> None:
        heater_leaf_a = _mesh_object("V_GUUTZ_SAFE-HEATER_HISPEC#1422", "Copper", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        heater_leaf_b = _mesh_object("V_GUUTZ_SAFE-HEATER_HISPEC#1424", "Copper", [1000.0, 0.0, 0.0], [1001.0, 1.0, 1.0])
        heater_leaf_a.hierarchy_path = ("Default", "V_GUUTZ_SAFE-HEATER_HISPEC#1422")
        heater_leaf_b.hierarchy_path = ("Default", "V_GUUTZ_SAFE-HEATER_HISPEC#1424")
        heater_leaf_a.scene_path = (
            "Default/HISPEC-0030-A0005/V_GUUTZ_SAFE-HEATER_HISPEC#1522/"
            "V_GUUTZ_SAFE-HEATER_HISPEC#1422"
        )
        heater_leaf_b.scene_path = (
            "Default/HISPEC-0030-A0005/V_GUUTZ_SAFE-HEATER_HISPEC#1522/"
            "V_GUUTZ_SAFE-HEATER_HISPEC#1424"
        )

        _body_objects, role_components = collapse_role_components(
            [heater_leaf_a, heater_leaf_b],
            [r"safe_heater"],
            [],
            group_gap_mm=0.01,
        )

        self.assertEqual(len(role_components), 1)
        self.assertEqual(role_components[0].name, "V_GUUTZ_SAFE_HEATER_HISPEC#1522")

    def test_role_component_detection_prefers_highest_hierarchy_role_match(self) -> None:
        heater_leaf = _mesh_object("solid_body", "Copper", [10.0, 0.0, 0.0], [11.0, 1.0, 1.0])
        heater_leaf.hierarchy_path = (
            "Default",
            "V_GUUTZ_SAFE-HEATER_HISPEC#1522",
            "V_GUUTZ_TEMP-SENSOR_HISPEC#17",
            "solid_body",
        )

        body_objects, role_components = collapse_role_components(
            [heater_leaf],
            [r"safe_heater"],
            [r"temp_sensor"],
            group_gap_mm=10.0,
        )

        self.assertEqual(body_objects, [])
        self.assertEqual([(component.kind, component.name) for component in role_components], [
            ("heater", "V_GUUTZ_SAFE_HEATER_HISPEC#1522")
        ])

    def test_role_component_detection_rejects_ambiguous_heater_sensor_match(self) -> None:
        ambiguous = _mesh_object("sensor_heater_combo", "Copper", [0.0, 0.0, 0.0], [5.0, 5.0, 1.0])

        with self.assertRaisesRegex(ValueError, "matches both heater and sensor"):
            collapse_role_components(
                [ambiguous],
                [r"heater"],
                [r"sensor"],
            )

    def test_cli_role_component_split_uses_only_configured_substrings(self) -> None:
        body = _mesh_object("body_panel", "Copper", [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])
        heater = _mesh_object("heater_strip_1", "Copper", [5.0, -2.0, -2.0], [6.0, 2.0, 2.0])
        sensor = _mesh_object("temperature_probe_A", "Copper", [-6.0, -1.0, -1.0], [-5.0, 1.0, 1.0])
        scene = GltfScene(
            path=SimpleNamespace(),
            objects=[body, heater, sensor],
            bounds_mm=(np.array([-5.0, -5.0, -5.0]), np.array([6.0, 5.0, 5.0])),
            warnings=[],
        )
        args = SimpleNamespace(
            no_detect_role_nodes=False,
            heater_name_pattern=None,
            heater_name_substring=None,
            sensor_name_pattern=None,
            sensor_name_substring=None,
            device_exclude_name_pattern=None,
            no_default_device_excludes=False,
            role_node_group_gap_mm=10.0,
        )
        warnings: list[str] = []

        voxel_scene, role_components = _split_role_components(scene, args, warnings)

        self.assertEqual([obj.name for obj in voxel_scene.objects], ["body_panel", "heater_strip_1", "temperature_probe_A"])
        self.assertEqual(role_components, [])
        self.assertEqual(args.role_components, [])
        self.assertEqual(warnings, [])

        args.heater_name_pattern = ["heater"]
        args.sensor_name_pattern = ["temperature"]
        warnings = []
        voxel_scene, role_components = _split_role_components(scene, args, warnings)

        self.assertEqual([obj.name for obj in voxel_scene.objects], ["body_panel", "heater_strip_1", "temperature_probe_A"])
        self.assertEqual(role_components, [])
        self.assertEqual(args.role_components, [])
        self.assertIn("Ignoring --heater-name-pattern/--sensor-name-pattern", warnings[0])

        args.heater_name_pattern = []
        args.sensor_name_pattern = []
        args.heater_name_substring = ["heater_strip"]
        args.sensor_name_substring = ["temperature-probe"]
        warnings = []
        voxel_scene, role_components = _split_role_components(scene, args, warnings)

        self.assertEqual([obj.name for obj in voxel_scene.objects], ["body_panel"])
        self.assertEqual([component.kind for component in role_components], ["heater", "sensor"])
        self.assertEqual(args.role_components, role_components)
        self.assertIn("voxelization uses 1 body object", warnings[0])

    def test_cli_legacy_physical_device_disable_flag_disables_role_detection(self) -> None:
        args = build_parser().parse_args(
            [
                "--mesh-dir",
                "meshes",
                "--graph-name",
                "graph",
                "--no-detect-physical-devices",
            ]
        )

        self.assertTrue(args.no_detect_role_nodes)

    def test_cli_accepts_ray_contains_backend(self) -> None:
        args = build_parser().parse_args(
            [
                "--mesh-dir",
                "meshes",
                "--graph-name",
                "graph",
                "--contains-backend",
                "ray",
            ]
        )

        self.assertEqual(args.contains_backend, "ray")

    def test_cli_defaults_to_ray_contains_backend(self) -> None:
        args = build_parser().parse_args(["--mesh-dir", "meshes", "--graph-name", "graph"])

        self.assertEqual(args.contains_backend, "ray")

    def test_cli_accepts_dense_matrix_node_limit(self) -> None:
        args = build_parser().parse_args(
            [
                "--mesh-dir",
                "meshes",
                "--graph-name",
                "graph",
                "--dense-matrix-node-limit",
                "0",
            ]
        )

        self.assertEqual(args.dense_matrix_node_limit, 0)

    def test_cli_accepts_ignored_component_substrings(self) -> None:
        args = build_parser().parse_args(
            [
                "--mesh-dir",
                "meshes",
                "--graph-name",
                "graph",
                "--ignore-component-substring",
                "fastener",
                "--ignored-component-substring",
                "shim",
            ]
        )

        self.assertEqual(args.ignore_component_substring, ["fastener", "shim"])

    def test_ignored_component_substring_filters_scene_before_role_detection(self) -> None:
        body = _mesh_object("body_panel", "Copper", [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])
        ignored_heater = _mesh_object("debug_SAFE-HEATER_fixture", "Copper", [5.0, -2.0, -2.0], [6.0, 2.0, 2.0])
        sensor = _mesh_object("temperature_probe_A", "Copper", [-6.0, -1.0, -1.0], [-5.0, 1.0, 1.0])
        scene = GltfScene(
            path=SimpleNamespace(),
            objects=[body, ignored_heater, sensor],
            bounds_mm=(np.array([-6.0, -5.0, -5.0]), np.array([6.0, 5.0, 5.0])),
            warnings=[],
        )
        args = SimpleNamespace(ignore_component_substring=["safe-heater"])
        warnings: list[str] = []

        filtered = _filter_ignored_components(scene, args, warnings)

        self.assertEqual([obj.name for obj in filtered.objects], ["body_panel", "temperature_probe_A"])
        self.assertEqual(args.ignored_component_names, ["debug_SAFE-HEATER_fixture"])
        self.assertIn("Ignored 1 CAD mesh object", warnings[0])

        split_args = SimpleNamespace(
            no_detect_role_nodes=False,
            heater_name_pattern=[],
            heater_name_substring=["safe-heater"],
            sensor_name_pattern=[],
            sensor_name_substring=["temperature-probe"],
            device_exclude_name_pattern=[],
            no_default_device_excludes=False,
            role_node_group_gap_mm=10.0,
        )
        voxel_scene, role_components = _split_role_components(filtered, split_args, warnings=[])

        self.assertEqual([obj.name for obj in voxel_scene.objects], ["body_panel"])
        self.assertEqual([component.kind for component in role_components], ["sensor"])

    def test_ignored_component_substring_matches_parent_hierarchy(self) -> None:
        body = _mesh_object("body_panel", "Copper", [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])
        leaf = _mesh_object("leaf_mesh_node", "Copper", [10.0, 0.0, 0.0], [11.0, 1.0, 1.0])
        leaf.hierarchy_path = ("Default", "IGNORE_THIS_ASSEMBLY", "leaf_mesh_node")
        leaf.scene_path = "Default/IGNORE_THIS_ASSEMBLY/leaf_mesh_node"
        scene = GltfScene(
            path=SimpleNamespace(),
            objects=[body, leaf],
            bounds_mm=(np.array([-5.0, -5.0, -5.0]), np.array([11.0, 5.0, 5.0])),
            warnings=[],
        )
        args = SimpleNamespace(ignore_component_substring=["ignore-this-assembly"])

        filtered = _filter_ignored_components(scene, args, warnings=[])

        self.assertEqual([obj.name for obj in filtered.objects], ["body_panel"])
        self.assertEqual(args.ignored_component_names, ["leaf_mesh_node"])

    def test_cli_role_substrings_are_normalized_like_cad_names(self) -> None:
        body = _mesh_object("body_panel", "Copper", [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])
        sensor = _mesh_object("THERMAL_PICKUP_A", "Copper", [-6.0, -1.0, -1.0], [-5.0, 1.0, 1.0])
        scene = GltfScene(
            path=SimpleNamespace(),
            objects=[body, sensor],
            bounds_mm=(np.array([-6.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0])),
            warnings=[],
        )
        args = SimpleNamespace(
            no_detect_role_nodes=False,
            heater_name_pattern=[],
            heater_name_substring=[],
            sensor_name_pattern=[],
            sensor_name_substring=["thermal-pickup"],
            device_exclude_name_pattern=[],
            no_default_device_excludes=False,
            role_node_group_gap_mm=10.0,
        )

        voxel_scene, role_components = _split_role_components(scene, args, warnings=[])

        self.assertEqual([obj.name for obj in voxel_scene.objects], ["body_panel"])
        self.assertEqual(len(role_components), 1)
        self.assertEqual(role_components[0].kind, "sensor")

    def test_cli_role_substrings_are_not_blocked_by_default_excludes(self) -> None:
        body = _mesh_object("body_panel", "Copper", [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])
        sensor = _mesh_object(
            "V_GUUTZ_EXTERNAL-SENSOR-HEATER-BREAKOUT-PCB_HISPEC",
            "Copper",
            [-6.0, -1.0, -1.0],
            [-5.0, 1.0, 1.0],
        )
        scene = GltfScene(
            path=SimpleNamespace(),
            objects=[body, sensor],
            bounds_mm=(np.array([-6.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0])),
            warnings=[],
        )
        args = SimpleNamespace(
            no_detect_role_nodes=False,
            heater_name_pattern=[],
            heater_name_substring=[],
            sensor_name_pattern=[],
            sensor_name_substring=["external-sensor-heater-breakout-pcb"],
            device_exclude_name_pattern=[],
            no_default_device_excludes=False,
            role_node_group_gap_mm=10.0,
        )

        voxel_scene, role_components = _split_role_components(scene, args, warnings=[])

        self.assertEqual([obj.name for obj in voxel_scene.objects], ["body_panel"])
        self.assertEqual(len(role_components), 1)
        self.assertEqual(role_components[0].kind, "sensor")

    def test_graph_build_adds_dedicated_heater_node_for_detected_role(self) -> None:
        materials = self.make_materials()
        leaves = [
            OctreeCell(
                cell_id="cell_body",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(0.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"body_panel": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="body_panel",
                dominant_material="Copper",
                confidence="high",
            )
        ]
        heater_obj = _mesh_object("heater_strip_1", "Copper", [5.0, -2.0, -2.0], [7.0, 2.0, 2.0])
        role_component = RoleComponent(name="heater_strip", kind="heater", objects=[heater_obj])

        result = build_graph(
            leaves,
            ContactReport(),
            materials,
            warnings=[],
            role_components=[role_component],
        )

        self.assertEqual(len(result.nodes), 2)
        body_node = next(node for node in result.nodes if not node["is_heater"])
        heater_node = next(node for node in result.nodes if node["is_heater"])
        self.assertTrue(heater_node["is_heater"])
        self.assertFalse(heater_node["is_sensor"])
        self.assertEqual(heater_node["role_source_components"], ["heater_strip_1"])
        self.assertEqual(heater_node["power_deposition_node_ids"], [body_node["node_id"]])
        self.assertEqual(heater_node["mass_kg"], 0.0)
        self.assertEqual(heater_node["volume_m3"], 0.0)
        self.assertEqual(heater_node["C_J_K"], 1.0)
        self.assertTrue(heater_node["C_manual_override"])
        self.assertEqual(len(result.edges), 1)
        self.assertEqual(result.edges[0]["edge_type"], "role_node_contact")
        self.assertEqual(result.edges[0]["G_W_K"], 0.0)

    def test_graph_build_uses_one_dedicated_role_node_for_multi_object_heater(self) -> None:
        materials = self.make_materials()
        leaves = [
            OctreeCell(
                cell_id="cell_body",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(10.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"body_panel": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="body_panel",
                dominant_material="Copper",
                confidence="high",
            )
        ]
        heater_left = _mesh_object("safe_heater_1", "Copper", [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])
        heater_right = _mesh_object("safe_heater_2", "Copper", [15.0, -5.0, -5.0], [25.0, 5.0, 5.0])
        role_component = RoleComponent(name="safe_heater", kind="heater", objects=[heater_left, heater_right])

        result = build_graph(
            leaves,
            ContactReport(),
            materials,
            warnings=[],
            role_components=[role_component],
        )

        heater_nodes = [node for node in result.nodes if node["is_heater"]]
        body_nodes = [node for node in result.nodes if not node["is_heater"]]
        self.assertEqual(len(result.nodes), 2)
        self.assertEqual(len(heater_nodes), 1)
        self.assertEqual(len(body_nodes), 1)
        heater = heater_nodes[0]
        body = body_nodes[0]
        self.assertEqual(heater["component_name"], "safe_heater")
        self.assertEqual(sorted(heater["source_components"]), ["safe_heater_1", "safe_heater_2"])
        self.assertNotIn("source_node_ids", heater)
        self.assertNotIn("source_cell_ids", heater)
        self.assertEqual(heater["power_deposition_node_ids"], [body["node_id"]])
        self.assertEqual(len(result.edges), 1)
        edge = result.edges[0]
        self.assertEqual({edge["node_i"], edge["node_j"]}, {heater["node_id"], body["node_id"]})
        self.assertEqual(edge["edge_type"], "role_node_contact")
        self.assertEqual(edge["G_W_K"], 0.0)
        matrices = build_octree_matrices(result.nodes, result.edges)
        body_row = list(matrices["node_ids"]).index(body["node_id"])
        heater_row = list(matrices["node_ids"]).index(heater["node_id"])
        self.assertEqual(matrices["G"][body_row, heater_row], 0.0)

    def test_large_octree_matrix_builder_uses_sparse_laplacian_without_dense_g(self) -> None:
        nodes = [
            {
                "node_id": node_id,
                "component_name": f"component_{node_id}",
                "material_name": "Copper",
                "C_J_K": 1.0,
                "radiation": {"G_rad_W_K": 0.0},
            }
            for node_id in range(4)
        ]
        edges = [
            {"edge_id": "edge_0", "node_i": 0, "node_j": 1, "G_W_K": 2.0},
            {"edge_id": "edge_1", "node_i": 1, "node_j": 2, "G_W_K": 3.0},
            {
                "edge_id": "marker",
                "node_i": 2,
                "node_j": 3,
                "G_W_K": 99.0,
                "edge_type": "role_node_contact",
            },
        ]

        matrices = build_octree_matrices(nodes, edges, dense_node_limit=1)

        self.assertNotIn("G", matrices)
        self.assertTrue(issparse(matrices["L"]))
        dense_l = matrices["L"].toarray()
        self.assertEqual(dense_l[0, 0], 2.0)
        self.assertEqual(dense_l[1, 1], 5.0)
        self.assertEqual(dense_l[2, 2], 3.0)
        self.assertEqual(dense_l[2, 3], 0.0)
        self.assertEqual(validate_graph({"graph_nodes": nodes, "graph_edges": edges, "warnings": []}, matrices)[0], [])

    def test_sparse_octree_output_writes_l_sparse_without_dense_matrix_files(self) -> None:
        nodes = [
            {
                "node_id": node_id,
                "component_name": f"component_{node_id}",
                "material_name": "Copper",
                "C_J_K": 1.0,
                "radiation": {"G_rad_W_K": 0.0},
            }
            for node_id in range(3)
        ]
        edges = [{"edge_id": "edge_0", "node_i": 0, "node_j": 1, "G_W_K": 2.0}]
        matrices = build_octree_matrices(nodes, edges, dense_node_limit=1)
        graph = {
            "metadata": {"graph_name": "sparse_output"},
            "parameters": {},
            "diagnostics": {},
            "graph_nodes": nodes,
            "graph_edges": edges,
            "warnings": [],
        }
        materials = {"Copper": Material("Copper", density_kg_m3=1.0, cp_J_kgK=1.0, k_W_mK=1.0, emissivity=0.5)}

        with TemporaryDirectory() as directory:
            output = Path(directory)
            _write_outputs(output, graph, matrices, materials, warnings=[])

            self.assertTrue((output / "L_sparse.json").exists())
            self.assertTrue((output / "C.npy").exists())
            self.assertTrue((output / "G_rad.npy").exists())
            self.assertFalse((output / "G.npy").exists())
            self.assertFalse((output / "L.npy").exists())

    def test_graph_build_adds_dedicated_role_node_when_detected_component_has_no_voxel_cells(self) -> None:
        materials = self.make_materials()
        leaves = [
            OctreeCell(
                cell_id="cell_1",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(0.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"body_panel": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="body_panel",
                dominant_material="Copper",
                confidence="high",
            )
        ]
        heater_obj = _mesh_object("heater_strip_1", "Copper", [5.01, -2.0, -2.0], [7.01, 2.0, 2.0])
        role_component = RoleComponent(name="heater_strip", kind="heater", objects=[heater_obj])
        warnings: list[str] = []

        result = build_graph(
            leaves,
            ContactReport(),
            materials,
            warnings=warnings,
            contact_detection_distance_mm=3.0,
            role_components=[role_component],
        )

        self.assertEqual(len(result.nodes), 2)
        heater_nodes = [node for node in result.nodes if node["is_heater"]]
        self.assertEqual(len(heater_nodes), 1)
        self.assertEqual(heater_nodes[0]["component_name"], "heater_strip")
        self.assertEqual(heater_nodes[0]["source_components"], ["heater_strip_1"])
        self.assertEqual(heater_nodes[0]["role_source_components"], ["heater_strip_1"])
        self.assertEqual(result.edges, [])
        self.assertIn("0 contact edges", " ".join(warnings))

    def test_graph_build_dedicated_role_node_survives_mesh_volume_failure(self) -> None:
        class DegenerateMesh:
            vertices = np.empty((0, 3))
            faces = np.empty((0, 3), dtype=int)
            triangles = np.empty((0, 3, 3))
            is_watertight = False

            @property
            def volume(self) -> float:
                raise ZeroDivisionError("center_mass = integrated[1:4] / volume")

        materials = self.make_materials()
        leaves = [
            OctreeCell(
                cell_id="cell_1",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(0.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"body_panel": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="body_panel",
                dominant_material="Copper",
                confidence="high",
            )
        ]
        sensor_obj = MeshObject(
            name="sensor_probe_1",
            material_name="Copper",
            mesh=DegenerateMesh(),
            vertices_mm=np.empty((0, 3)),
            bounds_mm=(np.array([5.01, -1.0, -1.0]), np.array([7.01, 1.0, 1.0])),
            watertight=False,
            scene_path="sensor_probe_1",
        )
        role_component = RoleComponent(name="sensor_probe", kind="sensor", objects=[sensor_obj])

        result = build_graph(
            leaves,
            ContactReport(),
            materials,
            warnings=[],
            role_components=[role_component],
        )

        sensor_nodes = [node for node in result.nodes if node["is_sensor"]]
        self.assertEqual(len(sensor_nodes), 1)
        self.assertEqual(sensor_nodes[0]["component_name"], "sensor_probe")
        self.assertEqual(sensor_nodes[0]["volume_m3"], 0.0)
        self.assertEqual(sensor_nodes[0]["mass_kg"], 0.0)
        self.assertEqual(sensor_nodes[0]["C_J_K"], 1.0)

    def test_graph_build_connects_dedicated_sensor_node_to_contacting_body_cell(self) -> None:
        materials = self.make_materials()
        leaves = [
            OctreeCell(
                cell_id="cell_1",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(0.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"body_panel": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="body_panel",
                dominant_material="Copper",
                confidence="high",
            )
        ]
        sensor_obj = _mesh_object("temperature_probe_A", "Copper", [4.0, -2.0, -2.0], [6.0, 2.0, 2.0])
        role_component = RoleComponent(name="temperature_probe", kind="sensor", objects=[sensor_obj])
        warnings: list[str] = []

        result = build_graph(
            leaves,
            ContactReport(),
            materials,
            warnings=warnings,
            role_components=[role_component],
        )

        sensor_nodes = [node for node in result.nodes if node["is_sensor"]]
        self.assertEqual(len(sensor_nodes), 1)
        sensor = sensor_nodes[0]
        self.assertEqual(sensor["component_name"], "temperature_probe")
        self.assertEqual(sensor["source_components"], ["temperature_probe_A"])
        self.assertEqual(len(result.edges), 1)
        self.assertEqual({result.edges[0]["node_i"], result.edges[0]["node_j"]}, {0, sensor["node_id"]})
        self.assertEqual(result.edges[0]["edge_type"], "role_node_contact")
        self.assertEqual(result.edges[0]["G_W_K"], 0.0)

    def test_graph_build_pairs_dedicated_heater_and_sensor_nodes_using_body_contacts(self) -> None:
        materials = self.make_materials()
        leaves = [
            OctreeCell(
                cell_id="cell_body",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(10.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"body_panel": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="body_panel",
                dominant_material="Copper",
                confidence="high",
            ),
        ]
        heater_obj = _mesh_object("heater_strip_1", "Copper", [4.0, -4.0, -4.0], [6.0, 4.0, 4.0])
        sensor_obj = _mesh_object("temperature_probe_A", "Copper", [4.0, -1.0, -1.0], [6.0, 1.0, 1.0])
        heater_component = RoleComponent(name="heater_strip", kind="heater", objects=[heater_obj])
        sensor_component = RoleComponent(name="temperature_probe", kind="sensor", objects=[sensor_obj])
        warnings: list[str] = []

        result = build_graph(
            leaves,
            ContactReport(),
            materials,
            warnings=warnings,
            role_components=[heater_component, sensor_component],
        )

        sensors = [node for node in result.nodes if node["is_sensor"]]
        heaters = [node for node in result.nodes if node["is_heater"]]
        bodies = [node for node in result.nodes if not node["is_heater"] and not node["is_sensor"]]
        self.assertEqual(len(sensors), 1)
        self.assertEqual(len(heaters), 1)
        self.assertEqual(len(bodies), 1)
        self.assertEqual(sensors[0]["sensor_connected_node_ids"], [bodies[0]["node_id"]])
        self.assertEqual(heaters[0]["power_deposition_node_ids"], [bodies[0]["node_id"]])
        self.assertTrue(sensors[0]["sensor_valid"])
        self.assertEqual(heaters[0]["assigned_sensor_id"], sensors[0]["node_id"])

    def test_graph_build_allows_configured_multiple_heaters_per_sensor(self) -> None:
        materials = self.make_materials()
        leaves = [
            OctreeCell(
                cell_id="cell_body",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(10.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"body_panel": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="body_panel",
                dominant_material="Copper",
                confidence="high",
            ),
        ]
        heater_a = _mesh_object("heater_strip_A", "Copper", [4.0, -4.0, -4.0], [6.0, 4.0, 4.0])
        heater_b = _mesh_object("heater_strip_B", "Copper", [4.0, 5.0, -4.0], [6.0, 7.0, 4.0])
        sensor_obj = _mesh_object("temperature_probe_A", "Copper", [4.0, -1.0, -1.0], [6.0, 1.0, 1.0])
        warnings: list[str] = []

        result = build_graph(
            leaves,
            ContactReport(),
            materials,
            warnings=warnings,
            role_components=[
                RoleComponent(name="heater_A", kind="heater", objects=[heater_a]),
                RoleComponent(name="heater_B", kind="heater", objects=[heater_b]),
                RoleComponent(name="temperature_probe", kind="sensor", objects=[sensor_obj]),
            ],
            max_heaters_per_sensor=2,
        )

        sensors = [node for node in result.nodes if node["is_sensor"]]
        heaters = [node for node in result.nodes if node["is_heater"]]
        self.assertEqual(len(sensors), 1)
        self.assertEqual(len(heaters), 2)
        self.assertEqual(
            sorted(heater["assigned_sensor_id"] for heater in heaters),
            [sensors[0]["node_id"], sensors[0]["node_id"]],
        )
        self.assertEqual(sorted(sensors[0]["assigned_heater_ids"]), sorted(heater["node_id"] for heater in heaters))

    def test_graph_build_adds_near_contact_edges_for_near_face_cells(self) -> None:
        materials = self.make_materials()
        leaves = [
            OctreeCell(
                cell_id="cell_1",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(0.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"part_1": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="part_1",
                dominant_material="Copper",
                confidence="low",
            ),
            OctreeCell(
                cell_id="cell_2",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(12.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"part_2": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="part_2",
                dominant_material="Copper",
                confidence="low",
            ),
        ]

        result = build_graph(leaves, None, materials, warnings=[], contact_detection_distance_mm=3.0)

        self.assertEqual(len(result.edges), 1)
        self.assertEqual(result.edges[0]["edge_type"], "near_same_material_contact")
        self.assertAlmostEqual(result.edges[0]["shared_area_m2"], 100.0e-6)
        self.assertAlmostEqual(result.edges[0]["distance_m"], 0.012)

    def test_candidate_cell_pairs_streams_only_spatial_neighbors(self) -> None:
        cells = [
            OctreeCell(
                cell_id="cell_1",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(0.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"part_1": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="part_1",
                dominant_material="Copper",
                confidence="high",
            ),
            OctreeCell(
                cell_id="cell_2",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(10.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"part_2": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="part_2",
                dominant_material="Copper",
                confidence="high",
            ),
            OctreeCell(
                cell_id="cell_3",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(23.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"part_3": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="part_3",
                dominant_material="Copper",
                confidence="high",
            ),
            OctreeCell(
                cell_id="cell_far",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(200.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"part_far": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="part_far",
                dominant_material="Copper",
                confidence="high",
            ),
        ]

        touching_pairs = {_pair_ids(a, b) for a, b in _candidate_cell_pairs(cells, 0.0)}
        near_pairs = {_pair_ids(a, b) for a, b in _candidate_cell_pairs(cells, 3.0)}

        self.assertIn(("cell_1", "cell_2"), touching_pairs)
        self.assertIn(("cell_2", "cell_3"), near_pairs)
        self.assertNotIn(("cell_1", "cell_far"), near_pairs)
        self.assertNotIn(("cell_2", "cell_far"), near_pairs)

    def test_exposed_area_uses_indexed_neighbor_pairs_for_mixed_cell_sizes(self) -> None:
        cells = [
            OctreeCell(
                cell_id="cell_large",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(0.0, 0.0, 0.0),
                size_mm=(20.0, 20.0, 20.0),
                occupancy={"part_large": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="part_large",
                dominant_material="Copper",
                confidence="high",
            ),
            OctreeCell(
                cell_id="cell_small_a",
                parent_id=None,
                children_ids=[],
                level=1,
                center_mm=(15.0, -5.0, 0.0),
                size_mm=(10.0, 10.0, 20.0),
                occupancy={"part_small_a": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="part_small_a",
                dominant_material="Copper",
                confidence="high",
            ),
            OctreeCell(
                cell_id="cell_small_b",
                parent_id=None,
                children_ids=[],
                level=1,
                center_mm=(15.0, 5.0, 0.0),
                size_mm=(10.0, 10.0, 20.0),
                occupancy={"part_small_b": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="part_small_b",
                dominant_material="Copper",
                confidence="high",
            ),
            OctreeCell(
                cell_id="cell_far",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(100.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"part_far": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="part_far",
                dominant_material="Copper",
                confidence="high",
            ),
        ]

        areas = _exposed_areas_m2(cells)

        self.assertAlmostEqual(areas["cell_large"], 2000.0e-6)
        self.assertAlmostEqual(areas["cell_small_a"], 600.0e-6)
        self.assertAlmostEqual(areas["cell_small_b"], 600.0e-6)
        self.assertAlmostEqual(areas["cell_far"], 600.0e-6)

    def test_graph_build_ignores_near_diagonal_cells_without_face_overlap(self) -> None:
        materials = self.make_materials()
        leaves = [
            OctreeCell(
                cell_id="cell_1",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(0.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"part_1": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="part_1",
                dominant_material="Copper",
                confidence="low",
            ),
            OctreeCell(
                cell_id="cell_2",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(12.0, 12.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={"part_2": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="part_2",
                dominant_material="Copper",
                confidence="low",
            ),
        ]

        result = build_graph(leaves, None, materials, warnings=[], contact_detection_distance_mm=3.0)

        self.assertEqual(result.edges, [])

    def test_graph_build_does_not_add_loose_component_bounds_contact_when_voxels_do_not_touch(self) -> None:
        materials = self.make_materials()
        leaves = [
            OctreeCell(
                cell_id="cell_1",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(0.0, 0.0, 0.0),
                size_mm=(4.0, 4.0, 4.0),
                occupancy={"part_1": 1.0},
                material_fractions={"Copper": 1.0},
                dominant_component="part_1",
                dominant_material="Copper",
                confidence="low",
            ),
            OctreeCell(
                cell_id="cell_2",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(30.0, 0.0, 0.0),
                size_mm=(4.0, 4.0, 4.0),
                occupancy={"part_2": 1.0},
                material_fractions={"Not assigned": 1.0},
                dominant_component="part_2",
                dominant_material="Not assigned",
                confidence="low",
            ),
        ]

        result = build_graph(
            leaves,
            None,
            materials,
            warnings=[],
            contact_detection_distance_mm=2.0,
            component_bounds_mm={
                "part_1": (np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0])),
                "part_2": (np.array([11.0, 0.0, 0.0]), np.array([20.0, 10.0, 10.0])),
            },
        )

        self.assertEqual(result.edges, [])

    def test_surface_triangle_assigns_open_non_watertight_leaf(self) -> None:
        materials = self.make_materials()
        triangle = np.array(
            [
                [[-4.0, -4.0, 0.0], [4.0, -4.0, 0.0], [0.0, 4.0, 0.0]],
            ],
            dtype=float,
        )
        mesh = SimpleNamespace(
            vertices=triangle.reshape(-1, 3),
            faces=np.array([[0, 1, 2]]),
            triangles=triangle,
            is_watertight=False,
        )
        obj = MeshObject(
            name="thin_panel",
            material_name="Copper",
            mesh=mesh,
            vertices_mm=triangle.reshape(-1, 3),
            bounds_mm=(np.array([-4.0, -4.0, 0.0]), np.array([4.0, 4.0, 0.0])),
            watertight=False,
        )
        scene = GltfScene(
            path=SimpleNamespace(),
            objects=[obj],
            bounds_mm=(np.array([-5.0, -5.0, -1.0]), np.array([5.0, 5.0, 1.0])),
            warnings=[],
        )
        params = OctreeParams(
            min_cell_size_mm=20.0,
            max_cell_size_mm=20.0,
            max_depth=0,
            samples_per_cell=4,
        )
        diagnostics = OctreeDiagnostics(debug_leaves=True)

        leaves = build_octree(scene, ContactReport(), materials, params, warnings=[], diagnostics=diagnostics)

        solid = [leaf for leaf in leaves if not leaf.is_empty]
        self.assertEqual(len(solid), 1)
        self.assertEqual(solid[0].dominant_component, "thin_panel")
        self.assertEqual(solid[0].dominant_material, "Copper")
        self.assertGreater(solid[0].occupancy["thin_panel"], 0.0)
        self.assertEqual(diagnostics.cells_surface_hit, 1)
        self.assertEqual(diagnostics.leaf_records[0]["acceptance_reason"], "triangle_surface_intersection")

    def test_parallel_octree_classification_matches_sequential_for_surface_mesh(self) -> None:
        materials = self.make_materials()
        triangle = np.array(
            [
                [[-4.0, -4.0, 0.0], [4.0, -4.0, 0.0], [0.0, 4.0, 0.0]],
            ],
            dtype=float,
        )
        mesh = SimpleNamespace(
            vertices=triangle.reshape(-1, 3),
            faces=np.array([[0, 1, 2]]),
            triangles=triangle,
            is_watertight=False,
        )
        obj = MeshObject(
            name="thin_panel",
            material_name="Copper",
            mesh=mesh,
            vertices_mm=triangle.reshape(-1, 3),
            bounds_mm=(np.array([-4.0, -4.0, 0.0]), np.array([4.0, 4.0, 0.0])),
            watertight=False,
        )
        scene = GltfScene(
            path=SimpleNamespace(),
            objects=[obj],
            bounds_mm=(np.array([-5.0, -5.0, -1.0]), np.array([5.0, 5.0, 1.0])),
            warnings=[],
        )
        sequential_params = OctreeParams(
            min_cell_size_mm=5.0,
            max_cell_size_mm=20.0,
            max_depth=2,
            samples_per_cell=4,
            voxel_workers=1,
        )
        parallel_params = OctreeParams(
            min_cell_size_mm=5.0,
            max_cell_size_mm=20.0,
            max_depth=2,
            samples_per_cell=4,
            voxel_workers=2,
            voxel_batch_size=2,
        )

        sequential = build_octree(scene, ContactReport(), materials, sequential_params, warnings=[])
        parallel = build_octree(scene, ContactReport(), materials, parallel_params, warnings=[])

        def signature(cells: list[OctreeCell]) -> list[tuple]:
            return [
                (
                    cell.cell_id,
                    cell.level,
                    cell.center_mm,
                    cell.size_mm,
                    cell.dominant_component,
                    cell.dominant_material,
                    dict(cell.occupancy),
                )
                for cell in cells
            ]

        self.assertEqual(signature(parallel), signature(sequential))

    def test_parallel_octree_worker_failure_falls_back_to_sequential(self) -> None:
        materials = self.make_materials()
        triangle = np.array(
            [
                [[-4.0, -4.0, 0.0], [4.0, -4.0, 0.0], [0.0, 4.0, 0.0]],
            ],
            dtype=float,
        )
        mesh = SimpleNamespace(
            vertices=triangle.reshape(-1, 3),
            faces=np.array([[0, 1, 2]]),
            triangles=triangle,
            is_watertight=False,
        )
        obj = MeshObject(
            name="thin_panel",
            material_name="Copper",
            mesh=mesh,
            vertices_mm=triangle.reshape(-1, 3),
            bounds_mm=(np.array([-4.0, -4.0, 0.0]), np.array([4.0, 4.0, 0.0])),
            watertight=False,
        )
        scene = GltfScene(
            path=SimpleNamespace(),
            objects=[obj],
            bounds_mm=(np.array([-5.0, -5.0, -1.0]), np.array([5.0, 5.0, 1.0])),
            warnings=[],
        )
        params = OctreeParams(
            min_cell_size_mm=20.0,
            max_cell_size_mm=20.0,
            max_depth=0,
            samples_per_cell=4,
            voxel_workers=2,
        )
        warnings: list[str] = []

        class FailingExecutor:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

            def map(self, *args, **kwargs):
                raise AttributeError("'range_iterator' object has no attribute 'get'")

        with patch("octree_graph.octree.ProcessPoolExecutor", FailingExecutor):
            leaves = build_octree(scene, ContactReport(), materials, params, warnings=warnings)

        solid = [leaf for leaf in leaves if not leaf.is_empty]
        self.assertEqual(len(solid), 1)
        self.assertEqual(solid[0].dominant_component, "thin_panel")
        self.assertIn("falling back to sequential", " ".join(warnings))
        self.assertIn("range_iterator", " ".join(warnings))

    def test_sample_points_are_symmetric_for_low_sample_counts(self) -> None:
        points = _sample_points(np.zeros(3), np.array([10.0, 10.0, 10.0]), 4)

        self.assertEqual(points.shape, (4, 3))
        self.assertTrue(np.any(points[:, 0] < 0.0))
        self.assertTrue(np.any(points[:, 0] > 0.0))
        self.assertTrue(np.any(points[:, 1] < 0.0))
        self.assertTrue(np.any(points[:, 1] > 0.0))
        self.assertTrue(np.any(points[:, 2] < 0.0))
        self.assertTrue(np.any(points[:, 2] > 0.0))

    def test_ray_contains_backend_bypasses_trimesh_contains(self) -> None:
        class ContainsShouldNotRunMesh:
            triangles = np.array(
                [
                    [[1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0]],
                    [[1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [1.0, -1.0, 1.0]],
                ],
                dtype=float,
            )

            def contains(self, points) -> list[bool]:
                raise AssertionError("trimesh.contains should not run")

        obj = MeshObject(
            name="ray_panel",
            material_name="Copper",
            mesh=ContainsShouldNotRunMesh(),
            vertices_mm=np.empty((0, 3), dtype=float),
            bounds_mm=(np.array([1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])),
            watertight=True,
        )

        self.assertIsInstance(
            _mesh_contains_point(
                obj,
                np.array([0.0, 0.0, 0.0]),
                OctreeParams(contains_backend="ray"),
            ),
            bool,
        )

    def test_objects_intersecting_bounds_skips_non_finite_object_bounds(self) -> None:
        valid = _mesh_object("valid", "Copper", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        invalid = _mesh_object("invalid", "Copper", [float("nan"), 0.0, 0.0], [1.0, 1.0, 1.0])

        hits = _objects_intersecting_bounds(
            [invalid, valid],
            np.array([-0.5, -0.5, -0.5]),
            np.array([0.5, 0.5, 0.5]),
        )

        self.assertEqual([obj.name for obj in hits], ["valid"])

    def test_objects_intersecting_bounds_skips_invalid_query_bounds(self) -> None:
        valid = _mesh_object("valid", "Copper", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        hits = _objects_intersecting_bounds(
            [valid],
            np.array([float("nan"), -0.5, -0.5]),
            np.array([0.5, 0.5, 0.5]),
        )

        self.assertEqual(hits, [])

    def test_objects_intersecting_bounds_normalizes_reversed_bounds(self) -> None:
        valid = _mesh_object("valid", "Copper", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        hits = _objects_intersecting_bounds(
            [valid],
            np.array([0.5, 0.5, 0.5]),
            np.array([-0.5, -0.5, -0.5]),
        )

        self.assertEqual([obj.name for obj in hits], ["valid"])

    def test_mesh_triangles_sanitizes_native_mesh_triangle_data(self) -> None:
        class BadTriangleMesh:
            triangles = np.array(
                [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    [[float("nan"), 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                ],
                dtype=float,
            )

        obj = MeshObject(
            name="part",
            material_name="Copper",
            mesh=BadTriangleMesh(),
            vertices_mm=np.empty((0, 3), dtype=float),
            bounds_mm=(np.zeros(3), np.ones(3)),
            watertight=False,
        )

        triangles = _mesh_triangles(obj)

        self.assertEqual(triangles.shape, (1, 3, 3))
        self.assertTrue(triangles.flags.c_contiguous)

    def test_triangle_box_intersection_rejects_aabb_only_overlap(self) -> None:
        triangle = np.array(
            [
                [0.4, 1.0, 0.0],
                [1.0, 0.4, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        )

        self.assertFalse(_triangle_intersects_aabb(triangle, np.zeros(3), np.array([0.5, 0.5, 0.5])))

    def test_triangle_index_large_query_uses_bounds_fallback(self) -> None:
        triangles = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[100.0, 100.0, 0.0], [101.0, 100.0, 0.0], [100.0, 101.0, 0.0]],
            ],
            dtype=float,
        )
        mesh = SimpleNamespace(triangles=triangles)
        obj = MeshObject(
            name="part",
            material_name="Copper",
            mesh=mesh,
            vertices_mm=triangles.reshape(-1, 3),
            bounds_mm=(np.array([0.0, 0.0, 0.0]), np.array([101.0, 101.0, 0.0])),
            watertight=False,
        )
        index = TriangleSpatialIndex.from_mesh(obj, target_bucket_size_mm=0.01)

        matches = index.query(np.array([-1000.0, -1000.0, -1000.0]), np.array([1000.0, 1000.0, 1000.0]))

        self.assertEqual(matches.tolist(), [0, 1])

    def test_triangle_index_large_query_filters_bounds_in_chunks(self) -> None:
        triangles = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
                [[10.0, 10.0, 0.0], [11.0, 10.0, 0.0], [10.0, 11.0, 0.0]],
                [[12.0, 10.0, 0.0], [13.0, 10.0, 0.0], [12.0, 11.0, 0.0]],
                [[20.0, 20.0, 0.0], [21.0, 20.0, 0.0], [20.0, 21.0, 0.0]],
            ],
            dtype=float,
        )
        mesh = SimpleNamespace(triangles=triangles)
        obj = MeshObject(
            name="part",
            material_name="Copper",
            mesh=mesh,
            vertices_mm=triangles.reshape(-1, 3),
            bounds_mm=(np.array([0.0, 0.0, 0.0]), np.array([21.0, 21.0, 0.0])),
            watertight=False,
        )
        index = TriangleSpatialIndex.from_mesh(obj, target_bucket_size_mm=1000.0)

        with patch("octree_graph.octree._TRIANGLE_QUERY_CHUNK_SIZE", 2):
            matches = index.query(np.array([-0.5, -0.5, -0.5]), np.array([12.5, 10.5, 0.5]))

        self.assertEqual(matches.tolist(), [0, 1, 2, 3])

    def test_triangle_index_uses_triangle_bounds_when_object_bounds_are_invalid(self) -> None:
        triangles = np.array(
            [
                [[4.0, 4.0, 4.0], [6.0, 4.0, 4.0], [4.0, 6.0, 4.0]],
            ],
            dtype=float,
        )
        mesh = SimpleNamespace(triangles=triangles)
        obj = MeshObject(
            name="part",
            material_name="Copper",
            mesh=mesh,
            vertices_mm=triangles.reshape(-1, 3),
            bounds_mm=(np.array([float("nan"), 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
            watertight=False,
        )

        index = TriangleSpatialIndex.from_mesh(obj, target_bucket_size_mm=1.0)
        matches = index.query(np.array([3.0, 3.0, 3.0]), np.array([7.0, 7.0, 5.0]))

        self.assertEqual(matches.tolist(), [0])

    def test_triangle_index_keeps_large_bucket_span_triangles_queryable(self) -> None:
        triangles = np.array(
            [
                [[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0]],
            ],
            dtype=float,
        )
        mesh = SimpleNamespace(triangles=triangles)
        obj = MeshObject(
            name="part",
            material_name="Copper",
            mesh=mesh,
            vertices_mm=triangles.reshape(-1, 3),
            bounds_mm=(np.array([0.0, 0.0, 0.0]), np.array([1000.0, 1000.0, 0.0])),
            watertight=False,
        )

        with patch("octree_graph.octree._TRIANGLE_BUCKET_INSERT_LIMIT", 1):
            index = TriangleSpatialIndex.from_mesh(obj, target_bucket_size_mm=1.0)
        matches = index.query(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))

        self.assertEqual(len(index.buckets), 0)
        self.assertEqual(index.unbucketed_triangle_indices.tolist(), [0])
        self.assertEqual(matches.tolist(), [0])

    def test_bbox_fallback_does_not_assign_aabb_only_leaf_when_refinement_budget_stops(self) -> None:
        materials = self.make_materials()
        mesh = SimpleNamespace(vertices=[], faces=[], triangles=np.empty((0, 3, 3)), is_watertight=False)
        obj = MeshObject(
            name="part_1",
            material_name="Copper",
            mesh=mesh,
            vertices_mm=np.empty((0, 3)),
            bounds_mm=(np.array([-5.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0])),
            watertight=False,
        )
        scene = GltfScene(
            path=SimpleNamespace(),
            objects=[obj],
            bounds_mm=(np.array([-5.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0])),
            warnings=[],
        )
        params = OctreeParams(
            min_cell_size_mm=1.0,
            max_cell_size_mm=2.0,
            max_leaf_cells=1,
            bbox_fallback=True,
            samples_per_cell=3,
        )

        diagnostics = OctreeDiagnostics()

        leaves = build_octree(scene, ContactReport(), materials, params, warnings=[], diagnostics=diagnostics)

        self.assertEqual(len(leaves), 1)
        self.assertIsNone(leaves[0].dominant_component)
        self.assertEqual(leaves[0].occupancy, {})
        self.assertIn("AABB overlap", " ".join(leaves[0].warnings))
        self.assertEqual(diagnostics.cells_subdivided, 0)

    def test_occupied_cells_exceed_leaf_budget_to_honor_max_cell_size(self) -> None:
        materials = self.make_materials()
        triangle = np.array(
            [
                [[-3.0, -3.0, 0.0], [3.0, -3.0, 0.0], [0.0, 3.0, 0.0]],
            ],
            dtype=float,
        )
        mesh = SimpleNamespace(
            vertices=triangle.reshape(-1, 3),
            faces=np.array([[0, 1, 2]]),
            triangles=triangle,
            is_watertight=False,
        )
        obj = MeshObject(
            name="panel",
            material_name="Copper",
            mesh=mesh,
            vertices_mm=triangle.reshape(-1, 3),
            bounds_mm=(np.array([-3.0, -3.0, 0.0]), np.array([3.0, 3.0, 0.0])),
            watertight=False,
        )
        scene = GltfScene(
            path=SimpleNamespace(),
            objects=[obj],
            bounds_mm=(np.array([-4.0, -4.0, -4.0]), np.array([4.0, 4.0, 4.0])),
            warnings=[],
        )
        params = OctreeParams(
            min_cell_size_mm=1.0,
            max_cell_size_mm=2.0,
            max_leaf_cells=1,
            max_depth=4,
            samples_per_cell=4,
        )
        warnings: list[str] = []
        diagnostics = OctreeDiagnostics()

        leaves = build_octree(scene, ContactReport(), materials, params, warnings=warnings, diagnostics=diagnostics)

        solid = [leaf for leaf in leaves if not leaf.is_empty]
        self.assertTrue(solid)
        self.assertGreater(len(leaves), 1)
        self.assertGreater(len(leaves), params.max_leaf_cells)
        self.assertLessEqual(max(max(leaf.size_mm) for leaf in solid), 2.0)
        self.assertIn("max_leaf_cells was exceeded to enforce max_cell_size_mm", " ".join(warnings))
        self.assertTrue(diagnostics.max_leaf_cells_reached)

    def test_occupied_cells_warn_when_max_size_blocked_by_depth(self) -> None:
        materials = self.make_materials()
        triangle = np.array(
            [
                [[-3.0, -3.0, 0.0], [3.0, -3.0, 0.0], [0.0, 3.0, 0.0]],
            ],
            dtype=float,
        )
        mesh = SimpleNamespace(
            vertices=triangle.reshape(-1, 3),
            faces=np.array([[0, 1, 2]]),
            triangles=triangle,
            is_watertight=False,
        )
        obj = MeshObject(
            name="panel",
            material_name="Copper",
            mesh=mesh,
            vertices_mm=triangle.reshape(-1, 3),
            bounds_mm=(np.array([-3.0, -3.0, 0.0]), np.array([3.0, 3.0, 0.0])),
            watertight=False,
        )
        scene = GltfScene(
            path=SimpleNamespace(),
            objects=[obj],
            bounds_mm=(np.array([-4.0, -4.0, -4.0]), np.array([4.0, 4.0, 4.0])),
            warnings=[],
        )
        params = OctreeParams(
            min_cell_size_mm=1.0,
            max_cell_size_mm=2.0,
            max_leaf_cells=256,
            max_depth=0,
            samples_per_cell=4,
        )
        warnings: list[str] = []
        diagnostics = OctreeDiagnostics()

        leaves = build_octree(scene, ContactReport(), materials, params, warnings=warnings, diagnostics=diagnostics)

        solid = [leaf for leaf in leaves if not leaf.is_empty]
        self.assertTrue(solid)
        self.assertGreater(max(max(leaf.size_mm) for leaf in solid), 2.0)
        self.assertEqual(solid[0].confidence, "low")
        self.assertIn("Some occupied cells exceed max_cell_size_mm", " ".join(warnings))
        self.assertIn("Cannot satisfy max_cell_size_mm", " ".join(solid[0].warnings))

    def test_occupied_cells_honor_max_cell_size_when_budget_allows(self) -> None:
        materials = self.make_materials()
        triangle = np.array(
            [
                [[-3.0, -3.0, 0.0], [3.0, -3.0, 0.0], [0.0, 3.0, 0.0]],
            ],
            dtype=float,
        )
        mesh = SimpleNamespace(
            vertices=triangle.reshape(-1, 3),
            faces=np.array([[0, 1, 2]]),
            triangles=triangle,
            is_watertight=False,
        )
        obj = MeshObject(
            name="panel",
            material_name="Copper",
            mesh=mesh,
            vertices_mm=triangle.reshape(-1, 3),
            bounds_mm=(np.array([-3.0, -3.0, 0.0]), np.array([3.0, 3.0, 0.0])),
            watertight=False,
        )
        scene = GltfScene(
            path=SimpleNamespace(),
            objects=[obj],
            bounds_mm=(np.array([-4.0, -4.0, -4.0]), np.array([4.0, 4.0, 4.0])),
            warnings=[],
        )
        params = OctreeParams(
            min_cell_size_mm=1.0,
            max_cell_size_mm=2.0,
            max_leaf_cells=256,
            max_depth=4,
            samples_per_cell=4,
        )
        warnings: list[str] = []
        diagnostics = OctreeDiagnostics()

        leaves = build_octree(scene, ContactReport(), materials, params, warnings=warnings, diagnostics=diagnostics)

        solid = [leaf for leaf in leaves if not leaf.is_empty]
        self.assertGreater(len(leaves), 1)
        self.assertTrue(solid)
        self.assertLessEqual(max(max(leaf.size_mm) for leaf in solid), 2.0)
        self.assertEqual(warnings, [])

    def test_adjacent_balance_targets_coarse_leaf_next_to_fine_leaf(self) -> None:
        fine = OctreeCell(
            cell_id="fine",
            parent_id=None,
            children_ids=[],
            level=3,
            center_mm=(0.0, 0.0, 0.0),
            size_mm=(1.0, 1.0, 1.0),
            occupancy={"fine_part": 1.0},
            material_fractions={"Copper": 1.0},
            dominant_component="fine_part",
            dominant_material="Copper",
            confidence="high",
        )
        coarse = OctreeCell(
            cell_id="coarse",
            parent_id=None,
            children_ids=[],
            level=1,
            center_mm=(2.5, 0.0, 0.0),
            size_mm=(4.0, 4.0, 4.0),
            occupancy={"coarse_part": 1.0},
            material_fractions={"Copper": 1.0},
            dominant_component="coarse_part",
            dominant_material="Copper",
            confidence="high",
        )
        far = OctreeCell(
            cell_id="far",
            parent_id=None,
            children_ids=[],
            level=1,
            center_mm=(20.0, 0.0, 0.0),
            size_mm=(8.0, 8.0, 8.0),
            occupancy={"far_part": 1.0},
            material_fractions={"Copper": 1.0},
            dominant_component="far_part",
            dominant_material="Copper",
            confidence="high",
        )
        params = OctreeParams(
            max_cell_size_mm=2.0,
            min_cell_size_mm=0.5,
            max_depth=5,
            max_adjacent_leaf_size_ratio=2.0,
        )

        targets = _adjacent_balance_refinement_targets([fine, coarse, far], params)

        self.assertEqual(targets, {"coarse"})

    def test_crowded_component_refinement_subdivides_dense_empty_regions(self) -> None:
        materials = self.make_materials()
        left = _mesh_object("small_part_left", "Copper", [-5.0, -1.0, -1.0], [-3.0, 1.0, 1.0])
        right = _mesh_object("small_part_right", "Copper", [3.0, -1.0, -1.0], [5.0, 1.0, 1.0])
        scene = GltfScene(
            path=SimpleNamespace(),
            objects=[left, right],
            bounds_mm=(np.array([-5.0, -1.0, -1.0]), np.array([5.0, 1.0, 1.0])),
            warnings=[],
        )
        coarse_params = OctreeParams(
            min_cell_size_mm=1.25,
            max_cell_size_mm=50.0,
            max_depth=4,
            crowded_component_refine_count=0,
        )
        crowded_params = OctreeParams(
            min_cell_size_mm=1.25,
            max_cell_size_mm=50.0,
            max_depth=4,
            crowded_component_refine_count=2,
            crowded_component_refine_distance_mm=3.0,
        )

        coarse = build_octree(scene, ContactReport(), materials, coarse_params, warnings=[])
        crowded = build_octree(scene, ContactReport(), materials, crowded_params, warnings=[])

        self.assertEqual(len(coarse), 1)
        self.assertGreater(len(crowded), len(coarse))
        self.assertLess(min(max(cell.size_mm) for cell in crowded), max(coarse[0].size_mm))

    def test_occupied_above_max_cell_size_dominates_discretionary_refinement_priority(self) -> None:
        params = OctreeParams(max_cell_size_mm=8.0)
        occupied_above_max = CellClassification(
            occupied=True,
            surface_hit=False,
            inside_hit=True,
            near_surface_hit=False,
            bbox_only_hit=False,
            surface_mesh_ids=set(),
            inside_mesh_ids={1},
            near_surface_mesh_ids=set(),
            candidate_mesh_ids={1},
            crowded_component_count=1,
            role_component_count=0,
            surface_component_count=0,
            near_surface_component_count=1,
            material_ids={"Aluminum"},
            part_ids={"large_part"},
            occupancy={"large_part": 1.0},
            material_fractions={"Aluminum": 1.0},
            dominant_component="large_part",
            dominant_material="Aluminum",
            volume_fraction=1.0,
            acceptance_reason="inside",
        )
        crowded_detail = CellClassification(
            occupied=True,
            surface_hit=True,
            inside_hit=False,
            near_surface_hit=True,
            bbox_only_hit=False,
            surface_mesh_ids={2, 3},
            inside_mesh_ids=set(),
            near_surface_mesh_ids={2, 3, 4, 5, 6},
            candidate_mesh_ids={2, 3, 4, 5, 6},
            crowded_component_count=8,
            role_component_count=0,
            surface_component_count=2,
            near_surface_component_count=5,
            material_ids={"Aluminum", "Copper"},
            part_ids={"a", "b"},
            occupancy={"a": 0.5, "b": 0.5},
            material_fractions={"Aluminum": 0.5, "Copper": 0.5},
            dominant_component="a",
            dominant_material="Aluminum",
            volume_fraction=0.5,
            acceptance_reason="triangle_surface_intersection",
            triangle_candidate_tests=512,
        )

        above_score, above_reasons = _refinement_priority(
            occupied_above_max,
            params,
            np.array([46.0, 46.0, 46.0]),
            mixed_parts=False,
            mixed_materials=False,
            high_contrast=False,
            crowded_component_refinement=False,
            role_component_refinement=False,
            multi_surface_refinement=False,
            surface_complexity_refinement=False,
            gap_preservation_refinement=False,
            needs_surface_refinement=False,
        )
        detail_score, detail_reasons = _refinement_priority(
            crowded_detail,
            params,
            np.array([4.0, 4.0, 4.0]),
            mixed_parts=True,
            mixed_materials=True,
            high_contrast=True,
            crowded_component_refinement=True,
            role_component_refinement=False,
            multi_surface_refinement=True,
            surface_complexity_refinement=True,
            gap_preservation_refinement=False,
            needs_surface_refinement=True,
        )

        self.assertIn("above_max_cell_size", above_reasons)
        self.assertNotIn("above_max_cell_size", detail_reasons)
        self.assertGreater(above_score, detail_score)

    def test_gap_preservation_refinement_prioritizes_low_fill_cells_between_components(self) -> None:
        params = OctreeParams(max_cell_size_mm=8.0, min_solid_fraction=0.12)
        gap_cell = CellClassification(
            occupied=True,
            surface_hit=True,
            inside_hit=False,
            near_surface_hit=True,
            bbox_only_hit=False,
            surface_mesh_ids={1},
            inside_mesh_ids=set(),
            near_surface_mesh_ids={2},
            candidate_mesh_ids={1, 2},
            crowded_component_count=2,
            role_component_count=0,
            surface_component_count=1,
            near_surface_component_count=2,
            material_ids={"Aluminum"},
            part_ids={"left_part"},
            occupancy={"left_part": 0.12},
            material_fractions={"Aluminum": 0.12},
            dominant_component="left_part",
            dominant_material="Aluminum",
            volume_fraction=0.12,
            acceptance_reason="triangle_surface_intersection",
            triangle_candidate_tests=64,
        )
        detailed_surface = CellClassification(
            occupied=True,
            surface_hit=True,
            inside_hit=False,
            near_surface_hit=True,
            bbox_only_hit=False,
            surface_mesh_ids={3},
            inside_mesh_ids=set(),
            near_surface_mesh_ids={3},
            candidate_mesh_ids={3},
            crowded_component_count=1,
            role_component_count=0,
            surface_component_count=1,
            near_surface_component_count=1,
            material_ids={"Copper"},
            part_ids={"detailed_part"},
            occupancy={"detailed_part": 0.5},
            material_fractions={"Copper": 0.5},
            dominant_component="detailed_part",
            dominant_material="Copper",
            volume_fraction=0.5,
            acceptance_reason="triangle_surface_intersection",
            triangle_candidate_tests=1024,
        )

        self.assertTrue(_needs_gap_preservation_refinement(gap_cell, params))
        self.assertFalse(_needs_gap_preservation_refinement(detailed_surface, params))

        gap_score, gap_reasons = _refinement_priority(
            gap_cell,
            params,
            np.array([6.0, 6.0, 6.0]),
            mixed_parts=False,
            mixed_materials=False,
            high_contrast=False,
            crowded_component_refinement=True,
            role_component_refinement=False,
            multi_surface_refinement=True,
            surface_complexity_refinement=True,
            gap_preservation_refinement=True,
            needs_surface_refinement=True,
        )
        detail_score, detail_reasons = _refinement_priority(
            detailed_surface,
            params,
            np.array([6.0, 6.0, 6.0]),
            mixed_parts=False,
            mixed_materials=False,
            high_contrast=False,
            crowded_component_refinement=False,
            role_component_refinement=False,
            multi_surface_refinement=False,
            surface_complexity_refinement=True,
            gap_preservation_refinement=False,
            needs_surface_refinement=True,
        )

        self.assertIn("gap_preservation", gap_reasons)
        self.assertNotIn("gap_preservation", detail_reasons)
        self.assertGreater(gap_score, detail_score)

    def test_detected_role_bounds_force_local_octree_refinement(self) -> None:
        materials = self.make_materials()
        sensor = _mesh_object("temperature_probe_A", "Copper", [0.0, 0.0, 0.0], [8.0, 8.0, 8.0])
        scene = GltfScene(
            path=SimpleNamespace(),
            objects=[sensor],
            bounds_mm=(np.array([0.0, 0.0, 0.0]), np.array([8.0, 8.0, 8.0])),
            warnings=[],
        )
        coarse_diagnostics = OctreeDiagnostics()
        role_diagnostics = OctreeDiagnostics()

        coarse = build_octree(
            scene,
            ContactReport(),
            materials,
            OctreeParams(
                min_cell_size_mm=2.0,
                max_cell_size_mm=100.0,
                max_depth=3,
                contact_refine_distance_mm=0.0,
            ),
            warnings=[],
            diagnostics=coarse_diagnostics,
        )
        role_refined = build_octree(
            scene,
            ContactReport(),
            materials,
            OctreeParams(
                min_cell_size_mm=2.0,
                max_cell_size_mm=100.0,
                max_depth=3,
                contact_refine_distance_mm=0.0,
                role_refine_component_names=("temperature_probe_A",),
                role_refine_max_depth=2,
            ),
            warnings=[],
            diagnostics=role_diagnostics,
        )

        self.assertEqual(len(coarse), 1)
        self.assertEqual(coarse_diagnostics.cells_subdivided, 0)
        self.assertEqual(coarse_diagnostics.cells_role_component_hit, 0)
        self.assertEqual(len(role_refined), 64)
        self.assertEqual(role_diagnostics.cells_subdivided, 9)
        self.assertEqual(role_diagnostics.cells_role_component_hit, 64)

    def test_empty_graph_guard_reports_actionable_failure(self) -> None:
        args = SimpleNamespace(bbox_fallback=False, max_leaf_cells=15000)
        leaves = [
            OctreeCell(
                cell_id="cell_1",
                parent_id=None,
                children_ids=[],
                level=0,
                center_mm=(0.0, 0.0, 0.0),
                size_mm=(10.0, 10.0, 10.0),
                occupancy={},
                material_fractions={},
                dominant_component=None,
                dominant_material=None,
                confidence="low",
            )
        ]

        with self.assertRaises(SystemExit) as context:
            _raise_if_empty_graph([], leaves, args)

        message = str(context.exception)
        self.assertIn("none were classified as solid graph nodes", message)
        self.assertIn("triangle-box surface intersections", message)

    def test_empty_first_pass_does_not_retry_with_bbox_fallback(self) -> None:
        args = SimpleNamespace(
            bbox_fallback=False,
            max_leaf_cells=15000,
            contact_detection_distance_mm=None,
            proximity_contact_distance_mm=None,
            role_contact_tolerance_mm=1.0e-6,
            radiation_reference_temperature_K=293.15,
        )
        params = OctreeParams(bbox_fallback=False)
        empty_leaf = OctreeCell(
            cell_id="cell_empty",
            parent_id=None,
            children_ids=[],
            level=0,
            center_mm=(0.0, 0.0, 0.0),
            size_mm=(10.0, 10.0, 10.0),
            occupancy={},
            material_fractions={},
            dominant_component=None,
            dominant_material=None,
            confidence="low",
        )
        with (
            patch("octree_graph.cli.build_octree", return_value=[empty_leaf]) as build_octree_mock,
            patch(
                "octree_graph.cli.build_graph",
                return_value=SimpleNamespace(nodes=[], edges=[], warnings=[]),
            ) as build_graph_mock,
        ):
            leaves, graph_result = _build_graph_with_optional_fallback(
                scene=SimpleNamespace(),
                voxel_scene=SimpleNamespace(),
                contact_report=ContactReport(),
                materials=self.make_materials(),
                params=params,
                args=args,
                warnings=[],
            )

        self.assertEqual(build_octree_mock.call_count, 1)
        self.assertFalse(args.bbox_fallback)
        self.assertFalse(params.bbox_fallback)
        self.assertEqual(leaves, [empty_leaf])
        self.assertEqual(graph_result.nodes, [])
        self.assertEqual(build_graph_mock.call_args.kwargs["contact_detection_distance_mm"], 0.0)
        self.assertEqual(
            build_graph_mock.call_args.kwargs["contact_interface_conductance_W_m2K"],
            DEFAULT_CONTACT_INTERFACE_CONDUCTANCE_W_M2K,
        )

    def test_cli_accepts_contact_interface_conductance(self) -> None:
        args = build_parser().parse_args(
            [
                "--mesh-dir",
                "meshes/example",
                "--graph-name",
                "graph",
                "--contact-interface-conductance-W-m2K",
                "25000",
            ]
        )

        self.assertEqual(args.contact_interface_conductance_W_m2K, 25000.0)


if __name__ == "__main__":
    unittest.main()
