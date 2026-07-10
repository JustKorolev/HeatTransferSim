"""Tests for octree material defaulting behavior."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from octree_graph.cli import (
    _build_graph_with_optional_fallback,
    _log_scene_memory_risk,
    _raise_if_empty_graph,
    _resolve_material_lookup_path,
    _split_role_components,
)
from octree_graph.load_contact_report import ContactReport
from octree_graph.graph_builder import (
    DEFAULT_ROLE_EXCLUDE_NAME_PATTERNS,
    RoleComponent,
    _candidate_cell_pairs,
    _exposed_areas_m2,
    build_graph,
    collapse_role_components,
)
from octree_graph.load_gltf import GltfScene, MeshObject
from octree_graph.load_contact_report import load_contact_report
from octree_graph.materials import (
    DEFAULT_ASSIGNED_MATERIAL_NAME,
    Material,
    infer_material_name_from_text,
    resolve_material,
)
from octree_graph.octree import (
    OctreeCell,
    OctreeDiagnostics,
    OctreeParams,
    _physical_material_name,
    _sample_points,
    _triangle_intersects_aabb,
    TriangleSpatialIndex,
    build_octree,
)


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
        self.assertEqual({component.name for component in role_components}, {"assembly/temperature_probe_A", "kapton_heater"})

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
        self.assertEqual(role_components[0].name, "kapton_heater")

    def test_role_component_detection_splits_distant_same_named_instances(self) -> None:
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
        self.assertEqual([component.name for component in role_components], ["safe_heater_1", "safe_heater_2"])
        self.assertEqual([len(component.objects) for component in role_components], [1, 2])

    def test_role_component_detection_rejects_ambiguous_heater_sensor_match(self) -> None:
        ambiguous = _mesh_object("sensor_heater_combo", "Copper", [0.0, 0.0, 0.0], [5.0, 5.0, 1.0])

        with self.assertRaisesRegex(ValueError, "matches both heater and sensor"):
            collapse_role_components(
                [ambiguous],
                [r"heater"],
                [r"sensor"],
            )

    def test_cli_role_component_split_excludes_components_from_voxel_scene(self) -> None:
        body = _mesh_object("body_panel", "Copper", [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])
        heater = _mesh_object("heater_strip_1", "Copper", [5.0, -2.0, -2.0], [6.0, 2.0, 2.0])
        scene = GltfScene(
            path=SimpleNamespace(),
            objects=[body, heater],
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

        self.assertEqual([obj.name for obj in voxel_scene.objects], ["body_panel", "heater_strip_1"])
        self.assertEqual(role_components, [])
        self.assertEqual(args.role_components, [])
        self.assertEqual(warnings, [])

        args.heater_name_substring = ["heater_strip"]
        voxel_scene, role_components = _split_role_components(scene, args, warnings)

        self.assertEqual([obj.name for obj in voxel_scene.objects], ["body_panel"])
        self.assertEqual(len(role_components), 1)
        self.assertEqual(role_components[0].kind, "heater")
        self.assertEqual(args.role_components, role_components)
        self.assertIn("excluded them from voxelization", warnings[0])

    def test_graph_build_adds_heater_role_node_and_contact_edge(self) -> None:
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
        heater_node = result.nodes[1]
        self.assertEqual(heater_node["node_type"], "heater")
        self.assertTrue(heater_node["is_heater"])
        self.assertFalse(heater_node["is_sensor"])
        self.assertNotIn("heater", heater_node["tags"])
        self.assertNotIn("sensor", heater_node["tags"])
        self.assertEqual(heater_node["source_components"], ["heater_strip_1"])
        self.assertGreater(heater_node["C_J_K"], 0.0)
        self.assertEqual(len(result.edges), 1)
        self.assertEqual(result.edges[0]["edge_type"], "role_node_contact")
        self.assertEqual(result.edges[0]["source"], "cad_role_node_contact")

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


if __name__ == "__main__":
    unittest.main()
