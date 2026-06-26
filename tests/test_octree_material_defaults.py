"""Tests for octree material defaulting behavior."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from octree_graph.load_contact_report import ContactReport
from octree_graph.materials import (
    DEFAULT_ASSIGNED_MATERIAL_NAME,
    Material,
    resolve_material,
)
from octree_graph.octree import _physical_material_name


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


if __name__ == "__main__":
    unittest.main()
