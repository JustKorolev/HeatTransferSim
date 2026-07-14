"""Tests for glTF external resource path normalization."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from octree_graph.cli import build_parser, _resolve_gltf_path
from octree_graph.load_gltf import (
    _PLACEHOLDER_IMAGE_URI,
    _prepare_gltf_for_load,
    _raw_gltf_mesh_node_paths,
    load_gltf_scene,
)


class GltfResourceResolutionTests(unittest.TestCase):
    def test_mesh_dir_selects_single_glb_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            glb = root / "Assembly.glb"
            glb.write_bytes(b"glb")
            args = build_parser().parse_args(["--mesh-dir", str(root), "--graph-name", "test"])

            self.assertEqual(_resolve_gltf_path(args), glb)

    def test_mesh_dir_rejects_gltf_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Assembly.gltf").write_text("{}", encoding="utf-8")
            (root / "Assembly.bin").write_bytes(b"abc")
            args = build_parser().parse_args(["--mesh-dir", str(root), "--graph-name", "test"])

            with self.assertRaisesRegex(ValueError, "no longer accepted"):
                _resolve_gltf_path(args)

    def test_mesh_dir_rejects_multiple_glb_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.glb").write_bytes(b"a")
            (root / "b.glb").write_bytes(b"b")
            args = build_parser().parse_args(["--mesh-dir", str(root), "--graph-name", "test"])

            with self.assertRaisesRegex(ValueError, "multiple .glb"):
                _resolve_gltf_path(args)

    def test_mesh_dir_rejects_gltf_even_when_glb_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Assembly.gltf").write_text("{}", encoding="utf-8")
            (root / "Assembly.glb").write_bytes(b"glb")
            args = build_parser().parse_args(["--mesh-dir", str(root), "--graph-name", "test"])

            with self.assertRaisesRegex(ValueError, "External-buffer .gltf"):
                _resolve_gltf_path(args)

    def test_cli_module_entrypoint_runs_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "octree_graph.cli", "--help"],
            capture_output=True,
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            timeout=10,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--mesh-dir", result.stdout)

    def test_resolves_missing_resource_folder_to_bin_next_to_gltf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gltf = root / "Assembly" / "Assembly.gltf"
            gltf.parent.mkdir()
            (gltf.parent / "Assembly.bin").write_bytes(b"abc")
            gltf.write_text(
                json.dumps({"asset": {"version": "2.0"}, "buffers": [{"uri": "./Assembly_resources/Assembly.bin"}]}),
                encoding="utf-8",
            )

            load_path, temporary_path, warnings = _prepare_gltf_for_load(gltf)

            try:
                self.assertIsNotNone(temporary_path)
                normalized = json.loads(load_path.read_text(encoding="utf-8"))
                self.assertEqual(normalized["buffers"][0]["uri"], "Assembly.bin")
                self.assertEqual(warnings, [])
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    def test_resolves_renamed_sibling_bin_when_embedded_uri_uses_old_export_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gltf = root / "HISPEC-CRYOSTAT-SMALL.gltf"
            (root / "HISPEC-CRYOSTAT-SMALL.bin").write_bytes(b"abcdef")
            gltf.write_text(
                json.dumps(
                    {
                        "asset": {"version": "2.0"},
                        "buffers": [
                            {
                                "byteLength": 3,
                                "uri": "./HISPEC-CRYOSTAT_smaller_resources/HISPEC-CRYOSTAT_smaller.bin",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            load_path, temporary_path, warnings = _prepare_gltf_for_load(gltf)

            try:
                self.assertIsNotNone(temporary_path)
                normalized = json.loads(load_path.read_text(encoding="utf-8"))
                self.assertEqual(normalized["buffers"][0]["uri"], "HISPEC-CRYOSTAT-SMALL.bin")
                self.assertEqual(warnings, [])
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    def test_replaces_missing_images_but_requires_missing_buffers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gltf = root / "assembly.gltf"
            (root / "assembly.bin").write_bytes(b"abc")
            gltf.write_text(
                json.dumps(
                    {
                        "asset": {"version": "2.0"},
                        "buffers": [{"uri": "assembly.bin"}],
                        "images": [{"uri": "missing_texture.jpeg"}],
                    }
                ),
                encoding="utf-8",
            )

            load_path, temporary_path, warnings = _prepare_gltf_for_load(gltf)

            try:
                self.assertIsNotNone(temporary_path)
                normalized = json.loads(load_path.read_text(encoding="utf-8"))
                self.assertEqual(normalized["images"][0]["uri"], _PLACEHOLDER_IMAGE_URI)
                self.assertEqual(len(warnings), 1)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

            gltf.write_text(
                json.dumps({"asset": {"version": "2.0"}, "buffers": [{"uri": "missing.bin"}]}),
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                _prepare_gltf_for_load(gltf)

    def test_load_gltf_scene_survives_degenerate_trimesh_mass_properties(self) -> None:
        class DegenerateGeometry:
            vertices = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=float,
            )
            faces = np.array([[0, 1, 2]], dtype=int)
            visual = SimpleNamespace(material=SimpleNamespace(name="Copper"))

            @property
            def is_watertight(self) -> bool:
                raise ZeroDivisionError("center_mass = integrated[1:4] / volume")

            def copy(self):
                raise AssertionError("loader should not copy live trimesh geometry")

        loaded = SimpleNamespace(
            graph=SimpleNamespace(
                nodes_geometry=["node_without_sensor_name"],
                get=lambda node_name: (np.eye(4), "sensor_probe_geometry"),
            ),
            geometry={"sensor_probe_geometry": DegenerateGeometry()},
        )
        fake_trimesh = SimpleNamespace(load=lambda load_path, force=None: loaded)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"trimesh": fake_trimesh}):
            glb = Path(tmp) / "assembly.glb"
            glb.write_bytes(b"glb")

            scene = load_gltf_scene(glb)

        self.assertEqual(len(scene.objects), 1)
        obj = scene.objects[0]
        self.assertEqual(obj.name, "node_without_sensor_name")
        self.assertIn("sensor_probe_geometry", obj.scene_path)
        self.assertFalse(obj.watertight)
        self.assertEqual(obj.mesh.triangles.shape, (1, 3, 3))
        self.assertIn("not reported watertight", " ".join(scene.warnings))

    def test_load_gltf_scene_preserves_hierarchy_path(self) -> None:
        class Geometry:
            vertices = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=float,
            )
            faces = np.array([[0, 1, 2]], dtype=int)
            visual = SimpleNamespace(material=SimpleNamespace(name="Copper"))
            is_watertight = True

        loaded = SimpleNamespace(
            graph=SimpleNamespace(
                nodes_geometry=["leaf_mesh_node"],
                get=lambda node_name: (np.eye(4), "leaf_geometry"),
                transforms=SimpleNamespace(
                    parents={
                        "leaf_mesh_node": "V_GUUTZ_SAFE-HEATER_HISPEC_1522",
                        "V_GUUTZ_SAFE-HEATER_HISPEC_1522": "HISPEC-0030-A0005",
                        "HISPEC-0030-A0005": "Default",
                    }
                ),
            ),
            geometry={"leaf_geometry": Geometry()},
        )
        fake_trimesh = SimpleNamespace(load=lambda load_path, force=None: loaded)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"trimesh": fake_trimesh}):
            glb = Path(tmp) / "assembly.glb"
            glb.write_bytes(b"glb")

            scene = load_gltf_scene(glb)

        self.assertEqual(
            scene.objects[0].hierarchy_path,
            (
                "Default",
                "HISPEC-0030-A0005",
                "V_GUUTZ_SAFE-HEATER_HISPEC_1522",
                "leaf_mesh_node",
            ),
        )
        self.assertIn("Default/HISPEC-0030-A0005", scene.objects[0].scene_path)

    def test_load_gltf_scene_uses_raw_hierarchy_paths_when_counts_differ(self) -> None:
        class Geometry:
            vertices = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=float,
            )
            faces = np.array([[0, 1, 2]], dtype=int)
            visual = SimpleNamespace(material=SimpleNamespace(name="Copper"))
            is_watertight = True

        loaded = SimpleNamespace(
            graph=SimpleNamespace(
                nodes_geometry=["leaf_mesh_node"],
                get=lambda node_name: (np.eye(4), "leaf_geometry"),
                transforms=SimpleNamespace(parents={}),
            ),
            geometry={"leaf_geometry": Geometry()},
        )
        fake_trimesh = SimpleNamespace(load=lambda load_path, force=None: loaded)
        raw_paths = [
            (
                "leaf_mesh_node",
                (
                    "Default#1",
                    "V_GUUTZ_SAFE-HEATER_HISPEC#1522",
                    "V_GUUTZ_SAFE-HEATER_HISPEC#1422",
                ),
            ),
            (
                "other_mesh_node",
                (
                    "Default#1",
                    "V_GUUTZ_SAFE-HEATER_HISPEC#1622",
                    "V_GUUTZ_SAFE-HEATER_HISPEC#1499",
                ),
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"trimesh": fake_trimesh}):
            glb = Path(tmp) / "assembly.glb"
            glb.write_bytes(b"glb")
            with patch("octree_graph.load_gltf._raw_gltf_mesh_node_paths", return_value=raw_paths):
                scene = load_gltf_scene(glb)

        self.assertEqual(
            scene.objects[0].hierarchy_path,
            (
                "Default#1",
                "V_GUUTZ_SAFE-HEATER_HISPEC#1522",
                "V_GUUTZ_SAFE-HEATER_HISPEC#1422",
            ),
        )
        self.assertIn("using raw hierarchy paths where ordinals overlap", " ".join(scene.warnings))

    def test_raw_glb_mesh_node_paths_include_node_indices_for_repeated_names(self) -> None:
        tree = {
            "asset": {"version": "2.0"},
            "nodes": [
                {"name": "root", "children": [1, 3]},
                {"name": "V_GUUTZ_SAFE-HEATER_HISPEC", "children": [2]},
                {"name": "V_GUUTZ_SAFE-HEATER_HISPEC", "mesh": 0},
                {"name": "V_GUUTZ_SAFE-HEATER_HISPEC", "children": [4]},
                {"name": "V_GUUTZ_SAFE-HEATER_HISPEC", "mesh": 1},
            ],
            "meshes": [{}, {}],
        }
        payload = json.dumps(tree).encode("utf-8")
        payload += b" " * ((4 - len(payload) % 4) % 4)
        header = b"glTF" + (2).to_bytes(4, "little") + (12 + 8 + len(payload)).to_bytes(4, "little")
        chunk = len(payload).to_bytes(4, "little") + (0x4E4F534A).to_bytes(4, "little") + payload

        with tempfile.TemporaryDirectory() as tmp:
            glb = Path(tmp) / "assembly.glb"
            glb.write_bytes(header + chunk)

            paths = _raw_gltf_mesh_node_paths(glb)

        self.assertEqual(len(paths), 2)
        self.assertNotEqual(paths[0][1], paths[1][1])
        self.assertIn("#1", "/".join(paths[0][1]))
        self.assertIn("#3", "/".join(paths[1][1]))


if __name__ == "__main__":
    unittest.main()
