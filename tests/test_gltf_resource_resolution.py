"""Tests for glTF external resource path normalization."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from octree_graph.cli import build_parser, _resolve_gltf_path
from octree_graph.load_gltf import _PLACEHOLDER_IMAGE_URI, _prepare_gltf_for_load


class GltfResourceResolutionTests(unittest.TestCase):
    def test_mesh_dir_selects_single_gltf_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gltf = root / "Assembly.gltf"
            gltf.write_text("{}", encoding="utf-8")
            (root / "Assembly.bin").write_bytes(b"abc")
            args = build_parser().parse_args(["--mesh-dir", str(root), "--graph-name", "test"])

            self.assertEqual(_resolve_gltf_path(args), gltf)

    def test_mesh_dir_rejects_multiple_gltf_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.gltf").write_text("{}", encoding="utf-8")
            (root / "b.gltf").write_text("{}", encoding="utf-8")
            args = build_parser().parse_args(["--mesh-dir", str(root), "--graph-name", "test"])

            with self.assertRaises(ValueError):
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


if __name__ == "__main__":
    unittest.main()
