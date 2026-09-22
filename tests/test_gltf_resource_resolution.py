"""Tests for glTF external resource path normalization."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from octree_graph.cli import build_parser, _resolve_gltf_path, _resolve_step_path
from octree_graph.load_gltf import (
    _PLACEHOLDER_IMAGE_URI,
    _prepare_gltf_for_load,
    _raw_gltf_mesh_node_paths,
    _resource_uri_for_temp_gltf,
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


class TempGltfResourceUriTests(unittest.TestCase):
    """The buffer URI must not depend on how the temp directory is SPELLED.

    The resource path is resolved before it gets here; the temp directory is not.
    On Windows those are routinely two spellings of one directory, because TEMP is
    reported with an 8.3 short name -- GitHub's runners have runneradmin as
    RUNNER~1, and any account over eight characters or containing a space is the
    same. os.path.relpath compares textually, so it used to emit a six-level ../
    chain out of one spelling and back into the other, leaving the rewritten glTF
    pointing at a path that does not exist.
    """

    def test_a_resource_beside_the_temp_gltf_is_just_its_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resource = root / "Assembly.bin"
            resource.write_bytes(b"abc")
            self.assertEqual(
                _resource_uri_for_temp_gltf(resource.resolve(), root), "Assembly.bin"
            )

    @staticmethod
    def _second_spelling_of(directory: Path):
        """Another spelling of the same real directory, or None if unavailable.

        On Windows: the 8.3 short name, which is what GitHub's runners report for
        TEMP (runneradmin -> RUNNER~1). This is the exact CI condition. Note that
        merely upper-casing would NOT reproduce it -- ntpath.relpath normalises
        case, so the bug would hide.

        On POSIX: a symlink, which relpath likewise will not see through.
        """
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            get_short = ctypes.windll.kernel32.GetShortPathNameW
            get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
            get_short.restype = wintypes.DWORD
            buffer = ctypes.create_unicode_buffer(1024)
            if not get_short(str(directory), buffer, 1024):
                return None
            short = Path(buffer.value)
            return short if str(short) != str(directory) else None

        link = directory.parent / f"{directory.name}-link"
        try:
            link.symlink_to(directory, target_is_directory=True)
        except (OSError, NotImplementedError):
            return None
        return link

    def test_a_differently_spelled_temp_dir_still_gives_the_filename(self) -> None:
        """The regression. Both spellings name one directory, so the answer is the
        bare filename either way -- and relpath cannot see that on its own."""
        with tempfile.TemporaryDirectory(prefix="averylongdirectoryname_") as tmp:
            root = Path(tmp)
            resource = root / "Assembly.bin"
            resource.write_bytes(b"abc")

            other = self._second_spelling_of(root)
            if other is None:
                self.skipTest("no second spelling of the temp directory available")
            self.assertEqual(
                Path(other).resolve(), root.resolve(), "not the same directory"
            )

            uri = _resource_uri_for_temp_gltf(resource.resolve(), other)
            self.assertEqual(uri, "Assembly.bin")
            self.assertNotIn("..", uri)

    def test_a_resource_genuinely_elsewhere_still_gets_a_relative_path(self) -> None:
        """Only the same-directory case shortcuts; a real subdirectory still
        resolves relative, so this does not paper over actual layouts."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "resources"
            nested.mkdir()
            resource = nested / "Assembly.bin"
            resource.write_bytes(b"abc")
            self.assertEqual(
                _resource_uri_for_temp_gltf(resource.resolve(), root),
                "resources/Assembly.bin",
            )


class StepOnlyInputTests(unittest.TestCase):
    """STEP is the only input format offered, and the mesh path is internal.

    A mesh export carries surfaces rather than solids, so the voxelizer shells a
    part instead of filling it. CRYOSTAT_V2 was built that way and its DC gain is
    a rank-1 ~1e9 K/W artifact of the shells -- a number that looks like a plant
    model and is not one. Silently falling back to a mesh is what made that
    reachable by accident, so finding no STEP is now an error.

    The path itself has to stay, because the Thermal Validation tab generates its
    own geometry as a GLB and drives this CLI to build it. That is machinery, not
    a format choice, so it is gated behind a suppressed flag.
    """

    def test_no_step_file_is_an_error_not_a_mesh_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = build_parser().parse_args(
                ["--mesh-dir", tmp, "--graph-name", "x"]
            )
            with self.assertRaises(SystemExit) as caught:
                _resolve_step_path(args)
            message = str(caught.exception)
            self.assertIn("STEP", message)
            self.assertIn("shelled", message, "the message should say WHY")

    def test_a_step_file_is_found_by_extension(self) -> None:
        for suffix in (".step", ".stp", ".STEP"):
            with self.subTest(suffix=suffix):
                with tempfile.TemporaryDirectory() as tmp:
                    step = Path(tmp) / f"assembly{suffix}"
                    step.write_text("ISO-10303-21;", encoding="utf-8")
                    args = build_parser().parse_args(
                        ["--mesh-dir", tmp, "--graph-name", "x"]
                    )
                    self.assertEqual(_resolve_step_path(args), step)

    def test_the_mesh_path_stays_reachable_for_validation(self) -> None:
        """Thermal Validation builds through the REAL octree importer; without this
        it would fall back to its deterministic graph and stop validating the
        production path."""
        with tempfile.TemporaryDirectory() as tmp:
            args = build_parser().parse_args(
                ["--mesh-dir", tmp, "--graph-name", "x", "--allow-gltf-input"]
            )
            self.assertIsNone(_resolve_step_path(args))

    def test_the_mesh_escape_hatch_is_not_advertised(self) -> None:
        """It is internal machinery, so it must not appear in --help."""
        self.assertNotIn("allow-gltf-input", build_parser().format_help())

    def test_validation_passes_the_flag(self) -> None:
        """If this ever stops being passed, validation degrades silently to its
        fallback graph -- a warning, not a failure, so nothing would notice."""
        import inspect

        from graph_visualizer.thermal_validation import ThermalValidationExperiment

        source = inspect.getsource(ThermalValidationExperiment._build_octree_graph)
        self.assertIn("--allow-gltf-input", source)
