"""Command line entry point for glTF to octree thermal graph conversion."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np

from .graph_builder import (
    DEFAULT_HEATER_NAME_PATTERNS,
    DEFAULT_SENSOR_NAME_PATTERNS,
    build_graph,
    collapse_physical_devices,
)
from .load_contact_report import load_contact_report
from .load_gltf import GltfScene, load_gltf_scene
from .materials import load_material_table
from .matrix_builder import build_matrices
from .octree import OctreeDiagnostics, OctreeParams, build_octree
from .validation import format_validation_report, validate_graph


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    progress = ConsoleProgress(enabled=not args.no_progress)
    warnings: list[str] = []
    try:
        progress.phase("Loading glTF scene")
        gltf_path = _resolve_gltf_path(args)
        scene = load_gltf_scene(gltf_path)
        progress.phase("Loading materials")
        material_lookup_path = _resolve_material_lookup_path(args.mesh_dir)
        contact_report = load_contact_report(material_lookup_path)
        materials, material_warnings = load_material_table(args.materials)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    warnings.extend(scene.warnings)
    warnings.extend(contact_report.warnings)
    warnings.extend(material_warnings)
    voxel_scene, physical_devices = _split_physical_devices(scene, args, warnings)
    params = OctreeParams(
        min_cell_size_mm=args.min_cell_size_mm,
        max_cell_size_mm=args.max_cell_size_mm,
        max_depth=args.max_depth,
        dominant_fraction_accept=args.dominant_fraction_accept,
        minority_fraction_ignore=args.minority_fraction_ignore,
        material_contrast_refine_threshold=args.material_contrast_refine_threshold,
        contact_refine_distance_mm=args.contact_refine_distance_mm,
        boundary_refine=args.boundary_refine,
        max_leaf_cells=args.max_leaf_cells,
        samples_per_cell=args.samples_per_cell,
        min_solid_fraction=args.min_solid_fraction,
        bbox_fallback=args.bbox_fallback,
    )
    leaves, graph_result, diagnostics = _build_graph_with_optional_fallback(
        scene,
        voxel_scene,
        contact_report,
        materials,
        params,
        args,
        warnings,
        include_diagnostics=True,
        progress=progress,
    )
    _raise_if_empty_graph(graph_result.nodes, leaves, args)
    progress.phase("Building matrices")
    matrices = build_matrices(graph_result.nodes, graph_result.edges)
    output = Path(args.output_root) / args.graph_name
    output.mkdir(parents=True, exist_ok=True)
    input_files = {
        "gltf": str(gltf_path),
        "materials": str(Path(args.materials)),
    }
    if material_lookup_path:
        input_files["material_lookup"] = str(Path(material_lookup_path))
    graph = {
        "metadata": {"graph_name": args.graph_name, "app_version": "octree_graph 0.1"},
        "input_files": input_files,
        "parameters": _parameters_payload(args),
        "materials_used": {name: material.to_dict() for name, material in materials.items()},
        "component_mapping": {obj.name: obj.name for obj in scene.objects},
        "physical_devices": _physical_devices_payload(physical_devices),
        "octree_cells": [cell.__dict__ for cell in leaves],
        "graph_nodes": graph_result.nodes,
        "graph_edges": graph_result.edges,
        "warnings": graph_result.warnings,
        "heater_sensor_tags": {},
        "validation_results": {},
    }
    graph["diagnostics"] = _diagnostics_payload(scene, leaves, graph_result, diagnostics, args)
    progress.phase("Validating graph")
    errors, validation_warnings = validate_graph(graph, matrices)
    graph["validation_results"] = {"errors": errors, "warnings": validation_warnings}
    progress.phase("Writing outputs")
    _write_outputs(output, graph, matrices, materials, warnings)
    _atomic_write_text(
        output / "validation_report.txt",
        format_validation_report(graph, errors, validation_warnings),
    )
    if errors:
        raise SystemExit(f"Graph written with validation errors; see {output / 'validation_report.txt'}")
    progress.done()
    print(f"Wrote octree graph with {len(graph_result.nodes)} nodes and {len(graph_result.edges)} edges to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", required=True, help="Directory containing exactly one .gltf/.glb and its .bin resources.")
    parser.add_argument("--materials", default="materials.json")
    parser.add_argument("--graph-name", required=True)
    parser.add_argument("--output-root", default="graphs")
    parser.add_argument("--min-cell-size-mm", type=float, default=5.0)
    parser.add_argument("--max-cell-size-mm", type=float, default=50.0)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--dominant-fraction-accept", type=float, default=0.95)
    parser.add_argument("--minority-fraction-ignore", type=float, default=0.02)
    parser.add_argument("--material-contrast-refine-threshold", type=float, default=5.0)
    parser.add_argument("--contact-refine-distance-mm", type=float, default=10.0)
    parser.add_argument("--boundary-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-leaf-cells", type=int, default=None)
    parser.add_argument("--samples-per-cell", type=int, default=9)
    parser.add_argument("--min-solid-fraction", type=float, default=0.12)
    parser.add_argument("--bbox-fallback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--contact-detection-distance-mm",
        type=float,
        default=None,
        help=(
            "Maximum voxel-surface gap for Python contact detection. "
            "Defaults to 0 for normal mesh-contained graphs and min_cell_size_mm for bbox-fallback graphs."
        ),
    )
    parser.add_argument("--proximity-contact-distance-mm", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--radiation-reference-temperature-K", type=float, default=293.15)
    parser.add_argument(
        "--octree-debug-leaves",
        action="store_true",
        help="Include per-solid-leaf octree acceptance records in octree_diagnostics.json.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable console progress output.")
    parser.add_argument(
        "--heater-name-pattern",
        action="append",
        default=None,
        help=(
            "Regex pattern for CAD component names/paths that should become physical heater nodes. "
            "Repeat to add multiple patterns. Defaults cover common heater names."
        ),
    )
    parser.add_argument(
        "--sensor-name-pattern",
        action="append",
        default=None,
        help=(
            "Regex pattern for CAD component names/paths that should become physical sensor nodes. "
            "Repeat to add multiple patterns. Defaults cover common sensor names."
        ),
    )
    parser.add_argument(
        "--no-detect-physical-devices",
        action="store_true",
        help="Disable CAD name/path detection for physical heater and sensor graph nodes.",
    )
    return parser


def _resolve_gltf_path(args: argparse.Namespace) -> Path:
    path = Path(args.mesh_dir)
    if not path.is_dir():
        raise ValueError(f"Expected --mesh-dir to be a directory, got {path}.")
    candidates = sorted(
        [
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in {".gltf", ".glb"}
        ],
        key=lambda item: item.name.lower(),
    )
    if not candidates:
        raise FileNotFoundError(f"No .gltf or .glb file found in mesh directory {path}.")
    if len(candidates) > 1:
        names = ", ".join(candidate.name for candidate in candidates[:8])
        extra = "" if len(candidates) <= 8 else f", ... and {len(candidates) - 8} more"
        raise ValueError(
            f"Mesh directory {path} contains multiple .gltf/.glb files: {names}{extra}. "
            "Keep one mesh scene file in the directory before running the generator."
        )
    return candidates[0]


def _resolve_material_lookup_path(mesh_dir: str | Path) -> Path | None:
    path = Path(mesh_dir) / "materials.xlsx"
    return path if path.is_file() else None


def _build_graph_with_optional_fallback(
    scene,
    voxel_scene,
    contact_report,
    materials,
    params: OctreeParams,
    args: argparse.Namespace,
    warnings: list[str],
    include_diagnostics: bool = False,
    progress: "ConsoleProgress | None" = None,
):
    if args.bbox_fallback:
        warnings.append(
            "--bbox-fallback is retained for CLI compatibility but no longer creates occupied voxels; "
            "triangle-box surface intersection and watertight containment are used instead."
        )
    diagnostics = OctreeDiagnostics(debug_leaves=bool(getattr(args, "octree_debug_leaves", False)))
    if progress is not None:
        progress.phase("Voxelizing octree")
    leaves = build_octree(
        voxel_scene,
        contact_report,
        materials,
        params,
        warnings,
        diagnostics=diagnostics,
        progress_callback=progress.octree if progress is not None else None,
    )
    contact_distance_mm = _resolve_contact_detection_distance(args, params)
    if progress is not None:
        progress.phase("Building thermal graph")
    graph_result = build_graph(
        leaves,
        contact_report,
        materials,
        warnings,
        radiation_reference_temperature_K=args.radiation_reference_temperature_K,
        contact_detection_distance_mm=contact_distance_mm,
        component_bounds_mm=_component_bounds_mm(voxel_scene),
        physical_devices=getattr(args, "physical_devices", None),
    )
    return (leaves, graph_result, diagnostics) if include_diagnostics else (leaves, graph_result)


def _split_physical_devices(
    scene: GltfScene,
    args: argparse.Namespace,
    warnings: list[str],
) -> tuple[GltfScene, list]:
    if getattr(args, "no_detect_physical_devices", False):
        args.physical_devices = []
        return scene, []
    heater_patterns = list(getattr(args, "heater_name_pattern", None) or DEFAULT_HEATER_NAME_PATTERNS)
    sensor_patterns = list(getattr(args, "sensor_name_pattern", None) or DEFAULT_SENSOR_NAME_PATTERNS)
    body_objects, devices = collapse_physical_devices(scene.objects, heater_patterns, sensor_patterns)
    args.physical_devices = devices
    if not devices:
        return scene, []
    device_names = ", ".join(f"{device.kind}:{device.name}" for device in devices[:8])
    extra = "" if len(devices) <= 8 else f", ... and {len(devices) - 8} more"
    warnings.append(
        f"Detected {len(devices)} physical heater/sensor CAD component(s); "
        f"excluded them from voxelization and added dedicated graph nodes: {device_names}{extra}."
    )
    bounds = _bounds_for_objects(body_objects) or scene.bounds_mm
    return GltfScene(path=scene.path, objects=body_objects, bounds_mm=bounds, warnings=scene.warnings), devices


def _bounds_for_objects(objects: list) -> tuple[np.ndarray, np.ndarray] | None:
    if not objects:
        return None
    mins = np.min([np.asarray(obj.bounds_mm[0], dtype=float) for obj in objects], axis=0)
    maxs = np.max([np.asarray(obj.bounds_mm[1], dtype=float) for obj in objects], axis=0)
    return mins, maxs


def _parameters_payload(args: argparse.Namespace) -> dict:
    payload = dict(vars(args))
    payload.pop("physical_devices", None)
    return payload


def _physical_devices_payload(devices: list) -> list[dict]:
    payload = []
    for device in devices:
        mins, maxs = device.bounds_mm
        payload.append(
            {
                "name": device.name,
                "kind": device.kind,
                "source_components": [obj.name for obj in device.objects],
                "bounds_mm": {
                    "min": np.asarray(mins, dtype=float).tolist(),
                    "max": np.asarray(maxs, dtype=float).tolist(),
                },
                "center_mm": np.asarray(device.center_mm, dtype=float).tolist(),
                "size_mm": np.asarray(device.size_mm, dtype=float).tolist(),
            }
        )
    return payload


class ConsoleProgress:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self._last_update = 0.0
        self._last_line_len = 0
        self._spinner_index = 0
        self._phase = ""

    def phase(self, label: str) -> None:
        if not self.enabled:
            return
        self._finish_line()
        self._phase = label
        print(f"{label}...", file=sys.stderr, flush=True)

    def octree(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not event.get("done") and now - self._last_update < 0.2:
            return
        self._last_update = now
        cells = int(event.get("cells_tested", 0))
        leaves = int(event.get("leaves", 0))
        queue = int(event.get("queue", 0))
        subdivided = int(event.get("cells_subdivided", 0))
        depth = int(event.get("max_depth_reached", 0))
        max_leaf_cells = event.get("max_leaf_cells")
        active_total = max(1, cells + queue)
        ratio = min(1.0, cells / float(active_total))
        width = 28
        filled = int(round(ratio * width))
        bar = "#" * filled + "-" * (width - filled)
        if isinstance(max_leaf_cells, int) and max_leaf_cells > 0:
            leaf_text = f"leaves={leaves}/{max_leaf_cells}"
        else:
            leaf_text = f"leaves={leaves}"
        line = (
            f"Voxelizing octree [{bar}] active={ratio * 100:5.1f}% "
            f"tested={cells} {leaf_text} queue={queue} subdivided={subdivided} depth={depth}"
        )
        if self.is_tty:
            padded = line.ljust(self._last_line_len)
            print(f"\r{padded}", end="", file=sys.stderr, flush=True)
            self._last_line_len = len(line)
            if event.get("done"):
                print(file=sys.stderr, flush=True)
                self._last_line_len = 0
        else:
            print(line, file=sys.stderr, flush=True)

    def done(self) -> None:
        if not self.enabled:
            return
        self._finish_line()

    def _finish_line(self) -> None:
        if self.is_tty and self._last_line_len:
            print(file=sys.stderr, flush=True)
            self._last_line_len = 0


def _diagnostics_payload(
    scene,
    leaves: list,
    graph_result,
    diagnostics: OctreeDiagnostics,
    args: argparse.Namespace,
) -> dict:
    payload = diagnostics.to_dict()
    payload.update(
        {
            "mesh_summary": _mesh_diagnostics(scene),
            "octree_summary": {
                "total_leaves": len(leaves),
                "solid_leaves": sum(1 for cell in leaves if not cell.is_empty),
                "empty_leaves": sum(1 for cell in leaves if cell.is_empty),
                "requested_min_cell_size_mm": float(args.min_cell_size_mm),
                "requested_max_cell_size_mm": float(args.max_cell_size_mm),
                "requested_max_depth": int(args.max_depth),
                "requested_samples_per_cell": int(args.samples_per_cell),
                "bbox_fallback_enabled": bool(args.bbox_fallback),
            },
            "graph_summary": _graph_diagnostics(graph_result.nodes, graph_result.edges),
        }
    )
    return payload


def _mesh_diagnostics(scene) -> dict:
    meshes = []
    for obj in getattr(scene, "objects", []):
        mesh = getattr(obj, "mesh", None)
        triangles = getattr(mesh, "triangles", [])
        vertices = getattr(mesh, "vertices", [])
        meshes.append(
            {
                "name": obj.name,
                "material_name": obj.material_name,
                "watertight": bool(obj.watertight),
                "triangle_count": int(len(triangles)),
                "vertex_count": int(len(vertices)),
                "bounds_mm": {
                    "min": np.asarray(obj.bounds_mm[0], dtype=float).tolist(),
                    "max": np.asarray(obj.bounds_mm[1], dtype=float).tolist(),
                },
            }
        )
    mins, maxs = scene.bounds_mm
    return {
        "mesh_count": len(meshes),
        "triangle_count": sum(mesh["triangle_count"] for mesh in meshes),
        "watertight_count": sum(1 for mesh in meshes if mesh["watertight"]),
        "scene_bounds_mm": {
            "min": np.asarray(mins, dtype=float).tolist(),
            "max": np.asarray(maxs, dtype=float).tolist(),
        },
        "meshes": meshes,
    }


def _graph_diagnostics(nodes: list[dict], edges: list[dict]) -> dict:
    return {
        "node_count_before_pruning": len(nodes),
        "node_count_after_pruning": len(nodes),
        "edge_count": len(edges),
        "connected_components": _connected_component_count(nodes, edges),
    }


def _connected_component_count(nodes: list[dict], edges: list[dict]) -> int:
    if not nodes:
        return 0
    node_ids = {int(node["node_id"]) for node in nodes}
    parent = {node_id: node_id for node_id in node_ids}

    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for edge in edges:
        node_i = int(edge["node_i"])
        node_j = int(edge["node_j"])
        if node_i in parent and node_j in parent:
            union(node_i, node_j)
    return len({find(node_id) for node_id in node_ids})


def _configured_contact_detection_distance(args: argparse.Namespace) -> float | None:
    if args.contact_detection_distance_mm is not None:
        return float(args.contact_detection_distance_mm)
    if args.proximity_contact_distance_mm is not None:
        return float(args.proximity_contact_distance_mm)
    return None


def _resolve_contact_detection_distance(args: argparse.Namespace, params: OctreeParams) -> float:
    configured = _configured_contact_detection_distance(args)
    if configured is not None:
        return max(0.0, configured)
    return 0.0


def _component_bounds_mm(scene) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        obj.name: (np.asarray(obj.bounds_mm[0], dtype=float), np.asarray(obj.bounds_mm[1], dtype=float))
        for obj in getattr(scene, "objects", [])
    }


def _raise_if_empty_graph(nodes: list[dict], leaves: list, args: argparse.Namespace) -> None:
    if nodes:
        return
    if not leaves:
        raise SystemExit("Octree generation produced no leaves; check that the glTF scene has valid mesh bounds.")
    guidance = [
        f"Octree generation produced {len(leaves)} leaves, but none were classified as solid graph nodes.",
        "This usually means no triangle-box surface intersections or watertight interiors were found before refinement stopped.",
    ]
    if args.max_leaf_cells is not None:
        guidance.append(
            "The max_leaf_cells cap can also stop refinement while cells are still unresolved; increase "
            "--max-leaf-cells or use a coarser min/max cell-size range if needed."
        )
    guidance.append("Check octree_diagnostics.json for surface-hit, bbox-only, and depth counters.")
    raise SystemExit(" ".join(guidance))


def _write_outputs(
    output: Path,
    graph: dict,
    matrices: dict[str, np.ndarray],
    materials: dict,
    warnings: list[str],
) -> None:
    _atomic_write_json(output / "graph.json", graph, indent=2)
    _atomic_write_json(output / "octree_diagnostics.json", graph.get("diagnostics", {}), indent=2)
    _write_csv(output / "nodes.csv", graph["graph_nodes"])
    _write_csv(output / "edges.csv", graph["graph_edges"])
    _atomic_write_json(output / "params.json", graph["parameters"], indent=2)
    _atomic_write_json(
        output / "materials_used.json",
        {name: material.to_dict() for name, material in materials.items()},
        indent=2,
    )
    with (output / "material_warnings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["warning"])
        writer.writerows([[warning] for warning in warnings])
    for key, value in matrices.items():
        np.save(output / f"{key}.npy", value)
    _write_browser_matrix_exports(output, matrices)
    (output / "simulations").mkdir(exist_ok=True)
    _atomic_write_json(output / "ui_state.json", {"selected_node_id": None, "filters": {}}, indent=2)


def _write_csv(path: Path, rows: list[dict]) -> None:
    flattened = [_flatten_for_csv(row) for row in rows]
    fields = sorted({key for row in flattened for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flattened)


def _flatten_for_csv(row: dict) -> dict:
    flattened = dict(row)
    radiation = flattened.pop("radiation", None)
    if isinstance(radiation, dict):
        for key, value in radiation.items():
            flattened[f"radiation_{key}"] = value
    return flattened


def _write_browser_matrix_exports(output: Path, matrices: dict[str, np.ndarray]) -> None:
    if "C" in matrices:
        _atomic_write_json(output / "C_diag.json", {"data": matrices["C"].tolist()})
    if "G_rad" in matrices:
        _atomic_write_json(output / "G_rad_diag.json", {"data": np.asarray(matrices["G_rad"], dtype=float).tolist()})
    if "L" in matrices:
        L = np.asarray(matrices["L"], dtype=float)
        row, col = np.nonzero(L)
        payload = {
            "shape": list(L.shape),
            "format": "coo",
            "row": row.astype(int).tolist(),
            "col": col.astype(int).tolist(),
            "data": L[row, col].astype(float).tolist(),
        }
        _atomic_write_json(output / "L_sparse.json", payload)


def _atomic_write_json(path: Path, payload: object, indent: int | None = None) -> None:
    text = json.dumps(payload, indent=indent) + "\n"
    _atomic_write_text(path, text)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    main()
