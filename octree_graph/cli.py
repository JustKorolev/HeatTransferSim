"""Command line entry point for glTF to octree thermal graph conversion."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import faulthandler
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import traceback
from typing import Any

import numpy as np
from scipy.sparse import issparse

from .graph_builder import (
    DEFAULT_CONTACT_INTERFACE_CONDUCTANCE_W_M2K,
    DEFAULT_MAX_HEATERS_PER_SENSOR,
    DEFAULT_ROLE_GROUP_GAP_MM,
    build_graph,
    collapse_role_components,
)
from .load_contact_report import load_contact_report
from .load_gltf import GltfScene, load_gltf_scene
from .materials import load_material_table
from .matrix_builder import DENSE_MATRIX_NODE_LIMIT, build_matrices
from .octree import OctreeCell, OctreeDiagnostics, OctreeParams, build_octree
from .validation import format_validation_report, validate_graph


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = Path(args.output_root) / args.graph_name
    output.mkdir(parents=True, exist_ok=True)
    with RunLogger(output / "conversion.log") as run_log:
        progress = ConsoleProgress(enabled=not args.no_progress, logger=run_log.log)
        try:
            _run_conversion(args, progress, run_log)
        except SystemExit as exc:
            run_log.log(f"Exiting: {exc}")
            raise
        except BaseException as exc:
            run_log.log(f"Unhandled exception: {type(exc).__name__}: {exc}")
            traceback.print_exc(file=run_log.handle)
            run_log.flush()
            raise


def _run_conversion(args: argparse.Namespace, progress: "ConsoleProgress", run_log: "RunLogger") -> None:
    warnings: list[str] = []
    output = Path(args.output_root) / args.graph_name
    checkpointer = BuildCheckpointer(output, args)
    checkpointer.phase("started", {"graph_name": args.graph_name})
    try:
        progress.phase("Loading glTF scene")
        gltf_path = _resolve_gltf_path(args)
        scene = load_gltf_scene(gltf_path)
        scene = _filter_ignored_components(scene, args, warnings)
        _log_scene_memory_risk(scene, args, run_log, warnings)
        progress.phase("Loading materials")
        material_lookup_path = _resolve_material_lookup_path(args.mesh_dir)
        contact_report = load_contact_report(material_lookup_path)
        materials, material_warnings = load_material_table(args.materials)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    warnings.extend(scene.warnings)
    warnings.extend(contact_report.warnings)
    warnings.extend(material_warnings)
    checkpointer.phase(
        "inputs_loaded",
        {
            "objects": len(getattr(scene, "objects", [])),
            "warnings": len(warnings),
            "materials": len(materials),
        },
    )
    voxel_scene, role_components = _split_role_components(scene, args, warnings)
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
        voxel_workers=args.voxel_workers,
        voxel_batch_size=args.voxel_batch_size,
        crowded_component_refine_count=args.crowded_component_refine_count,
        crowded_component_refine_distance_mm=args.crowded_component_refine_distance_mm,
        adaptive_refine_priority=args.adaptive_refine_priority,
        multi_surface_refine_count=args.multi_surface_refine_count,
        surface_complexity_refine_threshold=args.surface_complexity_refine_threshold,
        role_refine_component_names=_role_refine_component_names(role_components),
        role_refine_distance_mm=args.role_refine_distance_mm,
        role_refine_max_depth=args.role_refine_max_depth,
        contains_backend=args.contains_backend,
        balance_adjacent_leaf_sizes=args.balance_adjacent_leaf_sizes,
        max_adjacent_leaf_size_ratio=args.max_adjacent_leaf_size_ratio,
        balance_refine_passes=args.balance_refine_passes,
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
        checkpointer=checkpointer,
    )
    _raise_if_empty_graph(graph_result.nodes, leaves, args)
    connectivity_analysis = _graph_connectivity_analysis(graph_result.nodes, graph_result.edges)
    _annotate_graph_warning_tags(graph_result.nodes, graph_result.edges, args, connectivity_analysis)
    progress.phase("Building matrices")
    matrices = build_matrices(
        graph_result.nodes,
        graph_result.edges,
        dense_node_limit=args.dense_matrix_node_limit,
    )
    checkpointer.phase(
        "matrices_built",
        {
            "matrix_keys": sorted(str(key) for key in matrices),
            "sparse_keys": sorted(str(key) for key, value in matrices.items() if issparse(value)),
        },
    )
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
        "role_nodes": _role_components_payload(role_components),
        "octree_cells": [cell.__dict__ for cell in leaves],
        "graph_nodes": graph_result.nodes,
        "graph_edges": graph_result.edges,
        "warnings": graph_result.warnings,
        "heater_sensor_tags": {},
        "validation_results": {},
        "connectivity_analysis": connectivity_analysis,
    }
    graph["diagnostics"] = _diagnostics_payload(scene, leaves, graph_result, diagnostics, args)
    progress.phase("Validating graph")
    errors, validation_warnings = validate_graph(graph, matrices)
    graph["validation_results"] = {"errors": errors, "warnings": validation_warnings}
    graph["build_quality"] = _build_quality_report(graph, args)
    progress.phase("Writing outputs")
    _write_outputs(output, graph, matrices, materials, warnings)
    _atomic_write_text(
        output / "validation_report.txt",
        format_validation_report(graph, errors, validation_warnings),
    )
    if errors:
        raise SystemExit(f"Graph written with validation errors; see {output / 'validation_report.txt'}")
    checkpointer.phase(
        "completed",
        {
            "nodes": len(graph_result.nodes),
            "edges": len(graph_result.edges),
            "quality_grade": graph.get("build_quality", {}).get("quality_grade"),
            "quality_score": graph.get("build_quality", {}).get("quality_score"),
        },
    )
    progress.done()
    run_log.log(f"Completed graph with {len(graph_result.nodes)} nodes and {len(graph_result.edges)} edges.")
    print(f"Wrote octree graph with {len(graph_result.nodes)} nodes and {len(graph_result.edges)} edges to {output}")
    _print_role_summary(graph_result.nodes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", required=True, help="Directory containing exactly one embedded .glb scene file.")
    parser.add_argument("--materials", default="materials.json")
    parser.add_argument("--graph-name", required=True)
    parser.add_argument("--output-root", default="graphs")
    parser.add_argument(
        "--checkpoint-build",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write restart/debug checkpoints under build_checkpoints while generating the graph. "
            "Enabled by default."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval-s",
        type=float,
        default=30.0,
        help="Minimum seconds between octree progress checkpoint writes.",
    )
    parser.add_argument(
        "--dense-matrix-node-limit",
        type=int,
        default=DENSE_MATRIX_NODE_LIMIT,
        help=(
            "Write dense G.npy/L.npy only at or below this node count. Larger graphs write sparse "
            "L_sparse.json and skip dense conductance matrices to avoid RAM exhaustion. Use 0 to "
            "force sparse matrix output."
        ),
    )
    parser.add_argument("--min-cell-size-mm", type=float, default=5.0)
    parser.add_argument("--max-cell-size-mm", type=float, default=50.0)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument(
        "--ignore-component-substring",
        "--ignored-component-substring",
        dest="ignore_component_substring",
        action="append",
        default=None,
        help=(
            "Case-insensitive CAD component name/path substring to remove before role detection, "
            "voxelization, and graph building. Repeat to ignore multiple component substrings."
        ),
    )
    parser.add_argument("--dominant-fraction-accept", type=float, default=0.95)
    parser.add_argument("--minority-fraction-ignore", type=float, default=0.02)
    parser.add_argument("--material-contrast-refine-threshold", type=float, default=5.0)
    parser.add_argument("--contact-refine-distance-mm", type=float, default=10.0)
    parser.add_argument("--boundary-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-leaf-cells", type=int, default=None)
    parser.add_argument("--samples-per-cell", type=int, default=9)
    parser.add_argument("--min-solid-fraction", type=float, default=0.12)
    parser.add_argument(
        "--crowded-component-refine-count",
        type=int,
        default=0,
        help=(
            "If greater than 0, refine cells whose local neighborhood overlaps at least this many "
            "CAD component bounds. Useful for preserving empty gaps in dense small-part regions."
        ),
    )
    parser.add_argument(
        "--crowded-component-refine-distance-mm",
        type=float,
        default=0.0,
        help=(
            "Padding around each octree cell when counting nearby components for crowded-region refinement."
        ),
    )
    parser.add_argument(
        "--adaptive-refine-priority",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use geometry-risk scoring so max_leaf_cells budget is spent on ambiguous/high-detail regions first."
        ),
    )
    parser.add_argument(
        "--multi-surface-refine-count",
        type=int,
        default=2,
        help=(
            "Refine cells whose near-surface neighborhood contains surfaces from at least this many components. "
            "This targets narrow gaps and false-contact-prone regions."
        ),
    )
    parser.add_argument(
        "--surface-complexity-refine-threshold",
        type=int,
        default=64,
        help=(
            "Refine cells whose local triangle candidate count reaches this threshold. "
            "Use 0 to disable triangle-density refinement."
        ),
    )
    parser.add_argument(
        "--balance-adjacent-leaf-sizes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After adaptive voxelization, spend remaining leaf budget splitting coarse occupied leaves "
            "that directly touch much finer occupied leaves."
        ),
    )
    parser.add_argument(
        "--max-adjacent-leaf-size-ratio",
        type=float,
        default=4.0,
        help=(
            "Adjacent occupied leaves with a max-size ratio above this value trigger balancing refinement "
            "of the coarser leaf when budget allows."
        ),
    )
    parser.add_argument(
        "--balance-refine-passes",
        type=int,
        default=2,
        help="Maximum post-octree adjacent-size balancing passes.",
    )
    parser.add_argument(
        "--role-refine-distance-mm",
        type=float,
        default=0.0,
        help=(
            "Padding around detected heater/sensor CAD bounds for forced local octree refinement."
        ),
    )
    parser.add_argument(
        "--role-refine-max-depth",
        type=int,
        default=None,
        help=(
            "Optional depth cap for forced heater/sensor local refinement. "
            "Defaults to the global --max-depth."
        ),
    )
    parser.add_argument("--bbox-fallback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--voxel-workers",
        type=int,
        default=1,
        help=(
            "Worker processes for octree cell classification. "
            "Use 1 for sequential execution, or 0 for conservative auto selection capped at 2."
        ),
    )
    parser.add_argument(
        "--voxel-batch-size",
        type=int,
        default=64,
        help="Maximum queued octree cells classified per multiprocessing batch.",
    )
    parser.add_argument(
        "--contains-backend",
        choices=("trimesh", "ray"),
        default="ray",
        help=(
            "Inside/outside backend for watertight meshes. Defaults to 'ray' to avoid native "
            "trimesh.contains crashes on Windows; use 'trimesh' only when you need that backend."
        ),
    )
    parser.add_argument(
        "--contact-detection-distance-mm",
        type=float,
        default=None,
        help=(
            "Maximum voxel-surface gap for Python contact detection. "
            "Defaults to 0 for normal mesh-contained graphs and min_cell_size_mm for bbox-fallback graphs."
        ),
    )
    parser.add_argument(
        "--role-contact-tolerance-mm",
        type=float,
        default=1.0e-6,
        help=(
            "AABB contact tolerance for detected heater/sensor CAD parts that require dedicated "
            "fallback role nodes because they produced no solid voxel cells."
        ),
    )
    parser.add_argument(
        "--role-contact-tolerance-max-mm",
        type=float,
        default=1.0,
        help=(
            "Maximum AABB contact tolerance expansion for attaching detected heater/sensor role "
            "objects to body cells."
        ),
    )
    parser.add_argument(
        "--role-contact-tolerance-growth-factor",
        type=float,
        default=2.0,
        help="Multiplicative growth factor used while expanding role contact tolerance.",
    )
    parser.add_argument("--proximity-contact-distance-mm", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--contact-interface-conductance-W-m2K",
        dest="contact_interface_conductance_W_m2K",
        type=float,
        default=DEFAULT_CONTACT_INTERFACE_CONDUCTANCE_W_M2K,
        help=(
            "Thermal interface conductance used for inter-component contacts. "
            "Inter-component edge conductance is computed as 1/(L1/(k1*A) + 1/(h*A) + L2/(k2*A)); "
            "same-component internal conduction omits the interface term."
        ),
    )
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
            "Regex pattern for CAD component names/paths that should become heater nodes. "
            "Repeat to add multiple patterns. No heater detection is performed unless a heater pattern or substring is provided."
        ),
    )
    parser.add_argument(
        "--heater-name-substring",
        action="append",
        default=None,
        help=(
            "Case-insensitive CAD component name/path substring that should become heater nodes. "
            "Repeat to add multiple substrings."
        ),
    )
    parser.add_argument(
        "--sensor-name-pattern",
        action="append",
        default=None,
        help=(
            "Regex pattern for CAD component names/paths that should become sensor nodes. "
            "Repeat to add multiple patterns. No sensor detection is performed unless a sensor pattern or substring is provided."
        ),
    )
    parser.add_argument(
        "--sensor-name-substring",
        action="append",
        default=None,
        help=(
            "Case-insensitive CAD component name/path substring that should become sensor nodes. "
            "Repeat to add multiple substrings."
        ),
    )
    parser.add_argument(
        "--device-exclude-name-pattern",
        action="append",
        default=None,
        help=(
            "Regex pattern for CAD component names/paths that should not become heater/sensor nodes, "
            "even if they match heater or sensor words. Repeat to add multiple patterns."
        ),
    )
    parser.add_argument(
        "--no-default-device-excludes",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Default heater/sensor CAD exclusions are no longer "
            "applied; use --device-exclude-name-pattern for explicit exclusions."
        ),
    )
    parser.add_argument(
        "--role-node-group-gap-mm",
        type=float,
        default=DEFAULT_ROLE_GROUP_GAP_MM,
        help=(
            "Maximum AABB gap for collapsing same-named heater/sensor mesh pieces into one role node. "
            "Trailing instance numbers are preserved, so heater_1 and heater_2 stay separate."
        ),
    )
    parser.add_argument(
        "--max-heater-sensor-pair-distance-mm",
        type=float,
        default=25.0,
        help="Maximum AABB surface gap for automatically pairing each heater to one valid sensor.",
    )
    parser.add_argument(
        "--max-heaters-per-sensor",
        type=int,
        default=DEFAULT_MAX_HEATERS_PER_SENSOR,
        help=(
            "Maximum number of valid heaters that automatic pairing may assign to one sensor. "
            "The default preserves one-to-one pairing."
        ),
    )
    parser.add_argument(
        "--no-detect-role-nodes",
        action="store_true",
        help="Disable CAD name/path detection for heater and sensor graph nodes.",
    )
    parser.add_argument(
        "--no-detect-physical-devices",
        dest="no_detect_role_nodes",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _resolve_gltf_path(args: argparse.Namespace) -> Path:
    path = Path(args.mesh_dir)
    if not path.is_dir():
        raise ValueError(f"Expected --mesh-dir to be a directory, got {path}.")
    gltf_files = sorted(
        [
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() == ".gltf"
        ],
        key=lambda item: item.name.lower(),
    )
    if gltf_files:
        names = ", ".join(candidate.name for candidate in gltf_files[:8])
        extra = "" if len(gltf_files) <= 8 else f", ... and {len(gltf_files) - 8} more"
        raise ValueError(
            f"Mesh directory {path} contains .gltf file(s): {names}{extra}. "
            "External-buffer .gltf exports are no longer accepted for octree conversion; "
            "export exactly one embedded .glb file instead."
        )
    candidates = sorted(
        [
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() == ".glb"
        ],
        key=lambda item: item.name.lower(),
    )
    if not candidates:
        raise FileNotFoundError(f"No .glb file found in mesh directory {path}. Export exactly one embedded .glb.")
    if len(candidates) > 1:
        names = ", ".join(candidate.name for candidate in candidates[:8])
        extra = "" if len(candidates) <= 8 else f", ... and {len(candidates) - 8} more"
        raise ValueError(
            f"Mesh directory {path} contains multiple .glb files: {names}{extra}. "
            "Keep one embedded .glb scene file in the directory before running the generator."
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
    checkpointer: "BuildCheckpointer | None" = None,
):
    if args.bbox_fallback:
        warnings.append(
            "--bbox-fallback is retained for CLI compatibility but no longer creates occupied voxels; "
            "triangle-box surface intersection and watertight containment are used instead."
        )
    diagnostics = OctreeDiagnostics(debug_leaves=bool(getattr(args, "octree_debug_leaves", False)))
    if progress is not None:
        progress.phase("Voxelizing octree")

    def octree_progress(event: dict) -> None:
        if progress is not None:
            progress.octree(event)
        if checkpointer is not None:
            checkpointer.octree_progress(event)

    leaves = build_octree(
        voxel_scene,
        contact_report,
        materials,
        params,
        warnings,
        diagnostics=diagnostics,
        progress_callback=octree_progress if progress is not None or checkpointer is not None else None,
    )
    if checkpointer is not None:
        checkpointer.octree_complete(leaves, diagnostics)
    contact_distance_mm = _resolve_contact_detection_distance(args, params)
    if progress is not None:
        progress.phase("Building thermal graph")
    graph_result = build_graph(
        leaves,
        contact_report,
        materials,
        warnings,
        radiation_reference_temperature_K=args.radiation_reference_temperature_K,
        contact_interface_conductance_W_m2K=float(
            getattr(
                args,
                "contact_interface_conductance_W_m2K",
                DEFAULT_CONTACT_INTERFACE_CONDUCTANCE_W_M2K,
            )
        ),
        contact_detection_distance_mm=contact_distance_mm,
        component_bounds_mm=_component_bounds_mm(voxel_scene),
        role_components=getattr(args, "role_components", None),
        role_contact_tolerance_mm=args.role_contact_tolerance_mm,
        role_contact_tolerance_max_mm=float(getattr(args, "role_contact_tolerance_max_mm", args.role_contact_tolerance_mm)),
        role_contact_tolerance_growth_factor=float(getattr(args, "role_contact_tolerance_growth_factor", 2.0)),
        max_heater_sensor_pair_distance_mm=float(getattr(args, "max_heater_sensor_pair_distance_mm", 25.0)),
        max_heaters_per_sensor=int(getattr(args, "max_heaters_per_sensor", DEFAULT_MAX_HEATERS_PER_SENSOR)),
    )
    if checkpointer is not None:
        checkpointer.graph_complete(graph_result)
    return (leaves, graph_result, diagnostics) if include_diagnostics else (leaves, graph_result)


def _split_role_components(
    scene: GltfScene,
    args: argparse.Namespace,
    warnings: list[str],
) -> tuple[GltfScene, list]:
    if getattr(args, "no_detect_role_nodes", False):
        args.role_components = []
        return scene, []
    ignored_patterns = list(getattr(args, "heater_name_pattern", None) or []) + list(
        getattr(args, "sensor_name_pattern", None) or []
    )
    if ignored_patterns:
        warnings.append(
            "Ignoring --heater-name-pattern/--sensor-name-pattern because role detection is configured "
            "to use only --heater-name-substring and --sensor-name-substring."
        )
    heater_patterns = _substring_patterns(getattr(args, "heater_name_substring", None) or [])
    sensor_patterns = _substring_patterns(getattr(args, "sensor_name_substring", None) or [])
    if not heater_patterns and not sensor_patterns:
        args.role_components = []
        return scene, []
    exclude_patterns = list(getattr(args, "device_exclude_name_pattern", None) or [])
    group_gap_mm = float(getattr(args, "role_node_group_gap_mm", DEFAULT_ROLE_GROUP_GAP_MM))
    body_objects, role_components = collapse_role_components(
        scene.objects,
        heater_patterns,
        sensor_patterns,
        exclude_patterns=exclude_patterns,
        group_gap_mm=group_gap_mm,
    )
    args.role_components = role_components
    if not role_components:
        warnings.append(
            "No heater/sensor CAD components were detected from the configured name substrings; "
            "voxelization will include all mesh objects as body geometry."
        )
        return scene, []
    role_names = ", ".join(f"{component.kind}:{component.name}" for component in role_components[:8])
    extra = "" if len(role_components) <= 8 else f", ... and {len(role_components) - 8} more"
    role_summary = _role_detection_summary(role_components)
    warnings.append(
        f"Detected {len(role_components)} heater/sensor CAD component(s); "
        f"{role_summary}; "
        f"their CAD bounds will be attached to body cells while voxelization uses {len(body_objects)} body object(s): "
        f"{role_names}{extra}."
    )
    leaf_warning = _role_leaf_grouping_warning(role_components)
    if leaf_warning:
        warnings.append(leaf_warning)
    return _scene_with_objects(scene, body_objects), role_components


def _scene_with_objects(scene: GltfScene, objects: list) -> GltfScene:
    if not objects:
        return GltfScene(
            path=scene.path,
            objects=[],
            bounds_mm=scene.bounds_mm,
            warnings=list(scene.warnings),
        )
    bounds = _bounds_for_objects(objects)
    return GltfScene(
        path=scene.path,
        objects=list(objects),
        bounds_mm=bounds if bounds is not None else scene.bounds_mm,
        warnings=list(scene.warnings),
    )


def _filter_ignored_components(
    scene: GltfScene,
    args: argparse.Namespace,
    warnings: list[str],
) -> GltfScene:
    substrings = [
        _normalize_component_ignore_text(str(value))
        for value in (getattr(args, "ignore_component_substring", None) or [])
        if str(value).strip()
    ]
    if not substrings:
        setattr(args, "ignored_component_names", [])
        return scene
    kept = []
    ignored = []
    for obj in scene.objects:
        search_text = _ignored_component_search_text(obj)
        if any(substring in search_text for substring in substrings):
            ignored.append(obj)
        else:
            kept.append(obj)
    setattr(args, "ignored_component_names", [str(obj.name) for obj in ignored])
    if not ignored:
        warnings.append(
            "No CAD components matched --ignore-component-substring; "
            f"configured substring(s): {', '.join(getattr(args, 'ignore_component_substring', []) or [])}."
        )
        return scene
    examples = ", ".join(str(obj.name) for obj in ignored[:8])
    extra = "" if len(ignored) <= 8 else f", ... and {len(ignored) - 8} more"
    warnings.append(
        f"Ignored {len(ignored)} CAD mesh object(s) before role detection and voxelization "
        f"using --ignore-component-substring: {examples}{extra}."
    )
    if not kept:
        raise ValueError(
            "All CAD mesh objects were removed by --ignore-component-substring; "
            "relax the ignored component substring(s)."
        )
    return _scene_with_objects(scene, kept)


def _ignored_component_search_text(obj: Any) -> str:
    parts: list[str] = [str(getattr(obj, "name", ""))]
    scene_path = str(getattr(obj, "scene_path", "") or "")
    if scene_path:
        parts.append(scene_path)
    parts.extend(str(part) for part in (getattr(obj, "hierarchy_path", ()) or ()))
    return _normalize_component_ignore_text(" ".join(part for part in parts if part))


def _normalize_component_ignore_text(value: str) -> str:
    return str(value).replace("\\", "/").replace("-", "_").replace(" ", "_").lower()


def _substring_patterns(values: list[str]) -> list[str]:
    return [re.escape(_normalize_role_pattern_text(str(value))) for value in values if str(value).strip()]


def _normalize_role_pattern_text(value: str) -> str:
    return str(value).replace("\\", "/").replace("-", "_").replace(" ", "_")


def _role_detection_summary(role_components: list) -> str:
    by_kind = Counter(str(component.kind) for component in role_components)
    object_counts = [len(getattr(component, "objects", []) or []) for component in role_components]
    count_distribution = Counter(object_counts)
    common_counts = ", ".join(
        f"{count}obj:{frequency}" for count, frequency in sorted(count_distribution.items())[:4]
    )
    return f"roles_by_kind={dict(sorted(by_kind.items()))} objects_per_role={common_counts or 'none'}"


def _role_leaf_grouping_warning(role_components: list) -> str | None:
    by_kind = Counter(str(component.kind) for component in role_components)
    one_object_by_kind = Counter(
        str(component.kind)
        for component in role_components
        if len(getattr(component, "objects", []) or []) == 1
    )
    suspicious = [
        f"{kind}={count}"
        for kind, count in sorted(by_kind.items())
        if count >= 100 and one_object_by_kind.get(kind, 0) >= int(0.75 * count)
    ]
    if not suspicious:
        return None
    examples = ", ".join(
        f"{component.kind}:{component.name}"
        for component in role_components
        if len(getattr(component, "objects", []) or []) == 1
    )[:300]
    return (
        "Role detection appears to be grouping mostly one mesh leaf per role "
        f"({'; '.join(suspicious)}). This usually means the GLB hierarchy path was not recovered. "
        f"Examples: {examples}"
    )


def _bounds_for_objects(objects: list) -> tuple[np.ndarray, np.ndarray] | None:
    if not objects:
        return None
    mins = np.min([np.asarray(obj.bounds_mm[0], dtype=float) for obj in objects], axis=0)
    maxs = np.max([np.asarray(obj.bounds_mm[1], dtype=float) for obj in objects], axis=0)
    return mins, maxs


def _log_scene_memory_risk(scene: GltfScene, args: argparse.Namespace, run_log: "RunLogger", warnings: list[str]) -> None:
    worker_count = _resolved_cli_voxel_workers(args.voxel_workers)
    triangle_count = 0
    triangle_bytes = 0
    for obj in getattr(scene, "objects", []):
        triangles = np.asarray(getattr(getattr(obj, "mesh", None), "triangles", []), dtype=float)
        if triangles.ndim == 3 and triangles.shape[1:] == (3, 3):
            triangle_count += int(triangles.shape[0])
            triangle_bytes += int(triangles.nbytes)
    index_bytes = triangle_count * 2 * 3 * np.dtype(float).itemsize
    estimated_per_worker_bytes = triangle_bytes + index_bytes
    available_bytes = _available_memory_bytes()
    run_log.log(
        "Scene memory estimate: "
        f"objects={len(getattr(scene, 'objects', []))}, triangles={triangle_count}, "
        f"triangle_bytes={_format_bytes(triangle_bytes)}, "
        f"estimated_worker_payload={_format_bytes(estimated_per_worker_bytes)}, "
        f"requested_workers={args.voxel_workers}, resolved_workers={worker_count}, "
        f"available_memory={_format_bytes(available_bytes) if available_bytes is not None else 'unknown'}"
    )
    if worker_count <= 1 or available_bytes is None:
        return
    estimated_parallel_bytes = estimated_per_worker_bytes * worker_count
    if estimated_parallel_bytes <= available_bytes * 0.7:
        return
    message = (
        "Disabled multiprocessing because the estimated copied triangle/index payload "
        f"({_format_bytes(estimated_parallel_bytes)} across {worker_count} workers) is too large for "
        f"available memory ({_format_bytes(available_bytes)}). Using --voxel-workers 1."
    )
    warnings.append(message)
    run_log.log(message)
    args.voxel_workers = 1


def _resolved_cli_voxel_workers(requested: int) -> int:
    value = int(requested)
    if value == 0:
        cpu_count = os.cpu_count() or 2
        return max(1, min(2, cpu_count - 1))
    return max(1, value)


def _available_memory_bytes() -> int | None:
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            return None
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except (AttributeError, OSError, ValueError):
        return None


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024.0 or unit == "TiB":
            return f"{number:.1f} {unit}"
        number /= 1024.0


def _parameters_payload(args: argparse.Namespace) -> dict:
    payload = dict(vars(args))
    payload.pop("role_components", None)
    return payload


def _role_components_payload(role_components: list) -> list[dict]:
    payload = []
    for component in role_components:
        mins, maxs = component.bounds_mm
        payload.append(
            {
                "name": component.name,
                "kind": component.kind,
                "source_components": [obj.name for obj in component.objects],
                "bounds_mm": {
                    "min": np.asarray(mins, dtype=float).tolist(),
                    "max": np.asarray(maxs, dtype=float).tolist(),
                },
                "center_mm": np.asarray(component.center_mm, dtype=float).tolist(),
                "size_mm": np.asarray(component.size_mm, dtype=float).tolist(),
            }
        )
    return payload


def _role_refine_component_names(role_components: list) -> tuple[str, ...]:
    return tuple(sorted({obj.name for component in role_components for obj in component.objects}))


class BuildCheckpointer:
    """Write compact restart/debug checkpoints at safe build boundaries."""

    def __init__(self, output: Path, args: argparse.Namespace) -> None:
        self.enabled = bool(getattr(args, "checkpoint_build", True))
        self.output = Path(output)
        self.folder = self.output / "build_checkpoints"
        self.interval_s = max(1.0, float(getattr(args, "checkpoint_interval_s", 30.0)))
        self._last_octree_write = 0.0

    def phase(self, phase: str, payload: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        data = {
            "phase": str(phase),
            "timestamp_utc": _utc_timestamp(),
            "payload": _json_safe(payload or {}),
        }
        _atomic_write_json(self.folder / "latest.json", data, indent=2)
        _atomic_write_json(self.folder / f"{_safe_checkpoint_name(phase)}.json", data, indent=2)

    def octree_progress(self, event: dict) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not bool(event.get("done")) and now - self._last_octree_write < self.interval_s:
            return
        self._last_octree_write = now
        self.phase(
            "octree_progress",
            {
                "cells_tested": int(event.get("cells_tested", 0)),
                "cells_subdivided": int(event.get("cells_subdivided", 0)),
                "leaves": int(event.get("leaves", 0)),
                "queue": int(event.get("queue", 0)),
                "max_leaf_cells": event.get("max_leaf_cells"),
                "max_depth_reached": int(event.get("max_depth_reached", 0)),
                "voxel_workers": int(event.get("voxel_workers", 1)),
                "done": bool(event.get("done", False)),
            },
        )

    def octree_complete(self, leaves: list[OctreeCell], diagnostics: OctreeDiagnostics) -> None:
        if not self.enabled:
            return
        self.phase(
            "octree_complete",
            {
                "total_leaves": len(leaves),
                "solid_leaves": sum(1 for cell in leaves if not cell.is_empty),
                "empty_leaves": sum(1 for cell in leaves if cell.is_empty),
            },
        )
        _atomic_write_json(
            self.folder / "octree_leaves_complete.json",
            {
                "phase": "octree_complete",
                "timestamp_utc": _utc_timestamp(),
                "diagnostics": diagnostics.to_dict(),
                "octree_cells": [cell.__dict__ for cell in leaves],
            },
        )

    def graph_complete(self, graph_result: Any) -> None:
        if not self.enabled:
            return
        nodes = list(getattr(graph_result, "nodes", []) or [])
        edges = list(getattr(graph_result, "edges", []) or [])
        warnings = list(getattr(graph_result, "warnings", []) or [])
        self.phase(
            "graph_complete",
            {"nodes": len(nodes), "edges": len(edges), "warnings": len(warnings)},
        )
        _atomic_write_json(
            self.folder / "graph_complete.json",
            {
                "phase": "graph_complete",
                "timestamp_utc": _utc_timestamp(),
                "graph_nodes": nodes,
                "graph_edges": edges,
                "warnings": warnings,
            },
        )


def _safe_checkpoint_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("._") or "checkpoint"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def __enter__(self) -> "RunLogger":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")
        self.log("")
        self.log("Starting octree graph conversion")
        try:
            faulthandler.enable(file=self.handle, all_threads=True)
        except Exception as exc:
            self.log(f"Could not enable faulthandler: {type(exc).__name__}: {exc}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            faulthandler.disable()
        except Exception:
            pass
        self.log("Conversion process closing")
        self.flush()
        if self.handle is not None:
            self.handle.close()

    def log(self, message: str) -> None:
        if self.handle is None:
            return
        timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self.handle.write(f"[{timestamp}] {message}\n")
        self.flush()

    def flush(self) -> None:
        if self.handle is None:
            return
        self.handle.flush()
        try:
            os.fsync(self.handle.fileno())
        except OSError:
            pass


class ConsoleProgress:
    def __init__(self, enabled: bool = True, logger=None) -> None:
        self.enabled = enabled
        self.is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self._last_update = 0.0
        self._last_log = 0.0
        self._last_line_len = 0
        self._spinner_index = 0
        self._phase = ""
        self._logger = logger

    def phase(self, label: str) -> None:
        if self._logger is not None:
            self._logger(f"Phase: {label}")
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
        workers = int(event.get("voxel_workers", 1))
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
        if workers > 1:
            line += f" workers={workers}"
        if self._logger is not None and (event.get("done") or now - self._last_log >= 10.0):
            self._logger(line)
            self._last_log = now
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
    heaters = [node for node in nodes if bool(node.get("is_heater"))]
    sensors = [node for node in nodes if bool(node.get("is_sensor"))]
    connectivity = _graph_connectivity_analysis(nodes, edges)
    return {
        "node_count_before_pruning": len(nodes),
        "node_count_after_pruning": len(nodes),
        "edge_count": len(edges),
        "connected_components": int(connectivity.get("component_count", 0)),
        "heater_count": len(heaters),
        "sensor_count": len(sensors),
        "paired_heater_count": sum(1 for node in heaters if node.get("assigned_sensor_id") is not None),
        "valid_sensor_count": sum(1 for node in sensors if bool(node.get("sensor_valid", True))),
        "paired_sensor_count": sum(1 for node in sensors if node.get("assigned_heater_ids") or node.get("assigned_heater_id") is not None),
        "max_heaters_per_sensor": max((len(node.get("assigned_heater_ids") or []) for node in sensors), default=0),
    }


def _annotate_graph_warning_tags(
    nodes: list[dict],
    edges: list[dict],
    args: argparse.Namespace,
    connectivity_analysis: dict | None = None,
) -> None:
    degree: Counter[int] = Counter()
    for edge in edges:
        try:
            degree[int(edge["node_i"])] += 1
            degree[int(edge["node_j"])] += 1
        except (KeyError, TypeError, ValueError):
            continue
    disconnected_ids = {
        int(node_id)
        for node_id in (connectivity_analysis or {}).get("disconnected_node_ids", []) or []
    }
    for node in nodes:
        tags = _warning_tags_for_node(node, degree, args, disconnected_ids)
        node.setdefault("tags", {})
        if isinstance(node["tags"], dict):
            node["tags"]["warning_tags"] = tags


def _warning_tags_for_node(
    node: dict,
    degree: Counter[int],
    args: argparse.Namespace,
    disconnected_ids: set[int] | None = None,
) -> list[str]:
    tags: list[str] = []
    node_id = int(node.get("node_id", -1))
    size = node.get("size_mm") or []
    try:
        if max(float(value) for value in size) > float(args.max_cell_size_mm):
            tags.append("oversized_cell")
    except (TypeError, ValueError):
        pass
    if str(node.get("confidence", "high") or "high").lower() != "high":
        tags.append("low_confidence")
    if node.get("warnings"):
        tags.append("node_warning")
    if degree.get(node_id, 0) <= 0:
        tags.append("isolated_node")
    if disconnected_ids and node_id in disconnected_ids:
        tags.append("disconnected_component")
    if bool(node.get("is_heater")):
        if not bool(node.get("heater_valid", True)) or not bool(node.get("heater_attached", True)):
            tags.append("invalid_heater")
        if node.get("assigned_sensor_id") is None:
            tags.append("unpaired_heater")
    if bool(node.get("is_sensor")):
        if not bool(node.get("sensor_valid", True)):
            tags.append("invalid_sensor")
        if not (node.get("assigned_heater_ids") or node.get("assigned_heater_id") is not None):
            tags.append("unpaired_sensor")
    deduped: list[str] = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped


def _build_quality_report(graph: dict, args: argparse.Namespace) -> dict:
    nodes = list(graph.get("graph_nodes", []) or [])
    edges = list(graph.get("graph_edges", []) or [])
    validation = graph.get("validation_results", {}) or {}
    tag_counts: Counter[str] = Counter()
    for node in nodes:
        for tag in _node_warning_tags(node):
            tag_counts[str(tag)] += 1
    warning_count = len(graph.get("warnings", []) or [])
    validation_errors = len(validation.get("errors", []) or [])
    validation_warnings = len(validation.get("warnings", []) or [])
    score = 100
    score -= min(40, validation_errors * 20)
    score -= min(25, tag_counts.get("oversized_cell", 0) * 2)
    score -= min(20, tag_counts.get("low_confidence", 0))
    score -= min(15, tag_counts.get("isolated_node", 0) * 2)
    score -= min(25, tag_counts.get("disconnected_component", 0) * 2)
    score -= min(15, (tag_counts.get("unpaired_heater", 0) + tag_counts.get("unpaired_sensor", 0)) * 2)
    score -= min(10, validation_warnings)
    score = max(0, int(score))
    grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D"
    largest_nodes = sorted(
        (
            {
                "node_id": int(node.get("node_id", -1)),
                "component_name": node.get("component_name", ""),
                "material_name": node.get("material_name", ""),
                "max_size_mm": _max_node_size_mm(node),
                "confidence": node.get("confidence", ""),
                "warning_tags": _node_warning_tags(node),
            }
            for node in nodes
        ),
        key=lambda item: float(item["max_size_mm"]),
        reverse=True,
    )[:20]
    return {
        "quality_score": score,
        "quality_grade": grade,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "warning_count": warning_count,
            "validation_error_count": validation_errors,
            "validation_warning_count": validation_warnings,
            "requested_max_cell_size_mm": float(args.max_cell_size_mm),
        },
        "node_warning_tag_counts": dict(sorted(tag_counts.items())),
        "largest_nodes": largest_nodes,
        "blocking_issues": list(validation.get("errors", []) or []),
    }


def _node_warning_tags(node: dict) -> list[str]:
    tags = node.get("tags", {}) if isinstance(node.get("tags"), dict) else {}
    values = tags.get("warning_tags", node.get("warning_tags", []))
    return [str(value) for value in (values or [])]


def _max_node_size_mm(node: dict) -> float:
    try:
        return max(float(value) for value in (node.get("size_mm") or []))
    except (TypeError, ValueError):
        return 0.0


def _print_role_summary(nodes: list[dict]) -> None:
    heaters = [node for node in nodes if bool(node.get("is_heater"))]
    sensors = [node for node in nodes if bool(node.get("is_sensor"))]
    paired_heaters = [node for node in heaters if node.get("assigned_sensor_id") is not None]
    valid_sensors = [node for node in sensors if bool(node.get("sensor_valid", True))]
    paired_sensors = [node for node in sensors if node.get("assigned_heater_ids") or node.get("assigned_heater_id") is not None]
    max_heaters_per_sensor = max((len(node.get("assigned_heater_ids") or []) for node in sensors), default=0)
    print(
        "Heater/sensor detection: "
        f"heaters={len(heaters)} paired_heaters={len(paired_heaters)} "
        f"sensors={len(sensors)} valid_sensors={len(valid_sensors)} paired_sensors={len(paired_sensors)} "
        f"max_heaters_per_sensor={max_heaters_per_sensor}"
    )


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


def _graph_connectivity_analysis(nodes: list[dict], edges: list[dict]) -> dict:
    node_ids = sorted(_safe_node_id(node) for node in nodes if _safe_node_id(node) is not None)
    if not node_ids:
        return {
            "connected": True,
            "component_count": 0,
            "largest_component_id": None,
            "largest_component_size": 0,
            "disconnected_node_ids": [],
            "components": [],
        }
    adjacency: dict[int, set[int]] = {node_id: set() for node_id in node_ids}
    valid_edge_count = 0
    for edge in edges:
        try:
            a = int(edge["node_i"])
            b = int(edge["node_j"])
        except (KeyError, TypeError, ValueError):
            continue
        if a not in adjacency or b not in adjacency:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
        valid_edge_count += 1
    nodes_by_id = {int(node["node_id"]): node for node in nodes if _safe_node_id(node) is not None}
    seen: set[int] = set()
    raw_components: list[list[int]] = []
    for node_id in node_ids:
        if node_id in seen:
            continue
        stack = [node_id]
        seen.add(node_id)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        raw_components.append(sorted(component))
    raw_components.sort(key=lambda ids: (-len(ids), ids[0] if ids else -1))
    largest = raw_components[0] if raw_components else []
    largest_id = 0 if raw_components else None
    disconnected_ids = [node_id for component in raw_components[1:] for node_id in component]
    components = [
        _connectivity_component_summary(index, component, nodes_by_id)
        for index, component in enumerate(raw_components)
    ]
    return {
        "connected": len(raw_components) <= 1,
        "component_count": len(raw_components),
        "largest_component_id": largest_id,
        "largest_component_size": len(largest),
        "disconnected_node_ids": disconnected_ids,
        "node_count": len(node_ids),
        "edge_count": valid_edge_count,
        "components": components,
    }


def _connectivity_component_summary(component_id: int, node_ids: list[int], nodes_by_id: dict[int, dict]) -> dict:
    component_nodes = [nodes_by_id[node_id] for node_id in node_ids if node_id in nodes_by_id]
    centers = [
        np.asarray(node.get("center_mm"), dtype=float)
        for node in component_nodes
        if isinstance(node.get("center_mm"), (list, tuple)) and len(node.get("center_mm")) == 3
    ]
    bounds: dict[str, Any] | None = None
    if centers:
        stacked = np.vstack(centers)
        bounds = {"min": np.min(stacked, axis=0).tolist(), "max": np.max(stacked, axis=0).tolist()}
    components = Counter(str(node.get("component_name", "") or "?") for node in component_nodes)
    materials = Counter(str(node.get("material_name", "") or "?") for node in component_nodes)
    return {
        "component_id": int(component_id),
        "node_count": len(node_ids),
        "node_ids_sample": node_ids[:50],
        "node_ids_truncated": len(node_ids) > 50,
        "heater_count": sum(1 for node in component_nodes if bool(node.get("is_heater"))),
        "sensor_count": sum(1 for node in component_nodes if bool(node.get("is_sensor"))),
        "bounds_center_mm": bounds,
        "top_components": components.most_common(10),
        "top_materials": materials.most_common(10),
    }


def _safe_node_id(node: dict) -> int | None:
    try:
        return int(node["node_id"])
    except (KeyError, TypeError, ValueError):
        return None


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
            "The max_leaf_cells cap limits optional adaptive refinement; mandatory occupied-cell "
            "subdivision still runs until max_cell_size_mm, max_depth, or min_cell_size_mm stops it."
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
    _atomic_write_json(output / "build_quality.json", graph.get("build_quality", {}), indent=2)
    _atomic_write_json(output / "connectivity_analysis.json", graph.get("connectivity_analysis", {}), indent=2)
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
        if issparse(value):
            continue
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
        L_value = matrices["L"]
        if issparse(L_value):
            L_coo = L_value.tocoo()
            payload = {
                "shape": list(L_coo.shape),
                "format": "coo",
                "row": L_coo.row.astype(int).tolist(),
                "col": L_coo.col.astype(int).tolist(),
                "data": L_coo.data.astype(float).tolist(),
            }
            _atomic_write_json(output / "L_sparse.json", payload)
            return
        L = np.asarray(L_value, dtype=float)
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
