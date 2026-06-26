"""Command line entry point for glTF to octree thermal graph conversion."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .graph_builder import build_graph
from .load_contact_report import load_contact_report
from .load_gltf import load_gltf_scene
from .materials import load_material_table
from .matrix_builder import build_matrices
from .octree import OctreeParams, build_octree
from .validation import format_validation_report, validate_graph


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    warnings: list[str] = []
    try:
        scene = load_gltf_scene(args.gltf)
        contact_report = load_contact_report(args.contact_report)
        materials, material_warnings = load_material_table(args.materials)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    warnings.extend(scene.warnings)
    warnings.extend(contact_report.warnings)
    warnings.extend(material_warnings)
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
    leaves = build_octree(scene, contact_report, materials, params, warnings)
    graph_result = build_graph(
        leaves,
        contact_report,
        materials,
        warnings,
        radiation_reference_temperature_K=args.radiation_reference_temperature_K,
    )
    matrices = build_matrices(graph_result.nodes, graph_result.edges)
    output = Path(args.output_root) / args.graph_name
    output.mkdir(parents=True, exist_ok=True)
    graph = {
        "metadata": {"graph_name": args.graph_name, "app_version": "octree_graph 0.1"},
        "input_files": {
            "gltf": str(Path(args.gltf)),
            "contact_report": str(Path(args.contact_report)) if args.contact_report else None,
            "materials": str(Path(args.materials)),
        },
        "parameters": vars(args),
        "materials_used": {name: material.to_dict() for name, material in materials.items()},
        "component_mapping": {obj.name: obj.name for obj in scene.objects},
        "octree_cells": [cell.__dict__ for cell in leaves],
        "graph_nodes": graph_result.nodes,
        "graph_edges": graph_result.edges,
        "warnings": graph_result.warnings,
        "heater_sensor_tags": {},
        "validation_results": {},
    }
    errors, validation_warnings = validate_graph(graph, matrices)
    graph["validation_results"] = {"errors": errors, "warnings": validation_warnings}
    _write_outputs(output, graph, matrices, materials, warnings)
    _atomic_write_text(
        output / "validation_report.txt",
        format_validation_report(graph, errors, validation_warnings),
    )
    if errors:
        raise SystemExit(f"Graph written with validation errors; see {output / 'validation_report.txt'}")
    print(f"Wrote octree graph with {len(graph_result.nodes)} nodes and {len(graph_result.edges)} edges to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gltf", required=True)
    parser.add_argument("--contact-report", default=None)
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
    parser.add_argument("--radiation-reference-temperature-K", type=float, default=293.15)
    return parser


def _write_outputs(
    output: Path,
    graph: dict,
    matrices: dict[str, np.ndarray],
    materials: dict,
    warnings: list[str],
) -> None:
    _atomic_write_json(output / "graph.json", graph, indent=2)
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
