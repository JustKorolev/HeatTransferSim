"""Standalone connectivity analysis for generated octree graph folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .cli import (
    _annotate_graph_warning_tags,
    _atomic_write_json,
    _build_quality_report,
    _graph_connectivity_analysis,
)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    graph_folder = Path(args.graph_folder)
    graph_path = graph_folder / "graph.json"
    if not graph_path.is_file():
        raise SystemExit(f"Missing graph.json in {graph_folder}")
    with graph_path.open("r", encoding="utf-8") as handle:
        graph = json.load(handle)
    nodes = list(graph.get("graph_nodes", []) or [])
    edges = list(graph.get("graph_edges", []) or [])
    analysis = _graph_connectivity_analysis(nodes, edges)
    graph["connectivity_analysis"] = analysis
    if not bool(args.no_update_graph):
        _annotate_graph_warning_tags(nodes, edges, _args_for_quality(graph), analysis)
        graph["graph_nodes"] = nodes
        graph["build_quality"] = _build_quality_report(graph, _args_for_quality(graph))
        _atomic_write_json(graph_path, graph, indent=2)
        _atomic_write_json(graph_folder / "build_quality.json", graph["build_quality"], indent=2)
    _atomic_write_json(graph_folder / "connectivity_analysis.json", analysis, indent=2)
    print(
        "Connectivity analysis: "
        f"connected={analysis['connected']} "
        f"components={analysis['component_count']} "
        f"largest={analysis['largest_component_size']} "
        f"disconnected_nodes={len(analysis['disconnected_node_ids'])}"
    )
    print(f"Wrote {graph_folder / 'connectivity_analysis.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-folder",
        required=True,
        help="Existing generated graph folder containing graph.json.",
    )
    parser.add_argument(
        "--no-update-graph",
        action="store_true",
        help="Only write connectivity_analysis.json; do not update graph.json warning tags/build_quality.",
    )
    return parser


def _args_for_quality(graph: dict[str, Any]) -> SimpleNamespace:
    params = graph.get("parameters", {}) if isinstance(graph.get("parameters"), dict) else {}
    return SimpleNamespace(max_cell_size_mm=float(params.get("max_cell_size_mm", 1.0e99)))


if __name__ == "__main__":
    main()
