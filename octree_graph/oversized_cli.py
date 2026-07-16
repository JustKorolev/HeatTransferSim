"""Analyze oversized nodes in an existing octree graph folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from .cli import _atomic_write_json, _oversized_node_summary


DEFAULT_GRAPH_FOLDER = Path("graphs") / "HISPEC-CRYOSTAT-TEST-v4"


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    graph_folder = Path(args.graph_folder).resolve()
    graph_path = graph_folder / "graph.json"
    if not graph_path.is_file():
        raise SystemExit(f"graph.json not found: {graph_path}")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    max_cell_size_mm = _resolve_max_cell_size_mm(graph, args.max_cell_size_mm)
    summary = _oversized_node_summary(
        list(graph.get("graph_nodes", []) or []),
        SimpleNamespace(max_cell_size_mm=max_cell_size_mm),
    )
    if not args.no_write:
        _atomic_write_json(graph_folder / "oversized_node_summary.json", summary, indent=2)
    _print_summary(graph_folder, summary, verbose=bool(args.verbose))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "graph_folder",
        nargs="?",
        default=str(DEFAULT_GRAPH_FOLDER),
        help=f"Generated graph folder containing graph.json. Default: {DEFAULT_GRAPH_FOLDER}",
    )
    parser.add_argument(
        "--max-cell-size-mm",
        type=float,
        default=None,
        help="Override max_cell_size_mm instead of reading it from graph.json parameters.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Only print the summary; do not write oversized_node_summary.json.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the largest oversized nodes after the compact summary.",
    )
    return parser


def _resolve_max_cell_size_mm(graph: dict, override: float | None) -> float:
    if override is not None:
        return float(override)
    try:
        return float((graph.get("parameters") or {})["max_cell_size_mm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("max_cell_size_mm was not found in graph.json; pass --max-cell-size-mm.") from exc


def _print_summary(graph_folder: Path, summary: dict, *, verbose: bool) -> None:
    print(f"Graph folder: {graph_folder}")
    print(f"max_cell_size_mm: {summary.get('max_cell_size_mm')}")
    print(f"oversized nodes: {summary.get('total_oversized_nodes')}")
    print(f"  voxel:  {summary.get('voxel_oversized_nodes')}")
    print(f"  marker: {summary.get('marker_oversized_nodes')}")
    print(f"largest_size_mm: {summary.get('largest_size_mm')}")
    reason_counts = summary.get("reason_counts") or {}
    if reason_counts:
        print("reasons:")
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count}")
    if int(reason_counts.get("blocked_by_max_leaf_cells_old_code", 0) or 0) > 0:
        print(
            "WARNING: this graph contains old max_leaf_cells-blocked voxel warnings. "
            "It was not produced by the patched max-cell-size enforcement path."
        )
    if verbose:
        print("largest nodes:")
        for node in summary.get("largest_nodes", []) or []:
            print(
                "  "
                f"node_id={node.get('node_id')} "
                f"size={node.get('max_size_mm')} "
                f"level={node.get('level')} "
                f"reason={node.get('reason')} "
                f"component={node.get('component_name')}"
            )


if __name__ == "__main__":
    main()
