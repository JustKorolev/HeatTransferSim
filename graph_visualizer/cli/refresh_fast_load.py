"""Refresh a graph folder's low-memory (fast) load artifacts after editing.

The octree builder writes ``nodes.csv`` + the binary matrices; the GUI's
lightweight save (used for large graphs) rewrites ``graph.json`` but historically
left ``nodes.csv`` stale, which disqualifies ``fast_graph_io``'s low-memory loader
and forces every run through the multi-minute, tens-of-GB ``graph.json`` parse.

This regenerates ``nodes.csv``, ``node_ids.npy``, ``C.npy`` and ``L_sparse.npz``
from the graph so later runs load fast and lean -- WITHOUT opening the graph in
the GUI. It loads ``graph.json`` once here (transient RAM in a throwaway process)
so you pay that parse a single time instead of on every run.

Example:
    python refresh_fast_load.py graphs/no_mli_high_res
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from graph_visualizer.fast_edge_io import write_edges_npz_from_graph_json
from graph_visualizer.fast_graph_io import can_load_fast
from graph_visualizer.graph_io import load_graph_folder, write_fast_load_artifacts


def refresh_edges_only(folder: Path) -> None:
    """Write just ``edges.npz``, streaming graph.json instead of parsing it whole.

    A graph folder built before edges.npz existed has every other artifact already;
    only the edges are missing. The full loader would need ~45 GB on a 3M-cell
    graph, so stream the edges out instead -- bounded memory, no re-build.
    """
    folder = Path(folder)
    graph_json = folder / "graph.json"
    if not graph_json.is_file():
        raise SystemExit(f"No graph.json in {folder}")
    size_gib = graph_json.stat().st_size / 2**30
    print(f"streaming edges from {graph_json.name} ({size_gib:.1f} GiB) ...")
    t0 = time.perf_counter()
    count = write_edges_npz_from_graph_json(graph_json, folder)
    print(f"wrote edges.npz with {count:,} edges in {time.perf_counter() - t0:.1f}s")
    usable, reason = can_load_fast(folder)
    print(f"fast load: {'usable' if usable else 'unavailable -- ' + reason}")


def refresh_fast_load_artifacts(folder: Path) -> None:
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")

    before, reason = can_load_fast(folder)
    print(f"fast load before: {'usable' if before else 'unavailable -- ' + reason}")

    t0 = time.perf_counter()
    print(f"loading {folder} (parsing graph.json once) ...")
    model, matrices = load_graph_folder(str(folder))
    print(f"loaded {len(model.nodes):,} nodes in {time.perf_counter() - t0:.1f}s")

    write_fast_load_artifacts(model, matrices, folder)

    after, reason = can_load_fast(folder)
    if after:
        print("fast load after: usable -- future runs will load fast and lean.")
    else:
        # Should not happen after a clean refresh; report why if it does.
        print(f"fast load after: STILL unavailable -- {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", help="path to graphs/<name> folder")
    parser.add_argument(
        "--edges-only",
        action="store_true",
        help=(
            "only (re)build edges.npz, streaming graph.json instead of loading it. "
            "Use for a graph built before edges.npz existed, when a full load would "
            "not fit in RAM."
        ),
    )
    args = parser.parse_args()
    if args.edges_only:
        refresh_edges_only(Path(args.graph))
    else:
        refresh_fast_load_artifacts(Path(args.graph))


if __name__ == "__main__":
    main()
