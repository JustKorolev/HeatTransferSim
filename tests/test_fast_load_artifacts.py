"""Editing a large octree graph must keep the low-memory (fast) load valid.

The lightweight save path (used for large graphs) rewrites graph.json but, for a
graph above the dense-matrix threshold, used to skip nodes.csv. That left
nodes.csv older than graph.json, which disqualifies fast_graph_io's low-memory
loader (can_load_fast) and forces every later run through the multi-minute,
tens-of-GB graph.json parse. The fix rewrites nodes.csv in that branch.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from graph_visualizer import graph_io
from graph_visualizer.fast_graph_io import can_load_fast, launch_refresh_subprocess
from graph_visualizer.graph_io import save_graph_folder
from graph_visualizer.models import GraphMetadata, NodeProperties, ThermalGraphModel


def _two_node_octree_model(name: str, left_initial_K: float = 310.0) -> ThermalGraphModel:
    model = ThermalGraphModel(metadata=GraphMetadata(graph_name=name))
    left = NodeProperties.with_material(1, (0, 0, 0), material="copper")
    right = NodeProperties.with_material(2, (1, 0, 0), material="copper")
    left.material = right.material = "Copper"
    left.center_mm = (0.0, 0.0, 0.0)
    right.center_mm = (1.0, 0.0, 0.0)
    left.size_mm = right.size_mm = (1.0, 1.0, 1.0)
    left.C_J_K = right.C_J_K = 10.0
    left.initial_temperature_K = left_initial_K
    right.initial_temperature_K = 290.0
    model.add_node(left)
    model.add_node(right)
    # Non-empty octree data routes save through the lightweight path.
    model.octree_graph_data = {"graph_edges": []}
    return model


def _read_initial_temp(folder: Path, node_id: int) -> float:
    with (folder / "nodes.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["node_id"]) == node_id:
                return float(row["initial_temperature_K"])
    raise AssertionError(f"node {node_id} missing from nodes.csv")


def test_lightweight_save_of_large_graph_keeps_fast_load_valid(monkeypatch) -> None:
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        # First save builds the full artifact set (small-graph branch).
        save_graph_folder(_two_node_octree_model("fast_load"), folder)
        assert (folder / "nodes.csv").exists()
        # node_ids.npy is written by the octree BUILDER, not the save path; it is
        # present in a real graph folder, so create it to mirror production.
        np.save(folder / "node_ids.npy", np.array([1, 2], dtype=int))

        # Simulate an edit made well after the build: graph.json is about to be
        # rewritten, and nodes.csv currently predates it.
        old = 1_000_000.0
        os.utime(folder / "graph.json", (old, old))
        os.utime(folder / "nodes.csv", (old, old))

        # Force the >threshold branch that used to skip nodes.csv.
        monkeypatch.setattr(graph_io, "_DENSE_OCTREE_MATRIX_NODE_LIMIT", 1)
        edited = _two_node_octree_model("fast_load", left_initial_K=333.0)
        save_graph_folder(edited, folder)

        usable, reason = can_load_fast(folder)
        assert usable, reason
        # nodes.csv was regenerated from the edited model, not left stale.
        assert _read_initial_temp(folder, 1) == 333.0
        graph_mtime = (folder / "graph.json").stat().st_mtime
        nodes_mtime = (folder / "nodes.csv").stat().st_mtime
        assert nodes_mtime >= graph_mtime - 1.0


def test_refresh_subprocess_restores_a_stale_fast_load(tmp_path) -> None:
    """End-to-end: the 'Update graph' path (refresh_fast_load.py via
    launch_refresh_subprocess) regenerates nodes.csv from graph.json without the
    GUI, turning an unusable fast-load folder back into a usable one."""
    folder = tmp_path / "graph"
    folder.mkdir()
    save_graph_folder(_two_node_octree_model("refresh_subproc"), folder)
    np.save(folder / "node_ids.npy", np.array([1, 2], dtype=int))
    old = 1_000_000.0
    os.utime(folder / "nodes.csv", (old, old))
    assert can_load_fast(folder)[0] is False

    proc = launch_refresh_subprocess(folder)
    assert proc.wait(timeout=120) == 0, (folder / "refresh_fast_load.log").read_text(encoding="utf-8")

    usable, reason = can_load_fast(folder)
    assert usable, reason
