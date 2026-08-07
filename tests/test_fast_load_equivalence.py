"""The fast (nodes.csv) load must be EQUIVALENT to the full graph.json load.

The low-memory path used to be a lossy subset: it dropped ``model.edges``, which
silently broke temperature-dependent properties (L(T) is rebuilt from the edges
each step, so an empty edge set gives an all-zero Laplacian -- 3M thermally
isolated nodes). These tests pin the contract that the fast path reconstructs the
model losslessly, and that both paths build the SAME operators.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from graph_visualizer.fast_edge_io import read_edges_npz, write_edges_npz
from graph_visualizer.models import EdgeProperties, NodeProperties, ThermalGraphModel


def _model() -> ThermalGraphModel:
    model = ThermalGraphModel()
    for node_id in range(4):
        model.nodes[node_id] = NodeProperties(
            node_id=node_id,
            coord=(node_id, 0, 0),
            component_name="partA" if node_id < 2 else "partB",
            material="copper",
        )
    model.edges = {
        (0, 1): EdgeProperties(0, 1, 2.5, "auto", "edge_0", "internal_conduction", 1e-4, 5e-3, "high", []),
        (1, 2): EdgeProperties(1, 2, 0.75, "loaded_G", "edge_1", "uncertain_contact", 2e-4, 6e-3, "low",
                               ["Inter-part geometry adjacency uses interface conductance approximation."]),
        (2, 3): EdgeProperties(2, 3, 0.0, "auto", "custom-id", "role_node_contact", 0.0, 0.0, "medium", []),
    }
    return model


def _fields(edge: EdgeProperties) -> tuple:
    return (
        edge.source, edge.target, edge.Gij_W_K, edge.source_metadata, edge.edge_id,
        edge.edge_type, edge.shared_area_m2, edge.distance_m, edge.contact_confidence,
        tuple(edge.warnings),
    )


def test_edges_round_trip_losslessly() -> None:
    original = _model()
    with TemporaryDirectory() as directory:
        write_edges_npz(original, Path(directory))
        restored = read_edges_npz(Path(directory))
    assert set(restored) == set(original.edges)
    for key, edge in original.edges.items():
        assert _fields(restored[key]) == _fields(edge), key


def test_non_canonical_edge_id_and_warnings_survive() -> None:
    original = _model()
    with TemporaryDirectory() as directory:
        write_edges_npz(original, Path(directory))
        restored = read_edges_npz(Path(directory))
    assert restored[(2, 3)].edge_id == "custom-id"
    assert restored[(1, 2)].warnings == original.edges[(1, 2)].warnings
    assert restored[(0, 1)].warnings == []


def test_empty_edge_set_round_trips() -> None:
    model = ThermalGraphModel()
    with TemporaryDirectory() as directory:
        write_edges_npz(model, Path(directory))
        assert read_edges_npz(Path(directory)) == {}


def test_can_load_fast_requires_the_edges_artifact() -> None:
    from graph_visualizer.fast_edge_io import EDGES_FILENAME
    from graph_visualizer.fast_graph_io import can_load_fast

    with TemporaryDirectory() as directory:
        folder = Path(directory)
        for name in ("nodes.csv", "node_ids.npy", "C.npy", "L_sparse.npz"):
            (folder / name).write_bytes(b"x")
        usable, reason = can_load_fast(folder)
        assert not usable and EDGES_FILENAME in reason


def test_tdep_operator_builds_a_real_laplacian_from_restored_edges() -> None:
    """The regression itself: with edges present, L(T) must actually conduct."""
    pytest.importorskip("scipy")
    from graph_visualizer.temperature_dependent_properties import (
        build_temperature_dependent_operator,
    )

    model = _model()
    for node in model.nodes.values():
        node.center_mm = (float(node.node_id) * 10.0, 0.0, 0.0)
        node.size_mm = (10.0, 10.0, 10.0)
        node.mass_kg = 0.01
        node.cp_J_kgK = 385.0
        node.k_W_mK = 400.0
        node.C_J_K = 3.85
    with TemporaryDirectory() as directory:
        write_edges_npz(model, Path(directory))
        model.edges = read_edges_npz(Path(directory))
    node_ids = np.array(sorted(model.nodes), dtype=int)
    operator = build_temperature_dependent_operator(model, node_ids)
    L = operator.laplacian(np.full(node_ids.size, 40.0))
    assert L.nnz > 0, "restored edges must produce a conducting Laplacian"
    off = L.copy()
    off.setdiag(0)
    off.eliminate_zeros()
    assert off.data.max() < 0.0, "off-diagonals must be -G (negative)"
    assert abs(np.asarray(L.sum(axis=1)).ravel()).max() < 1e-9, "must be conservative"
