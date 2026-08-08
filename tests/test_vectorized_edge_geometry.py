"""The vectorized edge-geometry build must equal the per-edge scalar helpers.

``build_temperature_dependent_operator`` used to call ``_edge_geometry`` once per
edge, allocating small numpy arrays each time -- minutes of startup on a
multi-million-edge graph, paid on every run. The vectorized path replaces it, so
it has to reproduce the scalar result exactly, INCLUDING the fallbacks for nodes
without geometry, zero area and zero conduction length.
"""

from __future__ import annotations

import numpy as np
import pytest

from graph_visualizer.models import EdgeProperties, NodeProperties, ThermalGraphModel
from graph_visualizer.temperature_dependent_properties import (
    DEFAULT_BOLTED_CONTACT_CONDUCTANCE_W_M2K,
    _edge_geometry,
    _edge_interface_conductance,
    _vectorized_edge_geometry,
    build_temperature_dependent_operator,
)


def _node(node_id: int, *, component: str, center, size, side: float = 0.0) -> NodeProperties:
    node = NodeProperties(node_id=node_id, coord=(node_id, 0, 0), component_name=component)
    node.center_mm = center
    node.size_mm = size
    node.side_length_m = side
    node.material = "copper"
    node.mass_kg = 0.01
    node.cp_J_kgK = 385.0
    node.k_W_mK = 400.0
    node.C_J_K = 3.85
    return node


def _cases() -> tuple[list, list[tuple[int, int, EdgeProperties]]]:
    nodes = [
        _node(0, component="A", center=(0.0, 0.0, 0.0), size=(10.0, 20.0, 30.0), side=0.01),
        _node(1, component="A", center=(10.0, 0.0, 0.0), size=(10.0, 20.0, 30.0), side=0.01),
        _node(2, component="B", center=(10.0, 20.0, 0.0), size=(4.0, 6.0, 8.0), side=0.02),
        # No geometry at all -> legacy distance/2 fallback.
        NodeProperties(node_id=3, coord=(3, 0, 0), component_name="C"),
    ]
    nodes[3].side_length_m = 0.05
    edges = [
        # same component (bonded), saved area present
        (0, 1, EdgeProperties(0, 1, 5.0, "auto", "edge_0", "internal_conduction", 3.0e-4, 1.0e-2, "high", [])),
        # different component (bolted), NO saved area -> derived face area
        (1, 2, EdgeProperties(1, 2, 2.0, "auto", "edge_1", "uncertain_contact", 0.0, 2.0e-2, "low", [])),
        # endpoint without geometry -> distance/2 path
        (2, 3, EdgeProperties(2, 3, 1.0, "auto", "edge_2", "same_material_spatial", 0.0, 4.0e-2, "medium", [])),
    ]
    return nodes, edges


def test_vectorized_matches_scalar_for_every_case() -> None:
    nodes, edges = _cases()
    edge_i = np.array([e[0] for e in edges])
    edge_j = np.array([e[1] for e in edges])
    saved = np.array([float(e[2].shared_area_m2) for e in edges])
    dist = np.array([float(e[2].distance_m) for e in edges])
    area, len_i, len_j, h_ref = _vectorized_edge_geometry(
        nodes, edge_i, edge_j, saved, dist, DEFAULT_BOLTED_CONTACT_CONDUCTANCE_W_M2K
    )
    for row, (i, j, edge) in enumerate(edges):
        want_area, want_i, want_j = _edge_geometry(nodes[i], nodes[j], edge)
        want_h = _edge_interface_conductance(
            nodes[i], nodes[j], edge, DEFAULT_BOLTED_CONTACT_CONDUCTANCE_W_M2K
        )
        assert area[row] == pytest.approx(want_area, rel=1e-12), f"area row {row}"
        assert len_i[row] == pytest.approx(want_i, rel=1e-12), f"len_i row {row}"
        assert len_j[row] == pytest.approx(want_j, rel=1e-12), f"len_j row {row}"
        assert h_ref[row] == pytest.approx(want_h, rel=1e-12), f"h_ref row {row}"


def test_bonded_and_bolted_are_distinguished() -> None:
    nodes, edges = _cases()
    edge_i = np.array([e[0] for e in edges])
    edge_j = np.array([e[1] for e in edges])
    saved = np.array([float(e[2].shared_area_m2) for e in edges])
    dist = np.array([float(e[2].distance_m) for e in edges])
    _a, _i, _j, h_ref = _vectorized_edge_geometry(
        nodes, edge_i, edge_j, saved, dist, DEFAULT_BOLTED_CONTACT_CONDUCTANCE_W_M2K
    )
    assert h_ref[0] == 0.0  # same component -> bonded
    assert h_ref[1] == DEFAULT_BOLTED_CONTACT_CONDUCTANCE_W_M2K  # cross component


def test_operator_still_conducts_and_conserves() -> None:
    nodes, edges = _cases()
    model = ThermalGraphModel()
    for node in nodes:
        model.nodes[node.node_id] = node
    model.edges = {(i, j): edge for i, j, edge in edges}
    node_ids = np.array(sorted(model.nodes), dtype=int)
    operator = build_temperature_dependent_operator(model, node_ids)
    L = operator.laplacian(np.full(node_ids.size, 40.0))
    assert L.nnz > 0
    assert abs(np.asarray(L.sum(axis=1)).ravel()).max() < 1e-9
