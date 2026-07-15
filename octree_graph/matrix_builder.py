"""Dense matrix construction for octree thermal graphs."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix

DENSE_MATRIX_NODE_LIMIT = 6000
DENSE_MATRIX_MAX_TOTAL_BYTES = 768 * 1024 * 1024


def build_matrices(
    nodes: list[dict],
    edges: list[dict],
    *,
    dense_node_limit: int = DENSE_MATRIX_NODE_LIMIT,
) -> dict[str, np.ndarray]:
    node_ids = np.array([int(node["node_id"]) for node in nodes], dtype=int)
    size = len(node_ids)
    index = {node_id: row for row, node_id in enumerate(node_ids)}
    C = np.array([float(node["C_J_K"]) for node in nodes], dtype=float)
    G_rad = np.array(
        [
            float((node.get("radiation") or {}).get("G_rad_W_K", node.get("Grad_W_K", 0.0)))
            for node in nodes
        ],
        dtype=float,
    )
    initial_temperature_K = np.array(
        [float(node.get("initial_temperature_K", 293.15)) for node in nodes], dtype=float
    )
    if not _should_build_dense_matrices(size, dense_node_limit):
        L = _build_sparse_laplacian(edges, index, size)
        return {
            "node_ids": node_ids,
            "C": C,
            "L": L,
            "G_rad": G_rad,
            "initial_temperature_K": initial_temperature_K,
        }
    G = np.zeros((size, size), dtype=float)
    for edge in edges:
        if _is_visual_role_contact_edge(edge):
            continue
        i = index[int(edge["node_i"])]
        j = index[int(edge["node_j"])]
        value = max(0.0, float(edge["G_W_K"]))
        G[i, j] = value
        G[j, i] = value
    L = np.diag(G.sum(axis=1)) - G
    return {
        "node_ids": node_ids,
        "C": C,
        "G": G,
        "L": L,
        "G_rad": G_rad,
        "initial_temperature_K": initial_temperature_K,
    }


def _should_build_dense_matrices(size: int, dense_node_limit: int) -> bool:
    if size > int(dense_node_limit):
        return False
    dense_pair_bytes = int(size) * int(size) * np.dtype(float).itemsize * 2
    return dense_pair_bytes <= DENSE_MATRIX_MAX_TOTAL_BYTES


def _build_sparse_laplacian(edges: list[dict], index: dict[int, int], size: int):
    diagonal = np.zeros(size, dtype=float)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for edge in edges:
        if _is_visual_role_contact_edge(edge):
            continue
        i = index[int(edge["node_i"])]
        j = index[int(edge["node_j"])]
        conductance = max(0.0, float(edge["G_W_K"]))
        if conductance <= 0.0:
            continue
        diagonal[i] += conductance
        diagonal[j] += conductance
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([-conductance, -conductance])
    nonzero = np.nonzero(diagonal > 0.0)[0]
    rows.extend(nonzero.astype(int).tolist())
    cols.extend(nonzero.astype(int).tolist())
    data.extend(diagonal[nonzero].astype(float).tolist())
    return coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()


def _is_visual_role_contact_edge(edge: dict) -> bool:
    return (
        str(edge.get("edge_type", "")) == "role_node_contact"
        or str(edge.get("source", edge.get("source_metadata", ""))) == "cad_role_node_contact"
    )
