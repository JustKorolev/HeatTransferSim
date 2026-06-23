"""Dense matrix construction for octree thermal graphs."""

from __future__ import annotations

import numpy as np


def build_matrices(nodes: list[dict], edges: list[dict]) -> dict[str, np.ndarray]:
    node_ids = np.array([int(node["node_id"]) for node in nodes], dtype=int)
    size = len(node_ids)
    index = {node_id: row for row, node_id in enumerate(node_ids)}
    C = np.array([float(node["C_J_K"]) for node in nodes], dtype=float)
    G = np.zeros((size, size), dtype=float)
    for edge in edges:
        i = index[int(edge["node_i"])]
        j = index[int(edge["node_j"])]
        value = max(0.0, float(edge["G_W_K"]))
        G[i, j] = value
        G[j, i] = value
    L = np.diag(G.sum(axis=1)) - G
    A = -np.diag(1.0 / C) @ L if size and np.all(C > 0.0) else np.zeros_like(L)
    return {"node_ids": node_ids, "C": C, "G": G, "L": L, "A": A}
