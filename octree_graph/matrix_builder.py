"""Dense matrix construction for octree thermal graphs."""

from __future__ import annotations

import numpy as np


def build_matrices(nodes: list[dict], edges: list[dict]) -> dict[str, np.ndarray]:
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


def _is_visual_role_contact_edge(edge: dict) -> bool:
    return (
        str(edge.get("edge_type", "")) == "role_node_contact"
        or str(edge.get("source", edge.get("source_metadata", ""))) == "cad_role_node_contact"
    )
