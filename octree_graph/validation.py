"""Validation report generation for octree graph folders."""

from __future__ import annotations

import numpy as np


def validate_graph(graph: dict, matrices: dict[str, np.ndarray]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = list(graph.get("warnings", []))
    nodes = graph.get("graph_nodes", [])
    edges = graph.get("graph_edges", [])
    node_ids = [int(node["node_id"]) for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        errors.append("Duplicate node IDs.")
    edge_ids = [edge.get("edge_id") for edge in edges]
    if len(edge_ids) != len(set(edge_ids)):
        errors.append("Duplicate edge IDs.")
    for node in nodes:
        if not node.get("component_name") or not node.get("material_name"):
            errors.append(f"Node {node.get('node_id')} missing component/material assignment.")
        if float(node.get("C_J_K", 0.0)) <= 0.0:
            errors.append(f"Node {node.get('node_id')} has nonpositive thermal capacitance.")
    node_set = set(node_ids)
    for edge in edges:
        if int(edge["node_i"]) not in node_set or int(edge["node_j"]) not in node_set:
            errors.append(f"Edge {edge.get('edge_id')} references a missing node.")
        if float(edge.get("G_W_K", 0.0)) < 0.0:
            errors.append(f"Edge {edge.get('edge_id')} has negative conductance.")
    size = len(nodes)
    for key in ("C", "G", "L"):
        if key not in matrices:
            errors.append(f"Missing matrix {key}.")
    if "G" in matrices and matrices["G"].shape != (size, size):
        errors.append(f"G shape {matrices['G'].shape} does not match node count {size}.")
    if "L" in matrices and matrices["L"].shape != (size, size):
        errors.append(f"L shape {matrices['L'].shape} does not match node count {size}.")
    if "G" in matrices and not np.allclose(matrices["G"], matrices["G"].T):
        errors.append("G is not symmetric.")
    if "L" in matrices and not np.allclose(matrices["L"], matrices["L"].T):
        errors.append("L is not symmetric.")
    return errors, warnings


def format_validation_report(graph: dict, errors: list[str], warnings: list[str]) -> str:
    return "\n".join(
        [
            "Octree Thermal Graph Validation",
            "================================",
            f"leaf cells: {len(graph.get('octree_cells', []))}",
            f"graph nodes: {len(graph.get('graph_nodes', []))}",
            f"graph edges: {len(graph.get('graph_edges', []))}",
            "",
            "Errors:",
            *(f"- {error}" for error in errors),
            *(["- none"] if not errors else []),
            "",
            "Warnings:",
            *(f"- {warning}" for warning in warnings),
            *(["- none"] if not warnings else []),
        ]
    )
