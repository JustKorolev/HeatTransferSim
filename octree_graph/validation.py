"""Validation report generation for octree graph folders."""

from __future__ import annotations

import numpy as np
from scipy.sparse import issparse

_DENSE_SYMMETRY_CHECK_MAX_BYTES = 512 * 1024 * 1024


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
    for key in ("C", "L"):
        if key not in matrices:
            errors.append(f"Missing matrix {key}.")
    if "G" not in matrices and not issparse(matrices.get("L")):
        errors.append("Missing matrix G.")
    if "G" in matrices and matrices["G"].shape != (size, size):
        errors.append(f"G shape {matrices['G'].shape} does not match node count {size}.")
    if "L" in matrices and matrices["L"].shape != (size, size):
        errors.append(f"L shape {matrices['L'].shape} does not match node count {size}.")
    if "G" in matrices:
        result = _symmetric_check_result(matrices["G"])
        if result is False:
            errors.append("G is not symmetric.")
        elif result is None:
            warnings.append(_skipped_dense_symmetry_warning("G", matrices["G"]))
    if "L" in matrices:
        result = _symmetric_check_result(matrices["L"])
        if result is False:
            errors.append("L is not symmetric.")
        elif result is None:
            warnings.append(_skipped_dense_symmetry_warning("L", matrices["L"]))
    return errors, warnings


def _symmetric_check_result(matrix: np.ndarray) -> bool | None:
    if issparse(matrix):
        difference = matrix - matrix.T
        return not (difference.nnz and np.max(np.abs(difference.data)) > 1.0e-12)
    array = np.asarray(matrix)
    if array.nbytes > _DENSE_SYMMETRY_CHECK_MAX_BYTES:
        return None
    return bool(np.allclose(array, array.T, rtol=1.0e-7, atol=1.0e-12))


def _skipped_dense_symmetry_warning(name: str, matrix: np.ndarray) -> str:
    array = np.asarray(matrix)
    return (
        f"Skipped dense {name} symmetry validation because matrix shape {array.shape} "
        f"uses {_format_bytes(array.nbytes)}; use sparse L output for large graphs."
    )


def _format_bytes(value: int) -> str:
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024.0 or unit == "TiB":
            return f"{number:.1f} {unit}"
        number /= 1024.0


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
