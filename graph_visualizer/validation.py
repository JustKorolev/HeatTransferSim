"""Validation helpers for graph folders and in-memory graph matrices."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.sparse import issparse

from .models import ThermalGraphModel


class ValidationError(ValueError):
    """Raised when graph folder validation fails."""


def validate_model(model: ThermalGraphModel) -> list[str]:
    errors: list[str] = []
    node_ids = list(model.nodes)
    if len(node_ids) != len(set(node_ids)):
        errors.append("Node IDs must be unique.")
    coords = [node.coord for node in model.nodes.values()]
    if len(coords) != len(set(coords)):
        errors.append("Coordinates must be unique.")
    required_fields = (
        "coord",
        "material",
        "rho_kg_m3",
        "cp_J_kgK",
        "k_W_mK",
        "emissivity",
        "mass_kg",
        "C_J_K",
        "Grad_W_K",
        "initial_temperature_K",
        "side_length_m",
    )
    for node_id, node in model.nodes.items():
        for field_name in required_fields:
            if not hasattr(node, field_name):
                errors.append(f"Node {node_id} is missing required field {field_name}.")
        if node.C_J_K < 0.0:
            errors.append(f"Node {node_id} has negative thermal capacitance.")
        if node.Grad_W_K < 0.0:
            errors.append(f"Node {node_id} has negative Grad_W_K.")
        if not np.isfinite(float(node.initial_temperature_K)):
            errors.append(f"Node {node_id} has non-finite initial_temperature_K.")
        if node.side_length_m <= 0.0:
            errors.append(f"Node {node_id} must have positive side_length_m.")
    for edge in model.edges.values():
        if edge.source not in model.nodes or edge.target not in model.nodes:
            errors.append(f"Edge {edge.source}-{edge.target} references a missing node.")
        if edge.Gij_W_K < 0.0:
            errors.append(f"Edge {edge.source}-{edge.target} has negative conductance.")
    return errors


def validate_matrices(
    matrices: Mapping[str, np.ndarray], expected_node_ids: list[int] | np.ndarray
) -> list[str]:
    errors: list[str] = []
    required = ("node_ids", "coords", "C", "Grad")
    for key in required:
        if key not in matrices:
            errors.append(f"matrices.npz is missing required array {key}.")
    if "G" not in matrices and "L" not in matrices:
        errors.append("matrices.npz is missing required array G or sparse/dense L.")
    if errors:
        return errors

    node_ids = np.asarray(matrices["node_ids"], dtype=int)
    expected = np.asarray(expected_node_ids, dtype=int)
    size = len(expected)
    if set(node_ids.tolist()) != set(expected.tolist()):
        errors.append("node_ids in matrices.npz do not match graph3d.json node IDs.")
    if len(node_ids) != size:
        errors.append("node_ids length does not match number of graph nodes.")

    coords = np.asarray(matrices["coords"])
    C = np.asarray(matrices["C"])
    Grad = np.asarray(matrices["Grad"])
    G = None if "G" not in matrices else np.asarray(matrices["G"])
    L = matrices.get("L")
    if coords.shape != (size, 3):
        errors.append(f"coords must have shape ({size}, 3), got {coords.shape}.")
    if C.shape != (size,):
        errors.append(f"C must have shape ({size},), got {C.shape}.")
    if Grad.shape != (size,):
        errors.append(f"Grad must have shape ({size},), got {Grad.shape}.")
    if G is not None:
        if G.shape != (size, size):
            errors.append(f"G must have shape ({size}, {size}), got {G.shape}.")
        if G.ndim == 2 and not np.allclose(G, G.T, rtol=1.0e-7, atol=1.0e-12):
            errors.append("G must be symmetric within tolerance.")
        if np.any(G < -1.0e-12):
            errors.append("G contains negative conductances.")
    if L is not None:
        if issparse(L):
            if L.shape != (size, size):
                errors.append(f"L must have shape ({size}, {size}), got {L.shape}.")
            if L.nnz and not np.all(np.isfinite(L.data)):
                errors.append("L contains non-finite values.")
        else:
            dense_l = np.asarray(L)
            if dense_l.shape != (size, size):
                errors.append(f"L must have shape ({size}, {size}), got {dense_l.shape}.")
            if np.any(~np.isfinite(dense_l)):
                errors.append("L contains non-finite values.")
    if np.any(C < -1.0e-12):
        errors.append("C contains negative thermal capacitances.")
    if np.any(Grad < -1.0e-12):
        errors.append("Grad contains negative radiative conductances.")
    if "G_rad" in matrices:
        G_rad = np.asarray(matrices["G_rad"])
        if G_rad.shape not in {(size,), (size, size)}:
            errors.append(f"G_rad must have shape ({size},) or ({size}, {size}), got {G_rad.shape}.")
        if np.any(G_rad < -1.0e-12):
            errors.append("G_rad contains negative radiative conductances.")
    return errors


def validate_conductance_matrix(
    matrices: Mapping[str, np.ndarray], expected_node_ids: list[int] | np.ndarray
) -> list[str]:
    """Validate the loaded-G mode contract: node_ids plus symmetric nonnegative G."""
    errors: list[str] = []
    for key in ("node_ids", "G"):
        if key not in matrices:
            errors.append(f"matrices.npz is missing required array {key}.")
    if errors:
        return errors

    node_ids = np.asarray(matrices["node_ids"], dtype=int)
    expected = np.asarray(expected_node_ids, dtype=int)
    G = np.asarray(matrices["G"])
    size = len(node_ids)
    if G.shape != (size, size):
        errors.append(f"G must have shape ({size}, {size}), got {G.shape}.")
    if len(expected) != size:
        errors.append("node_ids length does not match number of graph nodes.")
    if set(node_ids.tolist()) != set(expected.tolist()):
        errors.append("node_ids in matrices.npz do not match graph node IDs.")
    if G.ndim == 2 and not np.allclose(G, G.T, rtol=1.0e-7, atol=1.0e-12):
        errors.append("G must be symmetric within tolerance.")
    if np.any(G < -1.0e-12):
        errors.append("G contains negative conductances.")
    return errors


def raise_if_errors(errors: list[str], heading: str = "Validation failed") -> None:
    if errors:
        raise ValidationError(heading + ":\n" + "\n".join(f"- {error}" for error in errors))
