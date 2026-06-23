"""Load and save sparse 3D thermal graph folders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import csv

import numpy as np

from .material_library import default_material_library, normalize_material_library
from .matrix_builder import apply_conductance_matrix, build_matrices, refresh_auto_edges
from .models import EdgeMode, GraphMetadata, ThermalGraphModel
from .validation import (
    raise_if_errors,
    validate_conductance_matrix,
    validate_matrices,
    validate_model,
)


GRAPH_FILE = "graph3d.json"
OCTREE_GRAPH_FILE = "graph.json"
MATRIX_FILE = "matrices.npz"
METADATA_FILE = "metadata.json"
MATERIAL_FILE = "material_library.json"


def save_graph_folder(model: ThermalGraphModel, folder_path: str | Path) -> dict[str, np.ndarray]:
    """Save graph JSON, matrices, metadata, and material library into a folder."""
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    model.metadata.edge_mode = EdgeMode.normalize(model.metadata.edge_mode)
    if model.octree_graph_data:
        return _save_octree_graph_folder_lightweight(model, folder)
    if model.metadata.edge_mode == EdgeMode.AUTO.value:
        refresh_auto_edges(model)
    errors = validate_model(model)
    raise_if_errors(errors, "Cannot save graph")
    matrices = build_matrices(model)
    matrix_errors = validate_matrices(matrices, model.ordered_node_ids())
    raise_if_errors(matrix_errors, "Cannot save graph matrices")

    model.metadata.graph_name = model.metadata.graph_name or folder.name
    with (folder / GRAPH_FILE).open("w", encoding="utf-8") as handle:
        json.dump(model.to_graph3d_dict(), handle, indent=2)
    with (folder / OCTREE_GRAPH_FILE).open("w", encoding="utf-8") as handle:
        json.dump(model.to_octree_graph_dict(), handle, indent=2)
    np.savez(folder / MATRIX_FILE, **matrices)
    _save_octree_outputs(model, matrices, folder)
    with (folder / METADATA_FILE).open("w", encoding="utf-8") as handle:
        json.dump(model.metadata.to_dict(), handle, indent=2)
    material_library = model.material_library or default_material_library()
    with (folder / MATERIAL_FILE).open("w", encoding="utf-8") as handle:
        json.dump(material_library, handle, indent=2)
    return matrices


def load_graph_folder(folder_path: str | Path) -> tuple[ThermalGraphModel, dict[str, np.ndarray]]:
    """Load and validate a legacy graph3d.json or octree graph.json folder."""
    folder = Path(folder_path)
    octree_path = folder / OCTREE_GRAPH_FILE
    graph_path = folder / GRAPH_FILE
    matrix_path = folder / MATRIX_FILE
    if octree_path.exists():
        with octree_path.open("r", encoding="utf-8") as handle:
            graph_data = json.load(handle)
        model = ThermalGraphModel.from_octree_graph_dict(graph_data)
        model.metadata.graph_name = model.metadata.graph_name or folder.name
        material_path = folder / "materials_used.json"
        if material_path.exists():
            with material_path.open("r", encoding="utf-8") as handle:
                loaded_materials = json.load(handle)
            if isinstance(loaded_materials, dict):
                model.material_library = normalize_material_library(loaded_materials)
    else:
        missing = [name for name, path in ((GRAPH_FILE, graph_path),) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Graph folder is incomplete/corrupted. Missing: " + ", ".join(missing)
            )

        with graph_path.open("r", encoding="utf-8") as handle:
            graph_data = json.load(handle)
        metadata = _load_metadata(folder / METADATA_FILE)
        material_library = _load_materials(folder / MATERIAL_FILE)
        model = ThermalGraphModel.from_graph3d_dict(
            graph_data, metadata=metadata, material_library=material_library
        )
        model.metadata.graph_name = model.metadata.graph_name or folder.name

    if not matrix_path.exists() and not (folder / "G.npy").exists():
        raise FileNotFoundError(
            "Graph folder is incomplete/corrupted. Missing matrices.npz or individual .npy matrices."
        )
    model_errors = validate_model(model)
    raise_if_errors(model_errors, "Loaded graph is invalid")

    if matrix_path.exists():
        with np.load(matrix_path, allow_pickle=False) as loaded:
            matrices = {key: loaded[key] for key in loaded.files}
    elif octree_path.exists():
        node_ids = np.array(model.ordered_node_ids(), dtype=int)
        matrices = {
            "node_ids": node_ids,
            "C": np.array([model.nodes[int(node_id)].C_J_K for node_id in node_ids], dtype=float),
        }
        return model, matrices
    else:
        matrices = build_matrices(model)
        for key in ("C", "G", "L", "A"):
            path = folder / f"{key}.npy"
            if path.exists():
                matrices[key] = np.load(path, allow_pickle=False)
    matrix_errors = validate_matrices(matrices, model.ordered_node_ids())
    raise_if_errors(matrix_errors, "Loaded matrices are invalid")
    if EdgeMode.normalize(model.metadata.edge_mode) == EdgeMode.LOADED_G.value:
        apply_conductance_matrix(model, matrices["node_ids"], matrices["G"])
    return model, matrices


def _save_octree_graph_folder_lightweight(
    model: ThermalGraphModel, folder: Path
) -> dict[str, np.ndarray]:
    """Persist octree metadata/tag edits without rebuilding large dense matrices."""
    errors = validate_model(model)
    raise_if_errors(errors, "Cannot save graph")
    model.metadata.graph_name = model.metadata.graph_name or folder.name
    with (folder / OCTREE_GRAPH_FILE).open("w", encoding="utf-8") as handle:
        json.dump(model.to_octree_graph_dict(), handle, indent=2)
    with (folder / METADATA_FILE).open("w", encoding="utf-8") as handle:
        json.dump(model.metadata.to_dict(), handle, indent=2)
    material_library = model.material_library or default_material_library()
    with (folder / MATERIAL_FILE).open("w", encoding="utf-8") as handle:
        json.dump(material_library, handle, indent=2)
    node_ids = np.array(model.ordered_node_ids(), dtype=int)
    matrices = {
        "node_ids": node_ids,
        "C": np.array([model.nodes[int(node_id)].C_J_K for node_id in node_ids], dtype=float),
    }
    _save_octree_outputs(model, matrices, folder)
    return matrices


def _save_octree_outputs(model: ThermalGraphModel, matrices: dict[str, np.ndarray], folder: Path) -> None:
    node_rows = [model.nodes[node_id].to_octree_node_dict() for node_id in model.ordered_node_ids()]
    with (folder / "nodes.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["node_id", "cell_id", "component_name", "material_name", "level", "volume_m3", "mass_kg", "C_J_K", "occupancy_fraction", "confidence"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(node_rows)
    with (folder / "edges.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["edge_id", "node_i", "node_j", "edge_type", "G_W_K", "shared_area_m2", "distance_m", "contact_confidence", "source"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(edge.to_octree_edge_dict() for edge in model.edges.values())
    with (folder / "params.json").open("w", encoding="utf-8") as handle:
        json.dump(model.octree_graph_data.get("parameters", {}), handle, indent=2)
    with (folder / "materials_used.json").open("w", encoding="utf-8") as handle:
        json.dump(model.material_library or default_material_library(), handle, indent=2)
    for key in ("C", "G", "L"):
        if key in matrices:
            np.save(folder / f"{key}.npy", matrices[key])
    if "C" in matrices and "L" in matrices and np.all(np.asarray(matrices["C"]) > 0.0):
        np.save(folder / "A.npy", -np.diag(1.0 / np.asarray(matrices["C"], dtype=float)) @ np.asarray(matrices["L"], dtype=float))
    ui_state = {
        "selected_node_id": None,
        "filters": {"materials": [], "components": [], "levels": []},
    }
    path = folder / "ui_state.json"
    if not path.exists():
        with path.open("w", encoding="utf-8") as handle:
            json.dump(ui_state, handle, indent=2)


def load_conductance_matrix_from_folder(
    model: ThermalGraphModel, folder_path: str | Path
) -> None:
    """Load only G and node_ids from a graph folder's matrices.npz into edges."""
    matrix_path = Path(folder_path) / MATRIX_FILE
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing {MATRIX_FILE}.")
    with np.load(matrix_path, allow_pickle=False) as loaded:
        matrices: dict[str, Any] = {key: loaded[key] for key in loaded.files}
    errors = validate_conductance_matrix(matrices, model.ordered_node_ids())
    raise_if_errors(errors, "Cannot load conductance matrix")
    apply_conductance_matrix(model, matrices["node_ids"], matrices["G"])


def _load_metadata(path: Path) -> GraphMetadata:
    if not path.exists():
        return GraphMetadata()
    with path.open("r", encoding="utf-8") as handle:
        return GraphMetadata.from_dict(json.load(handle))


def _load_materials(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return default_material_library()
    with path.open("r", encoding="utf-8") as handle:
        return normalize_material_library(json.load(handle))
