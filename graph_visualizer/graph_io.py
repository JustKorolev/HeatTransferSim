"""Load and save sparse 3D thermal graph folders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
MATRIX_FILE = "matrices.npz"
METADATA_FILE = "metadata.json"
MATERIAL_FILE = "material_library.json"


def save_graph_folder(model: ThermalGraphModel, folder_path: str | Path) -> dict[str, np.ndarray]:
    """Save graph JSON, matrices, metadata, and material library into a folder."""
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    model.metadata.edge_mode = EdgeMode.normalize(model.metadata.edge_mode)
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
    np.savez(folder / MATRIX_FILE, **matrices)
    with (folder / METADATA_FILE).open("w", encoding="utf-8") as handle:
        json.dump(model.metadata.to_dict(), handle, indent=2)
    material_library = model.material_library or default_material_library()
    with (folder / MATERIAL_FILE).open("w", encoding="utf-8") as handle:
        json.dump(material_library, handle, indent=2)
    return matrices


def load_graph_folder(folder_path: str | Path) -> tuple[ThermalGraphModel, dict[str, np.ndarray]]:
    """Load and validate a graph folder containing graph3d.json and matrices.npz."""
    folder = Path(folder_path)
    graph_path = folder / GRAPH_FILE
    matrix_path = folder / MATRIX_FILE
    missing = [name for name, path in ((GRAPH_FILE, graph_path), (MATRIX_FILE, matrix_path)) if not path.exists()]
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
    model_errors = validate_model(model)
    raise_if_errors(model_errors, "Loaded graph is invalid")

    with np.load(matrix_path, allow_pickle=False) as loaded:
        matrices = {key: loaded[key] for key in loaded.files}
    matrix_errors = validate_matrices(matrices, model.ordered_node_ids())
    raise_if_errors(matrix_errors, "Loaded matrices are invalid")
    if EdgeMode.normalize(model.metadata.edge_mode) == EdgeMode.LOADED_G.value:
        apply_conductance_matrix(model, matrices["node_ids"], matrices["G"])
    return model, matrices


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
