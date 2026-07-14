"""Load and save sparse 3D thermal graph folders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import csv
import os
import tempfile

import numpy as np

from .material_library import default_material_library, material_defaults, normalize_material_library
from .matrix_builder import (
    apply_conductance_matrix,
    build_matrices,
    refresh_auto_edges,
    refresh_geometry_edges,
    refresh_radiation_from_exposed_faces,
)
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
    _atomic_write_json(folder / GRAPH_FILE, model.to_graph3d_dict(), indent=2)
    _atomic_write_json(folder / OCTREE_GRAPH_FILE, model.to_octree_graph_dict(), indent=2)
    np.savez(folder / MATRIX_FILE, **matrices)
    _save_octree_outputs(model, matrices, folder)
    _atomic_write_json(folder / METADATA_FILE, model.metadata.to_dict(), indent=2)
    material_library = model.material_library or default_material_library()
    _atomic_write_json(folder / MATERIAL_FILE, material_library, indent=2)
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
        if isinstance(graph_data.get("materials_used"), (dict, list)):
            model.material_library = normalize_material_library(graph_data["materials_used"])
        material_path = folder / "materials_used.json"
        if material_path.exists():
            with material_path.open("r", encoding="utf-8") as handle:
                loaded_materials = json.load(handle)
            if isinstance(loaded_materials, dict):
                model.material_library = normalize_material_library(loaded_materials)
        _apply_loaded_material_properties(model)
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
        matrices = _normalize_loaded_matrices(model, matrices)
        matrices = _repair_empty_auto_conduction(model, matrices)
    elif octree_path.exists():
        node_ids = np.array(model.ordered_node_ids(), dtype=int)
        matrices = {
            "node_ids": node_ids,
            "coords": np.array([model.nodes[int(node_id)].coord for node_id in node_ids], dtype=int),
            "C": np.array([model.nodes[int(node_id)].C_J_K for node_id in node_ids], dtype=float),
            "Grad": np.array([model.nodes[int(node_id)].Grad_W_K for node_id in node_ids], dtype=float),
        }
        for key in ("C", "G", "L", "G_rad", "initial_temperature_K"):
            path = folder / f"{key}.npy"
            if path.exists():
                matrices[key] = np.load(path, allow_pickle=False)
        if "G" not in matrices:
            size = len(node_ids)
            matrices["G"] = np.zeros((size, size), dtype=float)
            index = {int(node_id): row for row, node_id in enumerate(node_ids)}
            for edge in model.edges.values():
                if edge.source in index and edge.target in index:
                    i = index[edge.source]
                    j = index[edge.target]
                    matrices["G"][i, j] = matrices["G"][j, i] = max(0.0, float(edge.Gij_W_K))
        if "L" not in matrices:
            matrices["L"] = np.diag(matrices["G"].sum(axis=1)) - matrices["G"]
        if "G_rad" not in matrices:
            matrices["G_rad"] = np.array(
                [
                    model.nodes[int(node_id)].G_rad_W_K
                    if model.nodes[int(node_id)].G_rad_W_K > 0.0
                    else model.nodes[int(node_id)].Grad_W_K
                    for node_id in node_ids
                ],
                dtype=float,
            )
        if "initial_temperature_K" not in matrices:
            matrices["initial_temperature_K"] = np.array(
                [model.nodes[int(node_id)].initial_temperature_K for node_id in node_ids],
                dtype=float,
            )
        matrices = _repair_empty_auto_conduction(model, matrices)
    else:
        matrices = {
            "node_ids": np.array(model.ordered_node_ids(), dtype=int),
        }
        for key in ("C", "G", "L", "A"):
            path = folder / f"{key}.npy"
            if path.exists():
                matrices[key] = np.load(path, allow_pickle=False)
        matrices = _normalize_loaded_matrices(model, matrices)
        matrices = _repair_empty_auto_conduction(model, matrices)
    if octree_path.exists():
        matrices = _refresh_octree_auto_geometry(model, matrices)
        _refresh_octree_radiation(model)
        matrices = _sync_radiation_matrix_from_model(model, matrices)
    matrix_errors = validate_matrices(matrices, model.ordered_node_ids())
    raise_if_errors(matrix_errors, "Loaded matrices are invalid")
    if EdgeMode.normalize(model.metadata.edge_mode) == EdgeMode.LOADED_G.value:
        apply_conductance_matrix(model, matrices["node_ids"], matrices["G"])
    return model, matrices


def _repair_empty_auto_conduction(
    model: ThermalGraphModel,
    matrices: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if EdgeMode.normalize(model.metadata.edge_mode) != EdgeMode.AUTO.value:
        return matrices
    if len(model.edges) > 0 or _matrix_has_conduction(matrices):
        return matrices
    refresh_auto_edges(model)
    if len(model.edges) == 0:
        refresh_geometry_edges(model)
    if len(model.edges) == 0:
        return matrices
    model.octree_graph_data.setdefault("warnings", [])
    model.octree_graph_data["warnings"].append(
        "Loaded graph had no conductive edges/matrix; regenerated geometry-based auto edges."
    )
    return build_matrices(model)


def _apply_loaded_material_properties(model: ThermalGraphModel) -> None:
    library = model.material_library or default_material_library()
    for node in model.nodes.values():
        defaults = material_defaults(node.material, library)
        node.rho_kg_m3 = float(defaults["rho_kg_m3"])
        node.cp_J_kgK = float(defaults["cp_J_kgK"])
        node.k_W_mK = float(defaults["k_W_mK"])
        node.emissivity = float(defaults["emissivity"])


def _refresh_octree_auto_geometry(
    model: ThermalGraphModel,
    matrices: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if EdgeMode.normalize(model.metadata.edge_mode) != EdgeMode.AUTO.value:
        return matrices
    if has_generated_role_contact_edges(model):
        model.octree_graph_data.setdefault("warnings", [])
        model.octree_graph_data["warnings"].append(
            "Preserved loaded heater/sensor role contact visual edges and rebuilt marker-only matrices."
        )
        return build_matrices(model)
    if not model.nodes:
        return matrices
    if any(node.center_mm is None or node.size_mm is None for node in model.nodes.values()):
        return matrices
    previous_edges = len(model.edges)
    refresh_geometry_edges(model)
    if len(model.edges) == 0:
        return matrices
    model.octree_graph_data.setdefault("warnings", [])
    if previous_edges:
        model.octree_graph_data["warnings"].append(
            "Regenerated auto geometry conductances from loaded material properties."
        )
    return build_matrices(model)


def has_generated_role_contact_edges(model: ThermalGraphModel) -> bool:
    return any(
        str(edge.edge_type) in {"consolidated_role_contact", "role_node_contact"}
        or str(edge.source_metadata) in {"voxel_role_consolidation", "cad_role_node_contact"}
        for edge in model.edges.values()
    )


def has_consolidated_role_edges(model: ThermalGraphModel) -> bool:
    return has_generated_role_contact_edges(model)


def _matrix_has_conduction(matrices: dict[str, np.ndarray]) -> bool:
    for key in ("G", "L"):
        if key not in matrices:
            continue
        values = np.asarray(matrices[key], dtype=float)
        if values.size and np.any(np.abs(values) > 1.0e-15):
            return True
    return False


def _refresh_octree_radiation(model: ThermalGraphModel) -> None:
    params = model.octree_graph_data.get("parameters", {}) if model.octree_graph_data else {}
    reference_temperature = params.get(
        "radiation_reference_temperature_K",
        model.metadata.T_sur_K,
    )
    try:
        updated = refresh_radiation_from_exposed_faces(model, float(reference_temperature))
    except (TypeError, ValueError):
        updated = refresh_radiation_from_exposed_faces(model)
    if updated:
        model.octree_graph_data.setdefault("warnings", [])
        model.octree_graph_data["warnings"].append(
            f"Refreshed radiative exposed-face areas for {updated} cells."
        )


def _sync_radiation_matrix_from_model(
    model: ThermalGraphModel,
    matrices: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    normalized = dict(matrices)
    node_ids = np.asarray(normalized.get("node_ids", model.ordered_node_ids()), dtype=int)
    normalized["G_rad"] = np.array(
        [model.nodes[int(node_id)].G_rad_W_K for node_id in node_ids],
        dtype=float,
    )
    normalized["Grad"] = np.array(
        [model.nodes[int(node_id)].Grad_W_K for node_id in node_ids],
        dtype=float,
    )
    return normalized


def _normalize_loaded_matrices(
    model: ThermalGraphModel, matrices: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    normalized = dict(matrices)
    node_ids = np.asarray(normalized.get("node_ids", model.ordered_node_ids()), dtype=int)
    if "coords" not in normalized:
        normalized["coords"] = np.array([model.nodes[int(node_id)].coord for node_id in node_ids], dtype=int)
    if "Grad" not in normalized:
        normalized["Grad"] = np.array([model.nodes[int(node_id)].Grad_W_K for node_id in node_ids], dtype=float)
    if "G" in normalized and "L" not in normalized:
        G = np.asarray(normalized["G"], dtype=float)
        normalized["L"] = np.diag(G.sum(axis=1)) - G
    if "G_rad" not in normalized:
        normalized["G_rad"] = np.array(
            [
                model.nodes[int(node_id)].G_rad_W_K
                if model.nodes[int(node_id)].G_rad_W_K > 0.0
                else model.nodes[int(node_id)].Grad_W_K
                for node_id in node_ids
            ],
            dtype=float,
        )
    if "initial_temperature_K" not in normalized:
        normalized["initial_temperature_K"] = np.array(
            [model.nodes[int(node_id)].initial_temperature_K for node_id in node_ids],
            dtype=float,
        )
    return normalized


def _save_octree_graph_folder_lightweight(
    model: ThermalGraphModel, folder: Path
) -> dict[str, np.ndarray]:
    """Persist octree metadata/tag edits without rebuilding large dense matrices."""
    errors = validate_model(model)
    raise_if_errors(errors, "Cannot save graph")
    model.metadata.graph_name = model.metadata.graph_name or folder.name
    _atomic_write_json(folder / OCTREE_GRAPH_FILE, model.to_octree_graph_dict(), indent=2)
    _atomic_write_json(folder / METADATA_FILE, model.metadata.to_dict(), indent=2)
    material_library = model.material_library or default_material_library()
    _atomic_write_json(folder / MATERIAL_FILE, material_library, indent=2)
    matrices = build_matrices(model)
    _save_octree_outputs(model, matrices, folder)
    return matrices


def _save_octree_outputs(model: ThermalGraphModel, matrices: dict[str, np.ndarray], folder: Path) -> None:
    node_rows = [
        _flatten_node_row(model.nodes[node_id].to_octree_node_dict())
        for node_id in model.ordered_node_ids()
    ]
    with (folder / "nodes.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "node_id",
            "cell_id",
            "component_name",
            "material_name",
            "level",
            "volume_m3",
            "mass_kg",
            "C_J_K",
            "initial_temperature_K",
            "radiation_is_exposed",
            "radiation_radiating_area_m2",
            "radiation_emissivity",
            "radiation_G_rad_W_K",
            "radiation_R_rad_K_W",
            "occupancy_fraction",
            "confidence",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(node_rows)
    with (folder / "edges.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["edge_id", "node_i", "node_j", "edge_type", "G_W_K", "shared_area_m2", "distance_m", "contact_confidence", "source"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(edge.to_octree_edge_dict() for edge in model.edges.values())
    _atomic_write_json(folder / "params.json", model.octree_graph_data.get("parameters", {}), indent=2)
    _atomic_write_json(folder / "materials_used.json", model.material_library or default_material_library(), indent=2)
    for key in ("C", "G", "L"):
        if key in matrices:
            np.save(folder / f"{key}.npy", matrices[key])
    if "G_rad" in matrices:
        np.save(folder / "G_rad.npy", matrices["G_rad"])
    _write_browser_matrix_exports(matrices, folder)
    (folder / "simulations").mkdir(exist_ok=True)
    ui_state = {
        "selected_node_id": None,
        "filters": {"materials": [], "components": [], "levels": []},
    }
    path = folder / "ui_state.json"
    if not path.exists():
        _atomic_write_json(path, ui_state, indent=2)


def _flatten_node_row(row: dict[str, Any]) -> dict[str, Any]:
    flattened = dict(row)
    radiation = flattened.pop("radiation", None)
    if isinstance(radiation, dict):
        for key, value in radiation.items():
            flattened[f"radiation_{key}"] = value
    return flattened


def _write_browser_matrix_exports(matrices: dict[str, np.ndarray], folder: Path) -> None:
    if "C" in matrices:
        _atomic_write_json(folder / "C_diag.json", {"data": np.asarray(matrices["C"], dtype=float).tolist()})
    if "G_rad" in matrices:
        _atomic_write_json(folder / "G_rad_diag.json", {"data": np.asarray(matrices["G_rad"], dtype=float).tolist()})
    if "L" in matrices:
        L = np.asarray(matrices["L"], dtype=float)
        if L.ndim == 2:
            row, col = np.nonzero(L)
            payload = {
                "shape": list(L.shape),
                "format": "coo",
                "row": row.astype(int).tolist(),
                "col": col.astype(int).tolist(),
                "data": L[row, col].astype(float).tolist(),
            }
            _atomic_write_json(folder / "L_sparse.json", payload)


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


def _atomic_write_json(path: Path, payload: Any, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            json.dump(_json_ready(payload), handle, indent=indent)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
