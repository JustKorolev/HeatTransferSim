"""glTF/.glb scene loading through trimesh."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MeshObject:
    name: str
    material_name: str | None
    mesh: Any
    vertices_mm: np.ndarray
    bounds_mm: tuple[np.ndarray, np.ndarray]
    watertight: bool


@dataclass
class GltfScene:
    path: Path
    objects: list[MeshObject]
    bounds_mm: tuple[np.ndarray, np.ndarray]
    warnings: list[str]


def load_gltf_scene(path: str | Path) -> GltfScene:
    file_path = Path(path)
    _raise_for_missing_external_resources(file_path)
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required to load glTF/GLB geometry.") from exc

    loaded = trimesh.load(file_path, force="scene")
    warnings: list[str] = []
    objects: list[MeshObject] = []
    for node_name in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph.get(node_name)
        mesh = loaded.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        material_name = getattr(getattr(mesh.visual, "material", None), "name", None)
        bounds = np.asarray(mesh.bounds, dtype=float)
        if bounds.shape != (2, 3):
            warnings.append(f"Object {node_name} has invalid bounds and was skipped.")
            continue
        if not bool(getattr(mesh, "is_watertight", False)):
            warnings.append(f"Object {node_name} is not reported watertight; occupancy may be unreliable.")
        objects.append(
            MeshObject(
                name=str(node_name),
                material_name=str(material_name).strip() if material_name else None,
                mesh=mesh,
                vertices_mm=np.asarray(mesh.vertices, dtype=float),
                bounds_mm=(bounds[0], bounds[1]),
                watertight=bool(getattr(mesh, "is_watertight", False)),
            )
        )
    if not objects:
        raise ValueError(f"No mesh objects found in {file_path}.")
    mins = np.min([obj.bounds_mm[0] for obj in objects], axis=0)
    maxs = np.max([obj.bounds_mm[1] for obj in objects], axis=0)
    extent = float(np.max(maxs - mins))
    if 0.0 < extent < 10.0:
        warnings.append(
            "glTF transformed bounds are smaller than 10 units; treating coordinates as meters "
            "and scaling geometry to millimeters."
        )
        for obj in objects:
            obj.mesh.apply_scale(1000.0)
            obj.vertices_mm = np.asarray(obj.mesh.vertices, dtype=float)
            bounds = np.asarray(obj.mesh.bounds, dtype=float)
            obj.bounds_mm = (bounds[0], bounds[1])
        mins = np.min([obj.bounds_mm[0] for obj in objects], axis=0)
        maxs = np.max([obj.bounds_mm[1] for obj in objects], axis=0)
    return GltfScene(path=file_path, objects=objects, bounds_mm=(mins, maxs), warnings=warnings)


def _raise_for_missing_external_resources(file_path: Path) -> None:
    """Fail early with actionable paths when a .gltf omits its resource folder."""
    if file_path.suffix.lower() == ".glb":
        return
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            tree = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return
    missing: list[Path] = []
    for section in ("buffers", "images"):
        for item in tree.get(section, []) or []:
            uri = item.get("uri") if isinstance(item, dict) else None
            if not uri or uri.startswith("data:") or "://" in uri:
                continue
            resource_path = (file_path.parent / uri).resolve()
            if not resource_path.exists():
                missing.append(resource_path)
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing[:12])
        extra = "" if len(missing) <= 12 else f"\n- ... and {len(missing) - 12} more"
        raise FileNotFoundError(
            "The .gltf file references external resource files that are missing.\n"
            "Copy the export's resource folder next to the .gltf, or re-export as a self-contained .glb.\n"
            f"Missing resources:\n{formatted}{extra}"
        )
