"""glTF/.glb scene loading through trimesh."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import urlparse

import numpy as np


@dataclass
class MeshObject:
    name: str
    material_name: str | None
    mesh: Any
    vertices_mm: np.ndarray
    bounds_mm: tuple[np.ndarray, np.ndarray]
    watertight: bool
    scene_path: str | None = None
    hierarchy_path: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class GltfScene:
    path: Path
    objects: list[MeshObject]
    bounds_mm: tuple[np.ndarray, np.ndarray]
    warnings: list[str]


@dataclass
class _ArrayTriangleMesh:
    vertices: np.ndarray
    faces: np.ndarray
    triangles: np.ndarray
    is_watertight: bool = False
    volume: float = 0.0


def load_gltf_scene(path: str | Path) -> GltfScene:
    file_path = Path(path)
    load_path, temporary_path, resource_warnings = _prepare_gltf_for_load(file_path)
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required to load glTF/GLB geometry.") from exc

    try:
        loaded = trimesh.load(load_path, force="scene")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    warnings: list[str] = list(resource_warnings)
    objects: list[MeshObject] = []
    raw_mesh_node_paths = _raw_gltf_mesh_node_paths(file_path)
    node_paths = _graph_node_paths(loaded.graph)
    nodes_geometry = list(loaded.graph.nodes_geometry)
    if raw_mesh_node_paths and len(raw_mesh_node_paths) != len(nodes_geometry):
        warnings.append(
            f"Raw glTF hierarchy has {len(raw_mesh_node_paths)} mesh node path(s), while trimesh exposed "
            f"{len(nodes_geometry)} geometry node(s); using raw hierarchy paths where ordinals overlap."
        )
    for ordinal, node_name in enumerate(nodes_geometry):
        transform, geometry_name = loaded.graph.get(node_name)
        hierarchy_path = node_paths.get(str(node_name), (str(node_name),))
        if ordinal < len(raw_mesh_node_paths):
            hierarchy_path = raw_mesh_node_paths[ordinal][1]
        obj = _mesh_object_from_geometry(
            node_name=str(node_name),
            geometry_name=str(geometry_name),
            geometry=loaded.geometry[geometry_name],
            transform=np.asarray(transform, dtype=float),
            warnings=warnings,
            hierarchy_path=hierarchy_path,
        )
        if obj is None:
            continue
        objects.append(obj)
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
            _scale_mesh_object(obj, 1000.0)
        mins = np.min([obj.bounds_mm[0] for obj in objects], axis=0)
        maxs = np.max([obj.bounds_mm[1] for obj in objects], axis=0)
    return GltfScene(path=file_path, objects=objects, bounds_mm=(mins, maxs), warnings=warnings)


def _mesh_object_from_geometry(
    node_name: str,
    geometry_name: str,
    geometry: Any,
    transform: np.ndarray,
    warnings: list[str],
    hierarchy_path: tuple[str, ...] | None = None,
) -> MeshObject | None:
    try:
        vertices = np.asarray(getattr(geometry, "vertices", []), dtype=float)
    except Exception:
        vertices = np.empty((0, 3), dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.size == 0:
        warnings.append(f"Object {node_name} has no valid vertices and was skipped.")
        return None
    vertices = _transform_vertices(vertices, transform)
    if not np.all(np.isfinite(vertices)):
        warnings.append(f"Object {node_name} has non-finite transformed vertices and was skipped.")
        return None

    try:
        faces = np.asarray(getattr(geometry, "faces", []), dtype=int)
    except Exception:
        faces = np.empty((0, 3), dtype=int)
    if faces.ndim != 2 or faces.shape[1] != 3:
        faces = np.empty((0, 3), dtype=int)
    valid_faces = faces[
        np.all((faces >= 0) & (faces < len(vertices)), axis=1)
    ] if faces.size else np.empty((0, 3), dtype=int)
    triangles = vertices[valid_faces] if valid_faces.size else np.empty((0, 3, 3), dtype=float)
    bounds = np.asarray([np.min(vertices, axis=0), np.max(vertices, axis=0)], dtype=float)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        warnings.append(f"Object {node_name} has invalid bounds and was skipped.")
        return None

    watertight = _safe_bool_attr(geometry, "is_watertight", default=False)
    if not watertight:
        warnings.append(f"Object {node_name} is not reported watertight; occupancy may be unreliable.")
    material_name = _safe_material_name(geometry)
    clean_path = tuple(str(part) for part in (hierarchy_path or (node_name,)) if str(part))
    base_scene_path = "/".join(clean_path) if clean_path else node_name
    scene_path = base_scene_path if geometry_name == node_name else f"{base_scene_path} {geometry_name}"
    mesh = _ArrayTriangleMesh(
        vertices=vertices,
        faces=valid_faces,
        triangles=triangles,
        is_watertight=watertight,
        volume=_bounds_volume_mm3(bounds),
    )
    return MeshObject(
        name=node_name,
        material_name=material_name,
        mesh=mesh,
        vertices_mm=vertices.astype(float, copy=True),
        bounds_mm=(bounds[0].astype(float, copy=True), bounds[1].astype(float, copy=True)),
        watertight=watertight,
        scene_path=scene_path,
        hierarchy_path=clean_path,
    )


def _graph_node_paths(graph: Any) -> dict[str, tuple[str, ...]]:
    """Best-effort extraction of full scene graph paths from trimesh."""
    try:
        geometry_nodes = [str(node) for node in graph.nodes_geometry]
    except Exception:
        return {}
    parent_by_child = _graph_parent_lookup(graph)
    paths: dict[str, tuple[str, ...]] = {}
    for node in geometry_nodes:
        parts: list[str] = []
        seen: set[str] = set()
        current: str | None = str(node)
        while current is not None and current not in seen:
            seen.add(current)
            parts.append(str(current))
            parent = parent_by_child.get(str(current))
            current = str(parent) if parent is not None else None
        parts.reverse()
        paths[str(node)] = tuple(parts) if parts else (str(node),)
    return paths


def _raw_gltf_mesh_node_paths(file_path: Path) -> list[tuple[str, tuple[str, ...]]]:
    tree = _raw_gltf_json(file_path)
    if not tree:
        return []
    nodes = tree.get("nodes", []) or []
    if not isinstance(nodes, list):
        return []
    child_ids = {
        int(child)
        for node in nodes
        if isinstance(node, dict)
        for child in (node.get("children", []) or [])
        if _can_int(child)
    }
    roots = [index for index in range(len(nodes)) if index not in child_ids]
    mesh_paths: list[tuple[str, tuple[str, ...]]] = []

    def label(index: int) -> str:
        node = nodes[index] if 0 <= index < len(nodes) and isinstance(nodes[index], dict) else {}
        name = str(node.get("name") or f"node_{index}")
        return f"{name}#{index}"

    def walk(index: int, parts: list[str]) -> None:
        if index < 0 or index >= len(nodes) or not isinstance(nodes[index], dict):
            return
        node = nodes[index]
        current = [*parts, label(index)]
        if "mesh" in node:
            mesh_paths.append((str(node.get("name") or f"node_{index}"), tuple(current)))
        for child in node.get("children", []) or []:
            if _can_int(child):
                walk(int(child), current)

    for root in roots:
        walk(root, [])
    return mesh_paths


def _raw_gltf_json(file_path: Path) -> dict[str, Any] | None:
    try:
        suffix = file_path.suffix.lower()
        if suffix == ".glb":
            data = file_path.read_bytes()
            if len(data) < 20 or data[:4] != b"glTF":
                return None
            offset = 12
            while offset + 8 <= len(data):
                chunk_len = int.from_bytes(data[offset : offset + 4], "little")
                chunk_type = int.from_bytes(data[offset + 4 : offset + 8], "little")
                offset += 8
                chunk = data[offset : offset + chunk_len]
                offset += chunk_len
                if chunk_type == 0x4E4F534A:
                    return json.loads(chunk.decode("utf-8").rstrip("\x00 \t\r\n"))
            return None
        if suffix == ".gltf":
            with file_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    except Exception:
        return None
    return None


def _can_int(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def _graph_parent_lookup(graph: Any) -> dict[str, str]:
    transforms = getattr(graph, "transforms", None)
    candidates = [
        getattr(graph, "parents", None),
        getattr(transforms, "parents", None),
        getattr(graph, "_parents", None),
        getattr(transforms, "_parents", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return {str(child): str(parent) for child, parent in candidate.items() if parent is not None}
    children_candidates = [
        getattr(graph, "children", None),
        getattr(transforms, "children", None),
        getattr(graph, "_children", None),
        getattr(transforms, "_children", None),
    ]
    for candidate in children_candidates:
        if not isinstance(candidate, dict) or not candidate:
            continue
        parents: dict[str, str] = {}
        for parent, children in candidate.items():
            for child in children or []:
                parents[str(child)] = str(parent)
        if parents:
            return parents
    return {}


def _transform_vertices(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if transform.shape != (4, 4):
        return vertices.astype(float, copy=True)
    homogenous = np.column_stack([vertices, np.ones(len(vertices), dtype=float)])
    return (homogenous @ transform.T)[:, :3]


def _scale_mesh_object(obj: MeshObject, scale: float) -> None:
    factor = float(scale)
    vertices = np.asarray(obj.vertices_mm, dtype=float) * factor
    triangles = np.asarray(getattr(obj.mesh, "triangles", []), dtype=float) * factor
    faces = np.asarray(getattr(obj.mesh, "faces", []), dtype=int)
    bounds = np.asarray([np.min(vertices, axis=0), np.max(vertices, axis=0)], dtype=float)
    obj.vertices_mm = vertices
    obj.bounds_mm = (bounds[0].astype(float, copy=True), bounds[1].astype(float, copy=True))
    obj.mesh = _ArrayTriangleMesh(
        vertices=vertices,
        faces=faces,
        triangles=triangles,
        is_watertight=bool(getattr(obj, "watertight", False)),
        volume=_bounds_volume_mm3(bounds),
    )


def _safe_bool_attr(obj: Any, name: str, default: bool = False) -> bool:
    try:
        return bool(getattr(obj, name, default))
    except Exception:
        return bool(default)


def _safe_material_name(geometry: Any) -> str | None:
    try:
        material_name = getattr(getattr(geometry.visual, "material", None), "name", None)
    except Exception:
        material_name = None
    return str(material_name).strip() if material_name else None


def _bounds_volume_mm3(bounds: np.ndarray) -> float:
    try:
        size = np.maximum(np.asarray(bounds[1], dtype=float) - np.asarray(bounds[0], dtype=float), 0.0)
        volume = float(np.prod(size))
    except Exception:
        return 0.0
    return volume if np.isfinite(volume) and volume > 0.0 else 0.0


_PLACEHOLDER_IMAGE_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _prepare_gltf_for_load(file_path: Path) -> tuple[Path, Path | None, list[str]]:
    """Return a glTF path whose external resources are resolvable.

    SolidWorks exports are often moved around as ``assembly.gltf`` plus a
    resource folder, or as a subfolder that already contains ``assembly.bin``.
    This prepares a temporary glTF with corrected URIs when the files can be
    found without forcing callers to duplicate resource folders.
    """
    if file_path.suffix.lower() == ".glb":
        return file_path, None, []
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            tree = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return file_path, None, []
    changed = False
    missing_buffers: list[Path] = []
    remaps: list[tuple[dict[str, Any], Path]] = []
    warnings: list[str] = []
    for item in tree.get("buffers", []) or []:
        if not isinstance(item, dict):
            continue
        uri = item.get("uri")
        if not _is_external_file_uri(uri):
            continue
        resolved = _resolve_resource_path(file_path, str(uri), item.get("byteLength"))
        if resolved is None:
            missing_buffers.append((file_path.parent / str(uri)).resolve())
            continue
        if resolved != _default_resource_path(file_path, str(uri)):
            remaps.append((item, resolved))
            changed = True
    for item in tree.get("images", []) or []:
        if not isinstance(item, dict):
            continue
        uri = item.get("uri")
        if not _is_external_file_uri(uri):
            continue
        resolved = _resolve_resource_path(file_path, str(uri))
        if resolved is None:
            item["uri"] = _PLACEHOLDER_IMAGE_URI
            changed = True
            warnings.append(f"Texture resource {uri!r} was not found; using a placeholder image for geometry load.")
            continue
        if resolved != _default_resource_path(file_path, str(uri)):
            remaps.append((item, resolved))
            changed = True
    if missing_buffers:
        _raise_for_missing_external_resources(missing_buffers)
    if not changed:
        return file_path, None, warnings
    handle, temporary_path = _open_temporary_gltf(file_path)
    with handle:
        for item, resolved in remaps:
            item["uri"] = _resource_uri_for_temp_gltf(resolved, temporary_path.parent)
        json.dump(tree, handle)
    return temporary_path, temporary_path, warnings


def _is_external_file_uri(uri: Any) -> bool:
    if not isinstance(uri, str) or not uri:
        return False
    if uri.startswith("data:"):
        return False
    parsed = urlparse(uri)
    return parsed.scheme in {"", "file"}


def _resolve_resource_path(file_path: Path, uri: str, min_byte_length: Any = None) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        candidate = Path(parsed.path)
        return candidate.resolve() if candidate.exists() else None
    parent = file_path.parent
    raw_path = Path(uri)
    candidates: list[Path] = [(parent / raw_path).resolve()]
    parts = raw_path.parts
    if len(parts) > 1 and parts[0].lower() == parent.name.lower():
        candidates.append((parent / Path(*parts[1:])).resolve())
    if raw_path.name:
        candidates.append((parent / raw_path.name).resolve())
    if raw_path.suffix.lower() == ".bin" and len(parts) > 1:
        candidates.extend(_renamed_buffer_candidates(file_path))
    for candidate in candidates:
        if _resource_candidate_is_valid(candidate, min_byte_length):
            return candidate
    if raw_path.name:
        matches: list[Path] = []
        for root in (parent, parent.parent):
            try:
                matches.extend(path.resolve() for path in root.rglob(raw_path.name) if path.is_file())
            except OSError:
                continue
        unique = sorted(set(matches), key=lambda path: (len(path.parts), str(path).lower()))
        if unique:
            return unique[0]
    return None


def _renamed_buffer_candidates(file_path: Path) -> list[Path]:
    """Common case: the glTF was renamed with its sibling .bin file."""
    parent = file_path.parent
    candidates = [
        file_path.with_suffix(".bin"),
        parent / f"{parent.name}.bin",
    ]
    try:
        candidates.extend(parent.glob("*.bin"))
    except OSError:
        pass
    return [path.resolve() for path in candidates]


def _resource_candidate_is_valid(path: Path, min_byte_length: Any = None) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if min_byte_length is None:
        return True
    try:
        required_size = int(min_byte_length)
    except (TypeError, ValueError):
        return True
    try:
        return path.stat().st_size >= required_size
    except OSError:
        return False


def _default_resource_path(file_path: Path, uri: str) -> Path:
    return (file_path.parent / Path(uri)).resolve()


def _open_temporary_gltf(file_path: Path) -> tuple[Any, Path]:
    kwargs = {
        "mode": "w",
        "encoding": "utf-8",
        "suffix": ".gltf",
        "prefix": f".{file_path.stem}.",
        "delete": False,
    }
    try:
        handle = tempfile.NamedTemporaryFile(dir=file_path.parent, **kwargs)
    except OSError:
        handle = tempfile.NamedTemporaryFile(**kwargs)
    return handle, Path(handle.name)


def _resource_uri_for_temp_gltf(resource_path: Path, temp_dir: Path) -> str:
    try:
        return os.path.relpath(resource_path, temp_dir).replace(os.sep, "/")
    except ValueError:
        return resource_path.as_uri()


def _raise_for_missing_external_resources(missing: list[Path]) -> None:
    """Fail early with actionable paths when required .bin buffers are missing."""
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing[:12])
        extra = "" if len(missing) <= 12 else f"\n- ... and {len(missing) - 12} more"
        raise FileNotFoundError(
            "The .gltf file references external buffer files that are missing.\n"
            "Place the .bin next to the .gltf or in the referenced resource folder. "
            "The loader also checks common sibling/nested SolidWorks export layouts.\n"
            f"Missing buffers:\n{formatted}{extra}"
        )
