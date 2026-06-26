"""Adaptive octree occupancy and subdivision."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from itertools import product
import math

import numpy as np

from .load_gltf import GltfScene, MeshObject
from .load_contact_report import ContactReport
from .materials import (
    DEFAULT_ASSIGNED_MATERIAL_NAME,
    Material,
    contrast_exceeds,
    is_unassigned_material_name,
    resolve_material,
)

_TRIMESH_CONTAINS_AVAILABLE: bool | None = None
_TRIANGLE_CACHE: dict[int, np.ndarray] = {}


@dataclass
class OctreeParams:
    min_cell_size_mm: float = 5.0
    max_cell_size_mm: float = 50.0
    max_depth: int = 8
    dominant_fraction_accept: float = 0.95
    minority_fraction_ignore: float = 0.02
    material_contrast_refine_threshold: float = 5.0
    contact_refine_distance_mm: float = 10.0
    boundary_refine: bool = True
    max_leaf_cells: int | None = None
    samples_per_cell: int = 9
    min_solid_fraction: float = 0.12
    bbox_fallback: bool = False


@dataclass
class OctreeCell:
    cell_id: str
    parent_id: str | None
    children_ids: list[str]
    level: int
    center_mm: tuple[float, float, float]
    size_mm: tuple[float, float, float]
    occupancy: dict[str, float]
    material_fractions: dict[str, float]
    dominant_component: str | None
    dominant_material: str | None
    confidence: str
    warnings: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.dominant_component

    @property
    def volume_m3(self) -> float:
        sx, sy, sz = self.size_mm
        return sx * sy * sz * 1.0e-9


def build_octree(
    scene: GltfScene,
    contact_report: ContactReport,
    materials: dict[str, Material],
    params: OctreeParams,
    warnings: list[str],
) -> list[OctreeCell]:
    mins, maxs = scene.bounds_mm
    center = (mins + maxs) * 0.5
    side = float(np.max(maxs - mins))
    root = (center, np.array([side, side, side], dtype=float))
    leaves: list[OctreeCell] = []
    counter = 0

    queue = deque([("cell_0", root[0], root[1], 0, None)])
    counter = 1

    def append_leaf(
        cell_id: str,
        center_mm: np.ndarray,
        size_mm: np.ndarray,
        level: int,
        parent: str | None,
        occupancy: dict[str, float],
        material_fractions: dict[str, float],
        dominant_component: str | None,
        dominant_material: str | None,
        confidence: str,
        cell_warnings: list[str],
    ) -> None:
        leaves.append(
            OctreeCell(
                cell_id=cell_id,
                parent_id=parent,
                children_ids=[],
                level=level,
                center_mm=tuple(float(v) for v in center_mm),
                size_mm=tuple(float(v) for v in size_mm),
                occupancy=occupancy,
                material_fractions=material_fractions,
                dominant_component=dominant_component,
                dominant_material=dominant_material,
                confidence=confidence,
                warnings=cell_warnings,
            )
        )

    while queue:
        cell_id, center_mm, size_mm, level, parent = queue.popleft()
        bbox_hits = _objects_intersecting_cell(scene.objects, center_mm, size_mm)
        occupancy, material_fractions, dominant_component, dominant_material, cell_warnings = _classify_cell(
            bbox_hits, center_mm, size_mm, contact_report, params, set(materials)
        )
        meaningful_materials = [
            resolve_material(name, materials, warnings)
            for name, frac in material_fractions.items()
            if frac > params.minority_fraction_ignore
        ]
        dominant_fraction = max(occupancy.values(), default=0.0)
        mixed_parts = sum(frac > params.minority_fraction_ignore for frac in occupancy.values()) > 1
        mixed_materials = sum(frac > params.minority_fraction_ignore for frac in material_fractions.values()) > 1
        high_contrast = contrast_exceeds(meaningful_materials, params.material_contrast_refine_threshold)
        contains_solid = bool(dominant_component)
        intersects_geometry = bool(bbox_hits)
        underfilled_solid = contains_solid and dominant_fraction < params.min_solid_fraction
        budget_allows_children = (
            params.max_leaf_cells is None
            or len(leaves) + len(queue) + 8 <= params.max_leaf_cells
        )
        can_subdivide = (
            level < params.max_depth
            and float(max(size_mm)) > params.min_cell_size_mm
            and budget_allows_children
        )
        should_subdivide = intersects_geometry and can_subdivide and (
            float(max(size_mm)) > params.max_cell_size_mm
            or mixed_parts
            or mixed_materials
            or high_contrast
            or underfilled_solid
            or (params.boundary_refine and 0.0 < dominant_fraction < params.dominant_fraction_accept)
            or (not contains_solid and float(max(size_mm)) > params.min_cell_size_mm)
        )
        if should_subdivide:
            quarter = size_mm * 0.25
            child_size = size_mm * 0.5
            for signs in product((-1.0, 1.0), repeat=3):
                child_id = f"cell_{counter}"
                counter += 1
                queue.append(
                    (
                        child_id,
                        center_mm + quarter * np.array(signs),
                        child_size,
                        level + 1,
                        cell_id,
                    )
                )
            continue

        confidence = "high"
        if dominant_fraction < params.dominant_fraction_accept:
            confidence = "medium" if 1.0 - dominant_fraction <= params.minority_fraction_ignore else "low"
            cell_warnings.append("Accepted with mixed/partial occupancy at refinement limit or tolerance.")
        if contains_solid and float(max(size_mm)) > params.max_cell_size_mm:
            append_leaf(
                cell_id,
                center_mm,
                size_mm,
                level,
                parent,
                {},
                {},
                None,
                None,
                "low",
                [
                    "Left empty because cell exceeded max_cell_size_mm but refinement was blocked "
                    "by max_depth or max_leaf_cells."
                ],
            )
            continue
        if contains_solid and dominant_fraction < params.min_solid_fraction:
            append_leaf(
                cell_id,
                center_mm,
                size_mm,
                level,
                parent,
                {},
                {},
                None,
                None,
                "low",
                [
                    "Left empty because sampled solid occupancy was below min_solid_fraction "
                    f"({dominant_fraction:.3g} < {params.min_solid_fraction:.3g})."
                ],
            )
            continue
        if (
            not dominant_component
            and bbox_hits
            and not can_subdivide
            and params.bbox_fallback
            and float(max(size_mm)) <= params.min_cell_size_mm * 1.01
        ):
            fallback = min(bbox_hits, key=lambda obj: float(np.prod(obj.bounds_mm[1] - obj.bounds_mm[0])))
            dominant_component = fallback.name
            dominant_material = _physical_material_name(fallback, contact_report, set(materials))
            occupancy = {dominant_component: 1.0}
            material_fractions = {dominant_material: 1.0}
            confidence = "low"
            cell_warnings.append("Assigned by bounding-box fallback because mesh containment was unreliable.")
        append_leaf(
            cell_id,
            center_mm,
            size_mm,
            level,
            parent,
            occupancy,
            material_fractions,
            dominant_component,
            dominant_material,
            confidence,
            cell_warnings,
        )
    return leaves


def _classify_cell(
    objects: list[MeshObject],
    center_mm: np.ndarray,
    size_mm: np.ndarray,
    contact_report: ContactReport,
    params: OctreeParams,
    known_materials: set[str],
) -> tuple[dict[str, float], dict[str, float], str | None, str | None, list[str]]:
    points = _sample_points(center_mm, size_mm, params.samples_per_cell)
    component_counts: Counter[str] = Counter()
    material_counts: Counter[str] = Counter()
    warnings: list[str] = []
    for point in points:
        hits = []
        for obj in objects:
            if np.any(point < obj.bounds_mm[0]) or np.any(point > obj.bounds_mm[1]):
                continue
            try:
                if _mesh_contains_point(obj, point):
                    hits.append(obj)
            except Exception:
                warnings.append(f"Inside/outside test failed for {obj.name}.")
        if hits:
            obj = hits[0]
            component_counts[obj.name] += 1
            material = _physical_material_name(obj, contact_report, known_materials)
            material_counts[material] += 1
    total = float(len(points))
    occupancy = {name: count / total for name, count in component_counts.items()}
    material_fractions = {name: count / total for name, count in material_counts.items()}
    dominant_component = component_counts.most_common(1)[0][0] if component_counts else None
    dominant_material = material_counts.most_common(1)[0][0] if material_counts else None
    return occupancy, material_fractions, dominant_component, dominant_material, warnings[:3]


def _physical_material_name(
    obj: MeshObject, contact_report: ContactReport, known_materials: set[str]
) -> str:
    if (
        obj.material_name
        and obj.material_name in known_materials
        and not is_unassigned_material_name(obj.material_name)
    ):
        return obj.material_name
    report_material = contact_report.material_for_component(obj.name)
    if (
        report_material
        and report_material in known_materials
        and not is_unassigned_material_name(report_material)
    ):
        return report_material
    return DEFAULT_ASSIGNED_MATERIAL_NAME


def _mesh_contains_point(obj: MeshObject, point: np.ndarray) -> bool:
    global _TRIMESH_CONTAINS_AVAILABLE
    if _TRIMESH_CONTAINS_AVAILABLE is not False:
        try:
            inside = bool(obj.mesh.contains([point])[0])
            _TRIMESH_CONTAINS_AVAILABLE = True
            return inside
        except (ImportError, ModuleNotFoundError):
            _TRIMESH_CONTAINS_AVAILABLE = False
        except Exception:
            return _ray_contains_point(obj, point)
    return _ray_contains_point(obj, point)


def _mesh_triangles(obj: MeshObject) -> np.ndarray:
    cache_key = id(obj.mesh)
    triangles = _TRIANGLE_CACHE.get(cache_key)
    if triangles is None:
        triangles = np.asarray(obj.mesh.triangles, dtype=float)
        _TRIANGLE_CACHE[cache_key] = triangles
    return triangles


def _ray_contains_point(obj: MeshObject, point: np.ndarray) -> bool:
    """Odd/even ray test used when trimesh.contains lacks optional rtree."""
    triangles = _mesh_triangles(obj)
    if triangles.size == 0:
        return False
    direction = np.array([1.0, 0.3713906763541037, 0.1937728766089219], dtype=float)
    direction /= np.linalg.norm(direction)
    eps = 1.0e-9
    v0 = triangles[:, 0, :]
    edge1 = triangles[:, 1, :] - v0
    edge2 = triangles[:, 2, :] - v0
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    a = np.einsum("ij,ij->i", edge1, h)
    mask = np.abs(a) > eps
    if not np.any(mask):
        return False
    f = np.zeros_like(a)
    f[mask] = 1.0 / a[mask]
    s = point - v0
    u = f * np.einsum("ij,ij->i", s, h)
    mask &= (u >= -eps) & (u <= 1.0 + eps)
    if not np.any(mask):
        return False
    q = np.cross(s, edge1)
    v = f * np.einsum("ij,j->i", q, direction)
    mask &= (v >= -eps) & ((u + v) <= 1.0 + eps)
    if not np.any(mask):
        return False
    t = f * np.einsum("ij,ij->i", edge2, q)
    hits = t[mask & (t > eps)]
    if hits.size == 0:
        return False
    unique_hits = np.unique(np.round(hits, decimals=8))
    return bool(unique_hits.size % 2 == 1)


def _objects_intersecting_cell(
    objects: list[MeshObject], center_mm: np.ndarray, size_mm: np.ndarray
) -> list[MeshObject]:
    half = size_mm * 0.5
    cell_min = center_mm - half
    cell_max = center_mm + half
    hits: list[MeshObject] = []
    for obj in objects:
        obj_min, obj_max = obj.bounds_mm
        if np.all(cell_max >= obj_min) and np.all(obj_max >= cell_min):
            hits.append(obj)
    return hits


def _sample_points(center_mm: np.ndarray, size_mm: np.ndarray, samples_per_cell: int) -> np.ndarray:
    half = size_mm * 0.5
    points = [center_mm]
    for signs in product((-1.0, 1.0), repeat=3):
        points.append(center_mm + 0.45 * half * np.array(signs))
    remaining = max(0, samples_per_cell - len(points))
    if remaining:
        grid_n = max(1, math.ceil(remaining ** (1.0 / 3.0)))
        offsets = np.linspace(-0.3, 0.3, grid_n)
        for offset in product(offsets, repeat=3):
            points.append(center_mm + half * np.array(offset))
            if len(points) >= samples_per_cell:
                break
    return np.asarray(points[: max(1, samples_per_cell)], dtype=float)
