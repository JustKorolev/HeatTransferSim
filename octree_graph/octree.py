"""Adaptive octree occupancy and subdivision for CAD assemblies."""

from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import product
import math
import os
from typing import Callable

import numpy as np

from .load_gltf import GltfScene, MeshObject
from .load_contact_report import ContactReport
from .materials import (
    DEFAULT_ASSIGNED_MATERIAL_NAME,
    Material,
    contrast_exceeds,
    infer_material_name_from_text,
    is_unassigned_material_name,
    resolve_material,
)

_TRIMESH_CONTAINS_AVAILABLE: bool | None = None
_TRIANGLE_CACHE: dict[int, np.ndarray] = {}
_TRIANGLE_INDEX_CACHE: dict[int, "TriangleSpatialIndex"] = {}
_WORKER_OBJECTS: list[MeshObject] = []
_WORKER_TRIANGLE_INDICES: dict[int, "TriangleSpatialIndex"] = {}
_WORKER_CONTACT_REPORT: ContactReport | None = None
_WORKER_PARAMS: "OctreeParams | None" = None
_WORKER_KNOWN_MATERIALS: set[str] = set()


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
    voxel_workers: int = 1
    voxel_batch_size: int = 64


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


@dataclass
class CellClassification:
    occupied: bool
    surface_hit: bool
    inside_hit: bool
    near_surface_hit: bool
    bbox_only_hit: bool
    surface_mesh_ids: set[int]
    inside_mesh_ids: set[int]
    near_surface_mesh_ids: set[int]
    candidate_mesh_ids: set[int]
    material_ids: set[str]
    part_ids: set[str]
    occupancy: dict[str, float]
    material_fractions: dict[str, float]
    dominant_component: str | None
    dominant_material: str | None
    volume_fraction: float | None
    acceptance_reason: str
    warnings: list[str] = field(default_factory=list)
    triangle_candidate_tests: int = 0
    triangle_intersection_tests: int = 0


@dataclass
class _CellWorkItem:
    cell_id: str
    center_mm: tuple[float, float, float]
    size_mm: tuple[float, float, float]
    level: int
    parent_id: str | None


@dataclass
class _TriangleMesh:
    triangles: np.ndarray


@dataclass
class OctreeDiagnostics:
    root_bounds_mm: dict[str, list[float]] = field(default_factory=dict)
    root_cell_size_mm: list[float] = field(default_factory=list)
    cells_tested: int = 0
    cells_subdivided: int = 0
    cells_rejected_empty: int = 0
    cells_accepted_exact: int = 0
    cells_accepted_bbox_fallback: int = 0
    cells_surface_hit: int = 0
    cells_inside_hit: int = 0
    cells_near_surface_hit: int = 0
    cells_bbox_only_hit: int = 0
    triangle_candidate_tests: int = 0
    triangle_intersection_tests: int = 0
    max_depth_reached: int = 0
    max_leaf_cells_reached: bool = False
    leaves_by_depth: dict[int, int] = field(default_factory=dict)
    leaves_by_cell_size_mm: dict[str, int] = field(default_factory=dict)
    debug_leaves: bool = False
    leaf_records: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "root_bounds_mm": self.root_bounds_mm,
            "root_cell_size_mm": self.root_cell_size_mm,
            "cells_tested": self.cells_tested,
            "cells_subdivided": self.cells_subdivided,
            "cells_rejected_empty": self.cells_rejected_empty,
            "cells_accepted_exact": self.cells_accepted_exact,
            "cells_accepted_bbox_fallback": self.cells_accepted_bbox_fallback,
            "cells_surface_hit": self.cells_surface_hit,
            "cells_inside_hit": self.cells_inside_hit,
            "cells_near_surface_hit": self.cells_near_surface_hit,
            "cells_bbox_only_hit": self.cells_bbox_only_hit,
            "triangle_candidate_tests": self.triangle_candidate_tests,
            "triangle_intersection_tests": self.triangle_intersection_tests,
            "max_depth_reached": self.max_depth_reached,
            "max_leaf_cells_reached": self.max_leaf_cells_reached,
            "leaves_by_depth": {str(key): value for key, value in sorted(self.leaves_by_depth.items())},
            "leaves_by_cell_size_mm": dict(sorted(self.leaves_by_cell_size_mm.items())),
            "leaf_records": self.leaf_records,
        }


@dataclass
class TriangleSpatialIndex:
    triangles: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    bucket_size_mm: float
    buckets: dict[tuple[int, int, int], list[int]]

    @classmethod
    def from_mesh(cls, obj: MeshObject, target_bucket_size_mm: float) -> "TriangleSpatialIndex":
        triangles = _mesh_triangles(obj)
        if triangles.size == 0:
            empty = np.empty((0, 3), dtype=float)
            return cls(triangles, empty, empty, max(float(target_bucket_size_mm), 1.0), {})
        bounds_min = np.min(triangles, axis=1)
        bounds_max = np.max(triangles, axis=1)
        mesh_min, mesh_max = obj.bounds_mm
        extent = float(np.max(np.asarray(mesh_max, dtype=float) - np.asarray(mesh_min, dtype=float)))
        bucket_size = max(float(target_bucket_size_mm), extent / 64.0, 1.0e-6)
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for index, (tri_min, tri_max) in enumerate(zip(bounds_min, bounds_max)):
            min_key = _bucket_key(tri_min, bucket_size)
            max_key = _bucket_key(tri_max, bucket_size)
            for ix in range(min_key[0], max_key[0] + 1):
                for iy in range(min_key[1], max_key[1] + 1):
                    for iz in range(min_key[2], max_key[2] + 1):
                        buckets.setdefault((ix, iy, iz), []).append(index)
        return cls(triangles, bounds_min, bounds_max, bucket_size, buckets)

    def query(self, cell_min: np.ndarray, cell_max: np.ndarray) -> np.ndarray:
        if self.triangles.size == 0:
            return np.empty((0,), dtype=int)
        min_key = _bucket_key(cell_min, self.bucket_size_mm)
        max_key = _bucket_key(cell_max, self.bucket_size_mm)
        bucket_span = (
            max_key[0] - min_key[0] + 1,
            max_key[1] - min_key[1] + 1,
            max_key[2] - min_key[2] + 1,
        )
        bucket_count = int(bucket_span[0] * bucket_span[1] * bucket_span[2])
        if bucket_count > max(4096, len(self.buckets) * 2):
            return self._query_all_bounds(cell_min, cell_max)
        matches: set[int] = set()
        for ix in range(min_key[0], max_key[0] + 1):
            for iy in range(min_key[1], max_key[1] + 1):
                for iz in range(min_key[2], max_key[2] + 1):
                    matches.update(self.buckets.get((ix, iy, iz), ()))
        if not matches:
            return np.empty((0,), dtype=int)
        candidates = np.fromiter(sorted(matches), dtype=int)
        overlap = np.all(self.bounds_max[candidates] >= cell_min, axis=1) & np.all(
            self.bounds_min[candidates] <= cell_max, axis=1
        )
        return candidates[overlap]

    def _query_all_bounds(self, cell_min: np.ndarray, cell_max: np.ndarray) -> np.ndarray:
        overlap = np.all(self.bounds_max >= cell_min, axis=1) & np.all(self.bounds_min <= cell_max, axis=1)
        return np.nonzero(overlap)[0].astype(int)


def build_octree(
    scene: GltfScene,
    contact_report: ContactReport | None,
    materials: dict[str, Material],
    params: OctreeParams,
    warnings: list[str],
    diagnostics: OctreeDiagnostics | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> list[OctreeCell]:
    contact_report = contact_report or ContactReport()
    mins, maxs = scene.bounds_mm
    center = (mins + maxs) * 0.5
    side = float(np.max(maxs - mins))
    root = (center, np.array([side, side, side], dtype=float))
    leaves: list[OctreeCell] = []
    triangle_indices = _build_triangle_indices(scene.objects, params)

    if diagnostics is not None:
        diagnostics.root_bounds_mm = {"min": mins.astype(float).tolist(), "max": maxs.astype(float).tolist()}
        diagnostics.root_cell_size_mm = root[1].astype(float).tolist()

    queue = deque([_CellWorkItem("cell_0", tuple(float(v) for v in root[0]), tuple(float(v) for v in root[1]), 0, None)])
    counter = 1

    def append_leaf(
        cell_id: str,
        center_mm: np.ndarray,
        size_mm: np.ndarray,
        level: int,
        parent: str | None,
        classification: CellClassification,
        confidence: str,
    ) -> None:
        if diagnostics is not None:
            _record_leaf_diagnostics(diagnostics, cell_id, size_mm, level, classification)
        leaves.append(
            OctreeCell(
                cell_id=cell_id,
                parent_id=parent,
                children_ids=[],
                level=level,
                center_mm=tuple(float(v) for v in center_mm),
                size_mm=tuple(float(v) for v in size_mm),
                occupancy=classification.occupancy,
                material_fractions=classification.material_fractions,
                dominant_component=classification.dominant_component,
                dominant_material=classification.dominant_material,
                confidence=confidence,
                warnings=classification.warnings,
            )
        )

    worker_count = _resolve_voxel_worker_count(params)
    batch_size = max(1, int(params.voxel_batch_size))

    def handle_classified_cell(
        work_item: _CellWorkItem,
        classification: CellClassification,
        remaining_batch_items: int,
    ) -> None:
        nonlocal counter
        center_mm = np.asarray(work_item.center_mm, dtype=float)
        size_mm = np.asarray(work_item.size_mm, dtype=float)
        level = int(work_item.level)
        if diagnostics is not None:
            diagnostics.cells_tested += 1
            diagnostics.max_depth_reached = max(diagnostics.max_depth_reached, level)
            diagnostics.triangle_candidate_tests += int(classification.triangle_candidate_tests)
            diagnostics.triangle_intersection_tests += int(classification.triangle_intersection_tests)
        if progress_callback is not None and diagnostics is not None:
            progress_callback(
                {
                    "phase": "octree",
                    "cells_tested": diagnostics.cells_tested,
                    "cells_subdivided": diagnostics.cells_subdivided,
                    "leaves": len(leaves),
                    "queue": len(queue) + int(remaining_batch_items),
                    "max_leaf_cells": params.max_leaf_cells,
                    "max_depth_reached": diagnostics.max_depth_reached,
                    "voxel_workers": worker_count,
                }
            )
        meaningful_materials = [
            resolve_material(name, materials, warnings)
            for name, frac in classification.material_fractions.items()
            if frac > params.minority_fraction_ignore
        ]
        dominant_fraction = max(classification.occupancy.values(), default=0.0)
        mixed_parts = sum(frac > params.minority_fraction_ignore for frac in classification.occupancy.values()) > 1
        mixed_materials = (
            sum(frac > params.minority_fraction_ignore for frac in classification.material_fractions.values()) > 1
        )
        high_contrast = contrast_exceeds(meaningful_materials, params.material_contrast_refine_threshold)
        needs_surface_refinement = params.boundary_refine and (
            classification.surface_hit or classification.near_surface_hit
        )
        effective_queue_len = len(queue) + int(remaining_batch_items)
        budget_allows_children = (
            params.max_leaf_cells is None
            or len(leaves) + effective_queue_len + 8 <= params.max_leaf_cells
        )
        if diagnostics is not None and params.max_leaf_cells is not None and not budget_allows_children:
            diagnostics.max_leaf_cells_reached = True
        can_subdivide = (
            level < params.max_depth
            and float(max(size_mm)) > params.min_cell_size_mm
            and budget_allows_children
        )
        should_subdivide = can_subdivide and (
            (classification.occupied and float(max(size_mm)) > params.max_cell_size_mm)
            or mixed_parts
            or mixed_materials
            or high_contrast
            or (needs_surface_refinement and float(max(size_mm)) > params.min_cell_size_mm)
            or (
                classification.occupied
                and 0.0 < dominant_fraction < params.dominant_fraction_accept
                and float(max(size_mm)) > params.min_cell_size_mm
            )
        )
        if should_subdivide:
            if diagnostics is not None:
                diagnostics.cells_subdivided += 1
            quarter = size_mm * 0.25
            child_size = size_mm * 0.5
            for signs in product((-1.0, 1.0), repeat=3):
                child_id = f"cell_{counter}"
                counter += 1
                queue.append(
                    _CellWorkItem(
                        child_id,
                        tuple(float(v) for v in center_mm + quarter * np.array(signs)),
                        tuple(float(v) for v in child_size),
                        level + 1,
                        work_item.cell_id,
                    )
                )
            return

        confidence = _classification_confidence(classification, params)
        if classification.occupied and float(max(size_mm)) > params.max_cell_size_mm:
            classification.warnings.append(
                "Accepted occupied cell above max_cell_size_mm because refinement was blocked "
                "by max_depth or max_leaf_cells."
            )
            confidence = "low"
        if classification.bbox_only_hit and params.bbox_fallback:
            classification.warnings.append(
                "Ignored legacy bbox fallback request; AABB overlap is not used as physical occupancy."
            )
        append_leaf(
            work_item.cell_id,
            center_mm,
            size_mm,
            level,
            work_item.parent_id,
            classification,
            confidence,
        )

    if worker_count <= 1:
        while queue:
            work_item = queue.popleft()
            classification = _classify_cell(
                scene.objects,
                triangle_indices,
                np.asarray(work_item.center_mm, dtype=float),
                np.asarray(work_item.size_mm, dtype=float),
                contact_report,
                params,
                set(materials),
                None,
            )
            handle_classified_cell(work_item, classification, remaining_batch_items=0)
    else:
        worker_objects = _prepare_worker_objects(scene.objects)
        batch: list[_CellWorkItem] = []
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_octree_worker,
                initargs=(worker_objects, contact_report, params, set(materials)),
            ) as executor:
                while queue:
                    batch: list[_CellWorkItem] = []
                    while queue and len(batch) < batch_size:
                        batch.append(queue.popleft())
                    classifications = list(
                        executor.map(
                            _classify_cell_work_item,
                            batch,
                            chunksize=max(1, min(8, batch_size // max(worker_count, 1))),
                        )
                    )
                    for index, (work_item, classification) in enumerate(zip(batch, classifications)):
                        handle_classified_cell(
                            work_item,
                            classification,
                            remaining_batch_items=len(batch) - index - 1,
                        )
        except Exception as exc:
            warnings.append(
                "Multiprocessing octree classification failed; falling back to sequential classification "
                f"for the remaining cells. Worker error: {type(exc).__name__}: {exc}"
            )
            worker_count = 1
            for index, work_item in enumerate(batch):
                classification = _classify_cell(
                    scene.objects,
                    triangle_indices,
                    np.asarray(work_item.center_mm, dtype=float),
                    np.asarray(work_item.size_mm, dtype=float),
                    contact_report,
                    params,
                    set(materials),
                    None,
                )
                handle_classified_cell(
                    work_item,
                    classification,
                    remaining_batch_items=len(batch) - index - 1,
                )
            while queue:
                work_item = queue.popleft()
                classification = _classify_cell(
                    scene.objects,
                    triangle_indices,
                    np.asarray(work_item.center_mm, dtype=float),
                    np.asarray(work_item.size_mm, dtype=float),
                    contact_report,
                    params,
                    set(materials),
                    None,
                )
                handle_classified_cell(work_item, classification, remaining_batch_items=0)
    if progress_callback is not None and diagnostics is not None:
        progress_callback(
            {
                "phase": "octree",
                "cells_tested": diagnostics.cells_tested,
                "cells_subdivided": diagnostics.cells_subdivided,
                "leaves": len(leaves),
                "queue": 0,
                "max_leaf_cells": params.max_leaf_cells,
                "max_depth_reached": diagnostics.max_depth_reached,
                "voxel_workers": worker_count,
                "done": True,
            }
        )
    return leaves


def _resolve_voxel_worker_count(params: OctreeParams) -> int:
    requested = int(getattr(params, "voxel_workers", 1))
    if requested == 0:
        cpu_count = os.cpu_count() or 2
        return max(1, min(2, cpu_count - 1))
    return max(1, requested)


def _prepare_worker_objects(objects: list[MeshObject]) -> list[MeshObject]:
    worker_objects: list[MeshObject] = []
    for obj in objects:
        triangles = np.asarray(_mesh_triangles(obj), dtype=float)
        bounds_min, bounds_max = obj.bounds_mm
        worker_objects.append(
            MeshObject(
                name=obj.name,
                material_name=obj.material_name,
                mesh=_TriangleMesh(triangles=triangles),
                vertices_mm=np.empty((0, 3), dtype=float),
                bounds_mm=(np.asarray(bounds_min, dtype=float), np.asarray(bounds_max, dtype=float)),
                watertight=bool(obj.watertight),
                scene_path=getattr(obj, "scene_path", None),
            )
        )
    return worker_objects


def _init_octree_worker(
    objects: list[MeshObject],
    contact_report: ContactReport,
    params: OctreeParams,
    known_materials: set[str],
) -> None:
    global _TRIMESH_CONTAINS_AVAILABLE
    global _TRIANGLE_CACHE
    global _TRIANGLE_INDEX_CACHE
    global _WORKER_OBJECTS
    global _WORKER_TRIANGLE_INDICES
    global _WORKER_CONTACT_REPORT
    global _WORKER_PARAMS
    global _WORKER_KNOWN_MATERIALS

    _TRIMESH_CONTAINS_AVAILABLE = False
    _TRIANGLE_CACHE = {}
    _TRIANGLE_INDEX_CACHE = {}
    _WORKER_OBJECTS = objects
    _WORKER_CONTACT_REPORT = contact_report
    _WORKER_PARAMS = params
    _WORKER_KNOWN_MATERIALS = set(known_materials)
    _WORKER_TRIANGLE_INDICES = _build_triangle_indices(_WORKER_OBJECTS, params)


def _classify_cell_work_item(work_item: _CellWorkItem) -> CellClassification:
    if _WORKER_PARAMS is None or _WORKER_CONTACT_REPORT is None:
        raise RuntimeError("Octree worker was not initialized.")
    return _classify_cell(
        _WORKER_OBJECTS,
        _WORKER_TRIANGLE_INDICES,
        np.asarray(work_item.center_mm, dtype=float),
        np.asarray(work_item.size_mm, dtype=float),
        _WORKER_CONTACT_REPORT,
        _WORKER_PARAMS,
        _WORKER_KNOWN_MATERIALS,
        None,
    )


def _record_leaf_diagnostics(
    diagnostics: OctreeDiagnostics,
    cell_id: str,
    size_mm: np.ndarray,
    level: int,
    classification: CellClassification,
) -> None:
    diagnostics.max_depth_reached = max(diagnostics.max_depth_reached, int(level))
    diagnostics.leaves_by_depth[level] = diagnostics.leaves_by_depth.get(level, 0) + 1
    size_key = "x".join(f"{float(value):.6g}" for value in size_mm)
    diagnostics.leaves_by_cell_size_mm[size_key] = diagnostics.leaves_by_cell_size_mm.get(size_key, 0) + 1
    if classification.occupied:
        diagnostics.cells_accepted_exact += 1
    else:
        diagnostics.cells_rejected_empty += 1
    if classification.surface_hit:
        diagnostics.cells_surface_hit += 1
    if classification.inside_hit:
        diagnostics.cells_inside_hit += 1
    if classification.near_surface_hit:
        diagnostics.cells_near_surface_hit += 1
    if classification.bbox_only_hit:
        diagnostics.cells_bbox_only_hit += 1
    if diagnostics.debug_leaves and classification.occupied:
        diagnostics.leaf_records.append(
            {
                "cell_id": cell_id,
                "acceptance_reason": classification.acceptance_reason,
                "source_meshes": sorted(classification.part_ids),
                "depth": int(level),
                "cell_size_mm": [float(value) for value in size_mm],
                "volume_fraction": classification.volume_fraction,
                "surface_hit": classification.surface_hit,
                "inside_hit": classification.inside_hit,
                "near_surface_hit": classification.near_surface_hit,
                "bbox_only_hit": classification.bbox_only_hit,
                "accepted_by_exact_geometry": classification.occupied,
                "accepted_by_bbox_fallback": False,
            }
        )


def _build_triangle_indices(
    objects: list[MeshObject], params: OctreeParams
) -> dict[int, TriangleSpatialIndex]:
    target_bucket_size = max(float(params.max_cell_size_mm), float(params.min_cell_size_mm), 1.0e-6)
    indices: dict[int, TriangleSpatialIndex] = {}
    for obj in objects:
        cache_key = id(obj.mesh)
        index = _TRIANGLE_INDEX_CACHE.get(cache_key)
        if index is None:
            index = TriangleSpatialIndex.from_mesh(obj, target_bucket_size)
            _TRIANGLE_INDEX_CACHE[cache_key] = index
        indices[id(obj)] = index
    return indices


def _classify_cell(
    objects: list[MeshObject],
    triangle_indices: dict[int, TriangleSpatialIndex],
    center_mm: np.ndarray,
    size_mm: np.ndarray,
    contact_report: ContactReport,
    params: OctreeParams,
    known_materials: set[str],
    diagnostics: OctreeDiagnostics | None = None,
) -> CellClassification:
    half = size_mm * 0.5
    cell_min = center_mm - half
    cell_max = center_mm + half
    near_margin = max(0.0, min(float(params.contact_refine_distance_mm), float(max(size_mm)) * 0.5))
    near_min = cell_min - near_margin
    near_max = cell_max + near_margin
    candidate_objects = _objects_intersecting_bounds(objects, near_min, near_max)
    candidate_mesh_ids = {id(obj) for obj in candidate_objects}
    surface_objects: list[MeshObject] = []
    near_surface_objects: list[MeshObject] = []
    warnings: list[str] = []
    triangle_candidate_tests = 0
    triangle_intersection_tests = 0

    for obj in candidate_objects:
        index = triangle_indices.get(id(obj))
        if index is None:
            continue
        near_candidates = index.query(near_min, near_max)
        triangle_candidate_tests += int(len(near_candidates))
        if near_candidates.size == 0:
            continue
        cell_candidates = index.query(cell_min, cell_max)
        triangle_candidate_tests += int(len(cell_candidates))
        surface_hit = False
        for triangle_index in cell_candidates:
            triangle_intersection_tests += 1
            if _triangle_intersects_aabb(index.triangles[int(triangle_index)], center_mm, half):
                surface_hit = True
                break
        if surface_hit:
            surface_objects.append(obj)
            continue
        near_surface_objects.append(obj)

    if diagnostics is not None:
        diagnostics.triangle_candidate_tests += triangle_candidate_tests
        diagnostics.triangle_intersection_tests += triangle_intersection_tests

    inside_counts: Counter[str] = Counter()
    material_counts: Counter[str] = Counter()
    points = _sample_points(center_mm, size_mm, params.samples_per_cell)
    watertight_candidates = [obj for obj in candidate_objects if bool(getattr(obj, "watertight", False))]
    for point in points:
        for obj in watertight_candidates:
            obj_min, obj_max = obj.bounds_mm
            if np.any(point < obj_min) or np.any(point > obj_max):
                continue
            try:
                if _mesh_contains_point(obj, point):
                    inside_counts[obj.name] += 1
                    material_counts[_physical_material_name(obj, contact_report, known_materials)] += 1
                    break
            except Exception:
                warnings.append(f"Inside/outside test failed for watertight mesh {obj.name}.")

    surface_mesh_ids = {id(obj) for obj in surface_objects}
    inside_mesh_ids = {id(obj) for obj in watertight_candidates if inside_counts.get(obj.name, 0) > 0}
    near_surface_mesh_ids = {id(obj) for obj in near_surface_objects}
    hit_objects = _unique_objects(surface_objects + [obj for obj in watertight_candidates if inside_counts.get(obj.name, 0) > 0])
    component_counts: Counter[str] = Counter()
    for obj in surface_objects:
        component_counts[obj.name] += max(1, int(math.ceil(float(params.samples_per_cell) * params.min_solid_fraction)))
        material_counts[_physical_material_name(obj, contact_report, known_materials)] += max(
            1, int(math.ceil(float(params.samples_per_cell) * params.min_solid_fraction))
        )
    component_counts.update(inside_counts)

    total = max(1, int(params.samples_per_cell))
    raw_occupancy = {name: min(1.0, count / float(total)) for name, count in component_counts.items()}
    occupancy = _normalize_fraction_map(raw_occupancy)
    material_fractions = _normalize_fraction_map(
        {name: min(1.0, count / float(total)) for name, count in material_counts.items()}
    )
    dominant_component = max(occupancy, key=occupancy.get) if occupancy else None
    dominant_material = max(material_fractions, key=material_fractions.get) if material_fractions else None
    occupied = bool(surface_objects or inside_counts)
    surface_hit = bool(surface_objects)
    inside_hit = bool(inside_counts)
    near_surface_hit = bool(near_surface_objects)
    bbox_only_hit = bool(candidate_objects and not occupied and not near_surface_hit)
    volume_fraction = max(occupancy.values(), default=0.0) if occupied else None
    if surface_hit and inside_hit:
        acceptance_reason = "surface_and_watertight_inside"
    elif surface_hit:
        acceptance_reason = "triangle_surface_intersection"
    elif inside_hit:
        acceptance_reason = "watertight_point_containment"
    elif near_surface_hit:
        acceptance_reason = "near_surface_empty"
    elif candidate_objects:
        acceptance_reason = "bbox_only_empty"
    else:
        acceptance_reason = "empty"
    if candidate_objects and not occupied:
        warnings.append("Candidate mesh AABB overlap did not produce triangle intersection or watertight containment.")
    return CellClassification(
        occupied=occupied,
        surface_hit=surface_hit,
        inside_hit=inside_hit,
        near_surface_hit=near_surface_hit,
        bbox_only_hit=bbox_only_hit,
        surface_mesh_ids=surface_mesh_ids,
        inside_mesh_ids=inside_mesh_ids,
        near_surface_mesh_ids=near_surface_mesh_ids,
        candidate_mesh_ids=candidate_mesh_ids,
        material_ids=set(material_fractions),
        part_ids={obj.name for obj in hit_objects},
        occupancy=occupancy,
        material_fractions=material_fractions,
        dominant_component=dominant_component,
        dominant_material=dominant_material,
        volume_fraction=volume_fraction,
        acceptance_reason=acceptance_reason,
        warnings=warnings[:5],
        triangle_candidate_tests=triangle_candidate_tests,
        triangle_intersection_tests=triangle_intersection_tests,
    )


def _normalize_fraction_map(values: dict[str, float]) -> dict[str, float]:
    clean = {name: float(value) for name, value in values.items() if value > 0.0}
    total = sum(clean.values())
    if total <= 1.0:
        return clean
    return {name: value / total for name, value in clean.items()}


def _unique_objects(objects: list[MeshObject]) -> list[MeshObject]:
    unique: list[MeshObject] = []
    seen: set[int] = set()
    for obj in objects:
        key = id(obj)
        if key in seen:
            continue
        seen.add(key)
        unique.append(obj)
    return unique


def _classification_confidence(classification: CellClassification, params: OctreeParams) -> str:
    if not classification.occupied:
        return "low" if classification.candidate_mesh_ids else "high"
    if classification.surface_hit and classification.inside_hit:
        return "high"
    if classification.inside_hit:
        return "high"
    if classification.surface_hit:
        fraction = classification.volume_fraction or 0.0
        return "medium" if fraction >= params.min_solid_fraction else "low"
    return "low"


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
    inferred_material = infer_material_name_from_text(obj.material_name, {name: None for name in known_materials})
    if inferred_material:
        return inferred_material
    inferred_material = infer_material_name_from_text(obj.name, {name: None for name in known_materials})
    if inferred_material:
        return inferred_material
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
        triangles = np.asarray(getattr(obj.mesh, "triangles", []), dtype=float)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
            triangles = np.empty((0, 3, 3), dtype=float)
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


def _objects_intersecting_bounds(
    objects: list[MeshObject], bounds_min: np.ndarray, bounds_max: np.ndarray
) -> list[MeshObject]:
    hits: list[MeshObject] = []
    for obj in objects:
        obj_min, obj_max = obj.bounds_mm
        if np.all(bounds_max >= obj_min) and np.all(obj_max >= bounds_min):
            hits.append(obj)
    return hits


def _sample_points(center_mm: np.ndarray, size_mm: np.ndarray, samples_per_cell: int) -> np.ndarray:
    count = max(1, int(samples_per_cell))
    half = np.asarray(size_mm, dtype=float) * 0.5
    if count == 1:
        return np.asarray([center_mm], dtype=float)
    grid_n = max(2, math.ceil(count ** (1.0 / 3.0)))
    offsets = np.linspace(-0.5, 0.5, grid_n + 2)[1:-1]
    points = [center_mm + half * np.array(offset, dtype=float) for offset in product(offsets, repeat=3)]
    points.sort(key=lambda point: (float(np.linalg.norm(point - center_mm)), point[0], point[1], point[2]))
    return np.asarray(points[:count], dtype=float)


def _triangle_intersects_aabb(triangle: np.ndarray, box_center: np.ndarray, box_half_size: np.ndarray) -> bool:
    tri = np.asarray(triangle, dtype=float) - box_center
    if tri.shape != (3, 3):
        return False
    eps = 1.0e-9
    tri_min = np.min(tri, axis=0)
    tri_max = np.max(tri, axis=0)
    if np.any(tri_min > box_half_size + eps) or np.any(tri_max < -box_half_size - eps):
        return False
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    if np.linalg.norm(normal) > eps and not _plane_intersects_aabb(normal, tri[0], box_half_size):
        return False
    axes = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    edges = [tri[1] - tri[0], tri[2] - tri[1], tri[0] - tri[2]]
    for edge in edges:
        for axis in axes:
            test_axis = np.cross(edge, axis)
            if np.dot(test_axis, test_axis) <= eps:
                continue
            projections = tri @ test_axis
            radius = np.dot(box_half_size, np.abs(test_axis))
            if float(np.min(projections)) > radius + eps or float(np.max(projections)) < -radius - eps:
                return False
    return True


def _plane_intersects_aabb(normal: np.ndarray, point: np.ndarray, half_size: np.ndarray) -> bool:
    radius = float(np.dot(half_size, np.abs(normal)))
    distance = float(np.dot(normal, point))
    return abs(distance) <= radius + 1.0e-9


def _bucket_key(center_mm: np.ndarray | tuple[float, float, float], bucket_size_mm: float) -> tuple[int, int, int]:
    center = np.asarray(center_mm, dtype=float)
    return tuple(np.floor(center / bucket_size_mm).astype(int))
