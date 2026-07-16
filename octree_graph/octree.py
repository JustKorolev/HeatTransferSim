"""Adaptive octree occupancy and subdivision for CAD assemblies."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import heapq
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
_TRIANGLE_CACHE: dict[int, tuple[object, np.ndarray]] = {}
_TRIANGLE_INDEX_CACHE: dict[int, "TriangleSpatialIndex"] = {}
_WORKER_OBJECTS: list[MeshObject] = []
_WORKER_TRIANGLE_INDICES: dict[int, "TriangleSpatialIndex"] = {}
_WORKER_CONTACT_REPORT: ContactReport | None = None
_WORKER_PARAMS: "OctreeParams | None" = None
_WORKER_KNOWN_MATERIALS: set[str] = set()
_TRIANGLE_QUERY_CHUNK_SIZE = 16384
_TRIANGLE_BUCKET_INSERT_LIMIT = 4096


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
    allow_max_cell_size_budget_overflow: bool = True
    samples_per_cell: int = 9
    min_solid_fraction: float = 0.12
    bbox_fallback: bool = False
    voxel_workers: int = 1
    voxel_batch_size: int = 64
    crowded_component_refine_count: int = 0
    crowded_component_refine_distance_mm: float = 0.0
    adaptive_refine_priority: bool = True
    multi_surface_refine_count: int = 2
    surface_complexity_refine_threshold: int = 64
    role_refine_component_names: tuple[str, ...] = field(default_factory=tuple)
    role_refine_distance_mm: float = 0.0
    role_refine_max_depth: int | None = None
    contains_backend: str = "ray"


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
    crowded_component_count: int
    role_component_count: int
    surface_component_count: int
    near_surface_component_count: int
    material_ids: set[str]
    part_ids: set[str]
    occupancy: dict[str, float]
    material_fractions: dict[str, float]
    dominant_component: str | None
    dominant_material: str | None
    volume_fraction: float | None
    acceptance_reason: str
    refinement_score: float = 0.0
    refinement_reasons: tuple[str, ...] = field(default_factory=tuple)
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
    cells_crowded_component_hit: int = 0
    cells_role_component_hit: int = 0
    cells_multi_surface_hit: int = 0
    cells_surface_complexity_hit: int = 0
    cells_refined_by_reason: dict[str, int] = field(default_factory=dict)
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
            "cells_crowded_component_hit": self.cells_crowded_component_hit,
            "cells_role_component_hit": self.cells_role_component_hit,
            "cells_multi_surface_hit": self.cells_multi_surface_hit,
            "cells_surface_complexity_hit": self.cells_surface_complexity_hit,
            "cells_refined_by_reason": dict(sorted(self.cells_refined_by_reason.items())),
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
    unbucketed_triangle_indices: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))

    @classmethod
    def from_mesh(cls, obj: MeshObject, target_bucket_size_mm: float) -> "TriangleSpatialIndex":
        triangles = _mesh_triangles(obj)
        if triangles.size == 0:
            empty = np.empty((0, 3), dtype=float)
            return cls(triangles, empty, empty, max(float(target_bucket_size_mm), 1.0), {})
        bounds_min, bounds_max = _triangle_bounds(triangles)
        object_bounds = _object_bounds_tuple(obj) or _bounds_from_triangle_bounds(bounds_min, bounds_max)
        extent = _bounds_extent_mm(object_bounds)
        try:
            target_bucket_size = float(target_bucket_size_mm)
        except (TypeError, ValueError):
            target_bucket_size = 1.0
        if not math.isfinite(target_bucket_size) or target_bucket_size <= 0.0:
            target_bucket_size = 1.0
        bucket_size = max(target_bucket_size, extent / 64.0, 1.0e-6)
        buckets: dict[tuple[int, int, int], list[int]] = {}
        unbucketed: list[int] = []
        for index, (tri_min, tri_max) in enumerate(zip(bounds_min, bounds_max)):
            min_key = _bucket_key(tri_min, bucket_size)
            max_key = _bucket_key(tri_max, bucket_size)
            span_x = max_key[0] - min_key[0] + 1
            span_y = max_key[1] - min_key[1] + 1
            span_z = max_key[2] - min_key[2] + 1
            bucket_insert_count = int(span_x * span_y * span_z)
            if (
                span_x <= 0
                or span_y <= 0
                or span_z <= 0
                or bucket_insert_count > _TRIANGLE_BUCKET_INSERT_LIMIT
            ):
                unbucketed.append(index)
                continue
            for ix in range(min_key[0], max_key[0] + 1):
                for iy in range(min_key[1], max_key[1] + 1):
                    for iz in range(min_key[2], max_key[2] + 1):
                        buckets.setdefault((ix, iy, iz), []).append(index)
        return cls(
            triangles,
            bounds_min,
            bounds_max,
            bucket_size,
            buckets,
            np.asarray(unbucketed, dtype=int),
        )

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
            if self.unbucketed_triangle_indices.size == 0:
                return np.empty((0,), dtype=int)
            candidates = self.unbucketed_triangle_indices.astype(int, copy=True)
        else:
            candidates = np.fromiter(sorted(matches), dtype=int)
            if self.unbucketed_triangle_indices.size:
                candidates = np.concatenate((candidates, self.unbucketed_triangle_indices)).astype(int, copy=False)
        return self._filter_candidates_by_bounds(candidates, cell_min, cell_max)

    def _query_all_bounds(self, cell_min: np.ndarray, cell_max: np.ndarray) -> np.ndarray:
        all_indices = np.arange(self.bounds_min.shape[0], dtype=int)
        return self._filter_candidates_by_bounds(all_indices, cell_min, cell_max)

    def _filter_candidates_by_bounds(
        self, candidates: np.ndarray, cell_min: np.ndarray, cell_max: np.ndarray
    ) -> np.ndarray:
        if candidates.size == 0:
            return np.empty((0,), dtype=int)
        cell_min = np.asarray(cell_min, dtype=float)
        cell_max = np.asarray(cell_max, dtype=float)
        if candidates.size <= _TRIANGLE_QUERY_CHUNK_SIZE:
            return self._filter_candidate_chunk(candidates, cell_min, cell_max)
        matches: list[np.ndarray] = []
        for start in range(0, int(candidates.size), _TRIANGLE_QUERY_CHUNK_SIZE):
            chunk = candidates[start : start + _TRIANGLE_QUERY_CHUNK_SIZE]
            filtered = self._filter_candidate_chunk(chunk, cell_min, cell_max)
            if filtered.size:
                matches.append(filtered)
        if not matches:
            return np.empty((0,), dtype=int)
        return np.concatenate(matches).astype(int, copy=False)

    def _filter_candidate_chunk(
        self, candidates: np.ndarray, cell_min: np.ndarray, cell_max: np.ndarray
    ) -> np.ndarray:
        mins = self.bounds_min[candidates]
        maxs = self.bounds_max[candidates]
        overlap = (
            (maxs[:, 0] >= cell_min[0])
            & (mins[:, 0] <= cell_max[0])
            & (maxs[:, 1] >= cell_min[1])
            & (mins[:, 1] <= cell_max[1])
            & (maxs[:, 2] >= cell_min[2])
            & (mins[:, 2] <= cell_max[2])
        )
        return candidates[overlap]


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
    span = maxs - mins
    side = float(max(float(span[0]), float(span[1]), float(span[2])))
    root = (center, np.array([side, side, side], dtype=float))
    leaves: list[OctreeCell] = []
    triangle_indices = _build_triangle_indices(scene.objects, params)

    if diagnostics is not None:
        diagnostics.root_bounds_mm = {"min": mins.astype(float).tolist(), "max": maxs.astype(float).tolist()}
        diagnostics.root_cell_size_mm = root[1].astype(float).tolist()

    queue: list[tuple[float, int, _CellWorkItem]] = []
    deferred_discretionary: list[tuple[_CellWorkItem, CellClassification]] = []
    counter = 1
    push_counter = 0
    queued_max_size_candidates = 0

    def push_cell(work_item: _CellWorkItem, priority: float = 0.0) -> None:
        nonlocal push_counter, queued_max_size_candidates
        heap_priority = -float(priority) if bool(getattr(params, "adaptive_refine_priority", True)) else 0.0
        heapq.heappush(queue, (heap_priority, push_counter, work_item))
        push_counter += 1
        if _is_max_size_candidate(work_item, params):
            queued_max_size_candidates += 1

    def pop_cell() -> _CellWorkItem:
        nonlocal queued_max_size_candidates
        work_item = heapq.heappop(queue)[2]
        if _is_max_size_candidate(work_item, params):
            queued_max_size_candidates = max(0, queued_max_size_candidates - 1)
        return work_item

    max_cell_budget_warning_emitted = False

    push_cell(
        _CellWorkItem("cell_0", tuple(float(v) for v in root[0]), tuple(float(v) for v in root[1]), 0, None),
        priority=0.0,
    )

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
        remaining_mandatory_candidates: int = 0,
        diagnostics_already_counted: bool = False,
    ) -> None:
        nonlocal counter, max_cell_budget_warning_emitted
        center_mm = np.asarray(work_item.center_mm, dtype=float)
        size_mm = np.asarray(work_item.size_mm, dtype=float)
        max_size_mm = float(max(size_mm))
        level = int(work_item.level)
        if diagnostics is not None and not diagnostics_already_counted:
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
                    "max_size_queue": queued_max_size_candidates + int(remaining_mandatory_candidates),
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
        crowded_component_refinement = _needs_crowded_component_refinement(classification, params)
        role_component_refinement = _needs_role_component_refinement(classification, params, level)
        multi_surface_refinement = _needs_multi_surface_refinement(classification, params)
        surface_complexity_refinement = _needs_surface_complexity_refinement(classification, params)
        gap_preservation_refinement = _needs_gap_preservation_refinement(classification, params)
        if diagnostics is not None:
            if multi_surface_refinement:
                diagnostics.cells_multi_surface_hit += 1
            if surface_complexity_refinement:
                diagnostics.cells_surface_complexity_hit += 1
        refinement_score, refinement_reasons = _refinement_priority(
            classification,
            params,
            size_mm,
            mixed_parts=mixed_parts,
            mixed_materials=mixed_materials,
            high_contrast=high_contrast,
            crowded_component_refinement=crowded_component_refinement,
            role_component_refinement=role_component_refinement,
            multi_surface_refinement=multi_surface_refinement,
            surface_complexity_refinement=surface_complexity_refinement,
            gap_preservation_refinement=gap_preservation_refinement,
            needs_surface_refinement=needs_surface_refinement,
        )
        classification.refinement_score = refinement_score
        classification.refinement_reasons = tuple(refinement_reasons)
        pending_max_size_candidates = queued_max_size_candidates + int(remaining_mandatory_candidates)
        effective_queue_len = len(queue) + int(remaining_batch_items)
        budget_allows_children = (
            params.max_leaf_cells is None
            or len(leaves) + effective_queue_len + 8 <= params.max_leaf_cells
        )
        if diagnostics is not None and params.max_leaf_cells is not None and not budget_allows_children:
            diagnostics.max_leaf_cells_reached = True
        above_max_cell_size = classification.occupied and max_size_mm > params.max_cell_size_mm
        can_subdivide = level < params.max_depth and max_size_mm > params.min_cell_size_mm
        discretionary_refinement = (
            mixed_parts
            or mixed_materials
            or high_contrast
            or crowded_component_refinement
            or role_component_refinement
            or multi_surface_refinement
            or surface_complexity_refinement
            or gap_preservation_refinement
            or (needs_surface_refinement and max_size_mm > params.min_cell_size_mm)
            or (
                classification.occupied
                and 0.0 < dominant_fraction < params.dominant_fraction_accept
                and max_size_mm > params.min_cell_size_mm
            )
        )
        discretionary_budget_allows_children = budget_allows_children and pending_max_size_candidates <= 0
        if (
            can_subdivide
            and not above_max_cell_size
            and discretionary_refinement
            and budget_allows_children
            and pending_max_size_candidates > 0
        ):
            deferred_discretionary.append((work_item, classification))
            return
        should_subdivide = can_subdivide and (
            above_max_cell_size or (discretionary_budget_allows_children and discretionary_refinement)
        )
        if should_subdivide:
            if above_max_cell_size and not budget_allows_children:
                if not bool(getattr(params, "allow_max_cell_size_budget_overflow", True)):
                    active_cells = len(leaves) + effective_queue_len
                    raise RuntimeError(
                        "max_leaf_cells is too low to satisfy max_cell_size_mm for occupied cells. "
                        f"Refusing to enqueue 8 children for {work_item.cell_id} at level {level}: "
                        f"active_or_leaf_cells={active_cells}, max_leaf_cells={params.max_leaf_cells}, "
                        f"cell_size_mm={max_size_mm:.6g}, max_cell_size_mm={float(params.max_cell_size_mm):.6g}. "
                        "Increase --max-leaf-cells, increase --max-cell-size-mm, or pass "
                        "--allow-max-cell-size-budget-overflow to permit mandatory max-size refinement to exceed the cap."
                    )
                if not max_cell_budget_warning_emitted:
                    warnings.append(
                        "Exceeded max_leaf_cells to honor max_cell_size_mm for occupied cells; "
                        "increase --max-leaf-cells if this graph is larger than expected."
                    )
                    max_cell_budget_warning_emitted = True
            if diagnostics is not None:
                diagnostics.cells_subdivided += 1
                for reason in refinement_reasons:
                    diagnostics.cells_refined_by_reason[reason] = diagnostics.cells_refined_by_reason.get(reason, 0) + 1
            quarter = size_mm * 0.25
            child_size = size_mm * 0.5
            for signs in product((-1.0, 1.0), repeat=3):
                child_id = f"cell_{counter}"
                counter += 1
                push_cell(
                    _CellWorkItem(
                        child_id,
                        tuple(float(v) for v in center_mm + quarter * np.array(signs)),
                        tuple(float(v) for v in child_size),
                        level + 1,
                        work_item.cell_id,
                    ),
                    priority=refinement_score + 0.01 * float(level + 1),
                )
            return

        confidence = _classification_confidence(classification, params)
        if above_max_cell_size:
            classification.warnings.append(
                "Accepted occupied cell above max_cell_size_mm because refinement was blocked "
                "by max_depth or min_cell_size_mm."
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
        while queue or deferred_discretionary:
            if deferred_discretionary and queued_max_size_candidates <= 0:
                work_item, classification = deferred_discretionary.pop()
                handle_classified_cell(
                    work_item,
                    classification,
                    remaining_batch_items=len(queue) + len(deferred_discretionary),
                    diagnostics_already_counted=True,
                )
                continue
            if not queue:
                break
            work_item = pop_cell()
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
                while queue or deferred_discretionary:
                    if deferred_discretionary and queued_max_size_candidates <= 0:
                        work_item, classification = deferred_discretionary.pop()
                        handle_classified_cell(
                            work_item,
                            classification,
                            remaining_batch_items=len(queue) + len(deferred_discretionary),
                            diagnostics_already_counted=True,
                        )
                        continue
                    if not queue:
                        break
                    batch: list[_CellWorkItem] = []
                    while queue and len(batch) < batch_size:
                        batch.append(pop_cell())
                    remaining_mandatory_by_index = _remaining_max_size_candidates_by_index(batch, params)
                    classifications = list(
                        executor.map(
                            _classify_cell_work_item,
                            batch,
                            chunksize=max(1, min(8, batch_size // max(worker_count, 1))),
                        )
                    )
                    for index, (work_item, classification) in enumerate(zip(batch, classifications)):
                        remaining_batch = batch[index + 1 :]
                        handle_classified_cell(
                            work_item,
                            classification,
                            remaining_batch_items=len(remaining_batch),
                            remaining_mandatory_candidates=remaining_mandatory_by_index[index],
                        )
        except Exception as exc:
            warnings.append(
                "Multiprocessing octree classification failed; falling back to sequential classification "
                f"for the remaining cells. Worker error: {type(exc).__name__}: {exc}"
            )
            worker_count = 1
            remaining_mandatory_by_index = _remaining_max_size_candidates_by_index(batch, params)
            for index, work_item in enumerate(batch):
                remaining_batch = batch[index + 1 :]
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
                    remaining_batch_items=len(remaining_batch),
                    remaining_mandatory_candidates=remaining_mandatory_by_index[index],
                )
            while queue or deferred_discretionary:
                if deferred_discretionary and queued_max_size_candidates <= 0:
                    work_item, classification = deferred_discretionary.pop()
                    handle_classified_cell(
                        work_item,
                        classification,
                        remaining_batch_items=len(queue) + len(deferred_discretionary),
                        diagnostics_already_counted=True,
                    )
                    continue
                if not queue:
                    break
                work_item = pop_cell()
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
                "max_size_queue": 0,
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


def _remaining_max_size_candidates_by_index(
    batch: list[_CellWorkItem],
    params: OctreeParams,
) -> list[int]:
    remaining_by_index = [0] * len(batch)
    remaining = 0
    for index in range(len(batch) - 1, -1, -1):
        remaining_by_index[index] = remaining
        if _is_max_size_candidate(batch[index], params):
            remaining += 1
    return remaining_by_index


def _is_max_size_candidate(item: _CellWorkItem, params: OctreeParams) -> bool:
    size_mm = np.asarray(item.size_mm, dtype=float)
    if size_mm.size == 0:
        return False
    return (
        int(item.level) < int(params.max_depth)
        and float(np.max(size_mm)) > float(params.max_cell_size_mm)
        and float(np.max(size_mm)) > float(params.min_cell_size_mm)
    )


def _prepare_worker_objects(objects: list[MeshObject]) -> list[MeshObject]:
    worker_objects: list[MeshObject] = []
    for obj in objects:
        triangles = np.array(_mesh_triangles(obj), dtype=float, copy=True)
        bounds = _object_bounds_tuple(obj)
        if bounds is None:
            continue
        bounds_min, bounds_max = bounds
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
    if int(classification.crowded_component_count) > 0:
        diagnostics.cells_crowded_component_hit += 1
    if int(classification.role_component_count) > 0:
        diagnostics.cells_role_component_hit += 1
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
                "crowded_component_count": int(classification.crowded_component_count),
                "role_component_count": int(classification.role_component_count),
                "surface_component_count": int(classification.surface_component_count),
                "near_surface_component_count": int(classification.near_surface_component_count),
                "refinement_score": float(classification.refinement_score),
                "refinement_reasons": list(classification.refinement_reasons),
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
    crowded_margin = max(0.0, float(params.crowded_component_refine_distance_mm))
    crowded_objects = (
        _objects_intersecting_bounds(objects, cell_min - crowded_margin, cell_max + crowded_margin)
        if int(params.crowded_component_refine_count) > 0
        else []
    )
    role_refine_names = set(getattr(params, "role_refine_component_names", ()) or ())
    role_refine_margin = max(0.0, float(getattr(params, "role_refine_distance_mm", 0.0)))
    role_refine_objects = (
        [
            obj
            for obj in _objects_intersecting_bounds(
                objects,
                cell_min - role_refine_margin,
                cell_max + role_refine_margin,
            )
            if obj.name in role_refine_names
        ]
        if role_refine_names
        else []
    )
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
            if not _point_in_object_bounds(point, obj):
                continue
            try:
                if _mesh_contains_point(obj, point, params):
                    inside_counts[obj.name] += 1
                    material_counts[_physical_material_name(obj, contact_report, known_materials)] += 1
                    break
            except Exception:
                warnings.append(f"Inside/outside test failed for watertight mesh {obj.name}.")

    surface_mesh_ids = {id(obj) for obj in surface_objects}
    inside_mesh_ids = {id(obj) for obj in watertight_candidates if inside_counts.get(obj.name, 0) > 0}
    near_surface_mesh_ids = {id(obj) for obj in near_surface_objects}
    surface_component_count = len({obj.name for obj in surface_objects})
    near_surface_component_count = len({obj.name for obj in surface_objects + near_surface_objects})
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
        crowded_component_count=len({id(obj) for obj in crowded_objects}),
        role_component_count=len({id(obj) for obj in role_refine_objects}),
        surface_component_count=surface_component_count,
        near_surface_component_count=near_surface_component_count,
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


def _needs_crowded_component_refinement(
    classification: CellClassification,
    params: OctreeParams,
) -> bool:
    threshold = int(params.crowded_component_refine_count)
    return threshold > 0 and int(classification.crowded_component_count) >= threshold


def _needs_multi_surface_refinement(
    classification: CellClassification,
    params: OctreeParams,
) -> bool:
    threshold = int(getattr(params, "multi_surface_refine_count", 0))
    return threshold > 0 and int(classification.near_surface_component_count) >= threshold


def _needs_surface_complexity_refinement(
    classification: CellClassification,
    params: OctreeParams,
) -> bool:
    threshold = int(getattr(params, "surface_complexity_refine_threshold", 0))
    return threshold > 0 and int(classification.triangle_candidate_tests) >= threshold


def _needs_gap_preservation_refinement(
    classification: CellClassification,
    params: OctreeParams,
) -> bool:
    if not bool(getattr(params, "boundary_refine", True)):
        return False
    if int(classification.near_surface_component_count) < 2:
        return False
    if not (classification.surface_hit or classification.near_surface_hit or classification.bbox_only_hit):
        return False
    fill_fraction = float(classification.volume_fraction or 0.0)
    low_fill_threshold = max(float(params.min_solid_fraction), float(params.minority_fraction_ignore))
    return (not classification.inside_hit) or fill_fraction <= low_fill_threshold


def _refinement_priority(
    classification: CellClassification,
    params: OctreeParams,
    size_mm: np.ndarray,
    *,
    mixed_parts: bool,
    mixed_materials: bool,
    high_contrast: bool,
    crowded_component_refinement: bool,
    role_component_refinement: bool,
    multi_surface_refinement: bool,
    surface_complexity_refinement: bool,
    gap_preservation_refinement: bool,
    needs_surface_refinement: bool,
) -> tuple[float, tuple[str, ...]]:
    score = 0.0
    reasons: list[str] = []

    def add(reason: str, value: float) -> None:
        nonlocal score
        score += float(value)
        reasons.append(reason)

    if role_component_refinement:
        add("role_region", 120.0)
    if gap_preservation_refinement:
        add("gap_preservation", 7500.0 + 250.0 * float(classification.near_surface_component_count))
    if multi_surface_refinement:
        add("multi_surface_ambiguity", 90.0 + 5.0 * float(classification.near_surface_component_count))
    if surface_complexity_refinement:
        add("surface_complexity", min(80.0, 10.0 + 0.25 * float(classification.triangle_candidate_tests)))
    if crowded_component_refinement:
        add("crowded_component_bounds", 30.0 + 2.0 * float(classification.crowded_component_count))
    if mixed_parts:
        add("mixed_parts", 45.0)
    if mixed_materials:
        add("mixed_materials", 35.0)
    if high_contrast:
        add("material_contrast", 25.0)
    if needs_surface_refinement:
        add("surface_or_near_surface", 20.0)
    if classification.occupied and float(max(size_mm)) > float(params.max_cell_size_mm):
        target = max(float(params.max_cell_size_mm), 1.0e-9)
        oversize_ratio = float(max(size_mm)) / target
        add("above_max_cell_size", 10000.0 + 100.0 * oversize_ratio)
    if classification.bbox_only_hit:
        add("bbox_only_candidate", 5.0)
    if classification.inside_hit and not classification.surface_hit and classification.near_surface_component_count <= 1:
        score -= 15.0
        reasons.append("simple_inside_deprioritized")
    if not reasons:
        reasons.append("default")
    return max(0.0, score), tuple(reasons)


def _needs_role_component_refinement(
    classification: CellClassification,
    params: OctreeParams,
    level: int,
) -> bool:
    if int(classification.role_component_count) <= 0:
        return False
    max_depth = getattr(params, "role_refine_max_depth", None)
    if max_depth is None:
        return True
    return int(level) < int(max_depth)


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


def _mesh_contains_point(obj: MeshObject, point: np.ndarray, params: OctreeParams) -> bool:
    global _TRIMESH_CONTAINS_AVAILABLE
    if str(getattr(params, "contains_backend", "ray")).lower() == "ray":
        return _ray_contains_point(obj, point)
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
    cached = _TRIANGLE_CACHE.get(cache_key)
    if cached is not None and cached[0] is obj.mesh:
        return cached[1]
    try:
        triangles = np.array(getattr(obj.mesh, "triangles", []), dtype=float, copy=True)
    except Exception:
        triangles = np.empty((0, 3, 3), dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        triangles = np.empty((0, 3, 3), dtype=float)
    elif triangles.size:
        finite_values = np.isfinite(triangles)
        finite = (
            finite_values[:, 0, 0]
            & finite_values[:, 0, 1]
            & finite_values[:, 0, 2]
            & finite_values[:, 1, 0]
            & finite_values[:, 1, 1]
            & finite_values[:, 1, 2]
            & finite_values[:, 2, 0]
            & finite_values[:, 2, 1]
            & finite_values[:, 2, 2]
        )
        if np.nonzero(~finite)[0].size:
            triangles = triangles[finite]
        triangles = np.ascontiguousarray(triangles, dtype=float)
    _TRIANGLE_CACHE[cache_key] = (obj.mesh, triangles)
    return triangles


def _triangle_bounds(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if triangles.size == 0:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty
    first = triangles[:, 0, :]
    second = triangles[:, 1, :]
    third = triangles[:, 2, :]
    bounds_min = np.minimum(np.minimum(first, second), third)
    bounds_max = np.maximum(np.maximum(first, second), third)
    return (
        np.ascontiguousarray(bounds_min, dtype=float),
        np.ascontiguousarray(bounds_max, dtype=float),
    )


def _bounds_from_triangle_bounds(
    bounds_min: np.ndarray, bounds_max: np.ndarray
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if bounds_min.size == 0 or bounds_max.size == 0:
        return None
    min_x = min_y = min_z = math.inf
    max_x = max_y = max_z = -math.inf
    for tri_min, tri_max in zip(bounds_min, bounds_max):
        try:
            tri_min_x, tri_min_y, tri_min_z = (float(value) for value in tri_min)
            tri_max_x, tri_max_y, tri_max_z = (float(value) for value in tri_max)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (tri_min_x, tri_min_y, tri_min_z, tri_max_x, tri_max_y, tri_max_z)):
            continue
        min_x = min(min_x, tri_min_x)
        min_y = min(min_y, tri_min_y)
        min_z = min(min_z, tri_min_z)
        max_x = max(max_x, tri_max_x)
        max_y = max(max_y, tri_max_y)
        max_z = max(max_z, tri_max_z)
    if not all(math.isfinite(value) for value in (min_x, min_y, min_z, max_x, max_y, max_z)):
        return None
    return (min_x, min_y, min_z), (max_x, max_y, max_z)


def _bounds_extent_mm(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None
) -> float:
    if bounds is None:
        return 0.0
    bounds_min, bounds_max = bounds
    try:
        span_x = max(0.0, float(bounds_max[0]) - float(bounds_min[0]))
        span_y = max(0.0, float(bounds_max[1]) - float(bounds_min[1]))
        span_z = max(0.0, float(bounds_max[2]) - float(bounds_min[2]))
    except (TypeError, ValueError, IndexError):
        return 0.0
    extent = max(span_x, span_y, span_z)
    return extent if math.isfinite(extent) else 0.0


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
    if not bool(mask.nonzero()[0].size):
        return False
    f = np.zeros_like(a)
    f[mask] = 1.0 / a[mask]
    s = point - v0
    u = f * np.einsum("ij,ij->i", s, h)
    mask &= (u >= -eps) & (u <= 1.0 + eps)
    if not bool(mask.nonzero()[0].size):
        return False
    q = np.cross(s, edge1)
    v = f * np.einsum("ij,j->i", q, direction)
    mask &= (v >= -eps) & ((u + v) <= 1.0 + eps)
    if not bool(mask.nonzero()[0].size):
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
    try:
        query_min = tuple(float(value) for value in bounds_min)
        query_max = tuple(float(value) for value in bounds_max)
    except (TypeError, ValueError):
        return []
    if len(query_min) != 3 or len(query_max) != 3:
        return []
    if not all(math.isfinite(value) for value in (*query_min, *query_max)):
        return []
    raw_query_min = query_min
    raw_query_max = query_max
    query_min = tuple(min(left, right) for left, right in zip(raw_query_min, raw_query_max))
    query_max = tuple(max(left, right) for left, right in zip(raw_query_min, raw_query_max))
    hits: list[MeshObject] = []
    for obj in objects:
        bounds = _object_bounds_tuple(obj)
        if bounds is None:
            continue
        obj_min, obj_max = bounds
        if _bounds_intersect_tuple(obj_min, obj_max, query_min, query_max):
            hits.append(obj)
    return hits


def _point_in_object_bounds(point: np.ndarray, obj: MeshObject) -> bool:
    bounds = _object_bounds_tuple(obj)
    if bounds is None:
        return False
    obj_min, obj_max = bounds
    x, y, z = (float(value) for value in point)
    return (
        obj_min[0] <= x <= obj_max[0]
        and obj_min[1] <= y <= obj_max[1]
        and obj_min[2] <= z <= obj_max[2]
    )


def _object_bounds_tuple(obj: MeshObject) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    cached = getattr(obj, "_bounds_tuple_mm", None)
    if cached is not None:
        return cached
    try:
        raw_min, raw_max = obj.bounds_mm
        obj_min = tuple(float(value) for value in raw_min)
        obj_max = tuple(float(value) for value in raw_max)
    except Exception:
        return None
    if len(obj_min) != 3 or len(obj_max) != 3:
        return None
    if not all(math.isfinite(value) for value in (*obj_min, *obj_max)):
        return None
    cached = (
        tuple(min(left, right) for left, right in zip(obj_min, obj_max)),
        tuple(max(left, right) for left, right in zip(obj_min, obj_max)),
    )
    try:
        setattr(obj, "_bounds_tuple_mm", cached)
    except Exception:
        pass
    return cached


def _bounds_intersect_tuple(
    a_min: tuple[float, float, float],
    a_max: tuple[float, float, float],
    b_min: tuple[float, float, float],
    b_max: tuple[float, float, float],
) -> bool:
    return (
        b_max[0] >= a_min[0]
        and a_max[0] >= b_min[0]
        and b_max[1] >= a_min[1]
        and a_max[1] >= b_min[1]
        and b_max[2] >= a_min[2]
        and a_max[2] >= b_min[2]
    )


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
    tri_min = np.minimum(np.minimum(tri[0], tri[1]), tri[2])
    tri_max = np.maximum(np.maximum(tri[0], tri[1]), tri[2])
    upper_miss = tri_min > box_half_size + eps
    lower_miss = tri_max < -box_half_size - eps
    if bool(upper_miss[0] or upper_miss[1] or upper_miss[2] or lower_miss[0] or lower_miss[1] or lower_miss[2]):
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
            p_min = min(float(projections[0]), float(projections[1]), float(projections[2]))
            p_max = max(float(projections[0]), float(projections[1]), float(projections[2]))
            if p_min > radius + eps or p_max < -radius - eps:
                return False
    return True


def _plane_intersects_aabb(normal: np.ndarray, point: np.ndarray, half_size: np.ndarray) -> bool:
    radius = float(np.dot(half_size, np.abs(normal)))
    distance = float(np.dot(normal, point))
    return abs(distance) <= radius + 1.0e-9


def _bucket_key(center_mm: np.ndarray | tuple[float, float, float], bucket_size_mm: float) -> tuple[int, int, int]:
    center = np.asarray(center_mm, dtype=float)
    return tuple(np.floor(center / bucket_size_mm).astype(int))
