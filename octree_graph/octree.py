"""Adaptive octree occupancy and subdivision for CAD assemblies."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import heapq
from itertools import product
import math
import os
from typing import Any, Callable

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
_WORKER_OBJECT_BOUNDS_INDEX: "ObjectBoundsIndex | None" = None
_WORKER_CONTACT_REPORT: ContactReport | None = None
_WORKER_PARAMS: "OctreeParams | None" = None
_WORKER_KNOWN_MATERIALS: set[str] = set()
_TRIANGLE_QUERY_CHUNK_SIZE = 16384
_TRIANGLE_BUCKET_INSERT_LIMIT = 4096
_EMBREE_ACTIVE: bool | None = None


def embree_active() -> bool:
    """True when trimesh's ray engine is the embreex (pyembree) backend.

    When active, point-in-solid containment is BVH-accelerated; when not,
    trimesh/our code fall back to a pure-Python ray test that is ~60x slower and
    makes full builds impractical. Result is cached (the binding cannot change
    mid-process)."""
    global _EMBREE_ACTIVE
    if _EMBREE_ACTIVE is None:
        try:
            import trimesh

            _EMBREE_ACTIVE = "pyembree" in type(trimesh.creation.box().ray).__module__
        except Exception:
            _EMBREE_ACTIVE = False
    return bool(_EMBREE_ACTIVE)


@dataclass
class OctreeParams:
    min_cell_size_mm: float = 5.0
    max_cell_size_mm: float = 50.0
    max_depth: int = 8
    dominant_fraction_accept: float = 0.95
    minority_fraction_ignore: float = 0.02
    material_contrast_refine_threshold: float = 5.0
    # Refine cells whose dominant material has conductivity BELOW this (W/mK) down
    # to min_cell_size, so low-k parts (G-10 / insulators, where steep thermal
    # gradients live) get fine cells while high-k metal bulk stays coarse. 0 = off.
    low_conductivity_refine_threshold: float = 0.0
    contact_refine_distance_mm: float = 10.0
    boundary_refine: bool = True
    max_leaf_cells: int | None = None
    samples_per_cell: int = 9
    min_solid_fraction: float = 0.12
    bbox_fallback: bool = False
    voxel_workers: int = 1
    voxel_batch_size: int = 64
    crowded_component_refine_count: int = 0
    crowded_component_refine_distance_mm: float = 0.0
    crowded_component_refine_neighbor_cells: float = 1.0
    adaptive_refine_priority: bool = True
    multi_surface_refine_count: int = 2
    surface_complexity_refine_threshold: int = 64
    role_refine_component_names: tuple[str, ...] = field(default_factory=tuple)
    role_refine_distance_mm: float = 0.0
    role_refine_max_depth: int | None = None
    contains_backend: str = "trimesh"  # BVH-accelerated; falls back to "ray" if deps absent
    balance_adjacent_leaf_sizes: bool = True
    max_adjacent_leaf_size_ratio: float = 4.0
    balance_refine_passes: int = 2


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
    # Interior-fill inheritance: when a parent cell is a homogeneous interior of a
    # single solid (no surface crosses it), its children are provably interior too,
    # so they skip containment testing and inherit this component/material directly.
    inherited_component: str | None = None
    inherited_material: str | None = None


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
    scene: "GltfScene | None",
    contact_report: ContactReport | None,
    materials: dict[str, Material],
    params: OctreeParams,
    warnings: list[str],
    diagnostics: OctreeDiagnostics | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    geometry_provider: "GeometryProvider | None" = None,
) -> list[OctreeCell]:
    contact_report = contact_report or ContactReport()
    known_materials = set(materials)
    # Geometry backend: default is the triangle (GLB) path; a geometry_provider
    # (e.g. the STEP B-rep backend) supplies bounds + per-cell classification and
    # forces serial classification (its handles aren't picklable to workers).
    if geometry_provider is not None:
        mins, maxs = geometry_provider.bounds_mm
    else:
        mins, maxs = scene.bounds_mm
    center = (mins + maxs) * 0.5
    span = maxs - mins
    side = float(max(float(span[0]), float(span[1]), float(span[2])))
    root = (center, np.array([side, side, side], dtype=float))
    leaves: list[OctreeCell] = []
    triangle_indices = (
        {} if geometry_provider is not None else _build_triangle_indices(scene.objects, params)
    )
    object_bounds_index = (
        None if geometry_provider is not None else ObjectBoundsIndex.build(scene.objects)
    )

    def classify(center_arg: Any, size_arg: Any) -> CellClassification:
        center_np = np.asarray(center_arg, dtype=float)
        size_np = np.asarray(size_arg, dtype=float)
        if geometry_provider is not None:
            return geometry_provider.classify_cell(center_np, size_np)
        return _classify_cell(
            scene.objects,
            triangle_indices,
            center_np,
            size_np,
            contact_report,
            params,
            known_materials,
            None,
            object_bounds_index,
        )

    def classify_work_item(work_item: _CellWorkItem) -> CellClassification:
        # Interior-fill children carry their component/material and need no
        # containment test; everyone else is classified normally.
        if work_item.inherited_component is not None:
            return _interior_fill_classification(
                work_item.inherited_component, work_item.inherited_material
            )
        return classify(work_item.center_mm, work_item.size_mm)

    if diagnostics is not None:
        diagnostics.root_bounds_mm = {"min": mins.astype(float).tolist(), "max": maxs.astype(float).tolist()}
        diagnostics.root_cell_size_mm = root[1].astype(float).tolist()

    queue: list[tuple[float, int, _CellWorkItem]] = []
    counter = 1
    push_counter = 0

    def push_cell(work_item: _CellWorkItem, priority: float = 0.0) -> None:
        nonlocal push_counter
        heap_priority = -float(priority) if bool(getattr(params, "adaptive_refine_priority", True)) else 0.0
        heapq.heappush(queue, (heap_priority, push_counter, work_item))
        push_counter += 1

    def pop_cell() -> _CellWorkItem:
        return heapq.heappop(queue)[2]

    max_size_limit_override_warning_emitted = False
    max_leaf_budget_override_warning_emitted = False

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

    worker_count = 1 if geometry_provider is not None else _resolve_voxel_worker_count(params)
    batch_size = max(1, int(params.voxel_batch_size))

    def handle_classified_cell(
        work_item: _CellWorkItem,
        classification: CellClassification,
        remaining_batch_items: int,
    ) -> None:
        nonlocal counter, max_size_limit_override_warning_emitted, max_leaf_budget_override_warning_emitted
        center_mm = np.asarray(work_item.center_mm, dtype=float)
        size_mm = np.asarray(work_item.size_mm, dtype=float)
        max_size_mm = float(max(size_mm))
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
        # Material/gradient-aware refinement: low-conductivity parts (G-10,
        # insulators) carry steep gradients and need fine cells, while high-k metal
        # bulk resolves fine with coarse cells. Refine cells whose dominant material
        # is below the conductivity threshold down to min_cell_size.
        low_conductivity_refinement = False
        low_k_threshold = float(getattr(params, "low_conductivity_refine_threshold", 0.0) or 0.0)
        if (
            low_k_threshold > 0.0
            and classification.occupied
            and classification.dominant_material
            and max_size_mm > params.min_cell_size_mm
        ):
            dominant_material = resolve_material(classification.dominant_material, materials, warnings)
            low_conductivity_refinement = float(dominant_material.k_W_mK) < low_k_threshold
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
            low_conductivity_refinement=low_conductivity_refinement,
        )
        classification.refinement_score = refinement_score
        classification.refinement_reasons = tuple(refinement_reasons)
        effective_queue_len = len(queue) + int(remaining_batch_items)
        budget_allows_children = (
            params.max_leaf_cells is None
            or len(leaves) + effective_queue_len + 8 <= params.max_leaf_cells
        )
        if diagnostics is not None and params.max_leaf_cells is not None and not budget_allows_children:
            diagnostics.max_leaf_cells_reached = True
        max_cell_size_mm = float(params.max_cell_size_mm)
        above_max_cell_size = classification.occupied and max_cell_size_mm > 0.0 and max_size_mm > max_cell_size_mm
        mandatory_max_size_subdivision = above_max_cell_size
        can_subdivide = level < params.max_depth and max_size_mm > params.min_cell_size_mm
        discretionary_refinement = (
            mixed_parts
            or mixed_materials
            or high_contrast
            or low_conductivity_refinement
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
        should_subdivide = mandatory_max_size_subdivision or (
            can_subdivide and budget_allows_children and discretionary_refinement
        )
        if mandatory_max_size_subdivision and not budget_allows_children:
            if diagnostics is not None:
                diagnostics.max_leaf_cells_reached = True
            if not max_leaf_budget_override_warning_emitted and params.max_leaf_cells is not None:
                warnings.append(
                    "max_leaf_cells was exceeded to enforce max_cell_size_mm on occupied cells. "
                    "Optional adaptive refinement remains capped by max_leaf_cells; increase "
                    "--max-leaf-cells if you want more refinement after the mandatory max-size pass."
                )
                max_leaf_budget_override_warning_emitted = True
        if mandatory_max_size_subdivision and not can_subdivide and not max_size_limit_override_warning_emitted:
            warnings.append(
                "max_depth or min_cell_size_mm was exceeded to enforce max_cell_size_mm on occupied cells. "
                "Optional adaptive refinement still respects --max-depth and --min-cell-size-mm after "
                "the mandatory max-size pass."
            )
            max_size_limit_override_warning_emitted = True
        if should_subdivide:
            if diagnostics is not None:
                diagnostics.cells_subdivided += 1
                for reason in refinement_reasons:
                    diagnostics.cells_refined_by_reason[reason] = diagnostics.cells_refined_by_reason.get(reason, 0) + 1
            # Interior-fill: a cell that is occupied, has no surface crossing it, and
            # is claimed by a single component is provably a uniform interior of that
            # solid -- as are all its descendants. Propagate the component/material so
            # children skip containment entirely (the expensive per-cell ray casts).
            interior_fill = (
                classification.occupied
                and not classification.surface_hit
                and classification.dominant_component is not None
                and len(classification.occupancy) == 1
            )
            child_component = classification.dominant_component if interior_fill else None
            child_material = classification.dominant_material if interior_fill else None
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
                        child_component,
                        child_material,
                    ),
                    priority=refinement_score + 0.01 * float(level + 1),
                )
            return

        confidence = _classification_confidence(classification, params)
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
            work_item = pop_cell()
            classification = classify_work_item(work_item)
            handle_classified_cell(work_item, classification, remaining_batch_items=0)
    else:
        # Self-healing multiprocessing. A worker OOM surfaces as a BrokenProcessPool;
        # rather than collapse the ENTIRE remaining build to serial (which is what
        # made real runs crawl), retry the remaining cells with HALF the workers --
        # which need proportionally less RAM -- stepping down until it fits or we
        # reach serial. The in-flight batch is held in ``pending_batch`` so a failure
        # re-runs exactly those cells: none dropped, none double-counted (a failed
        # ``executor.map`` raises before any of its results are handled).
        pending_batch: list[_CellWorkItem] = []

        def _drain_with_pool(mp_workers: int) -> None:
            nonlocal pending_batch
            worker_objects = _prepare_worker_objects(scene.objects)
            with ProcessPoolExecutor(
                max_workers=mp_workers,
                initializer=_init_octree_worker,
                initargs=(worker_objects, contact_report, params, set(materials)),
            ) as executor:
                while queue or pending_batch:
                    if pending_batch:
                        batch = pending_batch
                    else:
                        batch = []
                        while queue and len(batch) < batch_size:
                            work_item = pop_cell()
                            # Interior-fill items need no containment -- handle them
                            # in-process instead of paying a worker IPC round-trip.
                            if work_item.inherited_component is not None:
                                handle_classified_cell(
                                    work_item,
                                    _interior_fill_classification(
                                        work_item.inherited_component, work_item.inherited_material
                                    ),
                                    remaining_batch_items=0,
                                )
                            else:
                                batch.append(work_item)
                        if not batch:
                            continue
                    pending_batch = batch  # in-flight; cleared only after success
                    classifications = list(
                        executor.map(
                            _classify_cell_work_item,
                            batch,
                            chunksize=max(1, min(8, batch_size // max(mp_workers, 1))),
                        )
                    )
                    for index, (work_item, classification) in enumerate(zip(batch, classifications)):
                        handle_classified_cell(
                            work_item, classification, remaining_batch_items=len(batch) - index - 1
                        )
                    pending_batch = []

        while worker_count > 1:
            try:
                _drain_with_pool(worker_count)
                break
            except Exception as exc:  # noqa: BLE001
                next_workers = worker_count // 2
                warnings.append(
                    f"Multiprocessing octree classification failed at {worker_count} worker(s) "
                    f"({type(exc).__name__}: {exc}); retrying remaining cells at {next_workers} "
                    "worker(s)." + (" Continuing serially." if next_workers <= 1 else "")
                )
                worker_count = next_workers
        # Serial drain: any re-queued in-flight batch, then the rest of the queue.
        for index, work_item in enumerate(pending_batch):
            handle_classified_cell(
                work_item, classify_work_item(work_item),
                remaining_batch_items=len(pending_batch) - index - 1,
            )
        pending_batch = []
        while queue:
            work_item = pop_cell()
            handle_classified_cell(work_item, classify_work_item(work_item), remaining_batch_items=0)
    leaves = _balance_adjacent_leaf_sizes(
        leaves,
        classify,
        params,
        warnings,
        diagnostics,
        next_counter=counter,
        progress_callback=progress_callback,
    )
    _mark_final_oversized_leaves(leaves, params, warnings)
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


def _balance_adjacent_leaf_sizes(
    leaves: list[OctreeCell],
    classify: Callable[[Any, Any], CellClassification],
    params: OctreeParams,
    warnings: list[str],
    diagnostics: OctreeDiagnostics | None,
    *,
    next_counter: int,
    progress_callback: Callable[[dict], None] | None = None,
) -> list[OctreeCell]:
    if not bool(getattr(params, "balance_adjacent_leaf_sizes", True)):
        return leaves
    max_passes = max(0, int(getattr(params, "balance_refine_passes", 0)))
    if max_passes <= 0:
        return leaves

    def _emit(processed: int, total: int, pass_index: int, refined: int) -> None:
        # Balancing runs serially after voxelization with no queue -- without this
        # the progress bar sits at queue=0 and the whole thing looks frozen.
        if progress_callback is not None:
            progress_callback({
                "phase": "balancing", "pass": pass_index + 1, "passes": max_passes,
                "processed": processed, "total": total, "refined": refined,
            })

    counter = int(next_counter)
    current = list(leaves)
    for _pass_index in range(max_passes):
        if params.max_leaf_cells is not None and len(current) + 7 > int(params.max_leaf_cells):
            break
        _emit(0, len(current), _pass_index, 0)
        targets = _adjacent_balance_refinement_targets(current, params)
        if not targets:
            break
        refined_any = False
        refined_count = 0
        projected_leaf_count = len(current)
        updated: list[OctreeCell] = []
        total_leaves = len(current)
        for _leaf_idx, leaf in enumerate(current):
            if _leaf_idx % 25000 == 0:
                _emit(_leaf_idx, total_leaves, _pass_index, refined_count)
            if leaf.cell_id not in targets:
                updated.append(leaf)
                continue
            if params.max_leaf_cells is not None and projected_leaf_count + 7 > int(params.max_leaf_cells):
                updated.append(leaf)
                continue
            if leaf.is_empty or int(leaf.level) >= int(params.max_depth) or max(leaf.size_mm) <= float(params.min_cell_size_mm):
                updated.append(leaf)
                continue
            center_mm = np.asarray(leaf.center_mm, dtype=float)
            size_mm = np.asarray(leaf.size_mm, dtype=float)
            child_size = size_mm * 0.5
            quarter = size_mm * 0.25
            child_leaves: list[OctreeCell] = []
            for signs in product((-1.0, 1.0), repeat=3):
                child_id = f"cell_{counter}"
                counter += 1
                child_center = center_mm + quarter * np.asarray(signs, dtype=float)
                classification = classify(child_center, child_size)
                if diagnostics is not None:
                    diagnostics.cells_tested += 1
                    diagnostics.max_depth_reached = max(diagnostics.max_depth_reached, int(leaf.level) + 1)
                    diagnostics.triangle_candidate_tests += int(classification.triangle_candidate_tests)
                    diagnostics.triangle_intersection_tests += int(classification.triangle_intersection_tests)
                    _record_leaf_diagnostics(
                        diagnostics,
                        child_id,
                        child_size,
                        int(leaf.level) + 1,
                        classification,
                    )
                child_leaves.append(
                    OctreeCell(
                        cell_id=child_id,
                        parent_id=leaf.cell_id,
                        children_ids=[],
                        level=int(leaf.level) + 1,
                        center_mm=tuple(float(v) for v in child_center),
                        size_mm=tuple(float(v) for v in child_size),
                        occupancy=classification.occupancy,
                        material_fractions=classification.material_fractions,
                        dominant_component=classification.dominant_component,
                        dominant_material=classification.dominant_material,
                        confidence=_classification_confidence(classification, params),
                        warnings=list(classification.warnings),
                    )
                )
            if diagnostics is not None:
                diagnostics.cells_subdivided += 1
                diagnostics.cells_refined_by_reason["adjacent_size_balance"] = (
                    diagnostics.cells_refined_by_reason.get("adjacent_size_balance", 0) + 1
                )
            updated.extend(child_leaves)
            projected_leaf_count += 7
            refined_any = True
            refined_count += 1
        current = updated
        _emit(total_leaves, total_leaves, _pass_index, refined_count)
        if not refined_any:
            break
    return current


def _adjacent_balance_refinement_targets(leaves: list[OctreeCell], params: OctreeParams) -> set[str]:
    solid = [leaf for leaf in leaves if not leaf.is_empty]
    if len(solid) < 2:
        return set()
    ratio_threshold = max(1.0, float(getattr(params, "max_adjacent_leaf_size_ratio", 4.0)))
    bucket_size = max(float(params.max_cell_size_mm), float(params.min_cell_size_mm), 1.0e-6)
    buckets: dict[tuple[int, int, int], list[OctreeCell]] = {}
    for leaf in solid:
        for key in _leaf_bucket_keys(leaf, bucket_size, padding_mm=1.0e-6):
            buckets.setdefault(key, []).append(leaf)
    targets: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for leaf in solid:
        for key in _leaf_bucket_keys(leaf, bucket_size, padding_mm=1.0e-6):
            for other in buckets.get(key, []):
                if other.cell_id == leaf.cell_id:
                    continue
                pair_key = tuple(sorted((leaf.cell_id, other.cell_id)))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                if not _leaves_touch_or_overlap(leaf, other, tolerance_mm=1.0e-6):
                    continue
                leaf_size = float(max(leaf.size_mm))
                other_size = float(max(other.size_mm))
                smaller = max(min(leaf_size, other_size), 1.0e-9)
                larger = max(leaf_size, other_size)
                if larger / smaller <= ratio_threshold:
                    continue
                coarse = leaf if leaf_size >= other_size else other
                if (
                    int(coarse.level) < int(params.max_depth)
                    and float(max(coarse.size_mm)) > float(params.min_cell_size_mm)
                ):
                    targets.add(coarse.cell_id)
    return targets


def _leaf_bucket_keys(leaf: OctreeCell, bucket_size_mm: float, padding_mm: float = 0.0):
    mins, maxs = _leaf_bounds_mm(leaf)
    mins = mins - float(padding_mm)
    maxs = maxs + float(padding_mm)
    low = np.floor(mins / bucket_size_mm).astype(int)
    high = np.floor(maxs / bucket_size_mm).astype(int)
    for ix in range(int(low[0]), int(high[0]) + 1):
        for iy in range(int(low[1]), int(high[1]) + 1):
            for iz in range(int(low[2]), int(high[2]) + 1):
                yield (ix, iy, iz)


def _leaf_bounds_mm(leaf: OctreeCell) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(leaf.center_mm, dtype=float)
    half = np.asarray(leaf.size_mm, dtype=float) * 0.5
    return center - half, center + half


def _leaves_touch_or_overlap(a: OctreeCell, b: OctreeCell, tolerance_mm: float = 0.0) -> bool:
    a_min, a_max = _leaf_bounds_mm(a)
    b_min, b_max = _leaf_bounds_mm(b)
    return bool(
        np.all(a_min <= b_max + float(tolerance_mm))
        and np.all(b_min <= a_max + float(tolerance_mm))
    )


def _mark_final_oversized_leaves(
    leaves: list[OctreeCell],
    params: OctreeParams,
    warnings: list[str],
) -> None:
    oversized = [
        leaf
        for leaf in leaves
        if not leaf.is_empty and float(max(leaf.size_mm)) > float(params.max_cell_size_mm)
    ]
    if not oversized:
        return
    summary = (
        "Internal warning: occupied cells still exceed max_cell_size_mm after mandatory refinement. "
        "Inspect cell warnings and the octree diagnostics before using this graph."
    )
    if summary not in warnings:
        warnings.append(summary)
    for leaf in oversized:
        leaf.confidence = "low"
        warning = (
            "Accepted occupied cell above max_cell_size_mm after mandatory refinement; "
            "this violates the occupied voxel size contract."
        )
        if warning not in leaf.warnings:
            leaf.warnings.append(warning)


def _resolve_voxel_worker_count(params: OctreeParams) -> int:
    requested = int(getattr(params, "voxel_workers", 1))
    if requested == 0:
        cpu_count = os.cpu_count() or 2
        return max(1, min(2, cpu_count - 1))
    return max(1, requested)


@dataclass
class _WorkerMeshPayload:
    """Picklable per-object payload sent to octree workers: raw vertex/face arrays
    plus metadata. Workers reconstruct a real ``trimesh.Trimesh`` from these so their
    containment tests run on EMBREE (like the serial path) instead of the ~60x-slower
    pure-Python ray fallback. Sending arrays (not a live Trimesh, whose embree ray
    engine isn't picklable) keeps it serializable, and vertices+faces are smaller
    than triangle soup because shared vertices aren't duplicated."""

    name: str
    material_name: str | None
    vertices: np.ndarray
    faces: np.ndarray
    bounds_mm: tuple
    watertight: bool
    scene_path: str | None


def _mesh_vertices_faces(obj: MeshObject) -> tuple[np.ndarray, np.ndarray]:
    """Vertices+faces for an object's mesh; falls back to exploding the triangle
    soup into per-triangle vertices if the mesh has no face topology."""
    mesh = getattr(obj, "mesh", None)
    verts = getattr(mesh, "vertices", None)
    faces = getattr(mesh, "faces", None)
    if verts is not None and faces is not None:
        verts = np.asarray(verts, dtype=float)
        faces = np.asarray(faces)
        if verts.ndim == 2 and verts.shape[1] == 3 and faces.ndim == 2 and faces.shape[1] == 3 and len(faces):
            return np.ascontiguousarray(verts), np.ascontiguousarray(faces, dtype=np.int64)
    triangles = np.asarray(_mesh_triangles(obj), dtype=float)
    if triangles.ndim == 3 and triangles.shape[0] > 0:
        exploded = triangles.reshape(-1, 3)
        return (
            np.ascontiguousarray(exploded),
            np.arange(exploded.shape[0], dtype=np.int64).reshape(-1, 3),
        )
    return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)


def _reconstruct_worker_mesh(payload: "_WorkerMeshPayload"):
    """Rebuild a real embree-capable ``trimesh.Trimesh`` in the worker; fall back to
    the raw triangle-soup mesh (pure-Python ray path) only if trimesh is unavailable."""
    if payload.faces.shape[0] and payload.vertices.shape[0]:
        try:
            import trimesh

            return trimesh.Trimesh(
                vertices=payload.vertices, faces=payload.faces, process=False, validate=False
            )
        except Exception:  # noqa: BLE001
            pass
    if payload.faces.shape[0] and payload.vertices.shape[0]:
        triangles = payload.vertices[payload.faces]
    else:
        triangles = np.empty((0, 3, 3), dtype=float)
    return _TriangleMesh(triangles=triangles)


def _prepare_worker_objects(objects: list[MeshObject]) -> list["_WorkerMeshPayload"]:
    payloads: list[_WorkerMeshPayload] = []
    for obj in objects:
        bounds = _object_bounds_tuple(obj)
        if bounds is None:
            continue
        bounds_min, bounds_max = bounds
        vertices, faces = _mesh_vertices_faces(obj)
        payloads.append(
            _WorkerMeshPayload(
                name=obj.name,
                material_name=obj.material_name,
                vertices=vertices,
                faces=faces,
                bounds_mm=(np.asarray(bounds_min, dtype=float), np.asarray(bounds_max, dtype=float)),
                watertight=bool(obj.watertight),
                scene_path=getattr(obj, "scene_path", None),
            )
        )
    return payloads


def _init_octree_worker(
    objects: list["_WorkerMeshPayload"],
    contact_report: ContactReport,
    params: OctreeParams,
    known_materials: set[str],
) -> None:
    global _TRIMESH_CONTAINS_AVAILABLE
    global _TRIANGLE_CACHE
    global _TRIANGLE_INDEX_CACHE
    global _WORKER_OBJECTS
    global _WORKER_TRIANGLE_INDICES
    global _WORKER_OBJECT_BOUNDS_INDEX
    global _WORKER_CONTACT_REPORT
    global _WORKER_PARAMS
    global _WORKER_KNOWN_MATERIALS

    # None (not False): let the first containment call detect and use embree, same as
    # the serial path. Forcing False here made every worker fall back to the ~60x
    # slower pure-Python ray test -- which is why multiprocessing never actually sped
    # voxelization up. Workers reconstruct real trimeshes below so embree is available.
    _TRIMESH_CONTAINS_AVAILABLE = None
    _TRIANGLE_CACHE = {}
    _TRIANGLE_INDEX_CACHE = {}
    _WORKER_OBJECTS = [
        MeshObject(
            name=payload.name,
            material_name=payload.material_name,
            mesh=_reconstruct_worker_mesh(payload),
            vertices_mm=np.empty((0, 3), dtype=float),
            bounds_mm=payload.bounds_mm,
            watertight=payload.watertight,
            scene_path=payload.scene_path,
        )
        for payload in objects
    ]
    _WORKER_CONTACT_REPORT = contact_report
    _WORKER_PARAMS = params
    _WORKER_KNOWN_MATERIALS = set(known_materials)
    _WORKER_TRIANGLE_INDICES = _build_triangle_indices(_WORKER_OBJECTS, params)
    _WORKER_OBJECT_BOUNDS_INDEX = ObjectBoundsIndex.build(_WORKER_OBJECTS)


def _classify_cell_work_item(work_item: _CellWorkItem) -> CellClassification:
    if work_item.inherited_component is not None:
        return _interior_fill_classification(
            work_item.inherited_component, work_item.inherited_material
        )
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
        _WORKER_OBJECT_BOUNDS_INDEX,
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


def _interior_fill_classification(component: str, material: str | None) -> CellClassification:
    """Classification for a cell provably interior to a single solid.

    Produced without any containment test: a homogeneous-interior parent
    guarantees its children are uniformly inside the same component (see
    ``handle_classified_cell``). Occupancy/material are identical to what a full
    ``_classify_cell`` sweep would return for such a cell, so graph output is
    unchanged -- only the ray-casting work is skipped. Test counters are zeroed
    so diagnostics stay accurate.
    """
    materials = {material: 1.0} if material else {}
    return CellClassification(
        occupied=True,
        surface_hit=False,
        inside_hit=True,
        near_surface_hit=False,
        bbox_only_hit=False,
        surface_mesh_ids=set(),
        inside_mesh_ids=set(),
        near_surface_mesh_ids=set(),
        candidate_mesh_ids=set(),
        crowded_component_count=0,
        role_component_count=0,
        surface_component_count=0,
        near_surface_component_count=0,
        material_ids=set(materials),
        part_ids={component},
        occupancy={component: 1.0},
        material_fractions=materials,
        dominant_component=component,
        dominant_material=material,
        volume_fraction=1.0,
        acceptance_reason="inherited_interior_fill",
        warnings=[],
        triangle_candidate_tests=0,
        triangle_intersection_tests=0,
    )


def _classify_cell(
    objects: list[MeshObject],
    triangle_indices: dict[int, TriangleSpatialIndex],
    center_mm: np.ndarray,
    size_mm: np.ndarray,
    contact_report: ContactReport,
    params: OctreeParams,
    known_materials: set[str],
    diagnostics: OctreeDiagnostics | None = None,
    object_bounds_index: "ObjectBoundsIndex | None" = None,
) -> CellClassification:
    # Reuse a prebuilt vectorized bounds index when the caller supplies one
    # (the hot path); otherwise build a transient one so direct/test callers
    # keep working with the same overlap semantics.
    index = object_bounds_index if object_bounds_index is not None else ObjectBoundsIndex.build(objects)
    half = size_mm * 0.5
    cell_min = center_mm - half
    cell_max = center_mm + half
    near_margin = max(0.0, min(float(params.contact_refine_distance_mm), float(max(size_mm)) * 0.5))
    near_min = cell_min - near_margin
    near_max = cell_max + near_margin
    candidate_objects = index.query(near_min, near_max)
    crowded_margin = _crowded_component_refine_margin_mm(size_mm, params)
    crowded_objects = (
        index.query(cell_min - crowded_margin, cell_max + crowded_margin)
        if int(params.crowded_component_refine_count) > 0
        else []
    )
    role_refine_names = set(getattr(params, "role_refine_component_names", ()) or ())
    role_refine_margin = max(0.0, float(getattr(params, "role_refine_distance_mm", 0.0)))
    role_refine_objects = (
        [
            obj
            for obj in index.query(
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
    if len(points) and watertight_candidates:
        pts = np.asarray(points, dtype=float)
        # Each point is claimed by the FIRST watertight candidate (in order) that
        # contains it -- same as the old per-point/break loop, but batched: one
        # containment call per candidate over all still-unclaimed in-bounds points.
        remaining = np.ones(pts.shape[0], dtype=bool)
        for obj in watertight_candidates:
            if not remaining.any():
                break
            bounds = _object_bounds_tuple(obj)
            if bounds is None:
                continue
            (bx0, by0, bz0), (bx1, by1, bz1) = bounds
            in_bounds = (
                remaining
                & (pts[:, 0] >= bx0) & (pts[:, 0] <= bx1)
                & (pts[:, 1] >= by0) & (pts[:, 1] <= by1)
                & (pts[:, 2] >= bz0) & (pts[:, 2] <= bz1)
            )
            if not in_bounds.any():
                continue
            try:
                inside_sub = _mesh_contains_points(obj, pts[in_bounds], params)
            except Exception:
                warnings.append(f"Inside/outside test failed for watertight mesh {obj.name}.")
                continue
            claimed = np.nonzero(in_bounds)[0][np.asarray(inside_sub, dtype=bool)]
            if claimed.size:
                inside_counts[obj.name] += int(claimed.size)
                material_counts[_physical_material_name(obj, contact_report, known_materials)] += int(claimed.size)
                remaining[claimed] = False

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


def _crowded_component_refine_margin_mm(size_mm: np.ndarray, params: OctreeParams) -> float:
    absolute_margin = max(0.0, float(getattr(params, "crowded_component_refine_distance_mm", 0.0) or 0.0))
    if absolute_margin > 0.0:
        return absolute_margin
    neighbor_cells = max(0.0, float(getattr(params, "crowded_component_refine_neighbor_cells", 0.0) or 0.0))
    return neighbor_cells * float(max(size_mm))


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
    low_conductivity_refinement: bool = False,
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
    if low_conductivity_refinement:
        add("low_conductivity_material", 40.0)
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
    """Resolve a component's engineering material. The per-mesh material lookup sheet
    is authoritative; the GLB's own material name (a CAD *appearance* like
    "brushed_aluminum") is only a fallback, since it cannot distinguish materials
    such as Invar or fiberglass."""
    known_lookup = {name: None for name in known_materials}
    # 1. Material-lookup spreadsheet (authoritative).
    report_material = contact_report.material_for_component(obj.name)
    if report_material and not is_unassigned_material_name(report_material):
        if report_material in known_materials:
            return report_material
        inferred_report = infer_material_name_from_text(report_material, known_lookup)
        if inferred_report:
            return inferred_report
    # 2. GLB material (appearance) name, exact then via aliases.
    if obj.material_name and not is_unassigned_material_name(obj.material_name):
        if obj.material_name in known_materials:
            return obj.material_name
        inferred_material = infer_material_name_from_text(obj.material_name, known_lookup)
        if inferred_material:
            return inferred_material
    # 3. Component name text.
    inferred_material = infer_material_name_from_text(obj.name, known_lookup)
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


def _mesh_contains_points(obj: MeshObject, points: np.ndarray, params: OctreeParams) -> np.ndarray:
    """Batched point-in-mesh test: one call for a whole array of points.

    trimesh.contains() and the ray-parity fallback both vectorize over points, so
    testing all of a cell's samples at once (instead of one call per point) is the
    dominant speedup when FILLING watertight meshes (the STEP path). Returns a
    boolean array aligned with ``points``."""
    global _TRIMESH_CONTAINS_AVAILABLE
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.zeros(0, dtype=bool)
    if str(getattr(params, "contains_backend", "ray")).lower() != "ray" and _TRIMESH_CONTAINS_AVAILABLE is not False:
        try:
            result = np.asarray(obj.mesh.contains(points), dtype=bool)
            _TRIMESH_CONTAINS_AVAILABLE = True
            if result.shape[0] == points.shape[0]:
                return result
        except (ImportError, ModuleNotFoundError):
            _TRIMESH_CONTAINS_AVAILABLE = False
        except Exception:
            pass
    return _ray_contains_points(obj, points)


def _ray_contains_points(obj: MeshObject, points: np.ndarray) -> np.ndarray:
    """Vectorized odd/even ray test for an array of points (no trimesh needed).

    Precomputes the per-triangle Moller-Trumbore terms ONCE, then tests each point
    against all triangles. The parity ray runs to +infinity, so all triangles
    must be considered (no cell-local culling)."""
    points = np.asarray(points, dtype=float)
    out = np.zeros(points.shape[0], dtype=bool)
    if points.shape[0] == 0:
        return out
    triangles = _mesh_triangles(obj)
    if triangles.size == 0:
        return out
    direction = np.array([1.0, 0.3713906763541037, 0.1937728766089219], dtype=float)
    direction /= np.linalg.norm(direction)
    eps = 1.0e-9
    v0 = triangles[:, 0, :]
    edge1 = triangles[:, 1, :] - v0
    edge2 = triangles[:, 2, :] - v0
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    a = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(a) > eps
    f = np.zeros_like(a)
    f[valid] = 1.0 / a[valid]
    for index in range(points.shape[0]):
        s = points[index] - v0
        u = f * np.einsum("ij,ij->i", s, h)
        mask = valid & (u >= -eps) & (u <= 1.0 + eps)
        if not mask.any():
            continue
        q = np.cross(s, edge1)
        v = f * np.einsum("ij,j->i", q, direction)
        mask &= (v >= -eps) & ((u + v) <= 1.0 + eps)
        if not mask.any():
            continue
        t = f * np.einsum("ij,ij->i", edge2, q)
        hits = t[mask & (t > eps)]
        if hits.size and (np.unique(np.round(hits, decimals=8)).size % 2 == 1):
            out[index] = True
    return out


class ObjectBoundsIndex:
    """Vectorized AABB-overlap index over a fixed object list.

    ``_classify_cell`` needs the objects whose bounding boxes overlap a query box,
    up to a few times per cell. Doing that as a Python ``for obj in objects`` loop
    is O(N_objects) *per cell* -- for a large assembly that is billions of
    bbox tests across the whole octree. This precomputes every object's bounds
    into ``(N, 3)`` min/max arrays once, so each query is a single vectorized pass
    instead of a per-object Python loop.

    Objects with missing / non-finite bounds get ``+inf`` min and ``-inf`` max
    sentinel boxes so they can never satisfy the overlap test -- reproducing the
    old "skip objects with no usable bounds" behaviour while keeping the array
    rows aligned with ``objects`` order (so results come back in the same order
    the loop produced them).
    """

    # Bounds are stored as six per-axis *contiguous* 1-D arrays rather than two
    # (N, 3) arrays: the overlap test then becomes six 1-D comparisons over
    # cache-friendly columns instead of a broadcast + ``np.all(axis=1)`` reduction
    # over a 2-D temporary. Measured ~13x faster on a 7.8k-object assembly.
    __slots__ = ("objects", "_count", "_min_x", "_min_y", "_min_z", "_max_x", "_max_y", "_max_z")

    def __init__(self, objects: list[MeshObject], mins: np.ndarray, maxs: np.ndarray) -> None:
        self.objects = objects
        self._count = len(objects)
        self._min_x = np.ascontiguousarray(mins[:, 0])
        self._min_y = np.ascontiguousarray(mins[:, 1])
        self._min_z = np.ascontiguousarray(mins[:, 2])
        self._max_x = np.ascontiguousarray(maxs[:, 0])
        self._max_y = np.ascontiguousarray(maxs[:, 1])
        self._max_z = np.ascontiguousarray(maxs[:, 2])

    @classmethod
    def build(cls, objects: list[MeshObject]) -> "ObjectBoundsIndex":
        objects = list(objects)
        count = len(objects)
        mins = np.empty((count, 3), dtype=float)
        maxs = np.empty((count, 3), dtype=float)
        for i, obj in enumerate(objects):
            bounds = _object_bounds_tuple(obj)
            if bounds is None:
                mins[i, :] = np.inf  # never >= a finite query_min
                maxs[i, :] = -np.inf  # never <= a finite query_max
                continue
            (mins[i, 0], mins[i, 1], mins[i, 2]), (maxs[i, 0], maxs[i, 1], maxs[i, 2]) = bounds
        return cls(objects, mins, maxs)

    def query(self, bounds_min: np.ndarray, bounds_max: np.ndarray) -> list[MeshObject]:
        try:
            query_min = np.asarray(bounds_min, dtype=float).reshape(3)
            query_max = np.asarray(bounds_max, dtype=float).reshape(3)
        except (TypeError, ValueError):
            return []
        if not (np.isfinite(query_min).all() and np.isfinite(query_max).all()):
            return []
        if self._count == 0:
            return []
        lo0, hi0 = min(query_min[0], query_max[0]), max(query_min[0], query_max[0])
        lo1, hi1 = min(query_min[1], query_max[1]), max(query_min[1], query_max[1])
        lo2, hi2 = min(query_min[2], query_max[2]), max(query_min[2], query_max[2])
        # Inclusive overlap on every axis (matches _bounds_intersect_tuple):
        # obj_max >= query_min AND obj_min <= query_max.
        mask = (
            (self._max_x >= lo0)
            & (self._min_x <= hi0)
            & (self._max_y >= lo1)
            & (self._min_y <= hi1)
            & (self._max_z >= lo2)
            & (self._min_z <= hi2)
        )
        objects = self.objects
        return [objects[i] for i in np.nonzero(mask)[0]]


def _objects_intersecting_bounds(
    objects: list[MeshObject], bounds_min: np.ndarray, bounds_max: np.ndarray
) -> list[MeshObject]:
    # Delegates to ObjectBoundsIndex so there is a single overlap implementation;
    # hot-path callers reuse a prebuilt index instead of rebuilding one per query.
    return ObjectBoundsIndex.build(objects).query(bounds_min, bounds_max)


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
