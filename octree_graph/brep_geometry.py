"""B-rep (STEP) geometry backend for the octree voxelizer.

Classifies each octree cell by exact OpenCASCADE point-in-solid tests against the
STEP solids, so solid interiors *fill* (unlike the GLB surface path, which could
only shell non-watertight parts). Implements the ``GeometryProvider`` seam that
``build_octree`` consumes; runs in-process (OCC handles aren't picklable to
multiprocessing workers).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from .load_contact_report import ContactReport
from .load_step import StepScene
from .materials import Material
from .octree import CellClassification, OctreeParams, _physical_material_name


def _sample_grid(center_mm: np.ndarray, size_mm: np.ndarray, n_axis: int) -> np.ndarray:
    """Cell-interior sample points on an n×n×n grid (cell-relative centers)."""
    n = max(1, int(n_axis))
    lo = center_mm - size_mm * 0.5
    offs = (np.arange(n) + 0.5) / n
    xs = lo[0] + offs * size_mm[0]
    ys = lo[1] + offs * size_mm[1]
    zs = lo[2] + offs * size_mm[2]
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def _empty_classification(
    candidate_ids: set[int] | None = None, bbox_only: bool = False
) -> CellClassification:
    return CellClassification(
        occupied=False,
        surface_hit=False,
        inside_hit=False,
        near_surface_hit=False,
        bbox_only_hit=bool(bbox_only),
        surface_mesh_ids=set(),
        inside_mesh_ids=set(),
        near_surface_mesh_ids=set(),
        candidate_mesh_ids=set(candidate_ids or set()),
        crowded_component_count=0,
        role_component_count=0,
        surface_component_count=0,
        near_surface_component_count=0,
        material_ids=set(),
        part_ids=set(),
        occupancy={},
        material_fractions={},
        dominant_component=None,
        dominant_material=None,
        volume_fraction=None,
        acceptance_reason="bbox_only_empty" if bbox_only else "empty",
    )


class BRepGeometryProvider:
    """Occupancy classification from STEP B-rep solids (exact point-in-solid).

    Each solid becomes its own component (``<name>#<index>``) so repeated
    instances that share a CAD name stay distinct nodes; the raw name is used for
    material resolution (against ``Materials.xlsx`` via the same resolver as the
    GLB path). Occupancy is the ABSOLUTE fill fraction (inside samples / total),
    which is what ``graph_builder`` multiplies into cell mass.
    """

    supports_multiprocessing = False

    def __init__(
        self,
        scene: StepScene,
        contact_report: ContactReport | None,
        materials: dict[str, Material],
        params: OctreeParams,
        *,
        samples_per_axis: int = 3,
        classify_tol_mm: float = 1.0e-4,
    ) -> None:
        self.scene = scene
        self.solids = scene.solids
        self.contact_report = contact_report or ContactReport()
        self.known_materials = set(materials)
        self.params = params
        self.samples_per_axis = int(samples_per_axis)
        self.classify_tol_mm = float(classify_tol_mm)
        self.warnings: list[str] = list(scene.warnings)

        n = len(self.solids)
        self._lo = np.zeros((n, 3), dtype=float)
        self._hi = np.zeros((n, 3), dtype=float)
        self._component_id: list[str] = []
        self._material: list[str] = []
        for i, solid in enumerate(self.solids):
            lo, hi = solid.bounds_mm
            self._lo[i] = lo
            self._hi[i] = hi
            raw = solid.name or f"solid_{i}"
            self._component_id.append(f"{raw}#{i}")
            self._material.append(self._resolve_material(raw))

    @property
    def bounds_mm(self) -> tuple[np.ndarray, np.ndarray]:
        return self.scene.bounds_mm

    def _resolve_material(self, raw_name: str) -> str:
        # Reuse the GLB resolver: spreadsheet (authoritative) -> name inference
        # -> inert default. STEP carries no usable appearance name, so leave it None.
        obj = SimpleNamespace(name=raw_name, material_name=None)
        return _physical_material_name(obj, self.contact_report, self.known_materials)

    def _candidates(self, cmin: np.ndarray, cmax: np.ndarray) -> np.ndarray:
        overlap = (
            (self._hi[:, 0] >= cmin[0])
            & (self._lo[:, 0] <= cmax[0])
            & (self._hi[:, 1] >= cmin[1])
            & (self._lo[:, 1] <= cmax[1])
            & (self._hi[:, 2] >= cmin[2])
            & (self._lo[:, 2] <= cmax[2])
        )
        return np.nonzero(overlap)[0]

    def classify_cell(self, center_mm: np.ndarray, size_mm: np.ndarray) -> CellClassification:
        center_mm = np.asarray(center_mm, dtype=float)
        size_mm = np.asarray(size_mm, dtype=float)
        half = size_mm * 0.5
        cmin, cmax = center_mm - half, center_mm + half
        cand = self._candidates(cmin, cmax)
        if cand.size == 0:
            return _empty_classification()

        points = _sample_grid(center_mm, size_mm, self.samples_per_axis)
        total = len(points)
        candidate_ids = {id(self.solids[int(i)]) for i in cand}
        inside_counts: dict[str, int] = {}
        material_counts: dict[str, int] = {}
        component_material: dict[str, str] = {}
        inside_ids: set[int] = set()
        part_ids: set[str] = set()
        material_ids: set[str] = set()

        for i in cand:
            solid = self.solids[int(i)]
            cid = self._component_id[int(i)]
            mat = self._material[int(i)]
            count = 0
            for p in points:
                if solid.contains_point(p[0], p[1], p[2], tol=self.classify_tol_mm):
                    count += 1
            if count > 0:
                inside_counts[cid] = inside_counts.get(cid, 0) + count
                material_counts[mat] = material_counts.get(mat, 0) + count
                component_material[cid] = mat
                inside_ids.add(id(solid))
                part_ids.add(cid)
                material_ids.add(mat)

        if not inside_counts:
            # Candidate solids' bboxes overlap the cell but no sample landed
            # inside. At a coarse cell this almost always means the solid is
            # under-sampled (a thin/localized part between sample points), NOT
            # that the cell is empty -- so mark it NEAR-SURFACE to force continued
            # subdivision (via boundary refinement) until samples can resolve it.
            # If subdivision bottoms out at min_cell_size still empty, it becomes a
            # genuine empty leaf. This mirrors how the triangle path keeps
            # refining toward surfaces it hasn't resolved yet.
            n_candidate_components = len({self._component_id[int(i)] for i in cand})
            return CellClassification(
                occupied=False,
                surface_hit=False,
                inside_hit=False,
                near_surface_hit=True,
                bbox_only_hit=True,
                surface_mesh_ids=set(),
                inside_mesh_ids=set(),
                near_surface_mesh_ids=set(candidate_ids),
                candidate_mesh_ids=candidate_ids,
                crowded_component_count=0,
                role_component_count=0,
                surface_component_count=0,
                near_surface_component_count=n_candidate_components,
                material_ids=set(),
                part_ids=set(),
                occupancy={},
                material_fractions={},
                dominant_component=None,
                dominant_material=None,
                volume_fraction=0.0,
                acceptance_reason="brep_near_surface",
            )

        # ABSOLUTE fill fraction per component (inside / total): graph_builder uses
        # max(occupancy) directly as the cell fill fraction for mass.
        occupancy = {cid: min(1.0, cnt / total) for cid, cnt in inside_counts.items()}
        material_fractions = {m: min(1.0, cnt / total) for m, cnt in material_counts.items()}
        dominant_component = max(occupancy, key=occupancy.get)
        dominant_material = component_material[dominant_component]
        dominant_fraction = occupancy[dominant_component]
        minority = float(self.params.minority_fraction_ignore)
        n_components = sum(1 for f in occupancy.values() if f > minority)

        return CellClassification(
            occupied=True,
            # A partially-filled cell (a solid boundary crosses it) or a cell
            # shared by >1 component is a "surface" cell -> drives boundary
            # refinement so surfaces resolve finely while interiors stay coarse.
            surface_hit=bool(dominant_fraction < 1.0 or n_components > 1),
            inside_hit=True,
            near_surface_hit=False,
            bbox_only_hit=False,
            surface_mesh_ids=set(inside_ids),
            inside_mesh_ids=set(inside_ids),
            near_surface_mesh_ids=set(),
            candidate_mesh_ids=candidate_ids,
            crowded_component_count=n_components,
            role_component_count=0,
            surface_component_count=len(part_ids),
            near_surface_component_count=0,  # triangle-gap heuristic; N/A for exact B-rep
            material_ids=material_ids,
            part_ids=part_ids,
            occupancy=occupancy,
            material_fractions=material_fractions,
            dominant_component=dominant_component,
            dominant_material=dominant_material,
            volume_fraction=dominant_fraction,
            acceptance_reason="brep_inside",
        )
