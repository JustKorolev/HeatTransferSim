"""Load a STEP (ISO 10303) assembly as B-rep solids for octree voxelization.

Unlike the GLB path (surface triangles), STEP carries true B-rep *solids*, so a
cell's interior can be tested exactly with an OpenCASCADE point-in-solid query
(``BRepClass3d_SolidClassifier``) instead of relying on watertight triangle
containment. This is what lets the octree fill solids rather than shelling them.

Requires ``pythonocc-core`` (OpenCASCADE). The import is intentionally local to
the functions that need it so the rest of ``octree_graph`` imports without the
kernel installed; the GLB pipeline stays usable in a plain-CPython environment.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class StepSolid:
    """One B-rep solid from the STEP, with a cached point-in-solid classifier.

    ``name`` is the CAD component name (same namespace as the GLB: ``HISPEC-####``
    codes, ``V_VENDOR_`` names, ``^`` hierarchy separators). ``bounds_mm`` is the
    axis-aligned bounding box ``(min_xyz, max_xyz)`` in millimetres.
    """

    name: str
    shape: Any  # TopoDS_Shape (a solid)
    bounds_mm: tuple[np.ndarray, np.ndarray]
    _classifier: Any = field(default=None, repr=False)

    @property
    def size_mm(self) -> np.ndarray:
        lo, hi = self.bounds_mm
        return np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)

    def contains_point(self, x: float, y: float, z: float, tol: float = 1.0e-6) -> bool:
        """True if the point is inside (or on) the solid. Reuses a cached classifier."""
        from OCC.Core.gp import gp_Pnt
        from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON

        if self._classifier is None:
            from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier

            self._classifier = BRepClass3d_SolidClassifier(self.shape)
        self._classifier.Perform(gp_Pnt(float(x), float(y), float(z)), float(tol))
        return self._classifier.State() in (TopAbs_IN, TopAbs_ON)


@dataclass
class StepScene:
    """All solids loaded from a STEP file, plus the overall bounding box."""

    path: Path
    solids: list[StepSolid]
    bounds_mm: tuple[np.ndarray, np.ndarray]
    warnings: list[str] = field(default_factory=list)


def _shape_bounds_mm(shape: Any) -> tuple[np.ndarray, np.ndarray]:
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (
        np.array([xmin, ymin, zmin], dtype=float),
        np.array([xmax, ymax, zmax], dtype=float),
    )


def _iter_solids(shape: Any):
    """Yield each TopAbs_SOLID in a (possibly compound) shape."""
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer

    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        yield exp.Current()
        exp.Next()


def load_step_scene(path: str | Path, *, split_compounds: bool = True) -> StepScene:
    """Read a STEP file into a list of named B-rep solids.

    Each named shape in the STEP may be a single solid or a compound of several;
    with ``split_compounds`` (default) each solid becomes its own ``StepSolid``
    (sharing the parent name), mirroring how the GLB path treats one mesh object
    per component. Names come from the XCAF product structure via pythonocc's
    ``read_step_file_with_names_colors`` helper (which sets up the XCAF document
    correctly -- constructing ``TDocStd_Document`` by hand aborts in OCCT 7.9).
    """
    from OCC.Extend.DataExchange import read_step_file_with_names_colors

    path = Path(path)
    # The helper prints per-shape diagnostics to stdout; silence it (huge on a
    # 10k-part assembly) while still capturing the returned mapping.
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        shape_map = read_step_file_with_names_colors(str(path))

    solids: list[StepSolid] = []
    warnings: list[str] = []
    global_lo = np.array([np.inf, np.inf, np.inf], dtype=float)
    global_hi = np.array([-np.inf, -np.inf, -np.inf], dtype=float)

    for shape, meta in shape_map.items():
        name = ""
        if isinstance(meta, (tuple, list)) and meta:
            name = str(meta[0] or "")
        members = list(_iter_solids(shape)) if split_compounds else [shape]
        if not members:
            # No solids (a sheet body / wireframe): skip -- nothing to fill. The
            # caller can log these; they carry no volume.
            warnings.append(f"STEP part {name!r} has no solids (sheet body); skipped.")
            continue
        for solid in members:
            try:
                bounds = _shape_bounds_mm(solid)
            except Exception as exc:  # noqa: BLE001 - keep loading the rest
                warnings.append(f"Bounds failed for {name!r}: {exc}")
                continue
            lo, hi = bounds
            global_lo = np.minimum(global_lo, lo)
            global_hi = np.maximum(global_hi, hi)
            solids.append(StepSolid(name=name, shape=solid, bounds_mm=bounds))

    if not solids:
        raise ValueError(f"No B-rep solids found in STEP file {path}.")
    return StepScene(path=path, solids=solids, bounds_mm=(global_lo, global_hi), warnings=warnings)


def _tessellate_shape(shape: Any, deflection_mm: float, angular_deg: float):
    """Tessellate one solid to (vertices, faces) with outward-consistent winding.

    Per-face triangulations share vertices only after a merge (BRepMesh emits
    per-face nodes); the caller merges via trimesh(process=True) to get a
    watertight mesh. Reversed faces have their winding flipped so normals point
    out (needed for reliable point-in-mesh containment)."""
    import math

    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods

    BRepMesh_IncrementalMesh(shape, float(deflection_mm), False, math.radians(float(angular_deg)), True)
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        explorer.Next()
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is None:
            continue
        transform = location.Transformation()
        reversed_face = face.Orientation() == TopAbs_REVERSED
        base = len(verts)
        for i in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(i).Transformed(transform)
            verts.append((point.X(), point.Y(), point.Z()))
        for i in range(1, triangulation.NbTriangles() + 1):
            a, b, c = triangulation.Triangle(i).Get()
            if reversed_face:
                a, c = c, a
            faces.append((base + a - 1, base + b - 1, base + c - 1))
    if not faces:
        return None
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)


def load_step_as_scene(
    path: str | Path,
    *,
    deflection_mm: float = 0.5,
    angular_deg: float = 20.0,
    split_compounds: bool = True,
):
    """Load a STEP assembly as a ``GltfScene`` of WATERTIGHT triangle meshes.

    Each B-rep solid is tessellated (OpenCASCADE ``BRepMesh``) and vertex-merged
    into a closed ``trimesh.Trimesh``. Because the meshes are watertight, the
    existing triangle voxelizer *fills* them (point-in-mesh containment) instead
    of shelling -- so a STEP assembly runs through the entire GLB pipeline
    unchanged (role detection, ignore filters, contact/radiation, multiprocessing
    workers), with solids filled. ``deflection_mm`` controls tessellation
    fineness (smaller = more triangles = truer curved surfaces, slower).
    """
    import trimesh

    from .load_gltf import GltfScene, MeshObject

    path = Path(path)
    step_scene = load_step_scene(path, split_compounds=split_compounds)
    objects: list[Any] = []
    warnings = list(step_scene.warnings)
    global_lo = np.array([np.inf, np.inf, np.inf], dtype=float)
    global_hi = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    non_watertight = 0

    for index, solid in enumerate(step_scene.solids):
        tessellation = _tessellate_shape(solid.shape, deflection_mm, angular_deg)
        if tessellation is None:
            warnings.append(f"Tessellation produced no triangles for {solid.name!r}; skipped.")
            continue
        vertices, faces = tessellation
        # process=True merges the per-face duplicate vertices -> a closed mesh so
        # is_watertight is meaningful and trimesh.contains() fills reliably.
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        name = solid.name or f"solid_{index}"
        lo, hi = solid.bounds_mm
        global_lo = np.minimum(global_lo, lo)
        global_hi = np.maximum(global_hi, hi)
        watertight = bool(getattr(mesh, "is_watertight", False))
        if not watertight:
            non_watertight += 1
        objects.append(
            MeshObject(
                name=name,
                material_name=None,
                mesh=mesh,
                vertices_mm=np.asarray(mesh.vertices, dtype=float),
                bounds_mm=(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)),
                watertight=watertight,
                scene_path=name,
                hierarchy_path=(name,),
            )
        )
    if not objects:
        raise ValueError(f"No tessellated solids produced from STEP file {path}.")
    if non_watertight:
        warnings.append(
            f"{non_watertight}/{len(objects)} tessellated STEP solids are not watertight after merge; "
            "those cells fall back to ray containment (may under-fill)."
        )
    scene = GltfScene(path=path, objects=objects, bounds_mm=(global_lo, global_hi), warnings=warnings)
    # Return the B-rep StepScene too so the caller can build EXACT component-contact
    # tests (OCC solid-solid distance) from the original solids, not the tessellation.
    return scene, step_scene


def _bbox_gap_mm(lo_a, hi_a, lo_b, hi_b) -> float:
    """Axis-aligned gap between two bounding boxes (0 if they overlap)."""
    gap = 0.0
    for axis in range(3):
        d = max(lo_a[axis] - hi_b[axis], lo_b[axis] - hi_a[axis], 0.0)
        gap = max(gap, float(d))
    return gap


def make_component_contact_fn(step_scene: StepScene, tolerance_mm: float):
    """Return ``contact(name_a, name_b) -> bool``: True iff any solid of component
    A comes within ``tolerance_mm`` of any solid of component B, by EXACT B-rep
    distance (OpenCASCADE ``BRepExtrema_DistShapeShape``). Cached per unordered
    pair and bbox-prefiltered, so it's computed at most once per component pair
    that actually has a voxel interface. This is the "do the parts truly touch?"
    test, decided on the real solids instead of voxel adjacency or tessellation."""
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape

    tol = float(tolerance_mm)
    by_name: dict[str, list[StepSolid]] = {}
    for solid in step_scene.solids:
        by_name.setdefault(solid.name or "", []).append(solid)
    cache: dict[frozenset, bool] = {}

    def contact(name_a: str, name_b: str) -> bool:
        key = frozenset((name_a, name_b))
        cached = cache.get(key)
        if cached is not None:
            return cached
        solids_a = by_name.get(name_a, [])
        solids_b = by_name.get(name_b, [])
        result = False
        for solid_a in solids_a:
            lo_a, hi_a = solid_a.bounds_mm
            for solid_b in solids_b:
                lo_b, hi_b = solid_b.bounds_mm
                if _bbox_gap_mm(lo_a, hi_a, lo_b, hi_b) > tol:
                    continue  # bboxes already farther apart than tolerance
                try:
                    distance = BRepExtrema_DistShapeShape(solid_a.shape, solid_b.shape).Value()
                except Exception:
                    result = True  # can't measure -> assume contact (don't suppress)
                    break
                if distance <= tol:
                    result = True
                    break
            if result:
                break
        cache[key] = result
        return result

    return contact
