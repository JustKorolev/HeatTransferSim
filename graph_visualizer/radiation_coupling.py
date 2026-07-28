"""Connect the octree lumped graph to the ray-traced view-factor engine.

This is the integration layer for surface-to-surface radiative coupling:

  1. ``exposed_face_patches(model)`` reconstructs the radiating surfaces from the
     graph -- each exposed cell face (a face bordering empty space) becomes a
     ``SurfacePatch`` owned by its graph node. Exposure is the cell's full face
     area minus the parts covered by touching neighbours (same rule the exposed-
     area / radiation code already uses).
  2. ``compute_radiation_exchange_links(model, ...)`` ray-traces view factors
     among those patches, converts them to gray-diffuse exchange areas, and
     aggregates to node-level ``radiation_exchange_links`` the solver consumes.
  3. ``apply_radiation_coupling(model, ...)`` runs (2) and attaches the result to
     the model, so ``prepare_simulation`` picks it up.

First-cut simplifications (documented so they are not mistaken for exactness):
  * Each exposed face is a single patch (full face rectangle for occlusion,
    exposed area for the radiative weighting) -- partial-coverage sub-shapes are
    not resolved.
  * The escaped-ray fraction all goes to the exterior background; interior vs
    exterior classification is deferred (with modeled, cryocooler-driven shields
    the enclosed surfaces couple to those shields directly, so escapes are mostly
    outward anyway).
  * Intersection uses trimesh's numpy ray engine -- fine for moderate models; use
    an Embree backend and super-surface grouping for large assemblies.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import tempfile
from pathlib import Path

import numpy as np

from .matrix_builder import _is_cad_role_node, _node_bounds_mm

# Persistent, content-addressed cache for the (expensive) radiative-coupling
# precompute, so the same graph + params loads instantly after the first build.
_DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "heattransfersim_radiation_cache"
# Bump when the coupling algorithm changes so stale caches are invalidated.
_COUPLING_CACHE_VERSION = 2


def _coupling_cache_key(
    model, target_super_surfaces: int, rays_per_patch: int, seed: int,
    assume_enclosure_vacuum: bool, exterior_view_threshold: float,
) -> str:
    """Content hash of everything that changes the coupling: each body node's
    geometry + emissivity, and the trace parameters. Changing any -> cache miss."""
    hasher = hashlib.sha256()
    for node_id in sorted(int(v) for v in model.nodes):
        node = model.nodes[node_id]
        if _is_cad_role_node(node):
            continue
        center = node.center_mm or (0.0, 0.0, 0.0)
        size = node.size_mm or (0.0, 0.0, 0.0)
        hasher.update(struct.pack("i", node_id))
        hasher.update(struct.pack(
            "7d", float(center[0]), float(center[1]), float(center[2]),
            float(size[0]), float(size[1]), float(size[2]),
            float(getattr(node, "emissivity", 0.0) or 0.0),
        ))
    hasher.update(struct.pack(
        "iiidid", int(_COUPLING_CACHE_VERSION), int(target_super_surfaces), int(seed),
        float(rays_per_patch), 1 if assume_enclosure_vacuum else 0, float(exterior_view_threshold),
    ))
    return hasher.hexdigest()


def _load_coupling_cache(path: Path):
    try:
        data = json.loads(path.read_text())
        members = [{int(k): float(v) for k, v in entry.items()} for entry in data["members"]]
        links = [(int(i), int(j), float(g)) for i, j, g in data["links"]]
        env = {int(k): float(v) for k, v in data["env_fraction"].items()}
        return members, links, env, dict(data.get("diagnostics", {}))
    except Exception:  # noqa: BLE001 - a bad/absent cache just means recompute
        return None


def _save_coupling_cache(path: Path, members, links, env, diagnostics) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "members": [{str(k): v for k, v in entry.items()} for entry in members],
            "links": [[int(i), int(j), float(g)] for i, j, g in links],
            "env_fraction": {str(k): float(v) for k, v in env.items()},
            "diagnostics": {k: float(v) for k, v in diagnostics.items()},
        }
        path.write_text(json.dumps(payload))
    except Exception:  # noqa: BLE001 - caching is best-effort
        pass
from .radiation_view_factors import (
    SurfacePatch,
    compute_view_factors,
    exchange_links_by_group,
    total_exchange_areas,
)

_TOLERANCE_MM = 1.0e-7


def exposed_face_patches(
    model,
    *,
    min_exposed_fraction: float = 1.0e-3,
) -> list[SurfacePatch]:
    """Build one SurfacePatch per exposed cell face (face bordering empty space).

    A face's exposed area is its full area minus the area covered by neighbouring
    cells that abut it. Faces exposed above ``min_exposed_fraction`` of their full
    area become patches, grouped (``group_id``) by owning graph node.
    """
    bodies: list[tuple[int, np.ndarray, np.ndarray]] = []
    min_cell_dim_mm = float("inf")
    for node_id in model.ordered_node_ids():
        node = model.nodes[int(node_id)]
        if _is_cad_role_node(node):
            continue
        bounds = _node_bounds_mm(node)
        if bounds is None:
            continue
        mn = np.asarray(bounds[0], dtype=float)
        mx = np.asarray(bounds[1], dtype=float)
        bodies.append((int(node_id), mn, mx))
        smallest = float(np.min(mx - mn))
        if smallest > 0.0:
            min_cell_dim_mm = min(min_cell_dim_mm, smallest)
    if not bodies:
        return []
    # In-plane hash bucket = smallest cell dimension, so a uniform grid maps each
    # face to a single bucket (O(1) neighbour lookup) and larger octree faces span
    # only as many buckets as the small neighbours that could cover them.
    bucket_mm = max(min_cell_dim_mm if np.isfinite(min_cell_dim_mm) else 1.0, 1.0e-6)

    def _bucket_range(low: float, high: float) -> range:
        return range(int(math.floor(low / bucket_mm)), int(math.floor((high - _TOLERANCE_MM) / bucket_mm)) + 1)

    faces: dict[tuple[int, int, str], dict] = {}
    # (axis, plane coordinate) -> in-plane bucket -> list of face keys at that plane.
    plane_hash: dict[tuple[int, int], dict[tuple[int, int], list[tuple[int, int, str]]]] = {}
    for node_id, mn, mx in bodies:
        for axis in range(3):
            other = [i for i in range(3) if i != axis]
            interval_a = (float(mn[other[0]]), float(mx[other[0]]))
            interval_b = (float(mn[other[1]]), float(mx[other[1]]))
            full_area = (interval_a[1] - interval_a[0]) * (interval_b[1] - interval_b[0])
            for side, coord in (("lo", float(mn[axis])), ("hi", float(mx[axis]))):
                center = np.empty(3, dtype=float)
                center[axis] = coord
                center[other[0]] = 0.5 * (interval_a[0] + interval_a[1])
                center[other[1]] = 0.5 * (interval_b[0] + interval_b[1])
                key = (node_id, axis, side)
                faces[key] = {
                    "full": full_area,
                    "covered": 0.0,
                    "center": center,
                    "axis": axis,
                    "side": side,
                    "interval_a": interval_a,
                    "interval_b": interval_b,
                    "other": other,
                }
                plane_key = (axis, round(coord / _TOLERANCE_MM))
                buckets = plane_hash.setdefault(plane_key, {})
                for ba in _bucket_range(interval_a[0], interval_a[1]):
                    for bb in _bucket_range(interval_b[0], interval_b[1]):
                        buckets.setdefault((ba, bb), []).append(key)

    # A face is covered where an OPPOSING-side face at the same plane overlaps it
    # in-plane. Only faces sharing an in-plane bucket can overlap, so this is
    # O(faces) for a voxel grid instead of the O(faces^2)-per-plane pairwise scan.
    for key, face in faces.items():
        node_id, axis, side = key
        opposite = "hi" if side == "lo" else "lo"
        buckets = plane_hash.get((axis, round(face["center"][axis] / _TOLERANCE_MM)))
        if not buckets:
            continue
        interval_a = face["interval_a"]
        interval_b = face["interval_b"]
        seen: set[tuple[int, int, str]] = set()
        for ba in _bucket_range(interval_a[0], interval_a[1]):
            for bb in _bucket_range(interval_b[0], interval_b[1]):
                for other_key in buckets.get((ba, bb), ()):
                    if other_key[2] != opposite or other_key[0] == node_id or other_key in seen:
                        continue
                    seen.add(other_key)
                    other = faces[other_key]
                    overlap_a = min(interval_a[1], other["interval_a"][1]) - max(interval_a[0], other["interval_a"][0])
                    overlap_b = min(interval_b[1], other["interval_b"][1]) - max(interval_b[0], other["interval_b"][0])
                    if overlap_a > _TOLERANCE_MM and overlap_b > _TOLERANCE_MM:
                        face["covered"] += overlap_a * overlap_b

    patches: list[SurfacePatch] = []
    for (node_id, axis, side), face in faces.items():
        exposed_mm2 = face["full"] - face["covered"]
        if exposed_mm2 <= face["full"] * float(min_exposed_fraction):
            continue
        other = face["other"]
        interval_a = face["interval_a"]
        interval_b = face["interval_b"]
        coord = face["center"][axis]
        corners_mm = []
        for da, db in ((0, 0), (1, 0), (1, 1), (0, 1)):
            corner = np.empty(3, dtype=float)
            corner[axis] = coord
            corner[other[0]] = interval_a[da]
            corner[other[1]] = interval_b[db]
            corners_mm.append(corner)
        normal = np.zeros(3, dtype=float)
        normal[axis] = 1.0 if side == "hi" else -1.0
        node = model.nodes[int(node_id)]
        patches.append(
            SurfacePatch(
                center=face["center"] * 1.0e-3,
                normal=normal,
                corners=np.asarray(corners_mm, dtype=float) * 1.0e-3,
                area=float(exposed_mm2) * 1.0e-6,
                emissivity=max(1.0e-6, float(getattr(node, "emissivity", 0.0) or 0.0)),
                group_id=int(node_id),
            )
        )
    return patches


def _orientation_key(normal: np.ndarray) -> tuple[int, int, int]:
    n = np.asarray(normal, dtype=float)
    axis = int(np.argmax(np.abs(n)))
    key = [0, 0, 0]
    key[axis] = 1 if n[axis] >= 0.0 else -1
    return tuple(key)


def _choose_bin_size_mm(patches: list[SurfacePatch], target_super_surfaces: int) -> float:
    """Pick the finest spatial bin whose (bin x orientation) super-surface count
    stays within ``target_super_surfaces`` (more bins = more spatial resolution)."""
    centers_mm = np.array([p.center for p in patches], dtype=float) * 1.0e3
    span = float(np.max(np.linalg.norm(centers_mm - centers_mm.mean(axis=0), axis=1))) or 1.0
    extent = float(np.max(centers_mm.max(axis=0) - centers_mm.min(axis=0))) or span
    orientations = {_orientation_key(p.normal) for p in patches}
    best = extent  # coarsest: one bin
    for divisions in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128):
        bin_size = extent / divisions
        keys = {
            (tuple(np.floor(np.asarray(p.center) * 1.0e3 / bin_size).astype(int)), _orientation_key(p.normal))
            for p in patches
        }
        if len(keys) <= int(target_super_surfaces):
            best = bin_size
        else:
            break
    # Never finer than a single face and never so coarse it collapses orientations.
    return max(best, 1.0e-6) if len(orientations) else extent


def group_patches_to_super_surfaces(
    patches: list[SurfacePatch],
    *,
    target_super_surfaces: int = 2000,
) -> tuple[list[SurfacePatch], list[dict[int, float]]]:
    """Collapse exposed-face patches into super-surfaces (spatial bin x outward
    orientation) so the ray trace and exchange-area solve run at a tractable
    size. Returns (super_patches, members) where members[k] maps owning graph
    node -> that node's face area in super-surface k (used to aggregate node
    temperatures to the super and distribute exchange power back)."""
    if not patches:
        return [], []
    bin_size_mm = _choose_bin_size_mm(patches, target_super_surfaces)
    buckets: dict[tuple, list[int]] = {}
    for index, patch in enumerate(patches):
        bin_key = tuple(np.floor(np.asarray(patch.center, dtype=float) * 1.0e3 / bin_size_mm).astype(int))
        buckets.setdefault((bin_key, _orientation_key(patch.normal)), []).append(index)

    super_patches: list[SurfacePatch] = []
    members: list[dict[int, float]] = []
    for super_index, ((_bin, orientation), patch_indices) in enumerate(sorted(buckets.items())):
        group = [patches[i] for i in patch_indices]
        areas = np.array([p.area for p in group], dtype=float)
        total_area = float(areas.sum())
        if total_area <= 0.0:
            continue
        centers = np.array([p.center for p in group], dtype=float)
        centroid = (centers * areas[:, None]).sum(axis=0) / total_area
        emissivity = float((np.array([p.emissivity for p in group]) * areas).sum() / total_area)
        normal = np.array(orientation, dtype=float)
        # Represent the super-surface as one square patch (side = sqrt(total area))
        # at the centroid, facing the group's orientation -- coarse geometry for
        # the (fast) ray trace; the member map preserves the true per-node areas.
        axis = int(np.argmax(np.abs(normal)))
        tangent = [i for i in range(3) if i != axis]
        half = 0.5 * float(np.sqrt(total_area))
        corners = []
        for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            corner = centroid.copy()
            corner[tangent[0]] += du * half
            corner[tangent[1]] += dv * half
            corners.append(corner)
        super_patches.append(
            SurfacePatch(
                center=centroid,
                normal=normal / max(np.linalg.norm(normal), 1e-300),
                corners=np.asarray(corners, dtype=float),
                area=total_area,
                emissivity=emissivity,
                group_id=super_index,
            )
        )
        member_area: dict[int, float] = {}
        for patch in group:
            member_area[int(patch.group_id)] = member_area.get(int(patch.group_id), 0.0) + float(patch.area)
        members.append(member_area)
    return super_patches, members


def compute_radiation_exchange_links(
    model,
    *,
    rays_per_patch: int = 4000,
    seed: int = 0,
    max_patches: int = 4000,
) -> tuple[list[tuple[int, int, float]], dict[int, float], dict[str, float]]:
    """Ray-trace view factors over the model's exposed faces and return node-level
    radiative exchange links, per-node environment exchange area, and diagnostics.

    Returns ([], {}, diagnostics) when there is nothing to couple or the patch
    count exceeds ``max_patches`` (the numpy ray engine is not meant for very
    large assemblies -- swap in Embree/grouping for those)."""
    patches = exposed_face_patches(model)
    diagnostics = {"patches": float(len(patches))}
    if len(patches) < 2:
        diagnostics["skipped"] = 1.0
        return [], {}, diagnostics
    if len(patches) > int(max_patches):
        diagnostics["skipped_too_many_patches"] = 1.0
        return [], {}, diagnostics
    view_factors, view_factors_env = compute_view_factors(patches, rays_per_patch=rays_per_patch, seed=seed)
    emissivity = np.array([p.emissivity for p in patches], dtype=float)
    area = np.array([p.area for p in patches], dtype=float)
    exchange, exchange_env = total_exchange_areas(view_factors, view_factors_env, emissivity, area)
    links, env_by_group = exchange_links_by_group(patches, exchange, exchange_env)
    diagnostics["links"] = float(len(links))
    diagnostics["mean_env_view_fraction"] = float(np.mean(view_factors_env))
    return links, env_by_group, diagnostics


def _attach_coupling(model, members, links, env_fraction) -> None:
    model.radiation_super_members = members
    model.radiation_super_links = links
    model.radiation_env_fraction_by_node = env_fraction


def apply_radiation_coupling(
    model,
    *,
    target_super_surfaces: int = 2000,
    rays_per_patch: int = 4000,
    seed: int = 0,
    max_patches: int = 500000,
    use_cache: bool = True,
    cache_dir: Path | None = None,
    assume_enclosure_vacuum: bool = True,
    exterior_view_threshold: float = 0.5,
    progress_callback=None,
) -> dict[str, float]:
    """Compute surface-to-surface radiative coupling and attach it to the model in
    factored (grouped) form, consumed by ``prepare_simulation``:

      * ``model.radiation_super_members`` -- list (per super-surface) of
        {node_id: face area in this super-surface};
      * ``model.radiation_super_links`` -- list of (super_i, super_j, G_ij)
        exchange areas between super-surfaces.

    Exposed faces are first collapsed into <= ``target_super_surfaces`` super-
    surfaces, then view factors + gray exchange areas are solved at that (small)
    size and the coupling is applied in the solver as aggregate-exchange-distribute
    (three sparse matvecs) so it scales to large graphs. Returns diagnostics.

    The result is cached on disk keyed by a content hash of the geometry +
    emissivities + trace params, so re-preparing the same graph is instant.
    ``progress_callback(stage, fraction)`` is invoked at each stage if given."""

    def report(stage: str, fraction: float) -> None:
        if progress_callback is not None:
            try:
                progress_callback(stage, float(fraction))
            except Exception:  # noqa: BLE001
                pass

    cache_directory = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
    cache_key = None
    if use_cache:
        try:
            cache_key = _coupling_cache_key(
                model, target_super_surfaces, rays_per_patch, seed,
                assume_enclosure_vacuum, exterior_view_threshold,
            )
            cached = _load_coupling_cache(cache_directory / f"{cache_key}.json")
        except Exception:  # noqa: BLE001
            cached = None
        if cached is not None:
            members, links, env_fraction, diagnostics = cached
            _attach_coupling(model, members, links, env_fraction)
            diagnostics["cache_hit"] = 1.0
            report("cached", 1.0)
            return diagnostics

    report("extract", 0.0)
    patches = exposed_face_patches(model)
    diagnostics = {"patches": float(len(patches))}
    if len(patches) < 2:
        diagnostics["skipped"] = 1.0
        return diagnostics
    if len(patches) > int(max_patches):
        diagnostics["skipped_too_many_patches"] = 1.0
        return diagnostics
    report("group", 0.2)
    super_patches, members = group_patches_to_super_surfaces(patches, target_super_surfaces=target_super_surfaces)
    diagnostics["super_surfaces"] = float(len(super_patches))
    if len(super_patches) < 2:
        diagnostics["skipped"] = 1.0
        return diagnostics
    report("view_factors", 0.4)
    view_factors, view_factors_env = compute_view_factors(super_patches, rays_per_patch=rays_per_patch, seed=seed)
    report("exchange", 0.8)
    emissivity = np.array([p.emissivity for p in super_patches], dtype=float)
    area = np.array([p.area for p in super_patches], dtype=float)
    # Interior/exterior classification: a super-surface that sees mostly open sky
    # (escaped fraction above the threshold) is an exterior skin surface and keeps
    # its escape -> radiates to the ambient environment; one that sees mostly other
    # surfaces is enclosed (interior) and is treated as a vacuum (escape
    # redistributed into its coupling, no sink). Vacuum classification is disabled
    # entirely when assume_enclosure_vacuum is False (all escape -> ambient sink).
    if assume_enclosure_vacuum:
        interior_mask = np.asarray(view_factors_env, dtype=float) <= float(exterior_view_threshold)
    else:
        interior_mask = np.zeros(len(super_patches), dtype=bool)
    exchange, _exchange_env = total_exchange_areas(
        view_factors, view_factors_env, emissivity, area, interior_mask=interior_mask
    )
    links: list[tuple[int, int, float]] = []
    for i in range(len(super_patches)):
        for j in range(i + 1, len(super_patches)):
            if exchange[i, j] > 0.0:
                links.append((i, j, float(exchange[i, j])))
    # Per-node ambient fraction: exterior surfaces radiate their escaped fraction
    # to the environment; interior (vacuum) surfaces radiate none (0). Scales each
    # node's ambient term so radiation is not double-counted (background + coupling).
    env_numerator: dict[int, float] = {}
    env_denominator: dict[int, float] = {}
    for super_index, member_area in enumerate(members):
        env_fraction = 0.0 if bool(interior_mask[super_index]) else float(view_factors_env[super_index])
        for node_id, node_area in member_area.items():
            env_numerator[node_id] = env_numerator.get(node_id, 0.0) + node_area * env_fraction
            env_denominator[node_id] = env_denominator.get(node_id, 0.0) + node_area
    env_fraction_by_node = {
        node_id: (env_numerator[node_id] / env_denominator[node_id])
        for node_id in env_denominator
        if env_denominator[node_id] > 0.0
    }
    diagnostics["interior_super_surfaces"] = float(int(np.count_nonzero(interior_mask)))
    diagnostics["exterior_super_surfaces"] = float(len(super_patches) - int(np.count_nonzero(interior_mask)))
    _attach_coupling(model, members, links, env_fraction_by_node)
    diagnostics["super_links"] = float(len(links))
    diagnostics["mean_env_view_fraction"] = float(np.mean(view_factors_env))
    diagnostics["cache_hit"] = 0.0
    if use_cache and cache_key is not None:
        _save_coupling_cache(cache_directory / f"{cache_key}.json", members, links, env_fraction_by_node, diagnostics)
    report("done", 1.0)
    return diagnostics
