"""Ray-traced view factors and gray-diffuse exchange areas for radiative coupling.

This is the geometry half of surface-to-surface radiation (the solver half lives
in ``simulation_model`` / ``thermal_validation``). It computes, for a set of
oriented surface patches:

  * geometric view factors ``F_ij`` (fraction of diffuse emission from patch i
    that reaches patch j), via Monte-Carlo ray casting with occlusion; and
  * total exchange areas ``G_ij = A_i * script-F_ij`` [m^2] for gray-diffuse,
    opaque surfaces (multiple reflections folded in), which the solver consumes
    as ``radiation_exchange_links`` (net power = sigma * sum_j G_ij (T_j^4-T_i^4)).

The view-factor estimator emits cosine-weighted (Lambertian) rays from each patch
and tallies the first patch each ray hits; rays that escape feed the environment
fraction ``F_i,env``. Occlusion/self-shielding is handled automatically by taking
the nearest hit. Reciprocity (A_i F_ij = A_j F_ji) is enforced by symmetrizing.

The exchange-area extraction is the enclosure matrix method: with radiosity
J = (I - diag(rho) F)^-1 (diag(eps) Eb + diag(rho) F_env Eb_env), the net heat
Q = diag(A)((I-F)J - F_env Eb_env) is linear in Eb and has zero row sums, so it
is a graph Laplacian on Eb = sigma T^4 whose off-diagonals are the exchange areas.
For two surfaces with F=1 this reduces to the textbook A/(1/e1 + 1/e2 - 1).

Backend note: intersection uses trimesh's pure-numpy ``ray_triangle`` engine,
which is fine for validation and moderate patch counts. For large assemblies swap
in an Embree-backed intersector (``trimesh.ray.ray_pyembree``) or a BVH; the API
here is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SurfacePatch:
    """A flat quadrilateral radiating surface element.

    corners are 4 points (CCW); ``normal`` is the outward unit normal (the side
    that radiates). ``group_id`` identifies the graph node/super-surface this
    patch belongs to, so patch-level exchange can be aggregated to node level.
    """

    center: np.ndarray
    normal: np.ndarray
    corners: np.ndarray  # (4, 3)
    area: float
    emissivity: float = 1.0
    group_id: int = -1


def _orthonormal_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=float)
    n = n / max(np.linalg.norm(n), 1e-300)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u = u / max(np.linalg.norm(u), 1e-300)
    v = np.cross(n, u)
    return u, v, n


def _sample_points_on_quad(corners: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    c0, c1, c2, c3 = (np.asarray(corners, dtype=float)[k] for k in range(4))
    s = rng.random(count)[:, None]
    t = rng.random(count)[:, None]
    return (1 - s) * (1 - t) * c0 + s * (1 - t) * c1 + s * t * c2 + (1 - s) * t * c3


def _cosine_weighted_directions(normal: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    u, v, n = _orthonormal_basis(normal)
    r1 = rng.random(count)
    r2 = rng.random(count)
    phi = 2.0 * np.pi * r1
    cos_theta = np.sqrt(1.0 - r2)  # pdf ~ cos(theta)/pi (Lambertian)
    sin_theta = np.sqrt(r2)
    x = np.cos(phi) * sin_theta
    y = np.sin(phi) * sin_theta
    z = cos_theta
    return x[:, None] * u + y[:, None] * v + z[:, None] * n


def compute_view_factors(
    patches: list[SurfacePatch],
    rays_per_patch: int = 20000,
    seed: int = 0,
    enforce_reciprocity: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo view factors among ``patches`` with occlusion.

    Returns (F, F_env): F is a (P, P) matrix with F[i, j] the fraction of diffuse
    emission from patch i reaching patch j (F[i, i] = 0); F_env[i] is the fraction
    escaping to the environment (rays hitting nothing). Rows satisfy
    sum_j F[i, j] + F_env[i] = 1.
    """
    import trimesh

    n_patches = len(patches)
    if n_patches == 0:
        return np.zeros((0, 0)), np.zeros(0)

    vertices: list[np.ndarray] = []
    faces: list[list[int]] = []
    triangle_patch: list[int] = []
    for patch_index, patch in enumerate(patches):
        base = len(vertices)
        vertices.extend(np.asarray(patch.corners, dtype=float))
        faces.append([base, base + 1, base + 2])
        faces.append([base, base + 2, base + 3])
        triangle_patch.extend((patch_index, patch_index))
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=int), process=False)
    triangle_patch_arr = np.asarray(triangle_patch, dtype=int)

    rng = np.random.default_rng(seed)
    F = np.zeros((n_patches, n_patches), dtype=float)
    F_env = np.zeros(n_patches, dtype=float)

    def _tally(origins: np.ndarray, directions: np.ndarray, source: np.ndarray) -> None:
        hit_triangle = np.asarray(mesh.ray.intersects_first(origins, directions), dtype=int)
        hit_patch = np.where(hit_triangle >= 0, triangle_patch_arr[np.clip(hit_triangle, 0, None)], -1)
        for i in np.unique(source):
            hits = hit_patch[source == i]
            # Exclude residual (spurious) self-hits from the ray budget.
            valid_rays = float(max(int(np.count_nonzero(hits != i)), 1))
            landed = hits[hits >= 0]
            if landed.size:
                counts = np.bincount(landed, minlength=n_patches).astype(float)
                counts[i] = 0.0
                F[i] += counts / valid_rays
            F_env[i] += float(np.count_nonzero(hits == -1)) / valid_rays

    # Cast rays in large batches (one intersector call per ~1M rays) rather than
    # once per patch, so Embree throughput dominates and the Python per-call
    # overhead is amortized. Each patch's rays stay within a single batch so its
    # view-factor tally is normalized over its full ray budget.
    ray_batch = max(int(rays_per_patch), 1_000_000)
    buf_origins: list[np.ndarray] = []
    buf_dirs: list[np.ndarray] = []
    buf_src: list[np.ndarray] = []
    buffered = 0
    for i, patch in enumerate(patches):
        if buffered + rays_per_patch > ray_batch and buf_origins:
            _tally(np.concatenate(buf_origins), np.concatenate(buf_dirs), np.concatenate(buf_src))
            buf_origins, buf_dirs, buf_src, buffered = [], [], [], 0
        normal = np.asarray(patch.normal, dtype=float)
        normal = normal / max(np.linalg.norm(normal), 1e-300)
        # Push the ray origin off the surface so it does not self-intersect at
        # t~0. The offset scales with the patch size (a scene-diagonal offset
        # collapses for small/mm-scale scenes and every ray then self-hits), with
        # a small absolute floor to stay above the intersector's tolerance.
        offset = max(1.0e-3 * float(np.sqrt(max(patch.area, 0.0))), 1.0e-6)
        buf_origins.append(_sample_points_on_quad(patch.corners, rays_per_patch, rng) + offset * normal)
        buf_dirs.append(_cosine_weighted_directions(patch.normal, rays_per_patch, rng))
        buf_src.append(np.full(rays_per_patch, i, dtype=int))
        buffered += rays_per_patch
    if buf_origins:
        _tally(np.concatenate(buf_origins), np.concatenate(buf_dirs), np.concatenate(buf_src))

    if enforce_reciprocity:
        areas = np.array([p.area for p in patches], dtype=float)
        s = areas[:, None] * F  # A_i F_ij, should be symmetric
        s = 0.5 * (s + s.T)
        with np.errstate(divide="ignore", invalid="ignore"):
            F = np.where(areas[:, None] > 0.0, s / areas[:, None], 0.0)
        np.fill_diagonal(F, 0.0)
        # Recompute the environment fraction as the (nonnegative) closure remainder.
        F_env = np.clip(1.0 - F.sum(axis=1), 0.0, 1.0)
    return F, F_env


def total_exchange_areas(
    F: np.ndarray,
    F_env: np.ndarray,
    emissivity: np.ndarray,
    area: np.ndarray,
    close_enclosure: bool = False,
    interior_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gray-diffuse total exchange areas from view factors (enclosure matrix method).

    Returns (G, G_env): G is a (P, P) symmetric matrix of surface-surface exchange
    areas [m^2] (G[i, i] = 0); G_env[i] is the surface-to-environment exchange area.
    Net radiative power INTO surface i is sigma * (sum_j G[i, j] (T_j^4 - T_i^4)
    + G_env[i] (T_env^4 - T_i^4)). For two surfaces with F=1 this yields the
    textbook A / (1/e1 + 1/e2 - 1).

    Surfaces flagged in ``interior_mask`` (or all of them with
    ``close_enclosure=True``) are treated as a vacuum enclosure: their escaped
    view-factor fraction is redistributed into the surfaces they see (row
    renormalized to sum to 1) and their G_env is zeroed -- no background sink.
    Unflagged (exterior) surfaces keep their escaped fraction and so radiate to
    the ambient environment.
    """
    F = np.asarray(F, dtype=float).copy()
    F_env = np.asarray(F_env, dtype=float).reshape(-1).copy()
    eps = np.clip(np.asarray(emissivity, dtype=float).reshape(-1), 1e-6, 1.0)
    area = np.asarray(area, dtype=float).reshape(-1)
    n = F.shape[0]
    if interior_mask is not None:
        mask = np.asarray(interior_mask, dtype=bool).reshape(-1)
    else:
        mask = np.ones(n, dtype=bool) if close_enclosure else np.zeros(n, dtype=bool)
    if mask.any():
        # Interior (vacuum) surfaces: redistribute their escaped fraction into the
        # surfaces they see (row sums to 1) and drop the sink. Interior surfaces
        # that see nothing stay uncoupled (radiate into vacuum, receive nothing).
        row_sums = F.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(mask & (row_sums > 1e-12), 1.0 / row_sums, 1.0)
        F = F * scale[:, None]
        F_env = np.where(mask, 0.0, F_env)
    rho = 1.0 - eps
    identity = np.eye(n)
    inverse = np.linalg.inv(identity - rho[:, None] * F)  # (I - diag(rho) F)^-1
    reflected = (identity - F) @ inverse
    m_surface = area[:, None] * (reflected * eps[None, :])          # diag(A)(I-F)inv diag(eps)
    m_env = area * (reflected @ (rho * F_env)) - area * F_env       # coefficient of Eb_env
    exchange = -m_surface
    np.fill_diagonal(exchange, 0.0)
    exchange = 0.5 * (exchange + exchange.T)
    exchange = np.clip(exchange, 0.0, None)
    exchange_env = np.clip(-m_env, 0.0, None)
    return exchange, exchange_env


def parallel_rectangles_view_factor(a: float, b: float, c: float) -> float:
    """Analytic view factor between two directly-opposed, aligned rectangles
    (sides a, b) separated by distance c. Standard closed form (Modest/Incropera)."""
    x = float(a) / float(c)
    y = float(b) / float(c)
    term = np.log(np.sqrt((1 + x**2) * (1 + y**2) / (1 + x**2 + y**2)))
    term += x * np.sqrt(1 + y**2) * np.arctan(x / np.sqrt(1 + y**2))
    term += y * np.sqrt(1 + x**2) * np.arctan(y / np.sqrt(1 + x**2))
    term -= x * np.arctan(x) + y * np.arctan(y)
    return float((2.0 / (np.pi * x * y)) * term)


def axis_aligned_square_patch(
    center: tuple[float, float, float],
    normal_axis: str,
    normal_sign: float,
    side: float,
    *,
    emissivity: float = 1.0,
    group_id: int = -1,
) -> SurfacePatch:
    """Build a square patch of the given side length centered at ``center`` whose
    outward normal is +/- one coordinate axis. Convenience for tests/enclosures."""
    axes = {"x": 0, "y": 1, "z": 2}
    axis = axes[normal_axis]
    tangent_axes = [i for i in range(3) if i != axis]
    normal = np.zeros(3)
    normal[axis] = float(normal_sign)
    half = 0.5 * float(side)
    center = np.asarray(center, dtype=float)
    corners = []
    for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        corner = center.copy()
        corner[tangent_axes[0]] += du * half
        corner[tangent_axes[1]] += dv * half
        corners.append(corner)
    return SurfacePatch(
        center=center,
        normal=normal,
        corners=np.asarray(corners, dtype=float),
        area=float(side) ** 2,
        emissivity=float(emissivity),
        group_id=int(group_id),
    )


def exchange_links_by_group(
    patches: list[SurfacePatch],
    exchange: np.ndarray,
    exchange_env: np.ndarray | None = None,
) -> tuple[list[tuple[int, int, float]], dict[int, float]]:
    """Aggregate patch-level exchange areas to graph-node (group) level.

    Returns (links, env_by_group): ``links`` is the list of (group_i, group_j,
    G_ij) the solver stores as ``radiation_exchange_links`` (exchange areas summed
    over patch pairs that map to the same group pair; self pairs dropped), and
    ``env_by_group`` maps group -> total exchange area to the environment."""
    group_ids = [int(p.group_id) for p in patches]
    pair_totals: dict[tuple[int, int], float] = {}
    for i in range(len(patches)):
        for j in range(i + 1, len(patches)):
            gi, gj = group_ids[i], group_ids[j]
            if gi == gj:
                continue
            key = (gi, gj) if gi < gj else (gj, gi)
            pair_totals[key] = pair_totals.get(key, 0.0) + float(exchange[i, j])
    links = [(gi, gj, g) for (gi, gj), g in pair_totals.items() if g > 0.0]
    env_by_group: dict[int, float] = {}
    if exchange_env is not None:
        for i, group in enumerate(group_ids):
            env_by_group[group] = env_by_group.get(group, 0.0) + float(exchange_env[i])
    return links, env_by_group
