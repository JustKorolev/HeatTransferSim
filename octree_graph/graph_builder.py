"""Build thermal graph nodes and conductance edges from octree leaves."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterator

import numpy as np

from .load_gltf import MeshObject
from .load_contact_report import ContactReport
from .materials import DEFAULT_ASSIGNED_MATERIAL_NAME, Material, resolve_material
from .octree import OctreeCell, _physical_material_name


_DEFAULT_ROLE_CONTACT_G_W_K = 0.1
_ROLE_NODE_CONTACT_TOLERANCE_MM = 1.0e-6
DEFAULT_HEATER_NAME_PATTERNS = [
    r"heater",
    r"heat[_\s-]*strip",
    r"cartridge",
    r"kapton",
    r"resistor[_\s-]*heater",
]
DEFAULT_SENSOR_NAME_PATTERNS = [
    r"sensor",
    r"thermistor",
    r"\brtd\b",
    r"\bdiode\b",
    r"thermometer",
    r"temperature[_\s-]*probe",
    r"temp[_\s-]*sensor",
]
DEFAULT_ROLE_EXCLUDE_NAME_PATTERNS = [
    r"flex[_\s-]*cable",
    r"(?:^|_)cable(?:_|$)",
    r"(?:^|_)wire(?:_|$)",
    r"(?:^|_)harness(?:_|$)",
    r"(?:^|_)connector(?:_|$)",
    r"(?:^|_)breakout(?:_|$)",
    r"(?:^|_)pcb(?:_|$)",
    r"(?:^|_)board(?:_|$)",
]
DEFAULT_ROLE_GROUP_GAP_MM = 10.0


@dataclass
class RoleComponent:
    name: str
    kind: str
    objects: list[MeshObject]
    material_name: str | None = None

    @property
    def bounds_mm(self) -> tuple[np.ndarray, np.ndarray]:
        mins = np.min([np.asarray(obj.bounds_mm[0], dtype=float) for obj in self.objects], axis=0)
        maxs = np.max([np.asarray(obj.bounds_mm[1], dtype=float) for obj in self.objects], axis=0)
        return mins, maxs

    @property
    def center_mm(self) -> np.ndarray:
        mins, maxs = self.bounds_mm
        return (mins + maxs) * 0.5

    @property
    def size_mm(self) -> np.ndarray:
        mins, maxs = self.bounds_mm
        return np.maximum(maxs - mins, 0.0)


@dataclass
class GraphBuildResult:
    nodes: list[dict]
    edges: list[dict]
    warnings: list[str]


def build_graph(
    leaves: list[OctreeCell],
    contact_report: ContactReport | None,
    materials: dict[str, Material],
    warnings: list[str],
    default_contact_G_W_K: float = 0.1,
    radiation_reference_temperature_K: float = 293.15,
    contact_detection_distance_mm: float = 0.0,
    component_bounds_mm: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    role_components: list[RoleComponent] | None = None,
    role_contact_tolerance_mm: float = _ROLE_NODE_CONTACT_TOLERANCE_MM,
) -> GraphBuildResult:
    contact_report = contact_report or ContactReport()
    solid = [cell for cell in leaves if not cell.is_empty]
    exposed_areas_m2 = _exposed_areas_m2(solid)
    nodes: list[dict] = []
    for node_id, cell in enumerate(solid):
        material = resolve_material(cell.dominant_material, materials, warnings)
        radiating_area_m2 = exposed_areas_m2.get(cell.cell_id, 0.0)
        G_rad = (
            4.0
            * material.emissivity
            * 5.670374419e-8
            * radiating_area_m2
            * float(radiation_reference_temperature_K) ** 3
        )
        component = cell.dominant_component or ""
        occupancy_fraction = max(cell.occupancy.values(), default=0.0)
        mass = material.density_kg_m3 * cell.volume_m3 * occupancy_fraction
        nodes.append(
            {
                "node_id": node_id,
                "cell_id": cell.cell_id,
                "center_mm": list(cell.center_mm),
                "size_mm": list(cell.size_mm),
                "level": cell.level,
                "component_name": component,
                "material_name": material.name,
                "volume_m3": cell.volume_m3,
                "mass_kg": mass,
                "C_J_K": mass * material.cp_J_kgK,
                "initial_temperature_K": 293.15,
                "occupancy_fraction": occupancy_fraction,
                "is_heater": False,
                "is_sensor": False,
                "confidence": cell.confidence,
                "warnings": list(cell.warnings),
                "radiation": {
                    "is_exposed": radiating_area_m2 > 0.0,
                    "radiating_area_m2": float(radiating_area_m2),
                    "emissivity": float(material.emissivity),
                    "G_rad_W_K": float(G_rad),
                    "R_rad_K_W": float(1.0 / G_rad) if G_rad > 0.0 else None,
                },
                "tags": {"notes": ""},
            }
        )
    cell_to_node = {node["cell_id"]: node for node in nodes}
    role_nodes = _append_role_nodes(
        nodes,
        role_components or [],
        contact_report,
        materials,
        warnings,
    )
    edges: list[dict] = []
    edge_index = 0
    connected_pairs: set[tuple[str, str]] = set()
    connected_node_pairs: set[tuple[int, int]] = set()
    for a, b in _candidate_cell_pairs(solid, 0.0):
        area_mm2, distance_mm = _shared_face_area_and_distance(a, b)
        if area_mm2 <= 0.0:
            continue
        node_a = cell_to_node[a.cell_id]
        node_b = cell_to_node[b.cell_id]
        material_a = resolve_material(str(node_a["material_name"]), materials, warnings)
        material_b = resolve_material(str(node_b["material_name"]), materials, warnings)
        same_component = node_a["component_name"] == node_b["component_name"]
        if same_component:
            edge_type = "internal_conduction"
            k_eff = harmonic_mean(material_a.k_W_mK, material_b.k_W_mK)
            G = k_eff * (area_mm2 * 1.0e-6) / max(distance_mm * 1.0e-3, 1.0e-12)
            source = "geometry"
            confidence = "high"
        elif node_a["material_name"] == node_b["material_name"]:
            edge_type = "same_material_spatial"
            k_eff = material_a.k_W_mK
            G = k_eff * (area_mm2 * 1.0e-6) / max(distance_mm * 1.0e-3, 1.0e-12)
            source = "geometry"
            confidence = "medium"
        else:
            edge_type = "uncertain_contact"
            G = default_contact_G_W_K
            source = "geometry"
            confidence = "low"
        edges.append(
            {
                "edge_id": f"edge_{edge_index}",
                "node_i": int(node_a["node_id"]),
                "node_j": int(node_b["node_id"]),
                "edge_type": edge_type,
                "G_W_K": float(G),
                "shared_area_m2": float(area_mm2 * 1.0e-6),
                "distance_m": float(distance_mm * 1.0e-3),
                "contact_confidence": confidence,
                "source": source,
                "warnings": [] if confidence != "low" else ["Inter-part geometry adjacency has not been contact-classified."],
            }
        )
        connected_pairs.add(_cell_pair_key(a, b))
        connected_node_pairs.add(_node_pair_key(node_a, node_b))
        edge_index += 1
    if contact_detection_distance_mm > 0.0:
        edge_index = _add_near_contact_edges(
            solid,
            cell_to_node,
            connected_pairs,
            edges,
            edge_index,
            materials,
            warnings,
            default_contact_G_W_K,
            contact_detection_distance_mm,
            connected_node_pairs,
        )
    edge_index = _add_role_node_contact_edges(
        solid,
        cell_to_node,
        role_nodes,
        edges,
        edge_index,
        contact_detection_distance_mm,
        connected_node_pairs,
        warnings,
        role_contact_tolerance_mm,
    )
    return GraphBuildResult(nodes=nodes, edges=edges, warnings=warnings)


def classify_role_component_name(
    name: str,
    heater_patterns: list[str],
    sensor_patterns: list[str],
    exclude_patterns: list[str] | None = None,
) -> str | None:
    normalized = _normalize_role_name(name)
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in exclude_patterns or []):
        return None
    heater = any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in heater_patterns)
    sensor = any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in sensor_patterns)
    if heater and sensor:
        raise ValueError(f"CAD component name matches both heater and sensor detection patterns: {name!r}")
    if heater:
        return "heater"
    if sensor:
        return "sensor"
    return None


def collapse_role_components(
    objects: list[MeshObject],
    heater_patterns: list[str],
    sensor_patterns: list[str],
    exclude_patterns: list[str] | None = None,
    group_gap_mm: float = DEFAULT_ROLE_GROUP_GAP_MM,
) -> tuple[list[MeshObject], list[RoleComponent]]:
    body_objects: list[MeshObject] = []
    groups: dict[tuple[str, str], list[MeshObject]] = {}
    for obj in objects:
        kind = classify_role_component_name(
            _object_search_text(obj),
            heater_patterns,
            sensor_patterns,
            exclude_patterns,
        )
        if kind is None:
            body_objects.append(obj)
            continue
        groups.setdefault((kind, _role_group_name(obj.name)), []).append(obj)
    components: list[RoleComponent] = []
    for (kind, name), members in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        clusters = _spatial_role_clusters(members, group_gap_mm)
        for cluster_index, cluster in enumerate(clusters, start=1):
            component_name = name if len(clusters) == 1 else f"{name}_{cluster_index}"
            components.append(RoleComponent(name=component_name, kind=kind, objects=cluster))
    return body_objects, components


def _normalize_role_name(name: str) -> str:
    return str(name).replace("\\", "/").replace("-", "_").replace(" ", "_")


def _object_search_text(obj: MeshObject) -> str:
    scene_path = getattr(obj, "scene_path", None)
    if scene_path and scene_path != obj.name:
        return f"{scene_path} {obj.name}"
    return obj.name


def _role_group_name(name: str) -> str:
    normalized = _normalize_role_name(name)
    normalized = re.sub(r"(_?\d+)?(_geometry|_mesh|_body|_solid)?$", "", normalized, flags=re.IGNORECASE)
    return normalized or str(name)


def _spatial_role_clusters(members: list[MeshObject], group_gap_mm: float) -> list[list[MeshObject]]:
    if len(members) <= 1:
        return [list(members)]
    max_gap_mm = max(0.0, float(group_gap_mm))
    parent = list(range(len(members)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    bounds = [(np.asarray(obj.bounds_mm[0], dtype=float), np.asarray(obj.bounds_mm[1], dtype=float)) for obj in members]
    for left in range(len(members)):
        for right in range(left + 1, len(members)):
            gap_mm = float(np.linalg.norm(_aabb_gaps_mm(bounds[left][0], bounds[left][1], bounds[right][0], bounds[right][1])))
            if gap_mm <= max_gap_mm:
                union(left, right)
    clusters_by_root: dict[int, list[MeshObject]] = {}
    for index, obj in enumerate(members):
        clusters_by_root.setdefault(find(index), []).append(obj)
    return sorted(
        clusters_by_root.values(),
        key=lambda cluster: (
            float(np.min([np.asarray(obj.bounds_mm[0], dtype=float)[0] for obj in cluster])),
            min(obj.name for obj in cluster),
        ),
    )


def _append_role_nodes(
    nodes: list[dict],
    components: list[RoleComponent],
    contact_report: ContactReport,
    materials: dict[str, Material],
    warnings: list[str],
) -> list[dict]:
    role_nodes: list[dict] = []
    known_materials = set(materials)
    for component in components:
        node_id = len(nodes)
        material_name = _role_component_material_name(component, contact_report, known_materials)
        material = resolve_material(material_name, materials, warnings)
        center_mm = component.center_mm
        size_mm = component.size_mm
        volume_m3 = _role_component_volume_m3(component)
        mass_kg = material.density_kg_m3 * volume_m3
        is_heater = component.kind == "heater"
        is_sensor = component.kind == "sensor"
        node = {
            "node_id": node_id,
            "cell_id": f"{component.kind}_{node_id}",
            "coord": [node_id, 0, 0],
            "center_mm": [float(value) for value in center_mm],
            "size_mm": [float(max(value, 1.0e-6)) for value in size_mm],
            "level": -1,
            "node_type": component.kind,
            "component_name": component.name,
            "material_name": material.name,
            "volume_m3": float(volume_m3),
            "mass_kg": float(mass_kg),
            "C_J_K": float(mass_kg * material.cp_J_kgK),
            "initial_temperature_K": 293.15,
            "occupancy_fraction": 1.0,
            "is_heater": bool(is_heater),
            "is_sensor": bool(is_sensor),
            "confidence": "high",
            "warnings": [f"CAD {component.kind} component collapsed into a dedicated graph node."],
            "radiation": {
                "is_exposed": False,
                "radiating_area_m2": 0.0,
                "emissivity": float(material.emissivity),
                "G_rad_W_K": 0.0,
                "R_rad_K_W": None,
            },
            "tags": {
                "notes": f"Detected from CAD component {component.name!r}.",
            },
            "source_components": [obj.name for obj in component.objects],
            "source_bounds_mm": {
                "min": [float(value) for value in component.bounds_mm[0]],
                "max": [float(value) for value in component.bounds_mm[1]],
            },
        }
        nodes.append(node)
        role_nodes.append(node)
    return role_nodes


def _role_component_material_name(
    component: RoleComponent,
    contact_report: ContactReport,
    known_materials: set[str],
) -> str:
    for obj in component.objects:
        material = _physical_material_name(obj, contact_report, known_materials)
        if material != DEFAULT_ASSIGNED_MATERIAL_NAME:
            return material
    return _physical_material_name(component.objects[0], contact_report, known_materials)


def _role_component_volume_m3(component: RoleComponent) -> float:
    volume = 0.0
    for obj in component.objects:
        mesh_volume = getattr(obj.mesh, "volume", 0.0)
        try:
            mesh_volume = abs(float(mesh_volume))
        except (TypeError, ValueError):
            mesh_volume = 0.0
        if mesh_volume > 0.0:
            # Mesh coordinates are millimeters in this pipeline.
            volume += mesh_volume * 1.0e-9
    if volume > 0.0:
        return float(volume)
    size_mm = np.maximum(component.size_mm, 0.0)
    effective_size_mm = np.maximum(size_mm, 1.0)
    return float(np.prod(effective_size_mm) * 1.0e-9)


def _add_role_node_contact_edges(
    cells: list[OctreeCell],
    cell_to_node: dict[str, dict],
    role_nodes: list[dict],
    edges: list[dict],
    edge_index: int,
    max_gap_mm: float,
    connected_node_pairs: set[tuple[int, int]],
    warnings: list[str],
    role_contact_tolerance_mm: float,
) -> int:
    if not cells or not role_nodes:
        return edge_index
    search_gap_mm = max(0.0, float(role_contact_tolerance_mm))
    for role_node in role_nodes:
        contacts: list[tuple[OctreeCell, float, float, float]] = []
        for cell in cells:
            contact = _node_cell_contact(role_node, cell, search_gap_mm)
            if contact is None:
                continue
            area_mm2, gap_mm, distance_mm = contact
            contacts.append((cell, area_mm2, gap_mm, distance_mm))
        added_edges = 0
        for cell, area_mm2, gap_mm, distance_mm in contacts:
            body_node = cell_to_node.get(cell.cell_id)
            if body_node is None:
                continue
            node_pair = _node_pair_key(role_node, body_node)
            if node_pair in connected_node_pairs:
                continue
            conductance = _DEFAULT_ROLE_CONTACT_G_W_K
            if area_mm2 > 0.0 and distance_mm > 0.0 and role_node["material_name"] == body_node["material_name"]:
                conductance = max(_DEFAULT_ROLE_CONTACT_G_W_K, area_mm2 * 1.0e-6 / max(distance_mm * 1.0e-3, 1.0e-12))
            edges.append(
                {
                    "edge_id": f"edge_{edge_index}",
                    "node_i": int(role_node["node_id"]),
                    "node_j": int(body_node["node_id"]),
                    "edge_type": "role_node_contact",
                    "G_W_K": float(conductance),
                    "shared_area_m2": float(area_mm2 * 1.0e-6),
                    "distance_m": float(distance_mm * 1.0e-3),
                    "contact_confidence": "medium" if gap_mm <= search_gap_mm else "low",
                    "source": "cad_role_node_contact",
                    "warnings": [
                        f"Heater/sensor role node connected to contacting body cell with AABB gap {gap_mm:.3g} mm."
                    ],
                }
            )
            connected_node_pairs.add(node_pair)
            edge_index += 1
            added_edges += 1
        if added_edges == 0:
            nearest = _nearest_role_cell_gaps(role_node, cells, limit=3)
            nearest_text = "; ".join(
                f"{cell.cell_id} gap={gap_mm:.6g} mm center_distance={distance_mm:.6g} mm"
                for cell, gap_mm, distance_mm in nearest
            )
            nearest_clause = f" Nearest body cells: {nearest_text}." if nearest_text else ""
            warning = (
                f"Detected {role_node.get('node_type', 'heater/sensor')} role node "
                f"{role_node.get('node_id')} ({role_node.get('component_name', '?')}) has 0 contact edges; "
                f"it will be thermally isolated in the simulation. "
                f"Role contact tolerance was {search_gap_mm:.6g} mm.{nearest_clause}"
            )
            role_node.setdefault("warnings", []).append(warning)
            warnings.append(warning)
    return edge_index


def _nearest_role_cell_gaps(
    role_node: dict,
    cells: list[OctreeCell],
    limit: int = 3,
) -> list[tuple[OctreeCell, float, float]]:
    ranked = [
        (
            cell,
            _node_cell_gap_mm(role_node, cell),
            _node_cell_center_distance_mm(role_node, cell),
        )
        for cell in cells
    ]
    ranked.sort(key=lambda item: (item[1], item[2], item[0].cell_id))
    return ranked[: max(0, int(limit))]


def _node_cell_contact(
    node: dict,
    cell: OctreeCell,
    max_gap_mm: float,
) -> tuple[float, float, float] | None:
    node_min, node_max = _node_bounds_mm(node)
    cell_min, cell_max = _cell_bounds_mm(cell)
    gaps = _aabb_gaps_mm(node_min, node_max, cell_min, cell_max)
    gap_mm = float(np.linalg.norm(gaps))
    if gap_mm > max_gap_mm:
        return None
    overlaps = np.minimum(node_max, cell_max) - np.maximum(node_min, cell_min)
    separated_axes = [axis for axis, gap in enumerate(gaps) if gap > 1.0e-7]
    if len(separated_axes) > 1:
        return None
    if len(separated_axes) == 1:
        face_axis = separated_axes[0]
    else:
        touch_axes = [axis for axis, overlap in enumerate(overlaps) if abs(overlap) <= 1.0e-7]
        face_axis = touch_axes[0] if len(touch_axes) == 1 else int(np.argmin(np.maximum(overlaps, 0.0)))
    other_axes = [axis for axis in range(3) if axis != face_axis]
    if overlaps[other_axes[0]] <= 0.0 or overlaps[other_axes[1]] <= 0.0:
        return None
    area_mm2 = float(overlaps[other_axes[0]] * overlaps[other_axes[1]])
    distance_mm = _node_cell_center_distance_mm(node, cell)
    return max(0.0, area_mm2), gap_mm, distance_mm


def _node_cell_gap_mm(node: dict, cell: OctreeCell) -> float:
    node_min, node_max = _node_bounds_mm(node)
    cell_min, cell_max = _cell_bounds_mm(cell)
    return float(np.linalg.norm(_aabb_gaps_mm(node_min, node_max, cell_min, cell_max)))


def _node_cell_center_distance_mm(node: dict, cell: OctreeCell) -> float:
    return float(np.linalg.norm(np.asarray(node["center_mm"], dtype=float) - np.asarray(cell.center_mm, dtype=float)))


def _node_bounds_mm(node: dict) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(node["center_mm"], dtype=float)
    size = np.maximum(np.asarray(node["size_mm"], dtype=float), 1.0e-9)
    return center - size * 0.5, center + size * 0.5


def _cell_bounds_mm(cell: OctreeCell) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(cell.center_mm, dtype=float)
    size = np.maximum(np.asarray(cell.size_mm, dtype=float), 1.0e-9)
    return center - size * 0.5, center + size * 0.5


def _aabb_gaps_mm(
    amin: np.ndarray,
    amax: np.ndarray,
    bmin: np.ndarray,
    bmax: np.ndarray,
) -> np.ndarray:
    return np.maximum(np.maximum(bmin - amax, amin - bmax), 0.0)


def harmonic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 / (1.0 / a + 1.0 / b)


def _shared_face_area_and_distance(a: OctreeCell, b: OctreeCell) -> tuple[float, float]:
    ca = np.asarray(a.center_mm)
    cb = np.asarray(b.center_mm)
    sa = np.asarray(a.size_mm)
    sb = np.asarray(b.size_mm)
    amin, amax = ca - sa * 0.5, ca + sa * 0.5
    bmin, bmax = cb - sb * 0.5, cb + sb * 0.5
    touch_axes = []
    overlaps = []
    for axis in range(3):
        gap = max(bmin[axis] - amax[axis], amin[axis] - bmax[axis])
        if abs(gap) <= 1.0e-7:
            touch_axes.append(axis)
            overlaps.append(0.0)
        elif gap > 0.0:
            return 0.0, 0.0
        else:
            overlaps.append(min(amax[axis], bmax[axis]) - max(amin[axis], bmin[axis]))
    if len(touch_axes) != 1:
        return 0.0, 0.0
    face_axis = touch_axes[0]
    other = [axis for axis in range(3) if axis != face_axis]
    area = overlaps[other[0]] * overlaps[other[1]]
    distance = float(np.linalg.norm(ca - cb))
    return float(max(0.0, area)), distance


def _add_near_contact_edges(
    cells: list[OctreeCell],
    cell_to_node: dict[str, dict],
    connected_pairs: set[tuple[str, str]],
    edges: list[dict],
    edge_index: int,
    materials: dict[str, Material],
    warnings: list[str],
    default_contact_G_W_K: float,
    max_gap_mm: float,
    connected_node_pairs: set[tuple[int, int]],
) -> int:
    for cell, other in _candidate_cell_pairs(cells, max_gap_mm):
        pair_key = _cell_pair_key(cell, other)
        if pair_key in connected_pairs:
            continue
        contact = _near_contact_area_gap_and_distance(cell, other, max_gap_mm)
        if contact is None:
            continue
        area_mm2, gap_mm, distance_mm = contact
        node_a = cell_to_node[cell.cell_id]
        node_b = cell_to_node[other.cell_id]
        node_pair = _node_pair_key(node_a, node_b)
        if node_pair in connected_node_pairs:
            continue
        material_a = resolve_material(str(node_a["material_name"]), materials, warnings)
        material_b = resolve_material(str(node_b["material_name"]), materials, warnings)
        same_component = node_a["component_name"] == node_b["component_name"]
        same_material = node_a["material_name"] == node_b["material_name"]
        if same_component:
            edge_type = "near_internal_conduction"
            k_eff = harmonic_mean(material_a.k_W_mK, material_b.k_W_mK)
            G = k_eff * (area_mm2 * 1.0e-6) / max(distance_mm * 1.0e-3, 1.0e-12)
            confidence = "medium"
        elif same_material and area_mm2 > 0.0:
            edge_type = "near_same_material_contact"
            k_eff = material_a.k_W_mK
            G = k_eff * (area_mm2 * 1.0e-6) / max(distance_mm * 1.0e-3, 1.0e-12)
            confidence = "medium"
        else:
            edge_type = "near_component_contact"
            G = default_contact_G_W_K
            confidence = "low"
        edges.append(
            {
                "edge_id": f"edge_{edge_index}",
                "node_i": int(node_a["node_id"]),
                "node_j": int(node_b["node_id"]),
                "edge_type": edge_type,
                "G_W_K": float(G),
                "shared_area_m2": float(area_mm2 * 1.0e-6),
                "distance_m": float(distance_mm * 1.0e-3),
                "contact_confidence": confidence,
                "source": "geometry_contact_distance",
                "warnings": [
                    f"Added by voxel-surface contact pass with gap {gap_mm:.3g} mm."
                ],
            }
        )
        connected_pairs.add(pair_key)
        connected_node_pairs.add(node_pair)
        edge_index += 1
    return edge_index


def _node_pair_key(node_a: dict, node_b: dict) -> tuple[int, int]:
    return tuple(sorted((int(node_a["node_id"]), int(node_b["node_id"]))))


def _candidate_cell_pairs(cells: list[OctreeCell], max_gap_mm: float) -> Iterator[tuple[OctreeCell, OctreeCell]]:
    bucket_size = _cell_pair_bucket_size(cells, max_gap_mm)
    buckets: dict[tuple[int, int, int], list[OctreeCell]] = {}
    for cell in cells:
        for key in _cell_bucket_keys(cell, bucket_size, max_gap_mm):
            buckets.setdefault(key, []).append(cell)
    seen: set[tuple[str, str]] = set()
    for cell in cells:
        for key in _cell_bucket_keys(cell, bucket_size, max_gap_mm):
            for other in buckets.get(key, []):
                if other is cell:
                    continue
                pair_key = _cell_pair_key(cell, other)
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                yield cell, other


def _cell_pair_bucket_size(cells: list[OctreeCell], max_gap_mm: float) -> float:
    sizes = [float(min(cell.size_mm)) for cell in cells if min(cell.size_mm) > 0.0]
    if not sizes:
        return max(float(max_gap_mm), 1.0e-9)
    # The median keeps dense, mostly uniform voxel grids from collapsing into a few oversized buckets
    # when a small number of coarse leaves are present.
    return max(float(np.median(sizes)) + float(max_gap_mm), 1.0e-9)


def _cell_bucket_keys(cell: OctreeCell, bucket_size_mm: float, padding_mm: float) -> Iterator[tuple[int, int, int]]:
    mins, maxs = _cell_bounds_mm(cell)
    padding = max(0.0, float(padding_mm))
    lo = np.floor((mins - padding) / bucket_size_mm).astype(int)
    hi = np.floor((maxs + padding) / bucket_size_mm).astype(int)
    for ix in range(int(lo[0]), int(hi[0]) + 1):
        for iy in range(int(lo[1]), int(hi[1]) + 1):
            for iz in range(int(lo[2]), int(hi[2]) + 1):
                yield ix, iy, iz


def _cell_pair_key(a: OctreeCell, b: OctreeCell) -> tuple[str, str]:
    return tuple(sorted((a.cell_id, b.cell_id)))


def _near_contact_area_gap_and_distance(
    a: OctreeCell, b: OctreeCell, max_gap_mm: float
) -> tuple[float, float, float] | None:
    ca = np.asarray(a.center_mm, dtype=float)
    cb = np.asarray(b.center_mm, dtype=float)
    sa = np.asarray(a.size_mm, dtype=float)
    sb = np.asarray(b.size_mm, dtype=float)
    amin, amax = ca - sa * 0.5, ca + sa * 0.5
    bmin, bmax = cb - sb * 0.5, cb + sb * 0.5
    gaps = np.maximum(np.maximum(bmin - amax, amin - bmax), 0.0)
    gap_mm = float(np.linalg.norm(gaps))
    if gap_mm > max_gap_mm:
        return None
    near_axes = [axis for axis, gap in enumerate(gaps) if gap > 1.0e-7]
    if len(near_axes) > 1:
        return None
    overlaps = [
        min(amax[axis], bmax[axis]) - max(amin[axis], bmin[axis])
        for axis in range(3)
    ]
    if len(near_axes) == 1:
        face_axis = near_axes[0]
        other = [axis for axis in range(3) if axis != face_axis]
        if overlaps[other[0]] <= 0.0 or overlaps[other[1]] <= 0.0:
            return None
        area_mm2 = overlaps[other[0]] * overlaps[other[1]]
    else:
        touch_axes = [axis for axis, overlap in enumerate(overlaps) if abs(overlap) <= 1.0e-7]
        if len(touch_axes) != 1:
            return None
        face_axis = touch_axes[0]
        other = [axis for axis in range(3) if axis != face_axis]
        if overlaps[other[0]] <= 0.0 or overlaps[other[1]] <= 0.0:
            return None
        area_mm2 = overlaps[other[0]] * overlaps[other[1]]
    distance_mm = float(np.linalg.norm(ca - cb))
    return float(max(0.0, area_mm2)), gap_mm, distance_mm


def _exposed_areas_m2(cells: list[OctreeCell]) -> dict[str, float]:
    """Estimate each leaf cell's exterior area after subtracting solid face contacts."""
    areas_mm2: dict[str, float] = {}
    for cell in cells:
        sx, sy, sz = (float(v) for v in cell.size_mm)
        areas_mm2[cell.cell_id] = 2.0 * (sx * sy + sx * sz + sy * sz)
    for a, b in _candidate_cell_pairs(cells, 0.0):
        shared_area_mm2, _distance_mm = _shared_face_area_and_distance(a, b)
        if shared_area_mm2 <= 0.0:
            continue
        areas_mm2[a.cell_id] = max(0.0, areas_mm2[a.cell_id] - shared_area_mm2)
        areas_mm2[b.cell_id] = max(0.0, areas_mm2[b.cell_id] - shared_area_mm2)
    return {cell_id: area_mm2 * 1.0e-6 for cell_id, area_mm2 in areas_mm2.items()}
