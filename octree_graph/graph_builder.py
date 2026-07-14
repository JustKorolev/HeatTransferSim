"""Build thermal graph nodes and conductance edges from octree leaves."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import re
from typing import Any, Iterator

import numpy as np

from .load_gltf import MeshObject
from .load_contact_report import ContactReport
from .materials import DEFAULT_ASSIGNED_MATERIAL_NAME, Material, resolve_material
from .octree import OctreeCell, _physical_material_name, _triangle_intersects_aabb


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


@dataclass
class _ContactSample:
    point: np.ndarray
    area_mm2: float
    normal: np.ndarray


@dataclass
class _RoleContactCandidate:
    component_name: str
    body_object: MeshObject
    has_intersection: bool
    min_distance_mm: float
    contact_area_mm2: float
    patch_centroid_mm: np.ndarray
    role_to_patch_distance_mm: float
    sample_points_mm: list[np.ndarray]
    sample_areas_mm2: list[float]
    closest_points_mm: list[np.ndarray]


def build_graph(
    leaves: list[OctreeCell],
    contact_report: ContactReport | None,
    materials: dict[str, Material],
    warnings: list[str],
    default_contact_G_W_K: float = 0.1,
    radiation_reference_temperature_K: float = 293.15,
    contact_detection_distance_mm: float = 0.0,
    component_bounds_mm: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    body_objects: list[MeshObject] | None = None,
    role_components: list[RoleComponent] | None = None,
    role_contact_tolerance_mm: float = _ROLE_NODE_CONTACT_TOLERANCE_MM,
    role_contact_tolerance_max_mm: float | None = None,
    role_contact_tolerance_growth_factor: float = 2.0,
    max_heater_sensor_pair_distance_mm: float = 25.0,
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
    role_components = role_components or []
    _warn_overlapping_role_components(role_components, warnings)
    if body_objects:
        role_groups: dict[str, list[dict]] = {}
    else:
        role_groups = _tag_role_voxel_nodes(
            nodes,
            solid,
            role_components,
            warnings,
        )
    cell_to_node = {node["cell_id"]: node for node in nodes}
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
    if role_groups:
        nodes, edges = _consolidate_role_voxel_nodes(nodes, edges, role_groups, warnings)
    _map_roles_to_body_cells(
        nodes,
        solid,
        role_components,
        body_objects or [],
        warnings,
        role_contact_tolerance_mm=role_contact_tolerance_mm,
    )
    _attach_role_interfaces_to_body_nodes(
        nodes,
        solid,
        role_components,
        warnings,
        role_contact_tolerance_mm=role_contact_tolerance_mm,
        role_contact_tolerance_max_mm=(
            role_contact_tolerance_max_mm
            if role_contact_tolerance_max_mm is not None
            else role_contact_tolerance_mm
        ),
        role_contact_tolerance_growth_factor=role_contact_tolerance_growth_factor,
    )
    _attach_sensor_connections_and_pair_roles(
        nodes,
        edges,
        warnings,
        max_heater_sensor_pair_distance_mm=max_heater_sensor_pair_distance_mm,
        role_components=role_components,
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


def _warn_overlapping_role_components(components: list[RoleComponent], warnings: list[str]) -> None:
    overlap_tolerance_mm = 1.0e-6
    for left_index, left in enumerate(components):
        left_min, left_max = left.bounds_mm
        for right in components[left_index + 1 :]:
            right_min, right_max = right.bounds_mm
            if np.any(left_max < right_min) or np.any(right_max < left_min):
                continue
            distance_mm = _role_component_distance_mm(left, right)
            if distance_mm > overlap_tolerance_mm:
                continue
            if left.kind == right.kind == "heater":
                warnings.append(
                    f"Distinct heater role components {left.name!r} and {right.name!r} have overlapping CAD bounds; "
                    f"mesh separation is {distance_mm:.6g} mm; kept separate and requiring independent body deposition nodes."
                )
            elif left.kind == right.kind == "sensor":
                warnings.append(
                    f"Distinct sensor role components {left.name!r} and {right.name!r} have overlapping CAD bounds; "
                    f"mesh separation is {distance_mm:.6g} mm; kept separate unless grouping already merged them."
                )
            else:
                warnings.append(
                    f"Heater/sensor role components {left.name!r} and {right.name!r} have overlapping CAD bounds; "
                    f"mesh separation is {distance_mm:.6g} mm; using overlap only for diagnostics/pairing, not power deposition."
                )


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


def _tag_role_voxel_nodes(
    nodes: list[dict],
    cells: list[OctreeCell],
    components: list[RoleComponent],
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    role_by_object: dict[str, RoleComponent] = {}
    matched_by_component: dict[str, int] = {component.name: 0 for component in components}
    role_groups: dict[str, dict[str, Any]] = {
        component.name: {"component": component, "node_ids": set()} for component in components
    }
    for component in components:
        for obj in component.objects:
            role_by_object[obj.name] = component
    if not role_by_object:
        return {}
    for node, cell in zip(nodes, cells):
        role_hits: dict[str, list[str]] = {}
        for component_name, fraction in cell.occupancy.items():
            if fraction <= 0.0:
                continue
            component = role_by_object.get(component_name)
            if component is None:
                continue
            role_hits.setdefault(component.kind, []).append(component_name)
            matched_by_component[component.name] = matched_by_component.get(component.name, 0) + 1
            role_groups.setdefault(component.name, {"component": component, "node_ids": set()})["node_ids"].add(
                int(node["node_id"])
            )
        if not role_hits:
            continue
        if "heater" in role_hits:
            node["is_heater"] = True
        if "sensor" in role_hits:
            node["is_sensor"] = True
        source_components = sorted({name for names in role_hits.values() for name in names})
        hit_components = sorted(
            {
                role_by_object[name].name
                for name in source_components
                if name in role_by_object
            }
        )
        node["role_source_components"] = source_components
        if hit_components:
            component = next(
                (
                    candidate
                    for candidate in components
                    if candidate.name == hit_components[0]
                ),
                None,
            )
            if component is not None:
                mins, maxs = component.bounds_mm
                node["node_type"] = component.kind
                node["component_name"] = component.name
                node["source_components"] = sorted({obj.name for obj in component.objects})
                node["source_bounds_mm"] = {
                    "min": [float(value) for value in mins],
                    "max": [float(value) for value in maxs],
                }
        node.setdefault("tags", {}).setdefault("notes", "")
        note = f"Detected CAD role geometry in voxel cell: {', '.join(source_components)}."
        node["tags"]["notes"] = _append_note(str(node["tags"].get("notes", "")), note)
        node.setdefault("warnings", []).append(note)
    for component in components:
        if matched_by_component.get(component.name, 0) > 0:
            continue
        warning = (
            f"Detected {component.kind} CAD component {component.name!r}, but it produced 0 solid voxel cells; "
            "it will be mapped to contacted body octree cells from CAD mesh contact."
        )
        warnings.append(warning)
    return {
        name: group
        for name, group in role_groups.items()
        if group.get("node_ids")
    }


def _append_note(existing: str, note: str) -> str:
    existing = str(existing or "").strip()
    return note if not existing else f"{existing}\n{note}"


def _consolidate_role_voxel_nodes(
    nodes: list[dict],
    edges: list[dict],
    role_groups: dict[str, dict[str, Any]],
    warnings: list[str],
) -> tuple[list[dict], list[dict]]:
    if not role_groups:
        return nodes, edges
    node_by_id = {int(node["node_id"]): node for node in nodes}
    old_to_new: dict[int, int] = {}
    replacement_nodes: list[dict] = []
    next_node_id = max(node_by_id, default=-1) + 1
    for group_name, group in sorted(role_groups.items()):
        component = group["component"]
        node_ids = sorted(int(node_id) for node_id in group.get("node_ids", set()) if int(node_id) in node_by_id)
        if len(node_ids) <= 1:
            continue
        group_nodes = [node_by_id[node_id] for node_id in node_ids]
        consolidated = _make_consolidated_role_node(next_node_id, component, group_nodes)
        replacement_nodes.append(consolidated)
        for node_id in node_ids:
            old_to_new[node_id] = next_node_id
        warnings.append(
            f"Consolidated {len(node_ids)} voxelized {component.kind} cell(s) into node {next_node_id} "
            f"for CAD component {group_name!r}."
        )
        next_node_id += 1
    if not old_to_new:
        return nodes, edges
    removed_ids = set(old_to_new)
    kept_nodes = [node for node in nodes if int(node["node_id"]) not in removed_ids]
    consolidated_edges = _rewrite_edges_for_consolidated_roles(edges, old_to_new, removed_ids)
    return kept_nodes + replacement_nodes, consolidated_edges


def _make_consolidated_role_node(node_id: int, component: RoleComponent, group_nodes: list[dict]) -> dict:
    total_C = sum(float(node.get("C_J_K", 0.0)) for node in group_nodes)
    total_mass = sum(float(node.get("mass_kg", 0.0)) for node in group_nodes)
    total_volume = sum(float(node.get("volume_m3", 0.0)) for node in group_nodes)
    initial_temperature = _weighted_node_average(group_nodes, "initial_temperature_K", total_C, default=293.15)
    g_rad = sum(float((node.get("radiation") or {}).get("G_rad_W_K", 0.0)) for node in group_nodes)
    radiating_area = sum(float((node.get("radiation") or {}).get("radiating_area_m2", 0.0)) for node in group_nodes)
    material_name = _dominant_node_value(group_nodes, "material_name")
    confidence = "high" if all(str(node.get("confidence", "")) == "high" for node in group_nodes) else "medium"
    is_heater = component.kind == "heater"
    is_sensor = component.kind == "sensor"
    source_components = sorted({obj.name for obj in component.objects})
    mins, maxs = component.bounds_mm
    notes = f"Consolidated {len(group_nodes)} voxelized {component.kind} cell(s) from CAD component {component.name!r}."
    return {
        "node_id": int(node_id),
        "cell_id": f"{component.kind}_consolidated_{node_id}",
        "coord": [int(node_id), 0, 0],
        "center_mm": [float(value) for value in component.center_mm],
        "size_mm": [float(max(value, 1.0e-6)) for value in component.size_mm],
        "level": min(int(node.get("level", 0)) for node in group_nodes),
        "node_type": component.kind,
        "component_name": component.name,
        "material_name": material_name,
        "volume_m3": float(total_volume),
        "mass_kg": float(total_mass),
        "C_J_K": float(total_C),
        "initial_temperature_K": float(initial_temperature),
        "occupancy_fraction": 1.0,
        "is_heater": bool(is_heater),
        "is_sensor": bool(is_sensor),
        "confidence": confidence,
        "warnings": [notes],
        "radiation": {
            "is_exposed": radiating_area > 0.0,
            "radiating_area_m2": float(radiating_area),
            "emissivity": float(_weighted_radiation_value(group_nodes, "emissivity", default=0.0)),
            "G_rad_W_K": float(g_rad),
            "R_rad_K_W": float(1.0 / g_rad) if g_rad > 0.0 else None,
        },
        "tags": {"notes": notes},
        "source_components": source_components,
        "source_node_ids": [int(node["node_id"]) for node in group_nodes],
        "source_cell_ids": [str(node["cell_id"]) for node in group_nodes if node.get("cell_id")],
        "source_bounds_mm": {
            "min": [float(value) for value in mins],
            "max": [float(value) for value in maxs],
        },
        "role_source_components": source_components,
    }


def _rewrite_edges_for_consolidated_roles(
    edges: list[dict],
    old_to_new: dict[int, int],
    removed_ids: set[int],
) -> list[dict]:
    merged: dict[tuple[int, int], dict] = {}
    passthrough: list[dict] = []
    for edge in edges:
        old_i = int(edge["node_i"])
        old_j = int(edge["node_j"])
        new_i = old_to_new.get(old_i, old_i)
        new_j = old_to_new.get(old_j, old_j)
        if new_i == new_j:
            continue
        key = tuple(sorted((new_i, new_j)))
        if old_i not in removed_ids and old_j not in removed_ids:
            copied = dict(edge)
            copied["node_i"], copied["node_j"] = key
            passthrough.append(copied)
            continue
        if key not in merged:
            copied = dict(edge)
            copied["node_i"], copied["node_j"] = key
            copied["edge_type"] = "consolidated_role_contact"
            copied["source"] = "voxel_role_consolidation"
            copied["G_W_K"] = 0.0
            copied["shared_area_m2"] = 0.0
            copied["warnings"] = ["Merged from voxelized heater/sensor cell connections."]
            merged[key] = copied
        merged_edge = merged[key]
        merged_edge["G_W_K"] = float(merged_edge.get("G_W_K", 0.0)) + float(edge.get("G_W_K", 0.0))
        merged_edge["shared_area_m2"] = float(merged_edge.get("shared_area_m2", 0.0)) + float(edge.get("shared_area_m2", 0.0))
        merged_edge["distance_m"] = min(
            float(merged_edge.get("distance_m", edge.get("distance_m", 0.0))),
            float(edge.get("distance_m", 0.0)),
        )
    rewritten = passthrough + list(merged.values())
    for index, edge in enumerate(rewritten):
        edge["edge_id"] = f"edge_{index}"
    return rewritten


def _cell_to_node_lookup(nodes: list[dict]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for node in nodes:
        cell_id = node.get("cell_id")
        if cell_id:
            lookup[str(cell_id)] = node
        for source_cell_id in node.get("source_cell_ids", []) or []:
            lookup[str(source_cell_id)] = node
    return lookup


def _map_roles_to_body_cells(
    nodes: list[dict],
    cells: list[OctreeCell],
    role_components: list[RoleComponent],
    body_objects: list[MeshObject],
    warnings: list[str],
    *,
    role_contact_tolerance_mm: float,
) -> dict[str, dict[str, Any]]:
    if not role_components:
        return {}
    mappings: dict[str, dict[str, Any]] = {}
    body_by_name = {obj.name: obj for obj in body_objects}
    if not body_by_name:
        for component in role_components:
            _warn_unresolved_role(component, warnings, "no body CAD mesh objects were available for contact analysis")
        return mappings
    cell_to_node = _cell_to_node_lookup(nodes)
    for component in role_components:
        candidate = _select_role_body_contact(component, list(body_by_name.values()), role_contact_tolerance_mm)
        if candidate is None:
            nearest = _nearest_body_candidate_diagnostic(component, list(body_by_name.values()))
            detail = (
                f"nearest body component {nearest[0]!r} at {nearest[1]:.6g} mm"
                if nearest is not None
                else "no candidate body component was found"
            )
            _warn_unresolved_role(
                component,
                warnings,
                f"no mesh contact within {float(role_contact_tolerance_mm):.6g} mm; {detail}",
            )
            mappings[component.name] = {"status": "unresolved", "diagnostics": detail}
            continue
        node_areas = _project_contact_candidate_to_cells(candidate, cells, cell_to_node)
        if not node_areas:
            _warn_unresolved_role(
                component,
                warnings,
                f"selected body component {candidate.component_name!r} had a CAD contact patch, "
                "but no contact samples mapped to matching body octree cells",
            )
            mappings[component.name] = {"status": "unresolved", "selected_body_component": candidate.component_name}
            continue
        node_ids = sorted(node_areas)
        areas = [float(node_areas[node_id]) for node_id in node_ids]
        weights = _normalized_weights(areas, len(areas))
        representative_id = max(node_ids, key=lambda node_id: (node_areas[node_id], -node_id))
        role_node = next((node for node in nodes if int(node["node_id"]) == int(representative_id)), None)
        if role_node is None:
            continue
        if bool(role_node.get("is_heater")) or bool(role_node.get("is_sensor")):
            role_node.setdefault("warnings", []).append(
                f"Multiple CAD roles map to body node {representative_id}; latest role metadata is {component.name!r}."
            )
        _apply_role_mapping_to_node(role_node, component, candidate, node_ids, areas, weights)
        mappings[component.name] = {
            "status": "valid",
            "node_id": int(representative_id),
            "selected_body_component": candidate.component_name,
            "contact_node_ids": node_ids,
            "contact_areas_mm2": areas,
            "contact_weights": weights,
        }
    return mappings


def _apply_role_mapping_to_node(
    node: dict,
    component: RoleComponent,
    candidate: _RoleContactCandidate,
    node_ids: list[int],
    areas_mm2: list[float],
    weights: list[float],
) -> None:
    mins, maxs = component.bounds_mm
    role_sources = sorted({obj.name for obj in component.objects})
    node["node_type"] = component.kind
    node["component_name"] = component.name
    node["source_components"] = role_sources
    node["role_source_components"] = role_sources
    node["source_bounds_mm"] = {
        "min": [float(value) for value in mins],
        "max": [float(value) for value in maxs],
    }
    node["role_selected_body_component"] = candidate.component_name
    node["role_contact_node_ids"] = [int(value) for value in node_ids]
    node["role_contact_areas_mm2"] = [float(value) for value in areas_mm2]
    node["role_contact_weights"] = [float(value) for value in weights]
    node["role_contact_total_area_mm2"] = float(sum(areas_mm2))
    node["role_contact_status"] = "valid"
    node["role_contact_tolerance_mm"] = float(candidate.min_distance_mm)
    node["role_contact_diagnostics"] = {
        "selected_body_component": candidate.component_name,
        "min_distance_mm": float(candidate.min_distance_mm),
        "contact_area_mm2": float(candidate.contact_area_mm2),
        "role_to_patch_distance_mm": float(candidate.role_to_patch_distance_mm),
        "has_intersection": bool(candidate.has_intersection),
    }
    node.setdefault("tags", {}).setdefault("notes", "")
    node["tags"]["notes"] = _append_note(
        str(node["tags"].get("notes", "")),
        f"CAD {component.kind} role {component.name!r} mapped to body component {candidate.component_name!r}.",
    )
    if component.kind == "heater":
        node["is_heater"] = True
        node["power_deposition_node_ids"] = [int(value) for value in node_ids]
        node["power_deposition_weights"] = [float(value) for value in weights]
        node["heater_attached"] = True
        node["heater_valid"] = True
        node["heater_warning"] = ""
    elif component.kind == "sensor":
        node["is_sensor"] = True
        node["readout_node_ids"] = [int(value) for value in node_ids]
        node["readout_weights"] = [float(value) for value in weights]
        node["sensor_connected_node_ids"] = [int(value) for value in node_ids]
        node["sensor_valid"] = True
        node["sensor_monitor_only"] = False


def _warn_unresolved_role(component: RoleComponent, warnings: list[str], detail: str) -> None:
    center = np.asarray(component.center_mm, dtype=float)
    warnings.append(
        f"Unresolved {component.kind} CAD role {component.name!r} at "
        f"({center[0]:.6g}, {center[1]:.6g}, {center[2]:.6g}) mm: {detail}."
    )


def _select_role_body_contact(
    component: RoleComponent,
    body_objects: list[MeshObject],
    tolerance_mm: float,
) -> _RoleContactCandidate | None:
    tolerance = max(0.0, float(tolerance_mm))
    role_min, role_max = component.bounds_mm
    expanded_min = role_min - tolerance
    expanded_max = role_max + tolerance
    samples = _role_surface_samples(component)
    if not samples:
        return None
    candidates: list[_RoleContactCandidate] = []
    for body in body_objects:
        body_min, body_max = body.bounds_mm
        if np.any(expanded_max < body_min) or np.any(body_max < expanded_min):
            continue
        candidate = _evaluate_role_body_contact(component, samples, body, tolerance)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            0 if item.has_intersection else 1,
            item.role_to_patch_distance_mm,
            -item.contact_area_mm2,
            item.min_distance_mm,
            item.component_name,
        )
    )
    return candidates[0]


def _evaluate_role_body_contact(
    component: RoleComponent,
    samples: list[_ContactSample],
    body: MeshObject,
    tolerance_mm: float,
) -> _RoleContactCandidate | None:
    triangles = _mesh_triangles_array(body)
    if triangles.size == 0:
        return None
    contact_points: list[np.ndarray] = []
    contact_areas: list[float] = []
    closest_points: list[np.ndarray] = []
    min_distance = float("inf")
    for sample in samples:
        closest, distance = _closest_point_on_mesh(sample.point, triangles)
        min_distance = min(min_distance, distance)
        penetrates_body = _point_inside_mesh_object(sample.point, body)
        if distance <= tolerance_mm + 1.0e-9 or penetrates_body:
            contact_points.append(sample.point)
            contact_areas.append(float(sample.area_mm2))
            closest_points.append(closest)
    if not contact_points:
        return None
    total_area = float(sum(contact_areas))
    if total_area <= 0.0:
        return None
    patch_centroid = sum(point * area for point, area in zip(contact_points, contact_areas)) / total_area
    role_center = np.asarray(component.center_mm, dtype=float)
    return _RoleContactCandidate(
        component_name=body.name,
        body_object=body,
        has_intersection=bool(min_distance <= 1.0e-7),
        min_distance_mm=float(min_distance),
        contact_area_mm2=total_area,
        patch_centroid_mm=np.asarray(patch_centroid, dtype=float),
        role_to_patch_distance_mm=float(np.linalg.norm(np.asarray(patch_centroid, dtype=float) - role_center)),
        sample_points_mm=contact_points,
        sample_areas_mm2=contact_areas,
        closest_points_mm=closest_points,
    )


def _nearest_body_candidate_diagnostic(component: RoleComponent, body_objects: list[MeshObject]) -> tuple[str, float] | None:
    samples = _role_surface_samples(component)
    if not samples:
        return None
    best: tuple[str, float] | None = None
    for body in body_objects:
        triangles = _mesh_triangles_array(body)
        if triangles.size == 0:
            continue
        distance = min(_closest_point_on_mesh(sample.point, triangles)[1] for sample in samples)
        if best is None or distance < best[1]:
            best = (body.name, float(distance))
    return best


def _point_inside_mesh_object(point_mm: np.ndarray, obj: MeshObject) -> bool:
    point = np.asarray(point_mm, dtype=float)
    try:
        contains = getattr(obj.mesh, "contains", None)
        if callable(contains):
            result = np.asarray(contains([point]), dtype=bool)
            if result.size:
                return bool(result[0])
    except Exception:
        pass
    mins, maxs = obj.bounds_mm
    return bool(np.all(point >= np.asarray(mins, dtype=float) - 1.0e-9) and np.all(point <= np.asarray(maxs, dtype=float) + 1.0e-9))


def _project_contact_candidate_to_cells(
    candidate: _RoleContactCandidate,
    cells: list[OctreeCell],
    cell_to_node: dict[str, dict],
) -> dict[int, float]:
    selected_component = candidate.component_name
    node_areas: dict[int, float] = {}
    matching_cells = [cell for cell in cells if cell.dominant_component == selected_component]
    if not matching_cells:
        return node_areas
    bucket_size = _cell_pair_bucket_size(matching_cells, 0.0)
    buckets = _cell_bucket_index(matching_cells, bucket_size)
    for point, area in zip(candidate.closest_points_mm, candidate.sample_areas_mm2):
        cell = _cell_containing_point(point, matching_cells, buckets, bucket_size)
        if cell is None:
            continue
        node = cell_to_node.get(cell.cell_id)
        if node is None:
            continue
        node_id = int(node["node_id"])
        node_areas[node_id] = node_areas.get(node_id, 0.0) + max(0.0, float(area))
    return {node_id: area for node_id, area in node_areas.items() if area > 0.0}


def _cell_containing_point(
    point_mm: np.ndarray,
    cells: list[OctreeCell],
    buckets: dict[tuple[int, int, int], list[OctreeCell]],
    bucket_size_mm: float,
) -> OctreeCell | None:
    point = np.asarray(point_mm, dtype=float)
    candidates = _cells_intersecting_bounds(buckets, bucket_size_mm, point, point, padding_mm=1.0e-7)
    if not candidates:
        candidates = cells
    containing: list[tuple[float, str, OctreeCell]] = []
    for cell in candidates:
        cell_min, cell_max = _cell_bounds_mm(cell)
        if np.all(point >= cell_min - 1.0e-7) and np.all(point <= cell_max + 1.0e-7):
            center_distance = float(np.linalg.norm(point - np.asarray(cell.center_mm, dtype=float)))
            containing.append((center_distance, cell.cell_id, cell))
    if not containing:
        return None
    containing.sort(key=lambda item: (item[0], item[1]))
    return containing[0][2]


def _role_surface_samples(component: RoleComponent) -> list[_ContactSample]:
    samples: list[_ContactSample] = []
    for obj in component.objects:
        triangles = _mesh_triangles_array(obj)
        for triangle in triangles:
            area = _triangle_area_mm2(triangle)
            if area <= 0.0:
                continue
            center = np.mean(triangle, axis=0)
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            norm = float(np.linalg.norm(normal))
            if norm > 0.0:
                normal = normal / norm
            else:
                normal = np.zeros(3, dtype=float)
            samples.append(_ContactSample(point=np.asarray(center, dtype=float), area_mm2=area, normal=normal))
    if samples:
        return samples
    for obj in component.objects:
        mins, maxs = obj.bounds_mm
        center = (np.asarray(mins, dtype=float) + np.asarray(maxs, dtype=float)) * 0.5
        size = np.maximum(np.asarray(maxs, dtype=float) - np.asarray(mins, dtype=float), 1.0)
        samples.append(_ContactSample(point=center, area_mm2=float(np.prod(size[:2])), normal=np.zeros(3, dtype=float)))
    return samples


def _mesh_triangles_array(obj: MeshObject) -> np.ndarray:
    try:
        triangles = np.asarray(getattr(obj.mesh, "triangles", []), dtype=float)
    except Exception:
        return np.empty((0, 3, 3), dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.all(np.isfinite(triangles)):
        return np.empty((0, 3, 3), dtype=float)
    return triangles


def _triangle_area_mm2(triangle: np.ndarray) -> float:
    try:
        area = 0.5 * float(np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])))
    except Exception:
        return 0.0
    return area if np.isfinite(area) and area > 0.0 else 0.0


def _closest_point_on_mesh(point: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, float]:
    best_point = np.zeros(3, dtype=float)
    best_distance = float("inf")
    for triangle in triangles:
        closest = _closest_point_on_triangle(point, triangle)
        distance = float(np.linalg.norm(point - closest))
        if distance < best_distance:
            best_distance = distance
            best_point = closest
    return best_point, best_distance


def _closest_point_on_triangle(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    # Real-Time Collision Detection, Christer Ericson, closest point on triangle.
    p = np.asarray(point, dtype=float)
    a, b, c = np.asarray(triangle, dtype=float)
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab
    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


def _attach_role_interfaces_to_body_nodes(
    nodes: list[dict],
    cells: list[OctreeCell],
    role_components: list[RoleComponent],
    warnings: list[str],
    *,
    role_contact_tolerance_mm: float,
    role_contact_tolerance_max_mm: float,
    role_contact_tolerance_growth_factor: float,
) -> None:
    if not role_components:
        return
    component_by_name = {component.name: component for component in role_components}
    component_by_object = {
        obj.name: component
        for component in role_components
        for obj in component.objects
    }
    cell_to_node = _cell_to_node_lookup(nodes)
    max_search_tolerance = max(float(role_contact_tolerance_mm), float(role_contact_tolerance_max_mm))
    bucket_size = _cell_pair_bucket_size(cells, max_search_tolerance)
    cell_buckets = _cell_bucket_index(cells, bucket_size)
    for role_node in nodes:
        if not (bool(role_node.get("is_heater")) or bool(role_node.get("is_sensor"))):
            continue
        if str(role_node.get("role_contact_status", "")) == "valid":
            if bool(role_node.get("is_heater")):
                ids = [int(value) for value in role_node.get("power_deposition_node_ids", []) or []]
                role_node["power_deposition_node_ids"] = ids
                role_node["power_deposition_weights"] = _normalized_weights(
                    role_node.get("power_deposition_weights", []) or [],
                    len(ids),
                )
                role_node["heater_attached"] = bool(ids)
                role_node["heater_valid"] = bool(ids)
                role_node["heater_warning"] = "" if ids else "No CAD contact-area body cells found for heater power deposition."
            if bool(role_node.get("is_sensor")):
                ids = [int(value) for value in role_node.get("readout_node_ids", []) or role_node.get("sensor_connected_node_ids", []) or []]
                role_node["readout_node_ids"] = ids
                role_node["readout_weights"] = _normalized_weights(
                    role_node.get("readout_weights", []) or [],
                    len(ids),
                )
                role_node["sensor_connected_node_ids"] = ids
                role_node["sensor_valid"] = bool(ids)
                role_node["sensor_monitor_only"] = not bool(ids)
            continue
        component = component_by_name.get(str(role_node.get("component_name", "")))
        if component is None:
            for source_name in role_node.get("source_components", []) or role_node.get("role_source_components", []) or []:
                component = component_by_object.get(str(source_name))
                if component is not None:
                    break
        ids, weights, used_tolerance = _role_body_contact_node_weights(
            role_node,
            component,
            cells,
            cell_to_node,
            role_contact_tolerance_mm,
            role_contact_tolerance_max_mm,
            role_contact_tolerance_growth_factor,
            cell_buckets=cell_buckets,
            bucket_size_mm=bucket_size,
        )
        if bool(role_node.get("is_heater")):
            role_node["power_deposition_node_ids"] = ids
            role_node["power_deposition_weights"] = weights
            role_node["heater_attached"] = bool(ids)
            role_node["heater_valid"] = bool(ids)
            if ids:
                role_node["heater_warning"] = ""
            else:
                role_node["heater_warning"] = "No contacted body cells found for heater power deposition."
                warnings.append(
                    f"Detected heater role {role_node.get('node_id')} ({role_node.get('component_name', '?')}) "
                    "has no contacted body cells for power deposition."
                )
        if bool(role_node.get("is_sensor")):
            role_node["readout_node_ids"] = ids
            role_node["readout_weights"] = weights
            role_node["sensor_connected_node_ids"] = ids
            role_node["sensor_valid"] = bool(ids)
            role_node["sensor_monitor_only"] = not bool(ids)
            if not ids:
                warnings.append(
                    f"Detected sensor role {role_node.get('node_id')} ({role_node.get('component_name', '?')}) "
                    "has no contacted body cells for readout."
                )
        if ids and used_tolerance > max(0.0, float(role_contact_tolerance_mm)) + 1.0e-12:
            warnings.append(
                f"Role node {role_node.get('node_id')} used expanded contact tolerance "
                f"{used_tolerance:.6g} mm to attach {len(ids)} body node(s)."
            )


def _role_body_contact_node_weights(
    role_node: dict,
    role_component: RoleComponent | None,
    cells: list[OctreeCell],
    cell_to_node: dict[str, dict],
    tolerance_mm: float,
    max_tolerance_mm: float,
    growth_factor: float,
    *,
    cell_buckets: dict[tuple[int, int, int], list[OctreeCell]] | None = None,
    bucket_size_mm: float | None = None,
) -> tuple[list[int], list[float], float]:
    tolerance = max(0.0, float(tolerance_mm))
    max_tolerance = max(tolerance, float(max_tolerance_mm))
    growth = max(1.01, float(growth_factor))
    while True:
        contacts: dict[int, float] = {}
        candidate_cells = _role_candidate_cells(
            role_node,
            role_component,
            cells,
            tolerance,
            cell_buckets=cell_buckets,
            bucket_size_mm=bucket_size_mm,
        )
        for cell in candidate_cells:
            body_node = cell_to_node.get(cell.cell_id)
            if body_node is None:
                continue
            if bool(body_node.get("is_heater")) or bool(body_node.get("is_sensor")):
                continue
            if role_component is not None:
                contact = _role_cell_contact(role_node, role_component, cell, tolerance)
            else:
                contact = _node_cell_contact(role_node, cell, tolerance)
            if contact is None:
                continue
            area_mm2, _gap_mm, _distance_mm = contact
            node_id = int(body_node["node_id"])
            contacts[node_id] = contacts.get(node_id, 0.0) + max(float(area_mm2), 1.0)
        if contacts or tolerance >= max_tolerance:
            ids = sorted(contacts)
            weights = _normalized_weights([contacts[node_id] for node_id in ids], len(ids))
            return ids, weights, tolerance
        tolerance = min(max_tolerance, tolerance * growth if tolerance > 0.0 else max_tolerance)


def _normalized_weights(weights: list[float], count: int) -> list[float]:
    if count <= 0:
        return []
    values = [float(value) for value in list(weights)[:count] if np.isfinite(float(value)) and float(value) >= 0.0]
    if len(values) != count or sum(values) <= 0.0:
        return [1.0 / float(count)] * count
    total = float(sum(values))
    return [float(value) / total for value in values]


def _attach_sensor_connections_and_pair_roles(
    nodes: list[dict],
    edges: list[dict],
    warnings: list[str],
    *,
    max_heater_sensor_pair_distance_mm: float,
    role_components: list[RoleComponent] | None = None,
) -> None:
    node_by_id = {int(node["node_id"]): node for node in nodes}
    component_by_name = {component.name: component for component in role_components or []}
    heaters = [node for node in nodes if bool(node.get("is_heater"))]
    sensors = [node for node in nodes if bool(node.get("is_sensor"))]
    if not heaters and not sensors:
        return
    for heater in heaters:
        heater["assigned_sensor_id"] = None
        heater["sensor_pair_distance_mm"] = None
        deposition = [
            int(value)
            for value in heater.get("power_deposition_node_ids", []) or []
            if int(value) in node_by_id
            and (
                int(value) == int(heater["node_id"])
                or (
                    not bool(node_by_id[int(value)].get("is_heater"))
                    and not bool(node_by_id[int(value)].get("is_sensor"))
                )
            )
        ]
        heater["power_deposition_node_ids"] = deposition
        heater["power_deposition_weights"] = _normalized_weights(
            heater.get("power_deposition_weights", []) or [],
            len(deposition),
        )
        heater["heater_attached"] = bool(deposition)
        heater["heater_valid"] = bool(deposition)
        heater["heater_warning"] = "" if deposition else "No body power deposition nodes found."
        if not deposition:
            warnings.append(
                f"Heater node {int(heater['node_id'])} has no body power deposition nodes; excluded from MIMO control."
            )
    for sensor in sensors:
        connected: set[int] = {
            int(value)
            for value in sensor.get("readout_node_ids", []) or sensor.get("sensor_connected_node_ids", []) or []
            if int(value) in node_by_id
            and (
                int(value) == int(sensor["node_id"])
                or (
                    not bool(node_by_id[int(value)].get("is_heater"))
                    and not bool(node_by_id[int(value)].get("is_sensor"))
                )
            )
        }
        sensor_id = int(sensor["node_id"])
        connected_role_ids: set[int] = set()
        if not connected:
            for edge in edges:
                node_i = int(edge["node_i"])
                node_j = int(edge["node_j"])
                other_id: int | None = None
                if node_i == sensor_id:
                    other_id = node_j
                elif node_j == sensor_id:
                    other_id = node_i
                if other_id is None:
                    continue
                other = node_by_id.get(other_id)
                if other is None:
                    continue
                if bool(other.get("is_heater")) or bool(other.get("is_sensor")):
                    connected_role_ids.add(other_id)
                    continue
                connected.add(other_id)
        inherited_from_heaters = False
        if not connected and connected_role_ids:
            for role_id in sorted(connected_role_ids):
                role_node = node_by_id.get(role_id)
                if role_node is None or not bool(role_node.get("is_heater")):
                    continue
                connected.update(
                    int(value)
                    for value in role_node.get("power_deposition_node_ids", []) or []
                    if int(value) in node_by_id
                    and not bool(node_by_id[int(value)].get("is_heater"))
                    and not bool(node_by_id[int(value)].get("is_sensor"))
                )
                connected.update(_external_body_neighbors(int(role_id), edges, node_by_id))
            inherited_from_heaters = bool(connected)
        sensor["sensor_connected_node_ids"] = sorted(connected)
        sensor["readout_node_ids"] = sorted(connected)
        sensor["readout_weights"] = _normalized_weights(sensor.get("readout_weights", []) or [], len(connected))
        sensor["sensor_valid"] = bool(connected)
        sensor["assigned_heater_id"] = None
        sensor["assigned_heater_ids"] = []
        sensor["sensor_pair_distance_mm"] = None
        sensor["sensor_control_mode"] = str(sensor.get("sensor_control_mode") or "manual")
        sensor["sensor_manual_power_W"] = float(sensor.get("sensor_manual_power_W", 0.0) or 0.0)
        sensor["sensor_monitor_only"] = not bool(connected)
        if not connected:
            warnings.append(
                f"Sensor node {sensor_id} has no connected body nodes; marked monitor-only and excluded from MIMO control."
            )
        elif inherited_from_heaters:
            warnings.append(
                f"Sensor node {sensor_id} has no direct body-node contacts but contacts heater node(s) "
                f"{sorted(connected_role_ids)}; using heater-adjacent body node(s) {sorted(connected)} for readout."
            )
    max_distance = max(0.0, float(max_heater_sensor_pair_distance_mm))
    candidate_pairs: list[tuple[float, int, int, dict, dict]] = []
    for heater in sorted(heaters, key=lambda item: int(item["node_id"])):
        if not bool(heater.get("heater_valid", True)):
            continue
        for sensor in sensors:
            sensor_id = int(sensor["node_id"])
            if not bool(sensor.get("sensor_valid", False)):
                continue
            distance = _role_pair_distance_mm(heater, sensor, component_by_name)
            if distance <= max_distance:
                candidate_pairs.append((distance, int(heater["node_id"]), sensor_id, heater, sensor))
    assigned_heaters: set[int] = set()
    assigned_sensors: set[int] = set()
    for distance, heater_id, sensor_id, heater, sensor in _unique_pair_assignments(candidate_pairs):
        _assign_automatic_pair(heater, sensor, heater_id, sensor_id, distance, status="automatic_unique")
        assigned_heaters.add(heater_id)
        assigned_sensors.add(sensor_id)
    for heater in sorted(heaters, key=lambda item: int(item["node_id"])):
        heater_id = int(heater["node_id"])
        if heater_id in assigned_heaters or not bool(heater.get("heater_valid", True)):
            continue
        reuse_candidates = [
            item
            for item in candidate_pairs
            if int(item[1]) == heater_id
        ]
        if not reuse_candidates:
            continue
        distance, _heater_id, sensor_id, _heater, sensor = min(
            reuse_candidates,
            key=lambda item: (item[0], item[2]),
        )
        _assign_automatic_pair(heater, sensor, heater_id, sensor_id, distance, status="automatic_reused_sensor")
        assigned_heaters.add(heater_id)
    for heater in heaters:
        if bool(heater.get("heater_valid", True)) and heater.get("assigned_sensor_id") is None:
            warnings.append(
                f"Heater node {int(heater['node_id'])} has no valid sensor within {max_distance:g} mm."
            )
    for sensor in sensors:
        if not sensor.get("assigned_heater_ids"):
            sensor["sensor_monitor_only"] = True
            warnings.append(f"Sensor node {int(sensor['node_id'])} has no assigned heater; marked monitor-only.")


def _assign_automatic_pair(
    heater: dict,
    sensor: dict,
    heater_id: int,
    sensor_id: int,
    distance_mm: float,
    *,
    status: str,
) -> None:
    heater["assigned_sensor_id"] = int(sensor_id)
    heater["sensor_pair_distance_mm"] = float(distance_mm)
    heater["heater_sensor_pairing_status"] = status
    assigned = sorted({int(value) for value in sensor.get("assigned_heater_ids", []) or []} | {int(heater_id)})
    sensor["assigned_heater_ids"] = assigned
    sensor["assigned_heater_id"] = int(assigned[0])
    previous_distance = sensor.get("sensor_pair_distance_mm")
    sensor["sensor_pair_distance_mm"] = (
        float(distance_mm)
        if previous_distance is None
        else min(float(previous_distance), float(distance_mm))
    )
    sensor["sensor_monitor_only"] = False
    sensor["sensor_control_mode"] = "mimo"
    sensor["heater_sensor_pairing_status"] = status


def _unique_pair_assignments(
    candidate_pairs: list[tuple[float, int, int, dict, dict]]
) -> list[tuple[float, int, int, dict, dict]]:
    if not candidate_pairs:
        return []
    heaters = sorted({heater_id for _distance, heater_id, _sensor_id, _heater, _sensor in candidate_pairs})
    sensors = sorted({sensor_id for _distance, _heater_id, sensor_id, _heater, _sensor in candidate_pairs})
    pair_by_ids = {(heater_id, sensor_id): item for item in candidate_pairs for _distance, heater_id, sensor_id, _heater, _sensor in [item]}
    try:
        from scipy.optimize import linear_sum_assignment

        cost = np.full((len(heaters), len(sensors)), 1.0e18, dtype=float)
        for row, heater_id in enumerate(heaters):
            for col, sensor_id in enumerate(sensors):
                item = pair_by_ids.get((heater_id, sensor_id))
                if item is not None:
                    cost[row, col] = float(item[0])
        rows, cols = linear_sum_assignment(cost)
        return [
            pair_by_ids[(heaters[int(row)], sensors[int(col)])]
            for row, col in zip(rows, cols)
            if cost[int(row), int(col)] < 1.0e17
        ]
    except Exception:
        limit = min(len(heaters), len(sensors))
        if limit <= 8:
            best: tuple[float, list[tuple[float, int, int, dict, dict]]] | None = None
            for sensor_order in permutations(sensors, limit):
                chosen: list[tuple[float, int, int, dict, dict]] = []
                total = 0.0
                for heater_id, sensor_id in zip(heaters[:limit], sensor_order):
                    item = pair_by_ids.get((heater_id, sensor_id))
                    if item is None:
                        break
                    chosen.append(item)
                    total += float(item[0])
                if len(chosen) == limit and (best is None or total < best[0]):
                    best = (total, chosen)
            if best is not None:
                return best[1]
        assigned_heaters: set[int] = set()
        assigned_sensors: set[int] = set()
        chosen = []
        for item in sorted(candidate_pairs, key=lambda value: (value[0], value[1], value[2])):
            _distance, heater_id, sensor_id, _heater, _sensor = item
            if heater_id in assigned_heaters or sensor_id in assigned_sensors:
                continue
            assigned_heaters.add(heater_id)
            assigned_sensors.add(sensor_id)
            chosen.append(item)
        return chosen


def _role_pair_distance_mm(
    heater: dict,
    sensor: dict,
    component_by_name: dict[str, RoleComponent],
) -> float:
    heater_component = component_by_name.get(str(heater.get("component_name", "")))
    sensor_component = component_by_name.get(str(sensor.get("component_name", "")))
    if heater_component is not None and sensor_component is not None:
        distance = _role_component_distance_mm(heater_component, sensor_component)
        if np.isfinite(distance):
            return float(distance)
    return _node_aabb_gap_mm(heater, sensor)


def _role_component_distance_mm(left: RoleComponent, right: RoleComponent) -> float:
    left_samples = _role_surface_samples(left)
    right_triangles = np.concatenate(
        [triangles for triangles in (_mesh_triangles_array(obj) for obj in right.objects) if triangles.size],
        axis=0,
    ) if any(_mesh_triangles_array(obj).size for obj in right.objects) else np.empty((0, 3, 3), dtype=float)
    if not left_samples or right_triangles.size == 0:
        return float("inf")
    left_to_right = min(_closest_point_on_mesh(sample.point, right_triangles)[1] for sample in left_samples)
    right_samples = _role_surface_samples(right)
    left_triangles = np.concatenate(
        [triangles for triangles in (_mesh_triangles_array(obj) for obj in left.objects) if triangles.size],
        axis=0,
    ) if any(_mesh_triangles_array(obj).size for obj in left.objects) else np.empty((0, 3, 3), dtype=float)
    if not right_samples or left_triangles.size == 0:
        return float(left_to_right)
    right_to_left = min(_closest_point_on_mesh(sample.point, left_triangles)[1] for sample in right_samples)
    return float(min(left_to_right, right_to_left))


def _external_body_neighbors(
    node_id: int,
    edges: list[dict],
    node_by_id: dict[int, dict],
) -> set[int]:
    connected: set[int] = set()
    for edge in edges:
        node_i = int(edge["node_i"])
        node_j = int(edge["node_j"])
        if node_i == int(node_id):
            other_id = node_j
        elif node_j == int(node_id):
            other_id = node_i
        else:
            continue
        other = node_by_id.get(int(other_id))
        if other is None or bool(other.get("is_heater")) or bool(other.get("is_sensor")):
            continue
        connected.add(int(other_id))
    return connected


def _node_aabb_gap_mm(left: dict, right: dict) -> float:
    left_min, left_max = _node_bounds_mm(left)
    right_min, right_max = _node_bounds_mm(right)
    gaps = np.maximum(np.maximum(left_min - right_max, right_min - left_max), 0.0)
    return float(np.linalg.norm(gaps))


def _node_bounds_mm(node: dict) -> tuple[np.ndarray, np.ndarray]:
    bounds = node.get("source_bounds_mm") or {}
    if isinstance(bounds, dict) and "min" in bounds and "max" in bounds:
        return np.asarray(bounds["min"], dtype=float), np.asarray(bounds["max"], dtype=float)
    center = np.asarray(node.get("center_mm", (0.0, 0.0, 0.0)), dtype=float)
    size = np.asarray(node.get("size_mm", (0.0, 0.0, 0.0)), dtype=float)
    half = np.maximum(size, 0.0) * 0.5
    return center - half, center + half


def _weighted_node_average(group_nodes: list[dict], field: str, total_C: float, default: float) -> float:
    if total_C <= 0.0:
        values = [float(node.get(field, default)) for node in group_nodes]
        return float(sum(values) / max(1, len(values)))
    return float(
        sum(float(node.get(field, default)) * float(node.get("C_J_K", 0.0)) for node in group_nodes) / total_C
    )


def _weighted_radiation_value(group_nodes: list[dict], field: str, default: float) -> float:
    weights = [float((node.get("radiation") or {}).get("radiating_area_m2", 0.0)) for node in group_nodes]
    total = sum(weights)
    if total <= 0.0:
        values = [float((node.get("radiation") or {}).get(field, default)) for node in group_nodes]
        return float(sum(values) / max(1, len(values)))
    return float(
        sum(float((node.get("radiation") or {}).get(field, default)) * weight for node, weight in zip(group_nodes, weights))
        / total
    )


def _dominant_node_value(group_nodes: list[dict], field: str) -> str:
    counts: dict[str, float] = {}
    for node in group_nodes:
        value = str(node.get(field, ""))
        counts[value] = counts.get(value, 0.0) + float(node.get("C_J_K", 0.0))
    return max(counts, key=counts.get) if counts else ""


def _append_role_nodes(
    nodes: list[dict],
    components: list[RoleComponent],
    contact_report: ContactReport,
    materials: dict[str, Material],
    warnings: list[str],
) -> list[tuple[dict, RoleComponent]]:
    role_contacts: list[tuple[dict, RoleComponent]] = []
    known_materials = set(materials)
    next_node_id = max((int(node.get("node_id", -1)) for node in nodes), default=-1) + 1
    for component in components:
        node_id = next_node_id
        next_node_id += 1
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
            "role_source_components": [obj.name for obj in component.objects],
            "source_bounds_mm": {
                "min": [float(value) for value in component.bounds_mm[0]],
                "max": [float(value) for value in component.bounds_mm[1]],
            },
        }
        nodes.append(node)
        role_contacts.append((node, component))
    return role_contacts


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
        try:
            mesh_volume = abs(float(getattr(obj.mesh, "volume", 0.0)))
        except Exception:
            mesh_volume = 0.0
        if mesh_volume > 0.0 and np.isfinite(mesh_volume):
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
    role_contacts: list[tuple[dict, RoleComponent]],
    edges: list[dict],
    edge_index: int,
    max_gap_mm: float,
    connected_node_pairs: set[tuple[int, int]],
    warnings: list[str],
    role_contact_tolerance_mm: float,
) -> int:
    if not cells or not role_contacts:
        return edge_index
    search_gap_mm = max(0.0, float(role_contact_tolerance_mm))
    bucket_size = _cell_pair_bucket_size(cells, search_gap_mm)
    cell_buckets = _cell_bucket_index(cells, bucket_size)
    for role_node, role_component in role_contacts:
        contacts: list[tuple[OctreeCell, float, float, float]] = []
        for cell in _role_candidate_cells(
            role_node,
            role_component,
            cells,
            search_gap_mm,
            cell_buckets=cell_buckets,
            bucket_size_mm=bucket_size,
        ):
            contact = _role_cell_contact(role_node, role_component, cell, search_gap_mm)
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


def _role_candidate_cells(
    role_node: dict,
    role_component: RoleComponent | None,
    cells: list[OctreeCell],
    tolerance_mm: float,
    *,
    cell_buckets: dict[tuple[int, int, int], list[OctreeCell]] | None,
    bucket_size_mm: float | None,
) -> list[OctreeCell]:
    if cell_buckets is None or bucket_size_mm is None:
        return cells
    try:
        if role_component is not None:
            mins, maxs = role_component.bounds_mm
        else:
            mins, maxs = _node_bounds_mm(role_node)
    except Exception:
        return cells
    return _cells_intersecting_bounds(
        cell_buckets,
        float(bucket_size_mm),
        np.asarray(mins, dtype=float),
        np.asarray(maxs, dtype=float),
        padding_mm=max(0.0, float(tolerance_mm)),
    )


def _role_cell_contact(
    role_node: dict,
    role_component: RoleComponent,
    cell: OctreeCell,
    max_gap_mm: float,
) -> tuple[float, float, float] | None:
    contact = _node_cell_contact(role_node, cell, max_gap_mm)
    if contact is not None:
        return contact
    if not _role_component_intersects_cell_bounds(role_component, cell, max_gap_mm):
        return None
    return 0.0, 0.0, _node_cell_center_distance_mm(role_node, cell)


def _role_component_intersects_cell_bounds(
    role_component: RoleComponent,
    cell: OctreeCell,
    tolerance_mm: float,
) -> bool:
    cell_min, cell_max = _cell_bounds_mm(cell)
    expanded_min = cell_min - max(0.0, float(tolerance_mm))
    expanded_max = cell_max + max(0.0, float(tolerance_mm))
    center = (expanded_min + expanded_max) * 0.5
    half_size = np.maximum((expanded_max - expanded_min) * 0.5, 1.0e-9)
    for obj in role_component.objects:
        obj_min, obj_max = obj.bounds_mm
        if np.any(expanded_max < obj_min) or np.any(obj_max < expanded_min):
            continue
        triangles = np.asarray(getattr(obj.mesh, "triangles", []), dtype=float)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
            continue
        for triangle in triangles:
            if _triangle_intersects_aabb(triangle, center, half_size):
                return True
    return False


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
    bounds = node.get("source_bounds_mm") or {}
    if isinstance(bounds, dict) and "min" in bounds and "max" in bounds:
        return np.asarray(bounds["min"], dtype=float), np.asarray(bounds["max"], dtype=float)
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
    buckets = _cell_bucket_index(cells, bucket_size, max_gap_mm)
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


def _cell_bucket_index(
    cells: list[OctreeCell],
    bucket_size_mm: float,
    padding_mm: float = 0.0,
) -> dict[tuple[int, int, int], list[OctreeCell]]:
    buckets: dict[tuple[int, int, int], list[OctreeCell]] = {}
    for cell in cells:
        for key in _cell_bucket_keys(cell, bucket_size_mm, padding_mm):
            buckets.setdefault(key, []).append(cell)
    return buckets


def _cells_intersecting_bounds(
    buckets: dict[tuple[int, int, int], list[OctreeCell]],
    bucket_size_mm: float,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    padding_mm: float = 0.0,
) -> list[OctreeCell]:
    if not buckets:
        return []
    mins = np.asarray(bounds_min, dtype=float)
    maxs = np.asarray(bounds_max, dtype=float)
    if mins.shape != (3,) or maxs.shape != (3,) or not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
        return []
    padding = max(0.0, float(padding_mm))
    query_min = np.minimum(mins, maxs) - padding
    query_max = np.maximum(mins, maxs) + padding
    seen: set[str] = set()
    matches: list[OctreeCell] = []
    for key in _bounds_bucket_keys(query_min, query_max, bucket_size_mm):
        for cell in buckets.get(key, []):
            if cell.cell_id in seen:
                continue
            cell_min, cell_max = _cell_bounds_mm(cell)
            if np.any(query_max < cell_min) or np.any(cell_max < query_min):
                continue
            seen.add(cell.cell_id)
            matches.append(cell)
    return matches


def _cell_bucket_keys(cell: OctreeCell, bucket_size_mm: float, padding_mm: float) -> Iterator[tuple[int, int, int]]:
    mins, maxs = _cell_bounds_mm(cell)
    padding = max(0.0, float(padding_mm))
    yield from _bounds_bucket_keys(mins - padding, maxs + padding, bucket_size_mm)


def _bounds_bucket_keys(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    bucket_size_mm: float,
) -> Iterator[tuple[int, int, int]]:
    bucket_size = max(float(bucket_size_mm), 1.0e-9)
    lo = np.floor(np.asarray(bounds_min, dtype=float) / bucket_size).astype(int)
    hi = np.floor(np.asarray(bounds_max, dtype=float) / bucket_size).astype(int)
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
