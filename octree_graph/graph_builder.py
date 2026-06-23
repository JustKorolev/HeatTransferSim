"""Build thermal graph nodes and conductance edges from octree leaves."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .load_contact_report import ContactReport
from .materials import Material, resolve_material
from .octree import OctreeCell


@dataclass
class GraphBuildResult:
    nodes: list[dict]
    edges: list[dict]
    warnings: list[str]


def build_graph(
    leaves: list[OctreeCell],
    contact_report: ContactReport,
    materials: dict[str, Material],
    warnings: list[str],
    default_contact_G_W_K: float = 0.1,
) -> GraphBuildResult:
    solid = [cell for cell in leaves if not cell.is_empty]
    volumes_by_component: dict[str, float] = {}
    for cell in solid:
        if cell.dominant_component:
            volumes_by_component[cell.dominant_component] = (
                volumes_by_component.get(cell.dominant_component, 0.0) + cell.volume_m3
            )
    nodes: list[dict] = []
    for node_id, cell in enumerate(solid):
        material = resolve_material(cell.dominant_material, materials, warnings)
        component = cell.dominant_component or ""
        if component in contact_report.component_masses_kg and volumes_by_component.get(component, 0.0) > 0:
            mass = contact_report.component_masses_kg[component] * cell.volume_m3 / volumes_by_component[component]
        else:
            mass = material.density_kg_m3 * cell.volume_m3
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
                "occupancy_fraction": max(cell.occupancy.values(), default=0.0),
                "confidence": cell.confidence,
                "warnings": list(cell.warnings),
                "tags": {"heater": False, "sensor": False, "heater_id": None, "sensor_id": None, "notes": ""},
            }
        )
    cell_to_node = {node["cell_id"]: node for node in nodes}
    edges: list[dict] = []
    for edge_index, (a, b) in enumerate(combinations(solid, 2)):
        area_mm2, distance_mm = _shared_face_area_and_distance(a, b)
        if area_mm2 <= 0.0:
            continue
        node_a = cell_to_node[a.cell_id]
        node_b = cell_to_node[b.cell_id]
        material_a = resolve_material(str(node_a["material_name"]), materials, warnings)
        material_b = resolve_material(str(node_b["material_name"]), materials, warnings)
        same_component = node_a["component_name"] == node_b["component_name"]
        excel_contact = contact_report.has_pair(str(node_a["component_name"]), str(node_b["component_name"]))
        if same_component:
            edge_type = "internal_conduction"
            k_eff = harmonic_mean(material_a.k_W_mK, material_b.k_W_mK)
            G = k_eff * (area_mm2 * 1.0e-6) / max(distance_mm * 1.0e-3, 1.0e-12)
            source = "geometry"
            confidence = "high"
        elif excel_contact:
            edge_type = "excel_contact"
            G = default_contact_G_W_K
            source = "both"
            confidence = "medium"
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
                "warnings": [] if confidence != "low" else ["Geometry adjacency is not confirmed by Excel contact metadata."],
            }
        )
    return GraphBuildResult(nodes=nodes, edges=edges, warnings=warnings)


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
