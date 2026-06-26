"""Data structures for sparse 3D lumped thermal graphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .material_library import material_defaults


class EdgeMode(str, Enum):
    """How conductive edge conductances are populated."""

    AUTO = "auto"
    LOADED_G = "loaded_G"

    @classmethod
    def normalize(cls, value: str | "EdgeMode") -> str:
        """Normalize current and legacy saved conduction-mode labels."""
        raw = value.value if isinstance(value, EdgeMode) else str(value)
        if raw in {"auto", "auto_estimated"}:
            return cls.AUTO.value
        if raw in {"loaded_G", "loaded_matrix"}:
            return cls.LOADED_G.value
        return cls.AUTO.value


@dataclass
class HeaterProperties:
    heater_id: int = 0
    heater_min_power_W: float = 0.0
    heater_max_power_W: float = 30.0
    heater_efficiency: float = 1.0


@dataclass
class SensorProperties:
    sensor_id: int = 0
    sensor_noise_std_K: float = 0.0
    sensor_bias_K: float = 0.0
    sensor_time_constant_s: float = 0.0


@dataclass
class PIDControlSettings:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    integral_leak_per_s: float = 0.0
    setpoint: float = 293.15


@dataclass
class ManualHeaterSettings:
    power: float = 0.0


@dataclass
class PIDState:
    integral: float = 0.0
    previous_error: float | None = None


@dataclass
class HeaterControl:
    mode: str = "manual"
    pid: PIDControlSettings = field(default_factory=PIDControlSettings)
    manual: ManualHeaterSettings = field(default_factory=ManualHeaterSettings)
    pid_state: PIDState = field(default_factory=PIDState)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        *,
        initial_temperature_K: float = 293.15,
        default_manual_power_W: float = 0.0,
    ) -> "HeaterControl":
        raw = data or {}
        mode = str(raw.get("mode", "manual"))
        if mode not in {"pid", "mimo", "manual"}:
            mode = "manual"
        pid_data = raw.get("pid", {}) or {}
        manual_data = raw.get("manual", {}) or {}
        state_data = raw.get("pid_state", raw.get("pidState", {}) or {}) or {}
        return cls(
            mode=mode,
            pid=PIDControlSettings(
                kp=float(pid_data.get("kp", 0.0)),
                ki=float(pid_data.get("ki", 0.0)),
                kd=float(pid_data.get("kd", 0.0)),
                integral_leak_per_s=max(
                    0.0,
                    float(
                        pid_data.get(
                            "integral_leak_per_s",
                            pid_data.get("integralLeakPerS", 0.0),
                        )
                    ),
                ),
                setpoint=float(pid_data.get("setpoint", initial_temperature_K)),
            ),
            manual=ManualHeaterSettings(
                power=float(manual_data.get("power", default_manual_power_W))
            ),
            pid_state=PIDState(
                integral=float(state_data.get("integral", 0.0)),
                previous_error=(
                    None
                    if state_data.get("previous_error", state_data.get("previousError")) is None
                    else float(state_data.get("previous_error", state_data.get("previousError")))
                ),
            ),
        )

    def reset_pid_state(self) -> None:
        self.pid_state.integral = 0.0
        self.pid_state.previous_error = None


@dataclass
class NodeProperties:
    node_id: int
    coord: tuple[int, int, int]
    cell_id: str | None = None
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    level: int = 0
    center_mm: tuple[float, float, float] | None = None
    size_mm: tuple[float, float, float] | None = None
    component_name: str = ""
    occupancy_fraction: float = 1.0
    confidence: str = "high"
    warnings: list[str] = field(default_factory=list)
    notes: str = ""
    side_length_m: float = 1.0
    material: str = "generic electronics package"
    rho_kg_m3: float = 2200.0
    cp_J_kgK: float = 800.0
    k_W_mK: float = 2.0
    emissivity: float = 0.85
    mass_kg: float = 1.0
    C_J_K: float = 800.0
    C_manual_override: bool = False
    Grad_W_K: float = 0.0
    initial_temperature_K: float = 293.15
    is_exposed: bool = False
    radiating_area_m2: float = 0.0
    G_rad_W_K: float = 0.0
    R_rad_K_W: float | None = None
    has_heater: bool = False
    heater: HeaterProperties = field(default_factory=HeaterProperties)
    heater_control: HeaterControl = field(default_factory=HeaterControl)
    has_sensor: bool = False
    sensor: SensorProperties = field(default_factory=SensorProperties)
    has_cryocooler: bool = False
    controller_setpoint_K: float = 293.15
    controller_weight: float = 0.0
    sensor_settling_time_s: float = 0.0
    controller_kp_coarse: float = 0.0
    controller_ki_coarse: float = 0.0
    controller_kp_hold: float = 0.0
    controller_ki_hold: float = 0.0

    @classmethod
    def with_material(
        cls,
        node_id: int,
        coord: tuple[int, int, int],
        material: str = "generic electronics package",
        library: dict[str, dict[str, float]] | None = None,
    ) -> "NodeProperties":
        defaults = material_defaults(material, library)
        node = cls(
            node_id=int(node_id),
            coord=tuple(int(v) for v in coord),
            material=material,
            rho_kg_m3=defaults["rho_kg_m3"],
            cp_J_kgK=defaults["cp_J_kgK"],
            k_W_mK=defaults["k_W_mK"],
            emissivity=defaults["emissivity"],
        )
        node.heater.heater_id = node.node_id
        node.sensor.sensor_id = node.node_id
        node.recompute_heat_capacity()
        return node

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeProperties":
        copied = dict(data)
        node_id_value = copied.pop("id") if "id" in copied else copied.pop("node_id")
        node_id = int(node_id_value)
        center_mm = copied.pop("center_mm", None)
        size_mm = copied.pop("size_mm", None)
        component_name = copied.pop("component_name", copied.pop("component", ""))
        material_name = copied.pop("material_name", None)
        tags = copied.pop("tags", {}) or {}
        coord_value = copied.pop("coord", None)
        if coord_value is None and center_mm is not None:
            coord_value = [round(float(v)) for v in center_mm]
        coord = tuple(int(v) for v in (coord_value or (0, 0, 0)))
        heater_data = copied.pop("heater", {}) or {}
        heater_control_data = copied.pop("heater_control", copied.pop("heaterControl", {}) or {}) or {}
        sensor_data = copied.pop("sensor", {}) or {}
        controller_data = copied.pop("controller", copied.pop("mimo_controller", {}) or {}) or {}
        if isinstance(controller_data, dict):
            copied.setdefault(
                "controller_setpoint_K",
                float(controller_data.get("setpoint_K", controller_data.get("setpoint", 293.15))),
            )
            copied.setdefault("controller_weight", float(controller_data.get("weight", 0.0)))
            copied.setdefault(
                "sensor_settling_time_s",
                float(controller_data.get("sensor_settling_time_s", controller_data.get("settling_time_s", 0.0))),
            )
            legacy_kp = controller_data.get("kp_scale", copied.get("controller_kp_scale", 0.0))
            legacy_ki = controller_data.get("ki_scale", copied.get("controller_ki_scale", 0.0))
            copied.setdefault(
                "controller_kp_coarse",
                float(controller_data.get("kp_coarse", controller_data.get("kp", legacy_kp))),
            )
            copied.setdefault(
                "controller_ki_coarse",
                float(controller_data.get("ki_coarse", controller_data.get("ki", legacy_ki))),
            )
            copied.setdefault(
                "controller_kp_hold",
                float(controller_data.get("kp_hold", copied.get("controller_kp_hold", legacy_kp))),
            )
            copied.setdefault(
                "controller_ki_hold",
                float(controller_data.get("ki_hold", copied.get("controller_ki_hold", legacy_ki))),
            )
        if "controller_kp_scale" in copied and "controller_kp_coarse" not in copied:
            copied["controller_kp_coarse"] = float(copied["controller_kp_scale"])
            copied.setdefault("controller_kp_hold", float(copied["controller_kp_scale"]))
        if "controller_ki_scale" in copied and "controller_ki_coarse" not in copied:
            copied["controller_ki_coarse"] = float(copied["controller_ki_scale"])
            copied.setdefault("controller_ki_hold", float(copied["controller_ki_scale"]))
        copied.pop("controller_kp_scale", None)
        copied.pop("controller_ki_scale", None)
        copied.pop("heat_sink", None)
        copied.pop("heatSink", None)
        if "heater_id" in tags:
            heater_data.setdefault("heater_id", tags.get("heater_id") or node_id)
        if "sensor_id" in tags:
            sensor_data.setdefault("sensor_id", tags.get("sensor_id") or node_id)
        copied.setdefault("has_heater", bool(tags.get("heater", False)))
        copied.setdefault("has_sensor", bool(tags.get("sensor", False)))
        copied.pop("has_heat_sink", None)
        copied.setdefault(
            "has_cryocooler",
            bool(tags.get("cryocooler", tags.get("active_cooler", False))),
        )
        if tags.get("notes") and "notes" not in copied:
            copied["notes"] = str(tags.get("notes"))
        if material_name and "material" not in copied:
            copied["material"] = str(material_name)
        copied.pop("volume_m3", None)
        radiation = copied.pop("radiation", {}) or {}
        copied.setdefault("is_exposed", bool(radiation.get("is_exposed", False)))
        copied.setdefault("radiating_area_m2", float(radiation.get("radiating_area_m2", 0.0)))
        copied.setdefault("G_rad_W_K", float(radiation.get("G_rad_W_K", copied.get("Grad_W_K", 0.0))))
        copied.setdefault("R_rad_K_W", radiation.get("R_rad_K_W"))
        copied.setdefault("initial_temperature_K", float(copied.get("initial_temperature_K", 293.15)))
        copied.pop("dominant_component", None)
        copied.pop("dominant_material", None)
        copied.pop("pos", None)
        if size_mm is not None and "side_length_m" not in copied:
            copied["side_length_m"] = max(float(v) for v in size_mm) / 1000.0
        node = cls(
            node_id=node_id,
            coord=coord,
            center_mm=tuple(float(v) for v in center_mm) if center_mm is not None else None,
            size_mm=tuple(float(v) for v in size_mm) if size_mm is not None else None,
            component_name=str(component_name),
            **copied,
        )
        if not isinstance(node.heater, HeaterProperties):
            node.heater = HeaterProperties(**heater_data)
        else:
            node.heater = HeaterProperties(**heater_data)
        if node.has_heater:
            node.has_sensor = True
        if not isinstance(node.heater_control, HeaterControl):
            node.heater_control = HeaterControl.from_dict(
                heater_control_data,
                initial_temperature_K=node.initial_temperature_K,
                default_manual_power_W=float(node.heater.heater_max_power_W) * float(node.heater.heater_efficiency),
            )
        else:
            node.heater_control = HeaterControl.from_dict(
                heater_control_data,
                initial_temperature_K=node.initial_temperature_K,
                default_manual_power_W=float(node.heater.heater_max_power_W) * float(node.heater.heater_efficiency),
            )
        if not isinstance(node.sensor, SensorProperties):
            node.sensor = SensorProperties(**sensor_data)
        else:
            node.sensor = SensorProperties(**sensor_data)
        return node

    def to_dict(self, include_id_key: bool = False) -> dict[str, Any]:
        data = asdict(self)
        data["coord"] = list(self.coord)
        if self.center_mm is not None:
            data["center_mm"] = list(self.center_mm)
        if self.size_mm is not None:
            data["size_mm"] = list(self.size_mm)
        if include_id_key:
            data["id"] = data.pop("node_id")
        return data

    def to_octree_node_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "cell_id": self.cell_id or f"cell_{self.node_id}",
            "center_mm": list(self.center_mm or self.center),
            "size_mm": list(self.size_mm or (self.side_length_m * 1000.0,) * 3),
            "level": self.level,
            "component_name": self.component_name,
            "material_name": self.material,
            "volume_m3": self.volume_m3,
            "mass_kg": self.mass_kg,
            "C_J_K": self.C_J_K,
            "initial_temperature_K": self.initial_temperature_K,
            "occupancy_fraction": self.occupancy_fraction,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "radiation": {
                "is_exposed": self.is_exposed,
                "radiating_area_m2": self.radiating_area_m2,
                "emissivity": self.emissivity,
                "G_rad_W_K": self.G_rad_W_K,
                "R_rad_K_W": self.R_rad_K_W,
            },
            "tags": {
                "heater": self.has_heater,
                "sensor": self.has_sensor,
                "cryocooler": self.has_cryocooler,
                "heater_id": self.heater.heater_id if self.has_heater else None,
                "sensor_id": self.sensor.sensor_id if self.has_sensor else None,
                "notes": self.notes,
            },
            "heater_control": asdict(self.heater_control),
            "controller": {
                "setpoint_K": self.controller_setpoint_K,
                "weight": self.controller_weight,
                "sensor_settling_time_s": self.sensor_settling_time_s,
                "kp_coarse": self.controller_kp_coarse,
                "ki_coarse": self.controller_ki_coarse,
                "kp_hold": self.controller_kp_hold,
                "ki_hold": self.controller_ki_hold,
            },
        }

    def apply_material_defaults(self, library: dict[str, dict[str, float]] | None = None) -> None:
        defaults = material_defaults(self.material, library)
        self.rho_kg_m3 = defaults["rho_kg_m3"]
        self.cp_J_kgK = defaults["cp_J_kgK"]
        self.k_W_mK = defaults["k_W_mK"]
        self.emissivity = defaults["emissivity"]
        self.recompute_heat_capacity()

    def recompute_heat_capacity(self) -> None:
        if not self.C_manual_override:
            self.C_J_K = self.mass_kg * self.cp_J_kgK

    @property
    def center(self) -> tuple[float, float, float]:
        if self.center_mm is not None:
            return tuple(float(v) for v in self.center_mm)
        return tuple(float(v) for v in self.coord)

    @property
    def volume_m3(self) -> float:
        if self.size_mm is not None:
            sx, sy, sz = self.size_mm
            return max(0.0, float(sx) * float(sy) * float(sz) * 1.0e-9)
        return max(0.0, float(self.side_length_m) ** 3)


@dataclass
class EdgeProperties:
    source: int
    target: int
    Gij_W_K: float
    source_metadata: str = EdgeMode.AUTO.value
    edge_id: str | None = None
    edge_type: str = "internal_conduction"
    shared_area_m2: float = 0.0
    distance_m: float = 0.0
    contact_confidence: str = "medium"
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EdgeProperties":
        copied = dict(data)
        source_metadata = copied.pop("source_metadata", copied.pop("source_type", copied.get("source", "")))
        if "node_i" in copied:
            source = copied.pop("node_i")
        else:
            source = copied.pop("source")
        if "node_j" in copied:
            target = copied.pop("node_j")
        else:
            target = copied.pop("target")
        return cls(
            source=int(source),
            target=int(target),
            Gij_W_K=float(copied.pop("Gij_W_K", copied.pop("G_W_K", 0.0))),
            source_metadata=str(source_metadata),
            edge_id=copied.pop("edge_id", None),
            edge_type=str(copied.pop("edge_type", "internal_conduction")),
            shared_area_m2=float(copied.pop("shared_area_m2", 0.0)),
            distance_m=float(copied.pop("distance_m", 0.0)),
            contact_confidence=str(copied.pop("contact_confidence", "medium")),
            warnings=list(copied.pop("warnings", []) or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_octree_edge_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id or f"edge_{self.source}_{self.target}",
            "node_i": self.source,
            "node_j": self.target,
            "edge_type": self.edge_type,
            "G_W_K": self.Gij_W_K,
            "shared_area_m2": self.shared_area_m2,
            "distance_m": self.distance_m,
            "contact_confidence": self.contact_confidence,
            "source": self.source_metadata,
            "warnings": list(self.warnings),
        }


@dataclass
class GraphMetadata:
    graph_name: str = "untitled_graph"
    created_at: str = field(default_factory=lambda: utc_now_iso())
    updated_at: str = field(default_factory=lambda: utc_now_iso())
    T_sur_K: float = 293.15
    edge_mode: str = EdgeMode.AUTO.value
    app_version: str = "graph_visualizer 0.1"
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GraphMetadata":
        if not data:
            return cls()
        values = {field_name: data.get(field_name) for field_name in cls.__dataclass_fields__}
        if values.get("edge_mode") is not None:
            values["edge_mode"] = EdgeMode.normalize(values["edge_mode"])
        return cls(**{key: value for key, value in values.items() if value is not None})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThermalGraphModel:
    """Sparse graph model keyed by canonical integer node IDs."""

    metadata: GraphMetadata = field(default_factory=GraphMetadata)
    nodes: dict[int, NodeProperties] = field(default_factory=dict)
    edges: dict[tuple[int, int], EdgeProperties] = field(default_factory=dict)
    material_library: dict[str, dict[str, float]] = field(default_factory=dict)
    octree_graph_data: dict[str, Any] = field(default_factory=dict)
    controller_gain_matrix: dict[int, dict[int, float]] = field(default_factory=dict)

    def add_node(self, node: NodeProperties) -> None:
        self._check_unique(node.node_id, node.coord)
        self.nodes[node.node_id] = node
        self.touch()

    def update_node(self, original_node_id: int, node: NodeProperties) -> None:
        if original_node_id not in self.nodes:
            raise ValueError(f"Node {original_node_id} does not exist.")
        for existing_id, existing in self.nodes.items():
            if existing_id == original_node_id:
                continue
            if existing_id == node.node_id:
                raise ValueError(f"Duplicate node_id {node.node_id}.")
            if existing.coord == node.coord:
                raise ValueError(f"Duplicate coord {node.coord}.")
        old_edges = list(self.edges.values())
        del self.nodes[original_node_id]
        self.nodes[node.node_id] = node
        self.edges = {}
        for edge in old_edges:
            source = node.node_id if edge.source == original_node_id else edge.source
            target = node.node_id if edge.target == original_node_id else edge.target
            if source in self.nodes and target in self.nodes and source != target:
                self.set_edge(source, target, edge.Gij_W_K, edge.source_metadata)
        self.touch()

    def delete_node(self, node_id: int) -> None:
        self.nodes.pop(node_id, None)
        self.edges = {
            key: edge
            for key, edge in self.edges.items()
            if edge.source != node_id and edge.target != node_id
        }
        self.prune_controller_gain_matrix()
        self.touch()

    def set_edge(
        self,
        source: int,
        target: int,
        Gij_W_K: float,
        source_metadata: str = EdgeMode.AUTO.value,
        **metadata: Any,
    ) -> None:
        if source == target:
            return
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Both edge endpoints must exist.")
        key = edge_key(source, target)
        low, high = key
        self.edges[key] = EdgeProperties(low, high, float(Gij_W_K), source_metadata, **metadata)
        self.touch()

    def clear_edges(self) -> None:
        self.edges.clear()
        self.touch()

    def controller_gain(self, sensor_node_id: int, heater_node_id: int) -> float:
        row = self.controller_gain_matrix.get(int(sensor_node_id), {})
        return float(row.get(int(heater_node_id), 0.0))

    def set_controller_gain(self, sensor_node_id: int, heater_node_id: int, value: float) -> None:
        sensor_id = int(sensor_node_id)
        heater_id = int(heater_node_id)
        row = self.controller_gain_matrix.setdefault(sensor_id, {})
        number = float(value)
        if abs(number) <= 0.0:
            row.pop(heater_id, None)
        else:
            row[heater_id] = number
        if not row:
            self.controller_gain_matrix.pop(sensor_id, None)
        self.touch()

    def prune_controller_gain_matrix(self) -> None:
        sensor_ids = {node_id for node_id, node in self.nodes.items() if node.has_sensor}
        heater_ids = {node_id for node_id, node in self.nodes.items() if node.has_heater}
        pruned: dict[int, dict[int, float]] = {}
        for sensor_id, row in self.controller_gain_matrix.items():
            sensor_key = int(sensor_id)
            if sensor_key not in sensor_ids:
                continue
            kept = {
                int(heater_id): float(value)
                for heater_id, value in row.items()
                if int(heater_id) in heater_ids and abs(float(value)) > 0.0
            }
            if kept:
                pruned[sensor_key] = kept
        self.controller_gain_matrix = pruned

    def ordered_node_ids(self) -> list[int]:
        return sorted(self.nodes)

    def coord_index(self) -> dict[tuple[int, int, int], int]:
        return {node.coord: node_id for node_id, node in self.nodes.items()}

    def find_by_coord(self, coord: tuple[int, int, int]) -> NodeProperties | None:
        node_id = self.coord_index().get(coord)
        return self.nodes.get(node_id) if node_id is not None else None

    def to_graph3d_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                self.nodes[node_id].to_dict(include_id_key=True)
                for node_id in self.ordered_node_ids()
            ],
            "edges": [
                edge.to_dict()
                for _, edge in sorted(self.edges.items(), key=lambda item: item[0])
            ],
            "controller_gain_matrix": {
                str(sensor_id): {
                    str(heater_id): float(value)
                    for heater_id, value in sorted(row.items(), key=lambda item: int(item[0]))
                }
                for sensor_id, row in sorted(self.controller_gain_matrix.items(), key=lambda item: int(item[0]))
            },
        }

    def to_octree_graph_dict(self) -> dict[str, Any]:
        components = sorted({node.component_name for node in self.nodes.values() if node.component_name})
        data = dict(self.octree_graph_data) if self.octree_graph_data else {}
        data.update({
            "metadata": self.metadata.to_dict(),
            "input_files": data.get("input_files", {}),
            "parameters": data.get("parameters", {}),
            "materials_used": data.get("materials_used", sorted({node.material for node in self.nodes.values()})),
            "component_mapping": data.get("component_mapping", {name: name for name in components}),
            "octree_cells": [
                {
                    "cell_id": node.cell_id or f"cell_{node.node_id}",
                    "parent_id": node.parent_id,
                    "children_ids": list(node.children_ids),
                    "level": node.level,
                    "center_mm": list(node.center_mm or node.center),
                    "size_mm": list(node.size_mm or (node.side_length_m * 1000.0,) * 3),
                    "dominant_component": node.component_name,
                    "dominant_material": node.material,
                    "occupancy_fraction": node.occupancy_fraction,
                    "confidence": node.confidence,
                    "warnings": list(node.warnings),
                }
                for node in self.nodes.values()
            ],
            "graph_nodes": [
                self.nodes[node_id].to_octree_node_dict()
                for node_id in self.ordered_node_ids()
            ],
            "graph_edges": [
                edge.to_octree_edge_dict()
                for _, edge in sorted(self.edges.items(), key=lambda item: item[0])
            ],
            "warnings": [],
            "heater_sensor_tags": {
                str(node_id): node.to_octree_node_dict()["tags"]
                for node_id, node in self.nodes.items()
                if node.has_heater or node.has_sensor or node.has_cryocooler or node.notes
            },
            "controller_gain_matrix": {
                str(sensor_id): {
                    str(heater_id): float(value)
                    for heater_id, value in sorted(row.items(), key=lambda item: int(item[0]))
                }
                for sensor_id, row in sorted(self.controller_gain_matrix.items(), key=lambda item: int(item[0]))
            },
            "validation_results": data.get("validation_results", {}),
        })
        return data

    @classmethod
    def from_graph3d_dict(
        cls,
        data: dict[str, Any],
        metadata: GraphMetadata | None = None,
        material_library: dict[str, dict[str, float]] | None = None,
    ) -> "ThermalGraphModel":
        model = cls(metadata=metadata or GraphMetadata(), material_library=material_library or {})
        for raw_node in data.get("nodes", []):
            node = NodeProperties.from_dict(dict(raw_node))
            model.add_node(node)
        for raw_edge in data.get("edges", []):
            edge = EdgeProperties.from_dict(dict(raw_edge))
            if edge.source in model.nodes and edge.target in model.nodes:
                model.set_edge(edge.source, edge.target, edge.Gij_W_K, edge.source_metadata)
        model.controller_gain_matrix = _parse_controller_gain_matrix(data.get("controller_gain_matrix"))
        model.prune_controller_gain_matrix()
        return model

    @classmethod
    def from_octree_graph_dict(cls, data: dict[str, Any]) -> "ThermalGraphModel":
        metadata = GraphMetadata.from_dict(data.get("metadata"))
        model = cls(metadata=metadata, material_library=data.get("material_library") or {})
        model.octree_graph_data = dict(data)
        model.controller_gain_matrix = _parse_controller_gain_matrix(data.get("controller_gain_matrix"))
        for raw_node in data.get("graph_nodes", []):
            node = NodeProperties.from_dict(dict(raw_node))
            model.add_node(node)
        for raw_edge in data.get("graph_edges", []):
            edge = EdgeProperties.from_dict(dict(raw_edge))
            if edge.source in model.nodes and edge.target in model.nodes:
                model.set_edge(
                    edge.source,
                    edge.target,
                    edge.Gij_W_K,
                    edge.source_metadata,
                    edge_id=edge.edge_id,
                    edge_type=edge.edge_type,
                    shared_area_m2=edge.shared_area_m2,
                    distance_m=edge.distance_m,
                    contact_confidence=edge.contact_confidence,
                    warnings=edge.warnings,
                )
        model.prune_controller_gain_matrix()
        return model

    def _check_unique(self, node_id: int, coord: tuple[int, int, int]) -> None:
        if node_id in self.nodes:
            raise ValueError(f"Duplicate node_id {node_id}.")
        if coord in self.coord_index():
            raise ValueError(f"Duplicate coord {coord}.")

    def touch(self) -> None:
        self.metadata.updated_at = utc_now_iso()


def edge_key(source: int, target: int) -> tuple[int, int]:
    return (min(int(source), int(target)), max(int(source), int(target)))


def _parse_controller_gain_matrix(raw: Any) -> dict[int, dict[int, float]]:
    if not isinstance(raw, dict):
        return {}
    parsed: dict[int, dict[int, float]] = {}
    for raw_sensor_id, raw_row in raw.items():
        if not isinstance(raw_row, dict):
            continue
        try:
            sensor_id = int(raw_sensor_id)
        except (TypeError, ValueError):
            continue
        row: dict[int, float] = {}
        for raw_heater_id, raw_value in raw_row.items():
            try:
                heater_id = int(raw_heater_id)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if abs(value) > 0.0:
                row[heater_id] = value
        if row:
            parsed[sensor_id] = row
    return parsed


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
