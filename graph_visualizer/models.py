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
class NodeProperties:
    node_id: int
    coord: tuple[int, int, int]
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
    has_heater: bool = False
    heater: HeaterProperties = field(default_factory=HeaterProperties)
    has_sensor: bool = False
    sensor: SensorProperties = field(default_factory=SensorProperties)

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
        coord = tuple(int(v) for v in copied.pop("coord"))
        heater_data = copied.pop("heater", {}) or {}
        sensor_data = copied.pop("sensor", {}) or {}
        copied.pop("pos", None)
        node = cls(node_id=node_id, coord=coord, **copied)
        if not isinstance(node.heater, HeaterProperties):
            node.heater = HeaterProperties(**heater_data)
        else:
            node.heater = HeaterProperties(**heater_data)
        if not isinstance(node.sensor, SensorProperties):
            node.sensor = SensorProperties(**sensor_data)
        else:
            node.sensor = SensorProperties(**sensor_data)
        return node

    def to_dict(self, include_id_key: bool = False) -> dict[str, Any]:
        data = asdict(self)
        data["coord"] = list(self.coord)
        if include_id_key:
            data["id"] = data.pop("node_id")
        return data

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
        return tuple(float(v) for v in self.coord)


@dataclass
class EdgeProperties:
    source: int
    target: int
    Gij_W_K: float
    source_metadata: str = EdgeMode.AUTO.value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EdgeProperties":
        copied = dict(data)
        source_metadata = copied.pop("source_metadata", copied.pop("source_type", ""))
        return cls(
            source=int(copied.pop("source")),
            target=int(copied.pop("target")),
            Gij_W_K=float(copied.pop("Gij_W_K", 0.0)),
            source_metadata=str(source_metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        self.touch()

    def set_edge(
        self,
        source: int,
        target: int,
        Gij_W_K: float,
        source_metadata: str = EdgeMode.AUTO.value,
    ) -> None:
        if source == target:
            return
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Both edge endpoints must exist.")
        key = edge_key(source, target)
        low, high = key
        self.edges[key] = EdgeProperties(low, high, float(Gij_W_K), source_metadata)
        self.touch()

    def clear_edges(self) -> None:
        self.edges.clear()
        self.touch()

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
        }

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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
