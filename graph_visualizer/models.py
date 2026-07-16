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
    lambda_order: float = 1.0
    mu_order: float = 1.0
    integral_leak_per_s: float = 0.0
    setpoint: float = 293.15


@dataclass
class ManualHeaterSettings:
    power: float = 0.0


@dataclass
class PIDState:
    integral: float = 0.0
    previous_error: float | None = None
    error_history: list[float] = field(default_factory=list)


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
                lambda_order=max(
                    0.0,
                    float(pid_data.get("lambda_order", pid_data.get("lambda", pid_data.get("lambdaOrder", 1.0)))),
                ),
                mu_order=max(
                    0.0,
                    float(pid_data.get("mu_order", pid_data.get("mu", pid_data.get("muOrder", 1.0)))),
                ),
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
                error_history=[
                    float(value)
                    for value in state_data.get("error_history", state_data.get("errorHistory", []))
                ],
            ),
        )

    def reset_pid_state(self) -> None:
        self.pid_state.integral = 0.0
        self.pid_state.previous_error = None
        self.pid_state.error_history = []


@dataclass
class NodeProperties:
    node_id: int
    coord: tuple[int, int, int]
    cell_id: str | None = None
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    level: int = 0
    node_type: str = ""
    source_components: list[str] = field(default_factory=list)
    source_node_ids: list[int] = field(default_factory=list)
    source_cell_ids: list[str] = field(default_factory=list)
    role_source_components: list[str] = field(default_factory=list)
    source_bounds_mm: dict[str, Any] = field(default_factory=dict)
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
    is_heater: bool = False
    heater: HeaterProperties = field(default_factory=HeaterProperties)
    heater_control: HeaterControl = field(default_factory=HeaterControl)
    is_sensor: bool = False
    sensor: SensorProperties = field(default_factory=SensorProperties)
    assigned_sensor_id: int | None = None
    assigned_heater_id: int | None = None
    assigned_heater_ids: list[int] = field(default_factory=list)
    sensor_pair_distance_mm: float | None = None
    power_deposition_node_ids: list[int] = field(default_factory=list)
    power_deposition_weights: list[float] = field(default_factory=list)
    heater_attached: bool = True
    heater_valid: bool = True
    heater_warning: str = ""
    sensor_connected_node_ids: list[int] = field(default_factory=list)
    readout_node_ids: list[int] = field(default_factory=list)
    readout_weights: list[float] = field(default_factory=list)
    sensor_monitor_only: bool = False
    sensor_valid: bool = True
    sensor_control_mode: str = "manual"
    sensor_manual_power_W: float = 0.0
    has_cryocooler: bool = False
    controller_setpoint_K: float = 293.15
    controller_weight: float = 0.0
    sensor_settling_time_s: float = 0.0
    controller_kp_coarse: float = 0.0
    controller_ki_coarse: float = 0.0
    controller_kp_hold: float = 0.0
    controller_ki_hold: float = 0.0
    controller_kd_coarse: float = 0.0
    controller_kd_hold: float = 0.0
    controller_lambda_order: float = 1.0
    controller_mu_order: float = 1.0
    controller_integral_leak_per_s: float = 0.0

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
        node_type = str(copied.pop("node_type", ""))
        source_components = copied.pop("source_components", []) or []
        source_node_ids = copied.pop("source_node_ids", []) or []
        source_cell_ids = copied.pop("source_cell_ids", []) or []
        role_source_components = copied.pop("role_source_components", []) or []
        source_bounds_mm = copied.pop("source_bounds_mm", {}) or {}
        is_heater_value = copied.pop("is_heater", None)
        is_sensor_value = copied.pop("is_sensor", None)
        tags = copied.pop("tags", {}) or {}
        copied.pop("warning_tags", None)
        coord_value = copied.pop("coord", None)
        if coord_value is None and center_mm is not None:
            coord_value = [round(float(v)) for v in center_mm]
        coord, coord_warning = _safe_coord_tuple(coord_value, fallback=(node_id, 0, 0))
        heater_data = copied.pop("heater", {}) or {}
        heater_control_data = copied.pop("heater_control", copied.pop("heaterControl", {}) or {}) or {}
        sensor_data = copied.pop("sensor", {}) or {}
        controller_data = copied.pop("controller", copied.pop("mimo_controller", {}) or {}) or {}
        physical_device_data = copied.pop("physical_device", copied.pop("physicalDevice", {}) or {}) or {}
        if isinstance(physical_device_data, dict):
            device_kind = str(physical_device_data.get("kind", "")).lower()
            if is_heater_value is None and "heater" in device_kind:
                is_heater_value = True
            if is_sensor_value is None and "sensor" in device_kind:
                is_sensor_value = True
            if not source_components:
                source_components = list(physical_device_data.get("source_components", []) or [])
            if not role_source_components:
                role_source_components = list(source_components)
            device_bounds = physical_device_data.get("bounds_mm", {}) or {}
            if not source_bounds_mm and isinstance(device_bounds, dict):
                source_bounds_mm = device_bounds
            if not node_type and device_kind:
                node_type = device_kind if device_kind in {"heater", "sensor"} else f"physical_{device_kind}"
        raw_had_sensor_control_mode = "sensor_control_mode" in copied
        raw_had_controller_setpoint = (
            "controller_setpoint_K" in copied
            or (isinstance(controller_data, dict) and ("setpoint_K" in controller_data or "setpoint" in controller_data))
        )
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
            copied.setdefault(
                "controller_kd_coarse",
                float(controller_data.get("kd_coarse", controller_data.get("kd", copied.get("controller_kd_coarse", 0.0)))),
            )
            copied.setdefault(
                "controller_kd_hold",
                float(controller_data.get("kd_hold", copied.get("controller_kd_hold", 0.0))),
            )
            copied.setdefault(
                "controller_lambda_order",
                max(
                    0.0,
                    float(
                        controller_data.get(
                            "lambda_order",
                            controller_data.get("lambda", copied.get("controller_lambda_order", 1.0)),
                        )
                    ),
                ),
            )
            copied.setdefault(
                "controller_mu_order",
                max(
                    0.0,
                    float(
                        controller_data.get(
                            "mu_order",
                            controller_data.get("mu", copied.get("controller_mu_order", 1.0)),
                        )
                    ),
                ),
            )
            copied.setdefault(
                "controller_integral_leak_per_s",
                float(controller_data.get("integral_leak_per_s", copied.get("controller_integral_leak_per_s", 0.0))),
            )
        copied.pop("controller_integral_negative_error_leak_per_s", None)
        copied.pop("integral_negative_error_leak_per_s", None)
        copied.pop("negative_error_leak_per_s", None)
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
        copied.setdefault(
            "is_heater",
            bool(
                tags.get("heater")
                if "heater" in tags
                else is_heater_value
                if is_heater_value is not None
                else node_type == "heater"
            ),
        )
        copied.setdefault(
            "is_sensor",
            bool(
                tags.get("sensor")
                if "sensor" in tags
                else is_sensor_value
                if is_sensor_value is not None
                else node_type == "sensor"
            ),
        )
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
        center_tuple, center_warning = _finite_vector3(center_mm, field_name="center_mm")
        size_tuple, size_warning = _finite_vector3(size_mm, field_name="size_mm", positive=True)
        source_bounds_mm, bounds_warning = _finite_source_bounds(source_bounds_mm)
        existing_warnings = list(copied.get("warnings", []) or [])
        for warning in (coord_warning, center_warning, size_warning, bounds_warning):
            if warning:
                existing_warnings.append(warning)
        if existing_warnings:
            copied["warnings"] = existing_warnings
        radiation = copied.pop("radiation", {}) or {}
        copied.setdefault("is_exposed", bool(radiation.get("is_exposed", False)))
        copied.setdefault("radiating_area_m2", float(radiation.get("radiating_area_m2", 0.0)))
        copied.setdefault("G_rad_W_K", float(radiation.get("G_rad_W_K", copied.get("Grad_W_K", 0.0))))
        copied.setdefault("R_rad_K_W", radiation.get("R_rad_K_W"))
        copied.setdefault("initial_temperature_K", float(copied.get("initial_temperature_K", 293.15)))
        copied.pop("dominant_component", None)
        copied.pop("dominant_material", None)
        copied.pop("pos", None)
        if size_tuple is not None and "side_length_m" not in copied:
            copied["side_length_m"] = max(float(v) for v in size_tuple) / 1000.0
        is_sensor_for_migration = bool(copied.get("is_sensor", False))
        old_mode = str(heater_control_data.get("mode", "manual")) if isinstance(heater_control_data, dict) else "manual"
        old_pid = heater_control_data.get("pid", {}) if isinstance(heater_control_data, dict) else {}
        old_manual = heater_control_data.get("manual", {}) if isinstance(heater_control_data, dict) else {}
        if is_sensor_for_migration and not raw_had_sensor_control_mode:
            has_controller_gains = any(
                float(copied.get(key, 0.0) or 0.0) > 0.0
                for key in (
                    "controller_weight",
                    "controller_kp_coarse",
                    "controller_ki_coarse",
                    "controller_kd_coarse",
                    "controller_kp_hold",
                    "controller_ki_hold",
                    "controller_kd_hold",
                )
            )
            copied["sensor_control_mode"] = "mimo" if old_mode == "mimo" or has_controller_gains else "manual"
        copied["sensor_control_mode"] = _normalize_sensor_control_mode(copied.get("sensor_control_mode", "manual"))
        if is_sensor_for_migration and not raw_had_controller_setpoint and isinstance(old_pid, dict):
            if "setpoint" in old_pid:
                copied["controller_setpoint_K"] = float(old_pid.get("setpoint", copied.get("controller_setpoint_K", 293.15)))
        if "sensor_manual_power_W" not in copied and isinstance(old_manual, dict):
            copied["sensor_manual_power_W"] = float(old_manual.get("power", 0.0))
        copied["assigned_sensor_id"] = _optional_int(copied.get("assigned_sensor_id"))
        copied["assigned_heater_id"] = _optional_int(copied.get("assigned_heater_id"))
        assigned_heater_ids = _int_list(copied.get("assigned_heater_ids", []))
        if copied["assigned_heater_id"] is not None and int(copied["assigned_heater_id"]) not in assigned_heater_ids:
            assigned_heater_ids.append(int(copied["assigned_heater_id"]))
        copied["assigned_heater_ids"] = sorted(set(assigned_heater_ids))
        if copied["assigned_heater_id"] is None and copied["assigned_heater_ids"]:
            copied["assigned_heater_id"] = int(copied["assigned_heater_ids"][0])
        copied["sensor_pair_distance_mm"] = _optional_float(copied.get("sensor_pair_distance_mm"))
        copied["power_deposition_node_ids"] = [
            int(value)
            for value in (
                copied.get("power_deposition_node_ids", copied.get("heater_power_deposition_node_ids", [])) or []
            )
            if _can_int(value)
        ]
        copied["power_deposition_weights"] = [
            float(value)
            for value in (
                copied.get("power_deposition_weights", copied.get("heater_power_deposition_weights", [])) or []
            )
            if _can_float(value)
        ]
        copied["heater_attached"] = bool(copied.get("heater_attached", copied.get("heater_valid", True)))
        copied["heater_valid"] = bool(copied.get("heater_valid", copied.get("heater_attached", True)))
        copied["heater_warning"] = str(copied.get("heater_warning", "") or "")
        copied["sensor_connected_node_ids"] = [
            int(value)
            for value in (copied.get("sensor_connected_node_ids", []) or [])
            if _can_int(value)
        ]
        copied["readout_node_ids"] = [
            int(value)
            for value in (
                copied.get("readout_node_ids", copied.get("sensor_readout_node_ids", copied["sensor_connected_node_ids"]))
                or []
            )
            if _can_int(value)
        ]
        copied["readout_weights"] = [
            float(value)
            for value in (copied.get("readout_weights", copied.get("sensor_readout_weights", [])) or [])
            if _can_float(value)
        ]
        node = cls(
            node_id=node_id,
            coord=coord,
            center_mm=center_tuple,
            size_mm=size_tuple,
            node_type=node_type,
            source_components=[str(value) for value in source_components],
            source_node_ids=[int(value) for value in source_node_ids],
            source_cell_ids=[str(value) for value in source_cell_ids],
            role_source_components=[str(value) for value in role_source_components],
            source_bounds_mm=source_bounds_mm,
            component_name=str(component_name),
            **copied,
        )
        if not isinstance(node.heater, HeaterProperties):
            node.heater = HeaterProperties(**heater_data)
        else:
            node.heater = HeaterProperties(**heater_data)
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
        if node.is_heater and not int(node.heater.heater_id):
            node.heater.heater_id = node.node_id
        if node.is_sensor and not int(node.sensor.sensor_id):
            node.sensor.sensor_id = node.node_id
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
        tags: dict[str, Any] = {
            "cryocooler": self.has_cryocooler,
            "notes": self.notes,
        }
        if not self.is_cad_role_node:
            tags.update(
                {
                    "heater_id": self.heater.heater_id if self.is_heater else None,
                    "sensor_id": self.sensor.sensor_id if self.is_sensor else None,
                }
            )
        data = {
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
            "is_heater": self.is_heater,
            "is_sensor": self.is_sensor,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "radiation": {
                "is_exposed": self.is_exposed,
                "radiating_area_m2": self.radiating_area_m2,
                "emissivity": self.emissivity,
                "G_rad_W_K": self.G_rad_W_K,
                "R_rad_K_W": self.R_rad_K_W,
            },
            "tags": tags,
            "heater_control": asdict(self.heater_control),
            "controller": {
                "setpoint_K": self.controller_setpoint_K,
                "weight": self.controller_weight,
                "sensor_settling_time_s": self.sensor_settling_time_s,
                "kp_coarse": self.controller_kp_coarse,
                "ki_coarse": self.controller_ki_coarse,
                "kp_hold": self.controller_kp_hold,
                "ki_hold": self.controller_ki_hold,
                "kd_coarse": self.controller_kd_coarse,
                "kd_hold": self.controller_kd_hold,
                "lambda_order": self.controller_lambda_order,
                "mu_order": self.controller_mu_order,
                "integral_leak_per_s": self.controller_integral_leak_per_s,
            },
        }
        if self.node_type:
            data["node_type"] = self.node_type
        if self.source_components:
            data["source_components"] = list(self.source_components)
        if self.source_node_ids:
            data["source_node_ids"] = [int(value) for value in self.source_node_ids]
        if self.source_cell_ids:
            data["source_cell_ids"] = [str(value) for value in self.source_cell_ids]
        if self.role_source_components:
            data["role_source_components"] = list(self.role_source_components)
        if self.source_bounds_mm:
            data["source_bounds_mm"] = dict(self.source_bounds_mm)
        if self.assigned_sensor_id is not None:
            data["assigned_sensor_id"] = int(self.assigned_sensor_id)
        if self.assigned_heater_id is not None:
            data["assigned_heater_id"] = int(self.assigned_heater_id)
        assigned_heater_ids = sorted({int(value) for value in self.assigned_heater_ids if _can_int(value)})
        if assigned_heater_ids:
            data["assigned_heater_ids"] = assigned_heater_ids
            data["assigned_heater_id"] = int(assigned_heater_ids[0])
        if self.sensor_pair_distance_mm is not None:
            data["sensor_pair_distance_mm"] = float(self.sensor_pair_distance_mm)
        if self.power_deposition_node_ids:
            data["power_deposition_node_ids"] = [int(value) for value in self.power_deposition_node_ids]
            data["power_deposition_weights"] = [float(value) for value in self.power_deposition_weights]
        data["heater_attached"] = bool(self.heater_attached)
        data["heater_valid"] = bool(self.heater_valid)
        if self.heater_warning:
            data["heater_warning"] = str(self.heater_warning)
        if self.sensor_connected_node_ids:
            data["sensor_connected_node_ids"] = [int(value) for value in self.sensor_connected_node_ids]
        if self.readout_node_ids:
            data["readout_node_ids"] = [int(value) for value in self.readout_node_ids]
            data["readout_weights"] = [float(value) for value in self.readout_weights]
        data["sensor_monitor_only"] = bool(self.sensor_monitor_only)
        data["sensor_valid"] = bool(self.sensor_valid)
        data["sensor_control_mode"] = _normalize_sensor_control_mode(self.sensor_control_mode)
        data["sensor_manual_power_W"] = float(self.sensor_manual_power_W)
        return data

    @property
    def is_cad_role_node(self) -> bool:
        return bool(str(self.node_type) in {"heater", "sensor"} and self.source_components)

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
    _coord_index_cache: dict[tuple[int, int, int], int] | None = field(default=None, init=False, repr=False)

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
        sensor_ids = {node_id for node_id, node in self.nodes.items() if node.is_sensor}
        heater_ids = {node_id for node_id, node in self.nodes.items() if node.is_heater}
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
        if self._coord_index_cache is not None:
            return dict(self._coord_index_cache)
        index: dict[tuple[int, int, int], int] = {}
        for node_id, node in self.nodes.items():
            coord, _warning = _safe_coord_tuple(getattr(node, "coord", None), fallback=None)
            if coord is None:
                continue
            index.setdefault(coord, int(node_id))
        self._coord_index_cache = dict(index)
        return index

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
                if node.is_heater or node.is_sensor or node.has_cryocooler or node.notes
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
        top_level_tags = data.get("heater_sensor_tags", {}) or {}
        coord_index: dict[tuple[int, int, int], int] = {}
        for raw_node in data.get("graph_nodes", []):
            raw_node = dict(raw_node)
            raw_node_id = raw_node.get("node_id", raw_node.get("id"))
            tag_payload = top_level_tags.get(str(raw_node_id))
            if isinstance(tag_payload, dict):
                merged_tags = dict(raw_node.get("tags", {}) or {})
                merged_tags.update(tag_payload)
                raw_node["tags"] = merged_tags
            node = NodeProperties.from_dict(raw_node)
            if node.coord in coord_index:
                node.coord = _unique_loaded_coord(node.node_id, coord_index)
                model.octree_graph_data.setdefault("warnings", [])
                model.octree_graph_data["warnings"].append(
                    f"Adjusted duplicate loaded coordinate for node {node.node_id}."
                )
            if node.node_id in model.nodes:
                raise ValueError(f"Duplicate node_id {node.node_id}.")
            model.nodes[node.node_id] = node
            coord_index[node.coord] = node.node_id
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
        model.touch()
        return model

    def _check_unique(self, node_id: int, coord: tuple[int, int, int]) -> None:
        if node_id in self.nodes:
            raise ValueError(f"Duplicate node_id {node_id}.")
        if coord in self.coord_index():
            raise ValueError(f"Duplicate coord {coord}.")

    def touch(self) -> None:
        self._coord_index_cache = None
        self.metadata.updated_at = utc_now_iso()


def edge_key(source: int, target: int) -> tuple[int, int]:
    return (min(int(source), int(target)), max(int(source), int(target)))


def _unique_loaded_coord(node_id: int, existing: dict[tuple[int, int, int], int]) -> tuple[int, int, int]:
    base = (int(node_id), 0, 0)
    if base not in existing:
        return base
    offset = 1
    while True:
        candidate = (int(node_id), offset, 0)
        if candidate not in existing:
            return candidate
        offset += 1


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


def _can_int(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _can_float(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _int_list(values: Any) -> list[int]:
    if values in (None, ""):
        return []
    if isinstance(values, (str, bytes)):
        values = [values]
    try:
        iterator = iter(values)
    except TypeError:
        iterator = iter([values])
    result: list[int] = []
    for value in iterator:
        if _can_int(value):
            result.append(int(value))
    return result


def _normalize_sensor_control_mode(value: Any) -> str:
    mode = str(value or "manual").strip().lower()
    return "mimo" if mode in {"mimo", "mimo_pid", "pid"} else "manual"


def _safe_coord_tuple(
    values: Any,
    *,
    fallback: tuple[int, int, int] | None = (0, 0, 0),
) -> tuple[tuple[int, int, int] | None, str]:
    if values is None:
        return fallback, "" if fallback is not None else ""
    try:
        vector = tuple(values)
    except TypeError:
        return fallback, "Ignored invalid coord while loading graph."
    if len(vector) != 3:
        return fallback, "Ignored invalid coord while loading graph."
    coord: list[int] = []
    for value in vector:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return fallback, "Ignored invalid coord while loading graph."
        if not _is_finite(number):
            return fallback, "Ignored non-finite coord while loading graph."
        coord.append(int(round(number)))
    return (coord[0], coord[1], coord[2]), ""


def _finite_vector3(
    values: Any,
    *,
    field_name: str,
    positive: bool = False,
) -> tuple[tuple[float, float, float] | None, str]:
    if values is None:
        return None, ""
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None, f"Ignored invalid {field_name} while loading graph."
    if len(vector) != 3 or not all(_is_finite(value) for value in vector):
        return None, f"Ignored non-finite {field_name} while loading graph."
    if positive and any(value <= 0.0 for value in vector):
        return None, f"Ignored nonpositive {field_name} while loading graph."
    return vector, ""


def _finite_source_bounds(raw: Any) -> tuple[dict[str, list[float]], str]:
    if not isinstance(raw, dict) or "min" not in raw or "max" not in raw:
        return {}, ""
    mins, min_warning = _finite_vector3(raw.get("min"), field_name="source_bounds_mm.min")
    maxs, max_warning = _finite_vector3(raw.get("max"), field_name="source_bounds_mm.max")
    if mins is None or maxs is None:
        return {}, min_warning or max_warning or "Ignored invalid source_bounds_mm while loading graph."
    if any(float(lo) > float(hi) for lo, hi in zip(mins, maxs)):
        return {}, "Ignored inverted source_bounds_mm while loading graph."
    return {"min": [float(value) for value in mins], "max": [float(value) for value in maxs]}, ""


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
