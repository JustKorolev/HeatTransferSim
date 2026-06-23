"""Material table loading and thermal-property helpers."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass(frozen=True)
class Material:
    name: str
    density_kg_m3: float
    cp_J_kgK: float
    k_W_mK: float
    emissivity: float

    @property
    def rho_cp(self) -> float:
        return self.density_kg_m3 * self.cp_J_kgK

    def to_dict(self) -> dict[str, float | str]:
        data = asdict(self)
        data["rho_cp_J_m3K"] = self.rho_cp
        return data


DEFAULT_MATERIAL = Material(
    name="default_unknown",
    density_kg_m3=2200.0,
    cp_J_kgK=800.0,
    k_W_mK=2.0,
    emissivity=0.85,
)
PROJECT_MATERIALS_FILE = Path(__file__).resolve().parents[1] / "materials.json"


def load_material_table(path: str | Path | None = None) -> tuple[dict[str, Material], list[str]]:
    warnings: list[str] = []
    table_path = Path(path) if path is not None else PROJECT_MATERIALS_FILE
    with table_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        rows = []
        for name, values in raw.items():
            if isinstance(values, dict):
                row = dict(values)
                row.setdefault("name", name)
                rows.append(row)
    else:
        rows = raw
    materials: dict[str, Material] = {DEFAULT_MATERIAL.name: DEFAULT_MATERIAL}
    for row in rows:
        try:
            material = Material(
                name=str(row["name"]),
                density_kg_m3=float(row.get("density_kg_m3", row.get("rho_kg_m3"))),
                cp_J_kgK=float(row["cp_J_kgK"]),
                k_W_mK=float(row["k_W_mK"]),
                emissivity=float(row.get("emissivity", 0.85)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"Skipped invalid material row {row!r}: {exc}")
            continue
        materials[material.name] = material
    return materials, warnings


def resolve_material(
    name: str | None, materials: dict[str, Material], warnings: list[str]
) -> Material:
    if name and name in materials:
        return materials[name]
    warnings.append(f"Unknown material {name!r}; using {DEFAULT_MATERIAL.name}.")
    return materials.get(DEFAULT_MATERIAL.name, DEFAULT_MATERIAL)


def contrast_exceeds(materials: list[Material], threshold: float) -> bool:
    if len(materials) < 2:
        return False
    ks = [m.k_W_mK for m in materials if m.k_W_mK > 0.0]
    cps = [m.rho_cp for m in materials if m.rho_cp > 0.0]
    if len(ks) >= 2 and max(ks) / min(ks) >= threshold:
        return True
    if len(cps) >= 2 and max(cps) / min(cps) >= threshold:
        return True
    return False
