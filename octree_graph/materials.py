"""Material table loading and thermal-property helpers."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re


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
DEFAULT_ASSIGNED_MATERIAL_NAME = "6061-T6 Aluminum"
UNASSIGNED_MATERIAL_NAMES = {
    "",
    "default_unknown",
    "not assigned",
    "none",
    "null",
    "unknown",
    "unknown material",
    "unassigned",
}
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
    if not is_unassigned_material_name(name) and name and name in materials:
        return materials[name]
    fallback = default_assigned_material(materials)
    warnings.append(f"Unknown or unassigned material {name!r}; using {fallback.name}.")
    return fallback


def infer_material_name_from_text(text: str | None, materials: dict[str, Material]) -> str | None:
    if not text:
        return None
    normalized = _normalize_material_text(text)
    exact = _case_insensitive_material_lookup(str(text).strip(), materials)
    if exact and not is_unassigned_material_name(exact):
        return exact
    aliases = [
        (r"\b18\s*[-_]?\s*8\s*SS\b", "18-8 Stainless Steel"),
        (r"\b18\s*[-_]?\s*8\s*STAINLESS\b", "18-8 Stainless Steel"),
        (r"\b304\s*SS\b", "AISI 304 Stainless Steel"),
        (r"\bAISI\s*304\b", "AISI 304 Stainless Steel"),
        (r"\b17\s*[-_]?\s*7\s*PH\b", "17-7PH Stainless Steel"),
        (r"\b17\s*[-_]?\s*7\b", "17-7PH Stainless Steel"),
        (r"\b6061\s*[-_]?\s*T6\b", "6061-T6 Aluminum"),
        (r"\b6061\b", "6061-T6 Aluminum"),
        (r"\bAL(?:UMINUM)?\b", "6061-T6 Aluminum"),
        (r"\bCOPPER\b", "Copper"),
        (r"\bCU\b", "Copper"),
        (r"\bINVAR\s*36\b", "Invar36"),
        (r"\bINVAR\b", "Invar, AL 36"),
        (r"\bDELRIN\b", "Delrin 2700 NC010, Low Viscosity Acetal Copolymer (SS)"),
        (r"\bPHENOLIC\b", "Phenolic"),
    ]
    for pattern, material_name in aliases:
        if re.search(pattern, normalized) and material_name in materials:
            return material_name
    return None


def default_assigned_material(materials: dict[str, Material]) -> Material:
    """Return the material used when cells have missing/unknown material metadata."""
    return materials.get(DEFAULT_ASSIGNED_MATERIAL_NAME, DEFAULT_MATERIAL)


def _case_insensitive_material_lookup(name: str, materials: dict[str, Material]) -> str | None:
    wanted = name.casefold()
    for material_name in materials:
        if material_name.casefold() == wanted:
            return material_name
    return None


def _normalize_material_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9-]+", " ", text.upper())


def is_unassigned_material_name(name: str | None) -> bool:
    if name is None:
        return True
    normalized = str(name).strip().lower()
    return normalized in UNASSIGNED_MATERIAL_NAMES


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
