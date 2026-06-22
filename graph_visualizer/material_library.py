"""Default material properties for lumped thermal graph cells."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_MATERIAL_LIBRARY: dict[str, dict[str, float]] = {
    "copper": {
        "rho_kg_m3": 8960.0,
        "cp_J_kgK": 385.0,
        "k_W_mK": 401.0,
        "emissivity": 0.35,
    },
    "aluminum": {
        "rho_kg_m3": 2700.0,
        "cp_J_kgK": 897.0,
        "k_W_mK": 237.0,
        "emissivity": 0.09,
    },
    "stainless steel": {
        "rho_kg_m3": 8000.0,
        "cp_J_kgK": 500.0,
        "k_W_mK": 16.0,
        "emissivity": 0.45,
    },
    "titanium": {
        "rho_kg_m3": 4500.0,
        "cp_J_kgK": 522.0,
        "k_W_mK": 22.0,
        "emissivity": 0.30,
    },
    "brass": {
        "rho_kg_m3": 8500.0,
        "cp_J_kgK": 380.0,
        "k_W_mK": 110.0,
        "emissivity": 0.30,
    },
    "silicon": {
        "rho_kg_m3": 2330.0,
        "cp_J_kgK": 705.0,
        "k_W_mK": 148.0,
        "emissivity": 0.70,
    },
    "glass": {
        "rho_kg_m3": 2500.0,
        "cp_J_kgK": 840.0,
        "k_W_mK": 1.05,
        "emissivity": 0.90,
    },
    "ceramic/alumina": {
        "rho_kg_m3": 3900.0,
        "cp_J_kgK": 880.0,
        "k_W_mK": 25.0,
        "emissivity": 0.80,
    },
    "FR4 / PCB": {
        "rho_kg_m3": 1850.0,
        "cp_J_kgK": 1100.0,
        "k_W_mK": 0.30,
        "emissivity": 0.85,
    },
    "Kapton": {
        "rho_kg_m3": 1420.0,
        "cp_J_kgK": 1090.0,
        "k_W_mK": 0.12,
        "emissivity": 0.80,
    },
    "PEEK": {
        "rho_kg_m3": 1320.0,
        "cp_J_kgK": 1340.0,
        "k_W_mK": 0.25,
        "emissivity": 0.85,
    },
    "PTFE / Teflon": {
        "rho_kg_m3": 2200.0,
        "cp_J_kgK": 1000.0,
        "k_W_mK": 0.25,
        "emissivity": 0.95,
    },
    "epoxy": {
        "rho_kg_m3": 1200.0,
        "cp_J_kgK": 1000.0,
        "k_W_mK": 0.20,
        "emissivity": 0.85,
    },
    "vacuum/insulator placeholder": {
        "rho_kg_m3": 1.0,
        "cp_J_kgK": 1.0,
        "k_W_mK": 1.0e-9,
        "emissivity": 0.0,
    },
    "generic electronics package": {
        "rho_kg_m3": 2200.0,
        "cp_J_kgK": 800.0,
        "k_W_mK": 2.0,
        "emissivity": 0.85,
    },
}


def default_material_library() -> dict[str, dict[str, float]]:
    """Return an isolated copy of the default material library."""
    return deepcopy(DEFAULT_MATERIAL_LIBRARY)


def material_defaults(
    material: str, library: dict[str, dict[str, float]] | None = None
) -> dict[str, float]:
    """Return defaults for a material, falling back to the generic package."""
    material_library = library or DEFAULT_MATERIAL_LIBRARY
    if material in material_library:
        return dict(material_library[material])
    return dict(material_library["generic electronics package"])


def normalize_material_library(raw: Any) -> dict[str, dict[str, float]]:
    """Coerce a loaded JSON material library to the expected numeric shape."""
    if not isinstance(raw, dict):
        return default_material_library()
    normalized = default_material_library()
    for name, values in raw.items():
        if not isinstance(values, dict):
            continue
        current = material_defaults("generic electronics package", normalized)
        for key in ("rho_kg_m3", "cp_J_kgK", "k_W_mK", "emissivity"):
            try:
                current[key] = float(values.get(key, current[key]))
            except (TypeError, ValueError):
                pass
        normalized[str(name)] = current
    return normalized
