"""Default material properties for lumped thermal graph cells."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
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

PROJECT_MATERIALS_FILE = Path(__file__).resolve().parents[1] / "materials.json"


def default_material_library() -> dict[str, dict[str, float]]:
    """Return the project material library, falling back to built-in defaults."""
    if PROJECT_MATERIALS_FILE.exists():
        try:
            with PROJECT_MATERIALS_FILE.open("r", encoding="utf-8") as handle:
                return normalize_material_library(json.load(handle))
        except (OSError, json.JSONDecodeError):
            pass
    return deepcopy(DEFAULT_MATERIAL_LIBRARY)


# Cells with no real material assignment ("ZERO MATTER" / "Unassigned" void
# voxels) are modeled as G-10 fiberglass-epoxy INSULATION rather than vacuum: a
# real (if weak) conductor with genuine heat capacity and NIST cryo curves (see
# material_properties_cryo._G10, which also maps these names). This keeps them in
# the thermal network -- a null material (k~1e-9) leaves them disconnected, which
# injects a dense cluster of near-zero eigenvalues that stalls the modal-reduction
# slow-mode solve -- and is physically closer to the truth (these are insulation
# gaps, not vacuum). Overrides any degenerate library entry of the same name.
INSULATION_MATERIAL_DEFAULTS: dict[str, float] = {  # G-10 thermal props, but non-radiating
    "rho_kg_m3": 1800.0,   # G-10 fiberglass-epoxy laminate
    "cp_J_kgK": 1000.0,    # nominal; the NIST G-10 cryo curve overrides at runtime
    "k_W_mK": 0.6,         # nominal normal-direction k; cryo curve overrides at runtime
    # Emissivity 0: these are interior/unknown insulation-filler cells. A nonzero
    # emissivity would make them (cold, ~674k exposed faces) absorb enormous power
    # from the warm ambient -- q = eps*sigma*A*(T_env^4 - T^4) > 0 for T<<T_env --
    # which is both unphysical for insulation (low emissivity by design) and swamps
    # the heater power. Zero keeps them conductive but radiatively inert.
    "emissivity": 0.0,
}
_NULL_MATERIAL_NAMES = frozenset(
    {"", "none", "not assigned", "unassigned", "unassigned (ignored)", "zero matter"}
)


def _is_null_material(material: str) -> bool:
    """True for placeholder 'no real material' names (case/space-insensitive)."""
    return " ".join(str(material or "").strip().lower().split()) in _NULL_MATERIAL_NAMES


def is_unassigned_material(material: str) -> bool:
    """True for cells with no assigned material, EXCLUDING deliberate ``ZERO MATTER``.

    Used by the viewer's "hide unassigned material" filter. Unmatched CAD parts
    default to ``"Unassigned (ignored)"`` (octree_graph.materials.
    DEFAULT_ASSIGNED_MATERIAL_NAME); those, along with ``"Not assigned"`` / empty /
    ``"none"`` / ``"unassigned"``, are considered unassigned. ``ZERO MATTER`` is a
    deliberate inert-void assignment (keep-out volume) and stays visible, so it is
    explicitly not treated as unassigned here even though ``_is_null_material``
    groups it with the null placeholders for thermal-property purposes."""
    if " ".join(str(material or "").strip().lower().split()) == "zero matter":
        return False
    return _is_null_material(material)


def material_defaults(
    material: str, library: dict[str, dict[str, float]] | None = None
) -> dict[str, float]:
    """Return defaults for a material, falling back to the generic package.

    Placeholder 'no real material' cells are modeled as G-10 insulation (see
    INSULATION_MATERIAL_DEFAULTS), regardless of any degenerate library entry."""
    if _is_null_material(material):
        return dict(INSULATION_MATERIAL_DEFAULTS)
    material_library = library or DEFAULT_MATERIAL_LIBRARY
    if material in material_library:
        return dict(material_library[material])
    return dict(material_library["generic electronics package"])


def normalize_material_library(raw: Any) -> dict[str, dict[str, float]]:
    """Coerce a loaded JSON material library to the expected numeric shape."""
    if isinstance(raw, list):
        raw = {
            str(row.get("name")): row
            for row in raw
            if isinstance(row, dict) and row.get("name")
        }
    if not isinstance(raw, dict):
        return deepcopy(DEFAULT_MATERIAL_LIBRARY)
    normalized = deepcopy(DEFAULT_MATERIAL_LIBRARY)
    for name, values in raw.items():
        if not isinstance(values, dict):
            continue
        current = material_defaults("generic electronics package", normalized)
        key_aliases = {
            "rho_kg_m3": ("rho_kg_m3", "density_kg_m3"),
            "cp_J_kgK": ("cp_J_kgK",),
            "k_W_mK": ("k_W_mK",),
            "emissivity": ("emissivity",),
        }
        for key, aliases in key_aliases.items():
            raw_value = next((values[alias] for alias in aliases if alias in values), current[key])
            try:
                current[key] = float(raw_value)
            except (TypeError, ValueError):
                pass
        normalized[str(name)] = current
    return normalized
