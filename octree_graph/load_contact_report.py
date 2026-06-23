"""Excel ContactReport.xlsx loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any


@dataclass
class ContactReport:
    component_materials: dict[str, str] = field(default_factory=dict)
    component_masses_kg: dict[str, float] = field(default_factory=dict)
    contact_pairs: set[tuple[str, str]] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)

    def has_pair(self, a: str, b: str) -> bool:
        for alias_a in _component_aliases(a):
            for alias_b in _component_aliases(b):
                if tuple(sorted((alias_a, alias_b))) in self.contact_pairs:
                    return True
        return False

    def material_for_component(self, name: str) -> str | None:
        for alias in _component_aliases(name):
            material = self.component_materials.get(alias)
            if material:
                return material
        return None


def load_contact_report(path: str | Path | None) -> ContactReport:
    report = ContactReport()
    if not path:
        return report
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("openpyxl is required to read ContactReport.xlsx.") from exc

    workbook = load_workbook(Path(path), data_only=True, read_only=True)
    for sheet_name in ("Contact Summary", "Contact Pairs"):
        if sheet_name not in workbook.sheetnames:
            report.warnings.append(f"Missing sheet {sheet_name!r}.")
    if "Contact Summary" in workbook.sheetnames:
        _read_summary(workbook["Contact Summary"], report)
    if "Contact Pairs" in workbook.sheetnames:
        _read_pairs(workbook["Contact Pairs"], report)
    _add_component_aliases(report)
    return report


def _header_map(sheet: Any, required: tuple[str, ...]) -> tuple[dict[str, int], int]:
    for row_number, row in enumerate(sheet.iter_rows(min_row=1, max_row=50, values_only=True), 1):
        headers = [str(value or "").strip().lower() for value in row]
        if all(name in headers for name in required):
            return {name: index for index, name in enumerate(headers)}, row_number
    raise ValueError(f"Could not find header row in sheet {sheet.title!r}.")


def _pick(row: tuple[Any, ...], headers: dict[str, int], names: tuple[str, ...]) -> Any:
    for name in names:
        index = headers.get(name)
        if index is not None and index < len(row):
            return row[index]
    return None


def _read_summary(sheet: Any, report: ContactReport) -> None:
    headers, header_row = _header_map(sheet, ("instance name", "material"))
    for row in sheet.iter_rows(min_row=header_row + 1, values_only=True):
        component = _pick(row, headers, ("component", "component name", "name", "part"))
        component = component or _pick(row, headers, ("instance name",))
        if not component:
            continue
        name = str(component).strip()
        material = _pick(row, headers, ("material", "material name"))
        if material:
            report.component_materials[name] = str(material).strip()
        mass = _pick(row, headers, ("mass_kg", "mass kg", "mass"))
        if mass not in (None, ""):
            try:
                report.component_masses_kg[name] = float(mass)
            except (TypeError, ValueError):
                report.warnings.append(f"Invalid mass for {name}: {mass!r}")


def _read_pairs(sheet: Any, report: ContactReport) -> None:
    headers, header_row = _header_map(sheet, ("part a name", "part b name"))
    for row in sheet.iter_rows(min_row=header_row + 1, values_only=True):
        a = _pick(row, headers, ("component a", "part a", "source", "component_1", "component1"))
        b = _pick(row, headers, ("component b", "part b", "target", "component_2", "component2"))
        a = a or _pick(row, headers, ("part a name",))
        b = b or _pick(row, headers, ("part b name",))
        if a and b:
            a_name = str(a).strip()
            b_name = str(b).strip()
            for alias_a in _component_aliases(a_name):
                for alias_b in _component_aliases(b_name):
                    report.contact_pairs.add(tuple(sorted((alias_a, alias_b))))


def _add_component_aliases(report: ContactReport) -> None:
    candidates: dict[str, set[str]] = {}
    for name, material in report.component_materials.items():
        for alias in _component_aliases(name):
            candidates.setdefault(alias, set()).add(material)
    for alias, materials in candidates.items():
        if alias in report.component_materials or len(materials) != 1:
            continue
        report.component_materials[alias] = next(iter(materials))


def _component_aliases(name: str) -> set[str]:
    aliases = {name}
    current = name
    for _ in range(3):
        stripped = re.sub(r"([_-])\d+$", "", current)
        if stripped == current:
            break
        aliases.add(stripped)
        current = stripped
    return aliases
