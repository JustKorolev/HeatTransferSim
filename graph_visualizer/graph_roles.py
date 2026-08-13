"""The graph's heaters and sensors, read from the build's own nodes.csv.

The headless tab never loads a graph, so it used to take its sensor list from the
most recent run's ``sensors.csv`` -- which means the list is empty until a run has
happened, and a first run therefore had no way to name its sensors. The build
already knows: nodes.csv carries ``is_heater`` / ``is_sensor`` columns for every
node, and the controller artifacts (G, modal npz) are derived from exactly those
nodes, so this list is the superset every controller scheme is built from rather
than one scheme's view of it.

Scanning is a single streaming pass over two columns -- ~1.7 s for a 471k-node,
188 MB nodes.csv -- and the result is cached beside the graph as a small JSON keyed
on the CSV's size and mtime, so it is paid once per build rather than once per tab
switch.

CAVEAT: nodes.csv is written by the BUILD. Roles added or removed later in the app
live in graph.json, so a heater assigned in the GUI after the build will not appear
here. Callers that hold a controller artifact should mark ids the artifact does not
cover rather than trusting either list alone.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys

NODES_CSV = "nodes.csv"
ROLE_CACHE_FILENAME = "role_manifest.json"
CACHE_VERSION = 1


@dataclass
class RoleRow:
    node_id: int
    component_name: str = ""
    monitor_only: bool = False

    def as_dict(self) -> dict:
        return {
            "node_id": int(self.node_id),
            "component_name": str(self.component_name),
            "monitor_only": bool(self.monitor_only),
        }


@dataclass
class RoleManifest:
    heaters: list[RoleRow] = field(default_factory=list)
    sensors: list[RoleRow] = field(default_factory=list)
    # Where the rows came from, for the status line: "nodes.csv", "cache", or the
    # reason there are none.
    source: str = ""

    @property
    def heater_ids(self) -> list[int]:
        return [int(row.node_id) for row in self.heaters]

    @property
    def sensor_ids(self) -> list[int]:
        return [int(row.node_id) for row in self.sensors]


def _cache_key(nodes_csv: Path) -> dict:
    stat = nodes_csv.stat()
    return {"version": CACHE_VERSION, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _rows_from_payload(payload: list) -> list[RoleRow]:
    rows: list[RoleRow] = []
    for entry in payload or ():
        if not isinstance(entry, dict):
            continue
        try:
            node_id = int(entry["node_id"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(
            RoleRow(
                node_id=node_id,
                component_name=str(entry.get("component_name", "")),
                monitor_only=bool(entry.get("monitor_only", False)),
            )
        )
    return rows


def _read_cache(folder: Path, nodes_csv: Path) -> RoleManifest | None:
    path = folder / ROLE_CACHE_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("source_key") != _cache_key(nodes_csv):
        return None  # rebuilt (or truncated) since: rescan rather than serve stale roles
    return RoleManifest(
        heaters=_rows_from_payload(payload.get("heaters", [])),
        sensors=_rows_from_payload(payload.get("sensors", [])),
        source="cache",
    )


def _write_cache(folder: Path, nodes_csv: Path, manifest: RoleManifest) -> None:
    payload = {
        "source_key": _cache_key(nodes_csv),
        "heaters": [row.as_dict() for row in manifest.heaters],
        "sensors": [row.as_dict() for row in manifest.sensors],
    }
    try:
        (folder / ROLE_CACHE_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass  # a read-only graph folder still gets a working (uncached) list


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def scan_nodes_csv(nodes_csv: Path) -> RoleManifest:
    """Stream nodes.csv and collect its heater and sensor rows.

    csv.reader with column indices rather than DictReader: nodes.csv has ~45
    columns and roles are a handful of rows in a million, so building a dict per
    row costs more than the whole rest of the scan (3.9 s -> 2.3 s on a 423 MB,
    1.0M-node graph). Quoting is still parsed properly, which matters because a
    component name can contain a comma or a newline.
    """
    heaters: list[RoleRow] = []
    sensors: list[RoleRow] = []
    # A role's component name can be long, and one build wrote a giant role_json
    # field; the default field-size limit aborts the whole scan on such a row.
    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):  # 32-bit builds reject maxsize
        csv.field_size_limit(2**31 - 1)
    with nodes_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return RoleManifest(source=f"{NODES_CSV} is empty")
        index = {name: position for position, name in enumerate(header)}
        heater_at, sensor_at = index.get("is_heater"), index.get("is_sensor")
        if heater_at is None and sensor_at is None:
            # A build too old to record roles. The caller falls back to a run's
            # own manifest rather than reporting a graph with no heaters.
            return RoleManifest(source=f"{NODES_CSV} has no is_heater/is_sensor columns")
        id_at = index.get("node_id")
        name_at = index.get("component_name")
        monitor_at = index.get("sensor_monitor_only")

        def field(row: list[str], position: int | None) -> str | None:
            if position is None or position >= len(row):
                return None
            return row[position]

        for row in reader:
            is_heater = _truthy(field(row, heater_at))
            is_sensor = _truthy(field(row, sensor_at))
            if not (is_heater or is_sensor):
                continue
            try:
                node_id = int(field(row, id_at) or "")
            except (TypeError, ValueError):
                continue
            name = str(field(row, name_at) or "")
            if is_heater:
                heaters.append(RoleRow(node_id=node_id, component_name=name))
            if is_sensor:
                sensors.append(
                    RoleRow(
                        node_id=node_id,
                        component_name=name,
                        monitor_only=_truthy(field(row, monitor_at)),
                    )
                )
    heaters.sort(key=lambda row: row.node_id)
    sensors.sort(key=lambda row: row.node_id)
    return RoleManifest(heaters=heaters, sensors=sensors, source=NODES_CSV)


def load_role_manifest(folder: str | Path, *, refresh: bool = False) -> RoleManifest:
    """Heaters and sensors for the graph in ``folder``.

    Served from the cached role_manifest.json when it matches the current
    nodes.csv; otherwise scanned and cached. ``refresh`` forces the rescan.
    """
    folder = Path(folder)
    nodes_csv = folder / NODES_CSV
    if not nodes_csv.is_file():
        return RoleManifest(source=f"no {NODES_CSV} in {folder.name}")
    if not refresh:
        cached = _read_cache(folder, nodes_csv)
        if cached is not None:
            return cached
    try:
        manifest = scan_nodes_csv(nodes_csv)
    except (OSError, csv.Error) as exc:
        return RoleManifest(source=f"could not read {NODES_CSV}: {exc}")
    _write_cache(folder, nodes_csv, manifest)
    return manifest
