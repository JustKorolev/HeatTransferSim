"""The graph's heaters and sensors, read from the build's own nodes.csv.

The headless tab never loads a graph, so it used to take its sensor list from the
most recent run's ``sensors.csv`` -- which means the list is empty until a run has
happened, and a first run therefore had no way to name its sensors. The build
already knows: nodes.csv carries ``is_heater`` / ``is_sensor`` columns for every
node, and the controller artifacts (G, modal npz) are derived from exactly those
nodes, so this list is the superset every controller scheme is built from rather
than one scheme's view of it.

Scanning is a single streaming pass -- ~1.7 s for a 471k-node, 188 MB nodes.csv --
and the result is cached beside the graph as a small JSON keyed on the CSV's size
and mtime, so it is paid once per build rather than once per tab switch.

TWO WRITERS, TWO LAYOUTS. This file is not written one way:

* the octree builder flattens every key, so roles appear as ``is_heater`` /
  ``is_sensor`` columns;
* ``graph_io._write_nodes_csv`` -- what "Update graph" and the GUI's lightweight
  save run -- writes a fixed 18-column set with ``extrasaction="ignore"`` and packs
  each role node's COMPLETE dict into one ``role_json`` cell. Those flat columns do
  not exist in that file at all.

Both are read here, ``role_json`` first and authoritative, matching
``fast_graph_io._node_from_row``. Reading only the flat columns made every
refreshed graph report zero heaters and zero sensors.

CAVEAT: a graph edited in the GUI and NOT refreshed still has its build-time
nodes.csv, so roles assigned since then live only in graph.json and will not appear
here -- "Update graph" is what reconciles them. Callers holding a controller
artifact should mark ids the artifact does not cover rather than trusting either
list alone.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys

NODES_CSV = "nodes.csv"
ROLE_CACHE_FILENAME = "role_manifest.json"
# Must match graph_io._NODES_CSV_ROLE_COLUMN / fast_graph_io.ROLE_JSON_COLUMN.
ROLE_JSON_COLUMN = "role_json"
# 2: role_json is read (v1 read only the builder's flat is_heater/is_sensor
# columns, so every "Update graph"-refreshed nodes.csv cached as having no roles).
CACHE_VERSION = 2


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
    # The scan's own source text is stored and replayed. A cached EMPTY result still
    # has to say why it is empty ("nodes.csv has no is_heater/is_sensor columns");
    # reporting a bare "cache" turned the one line that explains an empty table into
    # the one line that explains nothing.
    origin = str(payload.get("source") or NODES_CSV)
    return RoleManifest(
        heaters=_rows_from_payload(payload.get("heaters", [])),
        sensors=_rows_from_payload(payload.get("sensors", [])),
        source=f"{origin} (cached)",
    )


def _write_cache(folder: Path, nodes_csv: Path, manifest: RoleManifest) -> None:
    payload = {
        "source_key": _cache_key(nodes_csv),
        "source": manifest.source,
        "heaters": [row.as_dict() for row in manifest.heaters],
        "sensors": [row.as_dict() for row in manifest.sensors],
    }
    try:
        (folder / ROLE_CACHE_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass  # a read-only graph folder still gets a working (uncached) list


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def _role_payload(cell: str | None) -> dict | None:
    """The node dict inside a ``role_json`` cell, or None if there is not one.

    Only role nodes carry this cell, so on a million-row file this parses a few
    dozen times. A malformed cell falls through to the flat columns rather than
    dropping the row: half a role list is worse than a slightly stale one.
    """
    if not cell or not str(cell).strip():
        return None
    try:
        data = json.loads(cell)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


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
        role_at = index.get(ROLE_JSON_COLUMN)
        if heater_at is None and sensor_at is None and role_at is None:
            # A build too old to record roles either way. The caller falls back to a
            # run's own manifest rather than reporting a graph with no heaters.
            return RoleManifest(
                source=f"{NODES_CSV} has no {ROLE_JSON_COLUMN} or is_heater/is_sensor columns"
            )
        id_at = index.get("node_id")
        name_at = index.get("component_name")
        monitor_at = index.get("sensor_monitor_only")

        def field(row: list[str], position: int | None) -> str | None:
            if position is None or position >= len(row):
                return None
            return row[position]

        for row in reader:
            # role_json first, and authoritative when present -- the same precedence
            # fast_graph_io._node_from_row uses. The two writers of this file do NOT
            # agree: the octree builder flattens every key (so is_heater/is_sensor
            # are columns), while graph_io._write_nodes_csv -- the one behind "Update
            # graph" -- writes a fixed 18-column set with extrasaction="ignore" and
            # puts the WHOLE role dict in role_json, so those columns do not exist at
            # all. Reading only the flat ones made every refreshed graph look like it
            # had no heaters or sensors.
            role = _role_payload(field(row, role_at))
            if role is not None:
                is_heater = bool(role.get("is_heater"))
                is_sensor = bool(role.get("is_sensor"))
            else:
                is_heater = _truthy(field(row, heater_at))
                is_sensor = _truthy(field(row, sensor_at))
            if not (is_heater or is_sensor):
                continue
            try:
                node_id = int(field(row, id_at) or "")
            except (TypeError, ValueError):
                continue
            name = str(field(row, name_at) or "")
            monitor = (
                bool(role.get("sensor_monitor_only"))
                if role is not None
                else _truthy(field(row, monitor_at))
            )
            if is_heater:
                heaters.append(RoleRow(node_id=node_id, component_name=name))
            if is_sensor:
                sensors.append(
                    RoleRow(node_id=node_id, component_name=name, monitor_only=monitor)
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
    except Exception as exc:  # noqa: BLE001
        # Deliberately broad: this feeds two GUI tables, and a nodes.csv with one
        # bad byte (UnicodeDecodeError, not OSError) must come back as "here is why
        # the list is empty" rather than as an exception that blanks the tab.
        return RoleManifest(source=f"could not read {NODES_CSV}: {type(exc).__name__}: {exc}")
    _write_cache(folder, nodes_csv, manifest)
    return manifest
