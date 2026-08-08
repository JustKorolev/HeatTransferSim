"""Lossless binary persistence of ``model.edges`` for the low-memory load path.

The compact ``nodes.csv`` path historically dropped edges entirely, on the
assumption that conduction always comes from the prebuilt ``L`` matrix. That is
true only for constant properties: with
``use_temperature_dependent_properties`` the engine rebuilds ``L(T)`` every step
from ``model.edges``, so an edgeless model silently produced an ALL-ZERO
Laplacian -- every node thermally isolated, heaters cooking their own cells while
the rest of the graph never moved.

This module stores every ``EdgeProperties`` field in a compact ``edges.npz`` so
the fast path reconstructs edges byte-for-byte identically to the full
``graph.json`` loader, at a few hundred MB instead of gigabytes:

* numeric fields as plain arrays,
* the three low-cardinality strings (``edge_type``, ``source_metadata``,
  ``contact_confidence``) as categorical codes + a small lookup table,
* ``edge_id`` as its integer suffix when it matches the canonical ``edge_<n>``
  form (the overwhelming majority), with any exceptions stored sparsely,
* ``warnings`` sparsely, since almost every edge has none.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .models import EdgeProperties, ThermalGraphModel

EDGES_FILENAME = "edges.npz"
_CANONICAL_PREFIX = "edge_"


def has_edges_artifact(folder: str | Path) -> bool:
    return (Path(folder) / EDGES_FILENAME).is_file()


def _encode_categorical(values: list[str]) -> tuple[np.ndarray, np.ndarray]:
    table, codes = np.unique(np.asarray(values, dtype=object).astype(str), return_inverse=True)
    return codes.astype(np.int32), table.astype(str)


def _canonical_edge_id(value: Any) -> int:
    """Integer suffix of an ``edge_<n>`` id, or -1 when it is not that shape."""
    if not isinstance(value, str) or not value.startswith(_CANONICAL_PREFIX):
        return -1
    suffix = value[len(_CANONICAL_PREFIX):]
    return int(suffix) if suffix.isdigit() else -1


def write_edges_npz(model: ThermalGraphModel, folder: str | Path) -> int:
    """Write every edge in ``model`` to ``edges.npz``. Returns the edge count."""
    folder = Path(folder)
    edges = list(model.edges.values())
    n = len(edges)
    source = np.fromiter((int(e.source) for e in edges), dtype=np.int64, count=n)
    target = np.fromiter((int(e.target) for e in edges), dtype=np.int64, count=n)
    conductance = np.fromiter((float(e.Gij_W_K) for e in edges), dtype=np.float64, count=n)
    area = np.fromiter((float(e.shared_area_m2) for e in edges), dtype=np.float64, count=n)
    distance = np.fromiter((float(e.distance_m) for e in edges), dtype=np.float64, count=n)
    type_codes, type_table = _encode_categorical([str(e.edge_type) for e in edges])
    meta_codes, meta_table = _encode_categorical([str(e.source_metadata) for e in edges])
    conf_codes, conf_table = _encode_categorical([str(e.contact_confidence) for e in edges])

    edge_id_num = np.fromiter((_canonical_edge_id(e.edge_id) for e in edges), dtype=np.int64, count=n)
    # edge_id values that are neither None nor canonical must survive verbatim.
    odd_idx = [i for i, e in enumerate(edges) if edge_id_num[i] < 0 and e.edge_id is not None]
    odd_val = [str(edges[i].edge_id) for i in odd_idx]
    warn_idx = [i for i, e in enumerate(edges) if e.warnings]
    warn_val = [json.dumps(list(edges[i].warnings)) for i in warn_idx]

    np.savez(
        folder / EDGES_FILENAME,
        source=source,
        target=target,
        conductance=conductance,
        shared_area_m2=area,
        distance_m=distance,
        edge_type_codes=type_codes,
        edge_type_table=type_table,
        source_metadata_codes=meta_codes,
        source_metadata_table=meta_table,
        contact_confidence_codes=conf_codes,
        contact_confidence_table=conf_table,
        edge_id_num=edge_id_num,
        edge_id_extra_idx=np.asarray(odd_idx, dtype=np.int64),
        edge_id_extra_val=np.asarray(odd_val, dtype=str),
        warning_idx=np.asarray(warn_idx, dtype=np.int64),
        warning_json=np.asarray(warn_val, dtype=str),
    )
    return n


def iter_octree_edges(graph_json_path: str | Path, chunk_bytes: int = 1 << 24):
    """Yield each ``graph_edges`` entry of a graph.json without loading the file.

    A 3M-cell graph.json is ~10 GB and the full loader needs ~45 GB of RAM to parse
    it, so an existing graph cannot simply be re-saved to gain ``edges.npz``. This
    walks the ``graph_edges`` array with a byte-level brace scanner (string- and
    escape-aware) and ``json.loads`` one object at a time, so peak memory is one
    edge plus the read buffer.
    """
    path = Path(graph_json_path)
    with path.open("rb") as handle:
        buffer = b""
        # Seek to the start of the graph_edges array.
        marker = b'"graph_edges"'
        start = -1
        while start < 0:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                return
            buffer += chunk
            start = buffer.find(marker)
            if start < 0:
                buffer = buffer[-len(marker):]
        bracket = buffer.find(b"[", start)
        while bracket < 0:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                return
            buffer += chunk
            bracket = buffer.find(b"[", start)
        buffer = buffer[bracket + 1:]

        depth = 0
        in_string = False
        escaped = False
        obj_start = -1
        index = 0  # cursor into buffer; NEVER re-slice per object (that is O(n^2))
        while True:
            size = len(buffer)
            while index < size:
                byte = buffer[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif byte == 0x5C:  # backslash
                        escaped = True
                    elif byte == 0x22:  # quote
                        in_string = False
                elif byte == 0x22:
                    in_string = True
                elif byte == 0x7B:  # {
                    if depth == 0:
                        obj_start = index
                    depth += 1
                elif byte == 0x7D:  # }
                    depth -= 1
                    if depth == 0 and obj_start >= 0:
                        yield json.loads(buffer[obj_start:index + 1].decode("utf-8"))
                        obj_start = -1
                elif byte == 0x5D and depth == 0:  # ] closing graph_edges
                    return
                index += 1
            # Buffer exhausted: drop everything already consumed, keeping only a
            # partially-read object, then refill. One copy per chunk, not per edge.
            keep_from = obj_start if obj_start >= 0 else index
            buffer = buffer[keep_from:]
            if obj_start >= 0:
                obj_start = 0
            index = len(buffer)
            chunk = handle.read(chunk_bytes)
            if not chunk:
                return
            buffer += chunk


def write_edges_npz_from_graph_json(graph_json_path: str | Path, folder: str | Path) -> int:
    """Build ``edges.npz`` by streaming graph.json (no full parse). Returns count."""
    from array import array

    from .models import edge_key

    source = array("q")
    target = array("q")
    conductance = array("d")
    area = array("d")
    distance = array("d")
    edge_id_num = array("q")
    type_codes = array("i")
    meta_codes = array("i")
    conf_codes = array("i")
    tables: dict[str, dict[str, int]] = {"type": {}, "meta": {}, "conf": {}}
    odd_idx: list[int] = []
    odd_val: list[str] = []
    warn_idx: list[int] = []
    warn_val: list[str] = []

    def code(kind: str, value: str) -> int:
        table = tables[kind]
        if value not in table:
            table[value] = len(table)
        return table[value]

    for row, data in enumerate(iter_octree_edges(graph_json_path)):
        low, high = edge_key(int(data["node_i"]), int(data["node_j"]))
        source.append(low)
        target.append(high)
        conductance.append(float(data.get("G_W_K", data.get("Gij_W_K", 0.0))))
        area.append(float(data.get("shared_area_m2", 0.0)))
        distance.append(float(data.get("distance_m", 0.0)))
        type_codes.append(code("type", str(data.get("edge_type", "internal_conduction"))))
        meta_codes.append(code("meta", str(data.get("source", ""))))
        conf_codes.append(code("conf", str(data.get("contact_confidence", "medium"))))
        raw_id = data.get("edge_id")
        number = _canonical_edge_id(raw_id)
        edge_id_num.append(number)
        if number < 0 and raw_id is not None:
            odd_idx.append(row)
            odd_val.append(str(raw_id))
        warnings = data.get("warnings") or []
        if warnings:
            warn_idx.append(row)
            warn_val.append(json.dumps(list(warnings)))

    def table_array(kind: str) -> np.ndarray:
        items = sorted(tables[kind], key=tables[kind].get)
        return np.asarray(items, dtype=str)

    np.savez(
        Path(folder) / EDGES_FILENAME,
        source=np.frombuffer(source, dtype=np.int64),
        target=np.frombuffer(target, dtype=np.int64),
        conductance=np.frombuffer(conductance, dtype=np.float64),
        shared_area_m2=np.frombuffer(area, dtype=np.float64),
        distance_m=np.frombuffer(distance, dtype=np.float64),
        edge_type_codes=np.frombuffer(type_codes, dtype=np.int32),
        edge_type_table=table_array("type"),
        source_metadata_codes=np.frombuffer(meta_codes, dtype=np.int32),
        source_metadata_table=table_array("meta"),
        contact_confidence_codes=np.frombuffer(conf_codes, dtype=np.int32),
        contact_confidence_table=table_array("conf"),
        edge_id_num=np.frombuffer(edge_id_num, dtype=np.int64),
        edge_id_extra_idx=np.asarray(odd_idx, dtype=np.int64),
        edge_id_extra_val=np.asarray(odd_val, dtype=str),
        warning_idx=np.asarray(warn_idx, dtype=np.int64),
        warning_json=np.asarray(warn_val, dtype=str),
    )
    return len(source)


def read_edges_npz(folder: str | Path) -> dict[tuple[int, int], EdgeProperties]:
    """Rebuild ``model.edges`` exactly as the full graph.json loader would."""
    folder = Path(folder)
    with np.load(folder / EDGES_FILENAME, allow_pickle=False) as data:
        source = data["source"]
        target = data["target"]
        conductance = data["conductance"]
        area = data["shared_area_m2"]
        distance = data["distance_m"]
        type_table = data["edge_type_table"]
        type_codes = data["edge_type_codes"]
        meta_table = data["source_metadata_table"]
        meta_codes = data["source_metadata_codes"]
        conf_table = data["contact_confidence_table"]
        conf_codes = data["contact_confidence_codes"]
        edge_id_num = data["edge_id_num"]
        extra = dict(zip(data["edge_id_extra_idx"].tolist(), data["edge_id_extra_val"].tolist()))
        warns = {
            int(i): json.loads(str(text))
            for i, text in zip(data["warning_idx"].tolist(), data["warning_json"].tolist())
        }

    edges: dict[tuple[int, int], EdgeProperties] = {}
    for row in range(source.size):
        number = int(edge_id_num[row])
        if number >= 0:
            edge_id: str | None = f"{_CANONICAL_PREFIX}{number}"
        else:
            edge_id = extra.get(row)
        low = int(source[row])
        high = int(target[row])
        edges[(low, high)] = EdgeProperties(
            source=low,
            target=high,
            Gij_W_K=float(conductance[row]),
            source_metadata=str(meta_table[int(meta_codes[row])]),
            edge_id=edge_id,
            edge_type=str(type_table[int(type_codes[row])]),
            shared_area_m2=float(area[row]),
            distance_m=float(distance[row]),
            contact_confidence=str(conf_table[int(conf_codes[row])]),
            warnings=list(warns.get(row, [])),
        )
    return edges
