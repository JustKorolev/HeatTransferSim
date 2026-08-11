"""Low-memory graph loading for headless simulation.

``graph_io.load_graph_folder`` parses ``graph.json`` -- 1.6 GB at 471k nodes,
~10 GB at 3M -- into one Python dict per node, per edge and per octree cell, then
builds the model from it and *keeps* the parsed payload on
``model.octree_graph_data``. That is where a 3M-cell headless run's ~45 GB goes.

A simulation does not need the raw parsed payload. This module rebuilds the model
from the compact per-node ``nodes.csv`` plus the binary matrix files, streaming
row by row so the full set of raw dicts never exists at once.

It DOES need the edges. An earlier version of this module skipped them, reasoning
that "conduction comes from the L matrix" -- true only for constant properties.
With ``use_temperature_dependent_properties`` the engine rebuilds ``L(T)`` every
step from ``model.edges``, so an edgeless model yielded an all-zero Laplacian and
silently simulated 3M thermally isolated nodes. Edges are now restored losslessly
from ``edges.npz`` (see ``fast_edge_io``), and ``can_load_fast`` requires that
file, so a graph lacking it falls back to the full loader rather than simulating a
subtly different model.

Caveat: ``nodes.csv`` carries no per-node controller setpoint (the octree writer
does not emit one), so setpoints must come from the run config. The loader
reports this via ``LoadReport.warnings`` and the runner surfaces it.
"""

from __future__ import annotations

import ast
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, save_npz

from .diagnostics import log_event
from .fast_edge_io import EDGES_FILENAME
from .models import GraphMetadata, NodeProperties, ThermalGraphModel


@dataclass
class LoadReport:
    node_count: int = 0
    edge_count: int = 0
    warnings: list[str] = field(default_factory=list)


_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no"}
# Columns that must stay strings even when they look numeric.
_TEXT_COLUMNS = {
    "cell_id",
    "component_name",
    "material_name",
    "confidence",
    "node_type",
    "heater_warning",
    "sensor_control_mode",
}


def _parse_cell(column: str, raw: str) -> Any:
    """CSV cells are text; recover bools, numbers and the Python-repr lists/dicts
    the octree writer emits (``csv`` stringifies them with ``str()``)."""
    if raw is None:
        return None
    text = raw.strip()
    if text == "":
        return None
    if column in _TEXT_COLUMNS:
        return raw
    lowered = text.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    first = text[0]
    if first in "[{(":
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return raw
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return raw


ROLE_JSON_COLUMN = "role_json"


def _row_to_node_dict(row: dict[str, str]) -> dict[str, Any]:
    """Invert the octree writer's CSV flattening back to a node dict that
    ``NodeProperties.from_dict`` understands (``radiation_*`` -> ``radiation``).

    A role node carries its COMPLETE dict in ``role_json`` (heaters/sensors/
    cryocoolers/pairing that the flat columns can't represent); when present it is
    authoritative and rebuilds the node exactly as the full graph.json loader
    would. Plain cells fall back to the flat columns."""
    role_json = row.get(ROLE_JSON_COLUMN)
    if role_json and str(role_json).strip():
        try:
            data = json.loads(role_json)
            if isinstance(data, dict):
                return data
        except (ValueError, TypeError):
            pass  # corrupt cell -> fall back to the flat reconstruction below
    data: dict[str, Any] = {}
    radiation: dict[str, Any] = {}
    for column, raw in row.items():
        if column is None or column == ROLE_JSON_COLUMN:
            continue
        value = _parse_cell(column, raw)
        if value is None:
            continue
        if column.startswith("radiation_"):
            radiation[column[len("radiation_") :]] = value
        else:
            data[column] = value
    if radiation:
        data["radiation"] = radiation
    return data


def fast_load_has_roles(folder: str | Path) -> bool:
    """True if nodes.csv carries the ``role_json`` column, i.e. heaters, sensors
    and cryocoolers survive the fast load. Older nodes.csv (written before this
    column existed) returns False, so a role-dependent run falls back to the full
    graph.json loader instead of silently dropping its roles. Reads only the header
    line, not the whole (multi-million-row) file."""
    path = Path(folder) / "nodes.csv"
    if not path.exists():
        return False
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    except (OSError, StopIteration):
        return False
    return ROLE_JSON_COLUMN in header


def _load_metadata(folder: Path) -> GraphMetadata:
    path = folder / "metadata.json"
    if not path.exists():
        return GraphMetadata()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return GraphMetadata()
    if not isinstance(payload, dict):
        return GraphMetadata()
    return GraphMetadata.from_dict(payload.get("metadata", payload))


def _load_sparse_L(folder: Path, report: LoadReport) -> csr_matrix | None:
    """Binary CSR if present; otherwise parse the legacy COO JSON ONCE and cache it
    as ``L_sparse.npz`` so later runs skip the (multi-minute, multi-GB) JSON parse."""
    from .graph_io import _load_sparse_laplacian, _load_sparse_laplacian_npz

    npz_path = folder / "L_sparse.npz"
    json_path = folder / "L_sparse.json"
    # Only trust the cache while it is at least as new as the JSON; a rebuild that
    # rewrote only L_sparse.json must not be silently overridden by a stale matrix.
    stale_cache = (
        npz_path.exists()
        and json_path.exists()
        and json_path.stat().st_mtime > npz_path.stat().st_mtime + 1.0
    )
    if stale_cache:
        report.warnings.append("L_sparse.npz is older than L_sparse.json; rebuilding it.")
    else:
        matrix = _load_sparse_laplacian_npz(npz_path)
        if matrix is not None:
            return matrix
    matrix = _load_sparse_laplacian(json_path)
    if matrix is None:
        return None
    try:
        save_npz(str(folder / "L_sparse.npz"), matrix)
        report.warnings.append(
            "Cached L_sparse.npz from L_sparse.json; later runs load the binary form."
        )
    except (OSError, ValueError) as exc:  # noqa: BLE001 - caching is best-effort
        report.warnings.append(f"Could not cache L_sparse.npz ({exc}).")
    return matrix


def can_load_fast(folder: str | Path) -> tuple[bool, str]:
    """(usable, reason). ``nodes.csv`` is written when the octree is BUILT, while
    later GUI edits (cryocooler assignments, material/capacitance changes,
    setpoints) are saved into ``graph.json``. Loading the stale CSV would silently
    drop those -- e.g. every cryocooler, leaving a run with no cooling at all -- so
    a graph.json newer than nodes.csv disqualifies the fast path."""
    folder = Path(folder)
    required = ("nodes.csv", "node_ids.npy", "C.npy", EDGES_FILENAME)
    for name in required:
        if not (folder / name).exists():
            return False, f"missing {name}"
    if not ((folder / "L_sparse.npz").exists() or (folder / "L_sparse.json").exists()):
        return False, "missing L_sparse.npz/json"
    graph_json = folder / "graph.json"
    nodes_csv = folder / "nodes.csv"
    if graph_json.exists() and graph_json.stat().st_mtime > nodes_csv.stat().st_mtime + 1.0:
        return False, "graph.json is newer than nodes.csv (edited after build)"
    return True, "ok"


REFRESH_LOG_FILENAME = "refresh_fast_load.log"


def edges_only_refresh_is_enough(folder: str | Path) -> bool:
    """True when ``edges.npz`` is the ONLY missing fast-load artifact.

    Then the cheap streaming rebuild suffices: graph.json is walked edge by edge
    with bounded memory, instead of the full parse that needs ~45 GB on a 3M-cell
    graph. Everything else on disk is already consistent."""
    folder = Path(folder)
    if (folder / EDGES_FILENAME).exists():
        return False
    if not (folder / "graph.json").is_file():
        return False
    others = ("nodes.csv", "node_ids.npy", "C.npy")
    if not all((folder / name).exists() for name in others):
        return False
    return (folder / "L_sparse.npz").exists() or (folder / "L_sparse.json").exists()


MODAL_BUILD_LOG_FILENAME = "build_modal_controller.log"


def launch_modal_build_subprocess(
    folder: str | Path,
    *,
    t_op_K: float = 50.0,
    n_modes: int = 140,
    order: int = 50,
    effort: float = 0.2,
    integral_gain: float = 0.06,
    design_dt_s: float = 1.0,
) -> "subprocess.Popen":
    """Start ``build_modal_controller.py`` on ``folder`` in a SEPARATE process.

    The simulation tab designs controllers from a graph held in the GUI process,
    which for a 3M-cell graph means ~45 GB for graph.json plus an splu of the
    3M x 3M DC operator on top. This runs the same design off the fast-load
    artifacts at ~20 GB, detached, so it survives losing a remote session. Output
    goes to ``<folder>/build_modal_controller.log``.
    """
    import subprocess

    folder = Path(folder)
    script = Path(__file__).resolve().parent.parent / "build_modal_controller.py"
    log_path = folder / MODAL_BUILD_LOG_FILENAME
    log_handle = log_path.open("w", encoding="utf-8")
    command = [
        sys.executable,
        str(script),
        str(folder),
        "--t-op", f"{float(t_op_K):g}",
        "--modes", str(int(n_modes)),
        "--order", str(int(order)),
        "--effort", f"{float(effort):g}",
        "--integral", f"{float(integral_gain):g}",
        "--dt", f"{float(design_dt_s):g}",
    ]
    try:
        proc = subprocess.Popen(  # noqa: S603
            command,
            cwd=str(script.parent),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
    finally:
        log_handle.close()
    return proc


def launch_refresh_subprocess(
    folder: str | Path, *, edges_only: bool = False
) -> "subprocess.Popen":
    """Start ``refresh_fast_load.py`` on ``folder`` in a SEPARATE process.

    The refresh has to parse the (multi-GB) graph.json once, so it must never run
    in the GUI process -- that would reintroduce exactly the load this whole path
    exists to avoid. Output goes to ``<folder>/refresh_fast_load.log`` so a failure
    is diagnosable. The caller polls ``proc.poll()`` for completion.

    ``edges_only`` streams graph.json to rebuild only ``edges.npz`` -- bounded
    memory, for a graph built before that artifact existed whose full load would
    not fit in RAM.
    """
    import subprocess

    folder = Path(folder)
    script = Path(__file__).resolve().parent.parent / "refresh_fast_load.py"
    log_path = folder / REFRESH_LOG_FILENAME
    log_handle = log_path.open("w", encoding="utf-8")
    command = [sys.executable, str(script), str(folder)]
    if edges_only:
        command.append("--edges-only")
    try:
        proc = subprocess.Popen(  # noqa: S603
            command,
            cwd=str(script.parent),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
    finally:
        # The child inherited its own descriptor; the parent's copy can close.
        log_handle.close()
    return proc


def validate_against_matrices(model: ThermalGraphModel, matrices: dict[str, Any]) -> str | None:
    """Cross-check the CSV-derived per-node data against the authoritative binary
    matrices. ``C.npy`` is rewritten whenever the graph is saved, so a mismatch in
    total heat capacity means nodes.csv is stale in ways the mtime check missed.
    Returns an error string, or None when consistent."""
    # Structural invariants the full loader gets from validate_matrices(): the row
    # order (node_ids) must line up with C and L, and every matrix row must have a
    # node behind it. A mismatch here would silently apply the wrong node's
    # capacitance/conduction to a row.
    node_ids = np.asarray(matrices.get("node_ids"), dtype=int).reshape(-1)
    C = np.asarray(matrices.get("C"), dtype=float).reshape(-1)
    if node_ids.size and C.size and node_ids.size != C.size:
        return f"node_ids length {node_ids.size} != C length {C.size}."
    L = matrices.get("L")
    if L is not None and node_ids.size and tuple(L.shape) != (node_ids.size, node_ids.size):
        return f"L shape {tuple(L.shape)} does not match {node_ids.size} nodes."
    if node_ids.size:
        missing = int(np.setdiff1d(node_ids, np.fromiter(model.nodes, dtype=int)).size)
        if missing:
            return f"{missing} matrix node ids have no nodes.csv row."
    if C.size == 0:
        return None
    node_total = float(sum(float(getattr(n, "C_J_K", 0.0)) for n in model.nodes.values()))
    matrix_total = float(np.sum(C))
    if matrix_total <= 0.0:
        return None
    relative = abs(node_total - matrix_total) / abs(matrix_total)
    if relative > 0.01:
        return (
            f"nodes.csv total heat capacity {node_total:.6g} J/K disagrees with C.npy "
            f"{matrix_total:.6g} J/K by {relative:.1%}; the CSV is stale."
        )
    return None


def load_graph_for_simulation(
    folder: str | Path,
) -> tuple[ThermalGraphModel, dict[str, Any], LoadReport]:
    """Model (nodes + metadata only) and matrices for a headless simulation.

    Skips ``graph.json`` and the edge objects entirely. Raises ``FileNotFoundError``
    if the compact inputs are missing -- the caller should fall back to
    ``graph_io.load_graph_folder``.
    """
    folder = Path(folder)
    report = LoadReport()
    usable, reason = can_load_fast(folder)
    if not usable:
        raise FileNotFoundError(f"Low-memory load unavailable for {folder}: {reason}.")
    log_event("fast_graph_io load start", folder=str(folder))

    node_ids = np.load(folder / "node_ids.npy").astype(int).reshape(-1)
    matrices: dict[str, Any] = {
        "node_ids": node_ids,
        "C": np.load(folder / "C.npy").astype(float).reshape(-1),
    }
    sparse_l = _load_sparse_L(folder, report)
    if sparse_l is None:
        raise FileNotFoundError(f"No usable L matrix in {folder}.")
    matrices["L"] = sparse_l
    if (folder / "G_rad.npy").exists():
        matrices["G_rad"] = np.load(folder / "G_rad.npy").astype(float).reshape(-1)
    if (folder / "initial_temperature_K.npy").exists():
        matrices["initial_temperature_K"] = (
            np.load(folder / "initial_temperature_K.npy").astype(float).reshape(-1)
        )

    model = ThermalGraphModel(metadata=_load_metadata(folder))
    # Edges make this path LOSSLESS. Without them temperature-dependent properties
    # rebuild L(T) from an empty edge set -> all-zero Laplacian -> every node
    # thermally isolated (silent, and ruinous). can_load_fast requires the file, so
    # a graph without it falls back to the full graph.json loader rather than
    # simulating a subtly different model.
    from .fast_edge_io import read_edges_npz

    model.edges = read_edges_npz(folder)
    report.edge_count = len(model.edges)
    # csv's field-size limit is small relative to the list-valued columns the octree
    # writer emits for role nodes.
    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):  # pragma: no cover - platform dependent
        csv.field_size_limit(2**31 - 1)
    saw_setpoint_column = False
    with (folder / "nodes.csv").open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        saw_setpoint_column = "controller_setpoint_K" in (reader.fieldnames or [])
        for row in reader:
            node = NodeProperties.from_dict(_row_to_node_dict(row))
            model.nodes[node.node_id] = node
    report.node_count = len(model.nodes)

    if not saw_setpoint_column:
        report.warnings.append(
            "nodes.csv carries no per-node controller setpoint, so setpoints come "
            "from the run config (pass --setpoint / setpoints_K); any setpoints "
            "saved only in graph.json are not applied on this fast path."
        )
    missing = int(np.setdiff1d(node_ids, np.fromiter(model.nodes, dtype=int)).size)
    if missing:
        report.warnings.append(
            f"{missing} node ids in node_ids.npy have no nodes.csv row; their per-node "
            "metadata falls back to defaults."
        )
    log_event(
        "fast_graph_io load complete",
        folder=str(folder),
        nodes=report.node_count,
        edges=report.edge_count,
        L_nnz=int(sparse_l.nnz),
    )
    return model, matrices, report
