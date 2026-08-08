"""The headless tab must say plainly whether the fast load is usable.

edges.npz is what makes the low-memory path lossless: without it, L(T) cannot be
rebuilt and the run silently falls back to constant properties. The status line
exists so that is visible BEFORE launching an overnight run rather than being
discovered in events.log afterwards.

_fast_load_status is pure logic (it never touches self or Qt), so it is exercised
directly -- this environment's PySide6 cannot load its Qt DLLs.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

for _name in ("PySide6", "PySide6.QtCore", "PySide6.QtWidgets", "PySide6.QtGui"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

from graph_visualizer.fast_edge_io import write_edges_npz  # noqa: E402
from graph_visualizer.fast_graph_io import edges_only_refresh_is_enough  # noqa: E402
from graph_visualizer.headless_run_tab import HeadlessRunTab  # noqa: E402
from graph_visualizer.models import ThermalGraphModel  # noqa: E402


def _folder(tmp_path: Path, *, with_edges: bool) -> Path:
    folder = tmp_path / "demo"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("nodes.csv", "node_ids.npy", "C.npy", "L_sparse.npz", "graph.json"):
        (folder / name).write_bytes(b"x")
    if with_edges:
        write_edges_npz(ThermalGraphModel(), folder)
    return folder


def test_missing_edges_is_called_out_and_routes_to_the_cheap_rebuild(tmp_path) -> None:
    folder = _folder(tmp_path, with_edges=False)
    status = HeadlessRunTab._fast_load_status(None, folder)
    assert "MISSING edges.npz" in status
    # Only edges.npz is absent -> the button must stream instead of a full load.
    assert edges_only_refresh_is_enough(folder) is True


def test_complete_folder_reports_ready(tmp_path) -> None:
    folder = _folder(tmp_path, with_edges=True)
    status = HeadlessRunTab._fast_load_status(None, folder)
    assert "READY" in status
    assert edges_only_refresh_is_enough(folder) is False


def test_status_is_encodable_for_a_cp1252_console(tmp_path) -> None:
    """Status text can reach a cp1252 console/log; keep it encodable."""
    for index, with_edges in enumerate((False, True)):
        base = tmp_path / f"case{index}"
        base.mkdir()
        folder = _folder(base, with_edges=with_edges)
        HeadlessRunTab._fast_load_status(None, folder).encode("cp1252")


def test_unusable_folder_explains_why(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "unavailable" in HeadlessRunTab._fast_load_status(None, empty)
