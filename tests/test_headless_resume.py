"""Resuming a headless run from its last checkpoint.

A multi-hour run that dies (the no_mli_high_res GPU crash took one out at step
400 of ~900, 53 minutes in) leaves checkpoints behind. Restarting from t=0 throws
that away, so the tab offers the previous run directly: pick it, and the run
reuses that directory, where run_simulation.py's _resume_if_checkpoint picks up
the newest checkpoint.

The scan/describe logic never touches Qt, so it is exercised directly -- this
environment's PySide6 cannot load its Qt DLLs.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

for _name in ("PySide6", "PySide6.QtCore", "PySide6.QtWidgets", "PySide6.QtGui"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

from graph_visualizer.headless_run_tab import HeadlessRunTab  # noqa: E402


def _run_with_checkpoints(root: Path, name: str, steps: list[int]) -> Path:
    run_dir = root / name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    for step in steps:
        np.savez(
            run_dir / "checkpoints" / f"ckpt_{step:08d}.npz",
            temperatures_K=np.full(4, 40.15),
            time_s=float(step) * 4.0,
            step=step,
        )
    return run_dir


def test_describe_checkpoint_reports_the_newest(tmp_path) -> None:
    run_dir = _run_with_checkpoints(tmp_path, "20260807-183010", [79, 159, 241, 325])
    assert HeadlessRunTab.describe_checkpoint(run_dir) == (325, 1300.0)


def test_run_without_checkpoints_is_not_offered(tmp_path) -> None:
    empty = tmp_path / "20260101-000000"
    (empty / "checkpoints").mkdir(parents=True)
    assert HeadlessRunTab.describe_checkpoint(empty) is None
    assert HeadlessRunTab.describe_checkpoint(tmp_path / "missing") is None


def test_corrupt_checkpoint_is_skipped_rather_than_raising(tmp_path) -> None:
    run_dir = tmp_path / "20260102-000000"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "ckpt_00000001.npz").write_bytes(b"not an npz")
    assert HeadlessRunTab.describe_checkpoint(run_dir) is None


def test_resumable_runs_lists_newest_first(tmp_path, monkeypatch) -> None:
    graph = "no_mli_high_res"
    root = tmp_path / "simulations" / graph
    _run_with_checkpoints(root, "20260807-120000", [10])
    _run_with_checkpoints(root, "20260807-183010", [325])
    (root / "20260807-999999").mkdir(parents=True)  # no checkpoints -> excluded

    tab = HeadlessRunTab.__new__(HeadlessRunTab)
    monkeypatch.setattr(
        HeadlessRunTab, "simulations_root", lambda self: tmp_path / "simulations"
    )
    found = tab.resumable_runs(graph)
    assert [p.name for p, _s, _t in found] == ["20260807-183010", "20260807-120000"]
    assert found[0][1] == 325 and found[0][2] == 1300.0


def test_unknown_graph_has_nothing_to_resume(tmp_path, monkeypatch) -> None:
    tab = HeadlessRunTab.__new__(HeadlessRunTab)
    monkeypatch.setattr(
        HeadlessRunTab, "simulations_root", lambda self: tmp_path / "simulations"
    )
    assert tab.resumable_runs("does_not_exist") == []
