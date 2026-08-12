"""A dying octree worker must leave evidence, and must not hang the parent.

A build lost its workers to what was almost certainly a native crash inside
embreex. Nothing was reported: the worker's interpreter died before it could raise,
and the parent sat in fut.result() with no timeout while the progress bar kept
redrawing a dead pool. The crash looked like slowness.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_a_worker_records_its_own_native_crash(tmp_path) -> None:
    script = tmp_path / "boom.py"
    script.write_text(
        f"import sys, os; sys.path.insert(0, r'{REPO}')\n"
        f"os.environ['OCTREE_CRASH_LOG_DIR'] = r'{tmp_path}'\n"
        "from octree_graph.octree import _enable_worker_faulthandler\n"
        "_enable_worker_faulthandler()\n"
        "import ctypes; ctypes.string_at(0)\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, str(script)], capture_output=True, timeout=120)
    log = tmp_path / "octree_worker_crash.log"
    assert log.is_file(), "a native worker death must leave a stack"
    text = log.read_text(encoding="utf-8").lower()
    assert "access violation" in text or "fatal" in text, text


def test_several_workers_append_rather_than_overwrite(tmp_path, monkeypatch) -> None:
    """Workers die in parallel and the informative one is not always the first."""
    from octree_graph.octree import _enable_worker_faulthandler

    monkeypatch.setenv("OCTREE_CRASH_LOG_DIR", str(tmp_path))
    _enable_worker_faulthandler()
    _enable_worker_faulthandler()
    assert (tmp_path / "octree_worker_crash.log").read_text(encoding="utf-8").count("worker pid=") == 2


def test_an_unwritable_log_target_does_not_stop_the_worker(tmp_path, monkeypatch) -> None:
    """Diagnostics must never be why a build fails to start."""
    from octree_graph.octree import _enable_worker_faulthandler

    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("OCTREE_CRASH_LOG_DIR", str(blocker / "sub"))
    _enable_worker_faulthandler()      # must not raise


def test_the_batch_timeout_is_generous_enough_not_to_fire_on_a_slow_build() -> None:
    """A false timeout on a healthy build would be worse than the hang it replaces:
    a deep cell against 10M triangles is legitimately slow."""
    from octree_graph.octree import _BATCH_TIMEOUT_BASE_S, _BATCH_TIMEOUT_PER_ITEM_S

    assert _BATCH_TIMEOUT_BASE_S >= 300.0
    assert _BATCH_TIMEOUT_PER_ITEM_S >= 1.0
    typical_batch = 512
    assert _BATCH_TIMEOUT_BASE_S + _BATCH_TIMEOUT_PER_ITEM_S * typical_batch >= 1800.0
