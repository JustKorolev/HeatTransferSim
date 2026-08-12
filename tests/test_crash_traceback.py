"""A native crash must leave evidence.

A segfault or access violation in a compiled extension kills the interpreter
outright: no Python exception, no finally, no report. The run directory simply
stops with status.json still saying "running" -- which is how a 760-step run
vanished with nothing to point at. faulthandler writes a stack for every thread
before the process dies, and it is the only evidence available for this class of
failure.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_a_native_crash_writes_a_traceback(tmp_path) -> None:
    script = tmp_path / "boom.py"
    script.write_text(
        f"import sys; sys.path.insert(0, r'{REPO}')\n"
        f"from run_simulation import _enable_crash_traceback\n"
        f"_enable_crash_traceback(r'{tmp_path}')\n"
        "import ctypes; ctypes.string_at(0)\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, str(script)], capture_output=True, timeout=120)
    log = tmp_path / "crash.log"
    assert log.is_file(), "a native crash must leave crash.log"
    text = log.read_text(encoding="utf-8")
    assert "access violation" in text.lower() or "fatal" in text.lower(), text


def test_it_appends_so_a_resume_keeps_the_earlier_crash(tmp_path) -> None:
    """A resumed run writes into the same directory; overwriting would destroy the
    record of the crash that made the resume necessary."""
    from run_simulation import _enable_crash_traceback

    _enable_crash_traceback(str(tmp_path))
    _enable_crash_traceback(str(tmp_path))
    assert (tmp_path / "crash.log").read_text(encoding="utf-8").count("run started") == 2


def test_no_run_dir_is_a_no_op() -> None:
    from run_simulation import _enable_crash_traceback

    _enable_crash_traceback(None)      # must not raise


def test_an_unwritable_target_does_not_stop_the_run(tmp_path) -> None:
    """Diagnostics must never be the reason a run fails to start."""
    from run_simulation import _enable_crash_traceback

    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    _enable_crash_traceback(str(blocker / "run"))   # must not raise
