"""A stopped headless run must finalize (plots/report), not be hard-killed.

The headless tab launches the run as a detached, console-less subprocess, so it
cannot deliver SIGINT/SIGTERM; on Windows terminate() is an uncatchable kill that
skips _finalize entirely (hence: no plots). "Stop Run" instead drops a
stop-request file that the run honors at the next step boundary and exits through
_finalize normally. These tests cover that handshake at the runner level.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from graph_visualizer.simulation_runner import (
    STOP_REQUEST_FILENAME,
    RunConfig,
    SimulationRunner,
)


def _runner(root: Path) -> SimulationRunner:
    run_dir = root / "run"
    cfg = RunConfig(graph_folder=str(root / "graph"), run_dir=str(run_dir))
    runner = SimulationRunner(cfg)
    runner.out_dir.mkdir(parents=True, exist_ok=True)
    return runner


def test_stop_request_file_triggers_a_graceful_stop() -> None:
    with TemporaryDirectory() as directory:
        runner = _runner(Path(directory))
        assert runner._should_stop() is False
        (runner.out_dir / STOP_REQUEST_FILENAME).write_text("stop", encoding="utf-8")
        assert runner._should_stop() is True
        # A graceful stop keeps its own status (so run() won't overwrite it with
        # "completed") and reaches _finalize via run()'s finally.
        assert runner._exit_status == "stopped"


def test_run_clears_a_stale_stop_request_at_start() -> None:
    with TemporaryDirectory() as directory:
        runner = _runner(Path(directory))
        (runner.out_dir / STOP_REQUEST_FILENAME).write_text("stop", encoding="utf-8")
        # Mimic run()'s startup cleanup so a leftover request from a prior run into
        # this same dir (a resume) does not immediately stop the new run.
        try:
            (runner.out_dir / STOP_REQUEST_FILENAME).unlink()
        except OSError:
            pass
        assert not (runner.out_dir / STOP_REQUEST_FILENAME).exists()
        assert runner._should_stop() is False
