"""A run must not silently target the built-in 293.15 K default.

NodeProperties.controller_setpoint_K defaults to 293.15 (room temperature) and
the low-memory nodes.csv path carries no per-node setpoint, so a run launched
without --setpoint kept that default on every sensor. A 40 K cryostat run then
commanded its heaters flat-out toward room temperature for the full duration --
observed as a completed 3600 s run whose tracking error started at -253 K and
never came near zero.

Also splits the reported tracking error: averaging monitor-only sensors (no heater
assigned, so the controller never acts on them) into the headline number made a
converged loop look broken -- 27 controlled sensors at 0.42 K RMS reported as
4.57 K once diluted by 64 unregulated ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from graph_visualizer.simulation_runner import RunConfig, SimulationRunner, _HardFailure


def _runner(tmp_path: Path, **kwargs) -> SimulationRunner:
    cfg = RunConfig(graph_folder=str(tmp_path / "g"), **kwargs)
    runner = SimulationRunner(cfg)
    runner.out_dir = tmp_path
    return runner


class _Prepared:
    def __init__(self, count: int) -> None:
        self.node_ids = np.arange(count)
        self.model = type("M", (), {"nodes": {}})()


def _write(runner, setpoints):
    runner._write_sensor_manifest(_Prepared(len(setpoints)), list(range(len(setpoints))),
                                  np.array(setpoints, dtype=float))


def test_untouched_default_setpoint_aborts_the_run(tmp_path) -> None:
    runner = _runner(tmp_path)
    with pytest.raises(_HardFailure, match="293.15"):
        _write(runner, [293.15] * 91)


def test_an_explicit_global_setpoint_is_fine(tmp_path) -> None:
    runner = _runner(tmp_path, global_setpoint_K=293.15)
    _write(runner, [293.15] * 91)  # deliberate room-temperature run


def test_per_sensor_setpoints_also_count_as_explicit(tmp_path) -> None:
    runner = _runner(tmp_path, setpoints_K={0: 293.15})
    _write(runner, [293.15] * 4)


def test_the_escape_hatch_works(tmp_path) -> None:
    runner = _runner(tmp_path, allow_default_setpoint=True)
    _write(runner, [293.15] * 4)


def test_a_cryogenic_setpoint_is_never_flagged(tmp_path) -> None:
    runner = _runner(tmp_path)
    _write(runner, [50.1662] * 91)


def test_mixed_setpoints_are_not_the_default(tmp_path) -> None:
    runner = _runner(tmp_path)
    _write(runner, [293.15, 50.0, 293.15])
