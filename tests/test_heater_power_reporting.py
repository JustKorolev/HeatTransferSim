"""Per-heater power series must report the ACTUATOR command, not the source row.

A heater deposits its command onto ``power_deposition_node_ids`` (the body cells
it touches), never onto its own marker node. Indexing the deposited source vector
at the heater's row therefore reads 0 W for every real heater -- which is exactly
what a 630-step no_mli_high_res run showed: ``heater_*_W`` all zero while
``power_in_W`` climbed to ~690 W of genuinely injected controller power.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from graph_visualizer.simulation_runner import RunConfig, SimulationRunner


class _FakePrepared:
    """Marker-row readout is zero (the bug); the actuator command is the truth."""

    def power_balance_W(self) -> dict:
        return {"heater_W": 12.0, "cryocooler_W": 0.0, "radiation_W": 0.0, "net_W": 12.0}

    def heater_power_by_node(self) -> dict:
        return {10: 0.0, 20: 0.0}

    def heater_actuator_power_by_node(self) -> dict:
        return {10: 5.0, 20: 7.0}


class _State:
    def __init__(self, t: float, temps: np.ndarray) -> None:
        self.time_s = t
        self.temperatures_K = temps


def _runner() -> SimulationRunner:
    with TemporaryDirectory() as directory:
        return SimulationRunner(RunConfig(graph_folder=str(Path(directory) / "g")))


def test_heater_series_uses_actuator_command_not_marker_row() -> None:
    runner = _runner()
    temps = np.array([40.0, 40.0])
    thr = runner.cfg.thresholds
    runner._collect(
        _FakePrepared(), _State(1.0, temps), temps, temps.copy(), 1.0,
        np.ones_like(temps), [], np.array([]), [10, 20], [], thr,
    )
    # Without the fix both series are 0.0 and the run looks like "heaters never
    # turned on" even though 12 W is being delivered.
    assert runner._series["heater_10_W"] == [5.0]
    assert runner._series["heater_20_W"] == [7.0]
