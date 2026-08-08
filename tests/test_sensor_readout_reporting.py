"""Reported sensor temperature must be the READOUT, not the marker node.

A sensor marker is a thermally isolated single-node component -- its role-contact
edges carry G = 0 W/K -- so its temperature never changes. Indexing the
temperature vector at the marker row therefore reports the initial temperature for
the entire run. That is what pinned no_mli_high_res at "40.15 K, tracking error
9.70" while the real readouts had already risen +12.9 K and closed the RMS error
from 9.04 K to 7.96 K. The controller was regulating correctly the whole time; only
the report was wrong.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from graph_visualizer.simulation_runner import RunConfig, SimulationRunner

pytest.importorskip("scipy")


class _Node:
    def __init__(self, **kw) -> None:
        self.readout_node_ids: list[int] = []
        self.readout_weights: list[float] = []
        self.sensor_connected_node_ids: list[int] = []
        for key, value in kw.items():
            setattr(self, key, value)


class _Prepared:
    """Marker row 0 frozen; body cells 1 and 2 heat up."""

    def __init__(self, readout: list[int]) -> None:
        self.node_ids = np.array([100, 1, 2])
        self.node_index_by_id = {100: 0, 1: 1, 2: 2}
        self.model = type("M", (), {"nodes": {100: _Node(readout_node_ids=readout)}})()

    def power_balance_W(self) -> dict:
        return {"heater_W": 0.0, "cryocooler_W": 0.0, "radiation_W": 0.0, "net_W": 0.0}

    def heater_actuator_power_by_node(self) -> dict:
        return {}


class _State:
    def __init__(self, t, temps):
        self.time_s = t
        self.temperatures_K = temps


def _runner() -> SimulationRunner:
    with TemporaryDirectory() as directory:
        return SimulationRunner(RunConfig(graph_folder=str(Path(directory) / "g")))


def test_reported_temperature_follows_body_cells_not_the_frozen_marker() -> None:
    runner = _runner()
    prepared = _Prepared([1, 2])
    runner._build_sensor_readout_operator(prepared, [100], [0])
    # marker stays at 40.15; body cells rise to 50 and 60 -> readout 55.
    temps = np.array([40.15, 50.0, 60.0])
    got = runner._sensor_readout_temperatures(temps, [0])
    assert got[0] == pytest.approx(55.0), "must average the body cells, not read the marker"

    runner._collect(prepared, _State(1.0, temps), temps, temps.copy(), 1.0,
                    np.ones(3), [0], np.array([49.85]), [], [], runner.cfg.thresholds)
    assert runner._series["sensor_0_K"][-1] == pytest.approx(55.0)
    # Tracking error must reflect the real readout (55 - 49.85), not 40.15 - 49.85.
    assert runner._series["rms_tracking_error_K"][-1] == pytest.approx(5.15, abs=1e-6)


def test_sensor_without_readout_cells_falls_back_to_its_own_row() -> None:
    runner = _runner()
    prepared = _Prepared([])
    runner._build_sensor_readout_operator(prepared, [100], [0])
    temps = np.array([40.15, 50.0, 60.0])
    got = runner._sensor_readout_temperatures(temps, [0])
    assert got[0] == pytest.approx(40.15)


def test_weights_are_honoured() -> None:
    runner = _runner()
    prepared = _Prepared([1, 2])
    prepared.model.nodes[100].readout_weights = [3.0, 1.0]
    runner._build_sensor_readout_operator(prepared, [100], [0])
    temps = np.array([40.15, 50.0, 60.0])
    got = runner._sensor_readout_temperatures(temps, [0])
    assert got[0] == pytest.approx((3 * 50.0 + 1 * 60.0) / 4.0)
