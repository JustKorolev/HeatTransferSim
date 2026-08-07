"""A heater that cannot conduct heat to its sensor must be caught at prepare time.

This is the no_mli_high_res failure: contact detection left the heater parts
thermally floating, so 690 W of controller power heated ~23.6k stranded nodes to
94 K while 99.2% of the graph sat bit-for-bit unchanged and every sensor reported
its initial temperature forever. The tracking error is then constant by
construction and the integrator winds up to saturation.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from graph_visualizer.simulation_runner import RunConfig, SimulationRunner

pytest.importorskip("scipy")

from scipy.sparse import csr_matrix  # noqa: E402


class _Node:
    def __init__(self, **kw) -> None:
        self.assigned_sensor_id = None
        self.power_deposition_node_ids: list[int] = []
        self.readout_node_ids: list[int] = []
        self.sensor_connected_node_ids: list[int] = []
        for key, value in kw.items():
            setattr(self, key, value)


class _Model:
    def __init__(self, nodes: dict) -> None:
        self.nodes = nodes


class _Prepared:
    """Two disjoint islands: nodes {0,1} and nodes {2,3}."""

    def __init__(self, *, connected: bool) -> None:
        adj = np.zeros((4, 4), dtype=float)
        adj[0, 1] = adj[1, 0] = 1.0
        adj[2, 3] = adj[3, 2] = 1.0
        if connected:
            adj[1, 2] = adj[2, 1] = 1.0
        self.A = csr_matrix(adj)
        self.node_ids = np.array([10, 11, 20, 21])
        self.node_index_by_id = {10: 0, 11: 1, 20: 2, 21: 3}
        self.heater_node_ids = [100]
        heater = _Node(assigned_sensor_id=200, power_deposition_node_ids=[10, 11])
        # Sensor reads the OTHER island when disconnected.
        sensor = _Node(readout_node_ids=[20, 21])
        self.model = _Model({100: heater, 200: sensor})


class _Params:
    enabled_heater_node_ids = None


def _runner() -> SimulationRunner:
    with TemporaryDirectory() as directory:
        return SimulationRunner(RunConfig(graph_folder=str(Path(directory) / "g")))


def test_disconnected_heater_and_sensor_aborts() -> None:
    runner = _runner()
    with pytest.raises(RuntimeError, match="connected component"):
        runner._check_actuator_connectivity(_Prepared(connected=False), _Params())


def test_connected_heater_and_sensor_passes() -> None:
    runner = _runner()
    # One island: deposition and readout share a component -> no abort.
    runner._check_actuator_connectivity(_Prepared(connected=True), _Params())
