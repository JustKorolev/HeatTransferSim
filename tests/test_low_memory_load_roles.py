"""The role-blind fast load must be skipped for role-dependent runs.

nodes.csv carries no heaters/sensors/cryocoolers (they live only in graph.json),
so a run that drives heaters, runs a controller, or uses the cryocooler must fall
back to the full graph.json loader -- otherwise it silently loads a model with
zero of them and simulates an inert block (the 20260806-180811 run: sensors=0
heaters=0 cryo=0, temperatures diverging with no source).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from graph_visualizer.simulation_parameters import SimulationParameters
from graph_visualizer.simulation_runner import RunConfig, SimulationRunner


def _runner(**cfg_kwargs) -> SimulationRunner:
    with TemporaryDirectory() as directory:
        cfg = RunConfig(graph_folder=str(Path(directory) / "graph"), **cfg_kwargs)
        return SimulationRunner(cfg)


def test_heater_inputs_run_needs_roles() -> None:
    params = replace(SimulationParameters(), input_mode="heater_inputs", cryocooler_enabled=False)
    runner = _runner(params=params, low_memory_load=True)
    assert runner._run_needs_graph_roles() is True


def test_cryocooler_run_needs_roles() -> None:
    params = replace(SimulationParameters(), input_mode="zero", cryocooler_enabled=True)
    runner = _runner(params=params)
    assert runner._run_needs_graph_roles() is True


def test_controller_path_forces_roles_even_without_params() -> None:
    runner = _runner(controller_path="somewhere/modal_controller.npz")
    assert runner._run_needs_graph_roles() is True


def test_passive_role_free_run_may_use_fast_path() -> None:
    params = replace(
        SimulationParameters(),
        input_mode="zero",
        cryocooler_enabled=False,
        mimo_controller_enabled=False,
    )
    runner = _runner(params=params, low_memory_load=True)
    assert runner._run_needs_graph_roles() is False
