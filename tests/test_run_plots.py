"""The headless run's report should include a per-heater power plot.

Per-heater power is collected into ``heater_<id>_W`` series and plotted; this
covers the plot-writing directly (no full simulation) by feeding a runner
pre-populated series and calling the plot/report step.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from graph_visualizer.simulation_runner import RunConfig, SimulationRunner

pytest.importorskip("matplotlib")


def _runner(root: Path) -> SimulationRunner:
    cfg = RunConfig(graph_folder=str(root / "graph"), run_dir=str(root / "run"))
    runner = SimulationRunner(cfg)
    runner.plots_dir.mkdir(parents=True, exist_ok=True)
    return runner


def test_write_plots_emits_a_heater_power_plot() -> None:
    with TemporaryDirectory() as directory:
        runner = _runner(Path(directory))
        runner._series = {
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "avg_temp_K": [300.0, 299.0, 298.0, 297.0],
            "max_temp_K": [301.0, 300.0, 299.0, 298.0],
            "min_temp_K": [299.0, 298.0, 297.0, 296.0],
            "max_temp_rate_K_per_s": [0.0, 1.0, 1.0, 1.0],
            "power_in_W": [1.0, 2.0, 3.0, 4.0],
            "power_out_W": [0.5, 0.5, 0.5, 0.5],
            "net_W": [0.5, 1.5, 2.5, 3.5],
            "heater_10_W": [1.0, 1.5, 2.0, 2.5],
            "heater_20_W": [0.0, 0.5, 1.0, 1.5],
        }
        runner._write_plots_and_report()

        for name in ("heater_power.png", "power_balance.png", "temp_rate.png"):
            assert (runner.plots_dir / name).exists(), name


def test_no_heater_series_skips_the_heater_plot_without_error() -> None:
    with TemporaryDirectory() as directory:
        runner = _runner(Path(directory))
        runner._series = {
            "time_s": [0.0, 1.0],
            "avg_temp_K": [300.0, 299.0],
        }
        runner._write_plots_and_report()  # must not raise
        assert not (runner.plots_dir / "heater_power.png").exists()


def test_plots_are_rewritten_in_place_with_no_temp_files_left() -> None:
    """Plots are now redrawn at every checkpoint, not only at finalize, so a 12 h
    unattended run is inspectable while it runs. Each figure must overwrite its own
    filename -- and the save is staged through a temp name, because a kill mid-save
    would otherwise leave a truncated PNG where a readable one used to be."""
    with TemporaryDirectory() as directory:
        runner = _runner(Path(directory))
        runner._series = {
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "avg_temp_K": [300.0, 299.0, 298.0, 297.0],
            "max_temp_K": [301.0, 300.0, 299.0, 298.0],
            "min_temp_K": [299.0, 298.0, 297.0, 296.0],
        }
        first = runner._write_plots()
        assert first, "expected at least one figure"
        stamps = {name: (runner.plots_dir / name).stat().st_mtime_ns for name in first}

        # A later flush with more samples redraws the same files.
        runner._series["time_s"].append(4.0)
        for key in ("avg_temp_K", "max_temp_K", "min_temp_K"):
            runner._series[key].append(runner._series[key][-1] - 1.0)
        second = runner._write_plots()

        assert set(second) == set(first), "a redraw must not create new plot names"
        assert not list(runner.plots_dir.glob("*.tmp")), "staged files must be renamed away"
        assert all(
            (runner.plots_dir / name).stat().st_mtime_ns >= stamps[name] for name in first
        ), "the newest plot must overwrite the old one in place"
        assert all((runner.plots_dir / name).stat().st_size > 0 for name in first)
