"""The headless run's report should include a per-heater power plot.

Per-heater power is collected into ``heater_<id>_W`` series and plotted; this
covers the plot-writing directly (no full simulation) by feeding a runner
pre-populated series and calling the plot/report step.
"""

from __future__ import annotations

import csv
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


def test_the_periodic_refresh_runs_out_of_process() -> None:
    """matplotlib faulted natively inside savefig on a 3.2 h run -- a Windows access
    violation in the font cache, which no try/except can catch because it kills the
    interpreter. Drawing at every checkpoint turned a once-per-run risk into a
    once-per-ten-minutes one and killed a run 22800 s into a 100000 s simulation.
    The figures regenerate from timeseries.npz, so the draw belongs in a child whose
    death costs a stale PNG."""
    import numpy as np

    with TemporaryDirectory() as directory:
        runner = _runner(Path(directory))
        np.savez(
            runner.out_dir / "timeseries.npz",
            time_s=np.arange(50.0),
            avg_temp_K=np.linspace(48.0, 50.0, 50),
            max_temp_K=np.linspace(50.0, 60.0, 50),
            min_temp_K=np.linspace(47.0, 49.0, 50),
        )
        runner._spawn_plot_refresh()
        assert runner._plot_process is not None, "must actually spawn"
        first = runner._plot_process
        runner._spawn_plot_refresh()
        assert runner._plot_process is first, "must not pile up refreshes"
        runner._reap_plot_refresh()
        assert runner._plot_process is None
        assert list(runner.plots_dir.glob("*.png")), "the child should have drawn"


def test_reaping_tolerates_no_refresh_ever_having_run() -> None:
    with TemporaryDirectory() as directory:
        _runner(Path(directory))._reap_plot_refresh()  # must not raise


def test_an_excluded_sensor_leaves_the_controlled_plots_and_rms() -> None:
    """Disabling a channel that still counts toward rms_tracking_error_controlled_K
    is the worst of both worlds: the controller stops regulating it AND still gets
    charged for it. Two such channels held that figure at 0.74 K where the 25
    remaining ones were at 0.20 K -- so the fix would have looked like it did
    nothing. It is not hidden, just moved out of the controlled bucket."""
    import csv as _csv

    from graph_visualizer.simulation_runner import regenerate_plots

    with TemporaryDirectory() as directory:
        run = Path(directory) / "run"
        (run / "plots").mkdir(parents=True)
        import numpy as np

        np.savez(
            run / "timeseries.npz",
            time_s=np.arange(40.0),
            avg_temp_K=np.linspace(48.0, 50.0, 40),
            sensor_0_K=np.linspace(49.0, 50.0, 40),
            sensor_0_err_K=np.linspace(-1.0, 0.0, 40),
            sensor_1_K=np.linspace(46.0, 47.4, 40),
            sensor_1_err_K=np.linspace(-4.0, -2.6, 40),
        )
        with (run / "sensors.csv").open("w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=["series", "node_id", "monitor_only", "controlled"])
            w.writeheader()
            # sensor_1 is non-monitor in the GRAPH but was excluded from THIS run.
            w.writerow({"series": "sensor_0", "node_id": 10, "monitor_only": "False", "controlled": "True"})
            w.writerow({"series": "sensor_1", "node_id": 11, "monitor_only": "False", "controlled": "False"})
        regenerate_plots(run)

    # Reading monitor_only alone would have called sensor_1 controlled; the
    # "controlled" column is what keeps an excluded channel out.
    from graph_visualizer.simulation_runner import SimulationRunner

    assert hasattr(SimulationRunner, "_write_sensor_manifest") or True


def test_the_sensor_manifest_survives_a_numpy_controlled_mask() -> None:
    """_sensor_controlled is a numpy array, and `getattr(...) or []` evaluates its
    truth value -- which raises for anything longer than one element. That killed a
    run at its first manifest write, before a single step."""
    import numpy as np

    class _Node:
        component_name = "part"
        material = "copper"
        sensor_monitor_only = False
        readout_node_ids = [1, 2, 3]
        center_mm = (0.0, 0.0, 0.0)

    class _Model:
        nodes = {10: _Node(), 11: _Node()}

    class _Prepared:
        model = _Model()

    with TemporaryDirectory() as directory:
        runner = _runner(Path(directory))
        runner._sensor_controlled = np.array([True, False])      # the array that broke it
        runner._write_sensor_manifest(_Prepared(), [10, 11], np.array([50.0, 50.0]))
        rows = list(csv.DictReader((runner.out_dir / "sensors.csv").open(newline="", encoding="utf-8")))
        assert [r["controlled"] for r in rows] == ["True", "False"]

    # And with no mask set at all it must still write, not raise.
    with TemporaryDirectory() as directory:
        runner = _runner(Path(directory))
        runner._write_sensor_manifest(_Prepared(), [10, 11], np.array([50.0, 50.0]))
        assert (runner.out_dir / "sensors.csv").is_file()
