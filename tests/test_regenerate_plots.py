"""Redrawing a finished run's figures, without re-running it.

Useful precisely when the plotting changed rather than the physics -- the
controlled-sensor filter, say. The timeseries is already on disk; only the figures
are stale.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from graph_visualizer.simulation_runner import regenerate_plots


def _run_dir(tmp_path, *, with_manifest=True):
    t = np.arange(0.0, 40.0, 4.0)
    series = {
        "time_s": t,
        "sensor_0_K": 50.0 + 0.1 * t,      # monitor-only
        "sensor_1_K": 49.0 + 0.2 * t,      # controlled
        "sensor_0_err_K": 0.1 * t,
        "sensor_1_err_K": 0.2 * t,
        "rms_tracking_error_K": 0.3 + 0.0 * t,
        "rms_tracking_error_controlled_K": 0.2 + 0.0 * t,
        "avg_temp_K": 50.0 + 0.0 * t,
        "max_temp_K": 60.0 + 0.0 * t,
        "min_temp_K": 40.0 + 0.0 * t,
        "cryo_tip_K": 45.0 + 0.0 * t,
        "heater_7_W": 1.0 + 0.0 * t,
        "power_in_W": 5.0 + 0.0 * t,
        "power_out_W": 5.0 + 0.0 * t,
        "net_W": 0.0 * t,
        "max_temp_rate_K_per_s": 0.0 * t,
        "energy_drift_rel": 0.0 * t,
    }
    np.savez(tmp_path / "timeseries.npz", **series)
    if with_manifest:
        with (tmp_path / "sensors.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["series", "monitor_only"])
            w.writerow(["sensor_0", "True"])
            w.writerow(["sensor_1", "False"])
    return tmp_path


def test_plots_are_written_from_the_saved_timeseries(tmp_path) -> None:
    written = regenerate_plots(_run_dir(tmp_path))
    assert "sensor_temps.png" in written
    for name in written:
        assert (tmp_path / "plots" / name).stat().st_size > 0, name


def test_the_run_data_is_not_touched(tmp_path) -> None:
    """Only the figures are rewritten -- a button that could damage a finished
    overnight run would not be worth having."""
    run = _run_dir(tmp_path)
    before = (run / "timeseries.npz").read_bytes()
    regenerate_plots(run)
    assert (run / "timeseries.npz").read_bytes() == before
    assert (run / "sensors.csv").is_file()


def test_a_run_with_no_timeseries_says_so(tmp_path) -> None:
    """A run that died before its first sample has nothing to plot; that should be
    a clear message, not an obscure traceback."""
    with pytest.raises(FileNotFoundError, match="timeseries"):
        regenerate_plots(tmp_path)


def test_an_older_run_without_sensors_csv_still_plots(tmp_path) -> None:
    """Runs predating the manifest must not become unplottable -- fall back to
    plotting whatever series exist rather than nothing."""
    written = regenerate_plots(_run_dir(tmp_path, with_manifest=False))
    assert "sensor_temps.png" in written


# --- surviving a hard kill ----------------------------------------------------- #
def _writer(tmp_path, series):
    from graph_visualizer.simulation_runner import SimulationRunner

    r = object.__new__(SimulationRunner)
    r.out_dir = tmp_path
    r._series = series
    return r


def test_the_timeseries_is_readable_mid_run(tmp_path) -> None:
    """It used to be written only at finalize, so a machine freeze or power cycle
    kept the multi-GB checkpoints and lost every series the run had recorded."""
    r = _writer(tmp_path, {"time_s": [0.0, 4.0, 8.0], "rms_tracking_error_K": [2.0, 1.0, 0.5]})
    r._write_timeseries()
    with np.load(tmp_path / "timeseries.npz") as data:
        assert data["time_s"].tolist() == [0.0, 4.0, 8.0]
    assert (tmp_path / "timeseries.csv").is_file()


def test_a_later_flush_replaces_the_earlier_one_atomically(tmp_path) -> None:
    """Written to a temp name and renamed, so a kill during the write cannot replace
    a good file with a truncated one."""
    series = {"time_s": [0.0], "rms_tracking_error_K": [2.0]}
    r = _writer(tmp_path, series)
    r._write_timeseries()
    series["time_s"].append(4.0)
    series["rms_tracking_error_K"].append(1.0)
    r._write_timeseries()
    with np.load(tmp_path / "timeseries.npz") as data:
        assert data["time_s"].tolist() == [0.0, 4.0]
    assert not (tmp_path / "timeseries.npz.tmp").exists(), "no temp file left behind"


def test_a_run_killed_after_one_flush_can_still_be_plotted(tmp_path) -> None:
    """The point of the whole thing: a crashed run keeps enough to see what it did."""
    from graph_visualizer.simulation_runner import regenerate_plots

    r = _writer(tmp_path, {
        "time_s": [0.0, 4.0], "rms_tracking_error_K": [2.0, 1.0],
        "avg_temp_K": [50.0, 50.1], "max_temp_K": [60.0, 60.1], "min_temp_K": [40.0, 40.1],
    })
    r._write_timeseries()          # then imagine the machine dies here
    assert "system_temp.png" in regenerate_plots(tmp_path)


# --- which RMS the report headlines -------------------------------------------- #
def _summary(series):
    from graph_visualizer.simulation_runner import SimulationRunner

    r = object.__new__(SimulationRunner)
    r._series = series
    return SimulationRunner._summary_metrics(r)


def test_the_headline_rms_is_the_controlled_sensors() -> None:
    """The all-sensor figure is dominated by sensors the controller cannot act on --
    64 of 91 on no_mli_high_res -- so it tracks the plant's passive drift while
    reading as a controller result."""
    out = _summary({
        "time_s": [0.0, 4.0],
        "rms_tracking_error_K": [9.0, 8.0],
        "rms_tracking_error_controlled_K": [2.0, 0.5],
        "rms_tracking_error_monitor_K": [11.0, 10.0],
    })
    assert out["final_rms_tracking_error_K"] == pytest.approx(0.5)
    assert out["peak_rms_tracking_error_K"] == pytest.approx(2.0)


def test_the_other_figures_are_kept_and_named() -> None:
    """Monitor drift still matters -- it just is not the controller's score."""
    out = _summary({
        "time_s": [0.0],
        "rms_tracking_error_K": [9.0],
        "rms_tracking_error_controlled_K": [2.0],
        "rms_tracking_error_monitor_K": [11.0],
    })
    assert out["final_rms_tracking_error_all_sensors_K"] == pytest.approx(9.0)
    assert out["final_rms_tracking_error_monitor_K"] == pytest.approx(11.0)


def test_a_run_with_no_controlled_series_falls_back() -> None:
    """Older runs recorded only the all-sensor figure; they must still summarize."""
    out = _summary({"time_s": [0.0], "rms_tracking_error_K": [9.0, 7.0]})
    assert out["final_rms_tracking_error_K"] == pytest.approx(7.0)
