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
