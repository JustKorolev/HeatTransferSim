"""Salvaging a crashed run's series from the fields it did write.

A run killed at the C level never reaches _finalize, so it writes no timeseries --
but every snapshot and checkpoint holds a FULL temperature field. On the run that
died at step 760 that is 19 fields spanning the whole run.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from tools.recover_run_series import _read_field, _samples, _sensor_manifest


def _run(tmp_path, n=6):
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "checkpoints").mkdir()
    for t in (4.0, 304.0):
        np.save(tmp_path / f"T_{t}.npy".replace("T_", "snapshots/T_"), np.full(n, 50.0 + t / 100))
    for step, t in ((99, 396.0), (181, 724.0)):
        np.savez(
            tmp_path / f"checkpoints/ckpt_{step:08d}.npz",
            temperatures_K=np.full(n, 60.0 + t / 100), time_s=t, step=step,
        )
    with (tmp_path / "sensors.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "node_id", "setpoint_K", "monitor_only"])
        w.writerow(["sensor_0", "10", "50.0", "False"])
        w.writerow(["sensor_1", "11", "50.0", "True"])
    return tmp_path


def test_snapshots_and_checkpoints_are_merged_in_time_order(tmp_path) -> None:
    """Both carry a full field, at different cadences; using only one throws away
    half the trajectory."""
    got = _samples(_run(tmp_path))
    assert [t for t, _, _ in got] == [4.0, 304.0, 396.0, 724.0]
    assert [k for _, _, k in got] == ["snapshot", "snapshot", "checkpoint", "checkpoint"]


def test_both_field_formats_read_back(tmp_path) -> None:
    run = _run(tmp_path)
    for _t, path, _kind in _samples(run):
        assert _read_field(path).shape == (6,)


def test_the_manifest_gives_the_runs_own_sensor_order(tmp_path) -> None:
    """Recovered series must key off sensors.csv, which records what the run
    actually used -- deriving it from the graph again could reorder them."""
    ids, setpoints, controlled = _sensor_manifest(_run(tmp_path))
    assert ids.tolist() == [10, 11]
    assert setpoints.tolist() == [50.0, 50.0]
    assert controlled.tolist() == [True, False]


def test_a_run_with_nothing_to_recover_says_so(tmp_path) -> None:
    from tools.recover_run_series import recover

    (tmp_path / "snapshots").mkdir()
    (tmp_path / "checkpoints").mkdir()
    with pytest.raises(FileNotFoundError, match="no snapshots or checkpoints"):
        recover(tmp_path, tmp_path)
