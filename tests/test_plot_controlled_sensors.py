"""Individual sensor series belong to the sensors the controller can act on.

The logging cap used to take the first N sensors by INDEX. On no_mli_high_res that
spent 32 of 32 slots almost entirely on monitor-only sensors, so the plots showed
everything except the loop being tuned, and 27 controlled sensors shared whatever
was left.
"""

from __future__ import annotations

import numpy as np

from graph_visualizer.simulation_runner import SimulationRunner


def _runner(controlled_mask, cap):
    r = object.__new__(SimulationRunner)
    r._sensor_controlled = np.array(controlled_mask, dtype=bool)
    r.cfg = type("Cfg", (), {"max_logged_sensors": cap})()
    r._log_event = lambda *a, **k: None
    return r


def _select(r):
    cap = max(0, int(r.cfg.max_logged_sensors))
    controlled_first = [int(j) for j in np.where(r._sensor_controlled)[0]]
    monitor_rest = [int(j) for j in np.where(~r._sensor_controlled)[0]]
    order = (controlled_first + monitor_rest)[:cap]
    keys = {f"sensor_{j}_K" for j in order if r._sensor_controlled[j]}
    return order, keys


def test_controlled_sensors_claim_the_logging_budget_first() -> None:
    # 5 sensors, only indices 3 and 4 controlled, room for 3 series.
    r = _runner([False, False, False, True, True], cap=3)
    order, keys = _select(r)
    assert order[:2] == [3, 4], "controlled sensors must come first"
    assert keys == {"sensor_3_K", "sensor_4_K"}
    assert len(order) == 3, "the remaining slot still goes to a monitor-only sensor"


def test_series_keep_their_original_sensor_index() -> None:
    """sensor_<j> has to keep matching the 'series' column in sensors.csv, or the
    plots and the manifest would refer to different sensors."""
    r = _runner([False, True, False, True], cap=4)
    order, keys = _select(r)
    assert set(order) == {0, 1, 2, 3}
    assert keys == {"sensor_1_K", "sensor_3_K"}, "indices are original, not renumbered"


def test_every_sensor_is_still_logged_when_the_cap_allows() -> None:
    r = _runner([True] * 4, cap=32)
    order, _keys = _select(r)
    assert order == [0, 1, 2, 3]
