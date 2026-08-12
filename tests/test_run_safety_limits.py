"""Two guards for overnight runs: bounded memory, and a bounded property rebuild."""

from __future__ import annotations

import numpy as np

from graph_visualizer import simulation_runner as R


class _Runner:
    def __init__(self, fraction):
        self.cfg = type("Cfg", (), {"max_rss_fraction": fraction})()
        self.events = []
        self._exit_status = "running"
        self._log_event = lambda kind, msg: self.events.append((kind, msg))

    hit = R.SimulationRunner._memory_ceiling_hit


def test_the_run_stops_before_the_machine_does(monkeypatch) -> None:
    """An overnight run that exhausts RAM froze Windows hard enough to need a power
    cycle, losing the run and the session. Stopping at a checkpoint costs only the
    remaining sim time."""
    monkeypatch.setattr(R, "_physical_memory_gib", lambda: 64.0)
    monkeypatch.setattr(R, "_process_rss_gib", lambda: 56.0)      # 87.5%
    r = _Runner(0.85)
    assert r.hit() is True
    assert r._exit_status == "stopped: memory ceiling"
    assert any(k == "memory_ceiling" for k, _ in r.events)


def test_normal_memory_use_does_not_stop_the_run(monkeypatch) -> None:
    monkeypatch.setattr(R, "_physical_memory_gib", lambda: 64.0)
    monkeypatch.setattr(R, "_process_rss_gib", lambda: 20.0)
    r = _Runner(0.85)
    assert r.hit() is False
    assert r._exit_status == "running"


def test_an_unmeasurable_machine_never_guesses_a_ceiling(monkeypatch) -> None:
    """Killing a healthy run because psutil is missing would be worse than the
    failure the guard exists to prevent."""
    monkeypatch.setattr(R, "_physical_memory_gib", lambda: 0.0)
    monkeypatch.setattr(R, "_process_rss_gib", lambda: 999.0)
    assert _Runner(0.85).hit() is False


def test_the_guard_can_be_switched_off(monkeypatch) -> None:
    monkeypatch.setattr(R, "_physical_memory_gib", lambda: 64.0)
    monkeypatch.setattr(R, "_process_rss_gib", lambda: 63.0)
    assert _Runner(0.0).hit() is False


# --- property rebuild gating -------------------------------------------------- #
class _Op:
    def __init__(self):
        self.calls = 0

    def rebuild(self, temps):
        self.calls += 1
        n = len(temps)
        return np.ones(n), np.ones(n), np.zeros((n, n))


def _sim(threshold):
    from graph_visualizer.simulation_model import PreparedSimulation
    from graph_visualizer.simulation_parameters import SimulationParameters
    from dataclasses import replace

    s = object.__new__(PreparedSimulation)
    s.temperature_dependent_operator = _Op()
    s.params = replace(SimulationParameters(), tdep_rebuild_delta_K=threshold)
    s.A = np.zeros((3, 3))
    s.z = np.append(np.full(3, 50.0), 0.0)
    return s


def test_a_small_drift_reuses_the_operator() -> None:
    """Properties are already lagged (semi-implicit, evaluated at step start), so
    holding them across a sub-threshold move is the same class of approximation with
    an explicit bound -- and it lets the stepper keep its stage-matrix cache."""
    s = _sim(0.25)
    s._refresh_temperature_dependent_operator(np.full(3, 50.0))
    assert s.temperature_dependent_operator.calls == 1
    s._refresh_temperature_dependent_operator(np.full(3, 50.1))   # 0.1 K < 0.25 K
    assert s.temperature_dependent_operator.calls == 1, "must reuse"
    assert s._tdep_rebuild_skips == 1


def test_crossing_the_threshold_rebuilds() -> None:
    s = _sim(0.25)
    s._refresh_temperature_dependent_operator(np.full(3, 50.0))
    s._refresh_temperature_dependent_operator(np.full(3, 50.5))
    assert s.temperature_dependent_operator.calls == 2


def test_drift_is_measured_from_the_last_REBUILD_not_the_last_step() -> None:
    """Otherwise many sub-threshold steps accumulate without ever triggering one,
    and the operator drifts arbitrarily far from the true temperature."""
    s = _sim(0.25)
    s._refresh_temperature_dependent_operator(np.full(3, 50.0))
    for t in (50.1, 50.2, 50.3):
        s._refresh_temperature_dependent_operator(np.full(3, t))
    assert s.temperature_dependent_operator.calls == 2, "50.3 is 0.3 K from the rebuild"


def test_zero_threshold_keeps_the_old_every_step_behaviour() -> None:
    s = _sim(0.0)
    for t in (50.0, 50.001, 50.002):
        s._refresh_temperature_dependent_operator(np.full(3, t))
    assert s.temperature_dependent_operator.calls == 3
