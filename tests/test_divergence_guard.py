"""The runner should fail fast when the solve stops conserving energy.

A run that produces energy from nowhere (the 3443 K / drift~1.0 case) is
worthless; the guard counts consecutive gross-drift steps so the loop can abort
with a clear reason instead of burning hours. This covers the counter logic in
_collect deterministically (no full simulation).
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from graph_visualizer.simulation_runner import RunConfig, SimulationRunner


class _FakePrepared:
    """Minimal stand-in: reports zero net power, so any real dU/dt reads as drift."""

    def power_balance_W(self) -> dict:
        return {"heater_W": 0.0, "cryocooler_W": 0.0, "radiation_W": 0.0, "net_W": 0.0}

    def heater_power_by_node(self) -> dict:
        return {}


class _State:
    def __init__(self, t: float, temps: np.ndarray) -> None:
        self.time_s = t
        self.temperatures_K = temps


def _runner() -> SimulationRunner:
    with TemporaryDirectory() as directory:
        return SimulationRunner(RunConfig(graph_folder=str(Path(directory) / "g")))


def _collect_step(runner, prev, temps, t) -> None:
    thr = runner.cfg.thresholds
    C_diag = np.ones_like(temps)
    runner._collect(
        _FakePrepared(), _State(t, temps), temps, prev, 1.0,
        C_diag, [], np.array([]), [], [], thr,
    )


def test_consecutive_gross_drift_accumulates_then_resets() -> None:
    runner = _runner()
    prev = np.array([40.0, 40.0])
    # First 3 steps: drift is only computed once len(time_s) > 2, so the counter
    # cannot arm yet.
    for i in range(3):
        _collect_step(runner, prev, prev.copy(), float(i))
    assert runner._consecutive_high_drift == 0

    # Now feed steps with a large real dU/dt but net_W=0 -> drift ~1.0 each.
    hot = prev.copy()
    for i in range(3, 8):
        hot = hot + 100.0
        _collect_step(runner, prev, hot.copy(), float(i))
        prev = hot.copy()
    assert runner._consecutive_high_drift >= 4

    # A conserving step (no temperature change, net_W=0 -> drift 0) resets it.
    _collect_step(runner, prev, prev.copy(), 99.0)
    assert runner._consecutive_high_drift == 0


def test_abort_threshold_is_enabled_by_default() -> None:
    runner = _runner()
    thr = runner.cfg.thresholds
    assert thr.energy_drift_abort_steps > 0
    assert thr.energy_drift_abort_rel >= 0.5  # only genuine, gross divergence


class _RampPrepared:
    """Reports a net power that CHANGES every step, like a live controller."""

    def __init__(self, powers):
        self._powers = list(powers)
        self._i = 0

    def power_balance_W(self) -> dict:
        value = self._powers[min(self._i, len(self._powers) - 1)]
        self._i += 1
        return {"heater_W": value, "cryocooler_W": 0.0, "radiation_W": 0.0, "net_W": value}

    def heater_actuator_power_by_node(self) -> dict:
        return {}


def test_drift_compares_against_the_power_that_drove_the_step() -> None:
    """_collect runs AFTER the step, so power_balance_W() reflects the new state --
    the power for the NEXT step. Comparing it to the ΔT just taken manufactured a
    steady ~0.13 "drift" on a run that conserved energy fine."""
    runner = _runner()
    thr = runner.cfg.thresholds
    # Each step's ΔT is produced by the PREVIOUS sample's power (C=1, dt=1).
    powers = [100.0, 150.0, 90.0, 140.0, 95.0, 145.0]
    prepared = _RampPrepared(powers)
    temps = np.zeros(2)
    for step, driving in enumerate([None] + powers[:-1]):
        prev = temps.copy()
        if driving is not None:
            temps = prev + np.array([driving / 2.0, driving / 2.0])  # dU/dt == driving
        runner._collect(prepared, _State(float(step), temps), temps, prev, 1.0,
                        np.ones(2), [], np.array([]), [], [], thr)
    drift = runner._series.get("energy_drift_rel", [])
    assert drift, "drift series must be populated"
    # Every recorded step conserved energy exactly -> drift ~0, not ~0.13.
    assert max(drift) < 1e-9, f"off-by-one reintroduced: {drift}"
