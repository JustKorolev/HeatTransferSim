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
        runner = SimulationRunner(RunConfig(graph_folder=str(Path(directory) / "g")))
    runner._logged = []
    runner._log_event = lambda kind, message: runner._logged.append((kind, message))
    return runner


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
    # Small enough to fire on a large graph. At 20 it could not: a 3M-node run
    # took ~100 s/step, so the guard needed half an hour of known-garbage solving
    # before it would act, and runs were stopped by hand long before that.
    assert thr.energy_drift_abort_steps <= 5


def test_high_temperature_rate_counts_consecutive_violations() -> None:
    """The rate guard was documented as hard and implemented as a soft log line, so
    a run that hit 601 K/s carried on. It now counts toward an abort."""
    runner = _runner()
    thr = runner.cfg.thresholds
    assert thr.max_temperature_rate_abort_steps > 0

    prev = np.array([40.0, 40.0])
    hot = prev.copy()
    for i in range(3):
        hot = hot + 10.0 * thr.max_temperature_rate_K_per_s  # dt=1 -> rate well over
        _collect_step(runner, prev, hot.copy(), float(i))
        prev = hot.copy()
    assert runner._consecutive_high_rate == 3

    # A calm step resets it, so a single real transient cannot abort a run.
    _collect_step(runner, prev, prev.copy(), 99.0)
    assert runner._consecutive_high_rate == 0


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
    # Drift is only defined once there is a previous power sample, so the first two
    # entries are the NaN pad _AlignedSeries inserts to keep the column lined up
    # with time_s. Without that pad this column was written two rows early.
    assert len(drift) == len(runner._series["time_s"]), "column must align with time_s"
    assert np.isnan(drift[:2]).all(), f"the undefined rows must be NaN, not data: {drift}"
    # Every recorded step conserved energy exactly -> drift ~0, not ~0.13.
    assert np.nanmax(drift) < 1e-9, f"off-by-one reintroduced: {drift}"


class _Osc:
    """Feeds _log_command_oscillation a command history directly."""

    def __init__(self, runner, pattern):
        runner._series = {"time_s": list(range(len(pattern))), "heater_1_W": list(pattern)}
        self.runner = runner


def test_alternating_commands_are_reported_as_a_limit_cycle() -> None:
    """Two runs burned 3.4 h and 7.7 h looking healthy -- energy conserved,
    temperatures bounded, tracking even improving -- while the actuators bang-banged
    at the Nyquist frequency. The tell is one number, so the run should say it."""
    runner = _runner()
    _Osc(runner, [10.0 + 5.0 * (-1) ** i for i in range(60)])
    runner._log_command_oscillation(_State(600.0, np.zeros(2)))
    kinds = [k for k, _m in runner._logged]
    assert "command_oscillation" in kinds
    message = next(m for k, m in runner._logged if k == "command_oscillation")
    assert "ALTERNATING" in message and "mimo_pi_kp" in message


def test_a_settled_loop_is_not_reported() -> None:
    runner = _runner()
    _Osc(runner, list(np.linspace(10.0, 12.0, 60)))     # smooth ramp, no alternation
    runner._log_command_oscillation(_State(600.0, np.zeros(2)))
    assert not [k for k, _m in runner._logged if k == "command_oscillation"]


def test_still_commands_are_not_reported() -> None:
    """A loop holding steady has ac1 dominated by float noise; amplitude gates it."""
    runner = _runner()
    _Osc(runner, [10.0] * 60)
    runner._log_command_oscillation(_State(600.0, np.zeros(2)))
    assert not [k for k, _m in runner._logged if k == "command_oscillation"]
