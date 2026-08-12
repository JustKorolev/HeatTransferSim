"""A resume must restore the CONTROLLER, not just the temperatures.

Checkpoints used to carry temperatures alone, so resuming silently restarted the
controller cold. For MIMO PI that is worse than losing the integral: the passive
reference is captured as ``y - G u_prev``, and with u_prev back at zero a plant
already warmed by 20 W of heating is taken to BE the unheated equilibrium, so the
feedforward is sized against a reference that is wrong by exactly the heating the
run had already achieved.
"""

from __future__ import annotations

import numpy as np

from graph_visualizer.simulation_runner import (
    _optional_state_arrays,
    _restore_controller_state,
)


class _Prepared:
    def __init__(self, **kw):
        self.controller_last_power_by_heater = {}
        self.controller_mimo_pi_integral = None
        self.controller_mimo_pi_passive_K = None
        self.controller_modal_integral = None
        self.__dict__.update(kw)


def _roundtrip(source, target):
    payload = dict(_optional_state_arrays(source))
    ids = sorted(source.controller_last_power_by_heater)
    payload["controller_heater_ids"] = np.array(ids, dtype=np.int64)
    payload["controller_last_power_W"] = np.array(
        [source.controller_last_power_by_heater[h] for h in ids], dtype=float
    )
    return _restore_controller_state(target, payload)


def test_mimo_pi_integral_and_passive_reference_survive_a_resume() -> None:
    source = _Prepared(
        controller_last_power_by_heater={10: 4.0, 11: 6.5},
        controller_mimo_pi_integral=np.array([1.5, -2.0]),
        controller_mimo_pi_passive_K=np.array([40.0, 41.0]),
        _mimo_pi_sensor_ids=[20, 21],
    )
    target = _Prepared(_mimo_pi_sensor_ids=[20, 21])
    _roundtrip(source, target)
    assert target.controller_last_power_by_heater == {10: 4.0, 11: 6.5}
    assert np.array_equal(target.controller_mimo_pi_integral, [1.5, -2.0])
    # The reference must come back as the PASSIVE temperature, not the warm one.
    assert np.array_equal(target.controller_mimo_pi_passive_K, [40.0, 41.0])


def test_a_checkpoint_from_before_this_existed_still_resumes() -> None:
    """Old runs must remain resumable; they just start the controller cold."""
    target = _Prepared(_mimo_pi_sensor_ids=[20, 21])
    restored = _restore_controller_state(target, {"temperatures_K": np.zeros(3)})
    assert restored == ""
    assert target.controller_mimo_pi_integral is None


def test_state_from_a_different_controller_is_not_applied() -> None:
    """Resuming after rebuilding G with a different sensor count must start clean:
    those integrator entries index channels that no longer mean the same thing."""
    source = _Prepared(
        controller_last_power_by_heater={10: 1.0},
        controller_mimo_pi_integral=np.array([1.0, 2.0, 3.0]),
        controller_mimo_pi_passive_K=np.array([40.0, 41.0, 42.0]),
        _mimo_pi_sensor_ids=[1, 2, 3],
    )
    target = _Prepared(_mimo_pi_sensor_ids=[20, 21])   # G rebuilt: 2 channels now
    _roundtrip(source, target)
    assert target.controller_mimo_pi_integral is None
    assert target.controller_mimo_pi_passive_K is None


# --- which state a resume actually starts from -------------------------------- #
class _Prep:
    def __init__(self, n=4):
        self.node_ids = np.arange(n)
        self.initial_temperatures_K = np.full(n, 293.0)
        self.temperatures = None
        self.controller_last_power_by_heater = {}
        self.controller_mimo_pi_integral = None
        self.controller_mimo_pi_passive_K = None
        self.controller_modal_integral = None

    def set_temperatures(self, t):
        self.temperatures = np.asarray(t, dtype=float).copy()


def _runner(tmp_path, uniform_K):
    from graph_visualizer.simulation_runner import SimulationRunner

    r = object.__new__(SimulationRunner)
    r.out_dir = tmp_path
    r.ckpt_dir = tmp_path / "checkpoints"
    r.cfg = type("Cfg", (), {"initial_temperature_uniform_K": uniform_K})()
    r._initial_state = None
    r.events = []
    r._log_event = lambda kind, msg: r.events.append((kind, msg))
    return r


def _write_ckpt(tmp_path, temps):
    d = tmp_path / "checkpoints"
    d.mkdir(exist_ok=True)
    np.savez(d / "ckpt_00000042.npz", temperatures_K=np.asarray(temps, float),
             time_s=123.0, step=42)


def test_a_resume_starts_from_the_checkpoint_not_the_initial_temperature(tmp_path) -> None:
    """The reported bug: a resume came up at whatever initial temperature the
    parameters held, discarding where the run had actually got to."""
    _write_ckpt(tmp_path, [70.0, 71.0, 72.0, 73.0])
    r = _runner(tmp_path, uniform_K=48.0)
    p = _Prep()
    assert r._resolve_initial_temperatures(p) is not None, "an override IS configured"
    r._resume_if_checkpoint(p)
    assert p.temperatures is not None
    assert p.temperatures.tolist() == [70.0, 71.0, 72.0, 73.0], "checkpoint must win"


def test_a_missing_checkpoint_is_reported_not_silent(tmp_path) -> None:
    """Silence here turns "resume" into "start over" with nothing to show for it --
    the run looks normal and the plant is simply back at its initial temperature."""
    r = _runner(tmp_path, uniform_K=48.0)
    r._resume_if_checkpoint(_Prep())
    assert any(kind == "resume" and "no checkpoint found" in msg for kind, msg in r.events), r.events


def test_the_restored_temperatures_are_logged(tmp_path) -> None:
    """So a resume can be confirmed from the log alone, rather than inferred from
    the first plot."""
    _write_ckpt(tmp_path, [70.0, 80.0, 90.0, 100.0])
    r = _runner(tmp_path, uniform_K=None)
    r._resume_if_checkpoint(_Prep())
    resumed = [m for k, m in r.events if k == "resumed"]
    assert resumed and "min=70.00" in resumed[0] and "max=100.00" in resumed[0], resumed
