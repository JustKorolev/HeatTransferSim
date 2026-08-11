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
