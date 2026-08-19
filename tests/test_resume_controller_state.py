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
import pytest

from graph_visualizer.simulation_runner import (
    _AlignedSeries,
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
class _State:
    """Stands in for SimulationModel.SimulationState: only time_s is read here."""

    def __init__(self, time_s=0.0):
        self.time_s = float(time_s)


class _Prep:
    def __init__(self, n=4):
        self.node_ids = np.arange(n)
        self.initial_temperatures_K = np.full(n, 293.0)
        self.temperatures = None
        self.controller_last_power_by_heater = {}
        self.controller_mimo_pi_integral = None
        self.controller_mimo_pi_passive_K = None
        self.controller_modal_integral = None
        # set_temperatures stamps a fresh history entry at t=0; the resume has to
        # put the clock back, so the stub has to have a clock to put back.
        self.history = [_State(0.0)]
        self.history_index = 0

    def set_temperatures(self, t):
        self.temperatures = np.asarray(t, dtype=float).copy()
        self.history = [_State(0.0)]
        self.history_index = 0


def _runner(tmp_path, uniform_K):
    from graph_visualizer.simulation_runner import SimulationRunner

    r = object.__new__(SimulationRunner)
    r.out_dir = tmp_path
    r.ckpt_dir = tmp_path / "checkpoints"
    r.cfg = type("Cfg", (), {"initial_temperature_uniform_K": uniform_K})()
    r._initial_state = None
    r.events = []
    r._series = _AlignedSeries()
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


# --- the clock, the baseline, and the history --------------------------------- #
def test_a_resume_puts_the_clock_back(tmp_path) -> None:
    """set_temperatures stamps its history entry at t=0, and step_forward builds
    state.time_s as self.time_s + dt. Left at zero, a resumed run restarts its time
    axis on top of the reloaded history and measures t_final_s from the resume, so
    it silently runs another full duration."""
    _write_ckpt(tmp_path, [70.0, 71.0, 72.0, 73.0])
    r = _runner(tmp_path, uniform_K=None)
    p = _Prep()
    r._resume_if_checkpoint(p)
    assert p.history[p.history_index].time_s == 123.0


def test_a_resume_rebases_the_metric_baseline(tmp_path) -> None:
    """initial_temperatures_K is what the first step measures dT/dt and energy drift
    against. Left at the graph's on-disk temperatures, a resume from 50 K on a graph
    saved at 293 K reports ~8 K/s for a jump that never happened."""
    _write_ckpt(tmp_path, [70.0, 71.0, 72.0, 73.0])
    r = _runner(tmp_path, uniform_K=None)
    p = _Prep()
    assert p.initial_temperatures_K.tolist() == [293.0] * 4, "the stale baseline"
    r._resume_if_checkpoint(p)
    assert p.initial_temperatures_K.tolist() == [70.0, 71.0, 72.0, 73.0]


def _write_series(tmp_path, times, **cols):
    np.savez(tmp_path / "timeseries.npz", time_s=np.asarray(times, float),
             **{k: np.asarray(v, float) for k, v in cols.items()})


def test_a_resume_carries_the_earlier_timeseries_forward(tmp_path) -> None:
    """_write_timeseries rewrites the file from self._series with mode "w". With that
    dict starting empty, the first flush after a resume REPLACED hours of history
    with the few rows since resuming -- the opposite of what the tooltip promises."""
    _write_ckpt(tmp_path, [70.0, 71.0, 72.0, 73.0])
    _write_series(tmp_path, [0.0, 60.0, 120.0], avg_temp_K=[50.0, 51.0, 52.0])
    r = _runner(tmp_path, uniform_K=None)
    r._resume_if_checkpoint(_Prep())
    assert r._series["time_s"] == [0.0, 60.0, 120.0]
    assert r._series["avg_temp_K"] == [50.0, 51.0, 52.0]
    assert any("3 earlier timeseries row(s)" in m for k, m in r.events if k == "resumed")


def test_samples_past_the_checkpoint_are_dropped(tmp_path) -> None:
    """Both are flushed together but a kill can land between them, leaving samples
    beyond the state being resumed from. Keeping them puts a backwards step in the
    time axis."""
    _write_ckpt(tmp_path, [70.0, 71.0, 72.0, 73.0])          # time_s = 123.0
    _write_series(tmp_path, [0.0, 60.0, 120.0, 180.0, 240.0])
    r = _runner(tmp_path, uniform_K=None)
    r._resume_if_checkpoint(_Prep())
    assert r._series["time_s"] == [0.0, 60.0, 120.0], r._series["time_s"]


def test_no_earlier_timeseries_is_reported_not_silent(tmp_path) -> None:
    _write_ckpt(tmp_path, [70.0, 71.0, 72.0, 73.0])
    r = _runner(tmp_path, uniform_K=None)
    r._resume_if_checkpoint(_Prep())
    assert any("no earlier timeseries" in m for k, m in r.events if k == "resumed")


def test_an_unreadable_timeseries_does_not_kill_the_resume(tmp_path) -> None:
    """Losing the history is bad; losing the run because the history is corrupt is
    worse, and a resume is exactly when a file may be half-written."""
    _write_ckpt(tmp_path, [70.0, 71.0, 72.0, 73.0])
    (tmp_path / "timeseries.npz").write_bytes(b"not an npz")
    r = _runner(tmp_path, uniform_K=None)
    r._resume_if_checkpoint(_Prep())
    assert r._series == {}
    assert any(k == "resume_series_error" for k, _ in r.events), r.events


# --- column alignment --------------------------------------------------------- #
def test_a_column_first_seen_after_a_resume_is_front_padded() -> None:
    """_collect appends time_s first, then setdefault-appends the rest. A key that
    appears for the first time at row i must already hold i values or its whole
    column is shifted by i in the csv/npz -- silently: the file parses and the plot
    just draws the wrong curve."""
    s = _AlignedSeries()
    s["time_s"] = [0.0, 60.0, 120.0]          # reloaded history
    s.setdefault("time_s", []).append(180.0)  # _collect's first act
    s.setdefault("heater_9_W", []).append(1.5)
    assert len(s["heater_9_W"]) == len(s["time_s"]) == 4
    assert np.isnan(s["heater_9_W"][:3]).all()
    assert s["heater_9_W"][3] == 1.5


def test_alignment_costs_nothing_on_a_fresh_run() -> None:
    """The first row of a fresh run must not be padded: time_s is already length 1
    by the time the other keys are created."""
    s = _AlignedSeries()
    s.setdefault("time_s", []).append(0.0)
    s.setdefault("avg_temp_K", []).append(50.0)
    assert s["avg_temp_K"] == [50.0]


# --- provenance --------------------------------------------------------------- #
def _provenance_runner(tmp_path, uniform_K):
    from graph_visualizer.simulation_runner import RunConfig, SimulationRunner

    r = object.__new__(SimulationRunner)
    r.out_dir = tmp_path
    r.ckpt_dir = tmp_path / "checkpoints"
    r.graph_name = "g"
    r.graph_folder = tmp_path
    r.cfg = RunConfig(graph_folder=str(tmp_path), initial_temperature_uniform_K=uniform_K)
    r._initial_state = None
    return r


def test_provenance_says_checkpoint_not_uniform_on_a_resume(tmp_path) -> None:
    """The initial temperature is IGNORED on a resume, so recording it as the start
    state is how a resumed run later reads as though it began cold from a uniform
    field -- and that is the one field anyone checks to find out."""
    import json

    _write_ckpt(tmp_path, [70.0, 71.0, 72.0, 73.0])
    _provenance_runner(tmp_path, uniform_K=50.1)._write_config_and_provenance()
    p = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert p["initial_state"]["source"] == "checkpoint"
    assert p["initial_state"]["initial_temperature_uniform_K_ignored"] == 50.1
    assert p["resumed"] is True


def test_a_fresh_run_still_records_its_uniform_start(tmp_path) -> None:
    import json

    _provenance_runner(tmp_path, uniform_K=50.1)._write_config_and_provenance()
    p = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert p["initial_state"] == {"source": "uniform", "temperature_K": 50.1}
    assert "resumed" not in p


def test_a_resume_keeps_the_earlier_provenance(tmp_path) -> None:
    """A resume reuses the directory, so this is the only surviving record that the
    first leg ran at all -- config.json is a dump of the CURRENT config."""
    import json

    _provenance_runner(tmp_path, uniform_K=50.1)._write_config_and_provenance()
    first = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    _write_ckpt(tmp_path, [70.0, 71.0, 72.0, 73.0])
    _provenance_runner(tmp_path, uniform_K=50.1)._write_config_and_provenance()
    second = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert second["previous"]["initial_state"] == first["initial_state"]


def test_resuming_a_run_that_logged_fewer_columns(tmp_path) -> None:
    """Real folders on this machine carry 106 or 107 columns depending on when they
    ran. Resuming a 106-column run with the 107th now being logged must front-pad the
    new column, not start it at row 0 next to rows from hours earlier."""
    _write_ckpt(tmp_path, [70.0, 71.0, 72.0, 73.0])
    _write_series(tmp_path, [0.0, 60.0, 120.0], avg_temp_K=[50.0, 51.0, 52.0])
    r = _runner(tmp_path, uniform_K=None)
    r._resume_if_checkpoint(_Prep())
    # ... the run continues, and _collect appends time_s first each step.
    r._series.setdefault("time_s", []).append(180.0)
    r._series.setdefault("avg_temp_K", []).append(53.0)
    r._series.setdefault("integral_held_count", []).append(1.0)   # new since that run
    n = len(r._series["time_s"])
    assert {len(v) for v in r._series.values()} == {n}, "every column must stay aligned"
    assert np.isnan(r._series["integral_held_count"][:3]).all()
    assert r._series["integral_held_count"][3] == 1.0


def test_a_column_that_stopped_being_logged_still_aligns(tmp_path) -> None:
    """The mirror case: a heater disabled since the earlier leg. Its old samples must
    stay on their own rows, so the gap goes at the END of that column."""
    _write_ckpt(tmp_path, [70.0, 71.0, 72.0, 73.0])
    np.savez(tmp_path / "timeseries.npz", time_s=np.array([0.0, 60.0, 120.0]),
             heater_9_W=np.array([1.0, 2.0]))   # short: stopped after two rows
    r = _runner(tmp_path, uniform_K=None)
    r._resume_if_checkpoint(_Prep())
    assert len(r._series["heater_9_W"]) == 3
    assert r._series["heater_9_W"][:2] == [1.0, 2.0]
    assert np.isnan(r._series["heater_9_W"][2])


def test_the_earlier_parameter_file_is_kept_on_a_resume(tmp_path) -> None:
    """A resume reuses the directory, and the tab writes the form's parameters into
    it before launching -- destroying the only record of what the earlier leg ran
    with, which is what its half of the timeseries has to be read against."""
    from graph_visualizer.headless_run_tab import HeadlessRunTab

    path = tmp_path / "simulation_parameters.json"
    path.write_text('{"mimo_pi_kp": 5.5}', encoding="utf-8")
    first = HeadlessRunTab._preserve_prior_parameters(path)
    assert first is not None and first.name == "simulation_parameters.leg1.json"
    assert first.read_text(encoding="utf-8") == '{"mimo_pi_kp": 5.5}'

    # Resumed twice: the second leg must not overwrite the first's archive.
    path.write_text('{"mimo_pi_kp": 2.0}', encoding="utf-8")
    second = HeadlessRunTab._preserve_prior_parameters(path)
    assert second.name == "simulation_parameters.leg2.json"
    assert first.read_text(encoding="utf-8") == '{"mimo_pi_kp": 5.5}', "leg1 clobbered"


def test_a_resume_continues_the_step_numbering(tmp_path) -> None:
    """Checkpoints are NAMED by the step counter. Left at 0, a resumed leg writes
    ckpt_00000020 onward, which sorts BELOW the ckpt_00001977 it resumed from -- so
    _available_checkpoints()[-1] keeps returning the old one and _prune_checkpoints
    (newest 3 by name) deletes the new ones. One run resumed from t=59310 s three
    times, discarding 11.3 h of completed progress on each.
    """
    _write_ckpt(tmp_path, [70.0, 71.0, 72.0, 73.0])          # ckpt_00000042, step=42
    r = _runner(tmp_path, uniform_K=None)
    r._resume_if_checkpoint(_Prep())
    assert r._resume_step == 42

    # ... so the next checkpoint this leg writes sorts ABOVE the one it resumed from.
    assert f"ckpt_{43:08d}.npz" > "ckpt_00000042.npz"


def test_a_fresh_run_starts_its_numbering_at_zero(tmp_path) -> None:
    r = _runner(tmp_path, uniform_K=None)
    r._resume_if_checkpoint(_Prep())
    assert int(getattr(r, "_resume_step", 0)) == 0


def test_a_checkpoint_without_a_step_field_still_resumes(tmp_path) -> None:
    """Old checkpoints must stay resumable; they just restart the numbering."""
    d = tmp_path / "checkpoints"
    d.mkdir()
    np.savez(d / "ckpt_00000042.npz", temperatures_K=np.zeros(4), time_s=123.0)
    r = _runner(tmp_path, uniform_K=None)
    r._resume_if_checkpoint(_Prep())
    assert r._resume_step == 0


def test_the_newest_checkpoint_is_chosen_by_time_not_by_filename(tmp_path) -> None:
    """Filenames are keyed on the step counter, and a leg resumed before that counter
    was restored numbered from 0 -- so files written hours later sort BEFORE the one
    they resumed from. This is the state real run directories are already in, and
    renumbering future writes cannot repair them."""
    d = tmp_path / "checkpoints"
    d.mkdir()
    np.savez(d / "ckpt_00001977.npz", temperatures_K=np.full(4, 60.0), time_s=59310.0, step=1977)
    np.savez(d / "ckpt_00001357.npz", temperatures_K=np.full(4, 46.0), time_s=100020.0, step=1357)
    r = _runner(tmp_path, uniform_K=None)
    p = _Prep()
    r._resume_if_checkpoint(p)
    assert p.history[p.history_index].time_s == 100020.0, "resumed from the stale one"
    assert p.temperatures.tolist() == [46.0] * 4


def test_an_unreadable_checkpoint_never_wins(tmp_path) -> None:
    """A half-written file must not strand the run at a state it cannot load, but it
    must not disappear either -- the spares exist for exactly this."""
    d = tmp_path / "checkpoints"
    d.mkdir()
    np.savez(d / "ckpt_00000010.npz", temperatures_K=np.full(4, 46.0), time_s=100020.0, step=10)
    (d / "ckpt_00000099.npz").write_bytes(b"truncated")
    r = _runner(tmp_path, uniform_K=None)
    p = _Prep()
    assert len(r._available_checkpoints()) == 2, "the bad file is kept as a spare"
    r._resume_if_checkpoint(p)
    assert p.temperatures.tolist() == [46.0] * 4


def test_a_corrupt_newest_checkpoint_falls_back_to_a_spare(tmp_path) -> None:
    """_CHECKPOINTS_KEPT exists so a corrupt write does not strand the run, but
    nothing acted on it: only the newest was opened, so a half-written newest failed
    the resume outright with two good spares sitting beside it."""
    d = tmp_path / "checkpoints"
    d.mkdir()
    np.savez(d / "ckpt_00000010.npz", temperatures_K=np.full(4, 46.0), time_s=100020.0, step=10)
    np.savez(d / "ckpt_00000020.npz", temperatures_K=np.full(4, 47.0), time_s=100050.0, step=20)
    np.savez(d / "ckpt_00000030.npz", time_s=100080.0, step=30)   # opens, no temperatures
    r = _runner(tmp_path, uniform_K=None)
    p = _Prep()
    r._resume_if_checkpoint(p)
    assert p.temperatures.tolist() == [47.0] * 4, "should use the newest READABLE one"
    assert any(k == "resume_checkpoint_unusable" for k, _ in r.events), r.events


def test_every_checkpoint_unreadable_stops_the_run(tmp_path) -> None:
    """Silently starting over from the initial temperature is the failure this whole
    area exists to prevent, so an all-corrupt directory must abort loudly."""
    from graph_visualizer.simulation_runner import _HardFailure

    d = tmp_path / "checkpoints"
    d.mkdir()
    for step in (10, 20):
        (d / f"ckpt_{step:08d}.npz").write_bytes(b"nope")
    r = _runner(tmp_path, uniform_K=None)
    with pytest.raises(_HardFailure):
        r._resume_if_checkpoint(_Prep())
