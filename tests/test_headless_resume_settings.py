"""Picking a run to resume must bring that run's OWN settings into the form.

It used to leave the graph's saved parameters and a default setpoint table on
screen, so continuing a run silently changed the setpoints, the controlled-sensor
filter and the heater limits it had been running with -- the state carried over
and the configuration did not.

The fixtures mirror a real run directory on this machine (simulation_parameters
.json + setpoints.json + heater_overrides.json + config.json), including the two
120 W heater overrides and the 2-of-27 sensor exclusion.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from graph_visualizer.headless_run_tab import HeadlessRunTab
from graph_visualizer.simulation_parameters import (
    SimulationParameters,
    save_simulation_parameters,
)
from graph_visualizer.simulation_controls_panel import MODE_HEADLESS
from test_simulation_controls_panel import _QtStub, _build

SENSORS = [10, 11, 12]
HEATERS = [20, 21]


def _tab(tmp_path, *, enabled=None, setpoints=None, overrides=None, controller=""):
    """A tab stubbed down to what the resume loader touches."""
    tab = object.__new__(HeadlessRunTab)
    tab.QtWidgets = _QtStub.QtWidgets
    tab.QtCore = _QtStub.QtCore
    tab.status: list[tuple[str, bool]] = []
    tab._status = lambda message, error=False: tab.status.append((message, error))
    tab._params_source = ""

    panel, _form = _build(MODE_HEADLESS)
    tab.panel = panel
    tab.snapshot_spin = panel.snapshot_spin
    tab.checkpoint_spin = panel.checkpoint_spin
    tab.initial_spin = panel.initial_spin
    tab.use_initial = panel.use_initial
    tab.initial_temperature_all_spin = panel.initial_temperature_all_spin
    tab.controller_scheme_combo = panel.controller_scheme_combo

    # Sensor table: the third column is the "controlled" tick.
    tab.setpoint_table = _QtStub.QtWidgets.QTableWidget(len(SENSORS), 3)
    tab._sensor_rows_manifest = [
        {"node_id": str(s), "component_name": f"c{s}", "monitor_only": "true" if s == 12 else "false"}
        for s in SENSORS
    ]
    for index in range(len(SENSORS)):
        tab.setpoint_table.setItem(index, 1, _QtStub.QtWidgets.QTableWidgetItem(""))
        cell = _QtStub.QtWidgets.QTableWidgetItem("")
        cell.setCheckState(_QtStub.QtCore.Qt.Checked)
        tab.setpoint_table.setItem(index, 2, cell)

    tab.heater_table = _QtStub.QtWidgets.QTableWidget(len(HEATERS), 5)
    tab._heater_rows_manifest = [
        {"node_id": str(h), "component_name": f"h{h}"} for h in HEATERS
    ]
    for index in range(len(HEATERS)):
        for column in (1, 2, 3, 4):
            tab.heater_table.setItem(index, column, _QtStub.QtWidgets.QTableWidgetItem(""))

    run_dir = tmp_path / "20260817-232713"
    run_dir.mkdir()
    params = replace(
        SimulationParameters(),
        mimo_pi_kp=5.5,
        mimo_pi_ki=1.0e-4,
        mimo_pi_measurement_filter_s=900.0,
        mimo_pi_passive_reference_K=24.6,
        dt_s=30.0,
        enabled_sensor_node_ids=enabled,
        mimo_pi_gain_matrix_path=controller,
    )
    save_simulation_parameters(run_dir / "simulation_parameters.json", params)
    if setpoints is not None:
        (run_dir / "setpoints.json").write_text(json.dumps(setpoints), encoding="utf-8")
    if overrides is not None:
        (run_dir / "heater_overrides.json").write_text(json.dumps(overrides), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps({"snapshot_interval_s": 1800.0, "checkpoint_interval_s": 600.0}),
        encoding="utf-8",
    )
    tab.resume_combo = _QtStub.QtWidgets.QComboBox()
    tab.resume_combo.addItem("(fresh run)", None)
    tab.resume_combo.addItem(run_dir.name, str(run_dir))
    tab.resume_combo.setCurrentIndex(1)
    return tab, run_dir


def test_the_runs_parameters_land_in_the_form(tmp_path) -> None:
    tab, _ = _tab(tmp_path)
    tab.load_resume_run_settings()
    assert tab.panel.mimo_pi_kp_spin.value() == 5.5
    assert tab.panel.mimo_pi_ki_spin.value() == 1.0e-4
    assert tab.panel.mimo_pi_filter_spin.value() == 900.0
    assert tab.panel.mimo_pi_passive_spin.value() == 24.6
    assert tab.panel.read().dt_s == 30.0


def test_the_runs_setpoints_land_in_the_table(tmp_path) -> None:
    """The reported symptom: the table came up at the default setpoint, so resuming
    would have continued the run against different targets."""
    tab, _ = _tab(tmp_path, setpoints={"10": 50.0, "11": 49.5})
    tab.load_resume_run_settings()
    assert tab.setpoint_table.item(0, 1).text() == "50"
    assert tab.setpoint_table.item(1, 1).text() == "49.5"
    assert tab.collect_setpoint_overrides() == {10: 50.0, 11: 49.5}


def test_the_excluded_sensors_come_back_unticked(tmp_path) -> None:
    """The 2-of-27 exclusion is a controller decision, not a cosmetic one: an
    unticked sensor is out of the loop entirely."""
    tab, _ = _tab(tmp_path, enabled=[10])
    tab.load_resume_run_settings()
    assert tab.setpoint_table.item(0, 2).checkState() == tab.QtCore.Qt.Checked
    assert tab.setpoint_table.item(1, 2).checkState() == tab.QtCore.Qt.Unchecked
    assert tab.collect_enabled_sensors() == [10]


def test_no_filter_reticks_every_controllable_sensor(tmp_path) -> None:
    """None is the no-filter convention -- every controllable sensor was in the
    loop. It must not be read as "leave the ticks alone"."""
    tab, _ = _tab(tmp_path, enabled=None)
    tab.setpoint_table.item(1, 2).setCheckState(tab.QtCore.Qt.Unchecked)  # stale form
    tab.load_resume_run_settings()
    assert tab.setpoint_table.item(0, 2).checkState() == tab.QtCore.Qt.Checked
    assert tab.setpoint_table.item(1, 2).checkState() == tab.QtCore.Qt.Checked
    # The monitor-only row is never controlled under either convention.
    assert tab.setpoint_table.item(2, 2).checkState() == tab.QtCore.Qt.Unchecked


def test_the_heater_overrides_come_back(tmp_path) -> None:
    tab, _ = _tab(tmp_path, overrides={"20": {"heater_max_power_W": 120.0}})
    tab.load_resume_run_settings()
    assert tab.heater_table.item(0, 1).text() == "120"
    assert tab.collect_heater_overrides() == {20: {"heater_max_power_W": 120.0}}


def test_a_heater_the_run_did_not_override_comes_back_blank(tmp_path) -> None:
    """Cleared, not merged: a blank cell means "use the Controller defaults", so a
    stale override left in the form must not ride into the resumed run."""
    tab, _ = _tab(tmp_path, overrides={"20": {"heater_max_power_W": 120.0}})
    tab.heater_table.setItem(1, 1, tab.QtWidgets.QTableWidgetItem("999"))  # stale
    tab.load_resume_run_settings()
    assert tab.heater_table.item(1, 1).text() == ""
    assert 21 not in tab.collect_heater_overrides()


def test_the_intervals_come_from_the_config(tmp_path) -> None:
    """They live in config.json rather than the parameter file, so set_params cannot
    have restored them."""
    tab, _ = _tab(tmp_path)
    tab.load_resume_run_settings()
    assert tab.snapshot_spin.value() == 1800.0
    assert tab.checkpoint_spin.value() == 600.0


def test_the_controller_is_matched_by_name_not_by_path(tmp_path) -> None:
    """A run's recorded path is the path on the machine that produced it, so matching
    literally fails for every run copied off that machine."""
    tab, _ = _tab(tmp_path, controller=r"C:\\Users\\someone-else\\graphs\\g\\sys_id\\G_exact_T50K")
    local = tmp_path / "sys_id" / "G_exact_T50K"
    tab.controller_scheme_combo.addItem("(none)", ("none", ""))
    tab.controller_scheme_combo.addItem("MIMO PI - G_exact_T50K", ("mimo_pi", str(local)))
    tab.load_resume_run_settings()
    assert tab.controller_scheme_combo.currentData() == ("mimo_pi", str(local))


def test_a_controller_this_graph_does_not_have_is_reported(tmp_path) -> None:
    tab, _ = _tab(tmp_path, controller="/elsewhere/G_from_another_graph")
    tab.controller_scheme_combo.addItem("(none)", ("none", ""))
    tab.load_resume_run_settings()
    assert any("not in this graph's list" in m for m, _ in tab.status), tab.status


def test_the_initial_temperature_is_disabled_while_resuming(tmp_path) -> None:
    """It is IGNORED on a resume -- the checkpoint supplies the state. Leaving it
    live is what made it look like the run would start from it."""
    tab, _ = _tab(tmp_path)
    tab._sync_initial_temperature_enabled()
    assert not tab.initial_spin.isEnabled()
    assert not tab.use_initial.isEnabled()
    assert "Ignored while resuming" in tab.initial_spin.tooltip

    tab.resume_combo.setCurrentIndex(0)  # (fresh run)
    tab._sync_initial_temperature_enabled()
    assert tab.initial_spin.isEnabled()


def test_a_run_missing_its_sidecars_loads_what_it_has(tmp_path) -> None:
    """Older runs predate setpoints.json. A partial load beats none, as long as it
    says which part it could not do."""
    tab, run_dir = _tab(tmp_path)
    assert not (run_dir / "setpoints.json").exists()
    tab.load_resume_run_settings()
    assert tab.panel.mimo_pi_kp_spin.value() == 5.5, "parameters still loaded"
    assert any("NOT found" in m and "setpoints" in m for m, _ in tab.status), tab.status


def test_an_unreadable_parameter_file_does_not_kill_the_tab(tmp_path) -> None:
    tab, run_dir = _tab(tmp_path)
    (run_dir / "simulation_parameters.json").write_text("{not json", encoding="utf-8")
    tab.load_resume_run_settings()
    assert any("NOT found" in m for m, _ in tab.status), tab.status


def test_selecting_a_fresh_run_loads_nothing(tmp_path) -> None:
    tab, _ = _tab(tmp_path, setpoints={"10": 50.0})
    tab.resume_combo.setCurrentIndex(0)
    tab.load_resume_run_settings()
    assert tab.setpoint_table.item(0, 1).text() == "", "the form must be left alone"
    assert tab.status == []


def test_refreshing_the_list_does_not_revert_the_form(tmp_path) -> None:
    """refresh_resume_runs rebuilds the combo and re-selects the same target, which
    fires the loader again. Reloading then would discard everything tuned since the
    first load -- refreshing a list is not a request to revert the form."""
    tab, _ = _tab(tmp_path)
    tab.load_resume_run_settings()
    assert tab.panel.mimo_pi_kp_spin.value() == 5.5
    tab.panel.mimo_pi_kp_spin.setValue(2.0)      # a deliberate edit
    tab.load_resume_run_settings()               # as a refresh would
    assert tab.panel.mimo_pi_kp_spin.value() == 2.0, "the edit was reverted"


def test_switching_to_a_different_run_does_load(tmp_path) -> None:
    """The guard must key on the target, not simply latch after the first load."""
    tab, first = _tab(tmp_path)
    tab.load_resume_run_settings()
    tab.panel.mimo_pi_kp_spin.setValue(2.0)

    second = tmp_path / "20260818-090000"
    second.mkdir()
    save_simulation_parameters(
        second / "simulation_parameters.json",
        replace(SimulationParameters(), mimo_pi_kp=9.0),
    )
    tab.resume_combo.addItem(second.name, str(second))
    tab.resume_combo.setCurrentIndex(2)
    tab.load_resume_run_settings()
    assert tab.panel.mimo_pi_kp_spin.value() == 9.0
