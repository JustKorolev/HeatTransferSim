"""Headless-tab settings must survive closing the app.

The tab always LOADED <graph>/simulation_parameters.json but only ever wrote into
the run directory, so every edit was lost on exit and the panel had to be set up
from scratch each session.
"""

from __future__ import annotations

import json

import pytest

from graph_visualizer.simulation_parameters import (
    SimulationParameters,
    load_simulation_parameters,
    save_simulation_parameters,
)


def test_saving_preserves_unknown_keys(tmp_path) -> None:
    """Other tools write into this file too. A round-trip that dropped their keys
    would quietly corrupt it every time a spinbox moved."""
    path = tmp_path / "simulation_parameters.json"
    save_simulation_parameters(path, SimulationParameters(dt_s=4.0), {"someone_elses": 7})
    params, extras = load_simulation_parameters(path)
    assert extras.get("someone_elses") == 7
    save_simulation_parameters(path, params, extras)
    assert json.loads(path.read_text(encoding="utf-8"))["someone_elses"] == 7


def test_edited_values_round_trip(tmp_path) -> None:
    path = tmp_path / "simulation_parameters.json"
    edited = SimulationParameters(
        dt_s=10.0, mimo_pi_kp=3.0, mimo_pi_ki=1.0e-4, mimo_rho_du=1.0e-2,
        tdep_rebuild_delta_K=0.25,
    )
    save_simulation_parameters(path, edited)
    back, _extras = load_simulation_parameters(path)
    for field in ("dt_s", "mimo_pi_kp", "mimo_pi_ki", "mimo_rho_du", "tdep_rebuild_delta_K"):
        assert getattr(back, field) == pytest.approx(getattr(edited, field)), field


def test_a_corrupt_file_does_not_block_saving(tmp_path) -> None:
    """A half-written file must not make the tab unable to save over it -- that
    would strand the user with settings they cannot persist."""
    path = tmp_path / "simulation_parameters.json"
    path.write_text("{not json", encoding="utf-8")
    params, extras = load_simulation_parameters(path)
    assert extras == {}
    save_simulation_parameters(path, params, extras)
    assert load_simulation_parameters(path)[0].dt_s == params.dt_s


def test_the_rebuild_threshold_has_a_control_and_is_read_back() -> None:
    """It was added to SimulationParameters with no widget, so the only way to set
    it was hand-editing JSON. A knob that is documented but unreachable is worse
    than no knob."""
    from tests.test_simulation_controls_panel import _build, MODE_HEADLESS

    panel, _form = _build(MODE_HEADLESS)
    assert "tdep_rebuild_delta_K" in panel.inputs
    panel.inputs["tdep_rebuild_delta_K"].setValue(0.5)
    assert panel.read(SimulationParameters()).tdep_rebuild_delta_K == pytest.approx(0.5)


# --- heater id picker ---------------------------------------------------------- #
def _heater_widget():
    from tests.test_simulation_controls_panel import _build, MODE_HEADLESS

    panel, _form = _build(MODE_HEADLESS)
    panel.build_readout_editor()
    return panel, panel.readout_editor_inputs["heater_id"]


def test_heater_id_is_a_picker_over_the_graphs_heaters() -> None:
    """It was a spin box over 0..1e9, so assigning a heater meant typing a 9-digit
    node id blind, with no feedback when it was wrong."""
    panel, widget = _heater_widget()
    panel.set_heater_id_choices([2988217, 2988220, 2988222])
    assert widget.value() == -1, "starts unassigned"
    widget.setValue(2988220)
    assert widget.value() == 2988220


def test_an_id_the_graph_no_longer_offers_is_kept_not_dropped() -> None:
    """Silently snapping to another heater would edit the node behind the user's
    back; the value is preserved and marked instead."""
    panel, widget = _heater_widget()
    panel.set_heater_id_choices([2988217, 2988220])
    widget.setValue(999999)
    assert widget.value() == 999999


def test_refreshing_the_choices_keeps_the_current_selection() -> None:
    """Switching graphs repopulates the list; losing the selection mid-edit would
    silently reassign the heater."""
    panel, widget = _heater_widget()
    panel.set_heater_id_choices([1, 2, 3])
    widget.setValue(2)
    panel.set_heater_id_choices([1, 2, 3, 4])
    assert widget.value() == 2
