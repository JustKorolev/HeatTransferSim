"""Per-heater limit overrides from the headless tab.

Every heater runs on the Controller section's single default max power and slew
rate. That is the right default -- one number for a rack of identical heaters -- but
it was also the ONLY option from this tab: the per-node editor it used to show had
no graph to name heaters from, so its heater-id combo was permanently empty and
nothing it held was ever read.

The override table names every heater the build declares and sends only the cells
the user filled, so an untouched table changes nothing about the run.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

for _name in ("PySide6", "PySide6.QtCore", "PySide6.QtWidgets", "PySide6.QtGui"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

from graph_visualizer.headless_run_tab import HeadlessRunTab  # noqa: E402


def _graph(tmp_path: Path, heaters: list[tuple[int, str]], sensors: list[int] = ()) -> Path:
    folder = tmp_path / "graphs" / "g"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "node_ids.npy").write_bytes(b"")
    with (folder / "nodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["node_id", "component_name", "is_heater", "is_sensor", "sensor_monitor_only"],
        )
        writer.writeheader()
        for node_id, name in heaters:
            writer.writerow({
                "node_id": node_id, "component_name": name,
                "is_heater": "True", "is_sensor": "False", "sensor_monitor_only": "False",
            })
        for node_id in sensors:
            writer.writerow({
                "node_id": node_id, "component_name": f"S{node_id}",
                "is_heater": "False", "is_sensor": "True", "sensor_monitor_only": "False",
            })
    return folder


def _tab(tmp_path: Path, monkeypatch, heaters, sensors=()) -> HeadlessRunTab:
    import test_simulation_controls_panel as panel_stubs

    _graph(tmp_path, heaters, sensors)
    monkeypatch.setattr(
        HeadlessRunTab, "simulations_root", lambda self: tmp_path / "simulations"
    )
    return HeadlessRunTab(panel_stubs._QtStub, None, graphs_root=lambda: tmp_path / "graphs")


def _fill(tab, row: int, column: int, text: str) -> None:
    """Type into a cell the way the user does (the Qt stub's items are read-only)."""
    import test_simulation_controls_panel as panel_stubs

    tab.heater_table.setItem(row, column, panel_stubs.QTableWidgetItem(text))


# --- the table ------------------------------------------------------------------ #
def test_the_table_lists_the_graphs_heaters(tmp_path, monkeypatch) -> None:
    """The combo it replaces was empty in this tab: nothing populated it, because
    populating it needed the graph this tab never loads."""
    tab = _tab(tmp_path, monkeypatch, [(2988217, "HTR_A"), (2988220, "HTR_B")], sensors=[3])
    assert tab.heater_table.rowCount() == 2
    assert "2988217" in tab.heater_table.item(0, 0).text()
    assert "HTR_A" in tab.heater_table.item(0, 0).text()


def test_an_untouched_table_overrides_nothing(tmp_path, monkeypatch) -> None:
    """Blank means "use the Controller defaults", so the run is byte-for-byte what
    it would have been before this table existed."""
    tab = _tab(tmp_path, monkeypatch, [(10, "A"), (11, "B")])
    assert tab.collect_heater_overrides() == {}


def test_only_the_filled_cells_are_sent(tmp_path, monkeypatch) -> None:
    tab = _tab(tmp_path, monkeypatch, [(10, "A"), (11, "B"), (12, "C")])
    _fill(tab, 0, 1, "45")        # max power only
    _fill(tab, 1, 2, " 2.5 ")     # slew only
    _fill(tab, 2, 3, "not a number")
    assert tab.collect_heater_overrides() == {
        10: {"heater_max_power_W": 45.0},
        11: {"heater_slew_rate_W_per_s": 2.5},
    }


def test_one_heater_can_override_every_field(tmp_path, monkeypatch) -> None:
    tab = _tab(tmp_path, monkeypatch, [(10, "A")])
    for column, text in ((1, "45"), (2, "2.5"), (3, "0.9")):
        _fill(tab, 0, column, text)
    assert tab.collect_heater_overrides() == {
        10: {
            "heater_max_power_W": 45.0,
            "heater_slew_rate_W_per_s": 2.5,
            "heater_efficiency": 0.9,
        }
    }


def test_clear_puts_every_heater_back_on_the_defaults(tmp_path, monkeypatch) -> None:
    tab = _tab(tmp_path, monkeypatch, [(10, "A")])
    _fill(tab, 0, 1, "45")
    tab.clear_heater_overrides()
    assert tab.collect_heater_overrides() == {}


# --- what the launch sends ------------------------------------------------------ #
def _launched(tab, monkeypatch, tmp_path) -> list[str]:
    """Run start_run with the subprocess stubbed, returning the command it built.

    Run directories are built relative to the CWD, so this runs from tmp_path --
    otherwise every launch writes into the repo's own simulations/ folder, and two
    launches in the same second share a directory and each other's files.
    """
    captured: dict = {}

    class _Process:
        pid = 1234

        def poll(self):
            return None

    def _popen(command, **_kwargs):
        captured["command"] = list(command)
        return _Process()

    monkeypatch.setattr("graph_visualizer.headless_run_tab.subprocess.Popen", _popen)
    monkeypatch.setattr(HeadlessRunTab, "_confirm_controller_ok", lambda self, artifact: True)
    monkeypatch.chdir(tmp_path)
    tab.start_run()
    return captured.get("command", [])


def test_the_run_is_given_the_overrides_file(tmp_path, monkeypatch) -> None:
    tab = _tab(tmp_path, monkeypatch, [(10, "A"), (11, "B")])
    _fill(tab, 0, 1, "45")
    _fill(tab, 0, 2, "2.5")
    command = _launched(tab, monkeypatch, tmp_path)

    assert "--heater-overrides-json" in command
    payload = json.loads(Path(command[command.index("--heater-overrides-json") + 1]).read_text())
    assert payload == {"10": {"heater_max_power_W": 45.0, "heater_slew_rate_W_per_s": 2.5}}


def test_no_overrides_means_no_file_and_no_flag(tmp_path, monkeypatch) -> None:
    tab = _tab(tmp_path, monkeypatch, [(10, "A")])
    command = _launched(tab, monkeypatch, tmp_path)
    assert "--heater-overrides-json" not in command
    assert not list(Path(tab.run_dir).glob("heater_overrides.json"))


def test_the_setpoint_table_is_the_only_source_of_targets(tmp_path, monkeypatch) -> None:
    """The global --setpoint is gone: every sensor's target comes from the table,
    which is prefilled, so the run is still given targets without any editing."""
    tab = _tab(tmp_path, monkeypatch, [(10, "A")], sensors=[20, 21])
    command = _launched(tab, monkeypatch, tmp_path)

    assert "--setpoint" not in command
    payload = json.loads(Path(command[command.index("--setpoints-json") + 1]).read_text())
    assert payload == {"20": 293.15, "21": 293.15}


# --- the CLI -------------------------------------------------------------------- #
def test_cli_reads_the_overrides_file(tmp_path) -> None:
    payload = tmp_path / "heaters.json"
    payload.write_text(json.dumps({"10": {"heater_max_power_W": 45.0}}), encoding="utf-8")
    script = Path(__file__).resolve().parent.parent / "run_simulation.py"
    result = subprocess.run(
        [sys.executable, str(script), "--graph", str(tmp_path / "nope"),
         "--heater-overrides-json", str(payload)],
        capture_output=True, text=True, cwd=str(tmp_path),
    )
    # The graph does not exist, so it must fail AFTER parsing the overrides file.
    assert "Per-heater overrides: 1 heater(s)" in result.stdout, result.stdout + result.stderr


def test_cli_rejects_a_malformed_overrides_file(tmp_path) -> None:
    payload = tmp_path / "bad.json"
    payload.write_text(json.dumps({"10": 45.0}), encoding="utf-8")  # not a field map
    script = Path(__file__).resolve().parent.parent / "run_simulation.py"
    result = subprocess.run(
        [sys.executable, str(script), "--graph", str(tmp_path),
         "--heater-overrides-json", str(payload)],
        capture_output=True, text=True, cwd=str(tmp_path),
    )
    assert result.returncode != 0
    assert "bad entry" in (result.stdout + result.stderr)


# --- what the run does with them ------------------------------------------------ #
def _runner(overrides):
    from graph_visualizer.simulation_runner import RunConfig, SimulationRunner

    runner = object.__new__(SimulationRunner)
    runner.cfg = RunConfig(graph_folder="unused", heater_overrides=overrides)
    runner._logged = []
    runner._log_event = lambda kind, message: runner._logged.append((kind, message))
    return runner


def _heater_node():
    from graph_visualizer.models import HeaterProperties, NodeProperties

    node = NodeProperties(node_id=10, coord=(0, 0, 0))
    node.is_heater = True
    node.heater = HeaterProperties(heater_id=10)
    return node


def test_the_run_writes_the_overrides_onto_the_heater(tmp_path) -> None:
    node = _heater_node()
    model = type("M", (), {"nodes": {10: node}})()
    _runner({10: {"heater_max_power_W": 45.0, "heater_slew_rate_W_per_s": 2.5}})._apply_heater_overrides(model)
    assert node.heater.heater_max_power_W == pytest.approx(45.0)
    assert node.heater.heater_slew_rate_W_per_s == pytest.approx(2.5)
    assert node.heater.heater_efficiency == pytest.approx(1.0), "unnamed fields keep the default"


def test_manual_power_is_written_on_the_node_and_forces_open_loop(tmp_path) -> None:
    """The open-loop step knob. sensor_manual_power_W lives on the heater NODE, not
    on its .heater limits record, and the manual command path is skipped entirely
    for a heater tagged "mimo" -- so setting the wattage without switching the mode
    would be accepted, logged as applied, and then silently ignored all run."""
    node = _heater_node()
    node.sensor_control_mode = "mimo"
    model = type("M", (), {"nodes": {10: node}})()
    runner = _runner({10: {"sensor_manual_power_W": 10.0}})
    runner._apply_heater_overrides(model)
    assert node.sensor_manual_power_W == pytest.approx(10.0)
    assert node.sensor_control_mode == "manual"
    assert any("manual" in message for _kind, message in runner._logged)


def test_manual_power_also_clears_the_legacy_mimo_tag(tmp_path) -> None:
    """_heater_controller_mode reads heater_control.mode as well, and "mimo" there
    wins over the node's own mode."""
    from graph_visualizer.simulation_model import _heater_controller_mode

    node = _heater_node()
    node.heater_control.mode = "mimo"
    model = type("M", (), {"nodes": {10: node}})()
    _runner({10: {"sensor_manual_power_W": 7.5}})._apply_heater_overrides(model)
    assert _heater_controller_mode(node) == "manual", "the heater would take no manual power"


def test_an_override_for_a_node_that_is_not_a_heater_is_reported(tmp_path) -> None:
    """Applying nothing and saying nothing would run all night on default limits
    while the user believes a heater was capped."""
    model = type("M", (), {"nodes": {}})()
    runner = _runner({999: {"heater_max_power_W": 45.0}})
    runner._apply_heater_overrides(model)
    assert any("WARNING" in message and "999" in message for _kind, message in runner._logged)


# --- and what the controller does with the slew ---------------------------------- #
def _sim_with_slew(monkeypatch, per_heater: dict[int, float], global_rate: float):
    from test_mimo_pi_controller import _pi_sim
    from graph_visualizer.models import HeaterProperties

    sim = _pi_sim(monkeypatch, enabled_heater_node_ids=None)
    sim.params = replace(sim.params, mimo_heater_slew_rate_W_per_s=global_rate)
    for heater_id, rate in per_heater.items():
        sim.model.nodes[heater_id].heater = HeaterProperties(
            heater_id=heater_id, heater_max_power_W=10.0, heater_slew_rate_W_per_s=rate
        )
    return sim


def test_a_heater_keeps_its_own_slew_limit_while_the_rest_use_the_global(monkeypatch) -> None:
    """The limit models a DRIVER, and heaters on different drivers ramp differently,
    so it has to be per heater rather than one number for the whole run."""
    from graph_visualizer.simulation_model import _controller_slew_limits

    sim = _sim_with_slew(monkeypatch, {10: 2.5}, global_rate=30.0)
    limits = _controller_slew_limits(sim.model, [10, 11], sim.params)
    assert limits[0] == pytest.approx(2.5), "its own rate wins"
    assert limits[1] == pytest.approx(30.0), "the rest fall back to the global"


def test_zero_still_means_use_the_global(monkeypatch) -> None:
    from graph_visualizer.simulation_model import _controller_slew_limits

    sim = _sim_with_slew(monkeypatch, {10: 0.0}, global_rate=30.0)
    assert _controller_slew_limits(sim.model, [10], sim.params)[0] == pytest.approx(30.0)


def test_the_command_actually_respects_a_per_heater_limit(monkeypatch) -> None:
    """The bound has to reach the allocator, not just the diagnostics."""
    sim = _sim_with_slew(monkeypatch, {10: 0.5}, global_rate=1000.0)
    sim.controller_last_power_by_heater = {10: 0.0, 11: 0.0}
    sim._mimo_pi_controller_power_vector(update_state=True)
    commands = dict(zip(
        sim.controller_allocator_diagnostics["heater_ids"],
        sim.controller_allocator_diagnostics["heater_commands_W"],
    ))
    dt = float(sim.params.dt_s)
    assert commands[10] <= 0.5 * dt + 1.0e-9, "capped heater must not step further than its rate"
    assert sim.controller_allocator_diagnostics["slew_rate_limit_W_per_s"][0] == pytest.approx(0.5)


def test_no_limit_anywhere_leaves_the_command_unbounded(monkeypatch) -> None:
    sim = _sim_with_slew(monkeypatch, {}, global_rate=0.0)
    sim.controller_last_power_by_heater = {10: 0.0, 11: 0.0}
    sim._mimo_pi_controller_power_vector(update_state=True)
    assert np.all(
        np.asarray(sim.controller_allocator_diagnostics["slew_rate_limit_W_per_s"]) == 0.0
    )


# --- every heater runs ----------------------------------------------------------- #
def test_a_headless_run_drives_every_heater(tmp_path, monkeypatch) -> None:
    """The enabled-I/O table is a live-tab control and its section is hidden here, so
    a heater unticked there would ride into an overnight run through the graph's
    saved parameters with nothing in this tab to show for it."""
    from graph_visualizer.simulation_parameters import SimulationParameters, save_simulation_parameters

    tab = _tab(tmp_path, monkeypatch, [(10, "A"), (11, "B")])
    save_simulation_parameters(
        tmp_path / "graphs" / "g" / "simulation_parameters.json",
        SimulationParameters(enabled_heater_node_ids=(11,)),
    )
    assert tab._collect_parameters().enabled_heater_node_ids is None


def test_the_graph_line_says_how_many_roles_were_found(tmp_path, monkeypatch) -> None:
    """When both tables come up empty, this line is the difference between "this
    graph declares no roles" and "its nodes.csv could not be read"."""
    tab = _tab(tmp_path, monkeypatch, [(10, "A")], sensors=[20])
    assert "1 heater(s), 1 sensor(s)" in tab.graph_info.text()


def test_a_graph_with_no_role_columns_says_so_rather_than_looking_empty(tmp_path, monkeypatch) -> None:
    import test_simulation_controls_panel as panel_stubs

    folder = tmp_path / "graphs" / "g"
    folder.mkdir(parents=True)
    (folder / "node_ids.npy").write_bytes(b"")
    (folder / "nodes.csv").write_text("node_id,component_name\n1,A\n", encoding="utf-8")
    monkeypatch.setattr(HeadlessRunTab, "simulations_root", lambda self: tmp_path / "simulations")
    tab = HeadlessRunTab(panel_stubs._QtStub, None, graphs_root=lambda: tmp_path / "graphs")
    assert "is_heater" in tab.graph_info.text(), tab.graph_info.text()


def test_a_failing_loader_does_not_blank_the_rest_of_the_tab(tmp_path, monkeypatch) -> None:
    """The three tables refresh together; one bad file used to abort the refresh
    before the graph info line was written, leaving stale text beside empty tables."""
    said: list[tuple[str, bool]] = []
    tab = _tab(tmp_path, monkeypatch, [(10, "A")], sensors=[20])
    monkeypatch.setattr(
        HeadlessRunTab, "load_sensor_setpoints",
        lambda self, announce=False: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    tab.on_status = lambda message, is_error: said.append((message, is_error))
    tab._populate_from_graph()

    assert any("boom" in message for message, _ in said), said
    assert tab.heater_table.rowCount() == 1, "the heater table still loaded"
    assert "roles:" in tab.graph_info.text(), "the info line was still written"


# --- the global figure is a CEILING, not a fallback --------------------------- #
class _Heater:
    def __init__(self, max_power_W=0.0, efficiency=1.0):
        self.heater_max_power_W = max_power_W
        self.heater_efficiency = efficiency


class _HeaterNode:
    def __init__(self, **kw):
        self.heater = _Heater(**kw)


def _limit(node, ceiling):
    from dataclasses import replace as _replace

    from graph_visualizer.simulation_model import _controller_heater_max_power
    from graph_visualizer.simulation_parameters import SimulationParameters

    params = _replace(SimulationParameters(), mimo_default_heater_max_power_W=ceiling)
    return _controller_heater_max_power(node, params)


def test_a_graph_rated_heater_is_clamped_to_the_global_ceiling() -> None:
    """The reported bug: the field was a FALLBACK, so a node carrying a build-time
    rating ignored it entirely. A run configured for 1.5 W commanded 12.4 W into one
    heater -- 8x -- and no artifact said so."""
    assert _limit(_HeaterNode(max_power_W=12.5), 1.5) == 1.5


def test_a_heater_rated_below_the_ceiling_keeps_its_own_rating() -> None:
    """The ceiling stops the allocator concentrating load; it must not RAISE a
    heater past what the hardware can take."""
    assert _limit(_HeaterNode(max_power_W=0.8), 1.5) == 0.8


def test_an_unrated_heater_takes_the_ceiling() -> None:
    assert _limit(_HeaterNode(), 1.5) == 1.5


def test_an_explicit_override_may_exceed_the_ceiling() -> None:
    """A deliberate 120 W has to keep working under a 1.5 W global, which is the
    whole reason the override needs marking: it is applied by mutating the node, so
    without the mark it looks exactly like a graph rating."""
    from graph_visualizer.simulation_model import EXPLICIT_MAX_POWER_ATTR

    node = _HeaterNode(max_power_W=120.0)
    setattr(node.heater, EXPLICIT_MAX_POWER_ATTR, True)
    assert _limit(node, 1.5) == 120.0


def test_efficiency_still_scales_the_limit() -> None:
    assert _limit(_HeaterNode(max_power_W=4.0, efficiency=0.5), 10.0) == 2.0


def test_a_zero_ceiling_means_no_ceiling() -> None:
    """Runs that never set the field must behave as before rather than having every
    rated heater silently clamped to zero."""
    assert _limit(_HeaterNode(max_power_W=12.5), 0.0) == 12.5
    assert _limit(_HeaterNode(), 0.0) == 0.0


def test_applying_an_override_marks_it_as_explicit(tmp_path) -> None:
    """End to end through the runner's override path, since that is what turns a
    table entry into the mark the limit rule reads."""
    from graph_visualizer import simulation_model
    from graph_visualizer.simulation_runner import SimulationRunner

    r = object.__new__(SimulationRunner)
    r.cfg = type("Cfg", (), {"heater_overrides": {7: {"heater_max_power_W": 120.0}}})()
    r.events = []
    r._log_event = lambda kind, msg: r.events.append((kind, msg))
    node = _HeaterNode(max_power_W=2.0)
    r._apply_heater_overrides(type("M", (), {"nodes": {7: node}})())
    assert node.heater.heater_max_power_W == 120.0
    assert getattr(node.heater, simulation_model.EXPLICIT_MAX_POWER_ATTR, False) is True
    assert _limit(node, 1.5) == 120.0, "an override must survive the ceiling"
