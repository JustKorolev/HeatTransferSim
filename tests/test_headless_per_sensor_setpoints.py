"""Per-sensor setpoints from the headless tab.

The simulation tab edits setpoints through its readout tables, which need a loaded
graph. The headless tab deliberately never loads one, so it took a single global
--setpoint for every sensor. RunConfig always had setpoints_K (node id -> target),
it just had no route from the CLI or the UI.

The global setpoint is gone: this table is the run's only source of targets. The
sensor list therefore comes from the BUILD (nodes.csv's is_sensor column, via
graph_roles) rather than from a previous run's sensors.csv, which a graph that has
never been run does not have. sensors.csv remains the fallback. Either way the
graph itself never enters the GUI process.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import types

import pytest
from pathlib import Path

for _name in ("PySide6", "PySide6.QtCore", "PySide6.QtWidgets", "PySide6.QtGui"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

from graph_visualizer.headless_run_tab import HeadlessRunTab  # noqa: E402


def _manifest(root: Path, graph: str, run: str, sensors: list[tuple[int, str, bool]]) -> Path:
    run_dir = root / "simulations" / graph / run
    run_dir.mkdir(parents=True)
    with (run_dir / "sensors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["series", "node_id", "component_name", "setpoint_K", "monitor_only"]
        )
        writer.writeheader()
        for index, (node_id, name, monitor) in enumerate(sensors):
            writer.writerow({
                "series": f"sensor_{index}", "node_id": node_id, "component_name": name,
                "setpoint_K": "50.1662", "monitor_only": str(monitor),
            })
    return run_dir


def _nodes_csv(tmp_path: Path, graph: str, roles: list[tuple[int, str, bool, bool]]) -> Path:
    """A minimal build-style nodes.csv: (node_id, component, is_heater, is_sensor)."""
    folder = tmp_path / "graphs" / graph
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "nodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["node_id", "component_name", "is_heater", "is_sensor", "sensor_monitor_only"],
        )
        writer.writeheader()
        for node_id, name, is_heater, is_sensor in roles:
            writer.writerow({
                "node_id": node_id, "component_name": name,
                "is_heater": str(is_heater), "is_sensor": str(is_sensor),
                "sensor_monitor_only": "False",
            })
    return folder


def _tab(tmp_path: Path, monkeypatch) -> HeadlessRunTab:
    tab = HeadlessRunTab.__new__(HeadlessRunTab)
    tab._graphs_root = lambda: tmp_path / "graphs"
    monkeypatch.setattr(
        HeadlessRunTab, "simulations_root", lambda self: tmp_path / "simulations"
    )
    return tab


def test_the_sensor_list_comes_from_the_build_not_a_previous_run(tmp_path, monkeypatch) -> None:
    """A graph that has never been run still has to name its sensors -- with no
    global setpoint left, an empty table means a run with no targets at all."""
    _nodes_csv(tmp_path, "g", [(10, "a", False, True), (20, "h", True, False), (11, "b", False, True)])
    rows = _tab(tmp_path, monkeypatch).sensor_manifest("g")
    assert [r["node_id"] for r in rows] == ["10", "11"], "sensors only, sorted"


def test_the_build_list_wins_over_an_older_run_manifest(tmp_path, monkeypatch) -> None:
    _nodes_csv(tmp_path, "g", [(10, "a", False, True)])
    _manifest(tmp_path, "g", "20260807-183010", [(999, "stale", False)])
    rows = _tab(tmp_path, monkeypatch).sensor_manifest("g")
    assert [r["node_id"] for r in rows] == ["10"]


def test_sensor_manifest_falls_back_to_the_newest_run(tmp_path, monkeypatch) -> None:
    """Covers a graph whose nodes.csv predates the role columns."""
    _manifest(tmp_path, "g", "20260101-000000", [(1, "old", False)])
    _manifest(tmp_path, "g", "20260807-183010", [(10, "a", False), (11, "b", True)])
    rows = _tab(tmp_path, monkeypatch).sensor_manifest("g")
    assert [r["node_id"] for r in rows] == ["10", "11"]


def test_runs_without_a_manifest_are_skipped(tmp_path, monkeypatch) -> None:
    _manifest(tmp_path, "g", "20260101-000000", [(1, "kept", False)])
    (tmp_path / "simulations" / "g" / "20260909-999999").mkdir(parents=True)
    rows = _tab(tmp_path, monkeypatch).sensor_manifest("g")
    assert [r["node_id"] for r in rows] == ["1"]
    assert _tab(tmp_path, monkeypatch).sensor_manifest("missing") == []


def test_only_filled_rows_become_overrides(tmp_path, monkeypatch) -> None:
    """A row cleared by hand keeps whatever the graph holds, so it sends nothing."""
    tab = _tab(tmp_path, monkeypatch)
    tab._sensor_rows_manifest = [{"node_id": "10"}, {"node_id": "11"}, {"node_id": "12"}]

    class _Cell:
        def __init__(self, text): self._text = text
        def text(self): return self._text

    class _Table:
        def __init__(self, values): self._values = values
        def item(self, row, _col): return self._values[row]

    tab.setpoint_table = _Table([_Cell(""), _Cell(" 42.5 "), _Cell("not a number")])
    assert tab.collect_setpoint_overrides() == {11: 42.5}

    tab.setpoint_table = _Table([_Cell(""), _Cell(""), _Cell("")])
    assert tab.collect_setpoint_overrides() == {}


def test_cli_applies_per_sensor_setpoints_over_the_global(tmp_path) -> None:
    """--setpoints-json layers on top of --setpoint rather than replacing it."""
    payload = tmp_path / "setpoints.json"
    payload.write_text(json.dumps({"11": 42.5, "12": 61.0}), encoding="utf-8")
    script = Path(__file__).resolve().parent.parent / "run_simulation.py"
    result = subprocess.run(
        [sys.executable, str(script), "--graph", str(tmp_path / "nope"),
         "--setpoint", "50", "--setpoints-json", str(payload)],
        # From tmp_path: the run builds its output directory relative to the CWD, so
        # without this every invocation leaves a simulations/nope/<timestamp> folder
        # in the repo (there are dozens of them).
        capture_output=True, text=True, cwd=str(tmp_path),
    )
    # The graph does not exist, so it must fail AFTER parsing the setpoints file.
    assert "Per-sensor setpoints: 2 sensor(s)" in result.stdout, result.stdout + result.stderr


def test_cli_rejects_a_malformed_setpoints_file(tmp_path) -> None:
    payload = tmp_path / "bad.json"
    payload.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    script = Path(__file__).resolve().parent.parent / "run_simulation.py"
    result = subprocess.run(
        [sys.executable, str(script), "--graph", str(tmp_path), "--setpoints-json", str(payload)],
        capture_output=True, text=True, cwd=str(tmp_path),
    )
    assert result.returncode != 0
    assert "must contain a JSON object" in (result.stdout + result.stderr)


def _tab_with_sensors(tmp_path, monkeypatch, count: int):
    """A tab whose per-sensor table is populated, built on the panel's Qt stubs."""
    import test_simulation_controls_panel as panel_stubs

    graph = _nodes_csv(
        tmp_path, "g", [(1000 + i, f"COO_{i}", False, True) for i in range(count)]
    )
    (graph / "node_ids.npy").write_bytes(b"")
    monkeypatch.setattr(
        HeadlessRunTab, "simulations_root", lambda self: tmp_path / "simulations"
    )
    tab = HeadlessRunTab(panel_stubs._QtStub, None, graphs_root=lambda: tmp_path / "graphs")
    tab.load_sensor_setpoints()
    return tab


def test_every_row_starts_at_the_default_setpoint(tmp_path, monkeypatch) -> None:
    """With no global setpoint left, an unedited row still has to state a target --
    a blank table would send nothing and the run would track the graph's own values."""
    from graph_visualizer.headless_run_tab import DEFAULT_SETPOINT_K

    tab = _tab_with_sensors(tmp_path, monkeypatch, 5)
    overrides = tab.collect_setpoint_overrides()
    assert len(overrides) == 5
    assert set(overrides.values()) == {DEFAULT_SETPOINT_K}


def test_set_all_uses_its_own_value(tmp_path, monkeypatch) -> None:
    """'Set all' used to read the global setpoint spin, which no longer exists."""
    tab = _tab_with_sensors(tmp_path, monkeypatch, 4)
    tab.setpoint_all_spin.setValue(45.0)
    tab.apply_setpoint_to_all_sensors()
    assert set(tab.collect_setpoint_overrides().values()) == {45.0}


def test_randomize_gives_every_sensor_its_own_target(tmp_path, monkeypatch) -> None:
    """The point of randomizing is the spread BETWEEN sensors; one shared value has
    none. Each row must get an independent draw from centre +/- spread."""
    tab = _tab_with_sensors(tmp_path, monkeypatch, 40)
    tab.sensor_random_center_spin.setValue(50.0)
    tab.sensor_random_spread_mK_spin.setValue(1000.0)  # +/- 1 K
    tab._randomize_setpoints()

    overrides = tab.collect_setpoint_overrides()
    assert len(overrides) == 40, "every sensor must get a value"
    values = list(overrides.values())
    assert len(set(values)) > 1, "values must differ between sensors"
    assert all(49.0 <= v <= 51.0 for v in values), f"outside centre +/- spread: {values}"


def test_zero_spread_makes_every_sensor_identical(tmp_path, monkeypatch) -> None:
    tab = _tab_with_sensors(tmp_path, monkeypatch, 12)
    tab.sensor_random_center_spin.setValue(50.0)
    tab.sensor_random_spread_mK_spin.setValue(0.0)
    tab._randomize_setpoints()
    assert set(tab.collect_setpoint_overrides().values()) == {50.0}


def test_randomize_without_a_sensor_list_says_so(tmp_path, monkeypatch) -> None:
    """There is no global setpoint to fall back to any more, so a randomize with no
    rows must report that rather than appear to have done something."""
    import test_simulation_controls_panel as panel_stubs

    graph = tmp_path / "graphs" / "g"
    graph.mkdir(parents=True)
    (graph / "node_ids.npy").write_bytes(b"")
    monkeypatch.setattr(
        HeadlessRunTab, "simulations_root", lambda self: tmp_path / "simulations"
    )
    said: list[tuple[str, bool]] = []
    tab = HeadlessRunTab(
        panel_stubs._QtStub, None, graphs_root=lambda: tmp_path / "graphs",
        on_status=lambda message, is_error: said.append((message, is_error)),
    )
    tab.sensor_random_center_spin.setValue(50.0)
    tab._randomize_setpoints()
    assert tab.collect_setpoint_overrides() == {}
    assert said and said[-1][1] is True
    assert "No sensors to randomize" in said[-1][0]
