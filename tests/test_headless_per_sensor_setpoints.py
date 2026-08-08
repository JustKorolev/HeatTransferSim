"""Per-sensor setpoints from the headless tab.

The simulation tab edits setpoints through its readout tables, which need a loaded
graph. The headless tab deliberately never loads one, so it took a single global
--setpoint for every sensor. RunConfig always had setpoints_K (node id -> target),
it just had no route from the CLI or the UI.

The sensor list comes from the manifest a previous run wrote (sensors.csv), so the
graph still never enters the GUI process.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import types
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


def _tab(tmp_path: Path, monkeypatch) -> HeadlessRunTab:
    tab = HeadlessRunTab.__new__(HeadlessRunTab)
    monkeypatch.setattr(
        HeadlessRunTab, "simulations_root", lambda self: tmp_path / "simulations"
    )
    return tab


def test_sensor_manifest_is_read_from_the_newest_run(tmp_path, monkeypatch) -> None:
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
    """Blank means 'use the global setpoint', so an untouched table sends nothing."""
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
        capture_output=True, text=True,
    )
    # The graph does not exist, so it must fail AFTER parsing the setpoints file.
    assert "Per-sensor setpoints: 2 sensor(s)" in result.stdout, result.stdout + result.stderr


def test_cli_rejects_a_malformed_setpoints_file(tmp_path) -> None:
    payload = tmp_path / "bad.json"
    payload.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    script = Path(__file__).resolve().parent.parent / "run_simulation.py"
    result = subprocess.run(
        [sys.executable, str(script), "--graph", str(tmp_path), "--setpoints-json", str(payload)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "must contain a JSON object" in (result.stdout + result.stderr)
