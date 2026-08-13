"""The graph's heater/sensor list, read from the build's nodes.csv.

The headless tab took its sensor list from the newest run's sensors.csv, so a graph
that had never been run listed no sensors -- and with the global setpoint row gone,
no sensors means a run with no targets at all. nodes.csv already knows: it carries
is_heater / is_sensor for every node, and the controller artifacts are derived from
exactly those nodes.
"""

from __future__ import annotations

import csv
import json

from graph_visualizer.graph_roles import (
    ROLE_CACHE_FILENAME,
    load_role_manifest,
    scan_nodes_csv,
)


def _nodes_csv(folder, rows) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "nodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "node_id", "component_name", "is_heater", "is_sensor", "sensor_monitor_only",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(node_id, name="", heater=False, sensor=False, monitor=False):
    return {
        "node_id": node_id, "component_name": name,
        "is_heater": str(heater), "is_sensor": str(sensor),
        "sensor_monitor_only": str(monitor),
    }


def test_heaters_and_sensors_are_separated_and_sorted(tmp_path):
    _nodes_csv(tmp_path, [
        _row(30, "s2", sensor=True, monitor=True),
        _row(10, "cell"),
        _row(20, "h1", heater=True),
        _row(5, "s1", sensor=True),
    ])
    manifest = scan_nodes_csv(tmp_path / "nodes.csv")
    assert manifest.heater_ids == [20]
    assert manifest.sensor_ids == [5, 30]
    assert [row.monitor_only for row in manifest.sensors] == [False, True]
    assert manifest.sensors[0].component_name == "s1"


def test_a_node_that_is_both_appears_in_both_lists(tmp_path):
    """The role columns are independent flags, not one category."""
    _nodes_csv(tmp_path, [_row(7, "both", heater=True, sensor=True)])
    manifest = scan_nodes_csv(tmp_path / "nodes.csv")
    assert manifest.heater_ids == [7] and manifest.sensor_ids == [7]


def test_the_scan_is_cached_beside_the_graph(tmp_path):
    """A 471k-row nodes.csv takes ~1.7 s to scan. Paying that on every graph switch
    would stall the window for data that only changes when the graph is rebuilt."""
    _nodes_csv(tmp_path, [_row(20, "h1", heater=True)])
    assert load_role_manifest(tmp_path).source == "nodes.csv"
    assert (tmp_path / ROLE_CACHE_FILENAME).is_file()
    assert load_role_manifest(tmp_path).source == "cache"
    assert load_role_manifest(tmp_path).heater_ids == [20]


def test_a_rebuilt_graph_invalidates_the_cache(tmp_path):
    """Serving a stale list would silently target heaters the new build dropped."""
    _nodes_csv(tmp_path, [_row(20, "h1", heater=True)])
    load_role_manifest(tmp_path)
    _nodes_csv(tmp_path, [_row(21, "h1", heater=True), _row(22, "h2", heater=True)])
    manifest = load_role_manifest(tmp_path)
    assert manifest.source == "nodes.csv", "must rescan, not serve the cache"
    assert manifest.heater_ids == [21, 22]


def test_a_corrupt_cache_is_rescanned_not_fatal(tmp_path):
    _nodes_csv(tmp_path, [_row(20, "h1", heater=True)])
    (tmp_path / ROLE_CACHE_FILENAME).write_text("{not json", encoding="utf-8")
    assert load_role_manifest(tmp_path).heater_ids == [20]


def test_a_graph_without_nodes_csv_reports_why(tmp_path):
    manifest = load_role_manifest(tmp_path)
    assert manifest.heaters == [] and manifest.sensors == []
    assert "nodes.csv" in manifest.source


def test_unreadable_rows_are_skipped_not_fatal(tmp_path):
    """One malformed row must not cost the whole list."""
    _nodes_csv(tmp_path, [_row("", "no id", heater=True), _row(20, "h1", heater=True)])
    assert load_role_manifest(tmp_path).heater_ids == [20]


def test_the_cache_round_trips_component_names_and_monitor_flags(tmp_path):
    _nodes_csv(tmp_path, [_row(5, "COO_A", sensor=True, monitor=True)])
    load_role_manifest(tmp_path)
    payload = json.loads((tmp_path / ROLE_CACHE_FILENAME).read_text(encoding="utf-8"))
    assert payload["sensors"][0]["component_name"] == "COO_A"
    cached = load_role_manifest(tmp_path)
    assert cached.sensors[0].monitor_only is True
    assert cached.sensors[0].component_name == "COO_A"


def test_a_build_too_old_to_record_roles_says_so(tmp_path):
    """Reporting "no heaters" for a graph whose nodes.csv simply predates the role
    columns would be wrong -- the caller has to know to fall back to a run manifest."""
    (tmp_path / "nodes.csv").write_text("node_id,component_name\n1,A\n", encoding="utf-8")
    manifest = load_role_manifest(tmp_path)
    assert manifest.heaters == [] and manifest.sensors == []
    assert "is_heater" in manifest.source


def test_a_component_name_with_a_comma_survives(tmp_path):
    """Quoting is why this parses CSV properly instead of splitting on commas."""
    (tmp_path / "nodes.csv").write_text(
        'node_id,component_name,is_heater,is_sensor,sensor_monitor_only\n'
        '7,"HTR, upper",True,False,False\n',
        encoding="utf-8",
    )
    assert load_role_manifest(tmp_path).heaters[0].component_name == "HTR, upper"


def test_an_empty_file_is_not_fatal(tmp_path):
    (tmp_path / "nodes.csv").write_text("", encoding="utf-8")
    assert load_role_manifest(tmp_path).heaters == []
