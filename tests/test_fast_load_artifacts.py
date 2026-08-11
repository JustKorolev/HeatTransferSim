"""Editing a large octree graph must keep the low-memory (fast) load valid.

The lightweight save path (used for large graphs) rewrites graph.json but, for a
graph above the dense-matrix threshold, used to skip nodes.csv. That left
nodes.csv older than graph.json, which disqualifies fast_graph_io's low-memory
loader (can_load_fast) and forces every later run through the multi-minute,
tens-of-GB graph.json parse. The fix rewrites nodes.csv in that branch.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from graph_visualizer import graph_io
from graph_visualizer.fast_graph_io import (
    can_load_fast,
    fast_load_has_roles,
    launch_refresh_subprocess,
    load_graph_for_simulation,
)
from graph_visualizer.graph_io import load_graph_folder, save_graph_folder
from graph_visualizer.models import GraphMetadata, NodeProperties, ThermalGraphModel


def _two_node_octree_model(name: str, left_initial_K: float = 310.0) -> ThermalGraphModel:
    model = ThermalGraphModel(metadata=GraphMetadata(graph_name=name))
    left = NodeProperties.with_material(1, (0, 0, 0), material="copper")
    right = NodeProperties.with_material(2, (1, 0, 0), material="copper")
    left.material = right.material = "Copper"
    left.center_mm = (0.0, 0.0, 0.0)
    right.center_mm = (1.0, 0.0, 0.0)
    left.size_mm = right.size_mm = (1.0, 1.0, 1.0)
    left.C_J_K = right.C_J_K = 10.0
    left.initial_temperature_K = left_initial_K
    right.initial_temperature_K = 290.0
    model.add_node(left)
    model.add_node(right)
    # Non-empty octree data routes save through the lightweight path.
    model.octree_graph_data = {"graph_edges": []}
    return model


def _read_initial_temp(folder: Path, node_id: int) -> float:
    with (folder / "nodes.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["node_id"]) == node_id:
                return float(row["initial_temperature_K"])
    raise AssertionError(f"node {node_id} missing from nodes.csv")


def test_lightweight_save_of_large_graph_keeps_fast_load_valid(monkeypatch) -> None:
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        # First save builds the full artifact set (small-graph branch).
        save_graph_folder(_two_node_octree_model("fast_load"), folder)
        assert (folder / "nodes.csv").exists()
        # node_ids.npy is written by the octree BUILDER, not the save path; it is
        # present in a real graph folder, so create it to mirror production.
        np.save(folder / "node_ids.npy", np.array([1, 2], dtype=int))

        # Simulate an edit made well after the build: graph.json is about to be
        # rewritten, and nodes.csv currently predates it.
        old = 1_000_000.0
        os.utime(folder / "graph.json", (old, old))
        os.utime(folder / "nodes.csv", (old, old))

        # Force the >threshold branch that used to skip nodes.csv.
        monkeypatch.setattr(graph_io, "_DENSE_OCTREE_MATRIX_NODE_LIMIT", 1)
        edited = _two_node_octree_model("fast_load", left_initial_K=333.0)
        save_graph_folder(edited, folder)

        usable, reason = can_load_fast(folder)
        assert usable, reason
        # nodes.csv was regenerated from the edited model, not left stale.
        assert _read_initial_temp(folder, 1) == 333.0
        graph_mtime = (folder / "graph.json").stat().st_mtime
        nodes_mtime = (folder / "nodes.csv").stat().st_mtime
        assert nodes_mtime >= graph_mtime - 1.0


def _cell(node_id: int, x: float, initial_K: float = 50.0) -> NodeProperties:
    node = NodeProperties.with_material(node_id, (node_id, 0, 0), material="copper")
    node.material = "Copper"
    node.center_mm = (x, 0.0, 0.0)
    node.size_mm = (1.0, 1.0, 1.0)
    node.C_J_K = 10.0
    node.initial_temperature_K = initial_K
    return node


def _model_with_roles() -> ThermalGraphModel:
    model = ThermalGraphModel(metadata=GraphMetadata(graph_name="roles"))
    body = _cell(1, 0.0)
    heater = _cell(2, 1.0)
    heater.is_heater = True
    heater.heater.heater_id = 2
    heater.heater.heater_max_power_W = 25.0
    heater.heater.heater_efficiency = 0.9
    heater.assigned_sensor_id = 3
    heater.sensor_control_mode = "mimo"
    heater.controller_setpoint_K = 55.0
    heater.controller_kp_coarse = 2.0
    heater.power_deposition_node_ids = [1]
    heater.power_deposition_weights = [1.0]
    sensor = _cell(3, 2.0)
    sensor.is_sensor = True
    sensor.sensor.sensor_id = 3
    sensor.controller_setpoint_K = 55.0
    sensor.assigned_heater_ids = [2]
    sensor.readout_node_ids = [1]
    sensor.readout_weights = [1.0]
    cryo = _cell(4, 3.0)
    cryo.has_cryocooler = True
    cryo.cryocooler_id = "cc1"
    cryo.cryocooler_enabled = True
    cryo.cryocooler_receiving_node_ids = [1]
    cryo.cryocooler_contact_areas_m2 = [0.01]
    for node in (body, heater, sensor, cryo):
        model.add_node(node)
    model.octree_graph_data = {"graph_edges": []}
    return model


def test_fast_load_preserves_roles_matching_the_full_loader(tmp_path) -> None:
    folder = tmp_path / "graph"
    folder.mkdir()
    save_graph_folder(_model_with_roles(), folder)
    np.save(folder / "node_ids.npy", np.array([1, 2, 3, 4], dtype=int))

    assert fast_load_has_roles(folder) is True

    full_model, _full = load_graph_folder(folder)
    fast_model, _fast, _report = load_graph_for_simulation(folder)

    # The fix's contract: the fast (nodes.csv) load reconstructs roles IDENTICALLY
    # to the full graph.json load -- no silently-dropped heaters/sensors/coolers.
    role_attrs = (
        "is_heater",
        "is_sensor",
        "has_cryocooler",
        "controller_setpoint_K",
        "sensor_control_mode",
        "assigned_sensor_id",
        "assigned_heater_ids",
        "cryocooler_receiving_node_ids",
        "cryocooler_contact_areas_m2",
        "power_deposition_node_ids",
        "readout_node_ids",
    )
    for node_id in (1, 2, 3, 4):
        fast, full = fast_model.nodes[node_id], full_model.nodes[node_id]
        for attr in role_attrs:
            assert getattr(fast, attr) == getattr(full, attr), (node_id, attr)
    assert fast_model.nodes[2].heater.heater_max_power_W == full_model.nodes[2].heater.heater_max_power_W

    # And the roles are actually present (the bug was sensors=0 heaters=0 cryo=0).
    assert [n for n in fast_model.nodes.values() if n.is_heater]
    assert [n for n in fast_model.nodes.values() if n.is_sensor]
    assert [n for n in fast_model.nodes.values() if n.has_cryocooler]


def test_fast_load_has_roles_false_for_legacy_nodes_csv(tmp_path) -> None:
    folder = tmp_path / "graph"
    folder.mkdir()
    (folder / "nodes.csv").write_text("node_id,C_J_K\n1,10\n", encoding="utf-8")
    assert fast_load_has_roles(folder) is False


def test_refresh_subprocess_restores_a_stale_fast_load(tmp_path) -> None:
    """End-to-end: the 'Update graph' path (refresh_fast_load.py via
    launch_refresh_subprocess) regenerates nodes.csv from graph.json without the
    GUI, turning an unusable fast-load folder back into a usable one."""
    folder = tmp_path / "graph"
    folder.mkdir()
    save_graph_folder(_two_node_octree_model("refresh_subproc"), folder)
    np.save(folder / "node_ids.npy", np.array([1, 2], dtype=int))
    old = 1_000_000.0
    os.utime(folder / "nodes.csv", (old, old))
    assert can_load_fast(folder)[0] is False

    proc = launch_refresh_subprocess(folder)
    assert proc.wait(timeout=120) == 0, (folder / "refresh_fast_load.log").read_text(encoding="utf-8")

    usable, reason = can_load_fast(folder)
    assert usable, reason


def test_a_graph_carrying_the_retired_pid_gains_still_loads() -> None:
    """Every existing graph.json has the per-heater PID gains in it.

    They were removed with the PID+QP allocator (nothing read them: heater control
    modes are only "mimo" or "manual"), so from_dict has to drop them rather than
    choke. A graph that will not open is a far worse outcome than a dead knob.
    """
    from graph_visualizer.models import NodeProperties

    node = NodeProperties.from_dict(
        {
            "node_id": 1,
            "coord": [0, 0, 0],
            "controller": {
                "setpoint_K": 312.0,
                "weight": 2.0,
                "kp_coarse": 0.75,
                "ki_coarse": 0.25,
                "kd_coarse": 0.4,
                "kp_hold": 0.5,
                "ki_hold": 0.125,
                "kd_hold": 0.2,
                "kp_scale": 0.9,   # the even older spelling
                "ki_scale": 0.3,
            },
            "controller_kp_coarse": 1.5,   # and the flattened form
        }
    )
    assert node.controller_setpoint_K == 312.0, "the fields that still matter survive"
    assert node.controller_weight == 2.0
    for dead in ("controller_kp_coarse", "controller_ki_hold", "controller_kd_coarse"):
        assert not hasattr(node, dead), dead
    # And they must not come back on the way out.
    assert "kp_coarse" not in node.to_octree_node_dict()["controller"]
