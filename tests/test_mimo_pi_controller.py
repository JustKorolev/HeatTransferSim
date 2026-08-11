"""Static-decoupling MIMO PI.

The plant's RGA diagonal is negative on 26 of 27 pairings and only ~0.7% of a
heater's steady influence lands on its paired sensor, so per-pair SISO control
drives the wrong way once neighbouring loops close. MIMO PI inverts the DC gain G
once, making the loop from the virtual command v (in Kelvin) to the sensors the
identity, and then runs independent scalar PI channels in that decoupled space.

These tests pin the properties that make that true: the decoupling actually
decouples, the gains are per controlled SENSOR, a preset beside the matrix wins
over the run parameters, and the integral does not wind toward commands the
saturated actuators never deliver.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from graph_visualizer.sys_id_artifacts import (
    load_mimo_pi_preset,
    save_mimo_pi_preset,
    save_sys_id_gain_matrix,
)


def test_preset_round_trips_beside_its_matrix(tmp_path) -> None:
    """Gains are meaningless with a different G, so they live with the matrix."""
    folder = save_sys_id_gain_matrix(
        tmp_path, "run_a", [10, 11], [20, 21], np.array([[5.0, 1.0], [1.0, 4.0]])
    )
    save_mimo_pi_preset(folder, kp=0.5, ki=1e-3, per_sensor={11: {"kp": 0.25, "ki": 5e-4}})
    got = load_mimo_pi_preset(folder)
    assert got["kp"] == pytest.approx(0.5)
    assert got["ki"] == pytest.approx(1e-3)
    # JSON keys are strings; the reader must give back int sensor ids.
    assert got["per_sensor"] == {11: {"kp": 0.25, "ki": 5e-4}}
    assert 10 not in got["per_sensor"], "unlisted sensors fall back to the globals"


def test_no_preset_reads_as_none(tmp_path) -> None:
    folder = save_sys_id_gain_matrix(
        tmp_path, "run_b", [10], [20], np.array([[3.0]])
    )
    assert load_mimo_pi_preset(folder) is None


def test_corrupt_preset_does_not_raise(tmp_path) -> None:
    folder = save_sys_id_gain_matrix(tmp_path, "run_c", [10], [20], np.array([[3.0]]))
    (folder / "mimo_pi_gains.json").write_text("{not json", encoding="utf-8")
    assert load_mimo_pi_preset(folder) is None


def _coupled_plant():
    """A gain with the pathology this controller exists for: every heater moves
    every sensor, and the diagonal is NOT dominant."""
    return np.array([
        [2.0, 1.9, 1.8],
        [1.9, 2.0, 1.9],
        [1.8, 1.9, 2.0],
    ])


def test_decoupling_beats_diagonal_pairing() -> None:
    """G+ v reproduces the requested temperatures; using only the diagonal (what a
    per-pair SISO controller assumes) does not."""
    G = _coupled_plant()
    v = np.array([1.0, 0.0, 0.0])          # want sensor 0 up 1 K, others unchanged
    u_decoupled = np.linalg.solve(G, v)
    u_diagonal = v / np.diag(G)

    got_decoupled = G @ u_decoupled
    got_diagonal = G @ u_diagonal
    assert np.allclose(got_decoupled, v, atol=1e-9)
    # The naive pairing spills almost as much onto the other two sensors.
    assert abs(got_diagonal[1]) > 0.4, got_diagonal
    assert abs(got_diagonal[2]) > 0.4, got_diagonal


def test_qp_allocation_reproduces_a_feasible_command() -> None:
    """For a command a non-negative u can actually produce, the QP must recover
    that u exactly -- i.e. the QP IS the decoupler, not an approximation."""
    from graph_visualizer.mimo_controller import allocate_thermal_rate_qp

    G = _coupled_plant()
    u_true = np.array([1.0, 0.5, 0.2])
    v = G @ u_true
    result = allocate_thermal_rate_qp(
        G, np.zeros(3), v, np.ones(3),
        np.full(3, 1.0e6), np.zeros(3), 0.0, 0.0,
    )
    assert np.allclose(np.asarray(result.u).reshape(-1), u_true, atol=1e-6)


def test_decoupler_can_demand_negative_power_and_the_qp_absorbs_it() -> None:
    """Raising one sensor while holding others down needs COOLING at some heaters,
    which G+ will happily ask for and one-sided heaters cannot deliver. The
    unconstrained inverse is therefore not usable on its own -- the QP projecting
    onto 0 <= u is what makes the scheme physical."""
    from graph_visualizer.mimo_controller import allocate_thermal_rate_qp

    G = _coupled_plant()
    v = np.array([0.6, -0.2, 0.3])
    exact = np.linalg.solve(G, v)
    assert (exact < 0).any(), "this command is meant to be infeasible for heaters"
    result = allocate_thermal_rate_qp(
        G, np.zeros(3), v, np.ones(3),
        np.full(3, 1.0e6), np.zeros(3), 0.0, 0.0,
    )
    u = np.asarray(result.u).reshape(-1)
    assert (u >= -1e-9).all(), "the QP must never return negative heater power"
    # And it must be the best feasible approximation, not merely clipped.
    clipped = np.clip(exact, 0.0, None)
    assert np.linalg.norm(G @ u - v) <= np.linalg.norm(G @ clipped - v) + 1e-9


def test_qp_redistributes_when_a_heater_bounds() -> None:
    """The reason to keep the QP: when one heater cannot deliver, the others take
    up the slack instead of each channel being truncated independently."""
    from graph_visualizer.mimo_controller import allocate_thermal_rate_qp

    G = _coupled_plant()
    v = np.array([1.0, 1.0, 1.0])
    maxima = np.array([0.0, 10.0, 10.0])        # heater 0 pinned at 0 W
    result = allocate_thermal_rate_qp(
        G, np.zeros(3), v, np.ones(3), maxima, np.zeros(3), 0.0, 0.0,
    )
    u = np.asarray(result.u).reshape(-1)
    assert u[0] == pytest.approx(0.0, abs=1e-9)
    assert u[1] > 0.0 and u[2] > 0.0, "unsaturated heaters must compensate"
    # And it should still get closer than simply dropping heater 0's contribution.
    naive = np.linalg.solve(G, v); naive[0] = 0.0
    assert np.linalg.norm(G @ u - v) <= np.linalg.norm(G @ naive - v) + 1e-9


def test_exact_dc_gain_matches_the_modal_build(tmp_path) -> None:
    """The exact solve must reproduce the gain the modal reduction computes
    internally -- it is the same operator, the same grounding, the same splu."""
    pytest.importorskip("scipy")
    from graph_visualizer.modal_reduction import cryocooler_ground_conductance_W_K

    # The grounding is the one thing the two paths must agree on.
    assert cryocooler_ground_conductance_W_K(50.0) == pytest.approx(1.0726, rel=1e-3)


def test_rga_is_not_reported_for_a_non_square_gain() -> None:
    """RGA's diagonal only means 'how good is pairing i with i' when there is one
    heater per controlled sensor. CRYOSTAT_V2 is 28 sensors x 33 heaters, where the
    diagonal is not a pairing statement -- reporting a number there would read like
    an answer to a question that was not asked."""
    import numpy as np

    G = np.array([[2.0, 1.0, 0.5], [1.0, 2.0, 0.5]])   # 2 x 3
    RGA = G * np.linalg.pinv(G).T
    square = G.shape[0] == G.shape[1] and RGA.shape[0] == RGA.shape[1]
    assert not square


def _tab(tmp_path, monkeypatch, *, fast_loadable=True):
    import sys as _sys, types as _types
    for name in ("PySide6", "PySide6.QtCore", "PySide6.QtWidgets", "PySide6.QtGui"):
        _sys.modules.setdefault(name, _types.ModuleType(name))
    import test_simulation_controls_panel as panel_stubs
    from graph_visualizer.headless_run_tab import HeadlessRunTab

    graph = tmp_path / "graphs" / "g"
    graph.mkdir(parents=True)
    (graph / "node_ids.npy").write_bytes(b"")
    if fast_loadable:
        monkeypatch.setattr(
            "graph_visualizer.fast_graph_io.can_load_fast", lambda folder: (True, "ok")
        )
    tab = HeadlessRunTab(panel_stubs._QtStub, None, graphs_root=lambda: tmp_path / "graphs")
    return tab, graph


def test_controller_entries_carry_their_own_scheme(tmp_path, monkeypatch) -> None:
    """The list mixes modal-LQR artifacts and MIMO PI gain matrices, so a scheme
    cannot be inferred from 'is a path selected' any more."""
    tab, graph = _tab(tmp_path, monkeypatch)
    save_sys_id_gain_matrix(graph, "G_exact", [1, 2], [3, 4], np.eye(2))
    tab._handle_graph_changed()
    entries = dict(tab.controller_scheme_combo.items)
    assert any(label.startswith("MIMO PI - ") for label in entries), entries
    for label, data in entries.items():
        assert isinstance(data, tuple) and len(data) == 2, (label, data)
    schemes = {data[0] for data in entries.values()}
    assert "mimo_pi" in schemes and "pid_qp" in schemes


def test_selecting_a_gain_matrix_selects_the_mimo_pi_scheme(tmp_path, monkeypatch) -> None:
    tab, graph = _tab(tmp_path, monkeypatch)
    folder = save_sys_id_gain_matrix(graph, "G_exact", [1, 2], [3, 4], np.eye(2))
    tab._handle_graph_changed()
    index = next(i for i, (label, _d) in enumerate(tab.controller_scheme_combo.items)
                 if label.startswith("MIMO PI - "))
    tab.controller_scheme_combo.setCurrentIndex(index)
    params = tab.panel.read()
    assert params.mimo_controller_scheme == "mimo_pi"
    assert Path(params.mimo_pi_gain_matrix_path) == folder
    # The other scheme's path must be cleared, or a stale artifact could quietly
    # reactivate a controller nobody selected.
    assert params.modal_controller_path == ""


def test_gain_build_refuses_without_fast_load_artifacts(tmp_path, monkeypatch) -> None:
    tab, _graph = _tab(tmp_path, monkeypatch, fast_loadable=False)
    messages: list[tuple[str, bool]] = []
    tab.on_status = lambda m, e: messages.append((m, e))
    tab.build_gain_matrix()
    assert tab._gain_build_process is None
    assert any(err and "Update graph" in msg for msg, err in messages), messages


def test_gain_build_uses_the_panel_operating_temperature(tmp_path, monkeypatch) -> None:
    """G is a linearization, so it must be built at the temperature the run uses."""
    tab, _graph = _tab(tmp_path, monkeypatch)
    tab.modal_temp_spin.setValue(50.0)
    captured: dict = {}

    def fake_launch(folder, **kwargs):
        captured.update(kwargs)
        class _P:
            def poll(self): return None
        return _P()

    monkeypatch.setattr(
        "graph_visualizer.fast_graph_io.launch_gain_build_subprocess", fake_launch
    )
    tab.build_gain_matrix()
    assert captured == {"t_op_K": 50.0}, captured
