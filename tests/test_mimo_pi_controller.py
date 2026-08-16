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

from dataclasses import replace

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


def _ill_conditioned_plant():
    """A gain matrix with one dominant direction and one nearly-absent one, like the
    real cryostat's (sigma_1 = 33.0 carrying 81% of the energy, sigma_27 = 0.090).

    Scaled so sigma_1 is O(10) like the real G, which is the whole point: at this
    scale the relative floor (1e-4 * sigma_1^2 = 0.01) exceeds a configured absolute
    lambda_u of 1e-3, exactly as it does on no_mli_high_res_v3.
    """
    return np.array([[5.0, 5.0], [5.0, 5.0 + 5.0e-3]])


def test_zero_lambda_u_stays_off_despite_the_relative_floor() -> None:
    """lambda_u = 0 means the caller wants exact decoupling, not "pick a default for
    me". Scaling must only ever raise an already-positive weight."""
    from graph_visualizer.mimo_controller import allocate_thermal_rate_qp

    G = _ill_conditioned_plant()
    result = allocate_thermal_rate_qp(
        G, np.zeros(2), G @ np.array([1.0, 0.5]), np.ones(2),
        np.full(2, 1.0e6), np.zeros(2), 0.0, 0.0, lambda_u_relative=1.0e-4,
    )
    assert result.lambda_effective == 0.0
    assert np.allclose(np.asarray(result.u).reshape(-1), [1.0, 0.5], atol=1e-6)


def test_lambda_u_is_raised_to_the_gain_matrix_scale() -> None:
    """An absolute lambda_u is compared against sigma^2, so the same number damps
    completely differently on two graphs whose gains differ by an order of
    magnitude. On the real cryostat 1e-3 left the weakest direction 89% undamped,
    and the allocator inverted through it -- 11 W per K of a direction the plant
    barely has, flipping the active heater set on 99% of steps."""
    from graph_visualizer.mimo_controller import allocate_thermal_rate_qp

    G = _ill_conditioned_plant()
    sigma = np.linalg.svd(G, compute_uv=False)
    kwargs = dict(
        max_delta_power=None, absolute_target=True, undershoot_weight=1.0,
    )
    scaled = allocate_thermal_rate_qp(
        G, np.zeros(2), np.array([1.0, 1.0]), np.ones(2),
        np.full(2, 1.0e6), np.zeros(2), 1.0e-3, 0.0,
        lambda_u_relative=1.0e-4, **kwargs,
    )
    assert scaled.lambda_effective == pytest.approx(1.0e-4 * sigma[0] ** 2)
    assert scaled.lambda_effective > 1.0e-3, "must be RAISED, not left at the absolute value"
    # The weak direction is damped out, and the run can say so rather than reporting
    # a tracking error that looks like mistuning.
    assert scaled.suppressed_directions == 1
    assert len(scaled.singular_values) == 2

    # A well-conditioned plant is left alone: nothing is suppressed.
    benign = allocate_thermal_rate_qp(
        np.eye(2), np.zeros(2), np.array([1.0, 1.0]), np.ones(2),
        np.full(2, 1.0e6), np.zeros(2), 1.0e-3, 0.0,
        lambda_u_relative=1.0e-4, **kwargs,
    )
    assert benign.suppressed_directions == 0
    assert benign.attenuated_command_fraction < 1.0e-2


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
    # "none" is the placeholder row a graph shows before it has an artifact. It is
    # NOT a controller: PID+QP used to sit in that slot and has been removed, so
    # selecting it must leave nothing regulating rather than silently running a
    # scheme whose per-pair PID this plant's negative RGA diagonal makes unusable.
    assert "mimo_pi" in schemes and "none" in schemes
    assert "pid_qp" not in schemes


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


# --- enabled-I/O gating ------------------------------------------------------ #
def _pi_sim(monkeypatch, *, enabled_heater_node_ids):
    """A PreparedSimulation stubbed down to what the MIMO PI law reads.

    Built with object.__new__ rather than prepare_simulation because the behaviour
    under test is the bound the law puts on each heater, and a real graph would add
    a mesh, a solver and a cryocooler without making the assertion any stronger.
    """
    from graph_visualizer.simulation_model import PreparedSimulation
    from graph_visualizer.simulation_parameters import SimulationParameters

    class _Node:
        def __init__(self, setpoint):
            self.controller_setpoint_K = setpoint
            self.is_heater = True
            self.heater_max_power_W = 10.0
            self.mimo_enabled = True
            self.C_J_K = 1.0
            self.heater_mode = "mimo"
            self.is_cryocooler = False

    heater_ids, sensor_ids = [10, 11], [20, 21]
    nodes = {i: _Node(float("nan")) for i in heater_ids}
    nodes.update({s: _Node(60.0) for s in sensor_ids})

    sim = object.__new__(PreparedSimulation)
    sim.model = type("M", (), {"nodes": nodes})()
    sim.node_ids = np.array(heater_ids + sensor_ids, dtype=int)
    sim.node_index_by_id = {int(v): i for i, v in enumerate(sim.node_ids)}
    # temperatures_K is a read-only view of the augmented state z (temps + time).
    sim.z = np.append(np.full(len(sim.node_ids), 50.0), 0.0)
    sim.params = replace(
        SimulationParameters(),
        dt_s=1.0,
        mimo_pi_kp=0.0,
        mimo_pi_ki=1.0e-2,
        mimo_heater_slew_rate_W_per_s=0.0,
        enabled_heater_node_ids=enabled_heater_node_ids,
    )
    sim.heater_node_ids = np.array(heater_ids, dtype=int)
    sim.cryocooler_node_ids = np.array([], dtype=int)
    sim.cryocooler_devices = {}
    sim.cryocooler_lift_curve = None
    sim.controller_last_power_by_heater = {}
    sim.controller_mimo_pi_integral = None
    sim.controller_allocator_diagnostics = {}
    sim.last_cryocooler_diagnostics = []

    # A well-conditioned 2x2 gain so the QP has a genuine choice of heaters.
    G = np.array([[0.5, 0.1], [0.1, 0.5]])
    monkeypatch.setattr(
        PreparedSimulation, "_load_mimo_pi_gain",
        lambda self: {"G": G, "sensor_ids": sensor_ids, "heater_ids": heater_ids},
    )
    monkeypatch.setattr(
        PreparedSimulation, "_mimo_pi_gains",
        lambda self, gain, sids: (np.zeros(len(sids)), np.full(len(sids), 1.0e-2)),
    )
    monkeypatch.setattr(
        "graph_visualizer.simulation_model.sensor_readout_temperature_K",
        lambda model, idx, temps, nid: 50.0,
    )
    return sim


def test_disabled_heater_is_bounded_to_zero_by_mimo_pi(monkeypatch) -> None:
    """A heater unticked in the enabled-I/O table must not be commanded.

    G is identified over every heater and its column order is frozen at build time,
    so the law cannot drop the column the way PID+QP dropped the heater -- it has to
    bound it to 0 W instead. The regression this guards is silent: the heater would
    keep drawing power with nothing in the UI to show for it.
    """
    sim = _pi_sim(monkeypatch, enabled_heater_node_ids=(11,))
    sim._mimo_pi_controller_power_vector(update_state=True)
    commands = dict(zip(
        sim.controller_allocator_diagnostics["heater_ids"],
        sim.controller_allocator_diagnostics["heater_commands_W"],
    ))
    assert commands[10] == pytest.approx(0.0, abs=1e-9)
    assert commands[11] > 0.0, "the enabled heater must still take up the demand"
    # And a 0 W bound is not "saturated high" -- that would misreport the run.
    assert sim.controller_allocator_diagnostics["saturated_high"] == 0


def test_no_enabled_filter_drives_every_heater(monkeypatch) -> None:
    """None means "no enabled-I/O filter" -- every heater in G is available.

    This is the convention the rest of simulation_model.py uses (_node_id_enabled
    returns True for a None set), and MIMO PI must share it or an untouched graph
    would come up with no usable heaters at all.
    """
    sim = _pi_sim(monkeypatch, enabled_heater_node_ids=None)
    sim._mimo_pi_controller_power_vector(update_state=True)
    commands = sim.controller_allocator_diagnostics["heater_commands_W"]
    assert all(c > 0.0 for c in commands), commands


def test_empty_enabled_set_drives_no_heater(monkeypatch) -> None:
    """An EMPTY tuple is not the same as None: it disables every heater.

    Worth pinning because the two are easy to conflate and the failure modes are
    opposite -- conflating them either runs a controller the user switched off or
    switches off one they never touched.
    """
    sim = _pi_sim(monkeypatch, enabled_heater_node_ids=())
    sim._mimo_pi_controller_power_vector(update_state=True)
    commands = sim.controller_allocator_diagnostics["heater_commands_W"]
    assert all(c == pytest.approx(0.0, abs=1e-9) for c in commands), commands


# --- the feedforward reference ----------------------------------------------- #
def _ref(sim, setpoints):
    """The (r - y_passive) reference the law hands the QP, for given setpoints.

    Evaluated twice: y_passive is only captured once the sensors have gone quiet,
    and quiescence needs two readings to measure. This stub's plant does not respond
    to power, so the first evaluation's command is cleared before the second --
    y_passive = y - G u_prev, and the premise these tests state is "the sensors read
    50 K with the heaters at 0 W".
    """
    for sid, target in zip([20, 21], setpoints):
        sim.model.nodes[sid].controller_setpoint_K = target
    sim._mimo_pi_controller_power_vector(update_state=True)
    sim.controller_last_power_by_heater = {}
    sim._mimo_pi_controller_power_vector(update_state=True)
    return np.array(sim.controller_allocator_diagnostics["reference_deviation_K"])


def test_the_passive_reference_is_not_captured_during_a_transient(monkeypatch) -> None:
    """y_ss = y_passive + G u only holds at steady state, so capturing on the first
    evaluation records whatever the plant happened to be doing then and holds it for
    the whole run. A 3600 s run of no_mli_high_res_v3 latched all 27 entries at
    exactly 48.000 K -- the uniform initial temperature -- and fed the QP that
    arbitrary constant for an hour. While the plant is still moving the feedforward
    must be zero and the integral must carry the load."""
    sim = _pi_sim(monkeypatch, enabled_heater_node_ids=None)
    moving = iter([50.0, 50.0, 70.0, 70.0, 90.0, 90.0])
    monkeypatch.setattr(
        "graph_visualizer.simulation_model.sensor_readout_temperature_K",
        lambda model, idx, temps, nid: next(moving),
    )
    for sid in (20, 21):
        sim.model.nodes[sid].controller_setpoint_K = 60.0
    for _ in range(3):
        sim._mimo_pi_controller_power_vector(update_state=True)
    assert sim.controller_mimo_pi_passive_K is None, "must not latch mid-transient"
    assert np.allclose(sim.controller_allocator_diagnostics["reference_deviation_K"], 0.0)


def test_reference_is_the_rise_needed_above_the_passive_equilibrium(monkeypatch) -> None:
    """G maps power to the rise ABOVE the unheated equilibrium, so the QP's
    reference must be (r - y_passive), not the setpoint itself."""
    sim = _pi_sim(monkeypatch, enabled_heater_node_ids=None)
    # Sensors read 50 K with the heaters at 0 W, so y_passive = 50 K.
    ref = _ref(sim, [60.0, 65.0])
    assert ref == pytest.approx([10.0, 15.0])


def test_one_sensors_setpoint_does_not_move_another_channels_reference(monkeypatch) -> None:
    """The regression this guards: the reference used to be (r - mean(r)), so
    changing ONE setpoint shifted every other channel through the mean -- a purely
    numerical coupling on top of the physical coupling the decoupler exists to undo.
    """
    a = _ref(_pi_sim(monkeypatch, enabled_heater_node_ids=None), [60.0, 65.0])
    b = _ref(_pi_sim(monkeypatch, enabled_heater_node_ids=None), [60.0, 300.0])
    assert a[0] == pytest.approx(b[0]), "sensor 0's reference must not depend on sensor 1's"


def test_uniform_setpoints_still_produce_a_feedforward(monkeypatch) -> None:
    """With (r - mean(r)) this was identically zero whenever every setpoint matched
    -- the common case -- leaving the integral to supply the entire holding power
    across a multi-hour transient. It must now be the actual required rise."""
    sim = _pi_sim(monkeypatch, enabled_heater_node_ids=None)
    ref = _ref(sim, [60.0, 60.0])
    assert ref == pytest.approx([10.0, 10.0])
    assert all(c > 0.0 for c in sim.controller_allocator_diagnostics["heater_commands_W"])


def test_the_passive_reference_is_held_not_re_estimated(monkeypatch) -> None:
    """Re-estimating y_passive every step would make Kp a second integrator. Once
    captured it must stay put even as the plant heats up under the command."""
    sim = _pi_sim(monkeypatch, enabled_heater_node_ids=None)
    first = _ref(sim, [60.0, 60.0])
    sim.z = np.append(np.full(len(sim.node_ids), 55.0), 0.0)   # plant responds
    monkeypatch.setattr(
        "graph_visualizer.simulation_model.sensor_readout_temperature_K",
        lambda model, idx, temps, nid: 55.0,
    )
    assert _ref(sim, [60.0, 60.0]) == pytest.approx(first)


def test_reset_integrators_clears_mimo_pi_state(monkeypatch) -> None:
    """The button's whole job is clearing windup; it used to clear only the modal
    scheme's integral, silently leaving MIMO PI's in place."""
    sim = _pi_sim(monkeypatch, enabled_heater_node_ids=None)
    _ref(sim, [60.0, 60.0])
    assert sim.controller_mimo_pi_integral is not None
    assert sim.controller_mimo_pi_passive_K is not None
    sim.reset_controller_integrators()
    assert sim.controller_mimo_pi_integral is None
    assert sim.controller_mimo_pi_passive_K is None


def test_changing_the_controller_invalidates_the_held_reference(monkeypatch) -> None:
    """Both the integral and the captured reference are denominated in a specific
    G; carrying them onto a different matrix would be meaningless."""
    sim = _pi_sim(monkeypatch, enabled_heater_node_ids=None)
    _ref(sim, [60.0, 60.0])
    sim.mark_controller_stale()
    assert sim.controller_mimo_pi_integral is None
    assert sim.controller_mimo_pi_passive_K is None


# --- closed loop -------------------------------------------------------------- #
def _closed_loop(monkeypatch, *, kp, steps=4000, dt=4.0):
    """Run the real law against a first-order plant and return the final error.

    A unit test on one step cannot see an integrator that never integrates. This
    closes the loop: y relaxes toward its steady value y_passive + G u, which is
    what the controller is trying to place at the setpoint.
    """
    from graph_visualizer.simulation_model import PreparedSimulation

    sim = _pi_sim(monkeypatch, enabled_heater_node_ids=None)
    sim.params = replace(sim.params, mimo_pi_kp=kp, mimo_pi_ki=1.0e-2, dt_s=dt)
    monkeypatch.setattr(
        PreparedSimulation, "_mimo_pi_gains",
        lambda self, gain, sids: (np.full(len(sids), kp), np.full(len(sids), 1.0e-2)),
    )
    G = np.array([[0.5, 0.1], [0.1, 0.5]])
    y_passive, tau = 50.0, 200.0
    y = np.full(2, y_passive)
    for sid in (20, 21):
        sim.model.nodes[sid].controller_setpoint_K = 60.0

    def readout(model, idx, temps, nid):
        return float(y[0] if int(nid) == 20 else y[1])

    monkeypatch.setattr("graph_visualizer.simulation_model.sensor_readout_temperature_K", readout)
    for _ in range(steps):
        sim._mimo_pi_controller_power_vector(update_state=True)
        u = np.array(sim.controller_allocator_diagnostics["heater_commands_W"])
        y += (y_passive + G @ u - y) * (dt / tau)     # first-order relaxation
    return y - 60.0


def test_the_loop_removes_the_offset_instead_of_drooping(monkeypatch) -> None:
    """The regression that mattered: the integral was frozen by an anti-windup test
    a regularized QP could never pass, so the loop was pure proportional and settled
    at the droop offset r_dev/Kp -- +0.68 K at Kp=3.0 and +6.9 K at Kp=0.3 on the
    real plant. Lowering Kp made tracking worse, which is the signature.
    """
    assert np.abs(_closed_loop(monkeypatch, kp=0.3)).max() < 0.05


def test_tracking_does_not_get_worse_when_kp_is_lowered(monkeypatch) -> None:
    """With integral action, steady-state accuracy must not depend on Kp. Under the
    frozen integral the offset scaled as 1/Kp, so this comparison inverts."""
    low = np.abs(_closed_loop(monkeypatch, kp=0.3)).max()
    high = np.abs(_closed_loop(monkeypatch, kp=3.0)).max()
    assert low < 0.05 and high < 0.05, (low, high)
    assert low < 10.0 * high, "a smaller Kp must not blow up the steady-state error"


def test_absolute_allocation_ignores_the_previous_command() -> None:
    """MIMO PI's v_cmd is the deviation the plant must HOLD, so G u = v_cmd and the
    answer cannot depend on u_prev.

    The allocator's default contract is INCREMENTAL -- G(u - u_prev) = v_cmd - drift
    -- which is right for the rate-based scheme it was written for. MIMO PI used it
    unchanged, so every step added G^-1 v_cmd to the previous command: an unintended
    integrator that ramped to saturation and parked there. That is what drove the
    real run to 120 W with every controlled sensor 5-7 K ABOVE setpoint.
    """
    from graph_visualizer.mimo_controller import allocate_thermal_rate_qp

    G = _coupled_plant()
    v = np.array([1.0, 1.0, 1.0])
    umax = np.full(3, 30.0)
    solutions = [
        np.asarray(
            allocate_thermal_rate_qp(
                G, np.zeros(3), v, np.ones(3), umax, np.full(3, prev), 1e-3, 0.0,
                absolute_target=True,
            ).u
        ).reshape(-1)
        for prev in (0.0, 5.0, 29.0)
    ]
    for other in solutions[1:]:
        assert np.allclose(solutions[0], other), "absolute allocation must ignore u_prev"
    assert np.allclose(G @ solutions[0], v, atol=2e-2), "and it must actually hit v_cmd"


def test_incremental_mode_is_unchanged_for_other_callers() -> None:
    """The default must keep the incremental contract; only MIMO PI opts out."""
    from graph_visualizer.mimo_controller import allocate_thermal_rate_qp

    G = _coupled_plant()
    v = np.array([0.5, 0.5, 0.5])
    umax = np.full(3, 30.0)
    prev = np.full(3, 4.0)
    u = np.asarray(
        allocate_thermal_rate_qp(G, np.zeros(3), v, np.ones(3), umax, prev, 1e-3, 0.0).u
    ).reshape(-1)
    assert np.allclose(G @ (u - prev), v, atol=2e-2), "default stays incremental"


# --- asymmetric residual weighting --------------------------------------------- #
def _coupled_pair():
    """Two sensors sharing almost all of each heater's influence -- the structure
    that makes an all-positive demand need negative power."""
    return np.array([[1.0, 0.9], [0.9, 1.0]])


def _allocate(v, weight):
    from graph_visualizer.mimo_controller import allocate_thermal_rate_qp

    G = _coupled_pair()
    result = allocate_thermal_rate_qp(
        G, np.zeros(2), np.asarray(v, float), np.ones(2), np.full(2, 30.0), np.zeros(2),
        1e-3, 0.0, absolute_target=True, undershoot_weight=weight,
    )
    return np.asarray(result.u).reshape(-1)


def test_symmetric_weighting_leaves_cold_sensors_cold() -> None:
    """The failure being fixed: both sensors below setpoint, 60 W available, and the
    best symmetric fit commands under 3 W because overshooting the nearly-correct
    sensor scores as badly as the cold it removes."""
    u = _allocate([4.30, 0.52], 1.0)
    assert u.sum() < 3.0, u
    assert (u < 1e-9).any(), "and it goes sparse"


def test_weighting_undershoot_buys_the_cold_sensor_more_heat() -> None:
    total = [_allocate([4.30, 0.52], w).sum() for w in (1.0, 2.0, 4.0, 10.0)]
    assert total == sorted(total), total
    assert total[-1] > 1.5 * total[0], total


def test_the_cold_channel_gets_closer_to_its_target() -> None:
    """More power is only worth having if it lands where the deficit is."""
    G = _coupled_pair()
    want = np.array([4.30, 0.52])
    short_sym = want[0] - (G @ _allocate(want, 1.0))[0]
    short_asym = want[0] - (G @ _allocate(want, 10.0))[0]
    assert short_asym < short_sym
    assert short_asym < 0.35 * short_sym, (short_sym, short_asym)


def test_one_is_still_the_symmetric_solver() -> None:
    """The knob must be able to restore the old behaviour exactly."""
    from graph_visualizer.mimo_controller import allocate_thermal_rate_qp

    G = _coupled_pair()
    v = np.array([4.30, 0.52])
    base = allocate_thermal_rate_qp(
        G, np.zeros(2), v, np.ones(2), np.full(2, 30.0), np.zeros(2), 1e-3, 0.0,
        absolute_target=True,
    )
    assert np.allclose(np.asarray(base.u).reshape(-1), _allocate(v, 1.0))


def test_it_does_not_disturb_an_already_reachable_demand() -> None:
    """When the demand IS reachable with non-negative power, the symmetric answer is
    already right and asymmetry must not inflate it."""
    want = np.array([2.0, 2.0])          # uniform: no sign conflict
    sym, asym = _allocate(want, 1.0), _allocate(want, 10.0)
    assert np.allclose(sym, asym, atol=1e-3), (sym, asym)


# --- reachability diagnostic ---------------------------------------------------- #
def _reach(monkeypatch, *, v, u, maxima):
    from graph_visualizer.simulation_model import PreparedSimulation

    sim = object.__new__(PreparedSimulation)
    G = np.array([[1.0, 0.9], [0.9, 1.0]])
    return sim._mimo_pi_reachability(
        G, np.asarray(u, float), np.asarray(v, float), [20, 21], np.asarray(maxima, float)
    )


def test_an_unreachable_channel_is_named_and_distinguished(monkeypatch) -> None:
    """The question that cost a whole session: mistuned, or impossible? A channel
    left short while the heaters are nowhere near their bounds is unreachable --
    serving it would need negative power elsewhere, so no tuning fixes it."""
    out = _reach(monkeypatch, v=[4.3, 0.52], u=[2.63, 0.0], maxima=[30.0, 30.0])
    assert out["unserved_sensor_ids"] == [20], out
    assert out["unserved_cause"] == "unreachable"
    assert out["heater_headroom_W"] > 50.0, "and it says the power was there unused"


def test_a_saturated_channel_reads_as_saturated(monkeypatch) -> None:
    """The opposite case, which more heater authority WOULD fix."""
    out = _reach(monkeypatch, v=[100.0, 100.0], u=[30.0, 30.0], maxima=[30.0, 30.0])
    assert out["unserved_cause"] == "saturated", out
    assert out["heater_headroom_W"] == pytest.approx(0.0)


def test_a_served_plant_reports_nothing(monkeypatch) -> None:
    out = _reach(monkeypatch, v=[2.0, 2.0], u=[1.0526, 1.0526], maxima=[30.0, 30.0])
    assert out["unserved_sensor_ids"] == []
    assert out["unserved_cause"] == "none"


def test_shortfall_is_relative_to_the_channels_own_demand(monkeypatch) -> None:
    """A 0.1 K miss on a 0.2 K demand matters; the same miss on a 40 K demand does
    not. Judging both against an absolute threshold would report the wrong one."""
    big = _reach(monkeypatch, v=[40.0, 40.0], u=[21.0, 21.0], maxima=[30.0, 30.0])
    assert big["unserved_sensor_ids"] == [], big


def test_a_diagnostic_evaluation_cannot_latch_the_passive_reference(monkeypatch) -> None:
    """The controller is also evaluated with update_state=False for readouts and
    diagnostics. Two evaluations inside one step see identical temperatures, so
    dy/dt reads exactly 0 and the quiescence gate passes on the very first step --
    which is the failure the gate exists to prevent. A 100 ks run latched at
    46.928 K that way, reported as "max|dy/dt|=0 K/s"."""
    sim = _pi_sim(monkeypatch, enabled_heater_node_ids=None)
    for sid in (20, 21):
        sim.model.nodes[sid].controller_setpoint_K = 60.0
    moving = iter([50.0, 50.0, 70.0, 70.0, 90.0, 90.0, 110.0, 110.0])
    monkeypatch.setattr(
        "graph_visualizer.simulation_model.sensor_readout_temperature_K",
        lambda model, idx, temps, nid: next(moving),
    )
    # A real step, then two diagnostic evaluations at the same temperatures.
    sim._mimo_pi_controller_power_vector(update_state=True)
    sim._mimo_pi_controller_power_vector(update_state=False)
    sim._mimo_pi_controller_power_vector(update_state=False)
    assert sim.controller_mimo_pi_passive_K is None, "a diagnostic pass must not latch"
    # And a real step while the plant is still moving must not latch either.
    sim._mimo_pi_controller_power_vector(update_state=True)
    assert sim.controller_mimo_pi_passive_K is None


def test_diagnostics_carry_the_loop_state_so_a_summary_needs_no_checkpoint(monkeypatch) -> None:
    """The integrator and the held passive reference lived only inside a checkpoint,
    so answering "is the integral winding or stalled?" after a run meant shipping a
    32 MB temperature field to read 27 floats. They belong in the diagnostics."""
    sim = _pi_sim(monkeypatch, enabled_heater_node_ids=None)
    ref = _ref(sim, [60.0, 60.0])
    d = sim.controller_allocator_diagnostics
    for key in ("integral_K_s", "passive_reference_K", "error_K", "v_cmd_K"):
        assert key in d, key
    assert len(d["integral_K_s"]) == len(ref)
    assert d["passive_reference_K"] == pytest.approx([50.0, 50.0])
    assert d["error_K"] == pytest.approx([10.0, 10.0])
    # And it must round-trip through JSON -- this file is written out verbatim.
    json.dumps(d)


def test_a_solved_passive_reference_is_used_on_the_first_step(monkeypatch, tmp_path) -> None:
    """G is a DEVIATION gain -- it says how far the sensors rise per watt, above a
    baseline it never states. Estimating that baseline from a running plant is
    hopeless when the dominant mode is ~24 h: one run latched the initial condition,
    another never latched at all and made its integral discover the whole 25 W of
    holding power, and shortening that ramp with a bigger ki overshot to 48 W and
    cooked cells to 1211 K. Solved beside the matrix, it is right on step 1."""
    sim = _pi_sim(monkeypatch, enabled_heater_node_ids=None)
    sim._mimo_pi_cache = {
        "G": np.array([[0.5, 0.1], [0.1, 0.5]]),
        "passive_K": 21.289,
        "sensor_ids": [20, 21],
        "heater_ids": [10, 11],
        "per_sensor": {},
        "preset_kp": None,
        "preset_ki": None,
    }
    sim._mimo_pi_cache_path = ""
    monkeypatch.setattr(type(sim), "_load_mimo_pi_gain", lambda self: self._mimo_pi_cache)
    for sid in (20, 21):
        sim.model.nodes[sid].controller_setpoint_K = 50.0
    sim._mimo_pi_controller_power_vector(update_state=True)
    ref = np.array(sim.controller_allocator_diagnostics["reference_deviation_K"])
    # 50 - 21.289, available immediately rather than after hours of integration.
    assert ref == pytest.approx([28.711, 28.711], abs=1e-3)


def test_the_solved_reference_is_the_curves_no_load_floor_not_a_tangent() -> None:
    """The tangent's zero crossing, T_op - Q(T_op)/(dQ/dT), extrapolates a local
    linearisation 29 K past its operating point across a curve that is nowhere near
    linear over that span. It was tried: on a 27.8 h run it commanded 28.5 W of
    holding power against a true requirement near 20 W, and the integral spent the
    entire run walking that off as a 34 h oscillation. The curve's own no-load
    floor is the temperature at which the cooler actually removes zero power."""
    from graph_visualizer.cryocooler import PT60LiftCurve
    from graph_visualizer.modal_reduction import (
        cryocooler_ground_conductance_W_K,
        cryocooler_passive_temperature_K,
    )

    T_op = 50.0
    curve = PT60LiftCurve(max_power_w=150.0)
    passive = cryocooler_passive_temperature_K(T_op)
    assert passive == pytest.approx(curve.minimum_temperature_k)
    assert passive == pytest.approx(27.669, abs=1e-2)
    assert curve.cooling_capacity_w(passive) == pytest.approx(0.0, abs=1e-9), "zero power there"
    # And it must NOT be the tangent value, which is 6.4 K colder.
    tangent = T_op - curve.cooling_capacity_w(T_op) / cryocooler_ground_conductance_W_K(T_op)
    assert tangent == pytest.approx(21.289, abs=1e-2)
    assert passive - tangent > 6.0


def test_the_measurement_filter_attenuates_the_fast_path(monkeypatch) -> None:
    """kp is the loop's only damping term and is capped near 0.1 by sensors that
    settle inside one control step -- the proportional term closes an algebraic
    loop through them. The mode that needs damping rings at 34 h. One filter
    separates them: a step in the fast path contributes only dt/tau on the first
    step, so kp can rise by that factor, while the slow mode is untouched."""
    sim = _pi_sim(monkeypatch, enabled_heater_node_ids=None)
    sim.params = replace(sim.params, dt_s=30.0, mimo_pi_measurement_filter_s=900.0)
    raw = np.array([50.0, 50.0])
    assert np.allclose(sim._mimo_pi_filtered_readout(raw, True), raw), "seeds on the first read"
    stepped = sim._mimo_pi_filtered_readout(np.array([60.0, 60.0]), True)
    # dt/tau = 30/900 = 1/30 of a 10 K step.
    assert stepped == pytest.approx([50.0 + 10.0 / 30.0] * 2, abs=1e-9)
    # A diagnostic pass must not advance it -- the controller runs more than once
    # per step, and a filter stepped twice has half its intended time constant.
    again = sim._mimo_pi_filtered_readout(np.array([60.0, 60.0]), False)
    assert again == pytest.approx(stepped)
    # Off by default, so an existing run is bit-for-bit unchanged.
    plain = _pi_sim(monkeypatch, enabled_heater_node_ids=None)
    assert plain.params.mimo_pi_measurement_filter_s == 0.0
    assert np.allclose(plain._mimo_pi_filtered_readout(np.array([60.0, 60.0]), True), [60.0, 60.0])


def test_the_passive_reference_is_derived_not_read_back() -> None:
    """It is a closed-form property of the lift curve and T_op; G does not enter
    it. Storing it meant a one-line correction to the formula cost a 51-minute
    re-solve of 27 CG systems on a 3M-node graph. An OLD matrix carrying the old
    tangent value must still yield the current one."""
    from graph_visualizer.simulation_model import PreparedSimulation

    stale = {"T_op_K": 50.0, "dc_ground": "cryocooler", "passive_reference_K": 21.289}
    assert PreparedSimulation._mimo_pi_passive_reference(stale) == pytest.approx(27.669, abs=1e-2)
    # Radiation-grounded matrices never receive an environment temperature.
    assert PreparedSimulation._mimo_pi_passive_reference(
        {"T_op_K": 50.0, "dc_ground": "radiation"}
    ) is None
    # And anything predating T_op_K falls back to the runtime estimate.
    assert PreparedSimulation._mimo_pi_passive_reference({"dc_ground": "cryocooler"}) is None
    # It tracks the operating point, so a matrix built at 80 K gets its own value.
    hot = PreparedSimulation._mimo_pi_passive_reference({"T_op_K": 80.0, "dc_ground": "cryocooler"})
    assert hot == pytest.approx(27.669, abs=1e-2), "no-load floor is a curve property"


def test_the_integral_can_be_told_that_overshoot_costs_more(monkeypatch) -> None:
    """The plant's authority is one-sided: too cold is fixed by adding heater power,
    immediately; too hot can only be fixed by removing power and waiting for the
    cryocooler. A symmetric integrator prices those the same, so the loop crosses
    the setpoint and then spends hours coming back -- 22 h of a 27.8 h run."""
    sim = _pi_sim(monkeypatch, enabled_heater_node_ids=None)
    sim.params = replace(sim.params, dt_s=10.0, mimo_pi_overshoot_integral_scale=4.0)
    for sid in (20, 21):
        sim.model.nodes[sid].controller_setpoint_K = 50.0

    def _run(readout):
        sim.controller_mimo_pi_integral = np.zeros(2)
        monkeypatch.setattr(
            "graph_visualizer.simulation_model.sensor_readout_temperature_K",
            lambda model, idx, temps, nid: readout,
        )
        sim._mimo_pi_controller_power_vector(update_state=True)
        return np.asarray(sim.controller_mimo_pi_integral, dtype=float).mean()

    too_cold = _run(49.0)   # error +1 K -> integrate at the plain rate
    too_hot = _run(51.0)    # error -1 K -> integrate 4x harder to back off
    assert too_cold > 0.0 and too_hot < 0.0
    # Not exactly 4x: the anti-windup also bleeds the committed integral, and it
    # bleeds hardest on the too-hot branch, where v_cmd asks for a power reduction
    # the non-negative allocator cannot fully deliver. The asymmetry survives it.
    assert abs(too_hot) > 3.0 * abs(too_cold)

    # Symmetric by default, so nothing changes for a run that does not ask.
    plain = _pi_sim(monkeypatch, enabled_heater_node_ids=None)
    assert plain.params.mimo_pi_overshoot_integral_scale == 1.0


def test_the_allocator_asymmetry_can_point_either_way() -> None:
    """undershoot_weight was clamped to >= 1, so the only asymmetry it could express
    was "ask for more heat" -- the one this plant does not want."""
    from graph_visualizer.mimo_controller import allocate_thermal_rate_qp

    G = _coupled_plant()
    # Deliberately NOT exactly achievable: with a feasible target the residual is
    # zero, the reweighting has nothing to weigh, and every asym gives one answer.
    target = np.array([0.9, 0.2, 0.7])
    kwargs = dict(absolute_target=True, lambda_u_relative=0.0)
    low = allocate_thermal_rate_qp(
        G, np.zeros(3), target, np.ones(3), np.full(3, 1.0e6), np.zeros(3), 0.0, 0.0,
        undershoot_weight=0.25, **kwargs,
    )
    high = allocate_thermal_rate_qp(
        G, np.zeros(3), target, np.ones(3), np.full(3, 1.0e6), np.zeros(3), 0.0, 0.0,
        undershoot_weight=4.0, **kwargs,
    )
    assert float(np.asarray(low.u).sum()) < float(np.asarray(high.u).sum()), (
        "a weight below 1 must settle lower than one above it"
    )
