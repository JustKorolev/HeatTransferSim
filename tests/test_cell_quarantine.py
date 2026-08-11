"""Quarantine of thermal dead ends, and the discrete-time LQR design.

Both cover the same 2026-08-10 ``no_mli_high_res`` failure from opposite ends: a
detached CAD solid absorbed 60 W of a 95 W run and cooked to 4894 K, while a
continuous-time LQR gain applied at dt = 4 s made every heater command flip sign
on every sample.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from graph_visualizer.cell_quarantine import (
    find_quarantined_cells,
    fully_orphaned_heaters,
    deposition_targets_lost,
)
from graph_visualizer.modal_reduction import discrete_lqr_gain, lqr_weights


def _laplacian(edges, n):
    """Conduction Laplacian from ``(i, j, G)`` triples."""
    L = np.zeros((n, n), dtype=float)
    for i, j, g in edges:
        L[i, j] -= g
        L[j, i] -= g
        L[i, i] += g
        L[j, j] += g
    return csr_matrix(L)


class _Node:
    def __init__(self, deposition=(), weights=()):
        self.power_deposition_node_ids = list(deposition)
        self.power_deposition_weights = list(weights)


class _Model:
    def __init__(self, nodes):
        self.nodes = nodes


# --------------------------------------------------------------- quarantine

def test_component_without_a_sink_is_quarantined_whole():
    """The interior of a detached solid has healthy local conductance and is
    invisible to any per-node test -- only reachability catches it."""
    # 0-1-2 reach the cooler at 0; 3-4-5 are a well-connected but detached island.
    L = _laplacian([(0, 1, 5.0), (1, 2, 5.0), (3, 4, 5.0), (4, 5, 5.0)], 6)
    result = find_quarantined_cells(L, sink_rows=[0])

    assert result.mask.tolist() == [False, False, False, True, True, True]
    assert result.count == 3
    assert result.quarantined_component_count == 1
    assert result.component_count == 2
    # Every island cell has conductance 5-10 W/K, so a per-node floor would miss them.
    assert sorted(result.unreachable_rows.tolist()) == [3, 4, 5]
    assert result.below_floor_rows.size == 0


def test_cells_with_no_edges_are_quarantined():
    L = _laplacian([(0, 1, 5.0)], 3)  # node 2 has no edges at all
    result = find_quarantined_cells(L, sink_rows=[0])
    assert result.mask.tolist() == [False, False, True]
    assert result.isolated_rows.tolist() == [2]


def test_sink_reachable_body_is_untouched():
    L = _laplacian([(0, 1, 5.0), (1, 2, 5.0)], 3)
    result = find_quarantined_cells(L, sink_rows=[2])
    assert not result.any_quarantined
    assert "no inert cells" in result.summary()


def test_conductance_floor_is_off_by_default():
    """A legitimately poorly-conducting cell (e.g. 1 mm G10 at ~1e-4 W/K) must
    survive the default settings -- a per-node floor is not safe unconditionally."""
    L = _laplacian([(0, 1, 5.0), (1, 2, 1.0e-4)], 3)
    assert not find_quarantined_cells(L, sink_rows=[0]).any_quarantined

    opted_in = find_quarantined_cells(L, sink_rows=[0], min_conductance_W_per_K=1.0e-3)
    assert opted_in.mask.tolist() == [False, False, True]
    assert opted_in.below_floor_rows.tolist() == [2]


def test_reachability_is_skipped_when_the_model_has_no_cooler():
    """A radiation-only or open-loop model is legitimate; quarantining the whole
    graph because nothing reaches a cryocooler would be absurd."""
    L = _laplacian([(0, 1, 5.0), (2, 3, 5.0)], 4)
    result = find_quarantined_cells(L, sink_rows=[])
    assert not result.any_quarantined
    assert "no cryocooler cells" in result.skipped_reason
    assert "skipped" in result.summary()


def test_orphaned_and_partially_lost_heaters_are_reported_separately():
    mask = np.array([False, False, True, True])
    index = {10: 0, 11: 1, 12: 2, 13: 3}
    model = _Model({
        100: _Node(deposition=[12, 13]),        # every target quarantined -> orphan
        101: _Node(deposition=[10, 12]),        # half lost -> still functional
        102: _Node(deposition=[10, 11]),        # untouched
    })
    heaters = [100, 101, 102]

    assert fully_orphaned_heaters(model, heaters, index, mask) == [100]
    lost = deposition_targets_lost(model, heaters, index, mask)
    assert lost == {100: [12, 13], 101: [12]}
    assert 102 not in lost


def test_summary_names_the_offending_nodes():
    L = _laplacian([(0, 1, 5.0), (2, 3, 5.0)], 4)
    node_ids = np.array([700, 701, 3755, 3756])
    text = find_quarantined_cells(L, sink_rows=[0]).summary(node_ids)
    assert "3755" in text and "2 cell(s)" in text


# ------------------------------------------------------------- discrete LQR

def _representative_plant(r=12, m=6, seed=0):
    """A stable plant spanning this cryostat's real time-constant range: the
    slowest mode is ~2813 s and the fastest is sub-second, which is exactly the
    spread that makes a sampled continuous-time gain unsafe."""
    rng = np.random.default_rng(seed)
    lam = np.logspace(np.log10(1.0 / 2813.0), np.log10(1.0 / 0.1), r)
    T = rng.normal(size=(r, r))
    A = T @ np.diag(-lam) @ np.linalg.inv(T)
    B = rng.normal(size=(r, m)) * 1.0e-3
    C = rng.normal(size=(m, r))
    return A, B, C


def _closed_loop_spectral_radius(A, B, K, dt):
    from scipy.signal import cont2discrete

    Ad, Bd, *_ = cont2discrete((A, B, np.zeros((1, A.shape[0])), np.zeros((1, B.shape[1]))), dt)
    return float(np.max(np.abs(np.linalg.eigvals(Ad - Bd @ K))))


@pytest.mark.parametrize("dt", [0.5, 1.0, 4.0, 8.0])
def test_discrete_gain_is_stable_at_its_design_rate(dt):
    """The whole point: designing at the sample rate puts the sampled closed-loop
    poles inside the unit circle by construction, at every dt."""
    A, B, C = _representative_plant()
    Q, R = lqr_weights(A, B, C, 0.1)
    K = discrete_lqr_gain(A, B, Q, R, dt)
    assert _closed_loop_spectral_radius(A, B, K, dt) < 1.0


def test_continuous_gain_is_unstable_when_sampled():
    """The measured failure: a continuous-time gain applied on a sampled loop
    leaves the unit circle, which is what produced commands alternating sign on
    every single step."""
    from scipy.linalg import solve_continuous_are

    A, B, C = _representative_plant()
    Q, R = lqr_weights(A, B, C, 0.1)
    K_cont = np.linalg.solve(R, B.T @ solve_continuous_are(A, B, Q, R))

    assert _closed_loop_spectral_radius(A, B, K_cont, 4.0) > 1.0
    K_disc = discrete_lqr_gain(A, B, Q, R, 4.0)
    assert _closed_loop_spectral_radius(A, B, K_disc, 4.0) < 1.0


def test_gain_converges_to_the_continuous_one_as_dt_shrinks():
    """Sanity check on the discretization: the two designs must agree in the
    limit, or the discrete solve is not solving the same problem."""
    from scipy.linalg import solve_continuous_are

    A, B, C = _representative_plant()
    Q, R = lqr_weights(A, B, C, 0.1)
    K_cont = np.linalg.solve(R, B.T @ solve_continuous_are(A, B, Q, R))

    errors = [
        np.linalg.norm(discrete_lqr_gain(A, B, Q, R, dt) - K_cont) / np.linalg.norm(K_cont)
        for dt in (1.0e-2, 1.0e-3, 1.0e-4)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1.0e-2


def test_design_timestep_must_be_positive_and_finite():
    A, B, C = _representative_plant()
    Q, R = lqr_weights(A, B, C, 0.1)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            discrete_lqr_gain(A, B, Q, R, bad)


# ------------------------------------------------------- end-to-end regression

def test_detached_solid_with_a_heater_no_longer_runs_away():
    """The 2026-08-10 failure, in miniature.

    A heater deposits into a small solid that has no conduction path to the
    cryocooler. Before the quarantine that cell absorbed its full command forever
    and cooked (4894 K on the real graph); now the power never enters the solve,
    the cell holds its initial temperature, and the reported power_in excludes it.
    """
    from graph_visualizer.matrix_builder import build_matrices
    from graph_visualizer.models import (
        GraphMetadata,
        HeaterProperties,
        NodeProperties,
        ThermalGraphModel,
    )
    from graph_visualizer.simulation_model import prepare_simulation
    from graph_visualizer.simulation_parameters import SimulationParameters

    def _cell(node_id, temperature=50.0, capacity=10.0):
        node = NodeProperties.with_material(node_id, (node_id, 0, 0), material="Copper")
        node.C_J_K = float(capacity)
        node.mass_kg = max(1.0e-12, capacity / max(float(node.cp_J_kgK), 1.0e-12))
        node.initial_temperature_K = float(temperature)
        node.Grad_W_K = node.G_rad_W_K = 0.0
        node.is_exposed = False
        node.center_mm = (float(node_id), 0.0, 0.0)
        node.size_mm = (1.0, 1.0, 1.0)
        return node

    model = ThermalGraphModel(metadata=GraphMetadata(graph_name="detached_solid"))
    # Main body: 1-2, with the cryocooler on 1. Island: 3 (no edge to the body).
    for node_id in (1, 2, 3):
        model.add_node(_cell(node_id))
    model.nodes[1].has_cryocooler = True
    model.set_edge(1, 2, 5.0)

    heater = _cell(4, capacity=1.0e-6)
    heater.is_heater = True
    heater.heater = HeaterProperties(heater_id=4, heater_max_power_W=30.0, heater_efficiency=1.0)
    heater.power_deposition_node_ids = [3]      # deposits ONLY into the island
    heater.power_deposition_weights = [1.0]
    model.add_node(heater)

    params = SimulationParameters(
        dt_s=1.0, t_final_s=10.0, input_mode="heater_inputs",
        cryocooler_enabled=True, gpu_solver_enabled=False,
        simulation_history_limit=0, browser_simulation_size_warning=1_000_000,
    )
    prepared = prepare_simulation(model, build_matrices(model), params)

    island_row = prepared.node_index_by_id[3]
    assert prepared.inert_cell_mask is not None
    assert bool(prepared.inert_cell_mask[island_row])
    assert prepared.orphaned_heater_ids == (4,)

    before = float(prepared.temperatures_K[island_row])
    for _ in range(50):
        prepared.step_with_forced_heater_powers({4: 30.0}, keep_cryocoolers_active=False)
    after = float(prepared.temperatures_K[island_row])

    # 30 W into a 10 J/K cell for 50 s would be +150 K without the quarantine.
    assert after == pytest.approx(before, abs=1.0e-9)

    # ...and the energy accounting reports it as commanded-but-undelivered rather
    # than as power that entered the system.
    prepared.heater_actuator_power_by_node = lambda **_kwargs: {4: 30.0}
    balance = prepared.power_balance_W()
    assert balance["heater_commanded_W"] == pytest.approx(30.0)
    assert balance["heater_W"] == pytest.approx(0.0)
    assert balance["heater_undelivered_W"] == pytest.approx(30.0)


def test_a_heater_split_across_good_and_dead_cells_keeps_its_live_share():
    """Partial loss must scale the delivered power by the surviving weights, not
    drop the heater entirely."""
    from graph_visualizer.matrix_builder import build_matrices
    from graph_visualizer.models import (
        GraphMetadata,
        HeaterProperties,
        NodeProperties,
        ThermalGraphModel,
    )
    from graph_visualizer.simulation_model import prepare_simulation
    from graph_visualizer.simulation_parameters import SimulationParameters

    def _cell(node_id, capacity=10.0):
        node = NodeProperties.with_material(node_id, (node_id, 0, 0), material="Copper")
        node.C_J_K = float(capacity)
        node.mass_kg = max(1.0e-12, capacity / max(float(node.cp_J_kgK), 1.0e-12))
        node.initial_temperature_K = 50.0
        node.Grad_W_K = node.G_rad_W_K = 0.0
        node.is_exposed = False
        node.center_mm = (float(node_id), 0.0, 0.0)
        node.size_mm = (1.0, 1.0, 1.0)
        return node

    model = ThermalGraphModel(metadata=GraphMetadata(graph_name="split_heater"))
    for node_id in (1, 2, 3):
        model.add_node(_cell(node_id))
    model.nodes[1].has_cryocooler = True
    model.set_edge(1, 2, 5.0)

    heater = _cell(4, capacity=1.0e-6)
    heater.is_heater = True
    heater.heater = HeaterProperties(heater_id=4, heater_max_power_W=30.0, heater_efficiency=1.0)
    heater.power_deposition_node_ids = [2, 3]     # 2 survives, 3 is the island
    heater.power_deposition_weights = [3.0, 1.0]  # 75% live, 25% dead
    model.add_node(heater)

    params = SimulationParameters(
        dt_s=1.0, t_final_s=10.0, input_mode="heater_inputs",
        cryocooler_enabled=True, gpu_solver_enabled=False,
        simulation_history_limit=0, browser_simulation_size_warning=1_000_000,
    )
    prepared = prepare_simulation(model, build_matrices(model), params)

    assert prepared.orphaned_heater_ids == ()          # not orphaned: half of it works
    assert 4 in prepared.heaters_missing_deposition    # but it did lose a target

    prepared.heater_actuator_power_by_node = lambda **_kwargs: {4: 40.0}
    balance = prepared.power_balance_W()
    assert balance["heater_commanded_W"] == pytest.approx(40.0)
    assert balance["heater_W"] == pytest.approx(30.0)   # 75% of 40 W
    assert balance["heater_undelivered_W"] == pytest.approx(10.0)
