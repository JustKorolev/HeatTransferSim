"""The DC-gain solve must scale to the real graph.

dc_gain_and_rga used to always call splu. A sparse LU of a 3D thermal mesh fills
in far faster than the node count grows, so a build that took seconds at 300k
nodes ran for tens of minutes at 3M without finishing. L_dc is SPD (a Laplacian
plus a nonnegative grounding diagonal), so CG solves the same system with only
matrix-vector products, one column per heater.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

from graph_visualizer.modal_reduction import (
    DC_GAIN_DIRECT_MAX_NODES,
    _dc_solve_iterative,
    dc_gain_and_rga,
)


def _spd_plant(n=300, m=5, k=7, seed=0):
    """A grounded random thermal network: symmetric weights, positive grounding."""
    rng = np.random.default_rng(seed)
    A = np.triu(rng.random((n, n)) < 0.02, 1)
    W = np.where(A, rng.random((n, n)) * 0.5 + 0.1, 0.0)
    W = W + W.T
    L = (csr_matrix(diags(W.sum(1)) - csr_matrix(W)) + diags(np.full(n, 0.05))).tocsr()
    F = np.zeros((n, m))
    for j, cell in enumerate(rng.choice(n, m, replace=False)):
        F[cell, j] = 1.0
    S = np.eye(n)[rng.choice(n, k, replace=False)]
    monitor = np.zeros(k, dtype=bool)
    monitor[-2:] = True
    return L, F, S, monitor


def test_iterative_solve_reproduces_the_direct_factorization() -> None:
    """The two paths must be interchangeable -- the size threshold is a performance
    decision, so it must not change the answer."""
    L, F, S, monitor = _spd_plant()
    G_direct, _ = dc_gain_and_rga(L, F, S, monitor)
    G_iter = S[~monitor] @ _dc_solve_iterative(L, F, rtol=1.0e-12)
    assert np.allclose(G_direct, G_iter, rtol=1.0e-7, atol=1.0e-12)


def test_a_zero_heater_column_is_not_solved_for() -> None:
    """A heater with no deposition cells has a zero load, hence a zero gain column.
    CG on a zero right-hand side has no meaningful relative residual, so it must be
    skipped rather than tripping the convergence check."""
    L, F, S, monitor = _spd_plant()
    F[:, 2] = 0.0
    X = _dc_solve_iterative(L, F, rtol=1.0e-12)
    assert np.all(X[:, 2] == 0.0)
    assert np.any(X[:, 0] != 0.0)


def test_an_ungrounded_operator_is_reported_not_silently_wrong() -> None:
    """Without a cryocooler or radiation path the Laplacian is singular and the
    steady state is not unique. A near-converged answer would look plausible and be
    meaningless, so the build must stop and say why."""
    L, F, S, _monitor = _spd_plant()
    ungrounded = (csr_matrix(L) - diags(np.full(L.shape[0], 0.05))).tocsr()
    with pytest.raises(RuntimeError, match="did not converge|singular"):
        _dc_solve_iterative(ungrounded, F, rtol=1.0e-14, maxiter=200)


def test_the_threshold_keeps_small_graphs_on_the_direct_path() -> None:
    """splu is exact and faster below the crossover; the change is meant to rescue
    large graphs, not to slow down every small one."""
    assert DC_GAIN_DIRECT_MAX_NODES >= 100_000


def test_progress_is_reported_per_column() -> None:
    """A silent multi-hour solve is indistinguishable from a hung one -- that is
    exactly how the original was noticed."""
    L, F, S, _m = _spd_plant()
    seen: list[str] = []
    _dc_solve_iterative(L, F, progress=seen.append, rtol=1.0e-12)
    assert seen, "the iterative solve must report progress"
    assert any("residual" in line for line in seen)


# --- parallel column solves --------------------------------------------------- #
def _projection(n, k, seed=1):
    rng = np.random.default_rng(seed)
    S = np.zeros((k, n))
    for i, cell in enumerate(rng.choice(n, k, replace=False)):
        S[i, cell] = 1.0
    return S


def test_parallel_and_serial_give_the_same_gain() -> None:
    """Parallelism is a scheduling change, not a numerical one. The columns are
    independent, so the result must be bit-for-bit what serial produces."""
    from graph_visualizer.modal_reduction import _dc_gain_iterative

    L, F, _S, _m = _spd_plant(n=300, m=6)
    S_ctrl = _projection(L.shape[0], 4)
    serial = _dc_gain_iterative(L, F, S_ctrl, workers=1)
    parallel = _dc_gain_iterative(L, F, S_ctrl, workers=3)
    assert np.array_equal(serial, parallel)


def test_worker_count_is_capped_by_the_memory_bus_not_the_core_count() -> None:
    """CG on a sparse operator is bandwidth bound: measured speedup saturates at
    ~3.2x around 8 workers and is WORSE at 16. Spawning one process per core would
    cost a full copy of the operator each (~250 MB at 3M nodes) to go slower."""
    from graph_visualizer.modal_reduction import DC_GAIN_MAX_WORKERS, _dc_default_workers

    assert _dc_default_workers(1000) <= DC_GAIN_MAX_WORKERS
    assert _dc_default_workers(2) == 2, "never more workers than there is work"


def test_a_zero_column_is_skipped_in_the_parallel_path_too() -> None:
    """A heater that deposits nowhere has no relative residual to check, so it must
    not reach a worker at all."""
    from graph_visualizer.modal_reduction import _dc_gain_iterative

    L, F, _S, _m = _spd_plant(n=300, m=6)
    F[:, 3] = 0.0
    S_ctrl = _projection(L.shape[0], 4)
    G = _dc_gain_iterative(L, F, S_ctrl, workers=3)
    assert np.all(G[:, 3] == 0.0)
    assert np.any(G[:, 0] != 0.0)


def test_a_non_converging_column_still_raises_from_a_worker() -> None:
    """The convergence guard must survive the process boundary -- an exception that
    got swallowed into the serial fallback would silently return a wrong G."""
    from graph_visualizer.modal_reduction import _dc_gain_iterative

    L, F, _S, _m = _spd_plant(n=300, m=6)
    ungrounded = (csr_matrix(L) - diags(np.full(L.shape[0], 0.05))).tocsr()
    S_ctrl = _projection(L.shape[0], 4)
    with pytest.raises(RuntimeError, match="did not converge|singular"):
        _dc_gain_iterative(ungrounded, F, S_ctrl, workers=3, rtol=1.0e-14, maxiter=200)


def test_a_crashed_worker_falls_back_to_serial_instead_of_losing_the_build(monkeypatch) -> None:
    """BrokenProcessPool is itself a RuntimeError, so an `except RuntimeError: raise`
    guard would abort the build when a worker segfaults -- which scipy's _sparsetools
    has done on this hardware. A dead pool means retry serially.
    """
    from concurrent.futures.process import BrokenProcessPool

    from graph_visualizer import modal_reduction as mr

    L, F, _S, _m = _spd_plant(n=300, m=5)
    S_ctrl = _projection(L.shape[0], 4)
    expected = mr._dc_gain_iterative(L, F, S_ctrl, workers=1)

    def explode(*_a, **_k):
        raise BrokenProcessPool("worker died")

    monkeypatch.setattr(mr, "_dc_gain_parallel", explode)
    notes: list[str] = []
    got = mr._dc_gain_iterative(L, F, S_ctrl, workers=4, progress=notes.append)
    assert np.array_equal(got, expected), "the serial fallback must give the real answer"
    assert any("falling back to serial" in n for n in notes), notes


def test_a_non_converged_column_is_not_retried_serially(monkeypatch) -> None:
    """The opposite case: a numerical failure would fail identically in serial, so
    retrying only wastes the time again and buries the reason."""
    from graph_visualizer import modal_reduction as mr

    L, F, _S, _m = _spd_plant(n=300, m=5)
    S_ctrl = _projection(L.shape[0], 4)
    calls: list[int] = []

    def not_converged(*_a, **_k):
        calls.append(1)
        raise mr.DCSolveNotConverged("column 1/5 did not converge")

    monkeypatch.setattr(mr, "_dc_gain_parallel", not_converged)
    with pytest.raises(mr.DCSolveNotConverged):
        mr._dc_gain_iterative(L, F, S_ctrl, workers=4)
    assert len(calls) == 1, "must not fall through to a serial retry"


def test_the_projection_is_shipped_sparsely_to_workers() -> None:
    """S is dense but structurally sparse: 618 MB at 3M nodes x 27 sensors, copied
    to every worker. Densely that is ~5 GB of pickling before the first solve."""
    import pickle

    from scipy.sparse import csr_matrix

    from graph_visualizer import modal_reduction as mr

    L, F, _S, _m = _spd_plant(n=300, m=4)
    S_ctrl = _projection(L.shape[0], 6)
    captured = {}

    def capture(Leff, live, proj, G, *rest):
        captured["proj"] = proj
        return mr._dc_gain_parallel(Leff, live, proj, G, *rest)

    original = mr._dc_gain_parallel
    try:
        mr._dc_gain_parallel = capture
        mr._dc_gain_iterative(L, F, S_ctrl, workers=2)
    finally:
        mr._dc_gain_parallel = original

    proj = captured["proj"]
    assert isinstance(proj, csr_matrix), type(proj)
    assert len(pickle.dumps(proj)) < len(pickle.dumps(S_ctrl))
