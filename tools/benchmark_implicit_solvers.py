"""Benchmark linear-solver strategies for the implicit thermal step.

The implicit TR-BDF2 / backward-Euler stepper solves, at every stage of every
substep, an SPD system

    M(h) x = b,   M(h) = diag(C) + alpha(h) * L

where L is the graph Laplacian (W/K), C the per-node capacitance (J/K), and
alpha(h) grows with the substep size h. The *time integrator* and its accuracy
are fixed by the scheme and the number of substeps; this script isolates the
one remaining choice -- how each SPD solve is performed -- and measures its
setup cost, per-solve cost, memory, and accuracy on a real saved graph.

Methods (each skipped cleanly if its library is missing):
  cg_none     scipy CG, no preconditioner (baseline iteration count)
  cg_jacobi   scipy CG + Jacobi preconditioner, rebuilt every solve (current)
  cg_bjac     scipy CG + block-Jacobi (contiguous dense-block inverses)
  superlu     scipy SuperLU factorization, factored once, reused (SciPy-only)
  cholmod     scikit-sparse CHOLMOD Cholesky, factored once, reused
  amg_cg      pyamg smoothed-aggregation AMG as a CG preconditioner, built once
  gpu_cg      cupy/cupyx CG + Jacobi on the GPU

Usage:
    python tools/benchmark_implicit_solvers.py graphs/VALIDATION_V1
    python tools/benchmark_implicit_solvers.py graphs/VALIDATION_V1 --substeps 4 64 --solves 20
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, load_npz
from scipy.sparse.linalg import LinearOperator, cg, splu

GAMMA = 2.0 - np.sqrt(2.0)  # TR-BDF2 stage parameter, matches SparseImplicitStepper


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_operator(folder: Path) -> tuple[np.ndarray, csr_matrix, np.ndarray]:
    """Return (C, L_csr, T0) loaded directly from the saved matrix files."""
    C = np.load(folder / "C.npy").astype(float).reshape(-1)
    T0_path = folder / "initial_temperature_K.npy"
    T0 = (
        np.load(T0_path).astype(float).reshape(-1)
        if T0_path.exists()
        else np.full_like(C, 293.15)
    )
    npz_path = folder / "L_sparse.npz"
    if npz_path.exists():
        L = load_npz(npz_path).tocsr()
    else:
        with (folder / "L_sparse.json").open() as handle:
            payload = json.load(handle)
        shape = tuple(int(v) for v in payload["shape"])
        L = coo_matrix(
            (
                np.asarray(payload["data"], dtype=float),
                (np.asarray(payload["row"], dtype=int), np.asarray(payload["col"], dtype=int)),
            ),
            shape=shape,
        ).tocsr()
    if C.size != L.shape[0]:
        raise ValueError(f"C length {C.size} does not match L shape {L.shape}.")
    return C, L, T0


def stage_matrix(C: np.ndarray, L: csr_matrix, h: float) -> csr_matrix:
    """The stiffer TR-BDF2 stage-1 matrix diag(C) + 0.5*gamma*h*L (SPD)."""
    alpha = 0.5 * GAMMA * float(h)
    return (diags(C, format="csr") + alpha * L).tocsr()


def realistic_rhs(C: np.ndarray, L: csr_matrix, M: csr_matrix, T0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A per-substep correction with known ground truth.

    The real solver solves for a *small smooth correction* (previous temperature
    is the initial guess), not the full field from zero. We take one explicit
    passive increment as the true correction, then b = M @ x_true.
    """
    h_ref = 1.0
    x_true = -h_ref * (L @ T0) / C  # O(1 K) smooth field, the scale of a substep
    b = np.asarray(M @ x_true, dtype=float).reshape(-1)
    return x_true, b


# --------------------------------------------------------------------------- #
# Memory helper
# --------------------------------------------------------------------------- #
def rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Preconditioner builders
# --------------------------------------------------------------------------- #
def build_block_jacobi(M: csr_matrix, block_size: int) -> LinearOperator:
    """Contiguous block-Jacobi preconditioner (matches SparseImplicitStepper)."""
    n = M.shape[0]
    num_blocks = (n + block_size - 1) // block_size
    padded_n = num_blocks * block_size
    blocks = np.zeros((num_blocks, block_size, block_size), dtype=float)
    idx = np.arange(block_size)
    blocks[:, idx, idx] = 1.0
    coo = M.tocoo()
    same = (coo.row // block_size) == (coo.col // block_size)
    blocks[coo.row[same] // block_size, coo.row[same] % block_size, coo.col[same] % block_size] = coo.data[same]
    inv_blocks = np.linalg.inv(blocks)

    def matvec(v):
        vec = np.asarray(v, dtype=float).reshape(-1)
        buf = np.zeros(padded_n) if padded_n != n else vec
        if padded_n != n:
            buf[:n] = vec
        applied = np.einsum("kij,kj->ki", inv_blocks, buf.reshape(num_blocks, block_size)).reshape(-1)
        return applied[:n]

    return LinearOperator(M.shape, matvec=matvec)


def _cg_with_count(M, b, prec, rtol):
    """One CG solve; returns (x, info, iterations)."""
    iters = 0

    def _cb(_xk):
        nonlocal iters
        iters += 1

    x, info = cg(M, b, rtol=rtol, atol=0.0, maxiter=5000, M=prec, callback=_cb)
    return x, info, iters


# --------------------------------------------------------------------------- #
# Method runners: each returns dict(setup_s, solve_s, rel_err, rel_resid, iters, note)
# --------------------------------------------------------------------------- #
def bench_cg_none(M, b, x_true, solves, rtol):
    t0 = time.perf_counter()
    for _ in range(solves):
        x, info, iters = _cg_with_count(M, b, None, rtol)
    solve_s = (time.perf_counter() - t0) / solves
    return _result(0.0, solve_s, M, b, x, x_true, iters=iters, note=f"info={info}")


def bench_cg_jacobi(M, b, x_true, solves, rtol):
    # No reusable setup: Jacobi is rebuilt each solve in the real code, so we
    # fold its (tiny) cost into per-solve timing.
    t0 = time.perf_counter()
    for _ in range(solves):
        inv_diag = 1.0 / M.diagonal()
        prec = LinearOperator(M.shape, matvec=lambda v: inv_diag * v)
        x, info, iters = _cg_with_count(M, b, prec, rtol)
    solve_s = (time.perf_counter() - t0) / solves
    return _result(0.0, solve_s, M, b, x, x_true, iters=iters, note=f"info={info}")


def bench_cg_bjac(M, b, x_true, solves, rtol, block_size=64):
    t0 = time.perf_counter()
    prec = build_block_jacobi(M, block_size)
    setup_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(solves):
        x, info, iters = _cg_with_count(M, b, prec, rtol)
    solve_s = (time.perf_counter() - t0) / solves
    return _result(setup_s, solve_s, M, b, x, x_true, iters=iters, note=f"block={block_size} info={info}")


def bench_superlu(M, b, x_true, solves, rtol):
    t0 = time.perf_counter()
    lu = splu(M.tocsc())
    setup_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(solves):
        x = lu.solve(b)
    solve_s = (time.perf_counter() - t0) / solves
    factor_nnz = lu.L.nnz + lu.U.nnz
    note = f"factor_nnz={factor_nnz:,} (~{factor_nnz * 12 / 1e6:.0f} MB)"
    return _result(setup_s, solve_s, M, b, x, x_true, note=note)


def bench_cholmod(M, b, x_true, solves, rtol):
    from sksparse.cholmod import cholesky  # type: ignore

    t0 = time.perf_counter()
    factor = cholesky(M.tocsc())
    setup_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(solves):
        x = factor(b)
    solve_s = (time.perf_counter() - t0) / solves
    return _result(setup_s, solve_s, M, b, np.asarray(x).reshape(-1), x_true)


def bench_amg_cg(M, b, x_true, solves, rtol):
    import pyamg  # type: ignore

    t0 = time.perf_counter()
    ml = pyamg.smoothed_aggregation_solver(M.tocsr())
    prec = ml.aspreconditioner(cycle="V")
    setup_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(solves):
        x, info, iters = _cg_with_count(M, b, prec, rtol)
    solve_s = (time.perf_counter() - t0) / solves
    return _result(setup_s, solve_s, M, b, x, x_true, iters=iters, note=f"levels={len(ml.levels)}")


def bench_gpu_cg(M, b, x_true, solves, rtol):
    import cupy as cp  # type: ignore
    from cupyx.scipy.sparse import csr_matrix as cp_csr  # type: ignore
    from cupyx.scipy.sparse.linalg import cg as cp_cg  # type: ignore
    from cupyx.scipy.sparse.linalg import LinearOperator as cp_linop  # type: ignore

    t0 = time.perf_counter()
    M_gpu = cp_csr(M)
    inv_diag = 1.0 / M_gpu.diagonal()
    prec = cp_linop(M_gpu.shape, matvec=lambda v: inv_diag * v)
    b_gpu = cp.asarray(b)
    cp.cuda.Stream.null.synchronize()
    setup_s = time.perf_counter() - t0
    iters = 0

    def _cb(_xk):
        nonlocal iters
        iters += 1

    t0 = time.perf_counter()
    for _ in range(solves):
        x_gpu, info = cp_cg(M_gpu, b_gpu, rtol=rtol, atol=0.0, maxiter=5000, M=prec, callback=_cb)
    cp.cuda.Stream.null.synchronize()
    solve_s = (time.perf_counter() - t0) / solves
    x = cp.asnumpy(x_gpu)
    return _result(setup_s, solve_s, M, b, x, x_true, iters=iters, note=f"info={int(info)}")


def _result(setup_s, solve_s, M, b, x, x_true, iters=None, note=""):
    x = np.asarray(x, dtype=float).reshape(-1)
    rel_err = float(np.linalg.norm(x - x_true) / max(np.linalg.norm(x_true), 1e-30))
    resid = np.asarray(M @ x, dtype=float).reshape(-1) - b
    rel_resid = float(np.linalg.norm(resid) / max(np.linalg.norm(b), 1e-30))
    return {
        "setup_s": setup_s,
        "solve_s": solve_s,
        "rel_err": rel_err,
        "rel_resid": rel_resid,
        "iters": iters,
        "note": note,
    }


METHODS = {
    "cg_none": bench_cg_none,
    "cg_jacobi": bench_cg_jacobi,
    "cg_bjac": bench_cg_bjac,
    "superlu": bench_superlu,
    "cholmod": bench_cholmod,
    "amg_cg": bench_amg_cg,
    "gpu_cg": bench_gpu_cg,
}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph_folder", type=Path)
    parser.add_argument("--dt", type=float, default=1.0, help="Outer display timestep dt_s.")
    parser.add_argument(
        "--substeps",
        type=int,
        nargs="+",
        default=[4, 64],
        help="Substep counts to test (defines h=dt/substeps and matrix stiffness).",
    )
    parser.add_argument("--solves", type=int, default=15, help="Solves timed per method (setup reused).")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Iterative tolerance (matches params default).")
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[64],
        help="Block sizes for the block-Jacobi method (one cg_bjac<N> row per size).",
    )
    args = parser.parse_args()

    print(f"Loading operator from {args.graph_folder} ...")
    t0 = time.perf_counter()
    C, L, T0 = load_operator(args.graph_folder)
    n = C.size
    print(
        f"  n = {n:,} nodes | L nnz = {L.nnz:,} | load {time.perf_counter() - t0:.1f}s | "
        f"baseline RSS {rss_mb() and f'{rss_mb():.0f} MB' or 'n/a'}"
    )

    for substeps in args.substeps:
        h = args.dt / substeps
        M = stage_matrix(C, L, h)
        x_true, b = realistic_rhs(C, L, M, T0)
        stiffness = 0.5 * GAMMA * h * float(np.max(np.abs(L.diagonal())) / np.max(C))
        print(f"\n{'='*92}")
        print(
            f"substeps={substeps}  ->  h={h:.4g}s   M = diag(C) + {0.5*GAMMA*h:.4g}*L   "
            f"(alpha*max(Lii)/max? ~{stiffness:.2g})"
        )
        print(f"{'method':<12}{'setup(s)':>10}{'solve(ms)':>11}{'iters':>8}{'rel_err':>12}{'rel_resid':>12}   note")
        print("-" * 100)
        # Expand cg_bjac into one runner per requested block size.
        methods = {}
        for name, fn in METHODS.items():
            if name == "cg_bjac":
                for bs in args.block_sizes:
                    methods[f"cg_bjac{bs}"] = (lambda M, b, x, s, r, _bs=bs: bench_cg_bjac(M, b, x, s, r, _bs))
            else:
                methods[name] = fn
        rows = {}
        for name, fn in methods.items():
            try:
                gc.collect()
                r = fn(M, b, x_true, args.solves, args.rtol)
                rows[name] = r
                iters_str = "-" if r.get("iters") is None else str(int(r["iters"]))
                print(
                    f"{name:<12}{r['setup_s']:>10.3f}{r['solve_s']*1e3:>11.2f}{iters_str:>8}"
                    f"{r['rel_err']:>12.2e}{r['rel_resid']:>12.2e}   {r['note']}"
                )
            except ImportError as exc:
                print(f"{name:<12}{'--':>10}{'skipped':>11}   library missing: {exc.name}")
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"{name:<12}{'--':>10}{'FAILED':>11}   {type(exc).__name__}: {str(exc)[:50]}")

        # Throughput at fixed accuracy: TR-BDF2 does 2 stage-solves per substep.
        # Setup is one-time, amortized to ~0 over a full simulation run.
        print(f"\n  simulated-seconds per wall-second (dt={args.dt}s, {substeps} substeps x 2 stage solves):")
        for name, r in rows.items():
            per_outer = 2 * substeps * r["solve_s"]
            sim_per_wall = args.dt / max(per_outer, 1e-12)
            print(f"    {name:<12} {sim_per_wall:>10.2f} x   ({per_outer*1e3:.1f} ms / outer step; setup {r['setup_s']:.2f}s once)")


if __name__ == "__main__":
    main()
