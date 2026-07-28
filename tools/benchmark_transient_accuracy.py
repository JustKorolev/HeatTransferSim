"""Quantify how the implicit stepper's substep cap distorts a fast transient.

Transient accuracy is set by the time-integration substep size, not the linear
solver. SparseImplicitStepper caps substeps at `adaptive_max_substeps` (default
4). When a step wants a large temperature change (the VALIDATION_V1 trace shows
a 42.7 K first step), the cap forces coarse substeps and TR-BDF2 -- being
L-stable -- over-damps / smears the transient instead of blowing up.

This script drives the *real* SparseImplicitStepper on the real saved operator
with a controlled constant source sized to reproduce that ~40 K first-step jump,
and compares substep caps {1,4,16,64} against the exact linear-ODE solution
(expm_multiply on the augmented system). Radiation off, constant source -> the
exponential solution is exact, so all error shown is time-integration error.

Usage:
    python tools/benchmark_transient_accuracy.py graphs/VALIDATION_V1
    python tools/benchmark_transient_accuracy.py graphs/VALIDATION_V1 --steps 4 --caps 1 4 16 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import expm_multiply

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_visualizer.simulation_model import SparseImplicitStepper  # noqa: E402


def load_operator(folder: Path):
    C = np.load(folder / "C.npy").astype(float).reshape(-1)
    T0_path = folder / "initial_temperature_K.npy"
    T0 = np.load(T0_path).astype(float).reshape(-1) if T0_path.exists() else np.full_like(C, 293.15)
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
    return C, L, T0


def build_source(C: np.ndarray, L: csr_matrix, target_rate_K_s: float, n_nodes: int):
    """Constant source (K/s) on n_nodes well-connected small-C nodes.

    Sized so max dT/dt ~= target_rate_K_s, reproducing the stiff first step.
    Returns source_K_s (what the stepper consumes) and the equivalent power (W).
    """
    connected = np.where(np.asarray(L.diagonal()).reshape(-1) > 0.0)[0]
    order = connected[np.argsort(C[connected])]
    chosen = order[:n_nodes]
    source = np.zeros_like(C)
    source[chosen] = float(target_rate_K_s)  # dT/dt contribution = P/C
    power_W = source * C
    return source, power_W, chosen


def reference_trajectory(C, L, T0, source_K_s, dt, steps):
    """Exact T after each step for C dT/dt = -L T + C*source (constant source)."""
    inv_C = 1.0 / C
    A = -(diags(inv_C, format="csr") @ L)
    s = source_K_s  # already dT/dt = A T + s
    n = C.size
    A_aug = bmat([[A, csr_matrix(s.reshape(-1, 1))], [csr_matrix((1, n)), csr_matrix((1, 1))]], format="csr")
    z = np.concatenate([T0, [1.0]])
    traj = []
    step_op = A_aug * float(dt)
    for _ in range(steps):
        z = expm_multiply(step_op, z)
        traj.append(z[:n].copy())
    return np.array(traj)


def run_stepper(C, L, T0, source_K_s, dt, steps, cap, target_delta_K, rtol):
    stepper = SparseImplicitStepper(
        dt_s=float(dt),
        rtol=float(rtol),
        maxiter=2000,
        solver="cg",
        state_operator=csr_matrix(L),
        capacitance_J_K=C,
        method="tr_bdf2",
        adaptive_substeps_enabled=True,
        adaptive_target_delta_K=float(target_delta_K),
        adaptive_max_substeps=int(cap),
        residual_check_enabled=False,
    )
    T = T0.copy()
    traj = []
    substeps_used = []
    t0 = time.perf_counter()
    for _ in range(steps):
        T = stepper.step(T, source_K_s)
        traj.append(T.copy())
        substeps_used.append(int(stepper.last_substeps))
    walltime = time.perf_counter() - t0
    return np.array(traj), substeps_used, walltime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("graph_folder", type=Path)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--caps", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--target-delta-K", type=float, default=1.0, help="adaptive_target_delta_K (default matches params).")
    parser.add_argument("--target-rate-K-s", type=float, default=40.0, help="Peak dT/dt of the controlled transient.")
    parser.add_argument("--source-nodes", type=int, default=200)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument(
        "--passive-perturb-K",
        type=float,
        default=0.0,
        help="If >0: zero source, and start from T0 with an alternating +/-K perturbation "
        "on connected nodes (a large, stiff passive relaxation). Tests big actual excursions.",
    )
    args = parser.parse_args()

    print(f"Loading operator from {args.graph_folder} ...")
    C, L, T0 = load_operator(args.graph_folder)
    print(f"  n = {C.size:,} nodes | L nnz = {L.nnz:,}")

    if args.passive_perturb_K > 0.0:
        connected = np.where(np.asarray(L.diagonal()).reshape(-1) > 0.0)[0]
        sign = np.where((connected % 2) == 0, 1.0, -1.0)
        T0 = T0.copy()
        T0[connected] += args.passive_perturb_K * sign
        source = np.zeros_like(C)
        print(
            f"  passive relaxation from a +/-{args.passive_perturb_K:.0f} K alternating perturbation "
            f"on {connected.size:,} connected nodes (source = 0)"
        )
    else:
        source, power_W, chosen = build_source(C, L, args.target_rate_K_s, args.source_nodes)
        print(
            f"  controlled transient: {args.target_rate_K_s:.0f} K/s on {args.source_nodes} nodes "
            f"(= {power_W[chosen].sum():.1f} W total); peak first-step demand ~{args.target_rate_K_s*args.dt:.0f} K"
        )

    print("\nComputing exact reference (expm_multiply) ...")
    t0 = time.perf_counter()
    ref = reference_trajectory(C, L, T0, source, args.dt, args.steps)
    print(f"  reference done in {time.perf_counter()-t0:.1f}s")
    ref_peak_delta = float(np.max(np.abs(ref[0] - T0)))
    print(f"  exact peak dT on step 1 = {ref_peak_delta:.2f} K")

    print(f"\n{'cap':>5}{'substeps(step1)':>16}{'max_err_K':>12}{'final_err_K':>13}{'rel_err_%':>11}{'walltime_s':>12}")
    print("-" * 72)
    ref_span = float(np.max(np.abs(ref - T0[None, :])))
    for cap in args.caps:
        traj, substeps_used, walltime = run_stepper(
            C, L, T0, source, args.dt, args.steps, cap, args.target_delta_K, args.rtol
        )
        err = np.abs(traj - ref)
        max_err = float(np.max(err))
        final_err = float(np.max(err[-1]))
        rel = 100.0 * max_err / max(ref_span, 1e-30)
        print(f"{cap:>5}{substeps_used[0]:>16}{max_err:>12.3f}{final_err:>13.3f}{rel:>11.2f}{walltime:>12.1f}")

    print(
        "\nInterpretation: 'cap' is adaptive_max_substeps. Larger max_err_K means the "
        "substep cap smeared the transient. The error should shrink toward 0 as the cap rises."
    )


if __name__ == "__main__":
    main()
