"""Pick the LQR effort weight rho against your actual reduced model.

Since the gain is designed in discrete time at the real sample rate, rho no
longer has anything to do with stability -- every value below is stable by
construction. What rho actually buys is a trade between settling speed and
commanded power, and the binding constraint is the heater clamp: a rho that asks
for more watts than a heater can deliver does not make the loop faster, it just
makes it saturate, and a saturated heater is an open loop.

So the useful question is not "what is a good rho" in the abstract, it is "what
is the smallest rho whose command for a typical tracking error still fits under
u_max". This reports exactly that, using the reduced plant stored in the
controller artifact, so the watts are your plant's watts rather than a textbook's.

    python tools/sweep_lqr_effort.py graphs/no_mli_high_res/modal_controller.npz
    python tools/sweep_lqr_effort.py <artifact> --dt 8 --step-K 1.0 --u-max 30

Requires an artifact built after the discrete-LQR change (it needs A_r/B_r); an
older one has only the gain and cannot be re-swept.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import cont2discrete

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_visualizer.modal_reduction import discrete_lqr_gain, lqr_weights  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifact", help="path to modal_controller*.npz")
    parser.add_argument("--dt", type=float, default=None,
                        help="sample rate [s] (default: the artifact's design_dt_s)")
    parser.add_argument("--step-K", type=float, default=1.0,
                        help="tracking error to size the command against [K]")
    parser.add_argument("--u-max", type=float, default=30.0, help="per-heater clamp [W]")
    parser.add_argument("--rho", type=float, nargs="*", default=None, help="values to try")
    args = parser.parse_args()

    data = np.load(Path(args.artifact), allow_pickle=False)
    # C_r is required, not optional. It is tempting to reconstruct the output map
    # from Q = C_ctrl^T C_ctrl via its PSD square root, but that recovers C_ctrl
    # only up to a rotation -- fine for gain norms, wrong for the state that a
    # given tracking error corresponds to, which is exactly what sizes the watts.
    missing = [k for k in ("A_r", "B_r", "C_r", "monitor", "design_dt_s") if k not in data]
    if missing:
        print(f"This artifact predates the discrete-LQR change (missing {', '.join(missing)}).")
        print("Rebuild it first -- the stored gain alone cannot be re-swept.")
        return 1

    A = np.asarray(data["A_r"], dtype=float)
    B = np.asarray(data["B_r"], dtype=float)
    dt = float(args.dt) if args.dt else float(data["design_dt_s"])
    monitor = np.asarray(data["monitor"], dtype=bool)
    C_ctrl = np.asarray(data["C_r"], dtype=float)[~monitor]

    rhos = args.rho or [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    Ad, Bd, *_ = cont2discrete((A, B, np.zeros((1, A.shape[0])), np.zeros((1, B.shape[1]))), dt)
    # A reduced state consistent with a uniform tracking error of step_K.
    target = np.full(C_ctrl.shape[0], float(args.step_K))
    x_err = np.linalg.lstsq(C_ctrl, target, rcond=None)[0]

    print(f"{Path(args.artifact).name}: r={A.shape[0]}, {B.shape[1]} heaters, dt={dt:g} s")
    print(f"sizing commands against a {args.step_K:g} K error, clamp {args.u_max:g} W/heater")
    print()
    print(f"{'rho':>10} {'|K|':>10} {'max|pole|':>10} {'63% [s]':>9} "
          f"{'peak u [W]':>11} {'total u [W]':>12}")
    fits = []
    for rho in rhos:
        Q, R = lqr_weights(A, B, C_ctrl, rho)
        K = discrete_lqr_gain(A, B, Q, R, dt)
        pole = float(np.abs(np.linalg.eigvals(Ad - Bd @ K)).max())
        tau = -dt / np.log(pole) if 0.0 < pole < 1.0 else float("inf")
        u = K @ x_err
        peak, total = float(np.abs(u).max()), float(np.abs(u).sum())
        flag = ""
        if peak > args.u_max:
            flag = f"  saturates ({peak / args.u_max:.1f}x clamp)"
        else:
            fits.append((rho, tau, peak))
        tau_text = "inf" if not np.isfinite(tau) else f"{tau:,.0f}"
        print(f"{rho:>10.4g} {np.linalg.norm(K):>10.4g} {pole:>10.4f} {tau_text:>9} "
              f"{peak:>11.4g} {total:>12.4g}{flag}")

    print()
    if fits:
        best = min(fits, key=lambda item: item[1])   # fastest that still fits
        print(f"=> smallest rho whose command fits under the clamp: {best[0]:.4g} "
              f"(63% in ~{best[1]:,.0f} s, peak {best[2]:.3g} W)")
        print("   Lower rho is not faster in practice from here -- it only commands "
              "saturation, and a clipped heater is an open loop.")
    else:
        print("=> every rho tried saturates. Either the setpoint step is larger than this "
              "actuator set can service, or the heaters are undersized for it.")
    print("   rho is only speed-vs-power now; stability is handled by designing at dt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
