"""Regenerate the artifact-derived SURF report figures (no eigensolve):

  fig1_hsv.png       Hankel singular values      <- graphs/CRYOSTAT_V2/modal_controller.npz
  fig2_step_overlay  full vs reduced step         <- docs/surf_report/step_data.npz
  fig3_lqr_gain.png  reduced-order LQR gain K     <- modal_controller.npz
  fig4_coupling.png  heater->sensor DC gain |G|   <- plots/cryostat_v2_modal_analysis/dc_gain.npz
                                                     (earlier well-conditioned build, cond ~37)

fig5_validation.png is produced by make_validation.py (runs the analytical suite).
The 469k->120-mode eigensolve behind step_data.npz is a one-off; step_data.npz is
cached here so this script stays fast and dependency-light.

Usage:  python docs/surf_report/make_figures.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
PURPLE, ORANGE, RED = "#6b46c1", "#dd6b20", "#c53030"

art = np.load(ROOT / "graphs" / "CRYOSTAT_V2" / "modal_controller.npz", allow_pickle=True)
hsv = art["hsv"]; r = int(art["r"]); K = art["K"]
hsv_n = hsv / hsv[0]

# fig1 — Hankel singular values
fig, ax = plt.subplots(figsize=(8, 5))
n = 60
ax.semilogy(np.arange(1, n + 1), np.clip(hsv_n[:n], 1e-18, None), "o-", ms=4, color=PURPLE)
ax.axvline(r, ls="--", color=ORANGE, label=f"retained order r = {r}")
ax.set_xlabel("balanced mode index"); ax.set_ylabel("Hankel singular value (normalized)")
ax.set_title("CRYOSTAT_V2 @ 55 K — Hankel singular values")
ax.grid(True, which="both", alpha=0.25); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "fig1_hsv.png", dpi=150); plt.close(fig)

# fig2 — full vs reduced step response (from cached step_data.npz)
sd_path = HERE / "step_data.npz"
if sd_path.exists():
    sd = np.load(sd_path)
    ts, yf, yr = sd["ts"], sd["yf"], sd["yr"]
    heater, sensor = int(sd["heater"]), int(sd["sensor"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    a1.plot(ts, yf, "-", lw=2.5, color=PURPLE, label="full model (120 modes)")
    a1.plot(ts, yr, "--", lw=1.8, color=ORANGE, label="reduced (r=30)")
    a1.set_ylabel("sensor temperature rise (K/W)")
    a1.set_title(f"CRYOSTAT_V2 @ 55 K — step response, heater {heater} → sensor {sensor}")
    a1.grid(alpha=0.25); a1.legend()
    a2.plot(ts, np.abs(yf - yr), "-", color=RED)
    a2.set_xlabel("time (s)"); a2.set_ylabel("|error| (K/W)"); a2.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "fig2_step_overlay.png", dpi=150); plt.close(fig)
else:
    print("note: step_data.npz missing; skipping fig2 (regenerate via rerun_validation)")

# fig3 — reduced-order LQR gain K
vmax = float(np.max(np.abs(K)))
fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(K, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
ax.set_xlabel(f"reduced state index ({K.shape[1]})"); ax.set_ylabel(f"heater index ({K.shape[0]})")
ax.set_title("CRYOSTAT_V2 — reduced-order LQR gain K  (u = −K x)")
fig.colorbar(im, label="gain (W per K-state)")
fig.tight_layout(); fig.savefig(OUT / "fig3_lqr_gain.png", dpi=150); plt.close(fig)

# fig4 — heater->sensor DC-gain coupling (earlier well-conditioned build)
dcp = ROOT / "plots" / "cryostat_v2_modal_analysis" / "dc_gain.npz"
if dcp.exists():
    G = np.load(dcp, allow_pickle=True)["G"]
    rowmax = np.max(np.abs(G), axis=1, keepdims=True); rowmax[rowmax == 0] = 1.0
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(np.abs(G) / rowmax, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_xlabel(f"heater index ({G.shape[1]})"); ax.set_ylabel(f"controlled sensor index ({G.shape[0]})")
    ax.set_title("CRYOSTAT_V2 @ 50 K — heater→sensor DC gain |G| (row-normalized)")
    fig.colorbar(im, label="|gain| / row-max")
    fig.tight_layout(); fig.savefig(OUT / "fig4_coupling.png", dpi=150); plt.close(fig)
    sv = np.linalg.svd(G, compute_uv=False)
    print(f"coupling: cond(G)={sv[0]/sv[-1]:.1f}, |G| range {np.abs(G).min():.3f}..{np.abs(G).max():.3f} K/W")
else:
    print("note: plots/cryostat_v2_modal_analysis/dc_gain.npz missing; skipping fig4")

print("done ->", OUT)
