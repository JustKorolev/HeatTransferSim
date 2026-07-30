"""Run analytical thermal-validation experiments (converged solver); collect a
clean error-vs-tolerance PASS table + cryo-regime / radiation-cooling sim-vs-
analytical transient overlays for the SURF report."""
import json, sys, traceback
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:/Users/andre/Documents/2025-2026 Academic Year/SURF/HeatTransferSim")
sys.path.insert(0, str(ROOT))
FIGDST = ROOT / "docs" / "surf_report" / "figures"
FIGDST.mkdir(parents=True, exist_ok=True)

from graph_visualizer.thermal_validation import (
    experiments_by_name,
    INSULATED_BLOCK, TWO_BLOCK_EXCHANGE, TWO_NODE_LUMPED, ONE_D_PRISM,
    DISTRIBUTED_ROD, RADIATION_COOLING, TDEP_HEATING, CRYO_REGIME,
    ENERGY_CONSERVATION,
)


def converged(params):
    params.solver_adaptive_max_substeps = 512
    params.solver_adaptive_target_delta_K = 1.0e-3
    params.solver_rtol = 1.0e-11
    params.gpu_solver_enabled = False
    params.use_octree_pipeline = False
    return params


SELECTED = [INSULATED_BLOCK, TWO_BLOCK_EXCHANGE, TWO_NODE_LUMPED, ONE_D_PRISM,
            DISTRIBUTED_ROD, RADIATION_COOLING, TDEP_HEATING, CRYO_REGIME,
            ENERGY_CONSERVATION]

PREF = ("error", "rmse", "imbalance", "conservation")
byname = experiments_by_name()


def run(name):
    exp = byname[name]
    params = converged(exp.default_parameters())
    with TemporaryDirectory() as tmp:
        build = exp.build(params, Path(tmp))
        res = exp.run(build, params)
    return res


def pick_metric(res):
    cand = [m for m in res.metrics if m.tolerance not in (None, 0)]
    for key in PREF:
        for m in cand:
            if key in m.name.lower():
                return m
    return cand[0] if cand else (res.metrics[0] if res.metrics else None)


rows = []
for name in SELECTED:
    try:
        res = run(name)
        m = pick_metric(res)
        rows.append({
            "experiment": name, "status": res.status,
            "metric": m.name if m else "(none)",
            "value": float(m.value) if m else None,
            "tolerance": float(m.tolerance) if (m and m.tolerance) else None,
            "units": m.units if m else "",
            "warnings": list(res.warnings)[:2],
        })
        print(f"[ok] {name:52s} {res.status}")
    except Exception as e:
        rows.append({"experiment": name, "status": f"ERROR: {e}"})
        print(f"[ERR] {name}: {e}"); traceback.print_exc()

json.dump(rows, open(FIGDST / "validation_summary.json", "w"), indent=2)

# ---- Figure: cryo-regime + radiation-cooling transient overlays ----
PURPLE, ORANGE = "#6b46c1", "#dd6b20"


def overlay(ax, res, title):
    t = np.asarray(res.times_s)
    maxerr = 0.0
    for k in res.simulated:
        sim = np.asarray(res.simulated[k]); ana = np.asarray(res.analytical[k])
        ax.plot(t, ana, "-", color=PURPLE, lw=2.4, label="analytical" if k == list(res.simulated)[0] else None)
        ax.plot(t, sim, "--", color=ORANGE, lw=1.4, label="simulated" if k == list(res.simulated)[0] else None)
        maxerr = max(maxerr, float(np.max(np.abs(sim - ana))))
    ax.set_title(f"{title}  (max |error| = {maxerr:.1e} K)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("temperature (K)")
    ax.grid(alpha=0.25); ax.legend()
    return maxerr


try:
    rc = run(RADIATION_COOLING)
    cr = run(CRYO_REGIME)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    e1 = overlay(a1, cr, "Cryo regime: heater + radiation + cp(T)")
    e2 = overlay(a2, rc, "Radiation cooling (lumped)")
    fig.suptitle("Thermal-solver validation against analytical references", y=1.02)
    fig.tight_layout(); fig.savefig(FIGDST / "fig5_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] fig5_validation.png cryo_err={e1:.2e} rad_err={e2:.2e}")
except Exception as e:
    print("[fig ERR]", e); traceback.print_exc()

print("\n=== SUMMARY ===")
for r in rows:
    if r.get("value") is None:
        print(f"{r['experiment']:52s} {r['status']}")
    else:
        print(f"{r['experiment']:52s} {r['status']:8s} {r['metric']}: "
              f"{r['value']:.3e} {r['units']} (tol {r['tolerance']:.1e})")
