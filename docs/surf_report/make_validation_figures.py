"""Presentation figures for the thermal-solver validation suite.

Runs the analytical validation experiments with the converged solver settings
(same configuration as make_validation.py), caches every sim/analytical series
to JSON so replotting is free, then renders the slide figures.

  python docs/surf_report/make_validation_figures.py            # run + plot
  python docs/surf_report/make_validation_figures.py --plot-only # replot cache
"""
import argparse
import json
import sys
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "plots" / "thermal_validation"
CACHE = OUT / "data"

from graph_visualizer.thermal_validation import (  # noqa: E402
    experiments_by_name,
    INSULATED_BLOCK, TWO_BLOCK_EXCHANGE, TWO_NODE_LUMPED, ONE_D_PRISM,
    DISTRIBUTED_ROD, RADIATION_COOLING, TDEP_HEATING, CRYO_REGIME,
    ENERGY_CONSERVATION,
)

SELECTED = [INSULATED_BLOCK, TWO_BLOCK_EXCHANGE, TWO_NODE_LUMPED, ONE_D_PRISM,
            DISTRIBUTED_ROD, RADIATION_COOLING, TDEP_HEATING, CRYO_REGIME,
            ENERGY_CONSERVATION]

# The prism default stops at 2 s -- long before copper's L^2/alpha ~ 86 s -- so only
# the near-face probe moves. Re-run it out to a full relaxation for the slide figure;
# same experiment and same tolerance, just a longer window.
PRISM_LONG = "prism_extended"


def prism_long_overrides(params):
    params.duration_s = 60.0
    params.dt_s = 0.01
    params.output_sample_interval_s = 0.25
    return params

# Short slide labels for the summary chart.
SHORT = {
    INSULATED_BLOCK: "Insulated block, constant heating",
    TWO_BLOCK_EXCHANGE: "Two-block exchange (contact)",
    TWO_NODE_LUMPED: "Two-node lumped conductance",
    ONE_D_PRISM: "1-D prism conduction (Fourier series)",
    DISTRIBUTED_ROD: "1-D distributed rod (mode decay)",
    RADIATION_COOLING: "Radiation cooling (lumped)",
    TDEP_HEATING: "Temperature-dependent heating",
    CRYO_REGIME: "Cryo regime: heater + radiation + cp(T)",
    ENERGY_CONSERVATION: "Global energy conservation",
}

# --- palette (dataviz reference instance, light surface) ---
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GOOD, WARNING = "#0ca30c", "#fab219"
SEQ4 = ["#86b6ef", "#3987e5", "#256abf", "#0d366b"]  # ordinal ramp, >=2:1 on light
SURFACE = "#fcfcfb"     # marker halo: reads as the page showing through
WARN_TEXT = "#8a5b00"   # amber dark enough to read as text on the light surface
BAND_ALPHA = 0.85       # tolerance-band fill against GRID

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "grid.color": GRID,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

# Set by --dark. Only save() and use_dark() read it; every figure function stays
# theme-agnostic and just uses the palette names.
DARK = False
STEM_SUFFIX = ""


def use_dark():
    """Repoint the palette at a dark surface and make the saved figures transparent.

    Transparent rather than dark-filled: a slide's background is rarely flat (this
    deck's is a gradient), so a figure that paints its own near-black sits on the
    slide as a visible rectangle. Nothing here is drawn ON the surface colour, so
    dropping it costs nothing -- except for marker halos, which need SOMETHING to
    stand in for the background; SURFACE becomes the deck's own near-black, which
    is right as long as the figure lands on a dark slide.

    Contrast is against that near-black, not against white: the light-surface ramp's
    dark end (#0d366b, 1.4:1 on #1a1a1a) would be invisible, so the ordinal ramp is
    rebuilt light-first. Grid and tolerance bands go the other way -- on a dark
    surface the light-surface grid (#e1e0d9) glares brighter than the data.
    """
    globals().update(
        INK="#f0efe9", INK2="#c9c7bf", MUTED="#96948c",
        GRID="#3a3a38", AXIS="#6b6862",
        BLUE="#5b9bf0", ORANGE="#ff8a5c",
        GOOD="#4ec94e", WARNING="#fac83c",
        # Light-first, because the light ramp's dark end (#0d366b) is 1.4:1 on
        # #1a1a1a. Kept clearly BLUE at the light end rather than running up to
        # near-white: #dce9fb read as ink rather than as the top of a blue family,
        # and against the near-white INK the top two steps were hard to separate.
        # Darkest still clears 3:1 on #1a1a1a.
        SEQ4=["#c3dcfb", "#8fb8f0", "#5f93e2", "#3d72c6"],
        SURFACE="#1a1a1a",
        WARN_TEXT="#f0b03c",
        # A light band at 0.85 would out-glare the curve it is meant to sit behind.
        BAND_ALPHA=0.55,
        DARK=True,
        STEM_SUFFIX="_dark",
    )
    plt.rcParams.update({
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "grid.color": GRID,
        # savefig(transparent=True) overrides these, but set them anyway so an
        # opaque save (interactive use, or transparent=False) still looks right
        # instead of putting dark ink on a near-white page.
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })

PREF = ("error", "rmse", "imbalance", "conservation")


def converged(params):
    """Solver settings tight enough that discretisation, not time stepping, dominates."""
    params.solver_adaptive_max_substeps = 512
    params.solver_adaptive_target_delta_K = 1.0e-3
    params.solver_rtol = 1.0e-11
    params.gpu_solver_enabled = False
    params.use_octree_pipeline = False
    return params


def slug(name):
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def run_and_cache(names, overrides=None, cache_as=None):
    byname = experiments_by_name()
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in names:
        exp = byname[name]
        try:
            params = converged(exp.default_parameters())
            if overrides:
                params = overrides(params)
            with TemporaryDirectory() as tmp:
                build = exp.build(params, Path(tmp))
                res = exp.run(build, params)
            payload = {
                "experiment_name": res.experiment_name,
                "status": res.status,
                "times_s": list(map(float, res.times_s)),
                "simulated": {k: list(map(float, v)) for k, v in res.simulated.items()},
                "analytical": {k: list(map(float, v)) for k, v in res.analytical.items()},
                "metrics": [m.to_dict() for m in res.metrics],
                "warnings": list(res.warnings),
                "material_properties": res.material_properties,
            }
            (CACHE / f"{slug(cache_as or name)}.json").write_text(json.dumps(payload))
            print(f"[ok]  {(cache_as or name):52s} {res.status}")
        except Exception as exc:  # keep going; a missing case just drops a panel
            print(f"[ERR] {name}: {exc}")
            traceback.print_exc()


def load(name):
    path = CACHE / f"{slug(name)}.json"
    return json.loads(path.read_text()) if path.exists() else None


def pick_metric(metrics):
    cand = [m for m in metrics if m.get("tolerance") not in (None, 0)]
    for key in PREF:
        for m in cand:
            if key in m["name"].lower():
                return m
    return cand[0] if cand else (metrics[0] if metrics else None)


def tidy(ax, xlabel=None, ylabel=None, title=None):
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11, pad=8)


def spread_labels(values, min_gap):
    """Nudge end-of-line label positions apart so close curves stay readable."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    for prev, cur in zip(order, order[1:]):
        if out[cur] - out[prev] < min_gap:
            out[cur] = out[prev] + min_gap
    return out


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{stem}{STEM_SUFFIX}"
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200, bbox_inches="tight",
                    transparent=DARK)
    plt.close(fig)
    print(f"[fig] {stem}.png{'  (transparent)' if DARK else ''}")


# ---------------------------------------------------------------- figure 1
def fig_summary():
    """Every case on one slide: error as a fraction of its acceptance tolerance."""
    rows = []
    for name in SELECTED:
        d = load(name)
        if not d:
            continue
        m = pick_metric(d["metrics"])
        if not m or not m.get("tolerance"):
            continue
        err, tol = abs(float(m["value"])), float(m["tolerance"])
        rows.append({
            "label": SHORT.get(name, name), "status": d["status"],
            "err": err, "tol": tol, "ratio": err / tol,
            "units": m.get("units", ""), "metric": m["name"],
        })
    if not rows:
        print("[skip] summary: no cached results")
        return

    FLOOR = 1e-10  # anything below this is round-off, not a resolvable error
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ypos = np.arange(len(rows))[::-1]
    for y, r in zip(ypos, rows):
        passing = r["status"].upper().startswith("PASS")
        color = GOOD if passing else WARNING
        x = max(r["ratio"], FLOOR)
        clipped = r["ratio"] < FLOOR
        ax.plot([FLOOR, x], [y, y], color=color, lw=3.0, alpha=0.55,
                solid_capstyle="round", zorder=2)
        ax.plot([x], [y], marker="<" if clipped else "o", ms=10 if clipped else 9,
                color=color, mec=SURFACE, mew=1.5, zorder=3)
        # Status word, then error and tolerance, each in its own x column so
        # nothing collides at any figure width.
        ax.text(3.6, y, ("PASS" if passing else "WARN"), va="center", ha="left",
                fontsize=9, fontweight="bold", color=GOOD if passing else WARN_TEXT,
                clip_on=False)
        ax.text(22.0, y, f"{r['err']:.1e} {r['units']}".strip(), va="center", ha="left",
                fontsize=9, color=INK, clip_on=False)
        ax.text(420.0, y, f"tol {r['tol']:.1e}", va="center", ha="left",
                fontsize=9, color=MUTED, clip_on=False)

    ax.axvline(1.0, color=INK2, ls="--", lw=1.4, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(FLOOR, 3.0)
    ax.set_ylim(-0.9, len(rows) - 0.25)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=10, color=INK)
    ax.set_xticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0])
    tidy(ax, xlabel="error / tolerance   (log scale — further left is better)")
    ax.grid(axis="y", visible=False)
    n_pass = sum(1 for r in rows if r["status"].upper().startswith("PASS"))
    headline = (f"Thermal solver vs. closed-form references: {n_pass}/{len(rows)} within tolerance"
                if n_pass == len(rows) else
                f"Thermal solver vs. closed-form references: {n_pass}/{len(rows)} within tolerance, "
                f"{len(rows) - n_pass} marginal")
    ax.set_title(headline, fontsize=12.5, pad=14, loc="left")
    ax.annotate("tolerance", xy=(1.0, -0.72), fontsize=9, color=INK2,
                ha="center", va="center")
    ax.annotate("← round-off floor", xy=(FLOOR, -0.72), xytext=(2, 0),
                textcoords="offset points", fontsize=8.5, color=MUTED,
                ha="left", va="center")
    save(fig, "val_1_summary")


# ---------------------------------------------------------------- figure 2
def fig_prism():
    """The strongest spatial test: transient conduction profile vs a Fourier series."""
    d = load(PRISM_LONG) or load(ONE_D_PRISM)
    if not d:
        print("[skip] prism: no cached result")
        return
    t = np.asarray(d["times_s"])
    keys = [k for k in d["simulated"] if k.startswith("x_over_L") and k in d["analytical"]]
    keys.sort()
    tol = next((float(m["tolerance"]) for m in d["metrics"]
                if m.get("tolerance") and "maximum absolute" in m["name"]), 3.0)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    step = max(1, len(t) // 14)
    worst = 0.0
    ramp = SEQ4[::-1]  # shallowest probe moves most -> give it the darkest step
    ends = []
    for i, k in enumerate(keys):
        c = ramp[i % len(ramp)]
        sim = np.asarray(d["simulated"][k])
        ana = np.asarray(d["analytical"][k])
        worst = max(worst, float(np.max(np.abs(sim - ana))))
        a1.plot(t, ana, "-", color=c, lw=2.0, zorder=2)
        a1.plot(t[::step], sim[::step], "o", ms=6, mfc="none", mec=c, mew=1.6, zorder=3)
        ends.append((float(ana[-1]), c, k.replace("x_over_L_", "x/L = ")))
        a2.plot(t, sim - ana, "-", color=c, lw=1.8, zorder=3)

    # Gap has to be a fraction of the axis range, not of the label spread, or
    # tightly-bunched curves still overlap.
    lo, hi = a1.get_ylim()
    for (yv, c, text), y in zip(ends, spread_labels([e[0] for e in ends], 0.05 * (hi - lo))):
        a1.annotate(text, xy=(t[-1], y), xytext=(8, 0), textcoords="offset points",
                    fontsize=9, color=c, va="center", ha="left", fontweight="bold")

    a1.legend(handles=[
        Line2D([], [], color=MUTED, lw=2.0, label="analytical (Fourier series)"),
        Line2D([], [], color=MUTED, lw=0, marker="o", ms=6, mfc="none", mew=1.6,
               label="simulated (graph solver)"),
    ], frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(0.97, 0.97))
    tidy(a1, "time (s)", "temperature (K)",
         f"Transient profile at 4 depths   (max |error| = {worst:.2f} K)")
    a1.set_xlim(t[0], t[-1] * 1.12)

    # Zoom to the error itself; the ±tol band is far wider, so say so in words
    # rather than flattening every curve onto the zero line.
    a2.axhline(0.0, color=AXIS, lw=1.0, zorder=2)
    tidy(a2, "time (s)", "simulated − analytical (K)",
         f"Error peaks at {worst:.2f} K — {tol / max(worst, 1e-12):.0f}× inside the ±{tol:g} K tolerance")
    a2.set_xlim(t[0], t[-1])
    lim = max(worst * 1.35, 1e-9)
    if tol <= lim:
        a2.axhspan(-tol, tol, color=GRID, alpha=BAND_ALPHA, zorder=1, label=f"tolerance  ±{tol:g} K")
        a2.legend(frameon=False, fontsize=9, loc="lower right")
    else:
        a2.annotate(f"tolerance band ±{tol:g} K extends well beyond this view",
                    xy=(0.5, 0.06), xycoords="axes fraction", fontsize=9,
                    color=MUTED, ha="center")
    a2.set_ylim(-lim, lim)

    fig.suptitle("1-D prism (copper): Dirichlet face + insulated end, 100-term series reference",
                 fontsize=12.5, y=1.03, x=0.005, ha="left")
    fig.tight_layout()
    save(fig, "val_2_prism_conduction")


# ---------------------------------------------------------------- figure 3
def fig_energy():
    """Conduction + radiation + heater at once: does the bookkeeping close?"""
    d = load(ENERGY_CONSERVATION)
    if not d:
        print("[skip] energy: no cached result")
        return
    t = np.asarray(d["times_s"])
    key = "stored_internal_energy_J"
    stored = np.asarray(d["simulated"][key])
    supplied = np.asarray(d["analytical"][key])
    resid = stored - supplied
    tol = next((float(m["tolerance"]) for m in d["metrics"]
                if "maximum energy imbalance" in m["name"]), 0.01 * np.max(np.abs(supplied)))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    a1.plot(t, supplied, "-", color=BLUE, lw=2.4, zorder=2)
    a1.plot(t, stored, "--", color=ORANGE, lw=1.6, zorder=3)
    a1.annotate("∫ net power in", xy=(t[-1], supplied[-1]), xytext=(-8, -16),
                textcoords="offset points", color=BLUE, fontsize=9,
                ha="right", fontweight="bold")
    a1.annotate("stored internal energy", xy=(t[int(len(t) * 0.55)], stored[int(len(t) * 0.55)]),
                xytext=(-4, 14), textcoords="offset points", color=ORANGE, fontsize=9,
                ha="right", fontweight="bold")
    tidy(a1, "time (s)", "energy (J)", "Heater + ambient radiation + conduction, all active")

    worst = float(np.max(np.abs(resid)))
    a2.plot(t, resid, "-", color=BLUE, lw=1.8, zorder=3)
    a2.axhline(0.0, color=AXIS, lw=1.0, zorder=2)
    tidy(a2, "time (s)", "stored − supplied (J)",
         f"Imbalance peaks at {worst:.1e} J — {tol / max(worst, 1e-30):.0f}× inside the ±{tol:.0f} J tolerance")
    lim = max(worst * 1.35, 1e-12)
    if tol <= lim:
        a2.axhspan(-tol, tol, color=GRID, alpha=BAND_ALPHA, zorder=1, label=f"tolerance  ±{tol:.1f} J")
        a2.legend(frameon=False, fontsize=9, loc="upper left")
    else:
        a2.annotate(f"tolerance band ±{tol:.0f} J extends well beyond this view",
                    xy=(0.5, 0.06), xycoords="axes fraction", fontsize=9,
                    color=MUTED, ha="center")
    a2.set_ylim(-lim, lim)
    fig.suptitle("Global energy conservation audit", fontsize=12.5, y=1.03, x=0.005, ha="left")
    fig.tight_layout()
    save(fig, "val_3_energy_conservation")


# ------------------------------------------------------- figure 6 (big three)
def fig_cryo_solo():
    """Cryo regime on its own, in the same two-panel layout as the prism and
    energy figures, so the three carry a slide sequence."""
    d = load(CRYO_REGIME)
    if not d:
        print("[skip] cryo solo: no cached result")
        return
    t = np.asarray(d["times_s"])
    keys = [k for k in d["simulated"] if k in d["analytical"]]
    tol = next((float(m["tolerance"]) for m in d["metrics"]
                if m.get("tolerance") and "maximum absolute" in m["name"]), 0.5)
    labels = {"average_temperature_K": "volume average",
              "hot_end_temperature_K": "hot end"}
    colors = {"average_temperature_K": BLUE, "hot_end_temperature_K": ORANGE}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    step = max(1, len(t) // 16)
    worst = 0.0
    ends = []
    for k in keys:
        c = colors.get(k, BLUE)
        sim, ana = np.asarray(d["simulated"][k]), np.asarray(d["analytical"][k])
        worst = max(worst, float(np.max(np.abs(sim - ana))))
        a1.plot(t, ana, "-", color=c, lw=2.4, zorder=2)
        a1.plot(t[::step], sim[::step], "o", ms=6, mfc="none", mec=c, mew=1.6, zorder=3)
        ends.append((float(ana[-1]), c, labels.get(k, k)))
        a2.plot(t, sim - ana, "-", color=c, lw=1.8, zorder=3)

    lo, hi = a1.get_ylim()
    for (_, c, text), y in zip(ends, spread_labels([e[0] for e in ends], 0.06 * (hi - lo))):
        a1.annotate(text, xy=(t[-1], y), xytext=(8, 0), textcoords="offset points",
                    fontsize=9, color=c, va="center", ha="left", fontweight="bold")
    a1.legend(handles=[
        Line2D([], [], color=MUTED, lw=2.4, label="reference (LSODA, rtol 1e-9)"),
        Line2D([], [], color=MUTED, lw=0, marker="o", ms=6, mfc="none", mew=1.6,
               label="simulated (implicit solver)"),
    ], frameon=False, fontsize=9, loc="lower right")
    tidy(a1, "time (s)", "temperature (K)",
         "40 K start, heater + radiation + cp(T) all active")
    a1.set_xlim(t[0], t[-1] * 1.12)

    a2.axhline(0.0, color=AXIS, lw=1.0, zorder=2)
    tidy(a2, "time (s)", "simulated − reference (K)",
         f"Error peaks at {worst:.1e} K — {tol / max(worst, 1e-12):.0f}× inside the ±{tol:g} K tolerance")
    a2.set_xlim(t[0], t[-1])
    lim = max(worst * 1.35, 1e-12)
    if tol <= lim:
        a2.axhspan(-tol, tol, color=GRID, alpha=BAND_ALPHA, zorder=1, label=f"tolerance  ±{tol:g} K")
        a2.legend(frameon=False, fontsize=9, loc="lower right")
    else:
        a2.annotate(f"tolerance band ±{tol:g} K extends well beyond this view",
                    xy=(0.5, 0.06), xycoords="axes fraction", fontsize=9,
                    color=MUTED, ha="center")
    a2.set_ylim(-lim, lim)
    fig.suptitle("Cryo regime: nonlinear coupled solve against an independent integrator",
                 fontsize=12.5, y=1.03, x=0.005, ha="left")
    fig.tight_layout()
    save(fig, "val_6_cryo_regime")


# ---------------------------------------------------------------- figure 4
def fig_cryo_and_radiation():
    """The regime the flight hardware actually runs in, restyled to match."""
    panels = [(CRYO_REGIME, "Cryo regime: heater + radiation + cp(T)"),
              (RADIATION_COOLING, "Radiation cooling (lumped)")]
    have = [(load(n), title) for n, title in panels]
    have = [(d, title) for d, title in have if d]
    if not have:
        print("[skip] cryo/radiation: no cached results")
        return
    fig, axes = plt.subplots(1, len(have), figsize=(6 * len(have), 4.8), squeeze=False)
    for ax, (d, title) in zip(axes[0], have):
        t = np.asarray(d["times_s"])
        worst = 0.0
        first = True
        step = max(1, len(t) // 16)
        for k in d["simulated"]:
            if k not in d["analytical"]:
                continue
            sim, ana = np.asarray(d["simulated"][k]), np.asarray(d["analytical"][k])
            if sim.shape != ana.shape or sim.size != t.size:
                continue
            worst = max(worst, float(np.max(np.abs(sim - ana))))
            ax.plot(t, ana, "-", color=BLUE, lw=2.4, zorder=2,
                    label="analytical" if first else None)
            ax.plot(t[::step], sim[::step], "o", ms=6, mfc="none", mec=ORANGE, mew=1.6,
                    zorder=3, label="simulated" if first else None)
            first = False
        tidy(ax, "time (s)", "temperature (K)", f"{title}\nmax |error| = {worst:.1e} K")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Coupled nonlinear cases: radiation and temperature-dependent cp",
                 fontsize=12.5, y=1.04, x=0.005, ha="left")
    fig.tight_layout()
    save(fig, "val_4_cryo_and_radiation")


# ---------------------------------------------------------------- figure 5
def fig_two_block():
    """The one case that lands over tolerance -- show it rather than hide it."""
    d = load(TWO_BLOCK_EXCHANGE)
    if not d:
        print("[skip] two-block: no cached result")
        return
    t = np.asarray(d["times_s"])
    # Both curves converge on the equilibrium line, so push their labels apart.
    pairs = [("hot_average_temperature_K", ORANGE, "hot block", 14),
             ("cold_average_temperature_K", BLUE, "cold block", -20)]
    m = pick_metric(d["metrics"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    tol = float(m["tolerance"]) if m and m.get("tolerance") else 0.05
    worst = 0.0
    for key, color, label, dy in pairs:
        if key not in d["simulated"] or key not in d["analytical"]:
            continue
        sim, ana = np.asarray(d["simulated"][key]), np.asarray(d["analytical"][key])
        worst = max(worst, float(np.max(np.abs(sim - ana))))
        step = max(1, len(t) // 16)
        a1.plot(t, ana, "-", color=color, lw=2.4, zorder=2)
        a1.plot(t[::step], sim[::step], "o", ms=6, mfc="none", mec=color, mew=1.6, zorder=3)
        a1.annotate(label, xy=(t[-1], ana[-1]), xytext=(-6, dy), textcoords="offset points",
                    color=color, fontsize=9, ha="right", fontweight="bold")
        a2.plot(t, sim - ana, "-", color=color, lw=1.8, zorder=3)
    eq = d["analytical"].get("equilibrium_temperature_K")
    if eq:
        val = float(np.asarray(eq).reshape(-1)[0])
        a1.axhline(val, color=MUTED, ls=":", lw=1.2)
        a1.annotate(f"equilibrium {val:.1f} K", xy=(t[int(len(t) * 0.35)], val),
                    xytext=(0, 6), textcoords="offset points", color=MUTED, fontsize=9)
    a1.legend(handles=[
        Line2D([], [], color=MUTED, lw=2.4, label="analytical"),
        Line2D([], [], color=MUTED, lw=0, marker="o", ms=6, mfc="none", mew=1.6,
               label="simulated"),
    ], frameon=False, fontsize=9, loc="lower right")
    tidy(a1, "time (s)", "temperature (K)",
         f"Contact-coupled pair relaxing to equilibrium  (max |error| = {worst:.3f} K)")
    a2.axhspan(-tol, tol, color=GRID, alpha=BAND_ALPHA, zorder=1, label=f"tolerance  ±{tol:g} K")
    a2.axhline(0.0, color=AXIS, lw=1.0, zorder=2)
    tidy(a2, "time (s)", "simulated − analytical (K)",
         "Early-transient overshoot: the only case over tolerance")
    a2.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Two-block thermal exchange across a contact conductance",
                 fontsize=12.5, y=1.03, x=0.005, ha="left")
    fig.tight_layout()
    save(fig, "val_5_two_block_exchange")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--dark", action="store_true",
                    help="render for a dark slide with a transparent background; "
                         "writes *_dark.png/pdf beside the light versions")
    args = ap.parse_args()
    if args.dark:
        use_dark()
    if not args.plot_only:
        run_and_cache(SELECTED)
        run_and_cache([ONE_D_PRISM], overrides=prism_long_overrides, cache_as=PRISM_LONG)
    fig_summary()
    fig_prism()
    fig_energy()
    fig_cryo_and_radiation()
    fig_two_block()
    fig_cryo_solo()


if __name__ == "__main__":
    main()
