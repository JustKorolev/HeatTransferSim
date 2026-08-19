"""Slide figures for closed-loop sensor tracking, from a headless run's timeseries.

  python docs/surf_report/make_controller_figures.py <run_dir> [--dark]

Reads timeseries.csv + sensors.csv only, so it works on a run that is still going.
Splits controlled from monitor sensors using the `controlled` column of sensors.csv:
a monitor sensor is not in the loop, so folding it into a tracking-error figure
overstates the error by an amount that has nothing to do with the controller.

The palette and the dark theme come from make_validation_figures, so both figure
sets stay one visual system rather than two that drift apart.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_validation_figures as style  # noqa: E402  (palette + use_dark + save)

TARGET_K = 0.05     # the tracking target these runs are aimed at


def target_band(ax, horizontal=True):
    """Draw the +/-TARGET_K band so it reads at any vertical scale.

    Fill alone was invisible where the band is a thin strip of the view (it is 18%
    of the height in the steady-state panel and under 2% in the transient one), and
    brightening the fill enough to fix that made it out-glare the curves sitting on
    it. Dotted edges carry the boundary instead.
    """
    span = ax.axhspan if horizontal else ax.axvspan
    line = ax.axhline if horizontal else ax.axvline
    span(-TARGET_K, TARGET_K, color=style.GRID, alpha=0.9, zorder=1)
    for edge in (-TARGET_K, TARGET_K):
        line(edge, color=style.INK2, ls=":", lw=0.9, alpha=0.7, zorder=2)


def load(run_dir: Path):
    manifest = {r["series"]: r for r in csv.DictReader(open(run_dir / "sensors.csv"))}
    rows = list(csv.DictReader(open(run_dir / "timeseries.csv")))
    if not rows:
        raise SystemExit(f"{run_dir/'timeseries.csv'} has no samples")
    hours = np.array([float(r["time_s"]) for r in rows]) / 3600.0
    err_cols = [c for c in rows[0] if c.endswith("_err_K")]
    controlled = [c for c in err_cols if manifest[c[:-6]]["controlled"] == "True"]
    monitor = [c for c in err_cols if manifest[c[:-6]]["controlled"] != "True"]
    E = np.array([[float(r[c]) for c in controlled] for r in rows], dtype=float)
    M = (np.array([[float(r[c]) for c in monitor] for r in rows], dtype=float)
         if monitor else np.zeros((len(rows), 0)))
    return hours, E, M, controlled, monitor


def worst_channel(E, labels):
    """The channel with the largest |error| at the end -- the one that sets the rms."""
    j = int(np.argmax(np.abs(E[-1])))
    return j, labels[j][:-6].replace("_", " ")


def rms(a, axis=None):
    return np.sqrt(np.mean(np.asarray(a) ** 2, axis=axis))


# ------------------------------------------------------------------ figure 1
def fig_transient_and_steady(hours, E, labels, out):
    """Transient beside a steady-state zoom, both linear in error.

    Two panels rather than one: the approach spans 5 K and the steady state 0.05 K,
    a factor of 100, so a single linear axis renders the settled behaviour as a flat
    line on the zero gridline. Log-|error| would show both at once but destroys the
    sign, and which side of setpoint a channel sits on is the whole point at steady
    state (see fig_convergence for the log view of the magnitude).
    """
    j61, name = worst_channel(E, labels)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 4.3),
                                 gridspec_kw={"width_ratios": [1.25, 1.0]})
    settle_h = 4.0
    for ax, (lo, hi), title in (
        (a1, (0.0, min(settle_h, hours[-1])), "Approach: 5 K initial offset closed"),
        (a2, (max(0.0, hours[-1] - 5.0), hours[-1]), "Steady state, last 5 h (note the axis)"),
    ):
        m = (hours >= lo) & (hours <= hi)
        for j in range(E.shape[1]):
            if j == j61:
                continue
            ax.plot(hours[m], E[m, j], "-", color=style.BLUE, lw=0.9, alpha=0.55, zorder=2)
        ax.plot(hours[m], E[m, j61], "-", color=style.ORANGE, lw=1.8, zorder=4)
        ax.axhline(0.0, color=style.AXIS, lw=1.0, zorder=3)
        # Only on the steady panel: at a 5 K scale the band is under 2% of the view,
        # so drawing it there adds a line at zero and no information.
        if ax is a2:
            target_band(ax)
        # Label the axis once: the two panels share the quantity and differ only in
        # scale, and repeating it costs width the steady panel needs.
        style.tidy(ax, "time (h)",
                   "sensor error  (measured − setpoint)  [K]" if ax is a1 else None, title)
        ax.set_xlim(lo, hi)
    a2.legend(handles=[
        Line2D([], [], color=style.BLUE, lw=1.4, alpha=0.8,
               label=f"{E.shape[1]-1} controlled sensors"),
        Line2D([], [], color=style.ORANGE, lw=1.8, label=name),
        Line2D([], [], color=style.GRID, lw=7, alpha=style.BAND_ALPHA,
               label=f"±{TARGET_K:g} K target"),
    ], frameon=False, fontsize=9, loc="center right")
    fig.suptitle("Closed-loop tracking: 25 controlled sensors", fontsize=12.5, y=1.03,
                 x=0.005, ha="left")
    fig.tight_layout()
    style.save(fig, out)


# ------------------------------------------------------------------ figure 2
def fig_convergence(hours, E, M, labels, out):
    """Error MAGNITUDE against time, log on both axes.

    This is the figure that needs log scales, and the only one. The envelope spans
    4.99 K down to 0.02 K -- 2.4 decades -- so on a linear axis everything after the
    first hour is pinned to zero and the settled floor, which is the result, is
    invisible. Log time as well because the approach is exponential: a decaying
    exponential is a straight line here, so a change of slope is a change of regime
    rather than an artefact of where the eye lands.
    """
    j61, name = worst_channel(E, labels)
    others = np.delete(E, j61, axis=1)
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    m = hours > 0
    ax.plot(hours[m], rms(E[m], axis=1), "-", color=style.BLUE, lw=2.2, zorder=4)
    ax.plot(hours[m], rms(others[m], axis=1), "--", color=style.GOOD, lw=2.0, zorder=5)
    ax.plot(hours[m], np.abs(E[m, j61]), "-", color=style.ORANGE, lw=1.6, zorder=3)
    if M.shape[1]:
        ax.plot(hours[m], rms(M[m], axis=1), ":", color=style.WARNING, lw=1.6, zorder=2)
    ax.axhline(TARGET_K, color=style.INK2, ls="--", lw=1.3, zorder=6)
    ax.annotate(f"{TARGET_K:g} K target", xy=(hours[1], TARGET_K), xytext=(2, 5),
                textcoords="offset points", ha="left", fontsize=9, color=style.INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Explicit, with a decade of headroom below: autoscaling clipped the 24-channel
    # rms where it dips to ~0.02 K, cutting off the best part of the result.
    floor = min(rms(np.delete(E, j61, axis=1)[m], axis=1).min(), TARGET_K)
    ax.set_ylim(floor / 2.5, np.abs(E).max() * 1.6)
    handles = [
        Line2D([], [], color=style.BLUE, lw=2.2, label="rms over all 25"),
        Line2D([], [], color=style.GOOD, lw=2.0, ls="--",
               label=f"rms over 24 (excl. {name})"),
        Line2D([], [], color=style.ORANGE, lw=1.6, label=f"{name}  |e|"),
    ]
    if M.shape[1]:
        handles.append(Line2D([], [], color=style.WARNING, lw=1.6, ls=":",
                              label=f"rms over {M.shape[1]} monitor (not controlled)"))
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower left",
              bbox_to_anchor=(0.0, 0.0))
    style.tidy(ax, "time (h, log)", "|error|  [K, log]",
               "Convergence: one channel sets the floor, the other 24 are 2.7x below it")
    fig.tight_layout()
    style.save(fig, out)


# ------------------------------------------------------------------ figure 3
def fig_final_distribution(hours, E, labels, out):
    """Where every channel ended up, worst-first, against the target band.

    A time series of 25 overlapping curves cannot answer "how many are inside
    tolerance"; this can, and it names the outlier instead of leaving it as an
    anonymous line. Averaged over the last hour so a single noisy sample does not
    decide a channel's reported position.
    """
    window = hours >= hours[-1] - 1.0
    final = E[window].mean(axis=0)
    spread = E[window].std(axis=0)
    order = np.argsort(final)
    names = [labels[j][:-6].replace("sensor_", "s") for j in order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    target_band(ax, horizontal=False)
    ax.axvline(0.0, color=style.AXIS, lw=1.2, zorder=3)
    inside = np.abs(final[order]) <= TARGET_K
    for k, j in enumerate(order):
        colour = style.BLUE if inside[k] else style.ORANGE
        ax.plot([0.0, final[j]], [k, k], "-", color=colour, lw=2.0, alpha=0.55, zorder=2)
        ax.errorbar(final[j], k, xerr=spread[j], fmt="o", ms=6, color=colour,
                    ecolor=colour, elinewidth=1.2, capsize=2.5, mec=style.SURFACE,
                    mew=1.2, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9, color=style.INK2)
    ax.set_ylim(-0.8, len(order) - 0.2)
    n_in = int(inside.sum())
    style.tidy(ax, f"mean error over the final hour  [K]   "
                   f"(bars: 1σ; dotted band ±{TARGET_K:g} K)", None,
               f"{n_in} of {len(order)} controlled sensors inside ±{TARGET_K:g} K")
    fig.tight_layout()
    style.save(fig, out)


# ------------------------------------------------------------------ figure 4
def fig_outlier_vs_pack(hours, E, labels, out):
    """The outlier against the band the rest of the pack occupies.

    Figure 2 shows it is the floor; this shows WHY the rms will not improve without
    it. The pack band is min..max of the other channels, so the gap between the band
    and the outlier is the part of the error no amount of settling removes.
    """
    j61, name = worst_channel(E, labels)
    others = np.delete(E, j61, axis=1)
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.fill_between(hours, others.min(axis=1), others.max(axis=1),
                    color=style.BLUE, alpha=0.30, lw=0, zorder=2,
                    label=f"{others.shape[1]} other controlled sensors (min–max)")
    ax.plot(hours, others.mean(axis=1), "-", color=style.BLUE, lw=1.6, zorder=3,
            label="their mean")
    ax.plot(hours, E[:, j61], "-", color=style.ORANGE, lw=2.2, zorder=4, label=name)
    ax.axhline(0.0, color=style.AXIS, lw=1.0, zorder=3)
    target_band(ax)
    # Start after the approach: clipping a -5 K dive at -1.2 K left vertical stubs at
    # the left edge that read as data. Figure 1 covers the transient.
    ax.set_xlim(1.0, hours[-1])
    ax.set_ylim(-1.0, 0.35)
    # The gap is still opening, slowly -- worth stating on the figure, because the
    # eye reads two flat lines as "settled" and this one is not.
    early = hours >= 5.0
    gap_now = float(others[-1].mean() - E[-1, j61])
    gap_then = float(others[early][0].mean() - E[early, j61][0])
    ax.annotate(f"gap {gap_then:.2f} K at 5 h  →  {gap_now:.2f} K at {hours[-1]:.0f} h",
                xy=(hours[-1], E[-1, j61]), xytext=(-8, -18),
                textcoords="offset points", ha="right", fontsize=9, color=style.ORANGE)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style.tidy(ax, "time (h)", "sensor error  [K]",
               f"{name} sits {abs(gap_now):.2f} K below the pack and is not closing")
    fig.tight_layout()
    style.save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--dark", action="store_true",
                    help="render for a dark slide on a transparent background")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: <run_dir>/plots)")
    args = ap.parse_args()
    if args.dark:
        style.use_dark()
    style.OUT = args.out or (args.run_dir / "plots")

    hours, E, M, labels, monitor = load(args.run_dir)
    j61, name = worst_channel(E, labels)
    others = np.delete(E, j61, axis=1)
    print(f"{len(hours)} samples over {hours[-1]:.2f} h; "
          f"{E.shape[1]} controlled, {M.shape[1]} monitor")
    print(f"rms all {E.shape[1]}: {rms(E[-1]):.4f} K | "
          f"rms without {name}: {rms(others[-1]):.4f} K | {name}: {E[-1, j61]:+.4f} K")

    fig_transient_and_steady(hours, E, labels, "ctl_1_transient_and_steady")
    fig_convergence(hours, E, M, labels, "ctl_2_convergence")
    fig_final_distribution(hours, E, labels, "ctl_3_final_distribution")
    fig_outlier_vs_pack(hours, E, labels, "ctl_4_outlier_vs_pack")


if __name__ == "__main__":
    main()
