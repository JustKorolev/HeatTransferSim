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


# Stamped on every figure whenever channels are dropped. An exclusion that is not
# on the face of the figure is one that gets quoted without its caveat: the run this
# was built for held 25 channels to 0.098 K, and dropping one reports 0.037 K for a
# configuration that was not run -- the dropped channel was in the loop, so its error
# was pulling on the allocation for all the others.
EXCLUSION_NOTE = ""


def apply_exclusions(E, labels, spec):
    """Drop channels named in ``spec`` (sensor ids, series names, or "worst").

    Cleaner to do here than in each figure, and it sets EXCLUSION_NOTE once so no
    figure can forget to disclose it.
    """
    global EXCLUSION_NOTE
    if not spec:
        return E, labels
    wanted = {t.strip() for t in spec.split(",") if t.strip()}
    keep, dropped = [], []
    worst = int(np.argmax(np.abs(E[-1]))) if "worst" in wanted else None
    for j, col in enumerate(labels):
        series = col[:-6]                       # "sensor_61_err_K" -> "sensor_61"
        token = series.replace("sensor_", "")
        if j == worst or series in wanted or token in wanted:
            dropped.append(series)
        else:
            keep.append(j)
    if not dropped:
        raise SystemExit(f"--exclude {spec!r} matched no channel of {len(labels)}")
    EXCLUSION_NOTE = (f"{len(dropped)} of {len(labels)} controlled channels excluded "
                      f"from these figures and from the rms")
    print(f"excluded {len(dropped)}: {', '.join(dropped)}")
    return E[:, keep], [labels[j] for j in keep]


def stamp(fig):
    """Put the exclusion note on the figure itself, bottom-left, small and permanent."""
    if EXCLUSION_NOTE:
        fig.text(0.005, -0.02, EXCLUSION_NOTE, fontsize=8.5, color=style.WARN_TEXT,
                 ha="left", va="top")


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
    scalars = {
        k: np.array([float(r[k]) for r in rows], dtype=float)
        for k in ("power_in_W", "power_out_W", "net_W", "energy_drift_rel")
        if k in rows[0]
    }
    return hours, E, M, controlled, monitor, scalars


# Deliberately generic. Naming one sensor on a slide invites "what is wrong with
# that one" when the point is the distribution, and the identity means nothing to an
# audience. The channel stays in every figure and in every quoted rms -- it is
# relabelled, not removed, because dropping it while still quoting the 25-channel rms
# would attribute its error to the other 24.
OUTLIER_LABEL = "widest channel"


def worst_channel(E, labels):
    """Index of the channel with the largest |error| at the end -- it sets the rms."""
    return int(np.argmax(np.abs(E[-1]))), OUTLIER_LABEL


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
        highlight = j61 if not EXCLUSION_NOTE else None
        for j in range(E.shape[1]):
            if j == highlight:
                continue
            ax.plot(hours[m], E[m, j], "-", color=style.BLUE, lw=0.9, alpha=0.55, zorder=2)
        if highlight is not None:
            ax.plot(hours[m], E[m, highlight], "-", color=style.ORANGE, lw=1.8, zorder=4)
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
    handles = [Line2D([], [], color=style.BLUE, lw=1.4, alpha=0.8,
                      label=f"{E.shape[1] - (0 if EXCLUSION_NOTE else 1)} controlled sensors")]
    if not EXCLUSION_NOTE:
        handles.append(Line2D([], [], color=style.ORANGE, lw=1.8, label=name))
    handles.append(Line2D([], [], color=style.GRID, lw=7, alpha=style.BAND_ALPHA,
                          label=f"±{TARGET_K:g} K target"))
    a2.legend(handles=handles, frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(f"Closed-loop tracking: {E.shape[1]} controlled sensors",
                 fontsize=12.5, y=1.03, x=0.005, ha="left")
    fig.tight_layout()
    stamp(fig)
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
    if not EXCLUSION_NOTE:
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
    handles = [Line2D([], [], color=style.BLUE, lw=2.2,
                      label=f"rms over all {E.shape[1]}")]
    if not EXCLUSION_NOTE:
        handles += [
            Line2D([], [], color=style.GOOD, lw=2.0, ls="--",
                   label=f"rms over {E.shape[1]-1} (excl. the widest)"),
            Line2D([], [], color=style.ORANGE, lw=1.6, label=f"{name}  |e|"),
        ]
    if M.shape[1]:
        handles.append(Line2D([], [], color=style.WARNING, lw=1.6, ls=":",
                              label=f"rms over {M.shape[1]} monitor (not controlled)"))
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower left",
              bbox_to_anchor=(0.0, 0.0))
    settled = float(rms(E[-1]))
    if EXCLUSION_NOTE:
        headline = (f"Convergence: rms over {E.shape[1]} channels settles at "
                    f"{settled:.3f} K, {TARGET_K/settled:.1f}x inside target")
    else:
        headline = (f"Convergence: one channel sets the {settled:.3f} K floor, the other "
                    f"{E.shape[1]-1} are {settled/rms(others[-1]):.1f}x below it")
    style.tidy(ax, "time (h, log)", "|error|  [K, log]", headline)
    fig.tight_layout()
    stamp(fig)
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
    # Rank, not identity: the sensor names carry no meaning for a reader and naming
    # the outlier is the thing this figure most invites.
    names = [str(k + 1) for k in range(len(order))]
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
    ax.set_ylabel("controlled channel, ranked by final error", color=style.INK2)
    ax.set_ylim(-0.8, len(order) - 0.2)
    n_in = int(inside.sum())
    style.tidy(ax, f"mean error over the final hour  [K]   "
                   f"(bars: 1σ; dotted band ±{TARGET_K:g} K)", None,
               f"{n_in} of {len(order)} controlled sensors inside ±{TARGET_K:g} K")
    fig.tight_layout()
    stamp(fig)
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
               f"One channel sits {abs(gap_now):.2f} K below the pack and is not closing")
    fig.tight_layout()
    stamp(fig)
    style.save(fig, out)


# ------------------------------------------------------------------ figure 5
def fig_power_balance(hours, sc, out):
    """Heat in against heat out, and their difference, both linear.

    Linear on both panels: in and out span 22-51 W, one order of magnitude, so a log
    axis would buy nothing and cost the reader the ability to read a wattage off the
    page. The difference gets its OWN panel rather than a third line -- it settles at
    -0.18 W against 22 W of throughput, so on the shared axis it is the zero line.
    """
    p_in, p_out, net = sc["power_in_W"], sc["power_out_W"], sc["net_W"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.8, 6.0), sharex=True,
                                 gridspec_kw={"height_ratios": [1.55, 1.0]})
    a1.plot(hours, p_in, "-", color=style.ORANGE, lw=1.8, zorder=4, label="heater power in")
    a1.plot(hours, p_out, "-", color=style.BLUE, lw=1.8, zorder=3, label="cryocooler lift out")
    a1.annotate(f"{p_in[-1]:.1f} W in / {p_out[-1]:.1f} W out at {hours[-1]:.0f} h",
                xy=(hours[-1], p_out[-1]), xytext=(-6, 40), textcoords="offset points",
                ha="right", fontsize=9, color=style.INK2)
    a1.legend(frameon=False, fontsize=9, loc="upper right")
    style.tidy(a1, None, "power  [W]", "Closing the 5 K offset costs a 51 W peak, settling to 23 W")
    a2.fill_between(hours, 0.0, net, where=net >= 0, color=style.ORANGE, alpha=0.35, lw=0)
    a2.fill_between(hours, 0.0, net, where=net < 0, color=style.BLUE, alpha=0.35, lw=0)
    a2.plot(hours, net, "-", color=style.INK2, lw=1.4, zorder=4)
    a2.axhline(0.0, color=style.AXIS, lw=1.2, zorder=3)
    top = max(2.0, float(np.percentile(net, 98)) * 1.2)
    a2.set_ylim(min(-1.6, net.min() * 1.2), top)
    if net.max() > top:
        a2.annotate(f"peak {net.max():+.1f} W, clipped", xy=(hours[int(net.argmax())], top),
                    xytext=(8, -12), textcoords="offset points", fontsize=9,
                    color=style.MUTED)
    style.tidy(a2, "time (h)", "net = in − out  [W]",
               f"Net settles to {net[-1]:+.2f} W: the structure is still equilibrating")
    fig.tight_layout()
    stamp(fig)
    style.save(fig, out)


# ------------------------------------------------------------------ figure 6
def fig_energy_drift(hours, sc, out):
    """First-law residual per step, log in the residual.

    The second figure that needs a log scale. The residual runs 4.3e-07 to 3.0e-02 --
    4.8 decades -- so linear would show the startup steps and render the entire
    settled run as a flat line on zero, which is exactly the part that has to be
    checked. The thresholds are drawn because the number is meaningless without them:
    it is a discretisation residual, not an error, and "small" only means anything
    against what would be acted on.

    The quantity is DIMENSIONLESS but it is not a percentage throughout, which is the
    easiest thing to get wrong about it. The denominator is
    max(|net_W|, |dU/dt|, 1.0), so once the plant nears equilibrium the 1 W floor
    wins and the ratio stops being a fraction OF anything -- it becomes an absolute
    residual in watts. On this run the floor is active for 79% of the samples, from
    1.9 h onward, so the shaded region is milliwatts and only the unshaded part left
    of it is a true fraction of the power.
    """
    drift = sc["energy_drift_rel"]
    ok = np.isfinite(drift) & (np.asarray(drift) > 0.0)
    fig, ax = plt.subplots(figsize=(9.8, 4.4))
    # Mark where the normalisation changes meaning.
    net = sc.get("net_W")
    if net is not None:
        floored = np.abs(net) < 1.0
        start = hours[np.argmax(floored & (hours > 0.5))] if floored.any() else None
        if start is not None and start < hours[-1]:
            ax.axvspan(start, hours[-1], color=style.GRID, alpha=0.55, lw=0, zorder=0)
            ax.annotate("denominator floored at 1 W from here:\nread this region as watts, "
                        "not as a fraction",
                        xy=(start, 1.0), xytext=(6, -2), textcoords="offset points",
                        fontsize=8.5, color=style.MUTED, va="top")
    ax.plot(hours[ok], drift[ok], "-", color=style.BLUE, lw=1.0, alpha=0.85, zorder=3)
    for level, colour, label in ((0.10, style.WARNING, "0.10  logged as a warning"),
                                 (0.90, style.ORANGE, "0.90  aborts the run")):
        ax.axhline(level, color=colour, ls="--", lw=1.3, zorder=4)
        ax.annotate(label, xy=(hours[ok][0], level), xytext=(3, 4),
                    textcoords="offset points", fontsize=9, color=colour)
    median = float(np.median(drift[ok]))
    ax.axhline(median, color=style.GOOD, ls=":", lw=1.4, zorder=4)
    ax.annotate(f"median {median:.1e}", xy=(hours[ok][0], median), xytext=(3, -14),
                textcoords="offset points", fontsize=9, color=style.GOOD)
    ax.set_yscale("log")
    ax.set_ylim(drift[ok].min() / 3.0, 2.0)
    peak = float(drift[ok].max())
    style.tidy(ax, "time (h)",
               "|net power − dU/dt| / max(|net|, |dU/dt|, 1 W)   [log, dimensionless]",
               f"Energy conservation: peak {peak:.1e} ({0.10/peak:.0f}x inside the "
               f"warning), median {median:.1e} ({0.10/median:.0f}x)")
    fig.tight_layout()
    stamp(fig)
    style.save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--dark", action="store_true",
                    help="render for a dark slide on a transparent background")
    ap.add_argument("--exclude", default="",
                    help="comma-separated sensor ids / series names to drop from the "
                         "figures AND the rms, or 'worst' for the widest channel. "
                         "Every figure is stamped with the exclusion.")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: <run_dir>/plots)")
    args = ap.parse_args()
    if args.dark:
        style.use_dark()
    style.OUT = args.out or (args.run_dir / "plots")

    hours, E, M, labels, monitor, scalars = load(args.run_dir)
    E, labels = apply_exclusions(E, labels, args.exclude)
    j61, name = worst_channel(E, labels)
    others = np.delete(E, j61, axis=1)
    print(f"{len(hours)} samples over {hours[-1]:.2f} h; "
          f"{E.shape[1]} controlled, {M.shape[1]} monitor")
    print(f"rms all {E.shape[1]}: {rms(E[-1]):.4f} K | "
          f"rms excluding the widest: {rms(others[-1]):.4f} K | "
          f"widest ({labels[j61][:-6]}): {E[-1, j61]:+.4f} K")

    fig_transient_and_steady(hours, E, labels, "ctl_1_transient_and_steady")
    fig_convergence(hours, E, M, labels, "ctl_2_convergence")
    fig_final_distribution(hours, E, labels, "ctl_3_final_distribution")
    if not args.exclude:
        fig_outlier_vs_pack(hours, E, labels, "ctl_4_outlier_vs_pack")
    else:
        print("skipping ctl_4 (outlier vs pack): its subject was excluded")
    if {"power_in_W", "power_out_W", "net_W"} <= set(scalars):
        fig_power_balance(hours, scalars, "ctl_5_power_balance")
    if "energy_drift_rel" in scalars:
        fig_energy_drift(hours, scalars, "ctl_6_energy_drift")


if __name__ == "__main__":
    main()
