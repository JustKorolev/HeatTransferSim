"""Relative Gain Array of a saved DC gain matrix, as a figure and a table.

G says how strongly each heater reaches each sensor. It does NOT say whether the
plant can be controlled one heater per sensor, and on a dense G it is actively
misleading about it: every entry can be large and positive while the pairing is
still unusable. That question needs the INVERSE, which is what the RGA supplies.

    Lambda = G .* pinv(G).T

Element (i, j) is heater j's gain on sensor i open-loop, divided by its gain with
every other loop closed. Rows and columns each sum to 1. Reading the diagonal:

    ~1        the other loops do not disturb this pairing -- pair them
    0.5-2     interaction present but workable
    ~0        this heater has no authority over this sensor
    >>1       the other loops fight this one; very sensitive to model error
    < 0       the gain REVERSES once the other loops close -- never pair

That last case is why this graph exists. It is the evidence that decides between
per-pair SISO and decoupling through G, and it cannot be read off G by eye.

This module owns the RGA itself -- the summary, the verdict wording and the
figure. :mod:`plant_report` calls it as one section of the full plant analysis
and decides where the files land.
"""

from __future__ import annotations

import csv
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

RGA_PNG = "rga.png"
RGA_CSV = "rga.csv"

# Beyond this many channels the tick labels stop being readable at any font size,
# so they are thinned rather than drawn on top of each other.
MAX_TICK_LABELS = 30
FOOTER_FONTSIZE = 7.5


def rga_summary(
    G: np.ndarray,
    RGA: np.ndarray,
    sensor_ids: Sequence[int],
    heater_ids: Sequence[int],
) -> dict[str, Any]:
    """The numbers the figure states, in a form other tools can read.

    The diagonal is reported ONLY for a square matrix. With unequal counts there
    is no one-heater-per-sensor pairing for it to describe, and a number that
    reads like a pairing verdict when none exists is worse than no number --
    the same rule ``exact_dc_gain`` applies when it logs the diagonal.
    """
    G = np.asarray(G, dtype=float)
    RGA = np.asarray(RGA, dtype=float)
    square = G.shape[0] == G.shape[1] and RGA.shape[0] == RGA.shape[1]

    # How much of a heater's total steady influence lands on its NOMINALLY paired
    # sensor. This is the plain-magnitude companion to the RGA: the RGA says the
    # pairing inverts, this says how little there was to invert in the first place.
    row_sums = np.abs(G).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        paired_fraction = (
            np.where(row_sums > 0.0, np.abs(np.diag(G)) / np.maximum(row_sums, 1e-300), np.nan)
            if square
            else np.array([])
        )
    paired_fraction = paired_fraction[np.isfinite(paired_fraction)] if paired_fraction.size else paired_fraction

    diag = np.diag(RGA) if square else np.array([])
    finite_diag = diag[np.isfinite(diag)] if diag.size else diag

    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_sensors": int(G.shape[0]),
        "n_heaters": int(G.shape[1]),
        "square": bool(square),
        # cond() is a decoration on this report, not its point, and its SVD raises
        # outright on a degenerate G. Losing the whole verdict because the
        # conditioning number could not be computed would be the wrong trade.
        "cond_G": _safe_cond(G),
        "max_abs_rga": float(np.max(np.abs(RGA))) if RGA.size else 0.0,
        "sensor_ids": [int(v) for v in sensor_ids],
        "heater_ids": [int(v) for v in heater_ids],
    }
    if square and finite_diag.size:
        summary.update(
            {
                "rga_diag": [float(v) for v in diag],
                "rga_diag_min": float(finite_diag.min()),
                "rga_diag_max": float(finite_diag.max()),
                "rga_diag_negative": int((finite_diag < 0).sum()),
                # ||Lambda - I||_sum: the standard scalar measure of how far this
                # plant is from being diagonally controllable. 0 is perfect.
                "rga_number": float(np.abs(RGA - np.eye(RGA.shape[0])).sum()),
            }
        )
        if paired_fraction.size:
            summary["median_paired_influence_fraction"] = float(np.median(paired_fraction))
    else:
        summary.update(
            {
                "rga_diag": None,
                "rga_diag_min": None,
                "rga_diag_max": None,
                "rga_diag_negative": None,
                "rga_number": None,
            }
        )
    return summary


def _safe_cond(G: np.ndarray) -> float:
    if G.size == 0:
        return float("nan")
    try:
        return float(np.linalg.cond(G))
    except np.linalg.LinAlgError:
        return float("nan")


def has_pairing_verdict(summary: dict[str, Any]) -> bool:
    """Whether the diagonal is allowed to be stated as a pairing answer.

    Square is necessary but not sufficient: a diagonal that came out non-finite
    has no verdict in it either, and every caller that formats the numbers has to
    agree on that or one of them will print "None of 27 negative".
    """
    return bool(summary.get("square")) and summary.get("rga_diag_negative") is not None


def verdict_lines(summary: dict[str, Any]) -> list[str]:
    """One-line-per-fact reading of the summary, for the figure and the status bar."""
    n_s, n_h = summary["n_sensors"], summary["n_heaters"]
    lines = [f"G is {n_s} controlled sensor(s) x {n_h} heater(s), cond(G) = {summary['cond_G']:.4g}"]
    if not has_pairing_verdict(summary):
        reason = (
            f"{n_s}x{n_h} is not square, so there is no one-heater-per-sensor pairing "
            "for it to describe"
            if not summary["square"]
            else "its diagonal is not finite"
        )
        lines.append(f"RGA diagonal not reported: {reason}.")
        return lines
    negative = summary["rga_diag_negative"]
    lines.append(
        f"RGA diagonal: {negative} of {n_s} NEGATIVE (min {summary['rga_diag_min']:+.4g}, "
        f"max {summary['rga_diag_max']:+.4g}); RGA number ||L-I||_sum = {summary['rga_number']:.4g}"
    )
    fraction = summary.get("median_paired_influence_fraction")
    if fraction is not None:
        lines.append(
            f"Median share of a heater's steady influence landing on its own sensor: "
            f"{fraction * 100.0:.2g}%"
        )
    if negative:
        lines.append(
            "A negative diagonal entry means that pairing's gain changes sign once the other "
            "loops close, so a PID tuned open-loop drives the WRONG WAY. This is why the "
            "scheme decouples through G rather than pairing."
        )
    else:
        lines.append("No sign reversal on the diagonal: per-pair SISO is not ruled out here.")
    return lines


def render_rga_figure(
    path: Path,
    RGA: np.ndarray,
    sensor_ids: Sequence[int],
    heater_ids: Sequence[int],
    summary: dict[str, Any],
    *,
    title: str = "Relative Gain Array",
    subtitle: str = "",
) -> Path:
    """Draw the heatmap (+ the diagonal, when there is one) and save it as a PNG.

    Uses the Agg canvas directly rather than pyplot. This is called from the GUI
    process, where matplotlib is already live on a Qt backend for the 2D graph
    view -- ``matplotlib.use("Agg")`` there would switch the backend out from
    under that widget. The object API renders off-screen without touching global
    state, and needs no plt.close() to avoid leaking figures.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import SymLogNorm
    from matplotlib.figure import Figure

    RGA = np.asarray(RGA, dtype=float)
    # The diagonal panel appears only when the diagonal is a pairing verdict, on
    # exactly the same test the text uses -- otherwise the figure would show a bar
    # chart the footer says means nothing.
    square = has_pairing_verdict(summary)
    n_s, n_h = RGA.shape

    header = title if not subtitle else f"{title}\n{subtitle}"
    width, height = (13.5, 7.2) if square else (9.5, 7.0)
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)

    # Explicit margins, NOT tight_layout: the colorbar is an axes tight_layout
    # refuses to place ("Axes that are not compatible with tight_layout"), and when
    # it gives up it leaves the footer sitting on top of the tick labels. The
    # bottom margin is sized from the footer that actually has to fit -- the text
    # is wrapped first, so a long verdict grows the margin instead of running off
    # the canvas.
    footer_lines = _wrap_footer(verdict_lines(summary), width, FOOTER_FONTSIZE)
    footer_in = len(footer_lines) * FOOTER_FONTSIZE * 1.5 / 72.0
    grid = fig.add_gridspec(
        1, 2 if square else 1,
        width_ratios=[1.5, 1.0] if square else None,
        wspace=0.42,
        left=1.05 / width,
        # The colorbar hangs off the right of the last column, and its rotated label
        # is outside that -- without this the label is simply clipped away.
        right=1.0 - 0.62 / width,
        # Room for the rotated node-id ticks and the x label, then the footer.
        bottom=(footer_in + 1.0) / height,
        top=1.0 - 0.72 / height,
    )
    ax_map = fig.add_subplot(grid[0, 0])
    ax_diag = fig.add_subplot(grid[0, 1]) if square else None

    # Symmetric log about zero: RGA entries here span several decades (the plant is
    # ill-conditioned, and pinv amplifies its weak directions), so a linear map
    # would render every entry but the largest as the same shade of nothing. The
    # colour carries the SIGN, which is the verdict; the log carries the magnitude.
    finite = RGA[np.isfinite(RGA)]
    extent = float(np.max(np.abs(finite))) if finite.size else 1.0
    extent = extent if extent > 0.0 else 1.0
    image = ax_map.imshow(
        RGA,
        cmap="RdBu_r",
        norm=SymLogNorm(linthresh=1.0, linscale=1.0, vmin=-extent, vmax=extent, base=10),
        aspect="auto",
        interpolation="nearest",
    )
    bar = fig.colorbar(image, ax=ax_map, fraction=0.046, pad=0.03)
    bar.set_label("RGA element (symlog)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    ax_map.set_title("RGA elements — blue is negative, i.e. sign reversal", fontsize=10)
    ax_map.set_xlabel("heater node id", fontsize=9)
    ax_map.set_ylabel("controlled sensor node id", fontsize=9)
    apply_channel_ticks(ax_map.set_xticks, ax_map.set_xticklabels, heater_ids, rotation=90)
    apply_channel_ticks(ax_map.set_yticks, ax_map.set_yticklabels, sensor_ids, rotation=0)
    if square:
        # Mark the pairing the diagonal is a verdict on, so "the diagonal" is a
        # visible object in the heatmap rather than something to count cells for.
        ax_map.plot(
            np.arange(n_h), np.arange(n_s),
            linestyle="none", marker="s", markersize=3.0,
            markerfacecolor="none", markeredgecolor="black", markeredgewidth=0.6,
        )

    if ax_diag is not None:
        diag = np.diag(RGA)
        colors = ["#c1272d" if value < 0 else "#2f6f9f" for value in diag]
        positions = np.arange(n_s)
        ax_diag.barh(positions, diag, color=colors, height=0.75)
        ax_diag.axvline(1.0, color="#2a8a4a", linewidth=1.2, linestyle="--", label="ideal (1)")
        ax_diag.axvline(0.0, color="black", linewidth=0.8)
        # Symlog for the same reason the heatmap uses it: one channel with a huge
        # relative gain would otherwise flatten all the others to invisible stubs.
        ax_diag.set_xscale("symlog", linthresh=1.0)
        ax_diag.set_ylim(n_s - 0.5, -0.5)          # match the heatmap's row order
        # Thinned on the same rule as the heatmap, so the two panels' row labels
        # stay in step instead of one of them silently going unreadable.
        apply_channel_ticks(ax_diag.set_yticks, ax_diag.set_yticklabels, sensor_ids, rotation=0)
        ax_diag.set_xlabel("diagonal RGA element (symlog)", fontsize=9)
        ax_diag.set_title(
            f"Pairing verdict: {summary['rga_diag_negative']} of {n_s} negative", fontsize=10
        )
        ax_diag.grid(True, axis="x", alpha=0.3)
        ax_diag.legend(fontsize=7, loc="lower right")
        ax_diag.tick_params(labelsize=7)

    fig.suptitle(header, fontsize=11)
    fig.text(
        0.008, 0.008, "\n".join(footer_lines),
        fontsize=FOOTER_FONTSIZE, va="bottom", ha="left", family="monospace",
    )

    # Write-then-rename, as the run's own plots do: a kill mid-save must not leave
    # a truncated PNG where a readable one used to be, and savefig cannot infer
    # the format from a ".png.tmp" name.
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + ".tmp")
    fig.savefig(staged, dpi=130, format="png")
    os.replace(staged, target)
    return target


def _wrap_footer(lines: Sequence[str], width_in: float, fontsize: float) -> list[str]:
    """Wrap the verdict to the figure's width, in monospace character units.

    0.6 em per character is DejaVu Sans Mono's advance width, so this estimates
    the usable column count rather than guessing a constant that is wrong on one
    of the two figure sizes.
    """
    columns = max(48, int((width_in - 0.2) * 72.0 / (fontsize * 0.6)))
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, columns) or [""])
    return wrapped


def apply_channel_ticks(set_ticks, set_labels, ids: Sequence[int], *, rotation: int) -> None:
    """Label every channel when they fit, every k-th when they do not."""
    count = len(ids)
    step = 1 if count <= MAX_TICK_LABELS else int(np.ceil(count / MAX_TICK_LABELS))
    positions = np.arange(0, count, step)
    set_ticks(positions)
    set_labels(
        [str(int(ids[i])) for i in positions],
        rotation=rotation,
        fontsize=6 if count > MAX_TICK_LABELS else 7,
    )


def write_rga_csv(
    path: Path,
    sensor_ids: Sequence[int],
    heater_ids: Sequence[int],
    RGA: np.ndarray,
) -> Path:
    """Long-form, matching gain_matrix.csv so the two join on (sensor, heater)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + ".tmp")
    with staged.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sensor_id", "heater_id", "rga"])
        writer.writeheader()
        for i, sensor_id in enumerate(sensor_ids):
            for j, heater_id in enumerate(heater_ids):
                writer.writerow(
                    {
                        "sensor_id": int(sensor_id),
                        "heater_id": int(heater_id),
                        "rga": float(RGA[i, j]),
                    }
                )
    os.replace(staged, target)
    return target
