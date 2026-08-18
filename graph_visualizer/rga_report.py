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
# Below this span, a log axis is all empty decades -- see diagonal_axis_scale.
SYMLOG_MIN_EXTENT = 10.0


# Cost added to a negative RGA element when choosing a pairing. Bristol's rule is
# never to pair on a negative element, so it has to lose to any finite positive
# one; large but finite so a plant with no positive option still returns something
# rather than failing the assignment outright.
NEGATIVE_PAIRING_PENALTY = 1.0e6


def select_pairing(RGA: np.ndarray) -> np.ndarray:
    """Which heater each sensor would be paired with, if one insists on pairing.

    NOT the index diagonal, and this distinction is the whole reason the function
    exists. ``sensor_ids`` and ``heater_ids`` are each sorted by node id
    INDEPENDENTLY, so G[i, i] pairs the i-th sensor with the i-th heater purely by
    sort order -- an arbitrary partner with no physical relationship. Reading the
    diagonal as "the pairing" measures a scheme nobody would ever choose.

    On the 27x27 cryostat the index diagonal reads 24 of 27 NEGATIVE, i.e. "per-
    pair control is impossible", while the actual best pairing is a clean
    permutation whose elements are ALL positive (min 1.22, median 1.77). Opposite
    conclusions from the same matrix.

    Chooses by Bristol's rule -- pair on elements positive and closest to 1 --
    solved as an assignment so each heater is used at most once. Returns an array
    of heater column indices, one per sensor, or -1 where no heater was available
    (fewer heaters than sensors).

    Cost is |log lambda|, not |lambda - 1|. The two agree near 1 and disagree
    exactly where it matters: |lambda - 1| rates a pairing of 0.007 as barely
    worse than one of 2, so the assignment happily sold one channel down to
    almost no authority to shave a little off several others. |log lambda| is
    symmetric in the RATIO -- lambda and 1/lambda cost the same, which is the
    right symmetry for a gain ratio -- and diverges as lambda approaches zero.
    """
    from scipy.optimize import linear_sum_assignment

    R = np.asarray(RGA, dtype=float)
    if R.size == 0:
        return np.zeros(R.shape[0], dtype=int) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        cost = np.abs(np.log(np.where(R > 0.0, R, np.nan)))
    cost = np.where(np.isfinite(cost), cost, NEGATIVE_PAIRING_PENALTY)
    rows, cols = linear_sum_assignment(cost)
    pairing = np.full(R.shape[0], -1, dtype=int)
    pairing[rows] = cols
    return pairing


def rga_summary(
    G: np.ndarray,
    RGA: np.ndarray,
    sensor_ids: Sequence[int],
    heater_ids: Sequence[int],
) -> dict[str, Any]:
    """The numbers the figure states, in a form other tools can read.

    Reported at the SELECTED pairing (see :func:`select_pairing`), not at the
    index diagonal -- the diagonal pairs partners by sort order and answers a
    question about a scheme nobody proposed.
    """
    G = np.asarray(G, dtype=float)
    RGA = np.asarray(RGA, dtype=float)
    n_s, n_h = G.shape

    pairing = select_pairing(RGA)
    paired = pairing >= 0
    rows = np.nonzero(paired)[0]
    values = RGA[rows, pairing[rows]] if rows.size else np.array([])
    finite = values[np.isfinite(values)] if values.size else values

    # How much of everything reaching a sensor comes from its PAIRED heater. The
    # plain-magnitude companion to the RGA, and it has to use the same pairing or
    # the two halves of the figure describe different schemes.
    row_sums = np.abs(G).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.full(n_s, np.nan)
        share[rows] = np.abs(G[rows, pairing[rows]]) / np.maximum(row_sums[rows], 1e-300)

    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_sensors": int(n_s),
        "n_heaters": int(n_h),
        "n_paired": int(rows.size),
        # cond() is a decoration on this report, not its point, and its SVD raises
        # outright on a degenerate G. Losing the whole verdict because the
        # conditioning number could not be computed would be the wrong trade.
        "cond_G": _safe_cond(G),
        "max_abs_rga": float(np.max(np.abs(RGA))) if RGA.size else 0.0,
        "sensor_ids": [int(v) for v in sensor_ids],
        "heater_ids": [int(v) for v in heater_ids],
        "pairing": [
            {
                "sensor_id": int(sensor_ids[int(i)]),
                "heater_id": int(heater_ids[int(pairing[i])]),
                "rga": float(RGA[int(i), int(pairing[i])]),
                "paired_influence_fraction": float(share[int(i)]),
            }
            for i in rows
        ],
        # Whether the arbitrary sort-order pairing happens to coincide with the
        # chosen one. Recorded because a report that silently switched pairings
        # between runs would be impossible to compare.
        "pairing_is_index_diagonal": bool(
            n_s == n_h and rows.size == n_s and np.array_equal(pairing, np.arange(n_s))
        ),
    }
    if finite.size:
        summary.update(
            {
                "rga_paired": [float(v) for v in values],
                "rga_paired_min": float(finite.min()),
                "rga_paired_max": float(finite.max()),
                "rga_paired_median": float(np.median(finite)),
                "rga_paired_negative": int((finite < 0).sum()),
                # Bristol's workable band. Outside it a loop still functions but
                # fights its neighbours and is very sensitive to model error.
                "rga_paired_workable": int(((finite >= 0.5) & (finite <= 2.0)).sum()),
                "rga_paired_above_5": int((finite > 5.0).sum()),
            }
        )
        if n_s == n_h:
            # ||Lambda - I||_sum only means "distance from decoupled" when the
            # identity IS the pairing, so it is reported for the square case and
            # relative to the chosen permutation.
            target = np.zeros_like(RGA)
            target[rows, pairing[rows]] = 1.0
            summary["rga_number"] = float(np.abs(RGA - target).sum())
        else:
            summary["rga_number"] = None
        finite_share = share[np.isfinite(share)]
        if finite_share.size:
            summary["median_paired_influence_fraction"] = float(np.median(finite_share))
    else:
        summary.update(
            {
                "rga_paired": None,
                "rga_paired_min": None,
                "rga_paired_max": None,
                "rga_paired_median": None,
                "rga_paired_negative": None,
                "rga_paired_workable": None,
                "rga_paired_above_5": None,
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
    """Whether a pairing could be formed and scored at all.

    Needs at least one sensor assigned a heater with a finite relative gain.
    Fewer heaters than sensors leaves some sensors unpaired, which is a real
    result rather than a failure -- but with none paired there is nothing to say.
    """
    return bool(summary.get("n_paired")) and summary.get("rga_paired_negative") is not None


def verdict_lines(summary: dict[str, Any]) -> list[str]:
    """One-line-per-fact reading of the summary, for the figure and the status bar."""
    n_s, n_h = summary["n_sensors"], summary["n_heaters"]
    lines = [f"G is {n_s} controlled sensor(s) x {n_h} heater(s), cond(G) = {summary['cond_G']:.4g}"]
    if not has_pairing_verdict(summary):
        lines.append(
            f"No pairing could be scored: {n_h} heater(s) for {n_s} sensor(s), "
            "or no finite relative gain to choose on."
        )
        return lines
    paired, negative = summary["n_paired"], summary["rga_paired_negative"]
    lines.append(
        f"Best pairing ({paired} of {n_s} sensors): RGA median "
        f"{summary['rga_paired_median']:.3g}, range {summary['rga_paired_min']:+.3g} to "
        f"{summary['rga_paired_max']:+.3g}; {negative} negative, "
        f"{summary['rga_paired_workable']} inside the workable 0.5-2 band, "
        f"{summary['rga_paired_above_5']} above 5."
    )
    if not summary.get("pairing_is_index_diagonal"):
        lines.append(
            "Chosen by assignment, NOT the matrix diagonal: sensor and heater ids are sorted "
            "independently, so G's diagonal pairs partners by sort order rather than by physics."
        )
    fraction = summary.get("median_paired_influence_fraction")
    if fraction is not None:
        lines.append(
            f"Median share of what reaches a sensor that comes from its paired heater: "
            f"{fraction * 100.0:.3g}%  (even spreading over {n_h} heaters would give "
            f"{100.0 / max(n_h, 1):.3g}%)"
        )
    if negative:
        lines.append(
            f"{negative} pairing(s) are NEGATIVE: that loop's gain changes sign once the others "
            "close, so a PID tuned open-loop drives the WRONG WAY there."
        )
    elif summary["rga_paired_above_5"]:
        lines.append(
            f"No sign reversal, so per-pair SISO is not ruled out -- but "
            f"{summary['rga_paired_above_5']} loop(s) above 5 fight their neighbours hard and "
            "are very sensitive to model error."
        )
    else:
        lines.append("No sign reversal and no extreme interaction: per-pair SISO is viable here.")
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
    # The bar panel appears only when a pairing could actually be scored, on
    # exactly the same test the text uses -- otherwise the figure would show a bar
    # chart the footer says means nothing.
    scored = has_pairing_verdict(summary)
    n_s, n_h = RGA.shape
    pairs = summary.get("pairing") or []
    heater_index = {int(h): j for j, h in enumerate(heater_ids)}
    sensor_index = {int(s): i for i, s in enumerate(sensor_ids)}
    pair_rows = np.array([sensor_index[int(p["sensor_id"])] for p in pairs], dtype=int)
    pair_cols = np.array([heater_index[int(p["heater_id"])] for p in pairs], dtype=int)
    pair_values = np.array([float(p["rga"]) for p in pairs], dtype=float)

    header = title if not subtitle else f"{title}\n{subtitle}"
    width, height = (13.5, 7.2) if scored else (9.5, 7.0)
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
        1, 2 if scored else 1,
        width_ratios=[1.5, 1.0] if scored else None,
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
    ax_diag = fig.add_subplot(grid[0, 1]) if scored else None

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
    ax_map.set_title("RGA elements — squares mark the chosen pairing", fontsize=10)
    ax_map.set_xlabel("heater node id", fontsize=9)
    ax_map.set_ylabel("controlled sensor node id", fontsize=9)
    apply_channel_ticks(ax_map.set_xticks, ax_map.set_xticklabels, heater_ids, rotation=90)
    apply_channel_ticks(ax_map.set_yticks, ax_map.set_yticklabels, sensor_ids, rotation=0)
    if pair_cols.size:
        # Mark the cells the verdict is actually about. These are NOT the diagonal:
        # on this plant the chosen pairing is a permutation, and marking the
        # diagonal instead drew a line through cells nobody proposed pairing on.
        ax_map.plot(
            pair_cols, pair_rows,
            linestyle="none", marker="s", markersize=3.5,
            markerfacecolor="none", markeredgecolor="black", markeredgewidth=0.8,
        )

    if ax_diag is not None:
        values = np.full(n_s, np.nan)
        values[pair_rows] = pair_values
        colors = [
            "#c1272d" if v < 0 else ("#e08214" if v > 2.0 else "#2f6f9f")
            for v in np.nan_to_num(values)
        ]
        ax_diag.barh(np.arange(n_s), np.nan_to_num(values), color=colors, height=0.75)
        ax_diag.axvline(1.0, color="#2a8a4a", linewidth=1.2, linestyle="--", label="ideal (1)")
        ax_diag.axvspan(0.5, 2.0, color="#2a8a4a", alpha=0.10, label="workable 0.5-2")
        ax_diag.axvline(0.0, color="black", linewidth=0.8)
        scale, scale_kwargs = diagonal_axis_scale(values[np.isfinite(values)])
        ax_diag.set_xscale(scale, **scale_kwargs)
        ax_diag.set_ylim(n_s - 0.5, -0.5)          # match the heatmap's row order
        # Thinned on the same rule as the heatmap, so the two panels' row labels
        # stay in step instead of one of them silently going unreadable.
        apply_channel_ticks(ax_diag.set_yticks, ax_diag.set_yticklabels, sensor_ids, rotation=0)
        ax_diag.set_xlabel(f"RGA at the paired heater ({scale})", fontsize=9)
        ax_diag.set_title(
            f"Best pairing: {summary['rga_paired_negative']} negative, "
            f"{summary['rga_paired_workable']} of {summary['n_paired']} workable",
            fontsize=10,
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


def diagonal_axis_scale(values: Sequence[float]) -> tuple[str, dict[str, Any]]:
    """Linear unless the diagonal actually spans decades.

    symlog with linthresh=1 is right when one channel has a relative gain of 50
    and another 0.05. It is actively WRONG when the whole diagonal sits inside
    +/-1 -- which is precisely what a plant with no paired authority looks like.
    Every bar then lands in the linear region, and the axis spends its range on
    decades that hold no data, so the figure reads as "nothing here" when the
    finding is that every pairing is negative.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)] if finite.size else finite
    extent = float(np.max(np.abs(finite))) if finite.size else 0.0
    if extent <= SYMLOG_MIN_EXTENT:
        return "linear", {}
    return "symlog", {"linthresh": 1.0}


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
