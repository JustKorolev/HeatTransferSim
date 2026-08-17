"""Figures, tables and a written report for :mod:`plant_analysis`.

Split from the numbers so the statistics can be computed and asserted on without
matplotlib anywhere near them, and so a figure that fails to draw costs a figure
rather than the whole report.

Every figure renders through the Agg canvas directly rather than pyplot: this
runs inside the GUI process, where matplotlib is already live on a Qt backend for
the 2D graph view, and ``matplotlib.use("Agg")`` there would switch the backend
out from under that widget.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .plant_analysis import (
    ANALYSIS_DIRNAME,
    REPORT_JSON,
    REPORT_MD,
    PlantAnalysis,
    compute_plant_analysis,
)
from .rga_report import (
    RGA_CSV,
    RGA_PNG,
    apply_channel_ticks,
    diagonal_axis_scale,
    has_pairing_verdict,
    render_rga_figure,
    write_rga_csv,
)

NEGATIVE = "#c1272d"
POSITIVE = "#2f6f9f"
ACCENT = "#e08214"


# ------------------------------------------------------------------- utilities
def _figure(width: float, height: float):
    """A constrained-layout Agg figure. Constrained rather than tight because
    several of these carry a colorbar, which tight_layout refuses to place and
    then silently gives up on, leaving labels stacked on top of each other."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def _save(fig, path: Path) -> Path:
    """Write-then-rename, as the run's own plots do, so a kill mid-save cannot
    leave a truncated PNG where a readable one used to be."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + ".tmp")
    fig.savefig(staged, dpi=130, format="png")
    os.replace(staged, target)
    return target


def _bar_channels(ax, values, ids, *, color, log: bool = False, rotation: int = 90) -> None:
    values = np.asarray(values, dtype=float)
    positions = np.arange(values.size)
    colors = color if isinstance(color, str) else list(color)
    ax.bar(positions, values, color=colors, width=0.8)
    if log:
        ax.set_yscale("log")
    apply_channel_ticks(ax.set_xticks, ax.set_xticklabels, ids, rotation=rotation)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(labelsize=7)


# --------------------------------------------------------------------- figures
def figure_gain_matrix(path: Path, G: np.ndarray, stats: dict[str, Any]) -> Path:
    """G itself, with its marginals.

    The picture is the argument: a plant of independent zones is a bright diagonal
    on a dark field, and this one is neither. The marginals say the same thing
    numerically -- if every column sum is about the same, no heater is special.
    """
    sensor_ids, heater_ids = stats["sensor_ids"], stats["heater_ids"]
    fig = _figure(12.0, 7.0)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 3.2], width_ratios=[3.2, 1.0])
    ax_top = fig.add_subplot(grid[0, 0])
    ax_map = fig.add_subplot(grid[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(grid[1, 1], sharey=ax_map)

    image = ax_map.imshow(G, cmap="magma", aspect="auto", interpolation="nearest")
    fig.colorbar(image, ax=ax_right, fraction=0.3, pad=0.08, label="K/W")
    ax_map.set_xlabel("heater node id", fontsize=9)
    ax_map.set_ylabel("controlled sensor node id", fontsize=9)
    apply_channel_ticks(ax_map.set_xticks, ax_map.set_xticklabels, heater_ids, rotation=90)
    apply_channel_ticks(ax_map.set_yticks, ax_map.set_yticklabels, sensor_ids, rotation=0)
    ax_map.tick_params(labelsize=7)

    ax_top.bar(np.arange(G.shape[1]), np.abs(G).sum(axis=0), color=POSITIVE, width=0.8)
    ax_top.set_ylabel("column sum\nK/W", fontsize=8)
    ax_top.tick_params(labelbottom=False, labelsize=7)
    ax_top.grid(True, axis="y", alpha=0.3)
    ax_top.set_title("Heater authority per column", fontsize=9)

    ax_right.barh(np.arange(G.shape[0]), np.abs(G).sum(axis=1), color=ACCENT, height=0.8)
    ax_right.set_xlabel("row sum K/W", fontsize=8)
    ax_right.tick_params(labelleft=False, labelsize=7)
    ax_right.grid(True, axis="x", alpha=0.3)

    structure = stats["gain_structure"]
    fig.suptitle(
        f"DC gain G — every entry between {structure['min_abs_K_per_W']:.3g} and "
        f"{structure['max_abs_K_per_W']:.3g} K/W "
        f"(dynamic range {structure['dynamic_range']:.1f}x)"
        if structure["dynamic_range"]
        else "DC gain G",
        fontsize=11,
    )
    return _save(fig, path)


def figure_spectrum(path: Path, stats: dict[str, Any]) -> Path:
    """The singular values and what fraction of the plant each one carries."""
    spectrum = stats["spectrum"]
    sigma = np.asarray(spectrum["singular_values"], dtype=float)
    cumulative = np.asarray(spectrum["cumulative_energy_fraction"], dtype=float)
    positions = np.arange(sigma.size)

    fig = _figure(10.0, 5.2)
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(positions, sigma, color=POSITIVE, width=0.75)
    ax.set_yscale("log")
    ax.set_xlabel("singular value index", fontsize=9)
    ax.set_ylabel("sigma  [K/W]", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(labelsize=8)
    for tol in (0.1, 0.01):
        ax.axhline(tol * sigma[0], color="grey", linewidth=0.8, linestyle=":")
        ax.text(
            positions[-1], tol * sigma[0], f" {tol:g}·σ₁",
            fontsize=7, va="bottom", ha="right", color="grey",
        )

    twin = ax.twinx()
    twin.plot(positions, cumulative * 100.0, color=NEGATIVE, marker="o", markersize=3, linewidth=1.2)
    twin.set_ylabel("cumulative energy  [%]", fontsize=9, color=NEGATIVE)
    twin.tick_params(labelsize=8, colors=NEGATIVE)
    twin.set_ylim(0, 105)

    fig.suptitle(
        f"Singular spectrum — σ₁ carries {spectrum['top_energy_fraction'] * 100:.4g}% of the plant, "
        f"cond(G) = {spectrum['condition_number']:.4g}, "
        f"{spectrum['directions_for_90pct']} direction(s) reach 90%",
        fontsize=11,
    )
    return _save(fig, path)


def figure_mode_shapes(path: Path, stats: dict[str, Any]) -> Path | None:
    """What the strongest directions look like across sensors and heaters."""
    modes = stats["modes"]["modes"]
    if not modes:
        return None
    sensor_ids, heater_ids = stats["sensor_ids"], stats["heater_ids"]
    fig = _figure(13.0, 2.6 * len(modes) + 0.8)
    grid = fig.add_gridspec(len(modes), 2)
    for row, mode in enumerate(modes):
        left = np.asarray(mode["sensor_pattern"], dtype=float)
        right = np.asarray(mode["heater_pattern"], dtype=float)
        ax_s = fig.add_subplot(grid[row, 0])
        _bar_channels(
            ax_s, left, sensor_ids,
            color=[NEGATIVE if v < 0 else POSITIVE for v in left],
        )
        ax_s.axhline(0.0, color="black", linewidth=0.8)
        ax_s.set_ylabel("sensor weight", fontsize=8)
        ax_s.set_title(
            f"direction {mode['index'] + 1}: σ = {mode['sigma']:.4g} K/W  "
            f"(sensor sign agreement {mode['sensor_sign_agreement'] * 100:.0f}%)",
            fontsize=9,
        )
        ax_h = fig.add_subplot(grid[row, 1])
        _bar_channels(
            ax_h, right, heater_ids,
            color=[NEGATIVE if v < 0 else POSITIVE for v in right],
        )
        ax_h.axhline(0.0, color="black", linewidth=0.8)
        ax_h.set_ylabel("heater weight", fontsize=8)
        ax_h.set_title(f"direction {mode['index'] + 1}: heater combination", fontsize=9)
    fig.suptitle(
        "Dominant directions — sign agreement near 100% means the whole structure moves together",
        fontsize=11,
    )
    return _save(fig, path)


def figure_pairing(path: Path, stats: dict[str, Any]) -> Path | None:
    """The two independent readings of "can this be paired one-to-one?"."""
    pairing = stats["pairing"]
    summary = pairing["rga_summary"]
    if not has_pairing_verdict(summary) or pairing["paired_influence_fraction"] is None:
        return None
    sensor_ids = stats["sensor_ids"]
    share = np.asarray(pairing["paired_influence_fraction"], dtype=float) * 100.0
    diag = np.asarray(summary["rga_diag"], dtype=float)

    fig = _figure(12.0, 5.4)
    grid = fig.add_gridspec(2, 1, hspace=0.05)
    ax_share = fig.add_subplot(grid[0, 0])
    _bar_channels(ax_share, share, sensor_ids, color=ACCENT)
    # The benchmark that makes this panel mean something: if a heater spread its
    # influence perfectly evenly over every sensor, its own sensor would still get
    # 100/n %. Landing AT that line means the pairing is not a pairing at all.
    even = 100.0 / max(len(sensor_ids), 1)
    ax_share.axhline(even, color="#2a8a4a", linewidth=1.1, linestyle="--")
    ax_share.annotate(
        f"perfectly even spreading = {even:.2f}%",
        xy=(0.995, even), xycoords=("axes fraction", "data"),
        ha="right", va="bottom", fontsize=7, color="#2a8a4a",
    )
    ax_share.set_ylim(0.0, max(float(share.max()) if share.size else 0.0, even) * 1.25)
    ax_share.set_ylabel("paired influence\n[% of row]", fontsize=8)
    ax_share.tick_params(labelbottom=False)
    ax_share.set_title(
        "Share of a heater's steady influence landing on its own sensor "
        f"(median {np.median(share):.2g}%)",
        fontsize=9,
    )

    ax_diag = fig.add_subplot(grid[1, 0], sharex=ax_share)
    _bar_channels(
        ax_diag, diag, sensor_ids,
        color=[NEGATIVE if v < 0 else POSITIVE for v in diag],
    )
    scale, scale_kwargs = diagonal_axis_scale(diag)
    ax_diag.set_yscale(scale, **scale_kwargs)
    ax_diag.axhline(1.0, color="#2a8a4a", linewidth=1.1, linestyle="--")
    ax_diag.axhline(0.0, color="black", linewidth=0.8)
    ax_diag.set_ylabel(f"RGA diagonal\n({scale})", fontsize=8)
    ax_diag.set_xlabel("controlled sensor node id", fontsize=9)
    ax_diag.set_title(
        f"RGA diagonal — {summary['rga_diag_negative']} of {len(sensor_ids)} negative "
        "(red = gain reverses once the other loops close)",
        fontsize=9,
    )
    fig.suptitle("Per-pair SISO viability", fontsize=11)
    return _save(fig, path)


def figure_actuators(path: Path, stats: dict[str, Any]) -> Path:
    """Per-heater authority, and how much of it duplicates another heater's."""
    actuators = stats["actuators"]
    heater_ids = stats["heater_ids"]
    norms = np.asarray(actuators["column_norm_K_per_W"], dtype=float)
    cosine = np.asarray(actuators["max_cosine_with_another_heater"], dtype=float)

    fig = _figure(12.0, 5.6)
    grid = fig.add_gridspec(2, 1, hspace=0.05)
    ax_norm = fig.add_subplot(grid[0, 0])
    _bar_channels(ax_norm, norms, heater_ids, color=POSITIVE)
    ax_norm.set_ylabel("column norm\n[K/W]", fontsize=8)
    ax_norm.tick_params(labelbottom=False)
    ax_norm.set_title("How much each heater can do", fontsize=9)

    ax_cos = fig.add_subplot(grid[1, 0], sharex=ax_norm)
    _bar_channels(
        ax_cos, cosine, heater_ids,
        color=[NEGATIVE if v > 0.99 else POSITIVE for v in cosine],
    )
    ax_cos.axhline(0.99, color=NEGATIVE, linewidth=1.0, linestyle="--")
    ax_cos.set_ylim(min(0.0, float(cosine.min()) if cosine.size else 0.0), 1.02)
    ax_cos.set_ylabel("max cosine with\nanother heater", fontsize=8)
    ax_cos.set_xlabel("heater node id", fontsize=9)
    ax_cos.set_title(
        f"Actuator redundancy — {actuators['n_near_duplicate_pairs']} heater(s) above 0.99, "
        "i.e. indistinguishable from another heater",
        fontsize=9,
    )
    fig.suptitle("Actuator set", fontsize=11)
    return _save(fig, path)


def figure_reachability(path: Path, stats: dict[str, Any]) -> Path:
    """What each channel costs to shape independently, and what is left after
    asking for a uniform lift."""
    sensor_ids = stats["sensor_ids"]
    cost = np.asarray(stats["reachability"]["independent_control_cost_W_per_K"], dtype=float)
    lift = stats["uniform_lift"]
    bounded = lift.get("nonnegative")
    residual = np.asarray(
        (bounded or lift["unconstrained"])["residual_per_sensor"], dtype=float
    )
    label = "u >= 0" if bounded else "unconstrained"

    fig = _figure(12.0, 5.6)
    grid = fig.add_gridspec(2, 1, hspace=0.05)
    ax_cost = fig.add_subplot(grid[0, 0])
    _bar_channels(ax_cost, np.maximum(cost, 1e-12), sensor_ids, color=POSITIVE, log=True)
    ax_cost.set_ylabel("W per K, alone\n(log)", fontsize=8)
    ax_cost.tick_params(labelbottom=False)
    ax_cost.set_title(
        "Cost of moving one sensor 1 K while holding every other one still "
        f"(median {stats['reachability']['median_cost_W_per_K']:.4g} W/K)",
        fontsize=9,
    )

    ax_res = fig.add_subplot(grid[1, 0], sharex=ax_cost)
    _bar_channels(
        ax_res, residual, sensor_ids,
        color=[NEGATIVE if abs(v) > 0.05 else POSITIVE for v in residual],
    )
    ax_res.axhline(0.0, color="black", linewidth=0.8)
    ax_res.set_ylabel(f"residual [K per K]\n({label})", fontsize=8)
    ax_res.set_xlabel("controlled sensor node id", fontsize=9)
    ax_res.set_title(
        "Left over after asking for +1 K on every sensor — this is plant authority, not tuning",
        fontsize=9,
    )
    fig.suptitle("Per-channel reachability", fontsize=11)
    return _save(fig, path)


def figure_operating_point(path: Path, stats: dict[str, Any]) -> Path | None:
    """The real allocation at the real setpoints, against the real caps."""
    operating = stats["operating_point"]
    if not operating.get("available"):
        return None
    sensor_ids, heater_ids = stats["sensor_ids"], stats["heater_ids"]
    error = np.asarray(operating["error_per_sensor_K"], dtype=float)
    power = np.asarray(operating["power_per_heater_W"], dtype=float)
    saturated = set(operating["saturated_heater_ids"])

    fig = _figure(12.0, 5.6)
    # Not shared: these two panels are indexed by DIFFERENT channels (sensors above,
    # heaters below), so they each need their own tick labels and the gap between.
    grid = fig.add_gridspec(2, 1, hspace=0.12)
    ax_err = fig.add_subplot(grid[0, 0])
    _bar_channels(
        ax_err, error, sensor_ids,
        color=[NEGATIVE if abs(v) > 0.1 else POSITIVE for v in error],
    )
    ax_err.axhline(0.0, color="black", linewidth=0.8)
    ax_err.set_ylabel("steady error [K]", fontsize=8)
    ax_err.set_xlabel("controlled sensor node id", fontsize=9)
    ax_err.set_title(
        f"Best achievable steady error at these setpoints: "
        f"{operating['error_rms_K']:.4g} K rms, worst {operating['error_max_abs_K']:.4g} K",
        fontsize=9,
    )

    ax_pow = fig.add_subplot(grid[1, 0])
    _bar_channels(
        ax_pow, power, heater_ids,
        color=[NEGATIVE if int(h) in saturated else POSITIVE for h in heater_ids],
    )
    ax_pow.set_ylabel("power [W]", fontsize=8)
    ax_pow.set_xlabel("heater node id", fontsize=9)
    ax_pow.set_title(
        f"Allocated power: {operating['total_power_W']:.4g} W total across "
        f"{operating['active_heaters']} heater(s), {operating['saturated_heaters']} at their cap",
        fontsize=9,
    )
    fig.suptitle(
        f"Operating point — setpoints {operating['setpoint_min_K']:g} to "
        f"{operating['setpoint_max_K']:g} K above a {operating['passive_reference_K']:g} K "
        "passive reference",
        fontsize=11,
    )
    return _save(fig, path)


# ---------------------------------------------------------------------- tables
def _write_csv(path: Path, fieldnames: Sequence[str], rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + ".tmp")
    with staged.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(staged, target)
    return target


def _column(stats: dict[str, Any], *path: str) -> list[Any] | None:
    """Nested lookup that yields None instead of raising when a section was
    skipped -- the per-channel table has to survive a missing operating point."""
    node: Any = stats
    for key in path:
        if not isinstance(node, dict) or node.get(key) is None:
            return None
        node = node[key]
    return node if isinstance(node, list) else None


def write_channel_table(path: Path, G: np.ndarray, stats: dict[str, Any]) -> Path:
    sensor_ids = stats["sensor_ids"]
    magnitude = np.abs(G)
    columns = {
        "paired_influence_fraction": _column(stats, "pairing", "paired_influence_fraction"),
        "rga_diagonal": _column(stats, "pairing", "rga_summary", "rga_diag"),
        "independent_control_cost_W_per_K": _column(
            stats, "reachability", "independent_control_cost_W_per_K"
        ),
        "uniform_lift_residual_unconstrained_K_per_K": _column(
            stats, "uniform_lift", "unconstrained", "residual_per_sensor"
        ),
        "uniform_lift_residual_nonneg_K_per_K": _column(
            stats, "uniform_lift", "nonnegative", "residual_per_sensor"
        ),
        "operating_point_error_K": _column(stats, "operating_point", "error_per_sensor_K"),
    }
    rows = []
    for i, sensor_id in enumerate(sensor_ids):
        row: dict[str, Any] = {
            "sensor_id": int(sensor_id),
            "row_sum_K_per_W": float(magnitude[i].sum()),
            "max_gain_K_per_W": float(magnitude[i].max()),
            "strongest_heater_id": int(stats["heater_ids"][int(np.argmax(magnitude[i]))]),
        }
        for name, values in columns.items():
            row[name] = "" if values is None else float(values[i])
        rows.append(row)
    return _write_csv(path, ["sensor_id", "row_sum_K_per_W", "max_gain_K_per_W",
                             "strongest_heater_id", *columns], rows)


def write_heater_table(path: Path, G: np.ndarray, stats: dict[str, Any]) -> Path:
    heater_ids = stats["heater_ids"]
    actuators = stats["actuators"]
    columns = {
        "uniform_lift_power_W_per_K": _column(
            stats, "uniform_lift", "nonnegative", "power_per_heater_W_per_K"
        ),
        "operating_point_power_W": _column(stats, "operating_point", "power_per_heater_W"),
    }
    rows = []
    for j, heater_id in enumerate(heater_ids):
        row: dict[str, Any] = {
            "heater_id": int(heater_id),
            "column_norm_K_per_W": float(actuators["column_norm_K_per_W"][j]),
            "max_gain_K_per_W": float(actuators["max_abs_K_per_W"][j]),
            "top_direction_participation": float(actuators["top_direction_participation"][j]),
            "max_cosine_with_another_heater": float(actuators["max_cosine_with_another_heater"][j]),
            "most_similar_heater_id": int(actuators["most_similar_heater_id"][j]),
        }
        for name, values in columns.items():
            row[name] = "" if values is None else float(values[j])
        rows.append(row)
    return _write_csv(
        path,
        ["heater_id", "column_norm_K_per_W", "max_gain_K_per_W", "top_direction_participation",
         "max_cosine_with_another_heater", "most_similar_heater_id", *columns],
        rows,
    )


def write_spectrum_table(path: Path, stats: dict[str, Any]) -> Path:
    spectrum = stats["spectrum"]
    rows = [
        {
            "index": i,
            "sigma_K_per_W": float(sigma),
            "energy_fraction": float(spectrum["energy_fraction"][i]),
            "cumulative_energy_fraction": float(spectrum["cumulative_energy_fraction"][i]),
        }
        for i, sigma in enumerate(spectrum["singular_values"])
    ]
    return _write_csv(
        path, ["index", "sigma_K_per_W", "energy_fraction", "cumulative_energy_fraction"], rows
    )


# ---------------------------------------------------------------------- report
def report_markdown(stats: dict[str, Any], figures: Sequence[Path], tables: Sequence[Path]) -> str:
    """The written report. Meant to be readable on its own and pasteable whole --
    every claim carries its number, so it does not depend on the figures to make
    sense."""
    meta = stats.get("source_metadata", {})
    structure, spectrum = stats["gain_structure"], stats["spectrum"]
    pairing, actuators = stats["pairing"], stats["actuators"]
    reach, lift, operating = stats["reachability"], stats["uniform_lift"], stats["operating_point"]
    summary = pairing["rga_summary"]
    out: list[str] = []
    add = out.append

    add(f"# Plant analysis — {meta.get('run_name', 'G matrix')}")
    add("")
    add(f"Generated {stats['created_at']} from the saved DC gain.")
    add("")
    add(f"- G: **{stats['n_sensors']} controlled sensors x {stats['n_heaters']} heaters**")
    if meta.get("T_op_K") is not None:
        add(f"- Linearized at T_op = **{meta['T_op_K']:g} K** (method: {meta.get('method', '?')})")
    if meta.get("dc_ground"):
        add(f"- DC grounded at the **{meta['dc_ground']}**")
    add("")

    add("## 1. Headline")
    add("")
    add(f"- Every heater reaches every sensor at **{structure['min_abs_K_per_W']:.3g} to "
        f"{structure['max_abs_K_per_W']:.3g} K/W** "
        f"(dynamic range {structure['dynamic_range']:.1f}x)"
        if structure["dynamic_range"] else "- Gain range unavailable")
    add(f"- **sigma_1 carries {spectrum['top_energy_fraction'] * 100:.4g}%** of the plant; "
        f"{spectrum['directions_for_90pct']} direction(s) reach 90%, "
        f"{spectrum['directions_for_99pct']} reach 99%")
    add(f"- cond(G) = **{spectrum['condition_number']:.4g}**; effective rank "
        + ", ".join(f"{v} at tol {k.split('_')[1]}" for k, v in spectrum["effective_rank"].items()))
    if has_pairing_verdict(summary):
        add(f"- RGA diagonal: **{summary['rga_diag_negative']} of {stats['n_sensors']} negative** "
            f"(min {summary['rga_diag_min']:+.4g}) — per-pair SISO is ruled out")
        if summary.get("median_paired_influence_fraction") is not None:
            add(f"- Median share of a heater's influence landing on its own sensor: "
                f"**{summary['median_paired_influence_fraction'] * 100:.2g}%**")
        if pairing.get("niederlinski_index") is not None:
            verdict = "unstable" if pairing["niederlinski_index"] < 0 else "not excluded"
            add(f"- Niederlinski index = **{pairing['niederlinski_index']:.4g}** "
                f"— diagonal control with integral action is {verdict}")
    else:
        add(f"- RGA diagonal not reported ({stats['n_sensors']}x{stats['n_heaters']} is not square, "
            "so there is no one-heater-per-sensor pairing to describe)")
    bounded = lift.get("nonnegative")
    if bounded:
        add(f"- Asking for +1 K on every sensor leaves **{bounded['residual_rms_K_per_K']:.4g} K rms "
            f"per K requested** once heaters are held to u >= 0 "
            f"(vs {lift['unconstrained']['residual_rms_K_per_K']:.4g} K unconstrained)")
    if actuators["n_near_duplicate_pairs"]:
        add(f"- **{actuators['n_near_duplicate_pairs']} heater(s)** are collinear above 0.99 with "
            "another heater — they are one actuator with two names")
    add("")

    add("## 2. Gain structure")
    add("")
    add("| quantity | value |")
    add("|---|---|")
    # No "|G|" in a label: the pipes would split the markdown row into extra cells.
    for label, key, unit in (
        ("smallest gain magnitude", "min_abs_K_per_W", "K/W"),
        ("largest gain magnitude", "max_abs_K_per_W", "K/W"),
        ("median gain magnitude", "median_abs_K_per_W", "K/W"),
        ("median row sum", "median_row_sum_K_per_W", "K/W"),
        ("median column sum", "median_column_sum_K_per_W", "K/W"),
    ):
        add(f"| {label} | {structure[key]:.4g} {unit} |")
    add(f"| fraction of entries positive | {structure['fraction_positive'] * 100:.1f}% |")
    if structure["median_offdiagonal_to_diagonal"] is not None:
        add(f"| median off-diagonal / diagonal | "
            f"{structure['median_offdiagonal_to_diagonal']:.4g} |")
    add("")

    add("## 3. Singular spectrum")
    add("")
    add("| i | sigma [K/W] | energy | cumulative |")
    add("|---|---|---|---|")
    for i, sigma in enumerate(spectrum["singular_values"]):
        # %.2f rounds the entire tail of an ill-conditioned plant to "0.00%", which
        # is the one part of this table anyone reads it for. Significant figures.
        add(f"| {i + 1} | {sigma:.4g} | {spectrum['energy_fraction'][i] * 100:.3g}% | "
            f"{spectrum['cumulative_energy_fraction'][i] * 100:.6g}% |")
    add("")

    add("## 4. Dominant directions")
    add("")
    for mode in stats["modes"]["modes"]:
        add(f"- **direction {mode['index'] + 1}**, sigma = {mode['sigma']:.4g} K/W — "
            f"sensor sign agreement {mode['sensor_sign_agreement'] * 100:.0f}%, "
            f"heater sign agreement {mode['heater_sign_agreement'] * 100:.0f}%; "
            f"strongest on sensor {mode['dominant_sensor_id']} / heater {mode['dominant_heater_id']}")
    add("")
    add("Sign agreement near 100% on the first direction means the whole structure moves "
        "together: that direction is a uniform lift, not a way of shaping one region "
        "against another.")
    add("")

    add("## 5. Actuator set")
    add("")
    add(f"- median max-cosine between a heater and its nearest neighbour: "
        f"**{actuators['median_max_cosine']:.4g}**"
        if actuators["median_max_cosine"] is not None else "- collinearity unavailable")
    add(f"- heaters collinear above 0.99 with another: **{actuators['n_near_duplicate_pairs']}**")
    add("")

    add("## 6. Per-channel reachability")
    add("")
    add(f"Cost of moving one sensor 1 K while holding every other one still, "
        f"median **{reach['median_cost_W_per_K']:.4g} W/K**"
        + (f", worst/median ratio **{reach['cost_ratio_worst_to_median']:.4g}**"
           if reach.get("cost_ratio_worst_to_median") else "") + ".")
    add("")
    add("| sensor | cost [W/K] |")
    add("|---|---|")
    for entry in reach["worst_channels"]:
        add(f"| {entry['sensor_id']} | {entry['cost_W_per_K']:.4g} |")
    add("")

    add("## 7. Uniform lift (+1 K on every sensor)")
    add("")
    unconstrained = lift["unconstrained"]
    add(f"- unconstrained: {unconstrained['power_W_per_K']:.4g} W/K, residual "
        f"{unconstrained['residual_rms_K_per_K']:.4g} K rms, worst "
        f"{unconstrained['residual_max_K_per_K']:.4g} K — but "
        f"{unconstrained['negative_power_heaters']} heater(s) would need NEGATIVE power "
        f"(down to {unconstrained['most_negative_W_per_K']:.4g} W/K), which no heater can deliver")
    if bounded:
        add(f"- with u >= 0: {bounded['power_W_per_K']:.4g} W/K across "
            f"{bounded['active_heaters']} heater(s), residual "
            f"{bounded['residual_rms_K_per_K']:.4g} K rms, worst "
            f"{bounded['residual_max_K_per_K']:.4g} K")
        add(f"- the cost of one-sided actuation is "
            f"**{lift['nonnegativity_penalty_rms_K_per_K']:.4g} K rms per K requested**")
    else:
        add(f"- bounded solve unavailable: {lift.get('nonnegative_error', 'not attempted')}")
    add("")

    add("## 8. Operating point")
    add("")
    if operating.get("available"):
        add(f"Setpoints {operating['setpoint_min_K']:g} to {operating['setpoint_max_K']:g} K "
            f"against a {operating['passive_reference_K']:g} K passive reference "
            f"(mean deviation {operating['requested_deviation_mean_K']:.4g} K, "
            f"spread {operating['requested_deviation_spread_K']:.4g} K).")
        add("")
        add(f"- best achievable steady error: **{operating['error_rms_K']:.4g} K rms**, worst "
            f"{operating['error_max_abs_K']:.4g} K")
        add(f"- power: **{operating['total_power_W']:.4g} W** across "
            f"{operating['active_heaters']} heater(s), peak "
            f"{operating['max_heater_power_W']:.4g} W")
        add(f"- heaters at their cap: **{operating['saturated_heaters']}**"
            + (f" ({', '.join(str(v) for v in operating['saturated_heater_ids'][:10])})"
               if operating["saturated_heater_ids"] else ""))
        add("")
        add("| sensor | error [K] |")
        add("|---|---|")
        for entry in operating["worst_channels"]:
            add(f"| {entry['sensor_id']} | {entry['error_K']:+.4g} |")
    else:
        add("Skipped — this is the one section that needs more than G, and it was not "
            "supplied: " + "; ".join(operating.get("missing", ["unknown"])) + ".")
        add("")
        add("Every other section above is scale-free and true of the plant regardless.")
    add("")

    add("## Files")
    add("")
    for path in list(figures) + list(tables):
        add(f"- `{Path(path).name}`")
    add("")
    return "\n".join(out)


# ---------------------------------------------------------------- orchestration
def write_plant_analysis(
    gain_folder: Path,
    *,
    enabled_sensor_ids: Sequence[int] | None = None,
    setpoints_K: dict[int, float] | None = None,
    heater_max_power_W: dict[int, float] | float | None = None,
    out_dir: Path | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> PlantAnalysis:
    """Compute everything and write it under ``<gain folder>/analysis/``.

    ``enabled_sensor_ids`` is the run's own channel selection. It is applied
    rather than ignored because most of this report is a property of the CLOSED
    loop: drop a channel and the RGA, the reachability costs and the achievable
    error all move, since each runs through the pseudo-inverse.

    A figure that fails to draw is reported and skipped. The numbers are already
    written by then, so a matplotlib problem costs a picture rather than the
    analysis -- the same trade the run's own plots make.
    """
    from .sys_id_artifacts import load_sys_id_gain_matrix_data

    folder = Path(gain_folder)
    data = load_sys_id_gain_matrix_data(folder)
    G = np.asarray(data.G, dtype=float)
    sensor_ids = [int(v) for v in data.sensor_ids]
    heater_ids = [int(v) for v in data.heater_ids]
    if G.size == 0 or not sensor_ids or not heater_ids:
        raise ValueError(f"{folder.name} holds no usable gain matrix to analyse.")

    excluded: list[int] = []
    if enabled_sensor_ids is not None:
        allowed = {int(v) for v in enabled_sensor_ids}
        keep = [i for i, sid in enumerate(sensor_ids) if sid in allowed]
        excluded = [sid for sid in sensor_ids if sid not in allowed]
        if not keep:
            raise ValueError(
                "None of this G matrix's sensors are ticked as controlled, so there is no "
                "loop left to analyse."
            )
        if excluded:
            G = G[keep, :]
            sensor_ids = [sensor_ids[i] for i in keep]

    _report(on_progress, f"computing statistics for {G.shape[0]}x{G.shape[1]} G ...")
    stats = compute_plant_analysis(
        G, sensor_ids, heater_ids,
        setpoints_K=setpoints_K,
        # The passive equilibrium the gain was built around. Only the artifact
        # knows it, and older artifacts do not carry it at all -- in which case
        # the operating-point section says so instead of guessing one.
        passive_reference_K=_passive_reference(data.metadata),
        heater_max_power_W=heater_max_power_W,
        metadata={"run_name": data.name, "created_at": data.created_at, **dict(data.metadata)},
    )
    stats["excluded_sensor_ids"] = excluded
    stats["gain_matrix_path"] = str(folder)

    target = Path(out_dir) if out_dir is not None else folder / ANALYSIS_DIRNAME
    target.mkdir(parents=True, exist_ok=True)

    tables = [
        write_channel_table(target / "channels.csv", G, stats),
        write_heater_table(target / "heaters.csv", G, stats),
        write_spectrum_table(target / "spectrum.csv", stats),
        # The RGA is the one section whose full matrix is worth having as numbers:
        # it is what anyone re-checking the pairing claim will want to join against
        # gain_matrix.csv, and it does not fit in the per-channel table.
        write_rga_csv(target / RGA_CSV, sensor_ids, heater_ids, stats["pairing"]["RGA"]),
    ]

    figures: list[Path] = []
    drawings: list[tuple[str, Callable[[], Path | None]]] = [
        ("gain matrix", lambda: figure_gain_matrix(target / "gain_matrix.png", G, stats)),
        ("spectrum", lambda: figure_spectrum(target / "spectrum.png", stats)),
        ("mode shapes", lambda: figure_mode_shapes(target / "mode_shapes.png", stats)),
        ("RGA", lambda: render_rga_figure(
            target / RGA_PNG,
            stats["pairing"]["RGA"],
            sensor_ids,
            heater_ids,
            stats["pairing"]["rga_summary"],
            subtitle=f"G matrix: {data.name}",
        )),
        ("pairing", lambda: figure_pairing(target / "pairing.png", stats)),
        ("actuators", lambda: figure_actuators(target / "actuators.png", stats)),
        ("reachability", lambda: figure_reachability(target / "reachability.png", stats)),
        ("operating point", lambda: figure_operating_point(target / "operating_point.png", stats)),
    ]
    skipped: list[str] = []
    for name, draw in drawings:
        _report(on_progress, f"drawing {name} ...")
        try:
            path = draw()
        except Exception as exc:  # noqa: BLE001 - a figure must not cost the report
            skipped.append(f"{name} ({type(exc).__name__}: {exc})")
            continue
        if path is not None:
            figures.append(path)
    stats["skipped_figures"] = skipped

    json_path = target / REPORT_JSON
    # RGA is a full matrix and already has its own CSV; keeping it out of the JSON
    # keeps that file something a person can open.
    serializable = {**stats, "pairing": {k: v for k, v in stats["pairing"].items() if k != "RGA"}}
    json_path.write_text(json.dumps(serializable, indent=2, default=float), encoding="utf-8")
    report_path = target / REPORT_MD
    report_path.write_text(report_markdown(stats, figures, tables), encoding="utf-8")

    return PlantAnalysis(
        out_dir=target,
        stats=stats,
        report_path=report_path,
        json_path=json_path,
        figures=figures,
        tables=tables,
    )


def _passive_reference(metadata: dict[str, Any]) -> float | None:
    value = metadata.get("passive_reference_K")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _report(on_progress: Callable[[str], None] | None, message: str) -> None:
    if on_progress is not None:
        on_progress(message)
