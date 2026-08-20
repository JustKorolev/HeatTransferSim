"""Draw the "Controlling the Sim" plant/loop overview.

Rebuilt as a script because the original was a one-off image with no source in
the repo, so every label change meant redrawing it by hand. The palette and
layout follow that original; the signal names do not (MV/PV are now HD/MT).

Saves transparent, for black slides.

    python docs/surf_report/make_loop_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

W, H = 1030.0, 545.0

CONTROL = {"facecolor": "#4A3FAF", "edgecolor": "#6E62D6"}
ADDS = {"facecolor": "#8A3A24", "edgecolor": "#B3573A"}
REMOVES = {"facecolor": "#16624A", "edgecolor": "#2E8C6B"}
PLANT = {"facecolor": "#555555", "edgecolor": "#7E7E7E"}

INK = "#F2F2F0"
SUB = "#DCDCDC"
MUTED = "#B8B8B8"
WIRE = "#9C9C9C"

LEGEND = (("Control", CONTROL), ("Adds heat", ADDS),
          ("Removes heat", REMOVES), ("Plant and sensing", PLANT))


def box(ax, x, y, w, h, style, title, sublines=()):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=7",
                                linewidth=1.6, zorder=3, **style))
    cx = x + w / 2.0
    top = y + (26 if sublines else h / 2.0)
    ax.text(cx, top, title, ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=INK, zorder=4)
    for i, line in enumerate(sublines):
        ax.text(cx, top + 24 + 21 * i, line, ha="center", va="center",
                fontsize=10, color=SUB, zorder=4)


def wire(ax, points):
    if len(points) > 2:
        ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
                color=WIRE, linewidth=1.8, solid_capstyle="butt", zorder=2)
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>", color=WIRE, linewidth=1.8,
                                mutation_scale=14, shrinkA=0, shrinkB=0), zorder=2)


def build() -> plt.Figure:
    fig = plt.figure(figsize=(W / 100.0, H / 100.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    # ------------------------------------------------------------ forward path
    ax.text(24, 54, "SP", fontsize=11, color=MUTED, zorder=5)
    wire(ax, [(24, 88), (62, 88)])
    ax.add_patch(Circle((92, 88), 26, facecolor="#4A3FAF", edgecolor="#6E62D6",
                        linewidth=1.6, zorder=4))
    ax.text(92, 88, "Σ", ha="center", va="center", fontsize=15,
            fontweight="bold", color=INK, zorder=5)
    ax.text(66, 124, "−", fontsize=13, fontweight="bold", color=MUTED, zorder=5)

    wire(ax, [(118, 88), (168, 88)])
    box(ax, 172, 42, 218, 92, CONTROL, "Controller", ("error = SP − MT",))

    wire(ax, [(390, 88), (440, 88)])
    box(ax, 444, 42, 216, 92, ADDS, "Heaters", ("27 channels", "u ≥ 0, capped"))

    ax.text(686, 66, "HD", fontsize=11, color=MUTED, zorder=5)
    wire(ax, [(660, 88), (712, 88)])
    box(ax, 716, 42, 234, 92, PLANT, "Cryostat", ("3.0M nodes", "8.7M links"))

    # ------------------------------------------------------------- disturbance
    box(ax, 620, 188, 260, 70, REMOVES, "Cryocooler", ("always removing heat",))
    wire(ax, [(790, 188), (790, 140)])

    # ------------------------------------------------------------ feedback path
    wire(ax, [(950, 88), (988, 88), (988, 355), (954, 355)])
    ax.text(998, 200, "MT", fontsize=11, color=MUTED, zorder=5)
    box(ax, 716, 310, 234, 92, PLANT, "Thermometers", ("91 sensors", "27 controlled"))

    wire(ax, [(716, 355), (566, 355)])
    box(ax, 306, 310, 256, 92, CONTROL, "Measurement", ("sampled temperatures",))
    wire(ax, [(306, 355), (92, 355), (92, 118)])

    # ------------------------------------------------------------------ legend
    ax.text(W / 2.0, 452, "SP = target temperature  ·  HD = heat delivered  ·  "
            "MT = measured temperature", fontsize=10, color=MUTED,
            ha="center", va="center", zorder=5)

    # DejaVu Sans advances ~0.6 em per character, so a 10 pt label is ~8.4 px per
    # character at 100 px/in; plus the swatch and the gap after it. Undercounting
    # this ran each swatch into the previous label.
    widths = [len(name) * 8.4 + 56 for name, _ in LEGEND]
    x = W / 2.0 - sum(widths) / 2.0
    for (name, style), width in zip(LEGEND, widths):
        ax.add_patch(Rectangle((x, 494), 15, 15, facecolor=style["facecolor"],
                               edgecolor=style["edgecolor"], linewidth=1.2, zorder=4))
        ax.text(x + 24, 502, name, fontsize=10, color=MUTED, va="center", zorder=5)
        x += width
    return fig


def main() -> None:
    here = Path(__file__).resolve().parent
    fig = build()
    for suffix in (".png", ".pdf", ".svg"):
        target = here / f"loop_diagram_dark{suffix}"
        fig.savefig(target, dpi=260, transparent=True)
        print(f"wrote {target}")
    plt.close(fig)

    from PIL import Image

    png = here / "loop_diagram_dark.png"
    image = Image.open(png).convert("RGBA")
    backdrop = Image.new("RGBA", image.size, (0, 0, 0, 255))
    preview = here / "loop_diagram_dark_preview.png"
    Image.alpha_composite(backdrop, image).convert("RGB").save(preview)
    print(f"wrote {preview}")


if __name__ == "__main__":
    main()
