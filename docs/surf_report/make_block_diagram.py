"""Draw the controller block diagram for the SURF talk.

Kept as a script rather than a hand-edited image so the numbers in the boxes can
be corrected when the plant analysis is re-run -- several of them (17.40 vs 17.29
W, K_p = 5.5, cond 366) are measured values that have already moved once.

Two themes, because the same diagram has to work in two places. ``dark`` is for
the talk, whose slides are black: it is NOT the light version with the background
removed, since transparency alone would leave black text invisible on black --
the ink is inverted and the accent colours lightened to hold contrast. ``light``
is for print and for the report.

Both save with a transparent background, so the slide's own background shows
through and the figure carries no white rectangle around it.

Coordinates are in a 1180 x 600 pixel-like frame with y increasing DOWNWARD, so
the layout reads the same way it is written.

    python docs/surf_report/make_block_diagram.py            # both themes
    python docs/surf_report/make_block_diagram.py --theme dark
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

W, H = 1180.0, 560.0

# DejaVu Sans (matplotlib's default) sets roughly 0.84 * fontsize per character,
# noticeably wider than the browser fonts this layout was first sketched against.
# Boxes are sized from that, and the sub-lines are kept under ~32 characters --
# a longer one silently draws past its box rather than wrapping.
TITLE_SIZE, SUB_SIZE, SUB_STEP = 10.0, 8.5, 17.0

# Accents are lightened rather than reused on dark: #2f6f9f on black is nearly
# unreadable, and #c1272d reads as brown. Fills stay opaque so a box still has an
# edge over a busy slide background.
THEMES = {
    "light": {
        "ink": "#1a1a1a", "sub": "#555555", "muted": "#666666", "wire": "#333333",
        "ghost": "#8a8a8a", "junction": "#ffffff",
        "live": {"facecolor": "#eaf2f8", "edgecolor": "#2f6f9f"},
        "fix": {"facecolor": "#e9f4ec", "edgecolor": "#2a8a4a"},
        "plant": {"facecolor": "#f0f0f0", "edgecolor": "#555555"},
        "off": {"facecolor": "#fdf1e3", "edgecolor": "#e08214", "linestyle": (0, (6, 4))},
        "accent_green": "#2a8a4a", "accent_orange": "#e08214", "accent_red": "#c1272d",
    },
    "dark": {
        "ink": "#f5f5f5", "sub": "#c4c4c4", "muted": "#a8a8a8", "wire": "#e2e2e2",
        "ghost": "#8f8f8f", "junction": "#101010",
        "live": {"facecolor": "#142c3c", "edgecolor": "#63a9d4"},
        "fix": {"facecolor": "#15301e", "edgecolor": "#5cc47e"},
        "plant": {"facecolor": "#262626", "edgecolor": "#9a9a9a"},
        "off": {"facecolor": "#33240f", "edgecolor": "#f0a95c", "linestyle": (0, (6, 4))},
        "accent_green": "#6fd18f", "accent_orange": "#f0a95c", "accent_red": "#ff7a7a",
    },
}


class Diagram:
    def __init__(self, theme: str) -> None:
        self.c = THEMES[theme]
        self.fig = plt.figure(figsize=(W / 100.0, H / 100.0))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)      # y downward, so the layout reads as written
        self.ax.axis("off")

    def box(self, x, y, w, h, kind, title, sublines=(), sub_colors=()):
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0,rounding_size=8",
                linewidth=2, zorder=3, **self.c[kind],
            )
        )
        cx = x + w / 2.0
        self.ax.text(cx, y + 22, title, ha="center", va="center",
                     fontsize=TITLE_SIZE, fontweight="600", color=self.c["ink"], zorder=4)
        for i, line in enumerate(sublines):
            color = sub_colors[i] if i < len(sub_colors) else self.c["sub"]
            self.ax.text(cx, y + 42 + SUB_STEP * i, line, ha="center", va="center",
                         fontsize=SUB_SIZE, color=color, zorder=4)

    def wire(self, points, dashed=False):
        """Polyline with an arrowhead on the final segment."""
        color = self.c["ghost"] if dashed else self.c["wire"]
        width = 1.7 if dashed else 2.0
        style = (0, (5, 4)) if dashed else "-"
        if len(points) > 2:
            self.ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
                         color=color, linewidth=width, linestyle=style,
                         solid_capstyle="butt", zorder=2)
        self.ax.annotate(
            "", xy=points[-1], xytext=points[-2],
            arrowprops=dict(arrowstyle="-|>", color=color, linewidth=width,
                            mutation_scale=16, shrinkA=0, shrinkB=0, linestyle=style),
            zorder=2,
        )

    def junction(self, cx, cy, marks):
        """Summing node. ``marks`` is (dx, dy, '+'/'-') per incoming signal."""
        self.ax.add_patch(Circle((cx, cy), 19, facecolor=self.c["junction"],
                                 edgecolor=self.c["wire"], linewidth=2, zorder=4))
        self.ax.text(cx, cy, "Σ", ha="center", va="center", fontsize=13,
                     fontweight="bold", color=self.c["ink"], zorder=5)
        for dx, dy, sign in marks:
            self.ax.text(cx + dx, cy + dy, sign, ha="center", va="center",
                         fontsize=12, fontweight="bold", color=self.c["ink"], zorder=5)

    def dot(self, x, y):
        self.ax.plot([x], [y], marker="o", markersize=5, color=self.c["wire"], zorder=5)

    def signal(self, x, y, text, **kw):
        self.ax.text(x, y, text, fontsize=11, fontweight="600", style="italic",
                     color=self.c["ink"], zorder=5, **kw)

    def note(self, x, y, text, color=None, size=9, **kw):
        self.ax.text(x, y, text, fontsize=size, zorder=5,
                     color=color or self.c["muted"], **kw)


def build(theme: str) -> plt.Figure:
    d = Diagram(theme)

    # ---------------------------------------------------- solved once, offline
    d.note(470, 24, "— solved once, offline, from the model —",
           color=d.c["accent_orange"], ha="center")
    d.box(100, 40, 280, 76, "off", "Passive reference  ŷ_passive",
          ("derived from the cooler curve,", "not measured — 17.40 vs 17.29 W"))
    d.box(590, 40, 250, 76, "off", "DC gain  G  (27 × 27)",
          ("solved once from  L T = P", "cond 366,  σ₁ = 81 %"))

    # ---------------------------------------------------------- disturbance
    d.box(900, 138, 220, 58, "plant", "Cryocooler", ("continuous heat removal",))

    # ------------------------------------------------------ feedforward path
    d.wire([(95, 290), (95, 160), (178, 160)])
    d.junction(200, 160, [(-28, -12, "+"), (20, -24, "−")])
    d.wire([(200, 116), (200, 139)])
    d.wire([(219, 160), (540, 160), (540, 266)])
    d.signal(290, 146, "r_dev = r − ŷ_passive")
    d.note(290, 182, "known before the run starts")

    # ---------------------------------------------------------- forward path
    d.signal(18, 282, "r")
    d.note(10, 316, "27 setpoints")
    d.wire([(34, 290), (178, 290)])
    d.dot(95, 290)
    d.junction(200, 290, [(-28, -12, "+"), (-14, 30, "−")])
    d.wire([(219, 290), (277, 290)])
    d.signal(243, 278, "e")

    d.box(280, 252, 210, 76, "live", "27 scalar PI channels",
          ("in decoupled coordinates", "K_p = 5.5"))

    d.wire([(490, 290), (518, 290)])
    d.junction(540, 290, [(-28, -12, "+"), (20, -26, "+")])
    d.wire([(559, 290), (587, 290)])
    d.signal(568, 278, "v")

    d.box(590, 242, 250, 96, "live", "Bounded allocator",
          ("u = argmin ‖G u − v‖² + ‖R u‖²",
           "subject to  u ≥ 0",
           "project, don't clip"),
          sub_colors=(d.c["sub"], d.c["sub"], d.c["accent_red"]))

    d.wire([(840, 290), (897, 290)])
    d.note(869, 272, "u ≥ 0", ha="center", color=d.c["ink"])

    d.box(900, 242, 220, 96, "plant", "Cryostat",
          ("27 heaters, 27 sensors", "~24 h dominant mode",
           "8.6 h power → temperature lag"))
    d.wire([(1010, 196), (1010, 238)])

    d.wire([(1120, 290), (1172, 290)])
    d.dot(1140, 290)
    d.signal(1148, 278, "y")
    # Right-anchored to the plant box, not the canvas: anywhere further right and
    # it runs under the feedback wire dropping at x = 1140, then off the edge.
    d.note(1120, 372, "27 temps [K]", ha="right")

    # --------------------------------------------------------- feedback path
    d.wire([(1140, 290), (1140, 430), (846, 430)])
    d.box(590, 402, 250, 58, "fix", "Measurement filter", ("τ_f = 900 s",))
    d.wire([(590, 430), (200, 430), (200, 313)])
    d.signal(400, 418, "y_f")
    d.note(715, 484,
           "hides the one-step parasitic path, passes the 34 h mode"
           "   —   K_p:  0.09 → 5.5",
           color=d.c["accent_green"], ha="center")

    # ---------------------------------------------------------- offline feed
    d.wire([(715, 118), (715, 238)], dashed=True)
    d.note(728, 218, "defines the decoupling", color=d.c["ghost"])

    # ---------------------------------------------------------------- caption
    d.note(560, 526,
           "The feedforward supplies the holding power; the PI only trims it.  "
           "Nothing in the live loop inverts G — the allocator solves against it under u ≥ 0.",
           size=9.5, ha="center")
    return d.fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", choices=("dark", "light", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=260)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    themes = ("dark", "light") if args.theme == "both" else (args.theme,)
    for theme in themes:
        fig = build(theme)
        png = here / f"controller_block_diagram_{theme}.png"
        for suffix in (".png", ".pdf", ".svg"):
            target = here / f"controller_block_diagram_{theme}{suffix}"
            # transparent=True so the slide's own background shows through and the
            # figure carries no white rectangle around it.
            fig.savefig(target, dpi=args.dpi, transparent=True)
            print(f"wrote {target}")
        plt.close(fig)
        if theme == "dark":
            print(f"wrote {_write_preview(png)}")


def _write_preview(png: Path) -> Path:
    """Flatten the dark theme onto black for checking.

    The deliverable is transparent with near-white ink, so any viewer that shows
    transparency as white renders it as a blank rectangle -- which reads as a
    broken export rather than as a file destined for a black slide. Same
    ``*_preview.png`` convention as the QR asset beside it.
    """
    from PIL import Image

    image = Image.open(png).convert("RGBA")
    backdrop = Image.new("RGBA", image.size, (0, 0, 0, 255))
    target = png.with_name(png.stem + "_preview.png")
    Image.alpha_composite(backdrop, image).convert("RGB").save(target)
    return target


if __name__ == "__main__":
    main()
