"""Consistent figure style and saving (titled PNGs named by deliverable)."""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402

# Colour-blind-safe categorical palette (Okabe-Ito) + semantic colours
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000", "#999999"]
SEM = {"plant": "#009E73", "animal": "#D55E00", "efficient": "#0072B2", "dominated": "#BBBBBB",
       "strong": "#009E73", "misleading": "#D55E00", "caveated": "#E69F00", "unverifiable": "#999999",
       "not attractive": "#DDDDDD", "front": "#0072B2", "original": "#000000", "random": "#BBBBBB"}
VIEW_COLORS = {"mass": "#0072B2", "energy": "#E69F00", "protein": "#009E73"}


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": C.DPI, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.6, "figure.titlesize": 12, "figure.titleweight": "bold",
    })


def fig_name(deliverable: int, title: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", title).strip("_")
    return f"D{deliverable}_{slug}.png"


def save(fig, deliverable: int, title: str, subtitle: str | None = None) -> str:
    """Add a numbered title (+ optional subtitle) above the axes and save to reports/figures.

    Title and subtitle are stacked with a fixed offset in points, so they never overlap
    whatever the figure height; ``bbox_inches='tight'`` then includes them.
    """
    from matplotlib.transforms import offset_copy

    C.FIGURES.mkdir(parents=True, exist_ok=True)
    y0 = 1.0
    if subtitle:
        fig.text(0.5, y0, subtitle, ha="center", va="bottom", fontsize=8.5, color="#444444",
                 transform=offset_copy(fig.transFigure, fig=fig, y=4, units="points"))
    fig.text(0.5, y0, f"D{deliverable} - {title}", ha="center", va="bottom", fontsize=12.5, fontweight="bold",
             transform=offset_copy(fig.transFigure, fig=fig, y=20 if subtitle else 6, units="points"))
    name = fig_name(deliverable, title)
    fig.savefig(C.FIGURES / name)
    plt.close(fig)
    return name


def log_axis(ax, axis: str = "x") -> None:
    """Log scale with readable 1-2-5 tick labels and no colliding minor labels."""
    import matplotlib.ticker as mt

    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    a.set_major_locator(mt.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(mt.NullFormatter())


def group_colors(labels) -> dict:
    """Stable distinct colours for up to 20 categories."""
    cmap = plt.get_cmap("tab20")
    return {lab: cmap(i % 20) for i, lab in enumerate(sorted(set(labels)))}


def short(name: str, n: int = 38) -> str:
    name = str(name)
    return name if len(name) <= n else name[: n - 1] + "…"
