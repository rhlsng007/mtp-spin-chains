"""
Consistent plotting style for every thesis figure.

Call set_thesis_style() once at the top of a plotting script, and use
save_figure() so every figure is written as both PDF (for LaTeX) and
PNG (for quick viewing and slides).
"""

import matplotlib
import matplotlib.pyplot as plt

from config import FIGURES_DIR

# Colour-blind-friendly palette (Okabe–Ito)
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}
PALETTE = list(COLORS.values())

# Fixed colours for recurring quantities, so they match across chapters
SERIES_COLORS = {
    "ED": COLORS["black"],
    "HVA": COLORS["blue"],
    "HEA": COLORS["orange"],
    "noisy": COLORS["red"],
    "mitigated": COLORS["green"],
}

# Typical LaTeX text widths in inches (A4, 1-inch margins ≈ 6.3 in)
SINGLE_COLUMN = 6.3
HALF_COLUMN = 3.1


def set_thesis_style(use_latex: bool = False) -> None:
    """
    Apply the thesis plotting style globally.

    use_latex=True renders text with a real LaTeX installation, which
    matches the thesis font exactly but is slower and requires LaTeX on
    the machine. The default uses matplotlib's built-in mathtext.
    """
    plt.rcParams.update({
        "figure.figsize": (SINGLE_COLUMN, SINGLE_COLUMN * 0.62),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "legend.frameon": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "mathtext.fontset": "cm",
        "text.usetex": use_latex,
    })


def save_figure(fig, name: str, subdir: str = "") -> None:
    """Save as figures/<subdir>/<name>.pdf and .png."""
    folder = FIGURES_DIR / subdir
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{ext}")
    print(f"[figure] saved {folder / name}.pdf / .png")
