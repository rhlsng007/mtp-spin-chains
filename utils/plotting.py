"""
Consistent plotting style for every thesis figure.

Two styles live here:

* set_thesis_style() + save_figure(): the Phase 0 working style used by the
  phaseN_plots.py scripts (unchanged, so re-plotting an old phase gives the
  same figures; only the PDF creation date is no longer embedded).
* The Phase 8 thesis style (T8.2) for the final figures in figures/thesis/:
  thesis_figure(), save_thesis_figure(), panel_labels(), audit_style() and
  the fixed colour maps below.

Thesis figures are drawn at their printed size (full width 6.3 in, half width
3.1 in) with constrained layout and saved without bbox cropping, so a figure
included with width=\\textwidth is reproduced 1:1 and every label is printed at
exactly THESIS_FONT_SIZES. Text uses Computer Modern (cmr10, shipped with
matplotlib) with DejaVu Serif as glyph fallback, and Computer Modern math, so
labels match a LaTeX thesis without a LaTeX installation.

Saved files are deterministic: the PDF creation date is dropped and the PNG
carries no timestamp, so re-running a script on unchanged data leaves the
tracked PDFs byte-identical (git sees no change).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

from config import FIGURES_DIR

# ---------------------------------------------------------------------------
# Colours (Okabe-Ito, colour-blind friendly)
# ---------------------------------------------------------------------------
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#7F7F7F",
}
PALETTE = [COLORS[k] for k in ("blue", "orange", "green", "red", "purple", "sky", "yellow", "black")]

# Fixed colours for recurring quantities, so they match across chapters
SERIES_COLORS = {
    "ED": COLORS["black"],
    "HVA": COLORS["blue"],
    "HEA": COLORS["orange"],
    "noisy": COLORS["red"],
    "mitigated": COLORS["green"],
}

# Phase 4 and 5 noise channels
CHANNEL_COLORS = {
    "depolarizing": COLORS["purple"],
    "amplitude_damping": COLORS["sky"],
    "dephasing": COLORS["green"],
    "realistic": COLORS["red"],
}
CHANNEL_LABELS = {
    "depolarizing": "depolarizing",
    "amplitude_damping": "amplitude damping",
    "dephasing": "dephasing",
    "realistic": "hardware-like",
}

# Phase 5 mitigation stages
MITIGATION_COLORS = {
    "raw": COLORS["red"],
    "readout": COLORS["sky"],
    "zne": COLORS["purple"],
    "combined": COLORS["green"],
    "ideal": COLORS["black"],
}

# Phase 7 cross-noise training modes
MODE_COLORS = {
    "clean_ed": COLORS["black"],
    "clean_vqe": COLORS["sky"],
    "noisy": COLORS["red"],
    "mixed": COLORS["purple"],
    "loo": COLORS["grey"],
}
MODE_LABELS = {
    "clean_ed": "train ED",
    "clean_vqe": "train noiseless circuit",
    "noisy": "train = test noise",
    "mixed": "train mixed",
    "loo": "leave one out",
}

ANSATZ_MARKERS = {"ED": None, "HVA": "o", "HEA": "s"}
# Depth convention used throughout: p = N/2 dashed, p = N solid
DEPTH_LINESTYLES = {"half": "--", "full": "-"}

# ---------------------------------------------------------------------------
# Sizes
# ---------------------------------------------------------------------------
# Typical LaTeX text widths in inches (A4, 1-inch margins = 6.3 in)
SINGLE_COLUMN = 6.3
HALF_COLUMN = 3.1
THESIS_WIDTHS = {"full": SINGLE_COLUMN, "half": HALF_COLUMN}
GOLDEN = 0.618

# Printed font sizes (pt) of thesis figures; audit_style() enforces this set
THESIS_FONT_SIZES = {"base": 9.0, "small": 8.0, "panel": 10.0}
ALLOWED_FONT_SIZES = tuple(sorted(set(THESIS_FONT_SIZES.values())))

# System sizes map to one fixed colour each, on a log-N scale from N = 4 to 14
# (the ED sizes), so N = 8 has the same colour in every figure of the thesis.
# Larger N (free-fermion series only) share the colour of N = 14.
_SIZE_CMAP = "viridis"
_SIZE_RANGE = (4, 14)
_SIZE_SPAN = (0.0, 0.8)           # avoid the pale yellow end of viridis


def size_color(n_qubits: int) -> str:
    """Fixed colour of system size N (log-N position on viridis)."""
    lo, hi = np.log(_SIZE_RANGE[0]), np.log(_SIZE_RANGE[1])
    t = (np.log(float(n_qubits)) - lo) / (hi - lo)
    t = float(np.clip(t, 0.0, 1.0))
    a, b = _SIZE_SPAN
    return to_hex(matplotlib.colormaps[_SIZE_CMAP](a + (b - a) * t))


# ---------------------------------------------------------------------------
# Phase 0 working style (used by phaseN_plots.py; unchanged)
# ---------------------------------------------------------------------------
def set_thesis_style(use_latex: bool = False) -> None:
    """
    Apply the Phase 0 working style globally.

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


def _metadata(ext: str) -> dict | None:
    """Deterministic file metadata: no creation date in PDFs."""
    if ext == "pdf":
        return {"CreationDate": None}
    return None


def save_figure(fig, name: str, subdir: str = "") -> None:
    """Save as figures/<subdir>/<name>.pdf and .png (Phase 0 working style)."""
    folder = FIGURES_DIR / subdir
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{ext}", metadata=_metadata(ext))
    print(f"[figure] saved {folder / name}.pdf / .png")


# ---------------------------------------------------------------------------
# Phase 8 thesis style (T8.2)
# ---------------------------------------------------------------------------
def thesis_rc(use_latex: bool = False) -> dict:
    """rcParams of the thesis style (printed size, fixed fonts and colours)."""
    base, small = THESIS_FONT_SIZES["base"], THESIS_FONT_SIZES["small"]
    return {
        "figure.dpi": 150,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.03,
        "figure.constrained_layout.w_pad": 0.03,
        "savefig.dpi": 300,
        "savefig.bbox": "standard",       # keep the printed size; no cropping
        "savefig.pad_inches": 0.0,
        "pdf.fonttype": 42,               # embed TrueType: selectable, editable text
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],   # DejaVu fills missing glyphs
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,       # proper minus signs with cmr10
        "axes.unicode_minus": True,
        "text.usetex": use_latex,
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "figure.titlesize": base,
        "legend.fontsize": small,
        "legend.title_fontsize": small,
        "xtick.labelsize": small,
        "ytick.labelsize": small,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.5,
        "errorbar.capsize": 1.5,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.borderaxespad": 0.3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
    }


def set_thesis_figure_style(use_latex: bool = False) -> None:
    """Apply the Phase 8 thesis style globally."""
    plt.rcParams.update(thesis_rc(use_latex))


@contextmanager
def thesis_style(use_latex: bool = False):
    """Context manager: thesis style inside, previous rcParams restored after."""
    with plt.rc_context(thesis_rc(use_latex)):
        yield


def thesis_figure(ncols: int = 1, nrows: int = 1, width: str = "full",
                  aspect: float | None = None, **subplot_kw):
    """
    New figure at printed size: width "full" (6.3 in) or "half" (3.1 in).

    aspect = height / width of the whole figure; default golden ratio for one
    panel per row, and 0.34 per row for three panels side by side.
    Returns (fig, axes) as plt.subplots does (squeeze=False is not applied).
    """
    w = THESIS_WIDTHS[width]
    if aspect is None:
        aspect = GOLDEN * nrows if ncols == 1 else (0.42 if ncols == 2 else 0.34) * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, w * aspect), **subplot_kw)
    return fig, axes


def panel_labels(axes, start: str = "a", inside: bool = False, loc=(0.04, 0.96), **kw) -> None:
    """
    Label panels (a), (b), ...: by default as a left-aligned title above each
    axis (never overlaps data); inside=True writes them into the top-left corner.
    """
    axes = list(np.ravel(axes))
    for k, ax in enumerate(axes):
        text = f"({chr(ord(start) + k)})"
        if inside:
            ax.text(loc[0], loc[1], text, transform=ax.transAxes, ha="left", va="top",
                    fontsize=THESIS_FONT_SIZES["panel"], **kw)
        else:
            ax.set_title(text, loc="left", fontsize=THESIS_FONT_SIZES["panel"], pad=3, **kw)


def mark_critical(ax, h_c: float = 1.0, **kw) -> None:
    """Thin dotted vertical line at the critical field."""
    style = dict(color=COLORS["grey"], lw=0.6, ls=":", zorder=0)
    style.update(kw)
    ax.axvline(h_c, **style)


def reference_line(ax, value: float, label: str | None = None, axis: str = "y", **kw) -> None:
    """Thin dashed line at an exact value (e.g. ln 2, pi, the exact central charge)."""
    style = dict(color=COLORS["black"], lw=0.6, ls="--", zorder=0)
    style.update(kw)
    (ax.axhline if axis == "y" else ax.axvline)(value, label=label, **style)


def add_secondary_scale(ax, factor: float, label: str, axis: str = "x"):
    """
    Second axis showing factor * (primary coordinate), e.g. the error budget
    lambda = (n_1q + 10 n_2q) p on top of a noise-strength axis. Works on log axes.
    """
    funcs = (lambda x: np.asarray(x) * factor, lambda y: np.asarray(y) / factor)
    sec = ax.secondary_xaxis("top", functions=funcs) if axis == "x" else \
        ax.secondary_yaxis("right", functions=funcs)
    (sec.set_xlabel if axis == "x" else sec.set_ylabel)(label)
    if axis == "x":
        ax.tick_params(axis="x", which="both", top=False)
    else:
        ax.tick_params(axis="y", which="both", right=False)
    return sec


def zero_strength_positions(strengths, decades_below: float = 1.0):
    """
    Positions on a log axis for a noise grid that starts at 0.

    Strength 0 is drawn one decade below the smallest positive strength and
    labelled "0" by label_zero_strength(). Returns (positions, x0).
    """
    s = np.asarray(strengths, dtype=float)
    x0 = s[s > 0].min() / 10.0 ** decades_below
    return np.where(s > 0, s, x0), x0


def label_zero_strength(ax, x0: float) -> None:
    """Add a '0' tick at x0 on a log x-axis (keeps the decade ticks)."""
    ticks = [t for t in ax.get_xticks() if t > x0 * 1.5]
    lo, hi = ax.get_xlim()
    ticks = [t for t in ticks if lo <= t <= hi]
    labels = [ax.xaxis.get_major_formatter()(t) for t in ticks]
    ax.set_xticks([x0] + ticks, ["0"] + labels)


def save_thesis_figure(fig, name: str, folder: Path, formats=("pdf", "png"),
                       png_dpi: int = 300) -> list:
    """Save a thesis figure at printed size; returns the written paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = folder / f"{name}.{ext}"
        fig.savefig(path, metadata=_metadata(ext), bbox_inches=None,
                    dpi=png_dpi if ext == "png" else None)
        paths.append(path)
    return paths


def audit_style(fig, width: str | float | None = None) -> list:
    """
    Check a thesis figure against the style rules (T8.2).

    Returns a list of (rule, passed, detail):
      * width: the figure is drawn at a thesis width (full or half);
      * fonts: every visible, non-empty text uses one of ALLOWED_FONT_SIZES;
      * series colours: every line or marker labelled ED, HVA or HEA (or
        starting with that word) uses its SERIES_COLORS colour.
    """
    results = []
    w = fig.get_size_inches()[0]
    targets = list(THESIS_WIDTHS.values()) if width is None else \
        [THESIS_WIDTHS[width] if isinstance(width, str) else float(width)]
    ok = any(abs(w - t) < 1e-6 for t in targets)
    results.append(("figure width", ok, f"{w:.3f} in"))

    fig.canvas.draw()
    bad = set()
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        size = round(float(t.get_fontsize()), 2)
        if not any(abs(size - a) < 1e-6 for a in ALLOWED_FONT_SIZES):
            bad.add((size, t.get_text()[:30]))
    detail = "all in " + ", ".join(f"{a:g}" for a in ALLOWED_FONT_SIZES) + " pt" if not bad else \
        "; ".join(f"{s:g} pt '{txt}'" for s, txt in sorted(bad)[:5])
    results.append(("font sizes", not bad, detail))

    wrong = []
    for artist in fig.findobj(lambda a: hasattr(a, "get_label") and hasattr(a, "get_color")):
        label = str(artist.get_label())
        key = label.split()[0] if label and not label.startswith("_") else ""
        if key in ("ED", "HVA", "HEA"):
            colors = np.atleast_2d(matplotlib.colors.to_rgba_array(artist.get_color()))
            want = matplotlib.colors.to_rgba(SERIES_COLORS[key])
            if not np.allclose(colors[0][:3], want[:3], atol=1e-3):
                wrong.append(label)
    results.append(("series colours", not wrong,
                    "ED, HVA, HEA as fixed" if not wrong else "wrong: " + ", ".join(wrong[:5])))
    return results
