"""
Phase 1 figures and numerical summary, made only from saved data.

Run from the project root, after experiments/phase1_ed_sweep.py:

    python experiments/phase1_plots.py

Figures (figures/phase1/*.pdf and *.png):
    tfim_entropy          half-chain entropy vs h, open and periodic chains
    tfim_gap              full and in-sector gaps, CFT finite-size scaling at h_c
    tfim_order            squared magnetization and Binder cumulant
    tfim_correlations     two-point correlator C(r) at several fields
    tfim_calabrese_cardy  entropy profile at h_c with fits; c versus N
    xxz_overview          half-chain entropy and Neel order vs anisotropy
    xxz_calabrese_cardy   XX-point entropy profile (even/odd ell) and c versus N
    cluster_string_order  string order vs h1 for every h2

Numbers quoted in the thesis go to results/phase1/phase1_summary.txt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import CFG, RESULTS_DIR
from analysis.entanglement import calabrese_cardy_fit, chord_log
from analysis.observables import centred_correlator
from utils.io import load_results, result_exists
from utils.plotting import COLORS, PALETTE, SINGLE_COLUMN, save_figure, set_thesis_style

SUBDIR = "phase1"
SIZES = list(CFG.system.sizes_exact)
LN2 = np.log(2.0)
SUMMARY = []


def note(line: str = "") -> None:
    print(line)
    SUMMARY.append(line)


def load(name: str) -> dict:
    if not result_exists(name, SUBDIR):
        raise FileNotFoundError(f"results/{SUBDIR}/{name} missing: "
                                "run experiments/phase1_ed_sweep.py first")
    return load_results(name, SUBDIR)[0]


def size_color(n: int) -> str:
    return PALETTE[SIZES.index(n) % len(PALETTE)] if n in SIZES else COLORS["black"]


def critical_line(ax, x: float, label: str = None) -> None:
    ax.axvline(x, color="0.5", ls=":", lw=1.2, label=label)


def panel_label(ax, text: str, loc: str = "upper left") -> None:
    x, y, va = {"upper left": (0.03, 0.96, "top"), "lower left": (0.03, 0.04, "bottom"),
                "upper right": (0.97, 0.96, "top"), "lower right": (0.97, 0.04, "bottom")}[loc]
    ax.text(x, y, text, transform=ax.transAxes, va=va,
            ha="right" if loc.endswith("right") else "left", fontweight="bold")


def extrapolate_inverse_n(sizes, values, n_last: int = 3) -> float:
    """Linear fit of values vs 1/N over the largest n_last sizes; intercept at 1/N = 0."""
    x = 1.0 / np.asarray(sizes[-n_last:], float)
    return float(np.polyfit(x, np.asarray(values[-n_last:]), 1)[1])


# ---------------------------------------------------------------------------
# TFIM
# ---------------------------------------------------------------------------
def fig_tfim_entropy():
    hc = CFG.tfim.h_critical
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.45), sharey=True)
    note("TFIM half-chain entropy (even-parity ground state)")
    for ax, prefix, label in ((axes[0], "ed_tfim_N", "open"), (axes[1], "ed_tfim_pbc_N", "periodic")):
        for n in SIZES[1:]:
            d = load(f"{prefix}{n}")
            h, S = d["h"], d["S_half"]
            ax.plot(h, S, color=size_color(n), label=f"$N={n}$")
            dS = np.gradient(S, h)
            note(f"  {label:8s} N={n:2d}: max at h = {h[np.argmax(S)]:.2f} (S = {S.max():.4f}), "
                 f"steepest drop at h = {h[np.argmin(dS)]:.2f}, "
                 f"S(h_c) = {S[np.argmin(abs(h - hc))]:.4f}")
        ax.axhline(LN2, color="0.4", ls="--", lw=1.0)
        critical_line(ax, hc)
        ax.set_xlabel("$h/J$")
        panel_label(ax, f"({'a' if label == 'open' else 'b'}) {label}", loc="lower left")
    axes[0].text(1.95, LN2 + 0.015, r"$\ln 2$", ha="right", va="bottom", color="0.3")
    axes[0].set_ylabel(r"$S_{\mathrm{vN}}(N/2)$")
    axes[1].legend(loc="center right", ncol=1, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "tfim_entropy", SUBDIR)
    plt.close(fig)
    note()


def fig_tfim_gap():
    hc = CFG.tfim.h_critical
    fig, axes = plt.subplots(1, 3, figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.36))
    full_at_hc, sector_at_hc = [], []
    for n in SIZES:
        d = load(f"ed_tfim_N{n}")
        h = d["h"]
        axes[0].semilogy(h, np.maximum(d["gap"], 1e-16), color=size_color(n), label=f"$N={n}$")
        axes[1].plot(h, d["gap_sector"], color=size_color(n))
        i = np.argmin(abs(h - hc))
        full_at_hc.append(d["gap"][i] * n)
        sector_at_hc.append(d["gap_sector"][i] * n)

    for ax in axes[:2]:
        critical_line(ax, hc)
        ax.set_xlabel("$h/J$")
    axes[0].set_ylabel(r"$E_1 - E_0$")
    axes[0].set_ylim(1e-16, 10)
    axes[1].set_ylabel(r"$\Delta_{\mathrm{even}}$")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")

    inv_n = 1.0 / np.asarray(SIZES, float)
    a_full = extrapolate_inverse_n(SIZES, full_at_hc)
    a_sec = extrapolate_inverse_n(SIZES, sector_at_hc)
    ax = axes[2]
    xs = np.linspace(0, inv_n.max() * 1.05, 20)
    for vals, scale, marker, color, label in ((full_at_hc, 1, "o", COLORS["blue"], "full"),
                                              (sector_at_hc, 4, "s", COLORS["orange"], "even / 4")):
        y = np.asarray(vals) / scale
        ax.plot(inv_n, y, marker, color=color, label=label)
        slope, icpt = np.polyfit(inv_n[-3:], y[-3:], 1)
        ax.plot(xs, slope * xs + icpt, "-", color=color, lw=1.0)
    ax.axhline(np.pi, color="0.4", ls="--", lw=1.0)
    ax.text(0.2, np.pi + 0.015, r"$\pi$", color="0.3", va="bottom")
    ax.set_xlim(0, inv_n.max() * 1.1)
    ax.set_ylim(None, np.pi + 0.12)
    ax.set_xlabel("$1/N$")
    ax.set_ylabel(r"$N\,\Delta(h_c)$")
    ax.legend(loc="lower left", fontsize=8)
    panel_label(ax, "(c)", loc="upper right")
    axes[0].set_ylim(1e-16, 1e4)
    axes[0].legend(fontsize=6.5, loc="lower right", ncol=2, handlelength=1.2, columnspacing=0.8)
    fig.tight_layout()
    save_figure(fig, "tfim_gap", SUBDIR)
    plt.close(fig)

    note("TFIM gap at h_c (boundary CFT: N*gap -> pi * v * x, v = 2J)")
    for n, f, s in zip(SIZES, full_at_hc, sector_at_hc):
        note(f"  N={n:2d}: N*(E1-E0) = {f:.4f}   N*gap_even = {s:.4f}")
    note(f"  1/N extrapolation: full {a_full:.4f} (pi = {np.pi:.4f}, "
         f"error {abs(a_full / np.pi - 1) * 100:.2f}%), "
         f"even {a_sec:.4f} (4 pi = {4 * np.pi:.4f}, error {abs(a_sec / (4 * np.pi) - 1) * 100:.2f}%)")
    note()


def fig_tfim_order():
    hc = CFG.tfim.h_critical
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.45))
    binder = {}
    for n in SIZES:
        d = load(f"ed_tfim_N{n}")
        axes[0].plot(d["h"], d["M2"], color=size_color(n), label=f"$N={n}$")
        axes[1].plot(d["h"], d["binder"], color=size_color(n))
        binder[n] = (d["h"], d["binder"])
    for ax in axes:
        critical_line(ax, hc)
        ax.set_xlabel("$h/J$")
    axes[0].set_ylabel(r"$\langle M^2 \rangle$")
    axes[1].set_ylabel(r"$U_4$")
    axes[0].legend(fontsize=8)
    panel_label(axes[0], "(a)", loc="lower left")
    panel_label(axes[1], "(b)", loc="lower left")
    fig.tight_layout()
    save_figure(fig, "tfim_order", SUBDIR)
    plt.close(fig)

    note("TFIM Binder-cumulant crossings (linear interpolation)")
    for n1, n2 in zip(SIZES[1:-1], SIZES[2:]):
        h, u1 = binder[n1]
        diff = u1 - binder[n2][1]
        k = np.flatnonzero(np.diff(np.sign(diff)) != 0)
        if len(k):
            k = k[0]
            hx = h[k] - diff[k] * (h[k + 1] - h[k]) / (diff[k + 1] - diff[k])
            note(f"  N={n1:2d} / N={n2:2d}: h* = {hx:.4f}")
    note()


def fig_tfim_correlations():
    n = SIZES[-1]
    d = load(f"ed_tfim_N{n}")
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN * 0.6, SINGLE_COLUMN * 0.45))
    for k, h_target in enumerate((0.5, 0.9, 1.0, 1.1, 1.5)):
        i = np.argmin(abs(d["h"] - h_target))
        C = centred_correlator(d["zz_corr"][i])
        ax.semilogy(np.arange(1, n), C, "o-", ms=3, color=PALETTE[k], label=f"$h={d['h'][i]:.1f}$")
    ax.set_xlabel("$r$")
    ax.set_ylabel(r"$\langle Z_i Z_{i+r} \rangle$")
    ax.set_title(f"TFIM, $N={n}$, open", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "tfim_correlations", SUBDIR)
    plt.close(fig)


def central_charges(key_fmt: str, boundary: str, parity=None) -> list:
    out = []
    for n in SIZES:
        if n < 6:
            continue
        profile = load(f"ed_cc_profiles_N{n}")[key_fmt.format(b=boundary)]
        fit = calabrese_cardy_fit(profile, n, boundary, ell_min=CFG.ed.cc_ell_min, parity=parity)
        out.append((n, fit["c"], fit["c_err"]))
    return out


def fig_tfim_cc():
    n = SIZES[-1]
    cc = load(f"ed_cc_profiles_N{n}")
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.45))
    ax = axes[0]
    for boundary, color in (("open", COLORS["blue"]), ("periodic", COLORS["orange"])):
        S = cc[f"tfim_{boundary}"]
        x = chord_log(np.arange(1, n), n, boundary)
        fit = calabrese_cardy_fit(S, n, boundary, ell_min=CFG.ed.cc_ell_min)
        pref = 6 if boundary == "open" else 3
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(x, S, "o", color=color, label=f"{boundary}, $c={fit['c']:.3f}$")
        ax.plot(xs, fit["c"] / pref * xs + fit["const"], "-", color=color, lw=1.2)
    ax.set_xlabel(r"$\ln[\,(\eta N/\pi)\sin(\pi\ell/N)\,]$")
    ax.set_ylabel(r"$S_{\mathrm{vN}}(\ell)$")
    ax.legend(fontsize=8, loc="center right")
    panel_label(ax, f"(a) $N={n}$")

    ax = axes[1]
    note("TFIM central charge at h_c (Calabrese-Cardy, ell_min = %d)" % CFG.ed.cc_ell_min)
    for boundary, marker, color in (("open", "o", COLORS["blue"]), ("periodic", "s", COLORS["orange"])):
        res = central_charges("tfim_{b}", boundary)
        ns, cs, es = map(np.asarray, zip(*res))
        ax.errorbar(1 / ns, cs, yerr=es, fmt=marker, color=color, label=boundary, capsize=2)
        for nn, c, e in res:
            note(f"  {boundary:8s} N={nn:2d}: c = {c:.4f} +/- {e:.4f}")
    ax.axhline(0.5, color="0.4", ls="--", lw=1.0)
    ax.set_xlim(0, None)
    ax.set_xlabel("$1/N$")
    ax.set_ylabel("$c$")
    ax.legend(fontsize=8)
    panel_label(ax, "(b)")
    fig.tight_layout()
    save_figure(fig, "tfim_calabrese_cardy", SUBDIR)
    plt.close(fig)
    note("  (eta = 2 for open, 1 for periodic chains; exact value c = 1/2)")
    note()


# ---------------------------------------------------------------------------
# XXZ
# ---------------------------------------------------------------------------
def fig_xxz_overview():
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.45))
    # Open chains dimerize: the cut at ell = N/2 crosses a strong bond when N/2 is
    # odd and a weak bond when N/2 is even, so the two families are drawn differently.
    for n in SIZES[1:]:
        d = load(f"ed_xxz_N{n}")
        ls = "-" if (n // 2) % 2 else "--"
        axes[0].plot(d["delta"], d["S_half"], ls, color=size_color(n), label=f"$N={n}$")
        axes[1].plot(d["delta"], d["Ms2"], ls, color=size_color(n))
    for ax in axes:
        critical_line(ax, 1.0)
        ax.set_xlabel(r"$\Delta$")
    axes[0].set_ylabel(r"$S_{\mathrm{vN}}(N/2)$")
    axes[1].set_ylabel(r"$\langle M_{\mathrm{s}}^2 \rangle$")
    axes[0].legend(fontsize=7, ncol=2, loc="lower left")
    axes[0].set_title(r"solid: $N/2$ odd, dashed: $N/2$ even", fontsize=9)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.tight_layout()
    save_figure(fig, "xxz_overview", SUBDIR)
    plt.close(fig)

    note("XXZ ground-state symmetry sector (expected (-1)^(N/2), Marshall sign rule)")
    for n in SIZES:
        d = load(f"ed_xxz_N{n}")
        labels = d["sector_labels"][np.unique(d["ground_sector"])].ravel().tolist()
        note(f"  N={n:2d}: spin-flip parity {labels}, expected {(-1) ** (n // 2)}")
    note()


def fig_xxz_cc():
    n = SIZES[-1]
    cc = load(f"ed_cc_profiles_N{n}")
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.45))
    ax = axes[0]
    S = cc["xxz_d0_open"]
    ell = np.arange(1, n)
    x = chord_log(ell, n, "open")
    for parity, marker, color in (("even", "o", COLORS["blue"]), ("odd", "s", COLORS["orange"])):
        m = ell % 2 == (0 if parity == "even" else 1)
        fit = calabrese_cardy_fit(S, n, "open", ell_min=CFG.ed.cc_ell_min, parity=parity)
        ax.plot(x[m], S[m], marker, color=color, label=rf"$\ell$ {parity}, $c={fit['c']:.2f}$")
    Sp = cc["xxz_d0_periodic"]
    fit = calabrese_cardy_fit(Sp, n, "periodic", ell_min=CFG.ed.cc_ell_min)
    ax.plot(chord_log(ell, n, "periodic"), Sp, "^", color=COLORS["green"],
            label=rf"periodic, $c={fit['c']:.3f}$")
    ax.set_xlabel(r"$\ln[\,(\eta N/\pi)\sin(\pi\ell/N)\,]$")
    ax.set_ylabel(r"$S_{\mathrm{vN}}(\ell)$")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(rf"$\Delta=0$, $N={n}$, open and periodic", fontsize=9)
    panel_label(ax, "(a)", loc="lower right")

    ax = axes[1]
    note("XXZ central charge (exact value c = 1)")
    series = [("d0", "periodic", None, r"$\Delta=0$ periodic", "s", COLORS["green"]),
              ("d0p5", "periodic", None, r"$\Delta=0.5$ periodic", "D", COLORS["purple"]),
              ("d0", "open", "even", r"$\Delta=0$ open, even $\ell$", "o", COLORS["blue"]),
              ("d0", "open", "odd", r"$\Delta=0$ open, odd $\ell$", "v", COLORS["orange"])]
    for tag, boundary, parity, label, marker, color in series:
        res = [r for r in central_charges(f"xxz_{tag}_{{b}}", boundary, parity)
               if np.isfinite(r[1])]
        ns, cs, es = map(np.asarray, zip(*res))
        ax.errorbar(1 / ns, cs, yerr=es, fmt=marker, color=color, label=label, capsize=2)
        for nn, c, e in res:
            note(f"  {label.replace('$', '').replace(chr(92), ''):28s} N={nn:2d}: c = {c:.4f} +/- {e:.4f}")
    ax.axhline(1.0, color="0.4", ls="--", lw=1.0)
    ax.set_xlim(0, None)
    ax.set_xlabel("$1/N$")
    ax.set_ylabel("$c$")
    ax.legend(fontsize=7)
    panel_label(ax, "(b)")
    fig.tight_layout()
    save_figure(fig, "xxz_calabrese_cardy", SUBDIR)
    plt.close(fig)
    note()


# ---------------------------------------------------------------------------
# Cluster-Ising
# ---------------------------------------------------------------------------
def fig_cluster():
    from experiments.phase1_ed_sweep import h2_tag
    h2_values = CFG.cluster.h2_values
    fig, axes = plt.subplots(1, len(h2_values), figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.38),
                             sharey=True)
    note("Cluster-Ising string order")
    for ax, h2 in zip(axes, h2_values):
        for n in SIZES[1:]:
            d = load(f"ed_cluster_N{n}_h2_{h2_tag(h2)}")
            ax.plot(d["h1"], d["string_order"], "o-", ms=2.5, color=size_color(n), label=f"$N={n}$")
        d = load(f"ed_cluster_N{SIZES[-1]}_h2_{h2_tag(h2)}")
        note(f"  h2 = {h2:+.1f}, N={SIZES[-1]}: " + ", ".join(
            f"S({h1:.1f}) = {s:.3f}" for h1, s in zip(d["h1"][::4], d["string_order"][::4])))
        ax.set_xlabel("$h_1/J$")
        ax.set_title(f"$h_2/J = {h2:g}$", fontsize=10)
    axes[0].set_ylabel(r"$\mathcal{S}$")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, "cluster_string_order", SUBDIR)
    plt.close(fig)
    note()


def main():
    set_thesis_style()
    note("PHASE 1 SUMMARY: exact diagonalization")
    note("=" * 60)
    fig_tfim_entropy()
    fig_tfim_gap()
    fig_tfim_order()
    fig_tfim_correlations()
    fig_tfim_cc()
    fig_xxz_overview()
    fig_xxz_cc()
    fig_cluster()
    out = RESULTS_DIR / SUBDIR / "phase1_summary.txt"
    out.write_text("\n".join(SUMMARY) + "\n", encoding="utf-8")
    print(f"[summary] written to {out}")


if __name__ == "__main__":
    main()
