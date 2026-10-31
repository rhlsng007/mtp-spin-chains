"""
Phase 3 figures, numerical summary and validation checks, made only from saved data.

Run from the project root, after experiments\\phase3_entanglement.py:

    python experiments\\phase3_plots.py

Figures (figures/phase3/*.pdf and *.png):
    ent_vs_field            half-chain entropy vs h, VQE over ED (even and symmetry-broken)  T3.1, T3.4
    ent_error_vs_field      |S_VQE - S_ref| vs h at p = N, all sizes                         T3.4
    cc_profiles             S(ell) at h_c vs the chord variable, with Calabrese-Cardy fits     T3.2
    central_charge_vs_depth c(p): open chains (HVA, HEA, ED) and periodic HVA                 T3.2
    max_entanglement        largest S_2 per depth vs the bound; fidelity vs the rank bound     T3.3
    entanglement_vs_fidelity entropy error vs infidelity, with the Fannes-Audenaert bound      T3.4

Numbers and PASS / FAIL / NOTE lines go to results/phase3/phase3_summary.txt.
Only the sizes found on disk are used.
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
from analysis.vqe_entanglement import LN2, audenaert_bound, entropy_bound, page_entropy
from utils.io import load_results, result_exists
from utils.plotting import COLORS, PALETTE, SERIES_COLORS, SINGLE_COLUMN, save_figure, set_thesis_style

SUBDIR = "phase3"
ANSATZE = ("hva", "hea")
LABEL = {"hva": "HVA", "hea": "HEA"}
LINESTYLE = {"hva": "-", "hea": "--"}
MARKER = {"hva": "o", "hea": "s"}
FLOOR = 1e-15
SUMMARY, CHECKS = [], []


def note(line: str = "") -> None:
    print(line)
    SUMMARY.append(line)


def check(name: str, ok: bool, detail: str) -> None:
    CHECKS.append(("PASS" if ok else "FAIL", name))
    note(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def observe(name: str, detail: str) -> None:
    CHECKS.append(("NOTE", name))
    note(f"  [NOTE] {name}: {detail}")


def load(name: str):
    return load_results(name, SUBDIR)[0] if result_exists(name, SUBDIR) else None


def size_color(n: int) -> str:
    sizes = list(CFG.system.sizes_noiseless)
    return PALETTE[sizes.index(n) % len(PALETTE)] if n in sizes else COLORS["black"]


def critical_line(ax):
    ax.axvline(CFG.tfim.h_critical, color="0.5", ls=":", lw=1.2)


def hc_index(h) -> int:
    return int(np.argmin(np.abs(np.asarray(h) - CFG.ent.cc_field)))


def cc(profile, n, boundary="open") -> dict:
    return calabrese_cardy_fit(profile, n, boundary, ell_min=CFG.ent.cc_ell_min)


def ordered_mask(ed: dict) -> np.ndarray:
    """Fields where both parity sectors belong to the ground space."""
    return ed["ground_space"].all(axis=1)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_ent_vs_field(ed, ent):
    n = max(ed)
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.42), sharey=True)
    e = ed[n]
    half = n // 2 - 1
    ordered = ordered_mask(e)
    for ax, a in zip(axes, ANSATZE):
        if (a, n) not in ent:
            continue
        d = ent[(a, n)]
        picks = sorted({1, max(n // 4, 1), n // 2, n})
        for k, p in enumerate(picks):
            ax.plot(d["h"], d["S"][p - 1, :, half], color=PALETTE[k % len(PALETTE)], label=f"$p={p}$")
        ax.plot(e["h"], e["S_even"][:, half], color=SERIES_COLORS["ED"], lw=1.4, label="ED, even")
        ax.plot(e["h"][ordered], e["S_broken"][ordered, half], color=SERIES_COLORS["ED"], lw=1.4,
                ls="--", label="ED, broken")
        critical_line(ax)
        ax.set_xlabel("$h/J$")
        ax.set_title(f"{LABEL[a]}, $N={n}$", fontsize=10)
    axes[0].set_ylabel(r"$S_{\mathrm{vN}}(N/2)$")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save_figure(fig, "ent_vs_field", SUBDIR)
    plt.close(fig)


def fig_ent_error_vs_field(ent):
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.42), sharey=True)
    for ax, a in zip(axes, ANSATZE):
        for (aa, n), d in sorted(ent.items()):
            if aa != a:
                continue
            half = n // 2 - 1
            err = np.abs(d["S"][-1, :, half] - d["S_ref"][-1, :, half])
            ax.semilogy(d["h"], np.maximum(err, FLOOR), color=size_color(n), label=f"$N={n}$")
        critical_line(ax)
        ax.set_xlabel("$h/J$")
        ax.set_title(f"{LABEL[a]}, $p=N$", fontsize=10)
    axes[0].set_ylabel(r"$|S_{\mathrm{VQE}} - S_{\mathrm{ref}}|$")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, "ent_error_vs_field", SUBDIR)
    plt.close(fig)


def fig_cc_profiles(ed, ent):
    n = max(ed)
    i = hc_index(ed[n]["h"])
    # S(ell) for ell > N/2 is a different cut and differs for states without mirror
    # symmetry (the HEA); the fits use ell <= N/2, so only those cuts are drawn
    keep = np.arange(1, n) <= n // 2
    x = chord_log(np.arange(1, n)[keep], n, "open")
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.42), sharey=True)
    for ax, a in zip(axes, ANSATZE):
        if (a, n) not in ent:
            continue
        d = ent[(a, n)]
        for k, p in enumerate(sorted({2, n // 2, n})):
            f = cc(d["S"][p - 1, i], n)
            prof = d["S"][p - 1, i][keep]
            ax.plot(x, prof, MARKER[a], ms=3.5, color=PALETTE[k % len(PALETTE)],
                    label=f"$p={p}$, $c={f['c']:.3f}$")
            xs = np.linspace(x.min(), x.max(), 40)
            ax.plot(xs, f["c"] / 6 * xs + f["const"], "-", lw=1.0, color=PALETTE[k % len(PALETTE)])
        f = cc(ed[n]["S_even"][i], n)
        prof = ed[n]["S_even"][i][keep]
        ax.plot(x, prof, "x", color=SERIES_COLORS["ED"], label=f"ED, $c={f['c']:.3f}$")
        ax.set_xlabel(r"$\ln[(2N/\pi)\sin(\pi\ell/N)]$")
        ax.set_title(f"{LABEL[a]}, $N={n}$, $h=h_c$", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel(r"$S_{\mathrm{vN}}(\ell)$")
    fig.tight_layout()
    save_figure(fig, "cc_profiles", SUBDIR)
    plt.close(fig)


def central_charges_open(ed, ent):
    """{(ansatz, N): array of c(p)} and {N: c_ED}, open chains at h_c."""
    c_vqe, c_ed = {}, {}
    for n, e in ed.items():
        if n < 6:
            continue
        i = hc_index(e["h"])
        c_ed[n] = cc(e["S_even"][i], n)["c"]
        for a in ANSATZE:
            if (a, n) in ent:
                c_vqe[(a, n)] = np.array([cc(ent[(a, n)]["S"][k, i], n)["c"]
                                          for k in range(len(ent[(a, n)]["depths"]))])
    return c_vqe, c_ed


def central_charges_pbc(pbc):
    out = {}
    for n, d in pbc.items():
        out[n] = (np.array([cc(s, n, "periodic")["c"] for s in d["S"]]),
                  cc(d["S_ed"], n, "periodic")["c"])
    return out


def fig_central_charge(c_vqe, c_ed, c_pbc):
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.45))
    ax = axes[0]
    for (a, n), cs in sorted(c_vqe.items()):
        if n < 8:
            continue
        ax.plot(np.arange(1, len(cs) + 1), cs, ls=LINESTYLE[a], marker=MARKER[a], ms=3,
                color=size_color(n))
    for n, c in c_ed.items():
        if n >= 8:
            ax.axhline(c, color=size_color(n), ls=":", lw=1.0)
    ax.axhline(0.5, color="0.4", ls="--", lw=1.0)
    handles = [plt.Line2D([], [], color="0.3", ls=LINESTYLE[a], marker=MARKER[a], ms=3,
                          label=LABEL[a]) for a in ANSATZE]
    handles += [plt.Line2D([], [], color=size_color(n), label=f"$N={n}$") for n in sorted(c_ed) if n >= 8]
    handles += [plt.Line2D([], [], color="0.3", ls=":", label="ED (open)")]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    ax.set_xlabel("depth $p$")
    ax.set_ylabel("$c$ (open chain)")
    ax.set_title("(a) open chains, $h=h_c$", fontsize=10)

    ax = axes[1]
    for n, (cs, c_ed_p) in sorted(c_pbc.items()):
        if n < 6:
            continue
        ax.plot(np.arange(1, len(cs) + 1), cs, "o-", ms=3, color=size_color(n), label=f"$N={n}$")
        ax.axhline(c_ed_p, color=size_color(n), ls=":", lw=1.0)
    ax.axhline(0.5, color="0.4", ls="--", lw=1.0)
    ax.set_xlabel("depth $p$")
    ax.set_ylabel("$c$ (periodic chain)")
    ax.set_title("(b) periodic HVA, $h=h_c$", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "central_charge_vs_depth", SUBDIR)
    plt.close(fig)


def fig_max_entanglement(maxent, ent, ed):
    n = max(n for (_, n) in maxent)
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.45))
    ax = axes[0]
    depths = np.arange(1, n + 1)
    ax.plot(depths, np.minimum(depths, n // 2), color="0.4", lw=1.0, label="rank bound")
    for a in ANSATZE:
        if (a, n) not in maxent:
            continue
        d = maxent[(a, n)]
        c = SERIES_COLORS[LABEL[a]]
        ax.plot(d["depths"], d["S2_max"] / LN2, MARKER[a], color=c, ms=4, label=f"{LABEL[a]} max $S_2$")
        ax.plot(d["depths"], d["random_S_mean"] / LN2, ls=LINESTYLE[a], color=c, lw=1.2,
                label=f"{LABEL[a]} random, mean $S$")
    ax.axhline(page_entropy(n) / LN2, color="0.4", ls="-.", lw=1.0, label="Page")
    if n in ed:
        half = n // 2 - 1
        ax.axhline(ed[n]["S_even"][:, half].max() / LN2, color=SERIES_COLORS["ED"], ls=":", lw=1.2,
                   label="ED max over $h$")
    ax.set_xlabel("depth $p$")
    ax.set_ylabel(r"$S(N/2)\,/\ln 2$")
    ax.set_title(f"(a) entanglement capacity, $N={n}$", fontsize=10)
    ax.legend(fontsize=6.5, loc="lower right")

    ax = axes[1]
    for a in ANSATZE:
        if (a, n) not in ent:
            continue
        d = ent[(a, n)]
        i = hc_index(d["h"])
        c = SERIES_COLORS[LABEL[a]]
        ax.semilogy(d["depths"], np.maximum(1 - d["fidelity_gs"][:, i], FLOOR), ls=LINESTYLE[a],
                    marker=MARKER[a], ms=3, color=c, label=f"{LABEL[a]} $1-F_{{\\mathrm{{GS}}}}$")
    a0 = next(a for a in ANSATZE if (a, n) in ent)
    d = ent[(a0, n)]
    i = hc_index(d["h"])
    # at h_c the ground space is the even state alone, so both ansaetze share this bound
    ax.semilogy(d["depths"], np.maximum(1 - d["rank_fidelity_bound"][:, i], FLOOR), ":",
                color=SERIES_COLORS["ED"], lw=1.4, label="smallest allowed by rank $2^p$")
    ax.set_xlabel("depth $p$")
    ax.set_ylabel("infidelity at $h_c$")
    ax.set_title(f"(b) achieved vs allowed by rank, $N={n}$", fontsize=10)
    ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    save_figure(fig, "max_entanglement", SUBDIR)
    plt.close(fig)


def fig_entanglement_vs_fidelity(ent):
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN * 0.62, SINGLE_COLUMN * 0.5))
    for (a, n), d in sorted(ent.items()):
        half = n // 2 - 1
        x = np.maximum(1 - d["fidelity_gs"].ravel(), FLOOR)
        y = np.maximum(np.abs(d["S"][:, :, half] - d["S_ref"][:, :, half]).ravel(), FLOOR)
        ax.loglog(x, y, MARKER[a], ms=1.8, alpha=0.5, color=size_color(n), mec="none")
    n = max(n for (_, n) in ent)
    xs = np.logspace(-15, 0, 200)
    ax.loglog(xs, audenaert_bound(1 - xs, 2 ** (n // 2)), color="0.2", lw=1.2,
              label=f"Fannes–Audenaert, $N={n}$")
    ax.set_xlabel(r"$1 - F_{\mathrm{GS}}$")
    ax.set_ylabel(r"$|S_{\mathrm{VQE}} - S_{\mathrm{ref}}|$ at $N/2$")
    handles = [plt.Line2D([], [], color="0.3", marker=MARKER[a], ls="none", label=LABEL[a])
               for a in ANSATZE]
    handles += [plt.Line2D([], [], color=size_color(nn), label=f"$N={nn}$")
                for nn in sorted({nn for (_, nn) in ent})]
    handles += [plt.Line2D([], [], color="0.2", label="Fannes–Audenaert bound")]
    ax.legend(handles=handles, fontsize=6.5, loc="lower right")
    ax.set_ylim(1e-15, 2)
    fig.tight_layout()
    save_figure(fig, "entanglement_vs_fidelity", SUBDIR)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Checks and summary
# ---------------------------------------------------------------------------
def summarize(ed, ent, maxent, pbc, c_vqe, c_ed, c_pbc):
    tol = CFG.ent.bound_tol
    note("Half-chain entropy at h_c (p = N) and in the ordered phase (h = 0.1)")
    for n in sorted(ed):
        e = ed[n]
        half = n // 2 - 1
        i_c, i_0 = hc_index(e["h"]), 0
        line = (f"  N={n:2d}: ED even S(h_c) = {e['S_even'][i_c, half]:.4f}, "
                f"S(0.1) = {e['S_even'][i_0, half]:.4f}; broken S(0.1) = {e['S_broken'][i_0, half]:.4f}")
        for a in ANSATZ_ORDER(ent, n):
            d = ent[(a, n)]
            line += f"; {LABEL[a]} S(h_c) = {d['S'][-1, i_c, half]:.4f}, S(0.1) = {d['S'][-1, i_0, half]:.4f}"
        note(line)
    note()

    note("Location of the entropy maximum over h (p = N, half chain)")
    for n in sorted(ed):
        e = ed[n]
        half = n // 2 - 1
        parts = []
        for a in ANSATZ_ORDER(ent, n):
            d = ent[(a, n)]
            parts.append(f"{LABEL[a]} h = {d['h'][np.argmax(d['S'][-1, :, half])]:.2f}")
        ordered = ordered_mask(e)
        broken_like = np.where(ordered, e["S_broken"][:, half], e["S_even"][:, half])
        parts.append(f"ED broken-or-even h = {e['h'][np.argmax(broken_like)]:.2f}")
        note(f"  N={n:2d}: " + ", ".join(parts))
    note()

    note("Central charge at h_c (Calabrese-Cardy, ell_min = %d, ell <= N/2)" % CFG.ent.cc_ell_min)
    for n in sorted(c_ed):
        parts = [f"ED open {c_ed[n]:.4f}"]
        for a in ANSATZE:
            if (a, n) in c_vqe:
                cs = c_vqe[(a, n)]
                parts.append(f"{LABEL[a]} p=N/2 {cs[n // 2 - 1]:.4f}, p=N {cs[-1]:.4f}")
        note(f"  N={n:2d}: " + "; ".join(parts))
    for n in sorted(c_pbc):
        cs, ce = c_pbc[n]
        if n >= 6:
            note(f"  periodic HVA N={n:2d}: c(p) = " + ", ".join(f"{c:.4f}" for c in cs)
                 + f"; ED periodic {ce:.4f}")
    note()

    note("Maximal half-chain entanglement per depth (T3.3)")
    for (a, n), d in sorted(maxent.items()):
        ratio = d["S2_max"] / (np.minimum(d["depths"], n // 2) * LN2)
        note(f"  {LABEL[a]} N={n:2d}: min S2_max / bound = {ratio.min():.4f}; random mean S at p = N: "
             f"{d['random_S_mean'][-1]:.3f} (Page {page_entropy(n):.3f}, max {n // 2} ln 2 = {n // 2 * LN2:.3f})")
    note()

    note("Validation checks")
    # 1 ED entropies reproduce Phase 1
    diffs = [np.abs(e["S_even"][:, n // 2 - 1] - e["S_half_phase1"]).max()
             for n, e in ed.items() if "S_half_phase1" in e]
    if diffs:
        check("ED entropies reproduce Phase 1", max(diffs) < 1e-10, f"max difference {max(diffs):.1e}")
    else:
        observe("ED entropies reproduce Phase 1", "Phase 1 files not found; skipped")

    # 2, 3 Schmidt rank and entropy bounds
    rank_ok, ent_ok, worst = True, True, -np.inf
    for (a, n), d in ent.items():
        for k, p in enumerate(d["depths"]):
            ell = np.arange(1, n)
            rmax = 2 ** np.minimum(np.minimum(ell, n - ell), p)
            rank_ok &= bool(np.all(d["rank"][k] <= rmax))
            b = entropy_bound(int(p), n)
            excess = max((d["S"][k] - b).max(), (d["S2"][k] - b).max())
            worst = max(worst, excess)
    ent_ok = worst <= tol
    check("Schmidt rank <= 2^p at every cut (all VQE states)", rank_ok,
          "one two-qubit gate per layer crosses any cut")
    check("S and S_2 <= min(p, ell, N - ell) ln 2 (all VQE states)", ent_ok,
          f"largest excess {worst:.1e}")

    # 4 Fannes-Audenaert, with 1 - F floored at the resolution of a computed fidelity
    worst = -np.inf
    for (a, n), d in ent.items():
        half = n // 2 - 1
        dS = np.abs(d["S"][:, :, half] - d["S_ref"][:, :, half])
        f = np.minimum(d["fidelity_gs"], 1.0 - CFG.ent.fidelity_resolution)
        worst = max(worst, (dS - audenaert_bound(f, 2 ** (n // 2))).max())
    check("|S_VQE - S_ref| <= Fannes-Audenaert bound (all VQE states)", worst <= tol,
          f"largest excess {worst:.1e} (negative = inside the bound); "
          f"1 - F floored at {CFG.ent.fidelity_resolution:g}")

    # 5 rank fidelity bound
    worst = max((d["fidelity_gs"] - d["rank_fidelity_bound"]).max() for d in ent.values())
    check("F_GS <= Schmidt-rank fidelity bound (all VQE states)", worst <= tol,
          f"largest excess {worst:.1e}")

    # 6 exact states reproduce ED entanglement
    for a in ANSATZE:
        if (a, 4) in ent:
            d = ent[(a, 4)]
            dS = np.abs(d["S"][-1] - d["S_ref"][-1]).max()
            check(f"{LABEL[a]} N=4, p=4 reproduces ED entanglement", dS < 1e-6,
                  f"max |S - S_ref| over all cuts and fields = {dS:.1e}")
    if pbc:
        dS = max(np.abs(d["S"][-1] - d["S_ed"]).max() for d in pbc.values())
        dc = max(abs(cs[-1] - ce) for n, (cs, ce) in c_pbc.items() if n >= 6 and np.isfinite(ce))
        check("periodic HVA at p = N/2 reproduces ED entanglement and c", dS < 1e-6 and dc < 1e-6,
              f"max |S - S_ED| = {dS:.1e}, max |c - c_ED| = {dc:.1e}")
        n_big = max(c_pbc)
        ce = c_pbc[n_big][1]
        check("periodic ED central charge close to 1/2", abs(ce - 0.5) < 0.02,
              f"N = {n_big}: c = {ce:.4f}")

    # 7 maximal entanglement
    worst, lowest = -np.inf, np.inf
    for (a, n), d in maxent.items():
        b = np.minimum(d["depths"], n // 2) * LN2
        worst = max(worst, (d["S2_max"] - b).max())
        lowest = min(lowest, (d["S2_max"] / b).min())
    check("maximal S_2 within the rank bound", worst <= tol, f"largest excess {worst:.1e}")
    check("both ansaetze saturate the rank bound at every depth", lowest > 0.999,
          f"smallest S2_max / bound = {lowest:.4f}")

    # 8 symmetry-broken reference
    n = max(ed)
    e = ed[n]
    half = n // 2 - 1
    check("ED limits deep in the ordered phase (h = 0.1)",
          abs(e["S_even"][0, half] - LN2) < 1e-3 and e["S_broken"][0, half] < 1e-2,
          f"N = {n}: even S = {e['S_even'][0, half]:.5f} (ln 2 = {LN2:.5f}), "
          f"broken S = {e['S_broken'][0, half]:.2e}")

    # observations
    for a in ANSATZE:
        asym = [np.abs(d["S"][-1] - d["S"][-1][:, ::-1]).max() for (aa, n), d in ent.items() if aa == a]
        if asym:
            observe(f"{LABEL[a]} mirror symmetry of S(ell)",
                    f"max |S(ell) - S(N - ell)| at p = N: {max(asym):.1e}"
                    + (" (the CNOT staircase runs one way; fits use ell <= N/2 as in Phase 1)"
                       if a == "hea" else ""))
    for (a, n), d in sorted(ent.items()):
        if n < 8:
            continue
        i = hc_index(d["h"])
        slack = d["rank_fidelity_bound"][:, i] - d["fidelity_gs"][:, i]
        k = int(np.argmax(slack))
        observe(f"{LABEL[a]} N={n} rank bound vs achieved fidelity at h_c",
                f"largest gap at p = {d['depths'][k]}: bound F <= {d['rank_fidelity_bound'][k, i]:.4f}, "
                f"achieved F_GS = {d['fidelity_gs'][k, i]:.4f}")
    note()


def ANSATZ_ORDER(ent, n):
    return [a for a in ANSATZE if (a, n) in ent]


def main():
    set_thesis_style()
    note("PHASE 3 SUMMARY: entanglement of VQE states, open TFIM")
    note("=" * 60)
    sizes = list(CFG.system.sizes_noiseless)
    ed = {n: d for n in sizes if (d := load(f"ent_ed_N{n}")) is not None}
    ent = {(a, n): d for a in ANSATZE for n in sizes if (d := load(f"ent_{a}_N{n}")) is not None}
    maxent = {(a, n): d for a in ANSATZE for n in sizes
              if (d := load(f"maxent_{a}_N{n}")) is not None}
    pbc = {n: d for n in CFG.ent.pbc_sizes if (d := load(f"pbc_hva_N{n}")) is not None}
    note(f"Found: ED {sorted(ed)}, states {sorted(ent)}, maxent {sorted(maxent)}, periodic {sorted(pbc)}")
    note()
    if not ed or not ent:
        sys.exit("run experiments/phase3_entanglement.py first")

    c_vqe, c_ed = central_charges_open(ed, ent)
    c_pbc = central_charges_pbc(pbc)

    fig_ent_vs_field(ed, ent)
    fig_ent_error_vs_field(ent)
    fig_cc_profiles(ed, ent)
    fig_central_charge(c_vqe, c_ed, c_pbc)
    if maxent:
        fig_max_entanglement(maxent, ent, ed)
    fig_entanglement_vs_fidelity(ent)

    summarize(ed, ent, maxent, pbc, c_vqe, c_ed, c_pbc)
    n_fail = sum(s == "FAIL" for s, _ in CHECKS)
    n_pass = sum(s == "PASS" for s, _ in CHECKS)
    note(f"Checks: {n_pass} passed, {n_fail} failed, {sum(s == 'NOTE' for s, _ in CHECKS)} notes")
    out = RESULTS_DIR / SUBDIR / "phase3_summary.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(SUMMARY) + "\n", encoding="utf-8")
    print(f"[summary] written to {out}")


if __name__ == "__main__":
    main()
