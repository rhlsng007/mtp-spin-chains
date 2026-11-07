"""
Thesis Figure 7: central charge extracted from VQE states at h = J against the
circuit depth (Phase 1 and 3 data).

    python experiments\\thesis\\fig07_central_charge_vs_depth.py

Panels (N in CFG.thesis.ent_cc_sizes)
  (a) HVA, open chains: Calabrese-Cardy c(p); dashed lines are the ED values
  (b) HEA, open chains: the same
  (c) HVA, periodic chains (p <= N/2): c(p), with the ED values and c = 1/2

Fits use the window CFG.ent.cc_ell_min <= l <= N/2 and the forms of Figure 2.

Validation
  * The open-chain fits reproduce the Phase 3 report (c_open_vs_depth, both
    ansaetze, every N) to its printed digits.
  * The Phase 3 ED even state at h = J has the Phase 1 entropy profile; the
    periodic reference profile of Phase 3 equals the Phase 1 periodic profile.
  * Periodic HVA at p = N/2 is exact (Ho and Hsieh), so its whole entropy
    profile equals the ED profile and c(N/2) equals the ED value.
  * HVA at p <= 2: c = 0 exactly; S(l) does not depend on l inside the fit
    window (area law of a shallow circuit).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from config import CFG
from experiments.thesis.figlib import FigureSpec, cc_fit, field_index, run_figure
from utils.plotting import (ANSATZ_MARKERS, COLORS, panel_labels, reference_line, size_color,
                            thesis_figure)

SPEC = FigureSpec(
    number=7,
    slug="central_charge_vs_depth",
    chapter="Entanglement of VQE states",
    phases=(1, 3),
    title="Central charge of VQE states at the critical point against circuit depth",
    caption=("Central charge from Calabrese-Cardy fits of the entropy profiles of optimized VQE "
             "states at h = J, against depth p, for N = 8, 10, 12. (a) Hamiltonian variational "
             "ansatz and (b) hardware-efficient ansatz on open chains; dashed lines are the exact "
             "values at the same N (finite-size biased above 1/2). (c) Periodic HVA, which becomes "
             "exact at p = N/2; the black dashed line marks c = 1/2. Shallow circuits obey an area "
             "law (c = 0); points below the axis (HEA, p = 1) are omitted."),
)

ANSATZE = (("hva", "HVA"), ("hea", "HEA"))
YLIM = (-0.15, 1.05)


def build(ctx):
    th, l0 = CFG.thesis, CFG.ent.cc_ell_min
    sizes, cc_sizes = list(th.vqe_sizes), list(th.ent_cc_sizes)
    ent = {a: {N: ctx.load(f"ent_{a}_N{N}", "phase3", require=("S", "depths", "h")) for N in sizes}
           for a, _ in ANSATZE}
    ed = {N: ctx.load(f"ent_ed_N{N}", "phase3", require=("S_even", "h")) for N in sizes}
    cc = {N: ctx.load(f"ed_cc_profiles_N{N}", "phase1", require=("tfim_open", "tfim_periodic"))
          for N in sizes}
    pbc = {N: ctx.load(f"pbc_hva_N{N}", "phase3", require=("S", "S_ed", "depths")) for N in sizes}
    report = ctx.load_json("phase3_report_data.json", "phase3")
    h = np.asarray(ed[sizes[0]]["h"])
    ih = field_index(h, CFG.tfim.h_critical)
    ctx.param(sizes=cc_sizes, field=CFG.tfim.h_critical, ell_min=l0, check_sizes=sizes)

    c_open = {}
    for a, _ in ANSATZE:
        for N in sizes:
            S = np.asarray(ent[a][N]["S"])[:, ih, :]
            c_open[a, N] = [cc_fit(row, N, "open", l0)[0] for row in S]
    c_pbc = {N: [cc_fit(row, N, "periodic", l0)[0] for row in np.asarray(pbc[N]["S"])] for N in sizes}
    c_ed_open = {N: cc_fit(cc[N]["tfim_open"], N, "open", l0)[0] for N in sizes}
    c_ed_pbc = {N: cc_fit(cc[N]["tfim_periodic"], N, "periodic", l0)[0] for N in sizes}

    # -- validation ------------------------------------------------------------
    mine, theirs = [], []
    for a, _ in ANSATZE:
        for N in sizes:
            rep = report.get("vqe", {}).get(f"{a}_N{N}", {}).get("c_open_vs_depth")
            if rep:
                mine += c_open[a, N]
                theirs += rep
    ctx.check_rounded("open-chain fits reproduce the Phase 3 report (c_open_vs_depth)", mine, theirs)
    d1 = [np.max(np.abs(np.asarray(ed[N]["S_even"])[ih] - cc[N]["tfim_open"])) for N in sizes]
    ctx.check_max("Phase 3 ED profile at h = J equals the Phase 1 profile (open)", d1, th.identity_tol)
    d2 = [np.max(np.abs(np.asarray(pbc[N]["S_ed"]) - cc[N]["tfim_periodic"])) for N in sizes]
    ctx.check_max("Phase 3 periodic reference equals the Phase 1 periodic profile", d2, th.identity_tol)
    dp, dc = [], []
    for N in sizes:
        p = list(np.asarray(pbc[N]["depths"], dtype=int))
        k = p.index(N // 2)
        dp.append(np.max(np.abs(np.asarray(pbc[N]["S"])[k] - pbc[N]["S_ed"])))
        if N >= 2 * l0 + 2:
            dc.append(abs(c_pbc[N][k] - c_ed_pbc[N]))
    ctx.check_max("periodic HVA at p = N/2 reproduces the ED profile (exact)", dp, th.pbc_entropy_tol)
    ctx.check_max("periodic HVA at p = N/2 gives the ED central charge", dc, th.pbc_entropy_tol)
    shallow = [abs(c) for N in cc_sizes for p, c in zip(np.asarray(ent["hva"][N]["depths"], dtype=int),
                                                         c_open["hva", N]) if p <= 2]
    ctx.check_max("HVA p <= 2: c = 0 (S(l) flat inside the fit window)", shallow, 1e-10)

    for a, name in ANSATZE:
        for N in cc_sizes:
            ctx.note(f"{name} open N={N}: c(p) = " + ", ".join(f"{c:.3f}" for c in c_open[a, N])
                     + f"; ED {c_ed_open[N]:.4f}")
    for N in cc_sizes:
        ctx.note(f"HVA periodic N={N}: c(p) = " + ", ".join(f"{c:.3f}" for c in c_pbc[N])
                 + f"; ED {c_ed_pbc[N]:.4f}")

    # -- figure --------------------------------------------------------------------
    fig, axes = thesis_figure(ncols=3, aspect=0.36)
    for ax, (a, name) in zip(axes[:2], ANSATZE):
        for N in cc_sizes:
            p = np.asarray(ent[a][N]["depths"], dtype=int)
            c = np.array(c_open[a, N])
            vis = c >= YLIM[0]
            ax.plot(p[vis], c[vis], ANSATZ_MARKERS[name] + "-", color=size_color(N), ms=3,
                    label=f"$N={N}$")
            reference_line(ax, c_ed_open[N], color=size_color(N), lw=0.7)
        ax.set_title(f"{name}, open", loc="right")
    ax = axes[2]
    for N in cc_sizes:
        p = np.asarray(pbc[N]["depths"], dtype=int)
        ax.plot(p, c_pbc[N], "o-", color=size_color(N), ms=3, label=f"$N={N}$")
        reference_line(ax, c_ed_pbc[N], color=size_color(N), lw=0.7)
    reference_line(ax, 0.5, lw=0.8)
    ax.set_title("HVA, periodic", loc="right")
    for ax in axes:
        ax.plot([], [], "--", color=COLORS["grey"], lw=0.7, label="ED, same $N$")
        ax.set_ylim(*YLIM)
        ax.set_xlabel("depth $p$")
        ax.set_ylabel(r"fitted $c$ at $h=J$")
        ax.legend(loc="lower right")
    panel_labels(axes)
    return fig


if __name__ == "__main__":
    raise SystemExit(run_figure(SPEC, build))
