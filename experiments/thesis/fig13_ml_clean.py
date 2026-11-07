"""
Thesis Figure 13: machine-learning phase classification on noiseless data
(Phases 6 and 7).

    python experiments\\thesis\\fig13_ml_clean.py

Panels
  (a) output curves P(h) of logistic regression trained on exact ED correlator
      features, N = 4 to 14 (thick) and exact free-fermion features up to
      N = 64 (thin); training fields shaded
  (b) h_c^ML against 1/N: ED with exact and shadow features, free fermions up to
      N = 64, and the PCA midpoint; h_c^ML converges from below
  (c) an ED-trained classifier applied to noiseless VQE states (shadows):
      HEA transfers, the HVA at p = N/2 does not (its ordered states are too weak)

h_c^ML is the field where the isotonic fit of the field-averaged output crosses
1/2; errors are the Phase 7 total uncertainty (seed spread and shot error in
quadrature).

Validation
  * Free fermions: the ED classifier outputs equal those on exact free-fermion
    correlators at every N <= 14 (both classifiers, feature sets, seeds), and
    the ED <M^2> equals the free-fermion value.
  * Exact-feature logistic regression is seed independent.
  * Deep-field accuracy is 100 % for every ED classifier.
  * PC1 correlates with <M^2> above CFG.ml.pca_corr_min (ED, free fermions).
  * Shadow and exact h_c^ML agree within hc_sigma * sigma + hc_floor (every N,
    both classifiers).
  * Means and total uncertainties equal clean_accuracy.csv (4 decimals).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from config import CFG
from experiments.thesis.figlib import FigureSpec, run_figure, summarize_hc, table_value
from utils.plotting import (ANSATZ_MARKERS, COLORS, SERIES_COLORS, mark_critical, panel_labels,
                            reference_line, size_color, thesis_figure)

SPEC = FigureSpec(
    number=13,
    slug="ml_clean",
    chapter="Machine-learning phase classification",
    phases=(1, 6, 7),
    title="Classifying the phases of noiseless states: finite-size drift and transfer to VQE",
    caption=("Logistic-regression phase classification trained deep in both phases (h < 0.6 and "
             "h > 1.4, shaded). (a) Field-averaged output for exact ED correlator features, "
             "N = 4 to 14 (thick), and exact free-fermion features up to N = 64 (thin). (b) Learned "
             "critical field h_c^ML against 1/N: exact and classical-shadow ED features, free "
             "fermions, and the midpoint of the first principal component; it approaches h_c from "
             "below. (c) The ED-trained classifier applied to shadows of noiseless VQE states, "
             "against the ED value at the same N."),
)


def build(ctx):
    th, ml = CFG.thesis, CFG.ml
    sizes, vsizes, ffs = list(ml.ed_sizes), list(ml.vqe_sizes), list(ml.ff_sizes)
    keys = ("h", "P", "hc", "hc_rep", "acc_deep", "pca_r_m2", "pca_h_mid", "m2_exact", "train_field",
            "kinds", "feature_sets", "classifiers")
    ed = {N: ctx.load(f"clean_ed_N{N}", "phase7/clean", require=keys) for N in sizes}
    ff = ctx.load("ff_exact", "phase7/ff", require=("sizes", "P", "hc", "m2", "pca_r_m2", "pca_h_mid"))
    vqe = {(a, N, p): ctx.load(f"clean_vqe_{a}_N{N}_p{p}", "phase7/clean", require=("ed_hc", "ed_hc_rep"))
           for a in CFG.vqe.ansatze for N in vsizes for p in (N // 2, N)}
    table = ctx.load_table("clean_accuracy.csv", "phase7/tables")
    first = ed[sizes[0]]
    h = np.asarray(first["h"])
    K = {k: i for i, k in enumerate(first["kinds"])}
    FS = {f: i for i, f in enumerate(first["feature_sets"])}
    CL = {c: i for i, c in enumerate(first["classifiers"])}
    ex, sh, fc, lr = K["exact"], K["shadow"], FS[th.ml_main_features], CL[th.ml_main_classifier]
    ffN = [int(n) for n in ff["sizes"]]
    ctx.param(ed_sizes=sizes, ff_sizes=ffN, vqe_sizes=vsizes, features=th.ml_main_features,
              classifier=th.ml_main_classifier, training_fields=f"h < {ml.train_h_ordered_max} or "
                                                                f"h > {ml.train_h_disordered_min}")

    # -- validation ------------------------------------------------------------
    dP, dm2 = [], []
    for N in sizes:
        if N in ffN:
            k = ffN.index(N)
            dP.append(np.nanmax(np.abs(np.asarray(ed[N]["P"])[ex] - np.asarray(ff["P"])[k])))
            dm2.append(np.max(np.abs(np.asarray(ed[N]["m2_exact"]) - np.asarray(ff["m2"])[k])))
    ctx.check_max("ED classifier outputs equal those on exact free-fermion correlators (N <= 14)", dP,
                  ml.identity_tol)
    ctx.check_max("ED <M^2> equals the free-fermion value", dm2, ml.identity_tol)
    seed = [np.nanmax(np.nanstd(np.asarray(ed[N]["hc"])[ex, :, lr], axis=-1)) for N in sizes]
    ctx.check_max("exact-feature logistic regression is seed independent (std of h_c^ML over seeds)",
                  seed, 1e-12)
    ctx.check_min("deep-field accuracy 100 % (every ED classifier, kind, feature set, seed)",
                  [np.nanmin(ed[N]["acc_deep"]) for N in sizes], 1.0)
    r = [float(np.min(ed[N]["pca_r_m2"])) for N in sizes] + [float(np.min(ff["pca_r_m2"]))]
    ctx.check_min(f"PC1 correlates with <M^2> (ED and free fermions)", r, ml.pca_corr_min)

    stats = {N: summarize_hc(ed[N]["hc"], ed[N]["hc_rep"]) for N in sizes}
    agree = []
    for N in sizes:
        for c in CL.values():
            m_ex, m_sh = stats[N]["mean"][ex, fc, c], stats[N]["mean"][sh, fc, c]
            sig = stats[N]["total"][sh, fc, c]
            agree.append(abs(m_sh - m_ex) <= ml.hc_sigma * sig + ml.hc_floor)
    ctx.check_true(f"shadow and exact h_c^ML agree within {ml.hc_sigma:g} sigma + {ml.hc_floor:g}",
                   all(agree), f"{sum(agree)} of {len(agree)}")
    mine, theirs = [], []
    for N in sizes:
        for kind, ik in K.items():
            for c, ic in CL.items():
                for col, key in (("hc_mean", "mean"), ("hc_total_sigma", "total")):
                    mine.append(stats[N][key][ik, fc, ic])
                    theirs.append(table_value(table, col, source="ed", N=N, kind=kind,
                                              features=th.ml_main_features, classifier=c))
    for (a, N, p), d in vqe.items():
        s = summarize_hc(d["ed_hc"], d["ed_hc_rep"])
        mine += [s["mean"][sh, fc, lr], s["total"][sh, fc, lr]]
        theirs += [table_value(table, col, source="vqe", ansatz=a, N=N, depth=p, trained_on="ed",
                               kind="shadow", features=th.ml_main_features, classifier=th.ml_main_classifier)
                   for col in ("hc_mean", "hc_total_sigma")]
    mine, theirs = np.array(mine, dtype=float), np.array(theirs, dtype=float)
    ok = np.isfinite(theirs)
    ctx.check_max("h_c^ML means and total sigmas equal clean_accuracy.csv", np.abs(mine[ok] - theirs[ok]),
                  th.table_tol, detail=f"{ok.sum()} table cells")

    hff = summarize_hc(ff["hc"])["mean"][:, fc, lr]
    ctx.note("h_c^ML (LR, exact): ED " + ", ".join(f"N={N}: {stats[N]['mean'][ex, fc, lr]:.3f}" for N in sizes)
             + "; free fermions " + ", ".join(f"N={n}: {v:.3f}" for n, v in zip(ffN, hff) if n > max(sizes)))
    big = np.array([n >= 16 for n in ffN])
    if big.sum() >= 2:
        fit = np.polyfit(1.0 / np.array(ffN, dtype=float)[big], hff[big], 1)
        ctx.note(f"linear fit h_c^ML = a + b/N over free-fermion N >= 16: a = {fit[1]:.3f} "
                 f"(fixed-threshold hypothesis; below h_c = 1)")
    for a in CFG.vqe.ansatze:
        for p_rule in ("N/2", "N"):
            vals = [summarize_hc(vqe[a, N, N // 2 if p_rule == "N/2" else N]["ed_hc"])["mean"][sh, fc, lr]
                    for N in vsizes]
            ctx.note(f"ED-trained LR on {a.upper()} shadows, p = {p_rule}: h_c^ML " +
                     ", ".join(f"{v:.3f}" for v in vals))

    # -- figure --------------------------------------------------------------------
    fig, axes = thesis_figure(ncols=3, aspect=0.36)
    ax = axes[0]
    tf = np.asarray(first["train_field"], dtype=bool)
    for lo, hi in ((h.min(), ml.train_h_ordered_max), (ml.train_h_disordered_min, h.max())):
        ax.axvspan(lo, hi, color=COLORS["grey"], alpha=0.12, lw=0)
    for n_ff, k in zip(ffN, range(len(ffN))):
        if n_ff > max(sizes):
            ax.plot(h, np.nanmean(np.asarray(ff["P"])[k, fc, lr], axis=-2), color=COLORS["grey"], lw=0.5)
    for N in sizes:
        ax.plot(h, np.nanmean(np.asarray(ed[N]["P"])[ex, fc, lr], axis=-2), color=size_color(N), lw=1.1,
                label=f"$N={N}$")
    reference_line(ax, 0.5, color=COLORS["grey"], lw=0.5, ls=":")
    mark_critical(ax)
    ax.set_xlim(h.min(), h.max())
    ax.set_xlabel(r"$h/J$")
    ax.set_ylabel(r"$P(h)$, logistic regression")
    ax.legend(loc="lower right", ncol=2, columnspacing=0.8)

    ax = axes[1]
    inv = lambda n: 1.0 / np.asarray(n, dtype=float)  # noqa: E731
    ax.plot(inv(ffN), hff, "-", color=COLORS["grey"], lw=0.8, label="free fermions, exact")
    ax.plot(inv(sizes), [stats[N]["mean"][ex, fc, lr] for N in sizes], "o", color=SERIES_COLORS["ED"],
            ms=3.5, label="ED, exact features")
    ax.errorbar(inv(sizes) * 1.02, [stats[N]["mean"][sh, fc, lr] for N in sizes],
                yerr=[stats[N]["total"][sh, fc, lr] for N in sizes], fmt="s", color=COLORS["sky"], ms=3,
                mfc="none", label="ED, shadows")
    ax.plot(inv(sizes), [float(ed[N]["pca_h_mid"][ex]) for N in sizes], "^", color=COLORS["purple"], ms=3,
            label="PCA midpoint")
    reference_line(ax, CFG.tfim.h_critical, lw=0.7)
    ax.set_xlim(0, 0.27)
    lo = min(np.nanmin(hff), min(float(ed[N]["pca_h_mid"][ex]) for N in sizes))
    ax.set_ylim(lo - 0.02, 1.06)
    ax.set_xlabel(r"$1/N$")
    ax.set_ylabel(r"$h_c^{\mathrm{ML}}/J$")
    ax.legend(loc="upper right")

    ax = axes[2]
    ax.plot(vsizes, [stats[N]["mean"][sh, fc, lr] for N in vsizes], "-", color=SERIES_COLORS["ED"], lw=0.9,
            label="ED, same $N$")
    for a, name in (("hva", "HVA"), ("hea", "HEA")):
        for p_rule, mfc in (("N/2", "none"), ("N", None)):
            st = [summarize_hc(vqe[a, N, N // 2 if p_rule == "N/2" else N]["ed_hc"],
                               vqe[a, N, N // 2 if p_rule == "N/2" else N]["ed_hc_rep"]) for N in vsizes]
            ax.errorbar(vsizes, [s["mean"][sh, fc, lr] for s in st], yerr=[s["total"][sh, fc, lr] for s in st],
                        fmt=ANSATZ_MARKERS[name] + ("--" if p_rule == "N/2" else "-"), color=SERIES_COLORS[name],
                        ms=3.5, mfc=mfc, lw=0.9, label=f"{name}, $p={p_rule}$")
    ax.set_xticks(vsizes)
    ax.set_xlabel("$N$")
    ax.set_ylabel(r"$h_c^{\mathrm{ML}}/J$, trained on ED")
    ax.legend(loc="lower right", ncol=1)
    panel_labels(axes)
    return fig


if __name__ == "__main__":
    raise SystemExit(run_figure(SPEC, build))
