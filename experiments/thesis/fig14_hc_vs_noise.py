"""
Thesis Figure 14 (central result): the learned critical field under hardware-
like noise, for five ways of training the classifier (Phases 6 and 7).

    python experiments\\thesis\\fig14_hc_vs_noise.py

Panels (N = CFG.ml.study_size, p = N/2, readout-corrected shadows,
logistic regression on correlator features)
  (a) HVA and (b) HEA: h_c^ML against the error budget lambda = p (n_1q + 10 n_2q)
      for the training modes: ED, noiseless circuit, train = test noise, mixed
      strengths, leave one strength out; the grey band is the ED value at the
      same N. Crosses at the top: no crossing (every field called disordered).
  (c) drift of the clean-trained classifiers in units of the combined
      uncertainty, against lambda, for N = 4, 6, 8: the sizes collapse onto one
      lambda scale, with the 2-sigma drift onset near lambda = 0.15.

Drift onset (Phase 7): the first strength with

    |h_c(p) - h_c(0)| > 2 sqrt(sigma_tot(p)^2 + sigma_tot(0)^2).

Validation
  * Error budget: the stored lambda equals p (n_1q + 10 n_2q) of the Phase 6
    datasets.
  * At zero noise (variant "gate") the noisy data reproduce the noiseless VQE
    classifiers: train = test noise equals the noiseless-circuit mode, and the
    ED-trained mode equals the ED-trained classifier on the noiseless circuit.
  * Readout correction is exact at zero noise for exact features ("ro" = "gate").
  * No crossing means every field is assigned to one phase: accuracy 20/38
    (all disordered) or 18/38 (all ordered).
  * Means and total uncertainties equal noise_accuracy.csv (4 decimals).
  * Drift onsets reproduce the Phase 7 results (N = 8).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from config import CFG
from experiments.thesis.figlib import (FigureSpec, drift_onset, run_figure, summarize_hc, table_value)
from utils.plotting import (MODE_COLORS, MODE_LABELS, COLORS, label_zero_strength, panel_labels,
                            reference_line, size_color, thesis_figure, zero_strength_positions)

SPEC = FigureSpec(
    number=14,
    slug="hc_vs_noise",
    chapter="Machine-learning phase classification",
    phases=(6, 7),
    title="The learned critical field under noise depends on how the classifier was trained",
    caption=("Learned critical field of logistic regression on readout-corrected classical shadows "
             "of noisy N = 8, p = 4 circuits under the hardware-like noise model, against the error "
             "budget lambda (the expected number of gate errors). (a) HVA, (b) HEA. Colours: "
             "training data. Classifiers trained on clean data (ED, noiseless circuit) drift once "
             "lambda exceeds about 0.15 to 0.3 and find no transition at lambda = 3.2 (crosses: "
             "every field called disordered); trained at the test noise level, or on mixed noise "
             "levels, they keep h_c^ML within 0.07 of the clean value. Grey band: ED at the same N. "
             "(c) Drift of the clean-trained classifiers in units of the combined uncertainty for "
             "N = 4, 6, 8; the dotted line marks the 2-sigma onset."),
)

ANSATZE = (("hva", "HVA"), ("hea", "HEA"))
# Phase 7 drift onsets at N = 8 (lambda; None = no drift), THESIS_CONTEXT.md Section 14
PHASE7_ONSETS = {"hva": {"clean_ed": 0.15, "clean_vqe": 0.32, "noisy": 3.2, "mixed": 0.69, "loo": 0.69},
                 "hea": {"clean_ed": 0.15, "clean_vqe": 0.32, "noisy": None, "mixed": 1.49, "loo": 1.49}}
ALL_DISORDERED, ALL_ORDERED = 20 / 38, 18 / 38


def build(ctx):
    th, ml = CFG.thesis, CFG.ml
    Np, var = ml.study_size, ml.default_variant
    sizes = list(ml.noisy_sizes)
    ch = ml.noisy_channels[0]
    keys = ("hc", "hc_rep", "acc", "ncross", "P", "strengths", "lam", "modes", "kinds", "feature_sets",
            "classifiers")
    noise = {(a, N, v): ctx.load(f"noise_{a}_N{N}_p{N // 2}_{ch}_{v}", "phase7/noise", require=keys)
             for a, _ in ANSATZE for N in sizes for v in ("gate", "ro")}
    clean = {(a, N): ctx.load(f"clean_vqe_{a}_N{N}_p{N // 2}", "phase7/clean", require=("P", "ed_P"))
             for a, _ in ANSATZE for N in sizes}
    ds = {(a, N): ctx.load(f"ds_noisy_{a}_N{N}_p{N // 2}_{ch}", "phase6/noisy",
                           require=("strengths", "lam", "n_1q", "n_2q")) for a, _ in ANSATZE for N in sizes}
    edf = ctx.load(f"clean_ed_N{Np}", "phase7/clean", require=("hc", "hc_rep"))
    table = ctx.load_table("noise_accuracy.csv", "phase7/tables")
    f0 = noise["hva", Np, var]
    M = {m: i for i, m in enumerate(f0["modes"])}
    K = {k: i for i, k in enumerate(f0["kinds"])}
    FS = {f: i for i, f in enumerate(f0["feature_sets"])}
    CL = {c: i for i, c in enumerate(f0["classifiers"])}
    kk, fc, lr = K[th.ml_main_kind], FS[th.ml_main_features], CL[th.ml_main_classifier]
    ctx.param(size=Np, depth=Np // 2, channel=ch, variant=var, kind=th.ml_main_kind,
              features=th.ml_main_features, classifier=th.ml_main_classifier, modes=list(M),
              drift_sigmas=th.drift_sigmas)

    # -- validation ------------------------------------------------------------
    lam_err = [np.max(np.abs(np.asarray(d["lam"]) - np.asarray(d["strengths"]) *
                             (int(d["n_1q"]) + 10 * int(d["n_2q"])))) for d in ds.values()]
    lam_err += [np.max(np.abs(np.asarray(noise[a, N, v]["lam"]) - np.asarray(ds[a, N]["lam"])))
                for a, _ in ANSATZE for N in sizes for v in ("gate", "ro")]
    ctx.check_max("lambda = p (n_1q + 10 n_2q), stored identically in Phases 6 and 7", lam_err, 1e-12)
    z1, z2, z3 = [], [], []
    for a, _ in ANSATZE:
        for N in sizes:
            g, r = noise[a, N, "gate"], noise[a, N, "ro"]
            Pg = np.asarray(g["P"])
            z1.append(np.nanmax(np.abs(Pg[M["noisy"], 0, K["exact"]] - np.asarray(clean[a, N]["P"])[K["exact"]])))
            z2.append(np.nanmax(np.abs(Pg[M["clean_ed"], 0, K["exact"]] - np.asarray(clean[a, N]["ed_P"])[K["exact"]])))
            z3.append(np.nanmax(np.abs(np.asarray(r["P"])[M["noisy"], 0, K["exact"]] - Pg[M["noisy"], 0, K["exact"]])))
    ctx.check_max("zero noise: train = test noise reproduces the noiseless-circuit classifier (exact)", z1,
                  ml.identity_tol)
    ctx.check_max("zero noise: ED-trained classifier on noisy data equals it on the noiseless circuit", z2,
                  ml.identity_tol)
    ctx.check_max("zero noise: readout correction exact for exact features (ro = gate)", z3, ml.identity_tol)
    bad = 0
    for d in noise.values():
        hc, acc = np.asarray(d["hc"]), np.asarray(d["acc"])
        sel = ~np.isfinite(hc)
        a = acc[sel]
        bad += int(np.sum(~(np.isclose(a, ALL_DISORDERED) | np.isclose(a, ALL_ORDERED))))
    ctx.check_true("no crossing <=> one phase everywhere: accuracy 20/38 or 18/38", bad == 0,
                   f"{bad} exceptions")

    stats = {key: summarize_hc(d["hc"], d["hc_rep"]) for key, d in noise.items()}
    mine, theirs = [], []
    for a, _ in ANSATZE:
        d, s = noise[a, Np, var], stats[a, Np, var]
        for m, im in M.items():
            for j, sj in enumerate(np.asarray(d["strengths"])):
                for col, key in (("hc_mean", "mean"), ("hc_total_sigma", "total")):
                    mine.append(s[key][im, j, kk, fc, lr])
                    theirs.append(table_value(table, col, ansatz=a, N=Np, variant=var, mode=m, strength=float(sj),
                                              kind=th.ml_main_kind, features=th.ml_main_features,
                                              classifier=th.ml_main_classifier))
    mine, theirs = np.array(mine), np.array(theirs)
    both = np.isfinite(mine) & np.isfinite(theirs)
    same_nan = bool(np.all(np.isfinite(mine) == np.isfinite(theirs)))
    ctx.check_true("h_c^ML means and total sigmas equal noise_accuracy.csv (4 decimals, same missing crossings)",
                   same_nan and bool(np.all(np.abs(mine[both] - theirs[both]) <= th.table_tol)),
                   f"{both.sum()} cells, max difference {np.max(np.abs(mine[both] - theirs[both])):.1e}")

    onsets, ok = {}, True
    for a, name in ANSATZE:
        d, s = noise[a, Np, var], stats[a, Np, var]
        lam = np.asarray(d["lam"])
        for m, im in M.items():
            k = drift_onset(s["mean"][im, :, kk, fc, lr], s["total"][im, :, kk, fc, lr], d["strengths"],
                            th.drift_sigmas)
            onsets[a, m] = float(lam[k]) if k >= 0 else None
            ref = PHASE7_ONSETS[a].get(m)
            ok &= (ref is None and k < 0) or (ref is not None and k >= 0 and abs(lam[k] - ref) <= 0.006 * ref + 0.005)
    ctx.check_true(f"drift onsets (N = {Np}) reproduce the Phase 7 results", ok,
                   "; ".join(f"{a.upper()} {m}: {'none' if v is None else f'{v:.2f}'}" for (a, m), v in onsets.items()))
    for a, name in ANSATZE:
        d, s = noise[a, Np, var], stats[a, Np, var]
        acc = np.nanmean(np.asarray(d["acc"])[:, :, kk, fc, lr], axis=-1)
        for m, im in M.items():
            mm = s["mean"][im, :, kk, fc, lr]
            ctx.note(f"{name} {m}: h_c^ML {mm[0]:.3f} (p = 0) -> "
                     f"{'none' if not np.isfinite(mm[-1]) else f'{mm[-1]:.3f}'} (lambda = {d['lam'][-1]:.2f}); "
                     f"accuracy {acc[im, 0]:.3f} -> {acc[im, -1]:.3f}; drift onset lambda = "
                     f"{'none' if onsets[a, m] is None else f'{onsets[a, m]:.2f}'}")
    for N in sizes:
        for a, name in ANSATZE:
            d, s = noise[a, N, var], stats[a, N, var]
            ons = [drift_onset(s["mean"][M[m], :, kk, fc, lr], s["total"][M[m], :, kk, fc, lr], d["strengths"],
                               th.drift_sigmas) for m in ("clean_ed", "clean_vqe")]
            ctx.note(f"N={N} {name}: clean-trained drift onsets lambda = " + ", ".join(
                "none" if k < 0 else f"{d['lam'][k]:.2f}" for k in ons) + f" (largest lambda {d['lam'][-1]:.2f})")

    # -- figure --------------------------------------------------------------------
    es = summarize_hc(edf["hc"], edf["hc_rep"])
    e_m, e_s = es["mean"][kk, fc, lr], es["total"][kk, fc, lr]
    fig, axes = thesis_figure(ncols=3, aspect=0.38)
    for ax, (a, name) in zip(axes[:2], ANSATZE):
        d, s = noise[a, Np, var], stats[a, Np, var]
        lam = np.asarray(d["lam"])
        x, x0 = zero_strength_positions(lam)
        ax.axhspan(e_m - e_s, e_m + e_s, color=COLORS["grey"], alpha=0.3, lw=0)
        for m, im in M.items():
            y, e = s["mean"][im, :, kk, fc, lr], s["total"][im, :, kk, fc, lr]
            off = 1.0 + 0.04 * (im - 2)                  # small horizontal offsets
            ax.errorbar(x * off, y, yerr=e, fmt="o-", color=MODE_COLORS[m], ms=2.5, lw=0.9, label=MODE_LABELS[m])
            miss = ~np.isfinite(y)
            if miss.any():
                ax.plot(x[miss] * off, np.full(miss.sum(), 1.04), "x", color=MODE_COLORS[m], ms=3.5, clip_on=False)
        ax.set_xscale("log")
        finite = np.concatenate([s["mean"][im, :, kk, fc, lr] for im in M.values()])
        ax.set_ylim(max(0.3, np.nanmin(finite) - 0.04), 1.06)
        reference_line(ax, CFG.tfim.h_critical, lw=0.6, ls=":")
        ax.set_xlabel(r"error budget $\lambda$")
        ax.set_ylabel(r"$h_c^{\mathrm{ML}}/J$")
        ax.set_title(f"{name}, $N={Np}$, $p={Np // 2}$", loc="right")
        label_zero_strength(ax, x0)
    axes[1].legend(loc="lower left")
    ax = axes[2]
    for N in sizes:
        for a, name in ANSATZE:
            d, s = noise[a, N, var], stats[a, N, var]
            lam = np.asarray(d["lam"])
            im = M["clean_ed"]
            y, e = s["mean"][im, :, kk, fc, lr], s["total"][im, :, kk, fc, lr]
            z = np.abs(y - y[0]) / np.sqrt(e ** 2 + e[0] ** 2)
            pos = lam > 0
            ax.plot(lam[pos], np.where(np.isfinite(z[pos]), z[pos], np.nan), "o-" if a == "hva" else "s--",
                    color=size_color(N), ms=2.5, lw=0.9, mfc="none" if a == "hea" else None,
                    label=f"$N={N}$" if a == "hva" else None)
    ax.axhline(th.drift_sigmas, color=COLORS["black"], lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"error budget $\lambda$")
    ax.set_ylabel(r"$|\Delta h_c^{\mathrm{ML}}|\,/\,\sigma$, trained on ED")
    ax.set_title("solid: HVA, dashed: HEA", loc="right")
    ax.legend(loc="upper left")
    panel_labels(axes)
    return fig


if __name__ == "__main__":
    raise SystemExit(run_figure(SPEC, build))
