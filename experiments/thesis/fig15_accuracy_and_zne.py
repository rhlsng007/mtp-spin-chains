"""
Thesis Figure 15: classification accuracy under noise, and zero-noise
extrapolation of the classifier input (Phases 6 and 7).

    python experiments\\thesis\\fig15_accuracy_and_zne.py

Panels
  (a) HVA and (b) HEA, N = CFG.ml.study_size: accuracy on all labelled test
      fields against lambda for the five training modes (shadows, logistic
      regression); 20/38 is the value of calling every field disordered
  (c) ZNE inside its window (two strengths per size): shift of h_c^ML from the
      noiseless circuit for unmitigated and ZNE-mitigated exact features
      (classifier trained on the noiseless circuit), N = 4, 6, 8, both ansaetze

Validation
  * Accuracy in [0, 1]; h_c^ML is missing exactly when the isotonic fit of the
    field-averaged output never crosses 1/2.
  * ZNE removes at least 1 - CFG.thesis.zne_feature_ratio_max of the exact
    feature error at every window point (feature RMS to the noiseless circuit).
  * Zero-noise extrapolation uses the Phase 5 Richardson weights (15/8, -5/4, 3/8).
  * Means equal zne_accuracy.csv and noise_accuracy.csv (4 decimals).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from config import CFG
from experiments.thesis.figlib import (FigureSpec, crossing_consistent, extrapolation_weights, run_figure,
                                       summarize_hc, table_value)
from utils.plotting import (ANSATZ_MARKERS, COLORS, MITIGATION_COLORS, MODE_COLORS, MODE_LABELS,
                            label_zero_strength, panel_labels, reference_line, size_color, thesis_figure,
                            zero_strength_positions)

SPEC = FigureSpec(
    number=15,
    slug="accuracy_and_zne",
    chapter="Machine-learning phase classification",
    phases=(5, 6, 7),
    title="Classification accuracy under noise, and recovery by zero-noise extrapolation",
    caption=("(a), (b) Accuracy of logistic regression on readout-corrected shadows of noisy N = 8, "
             "p = 4 circuits against the error budget, for the five training modes; the dotted line "
             "(20/38) is the accuracy of calling every field disordered. Training at the test noise "
             "level or on mixed levels keeps the accuracy above 0.9 up to lambda = 3.2. (c) Inside "
             "the ZNE window (lambda = 0.07 to 0.17 at N = 8), the shift of h_c^ML from the "
             "noiseless circuit for unmitigated (open) and ZNE-mitigated (filled) exact features; "
             "the classifier is trained on the noiseless circuit."),
)

ANSATZE = (("hva", "HVA"), ("hea", "HEA"))
ALL_DISORDERED, ALL_ORDERED = 20 / 38, 18 / 38


def build(ctx):
    th, ml = CFG.thesis, CFG.ml
    Np, var = ml.study_size, ml.default_variant
    sizes = list(ml.noisy_sizes)
    ch = ml.noisy_channels[0]
    noise = {a: ctx.load(f"noise_{a}_N{Np}_p{Np // 2}_{ch}_{var}", "phase7/noise",
                         require=("acc", "hc", "P", "lam", "strengths", "modes", "kinds", "feature_sets",
                                  "classifiers"))
             for a, _ in ANSATZE}
    zne = {(a, N): ctx.load(f"zne_{a}_N{N}_p{N // 2}_{ch}", "phase7/zne",
                            require=("hc", "hc_rep", "modes", "methods", "lam", "strengths", "feature_rms_to_clean"))
           for a, _ in ANSATZE for N in sizes}
    clean = {(a, N): ctx.load(f"clean_vqe_{a}_N{N}_p{N // 2}", "phase7/clean", require=("hc", "hc_rep"))
             for a, _ in ANSATZE for N in sizes}
    dsz = {(a, N): ctx.load(f"ds_zne_{a}_N{N}_p{N // 2}_{ch}", "phase6/zne", require=("coefficients", "scales"))
           for a, _ in ANSATZE for N in sizes}
    t_noise = ctx.load_table("noise_accuracy.csv", "phase7/tables")
    t_zne = ctx.load_table("zne_accuracy.csv", "phase7/tables")
    f0 = noise["hva"]
    M = {m: i for i, m in enumerate(f0["modes"])}
    K = {k: i for i, k in enumerate(f0["kinds"])}
    FS = {f: i for i, f in enumerate(f0["feature_sets"])}
    CL = {c: i for i, c in enumerate(f0["classifiers"])}
    kk, fc, lr = K[th.ml_main_kind], FS[th.ml_main_features], CL[th.ml_main_classifier]
    z0 = zne["hva", sizes[0]]
    ZM = {m: i for i, m in enumerate(z0["modes"])}
    ZU = {u: i for i, u in enumerate(z0["methods"])}
    ctx.param(size=Np, variant=var, kind_accuracy=th.ml_main_kind, zne_kind="exact", zne_mode="clean_vqe",
              zne_methods=["gate", "zne_gate"], zne_sizes=sizes)

    # -- validation ------------------------------------------------------------
    rng_ok, bad = True, 0
    for d in noise.values():
        acc = np.asarray(d["acc"])
        rng_ok &= bool(np.nanmin(acc) >= 0 and np.nanmax(acc) <= 1)
        bad += crossing_consistent(d["hc"], d["P"], getattr(ml, "decision_threshold", 0.5))[0]
    ctx.check_true("accuracy in [0, 1]; missing h_c^ML <=> isotonic fit never crosses 1/2", rng_ok and bad == 0,
                   f"{bad} inconsistent entries")
    ratios, ratios_ro = [], []
    for d in zne.values():
        r = np.asarray(d["feature_rms_to_clean"])
        ratios.append(np.max(r[ZU["zne_gate"]] / r[ZU["gate"]]))
        ratios_ro.append(np.max(r[ZU["zne_ro"]] / r[ZU["ro"]]))
    ctx.note(f"feature RMS ratio ZNE / unmitigated: exact gate {min(ratios):.3f} to {max(ratios):.3f}; "
             f"readout-corrected {min(ratios_ro):.3f} to {max(ratios_ro):.3f}")
    ctx.check_max(f"ZNE leaves less than {th.zne_feature_ratio_max:.0%} of the exact feature error "
                  "(gate noise; every size, ansatz and window point)", ratios, th.zne_feature_ratio_max)
    wR = extrapolation_weights(np.asarray(dsz["hva", sizes[0]]["scales"], dtype=float), 2)
    ctx.check_max("Phase 6 ZNE datasets use the Richardson weights (15/8, -5/4, 3/8)",
                  [np.max(np.abs(np.asarray(d["coefficients"]) - wR)) for d in dsz.values()], 1e-12)

    mine, theirs = [], []
    for a, _ in ANSATZE:
        d = noise[a]
        acc = np.nanmean(np.asarray(d["acc"])[:, :, kk, fc, lr], axis=-1)
        for m, im in M.items():
            for j, sj in enumerate(np.asarray(d["strengths"])):
                mine.append(acc[im, j])
                theirs.append(table_value(t_noise, "acc", ansatz=a, N=Np, variant=var, mode=m, strength=float(sj),
                                          kind=th.ml_main_kind, features=th.ml_main_features,
                                          classifier=th.ml_main_classifier))
    shift = {}
    for (a, N), d in zne.items():
        s = summarize_hc(d["hc"], d["hc_rep"])
        ref = summarize_hc(clean[a, N]["hc"])["mean"][K["exact"], fc, lr]
        for meth in ("gate", "zne_gate"):
            for w, sw in enumerate(np.asarray(d["strengths"])):
                v = s["mean"][ZM["clean_vqe"], ZU[meth], w, K["exact"], fc, lr]
                shift[a, N, meth, w] = (float(d["lam"][w]), v - ref, v, ref)
                mine.append(v)
                theirs.append(table_value(t_zne, "hc_mean", ansatz=a, N=N, mode="clean_vqe", method=meth,
                                          strength=float(sw), kind="exact", features=th.ml_main_features,
                                          classifier=th.ml_main_classifier))
    mine, theirs = np.array(mine, dtype=float), np.array(theirs, dtype=float)
    both = np.isfinite(mine) & np.isfinite(theirs)
    ctx.check_true("accuracies and ZNE h_c^ML equal noise_accuracy.csv and zne_accuracy.csv (4 decimals)",
                   bool(np.all(np.isfinite(mine) == np.isfinite(theirs))) and
                   bool(np.all(np.abs(mine[both] - theirs[both]) <= th.table_tol)),
                   f"{both.sum()} cells, max difference {np.max(np.abs(mine[both] - theirs[both])):.1e}")
    zz = [abs(v[1]) for k, v in shift.items() if k[2] == "zne_gate"]
    uu = [abs(v[1]) for k, v in shift.items() if k[2] == "gate"]
    ctx.note(f"ZNE window: |h_c^ML - noiseless| unmitigated {min(uu):.3f} to {max(uu):.3f}, ZNE {min(zz):.4f} to "
             f"{max(zz):.4f} over {len(zz)} points (exact features, trained on the noiseless circuit)")
    for a, name in ANSATZE:
        for w in range(2):
            lam, du, vu, ref = shift[a, Np, "gate", w]
            _, dz, vz, _ = shift[a, Np, "zne_gate", w]
            ctx.note(f"{name} N={Np} lambda = {lam:.2f}: unmitigated {vu:.3f}, ZNE {vz:.3f}, noiseless {ref:.3f}")
    for a, name in ANSATZE:
        acc = np.nanmean(np.asarray(noise[a]["acc"])[:, :, kk, fc, lr], axis=-1)
        ctx.note(f"{name}: accuracy at lambda = {noise[a]['lam'][-1]:.2f}: " +
                 ", ".join(f"{m} {acc[im, -1]:.3f}" for m, im in M.items()))

    # -- figure --------------------------------------------------------------------
    fig, axes = thesis_figure(ncols=3, aspect=0.38)
    for ax, (a, name) in zip(axes[:2], ANSATZE):
        d = noise[a]
        lam = np.asarray(d["lam"])
        x, x0 = zero_strength_positions(lam)
        acc = np.nanmean(np.asarray(d["acc"])[:, :, kk, fc, lr], axis=-1)
        for m, im in M.items():
            ax.semilogx(x, acc[im], "o-", color=MODE_COLORS[m], ms=2.5, lw=0.9, label=MODE_LABELS[m])
        reference_line(ax, ALL_DISORDERED, color=COLORS["grey"], lw=0.6, ls=":",
                       label="all fields called disordered")
        ax.set_ylim(0.45, 1.0)
        ax.set_xlabel(r"error budget $\lambda$")
        ax.set_ylabel("accuracy, all test fields")
        ax.set_title(f"{name}, $N={Np}$", loc="right")
        label_zero_strength(ax, x0)
    axes[0].legend(loc="lower left")
    ax = axes[2]
    for (a, N, meth, w), (lam, dv, _, _) in shift.items():
        name = dict(ANSATZE)[a]
        ax.plot(lam, dv, ANSATZ_MARKERS[name], color=size_color(N), ms=4,
                mfc=size_color(N) if meth == "zne_gate" else "none")
    for (a, N, meth, w), (lam, dv, _, _) in shift.items():
        if meth == "gate":
            lz, dz = shift[a, N, "zne_gate", w][:2]
            ax.annotate("", (lz, dz), (lam, dv), arrowprops=dict(arrowstyle="->", lw=0.5, color=COLORS["grey"]))
    reference_line(ax, 0.0, lw=0.6)
    lo_y = min(v[1] for v in shift.values())
    ax.set_ylim(lo_y * 1.9 if lo_y < 0 else -0.01, max(0.004, -0.25 * lo_y))
    ax.set_xlim(0.0, 1.15 * max(v[0] for v in shift.values()))
    ax.set_xlabel(r"error budget $\lambda$")
    ax.set_ylabel(r"$h_c^{\mathrm{ML}} - h_c^{\mathrm{ML}}(\mathrm{noiseless})$")
    hs = [ax.plot([], [], "o", color=size_color(N), ms=4, label=f"$N={N}$")[0] for N in sizes]
    hs += [ax.plot([], [], "o", color=COLORS["black"], ms=4, mfc="none", label="unmitigated")[0],
           ax.plot([], [], "o", color=COLORS["black"], ms=4, label="ZNE")[0],
           ax.plot([], [], ANSATZ_MARKERS["HEA"], color=COLORS["black"], ms=4, mfc="none", label="squares: HEA")[0]]
    ax.legend(handles=hs, loc="lower left", ncol=2, columnspacing=0.8)
    ax.set_title("ZNE window", loc="right")
    panel_labels(axes)
    return fig


if __name__ == "__main__":
    raise SystemExit(run_figure(SPEC, build))
