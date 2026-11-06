"""
Phase 7: validation checks, figures, accuracy tables and summary.

Run from the project root after experiments\\phase7_classify.py:

    python experiments\\phase7_plots.py

Writes figures/phase7/*.pdf|png, results/phase7/tables/*.csv and
results/phase7/phase7_summary.txt. Missing result files turn their checks
into notes ("skipped"), so the script also runs on a partial set of results.

Checks against exact results (pass/fail):

    C1  classifiers trained on ED exact features give the same outputs as on the
        exact free-fermion correlators (same N)
    C2  ED h_c^ML(N) equals the free-fermion value, increases with N, stays below h_c,
        and the free-fermion series keeps increasing up to N = 64
    C3  noisy strength 0 ("gate") reproduces the noiseless VQE classifiers exactly
        (common random numbers), also for the ED-trained classifier
    C4  readout correction is exact at strength 0: "ro" equals "gate" on exact features
    C5  shadow and exact h_c^ML agree for ED within hc_sigma combined sigma + hc_floor
    C6  PC1 tracks the exact M^2: |corr| >= pca_corr_min (ED and free fermions)
    C7  shadows rebuilt from all stored snapshots equal the stored shadows
    C8  ZNE brings exact features and h_c^ML back toward the noiseless circuit
    C9  clean ED shadows are classified without error on the deep test fields (logistic)
    C10 a live rerun reproduces a saved result bit for bit; exact logistic results do not
        depend on the seed
"""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

from config import CFG, RESULTS_DIR                               # noqa: E402
from ml import classifiers as C                                   # noqa: E402
from ml import dataset as D                                       # noqa: E402
from utils.io import load_results                                 # noqa: E402
from utils.plotting import COLORS, PALETTE, SERIES_COLORS, save_figure, set_thesis_style   # noqa: E402

SUB = "phase7"
FIG = "phase7"
TABLES = RESULTS_DIR / SUB / "tables"
KIND = {k: i for i, k in enumerate(C.KINDS)}
FSET = {f: i for i, f in enumerate(CFG.ml.feature_sets_phase7)}
CLF = {c: i for i, c in enumerate(C.CLASSIFIERS)}
MODE = {m: i for i, m in enumerate(CFG.ml.noise_modes)}
MODE_LABEL = {"clean_ed": "train ED", "clean_vqe": "train noiseless circuit", "noisy": "train = test noise",
              "mixed": "train mixed", "loo": "train mixed, test level held out"}
MODE_COLOR = {"clean_ed": COLORS["black"], "clean_vqe": COLORS["blue"], "noisy": COLORS["red"],
              "mixed": COLORS["green"], "loo": COLORS["purple"]}
CLF_STYLE = {"logistic": "-", "mlp": "--"}
CLF_LABEL = {"logistic": "LR", "mlp": "MLP"}

checks: list = []
notes: list = []


def check(name: str, ok: bool, msg: str) -> None:
    checks.append((name, bool(ok), msg))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")


def note(name: str, msg: str) -> None:
    notes.append((name, msg))
    print(f"[note] {name}: {msg}")


def load(name: str, sub: str):
    try:
        return load_results(name, subdir=f"{SUB}/{sub}")[0]
    except FileNotFoundError:
        return None


def noisy_depth(N: int) -> int:
    return D.resolve_depth(CFG.ml.noisy_depth_rule, N)


def noise_file(a, N, variant, channel=None):
    ch = channel or CFG.ml.noisy_channels[0]
    return load(f"noise_{a}_N{N}_p{noisy_depth(N)}_{ch}_{variant}", "noise")


def zne_file(a, N, channel=None):
    ch = channel or CFG.ml.noisy_channels[0]
    return load(f"zne_{a}_N{N}_p{noisy_depth(N)}_{ch}", "zne")


def hc_stats(a, prefix=""):
    return C.summarize_hc(a[prefix + "hc"], a[prefix + "hc_rep"])


def fmt(x, nd=3):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def strength_axis(strengths):
    """x positions for a log axis with p = 0 drawn one decade-third left of the first nonzero strength."""
    s = np.asarray(strengths, dtype=float)
    x = s.copy()
    x0 = s[s > 0].min() / 3.0
    x[s == 0] = x0
    return x, x0


def label_zero(ax, x0):
    ax.set_xscale("log")
    ticks = [x0] + [t for t in (1e-5, 1e-4, 1e-3, 1e-2) if t >= x0 * 2]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0"] + [f"$10^{{{int(np.log10(t))}}}$" for t in ticks[1:]])
    ax.axvspan(x0 / 1.6, x0 * 1.6, color="0.9", zorder=0)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_all() -> dict:
    R = {"ed": {}, "vqe": {}, "noise": {}, "zne": {}, "shots": {}, "learning": {}}
    for N in CFG.ml.ed_sizes:
        a = load(f"clean_ed_N{N}", "clean")
        if a is not None:
            R["ed"][N] = a
    for ans in CFG.vqe.ansatze:
        for N in CFG.ml.vqe_sizes:
            for p in sorted({D.resolve_depth(r, N) for r in CFG.ml.vqe_depth_rules}):
                a = load(f"clean_vqe_{ans}_N{N}_p{p}", "clean")
                if a is not None:
                    R["vqe"][(ans, N, p)] = a
            if N in CFG.ml.noisy_sizes:
                for v in D.NOISY_VARIANTS:
                    a = noise_file(ans, N, v)
                    if a is not None:
                        R["noise"][(ans, N, v)] = a
                a = zne_file(ans, N)
                if a is not None:
                    R["zne"][(ans, N)] = a
    R["ff"] = load("ff_exact", "ff")
    Ns = CFG.ml.study_size
    for e in ("shots", "learning"):
        a = load(f"{e}_ed_N{Ns}", e)
        if a is not None:
            R[e]["ed"] = a
        for ans in CFG.vqe.ansatze:
            a = load(f"{e}_{ans}_N{Ns}_p{noisy_depth(Ns)}_{CFG.ml.noisy_channels[0]}", e)
            if a is not None:
                R[e][ans] = a
    return R


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def checks_clean(R):
    ff = R["ff"]
    ed = R["ed"]
    lr, k_ex, k_sh, fc = CLF["logistic"], KIND["exact"], KIND["shadow"], FSET["corr"]
    # C1, C2
    if ff is None or not ed:
        note("C1/C2", "skipped (ff or clean ED results missing)")
    else:
        sizes = [int(v) for v in np.asarray(ff["sizes"])]
        dP, dh = 0.0, 0.0
        for N, a in ed.items():
            if N not in sizes:
                continue
            i = sizes.index(N)
            dP = max(dP, float(np.max(np.abs(a["P"][k_ex] - ff["P"][i]))))
            dh = max(dh, float(np.nanmax(np.abs(a["hc"][k_ex] - ff["hc"][i]))))
        check("C1 free-fermion identity", dP < CFG.ml.identity_tol,
              f"max |P_ED - P_ff| = {dP:.1e} over N = {sorted(ed)} (both classifiers, feature sets, seeds); "
              f"max |h_c difference| = {dh:.1e}")
        Ns = sorted(ed)
        hc_ed = np.array([np.nanmean(ed[N]["hc"][k_ex, fc, lr]) for N in Ns])
        hc_ff = np.nanmean(ff["hc"][:, fc, lr], axis=-1)
        mono_ed = bool(np.all(np.diff(hc_ed) > 0))
        mono_ff = bool(np.all(np.diff(hc_ff) > 0))
        below = bool(np.all(hc_ff < CFG.tfim.h_critical))
        check("C2 finite-size drift", mono_ed and mono_ff and below,
              "ED h_c^ML (LR, exact) = " + ", ".join(f"{fmt(v)}" for v in hc_ed) + f" for N = {Ns}; "
              f"free fermions to N = {sizes[-1]}: " + ", ".join(fmt(v) for v in hc_ff) +
              f"; increasing: ED {mono_ed}, ff {mono_ff}; all below h_c: {below}")
        x = 1.0 / np.array(sizes[-3:], dtype=float)
        slope, inter = np.polyfit(x, hc_ff[-3:], 1)
        note("N-ff", f"linear 1/N extrapolation of the last three free-fermion sizes ({sizes[-3:]}): "
                     f"h_c^ML(inf) = {inter:.3f} (slope {slope:.2f}); a fixed-threshold classifier "
                     f"converges below h_c = 1 (hypothesis, see report)")
        r = np.asarray(ff["pca_r_m2"])
        r_ed = [float(ed[N]["pca_r_m2"][k_ex]) for N in Ns]
        ok = bool(np.all(np.array(r_ed) >= CFG.ml.pca_corr_min) and np.all(r >= CFG.ml.pca_corr_min))
        evr = [float(ed[N]["pca_evr"][k_ex, 0]) for N in Ns]
        check("C6 PCA tracks M^2", ok,
              "|corr(PC1, M^2)| ED exact = " + ", ".join(f"{v:.4f}" for v in r_ed) +
              f"; free fermions min {np.min(r):.4f}; PC1 explained variance " +
              ", ".join(f"{v:.3f}" for v in evr))
    if not ed:
        return
    # C5, C9
    bad, rows, deep = [], [], []
    for N, a in sorted(ed.items()):
        s = hc_stats(a)
        for fs, jf in FSET.items():
            e = s["mean"][k_ex, jf, lr]
            sh = s["mean"][k_sh, jf, lr]
            sig = s["total"][k_sh, jf, lr]
            tol = CFG.ml.hc_sigma * sig + CFG.ml.hc_floor
            rows.append(f"N={N} {fs}: {fmt(sh)} vs {fmt(e)} (tol {tol:.3f})")
            if not abs(sh - e) <= tol:
                bad.append(f"N={N} {fs}")
        deep.append(float(np.nanmin(a["acc_deep"][k_sh, :, lr])))
    check("C5 shadow vs exact h_c^ML (ED, LR)", not bad, "; ".join(rows) + (f"; outside: {bad}" if bad else ""))
    check("C9 deep-field accuracy (ED shadows, LR)", min(deep) == 1.0,
          "minimum deep-field accuracy per N = " + ", ".join(f"{v:.3f}" for v in deep))
    mlp_deep = [float(np.nanmin(a["acc_deep"][k_sh, :, CLF["mlp"]])) for _, a in sorted(ed.items())]
    note("N-mlp-deep", "MLP minimum deep-field accuracy per N = " + ", ".join(f"{v:.3f}" for v in mlp_deep))
    seed_spread = max(float(np.nanmax(np.nanstd(a["hc"][k_ex, :, lr], axis=-1))) for a in ed.values())
    check("C10b exact logistic is seed independent", seed_spread < 1e-12,
          f"largest seed spread of exact LR h_c^ML = {seed_spread:.1e}")


def checks_noise(R):
    # C3: strength 0 gate == noiseless VQE dataset (clean file), including the ED-trained classifier
    worst, worst_ed, n = 0.0, 0.0, 0
    for (ans, N, v), a in R["noise"].items():
        if v != "gate":
            continue
        c = R["vqe"].get((ans, N, noisy_depth(N)))
        if c is None:
            continue
        n += 1
        worst = max(worst, float(np.max(np.abs(a["P"][MODE["noisy"], 0] - c["P"]))),
                    float(np.max(np.abs(a["P"][MODE["clean_vqe"], 0] - c["P"]))))
        if "ed_P" in c:
            worst_ed = max(worst_ed, float(np.max(np.abs(a["P"][MODE["clean_ed"], 0] - c["ed_P"]))))
    if n == 0:
        note("C3", "skipped (no gate noise file with a matching clean VQE file)")
    else:
        check("C3 strength-0 identity", max(worst, worst_ed) < CFG.ml.identity_tol,
              f"{n} datasets: max |P(noisy, p=0) - P(noiseless VQE)| = {worst:.1e}; "
              f"ED-trained classifier {worst_ed:.1e}")
    # C4: ro == gate at strength 0, exact features
    worst, n = 0.0, 0
    for (ans, N, v), a in R["noise"].items():
        if v != "ro" or (ans, N, "gate") not in R["noise"]:
            continue
        g = R["noise"][(ans, N, "gate")]
        n += 1
        for m in ("noisy", "clean_ed", "clean_vqe"):
            worst = max(worst, float(np.max(np.abs(a["P"][MODE[m], 0, KIND["exact"]] -
                                                   g["P"][MODE[m], 0, KIND["exact"]]))))
    if n == 0:
        note("C4", "skipped (needs both ro and gate files)")
    else:
        check("C4 readout correction exact at p = 0", worst < CFG.ml.identity_tol,
              f"{n} datasets: max |P_ro - P_gate| (exact features, p = 0) = {worst:.1e}")


def checks_zne(R):
    if not R["zne"]:
        note("C8", "skipped (no ZNE results)")
        return
    lr, k_ex, fc = CLF["logistic"], KIND["exact"], FSET["corr"]
    feat_ok, hc_ok, rows = True, True, []
    for (ans, N), z in sorted(R["zne"].items()):
        meth = list(z["methods"])
        d = z["feature_rms_to_clean"]
        c = R["vqe"].get((ans, N, noisy_depth(N)))
        hc_clean = np.nanmean(c["hc"][k_ex, fc, lr]) if c is not None else np.nan
        im = list(z["modes"]).index("clean_vqe")
        for w, lam in enumerate(z["lam"]):
            r_gate = d[meth.index("zne_gate"), w] / d[meth.index("gate"), w]
            r_ro = d[meth.index("zne_ro"), w] / d[meth.index("ro"), w]
            feat_ok &= bool(r_gate < 1 and r_ro < 1)
            hg = np.nanmean(z["hc"][im, meth.index("gate"), w, k_ex, fc, lr])
            hz = np.nanmean(z["hc"][im, meth.index("zne_gate"), w, k_ex, fc, lr])
            if np.isfinite(hc_clean):
                hc_ok &= bool(abs(hz - hc_clean) <= abs(hg - hc_clean) + 1e-3)
            rows.append(f"{ans} N={N} lambda={lam:.2f}: feature error ratio {r_gate:.3f} (gate) {r_ro:.3f} (ro); "
                        f"h_c {fmt(hg)} -> {fmt(hz)} (noiseless {fmt(hc_clean)})")
    check("C8 ZNE restores features and h_c^ML (exact, LR)", feat_ok and hc_ok, "; ".join(rows))


def checks_shots_rerun(R):
    errs = [float(np.nanmax(a["rebuild_err"])) for a in R["shots"].values()]
    if errs:
        check("C7 shadow rebuild", max(errs) <= 1e-12,
              f"max |rebuilt - stored| at M = {CFG.ml.n_shadow_snapshots}: {max(errs):.1e} ({len(errs)} files)")
    else:
        note("C7", "skipped (no shots results)")
    N = CFG.ml.study_size
    a = R["ed"].get(N)
    if a is None:
        note("C10", "skipped (clean ED file missing)")
        return
    ds = D.load_dataset("ed", N)
    h, y = np.asarray(ds["h"]), np.asarray(ds["y"])
    tr, te = C.split_array(ds, C.dataset_array(ds, "shadow"), "shadow", "corr")
    res = C.evaluate([tr], y[np.asarray(ds["train_field"], dtype=bool)], te, h, y)
    d = float(np.max(np.abs(res["P"] - a["P"][KIND["shadow"], FSET["corr"]])))
    check("C10 live rerun", d <= 1e-12, f"ED N = {N}, shadows, both classifiers, all seeds: max |P_rerun - P_saved| = {d:.1e}")


# ---------------------------------------------------------------------------
# Tables (deliverable)
# ---------------------------------------------------------------------------
def write_csv(path: Path, header: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[table] {path}")


STAT_HEADER = ["hc_mean", "hc_seed_std", "hc_shot_se", "hc_total_sigma", "n_valid_seeds",
               "acc", "acc_window", "acc_deep"]


def stat_cells(a, idx, prefix=""):
    s = C.summarize_hc(a[prefix + "hc"][idx], a[prefix + "hc_rep"][idx])
    acc = [np.nanmean(a[prefix + k][idx]) for k in ("acc", "acc_window", "acc_deep")]
    return [fmt(s["mean"], 4), fmt(s["seed_std"], 4), fmt(s.get("shot_se", np.nan), 4),
            fmt(s.get("total", np.nan), 4), int(s["n_valid"])] + [fmt(v, 4) for v in acc]


def tables(R):
    rows = []
    for N, a in sorted(R["ed"].items()):
        for k, ik in KIND.items():
            for f, jf in FSET.items():
                for c, ic in CLF.items():
                    rows.append(["ed", "ed", N, 0, "ed", k, f, c] + stat_cells(a, (ik, jf, ic)))
    for (ans, N, p), a in sorted(R["vqe"].items()):
        for pre, train in (("", "same"), ("ed_", "ed")):
            if pre + "hc" not in a:
                continue
            for k, ik in KIND.items():
                for f, jf in FSET.items():
                    for c, ic in CLF.items():
                        rows.append(["vqe", ans, N, p, train, k, f, c] + stat_cells(a, (ik, jf, ic), pre))
    write_csv(TABLES / "clean_accuracy.csv",
              ["source", "ansatz", "N", "depth", "trained_on", "kind", "features", "classifier"] + STAT_HEADER, rows)
    rows = []
    for (ans, N, v), a in sorted(R["noise"].items()):
        for m, im in MODE.items():
            for js, (s, lam) in enumerate(zip(a["strengths"], a["lam"])):
                for k, ik in KIND.items():
                    for f, jf in FSET.items():
                        for c, ic in CLF.items():
                            rows.append([ans, N, int(a["depth"]), v, m, f"{s:.4g}", f"{lam:.4g}", k, f, c]
                                        + stat_cells(a, (im, js, ik, jf, ic)))
    write_csv(TABLES / "noise_accuracy.csv",
              ["ansatz", "N", "depth", "variant", "mode", "strength", "lambda", "kind", "features", "classifier"]
              + STAT_HEADER, rows)
    rows = []
    for (ans, N), z in sorted(R["zne"].items()):
        for im, m in enumerate(z["modes"]):
            for iu, meth in enumerate(z["methods"]):
                for w, (s, lam) in enumerate(zip(z["strengths"], z["lam"])):
                    for k, ik in KIND.items():
                        for f, jf in FSET.items():
                            for c, ic in CLF.items():
                                if not np.any(np.isfinite(z["P"][im, iu, w, ik, jf, ic])):
                                    continue
                                rows.append([ans, N, m, meth, f"{s:.4g}", f"{lam:.4g}", k, f, c]
                                            + stat_cells(z, (im, iu, w, ik, jf, ic)))
    write_csv(TABLES / "zne_accuracy.csv",
              ["ansatz", "N", "mode", "method", "strength", "lambda", "kind", "features", "classifier"]
              + STAT_HEADER, rows)


# ---------------------------------------------------------------------------
# Summary notes on the results
# ---------------------------------------------------------------------------
def result_notes(R):
    lr, mlp, k_ex, k_sh, fc = CLF["logistic"], CLF["mlp"], KIND["exact"], KIND["shadow"], FSET["corr"]
    for N, a in sorted(R["ed"].items()):
        s = hc_stats(a)
        note(f"ED N={N}", "h_c^ML exact LR/MLP " + "/".join(fmt(s["mean"][k_ex, fc, i]) for i in (lr, mlp)) +
             "; shadows LR " + f"{fmt(s['mean'][k_sh, fc, lr])} +- {fmt(s['total'][k_sh, fc, lr])}, MLP "
             f"{fmt(s['mean'][k_sh, fc, mlp])} +- {fmt(s['total'][k_sh, fc, mlp])}; window accuracy (shadows) LR "
             f"{np.nanmean(a['acc_window'][k_sh, fc, lr]):.3f}, MLP {np.nanmean(a['acc_window'][k_sh, fc, mlp]):.3f}; "
             f"PCA h_mid {fmt(a['pca_h_mid'][k_ex])}, steepest {fmt(a['pca_h_steep'][k_ex], 2)}")
    for (ans, N, p), a in sorted(R["vqe"].items()):
        s = hc_stats(a)
        msg = f"h_c^ML shadows LR {fmt(s['mean'][k_sh, fc, lr])}, MLP {fmt(s['mean'][k_sh, fc, mlp])}"
        if "ed_hc" in a:
            se = hc_stats(a, "ed_")
            msg += (f"; ED-trained LR {fmt(se['mean'][k_sh, fc, lr])} (accuracy "
                    f"{np.nanmean(a['ed_acc'][k_sh, fc, lr]):.3f})")
        note(f"VQE {ans} N={N} p={p}", msg)
    v = CFG.ml.default_variant
    for (ans, N, var), a in sorted(R["noise"].items()):
        if var != v:
            continue
        s = hc_stats(a)
        for m, im in MODE.items():
            mean, tot = s["mean"][im, :, k_sh, fc, lr], s["total"][im, :, k_sh, fc, lr]
            ref, rs = mean[0], tot[0]
            drift = np.abs(mean - ref) > 2 * np.sqrt(tot ** 2 + rs ** 2)
            onset = np.flatnonzero(drift & (np.asarray(a["strengths"]) > 0))
            on = (f"p = {a['strengths'][onset[0]]:.2g} (lambda = {a['lam'][onset[0]]:.2f})"
                  if onset.size else "none")
            acc = np.nanmean(a["acc"][im, :, k_sh, fc, lr], axis=-1)
            note(f"noise {ans} N={N} {v} {m}",
                 f"LR shadows h_c^ML {fmt(mean[0])} (p = 0) -> {fmt(mean[-1])} (p = {a['strengths'][-1]:.0e}); "
                 f"first 2-sigma drift at {on}; accuracy {acc[0]:.3f} -> {acc[-1]:.3f}; "
                 f"no crossing at {int(np.sum(s['n_valid'][im, :, k_sh, fc, lr] == 0))} strengths")
    for key, a in R["shots"].items():
        s = hc_stats(a)
        M = np.asarray(a["shots"], dtype=float)
        im = list(a["modes"]).index("noisy")
        se = s["shot_se"][im, 0, :, lr]
        ok = np.isfinite(se) & (se > 0)
        slope = np.polyfit(np.log(M[ok]), np.log(se[ok]), 1)[0] if ok.sum() >= 2 else np.nan
        note(f"shots {key}", "LR h_c^ML vs M = " + ", ".join(f"{int(m)}: {fmt(x)}" for m, x in
                                                           zip(M, s["mean"][im, 0, :, lr])) +
             f"; shot-noise SE slope in log M = {fmt(slope, 2)} (expected -0.5)")
    for key, a in R["learning"].items():
        acc = np.nanmean(a["acc"][..., lr, :], axis=-1)
        note(f"learning {key}", "LR accuracy vs training rows " +
             ", ".join(f"{int(n)}: {x:.3f}" for n, x in zip(a["n_train_rows"], acc[0])))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_output_curves(R):
    if not R["ed"]:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0), sharey=True)
    for ax, (c, ic) in zip(axes, CLF.items()):
        for i, (N, a) in enumerate(sorted(R["ed"].items())):
            col = PALETTE[i % len(PALETTE)]
            h = a["h"]
            ax.plot(h, np.nanmean(a["P"][KIND["exact"], FSET["corr"], ic], axis=0), color=col, lw=1.2)
            Psh = a["P"][KIND["shadow"], FSET["corr"], ic]
            ax.errorbar(h, np.nanmean(Psh, axis=0), yerr=np.nanstd(Psh, axis=0, ddof=1), fmt="o", ms=2.5,
                        color=col, elinewidth=0.8, label=f"N = {N}")
        ax.axvline(CFG.tfim.h_critical, color="0.5", ls=":")
        ax.axhline(0.5, color="0.5", ls=":")
        for x in (CFG.ml.train_h_ordered_max, CFG.ml.train_h_disordered_min):
            ax.axvline(x, color="0.8", lw=0.8)
        ax.set_xlabel("$h/J$")
        ax.set_title(CLF_LABEL[c])
    axes[0].set_ylabel(r"$P(h)$: mean $p(y=1\,|\,x)$")
    axes[1].legend(fontsize=7, loc="lower right")
    save_figure(fig, "fig1_output_curves_ed", FIG)
    plt.close(fig)


def fig_hc_vs_N(R):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    lr, k_ex, k_sh, fc = CLF["logistic"], KIND["exact"], KIND["shadow"], FSET["corr"]
    if R["ff"] is not None:
        sz = np.asarray(R["ff"]["sizes"], dtype=float)
        ax.plot(1 / sz, np.nanmean(R["ff"]["hc"][:, fc, lr], axis=-1), "-", color="0.6", label="free fermions, exact (LR)")
    if R["ed"]:
        Ns = np.array(sorted(R["ed"]), dtype=float)
        for kind, mk in (("exact", "s"), ("shadow", "o")):
            for c, ic in CLF.items():
                st = [hc_stats(R["ed"][int(N)]) for N in Ns]
                m = [s["mean"][KIND[kind], fc, ic] for s in st]
                e = [s["total"][KIND[kind], fc, ic] for s in st]
                ax.errorbar(1 / Ns + (0.002 if c == "mlp" else 0), m, yerr=e, fmt=mk, mfc="none" if c == "mlp" else None,
                            color=SERIES_COLORS["ED"], ms=4, label=f"ED {kind} ({CLF_LABEL[c]})")
    for ans in CFG.vqe.ansatze:
        pts = [(N, hc_stats(a)) for (an, N, p), a in R["vqe"].items() if an == ans and p == noisy_depth(N)]
        if pts:
            pts.sort()
            ax.errorbar([1 / N for N, _ in pts], [s["mean"][k_sh, fc, lr] for _, s in pts],
                        yerr=[s["total"][k_sh, fc, lr] for _, s in pts], fmt="^-", ms=4,
                        color=SERIES_COLORS[ans.upper()], label=f"{ans.upper()} p = N/2, shadows (LR)")
    ax.axhline(CFG.tfim.h_critical, color="0.5", ls=":")
    ax.set_xlabel("$1/N$")
    ax.set_ylabel(r"$h_c^{\mathrm{ML}}$")
    ax.set_xlim(left=0)
    ax.legend(fontsize=6.5)
    save_figure(fig, "fig2_hc_vs_N", FIG)
    plt.close(fig)


def fig_pca(R):
    if not R["ed"]:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
    for i, (N, a) in enumerate(sorted(R["ed"].items())):
        col = PALETTE[i % len(PALETTE)]
        h = a["h"]
        for kind, ls in (("exact", "-"), ("shadow", "o")):
            pc = a["pca_pc1"][KIND[kind]]
            scale = np.nanmax(np.abs(a["pca_pc1"][KIND["exact"]]))
            if kind == "exact":
                axes[0].plot(h, pc / scale, ls, color=col, lw=1.2, label=f"N = {N}")
            else:
                axes[0].plot(h, pc / scale, ls, color=col, ms=2.5)
        axes[1].plot(a["m2_exact"], a["pca_pc1"][KIND["exact"]] / np.nanmax(np.abs(a["pca_pc1"][KIND["exact"]])),
                     "-", color=col, lw=1.2)
    axes[0].axvline(CFG.tfim.h_critical, color="0.5", ls=":")
    axes[0].set_xlabel("$h/J$")
    axes[0].set_ylabel("PC1 (normalized)")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("$M^2$ (exact)")
    axes[1].set_ylabel("PC1 (exact features)")
    save_figure(fig, "fig3_pca", FIG)
    plt.close(fig)


def _noise_panel(ax, a, quantity, clfs=("logistic", "mlp"), kind="shadow", fs="corr"):
    x, x0 = strength_axis(a["strengths"])
    for m, im in MODE.items():
        for c in clfs:
            ic = CLF[c]
            if quantity == "hc":
                s = hc_stats(a)
                yv, ye = s["mean"][im, :, KIND[kind], FSET[fs], ic], s["total"][im, :, KIND[kind], FSET[fs], ic]
            else:
                v = a[quantity][im, :, KIND[kind], FSET[fs], ic]
                yv, ye = np.nanmean(v, axis=-1), np.nanstd(v, axis=-1, ddof=1)
            ax.errorbar(x * (1.06 if c == "mlp" else 1.0), yv, yerr=ye, fmt="o" + CLF_STYLE[c], ms=3, lw=1.1,
                        color=MODE_COLOR[m], label=MODE_LABEL[m] if c == clfs[0] else None)
    for c in clfs:
        ax.plot([], [], CLF_STYLE[c], color="0.3", label=CLF_LABEL[c])
    label_zero(ax, x0)
    top = ax.secondary_xaxis("top", functions=(lambda s: s * a["lam"][-1] / a["strengths"][-1],
                                               lambda l: l * a["strengths"][-1] / a["lam"][-1]))
    top.set_xlabel(r"error budget $\lambda$", fontsize=9)
    return x0


def fig_noise(R, quantity, name, ylabel):
    N, v = CFG.ml.study_size, CFG.ml.default_variant
    files = [(ans, R["noise"].get((ans, N, v))) for ans in CFG.vqe.ansatze]
    files = [(ans, a) for ans, a in files if a is not None]
    if not files:
        return
    fig, axes = plt.subplots(1, len(files), figsize=(3.3 * len(files), 3.4), sharey=True, squeeze=False)
    for ax, (ans, a) in zip(axes[0], files):
        _noise_panel(ax, a, quantity)
        if quantity == "hc" and N in R["ed"]:
            s = hc_stats(R["ed"][N])
            m, e = s["mean"][KIND["shadow"], FSET["corr"], CLF["logistic"]], s["total"][KIND["shadow"], FSET["corr"], CLF["logistic"]]
            ax.axhspan(m - e, m + e, color="0.85", zorder=0, label="ED, shadows (LR)")
        ax.set_title(f"{ans.upper()}, N = {N}, p = {noisy_depth(N)}, variant {v}", fontsize=9, pad=30)
        ax.set_xlabel("noise strength $p$")
    axes[0][0].set_ylabel(ylabel)
    axes[0][-1].legend(fontsize=5.5, loc="best")
    save_figure(fig, name, FIG)
    plt.close(fig)


def fig_curves_under_noise(R):
    N, v = CFG.ml.study_size, CFG.ml.default_variant
    files = [(ans, R["noise"].get((ans, N, v))) for ans in CFG.vqe.ansatze]
    files = [(ans, a) for ans, a in files if a is not None]
    if not files:
        return
    modes = ("clean_ed", "clean_vqe")
    fig, axes = plt.subplots(len(modes), len(files), figsize=(3.2 * len(files), 5.0), sharex=True, sharey=True,
                             squeeze=False)
    sel = [0, 4, 7, 9, 10]
    for j, (ans, a) in enumerate(files):
        for i, m in enumerate(modes):
            ax = axes[i][j]
            for k, js in enumerate(sel):
                if js >= len(a["strengths"]):
                    continue
                P = a["P"][MODE[m], js, KIND["shadow"], FSET["corr"], CLF["logistic"]]
                ax.plot(a["h"], np.nanmean(P, axis=0), "-o", ms=2, lw=1, color=PALETTE[k],
                        label=f"p = {a['strengths'][js]:.0e}" if a["strengths"][js] > 0 else "p = 0")
            ax.axvline(CFG.tfim.h_critical, color="0.5", ls=":")
            ax.axhline(0.5, color="0.5", ls=":")
            ax.set_title(f"{ans.upper()}: {MODE_LABEL[m]}", fontsize=8.5)
            if i == len(modes) - 1:
                ax.set_xlabel("$h/J$")
        axes[0][0].set_ylabel("$P(h)$")
        axes[1][0].set_ylabel("$P(h)$")
    axes[0][-1].legend(fontsize=6)
    save_figure(fig, "fig6_curves_under_noise", FIG)
    plt.close(fig)


def fig_variants_sizes(R):
    N = CFG.ml.study_size
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0), sharey=True)
    lr = CLF["logistic"]
    styles = {"gate": "-", "meas": ":", "ro": "--"}
    for ans in CFG.vqe.ansatze:
        for v, ls in styles.items():
            a = R["noise"].get((ans, N, v))
            if a is None:
                continue
            x, x0 = strength_axis(a["strengths"])
            s = hc_stats(a)
            axes[0].errorbar(x, s["mean"][MODE["clean_vqe"], :, KIND["shadow"], FSET["corr"], lr],
                             yerr=s["total"][MODE["clean_vqe"], :, KIND["shadow"], FSET["corr"], lr], fmt="o" + ls,
                             ms=2.5, color=SERIES_COLORS[ans.upper()], label=f"{ans.upper()} {v}")
            label_zero(axes[0], x0)
        for i, Nn in enumerate(CFG.ml.noisy_sizes):
            a = R["noise"].get((ans, Nn, CFG.ml.default_variant))
            if a is None:
                continue
            x, x0 = strength_axis(a["strengths"])
            s = hc_stats(a)
            axes[1].errorbar(x, s["mean"][MODE["noisy"], :, KIND["shadow"], FSET["corr"], lr],
                             yerr=s["total"][MODE["noisy"], :, KIND["shadow"], FSET["corr"], lr],
                             fmt=["s", "^", "o"][i % 3] + ("-" if ans == "hva" else "--"), ms=3,
                             color=SERIES_COLORS[ans.upper()], alpha=0.5 + 0.25 * i, label=f"{ans.upper()} N = {Nn}")
            label_zero(axes[1], x0)
    axes[0].set_title(f"train noiseless circuit, N = {N} (LR)", fontsize=9)
    axes[1].set_title(f"train = test noise, variant {CFG.ml.default_variant} (LR)", fontsize=9)
    axes[0].set_ylabel(r"$h_c^{\mathrm{ML}}$")
    for ax in axes:
        ax.set_xlabel("noise strength $p$")
        ax.legend(fontsize=6)
    save_figure(fig, "fig7_variants_and_sizes", FIG)
    plt.close(fig)


def fig_zne(R):
    if not R["zne"]:
        return
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    lr, k_ex, k_sh, fc = CLF["logistic"], KIND["exact"], KIND["shadow"], FSET["corr"]
    labels, xi = [], 0
    for (ans, N), z in sorted(R["zne"].items()):
        meth = list(z["methods"])
        im = list(z["modes"]).index("clean_vqe")
        s = hc_stats(z)
        c = R["vqe"].get((ans, N, noisy_depth(N)))
        ref = np.nanmean(c["hc"][k_ex, fc, lr]) if c is not None else np.nan
        for w, lam in enumerate(z["lam"]):
            for j, (m, k, col) in enumerate((("gate", k_ex, COLORS["red"]), ("zne_gate", k_ex, COLORS["green"]),
                                              ("ro", k_sh, COLORS["orange"]), ("zne_ro", k_sh, COLORS["sky"]))):
                ax.errorbar(xi + 0.15 * (j - 1.5), s["mean"][im, meth.index(m), w, k, fc, lr],
                            yerr=s["total"][im, meth.index(m), w, k, fc, lr], fmt="o", color=col, ms=4,
                            label=m + (" (exact)" if k == k_ex else " (shadows)") if xi == 0 else None)
            ax.plot([xi - 0.35, xi + 0.35], [ref, ref], color="k", lw=1, label="noiseless (exact)" if xi == 0 else None)
            labels.append(f"{ans.upper()} {N}\n$\\lambda$={lam:.2f}")
            xi += 1
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel(r"$h_c^{\mathrm{ML}}$ (train noiseless circuit, LR)")
    ax.legend(fontsize=6)
    save_figure(fig, "fig8_zne", FIG)
    plt.close(fig)


def fig_shots_learning(R):
    if not R["shots"] and not R["learning"]:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))
    lr = CLF["logistic"]
    for key, a in R["shots"].items():
        s = hc_stats(a)
        im = list(a["modes"]).index("noisy")
        col = SERIES_COLORS["ED"] if key == "ed" else SERIES_COLORS[key.upper()]
        for i, js in enumerate(a["strength_index"]):
            lab = "ED" if key == "ed" else f"{key.upper()} p = {a['strengths'][i]:.0e}"
            axes[0].errorbar(a["shots"] * (1 + 0.04 * i), s["mean"][im, i, :, lr], yerr=s["total"][im, i, :, lr],
                             fmt="o-", ms=3, color=col, alpha=1 - 0.2 * i, label=lab)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("snapshots per shadow $M$")
    axes[0].set_ylabel(r"$h_c^{\mathrm{ML}}$ (train = test, LR)")
    axes[0].legend(fontsize=5.5)
    for key, a in R["learning"].items():
        col = SERIES_COLORS["ED"] if key == "ed" else SERIES_COLORS[key.upper()]
        for i in range(a["acc"].shape[0]):
            acc = a["acc"][i, :, lr]
            lab = "ED" if key == "ed" else f"{key.upper()} p = {a['strengths'][i]:.0e}"
            axes[1].errorbar(a["n_train_rows"], np.nanmean(acc, -1), yerr=np.nanstd(acc, -1, ddof=1), fmt="o-",
                             ms=3, color=col, alpha=1 - 0.25 * i, label=lab)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("training rows")
    axes[1].set_ylabel("test accuracy (LR)")
    axes[1].legend(fontsize=5.5)
    save_figure(fig, "fig9_shots_and_learning", FIG)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    warnings.simplefilter("ignore", RuntimeWarning)
    set_thesis_style()
    R = load_all()
    print(f"[phase7] loaded: {len(R['ed'])} ED, {len(R['vqe'])} VQE, {len(R['noise'])} noise, {len(R['zne'])} ZNE, "
          f"ff {'yes' if R['ff'] is not None else 'no'}, {len(R['shots'])} shots, {len(R['learning'])} learning")
    checks_clean(R)
    checks_noise(R)
    checks_zne(R)
    checks_shots_rerun(R)
    result_notes(R)
    tables(R)
    fig_output_curves(R)
    fig_hc_vs_N(R)
    fig_pca(R)
    fig_noise(R, "hc", "fig4_hc_vs_noise", r"$h_c^{\mathrm{ML}}$ (shadows)")
    fig_noise(R, "acc", "fig5_accuracy_vs_noise", "test accuracy (shadows)")
    fig_curves_under_noise(R)
    fig_variants_sizes(R)
    fig_zne(R)
    fig_shots_learning(R)
    n_pass = sum(ok for _, ok, _ in checks)
    n_fail = len(checks) - n_pass
    lines = [f"Phase 7 summary: {n_pass} checks passed, {n_fail} failed, {len(notes)} notes", ""]
    lines += [f"[{'PASS' if ok else 'FAIL'}] {n}: {m}" for n, ok, m in checks] + [""]
    lines += [f"[note] {n}: {m}" for n, m in notes]
    path = RESULTS_DIR / SUB / "phase7_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[phase7] {n_pass} checks passed, {n_fail} failed, {len(notes)} notes; summary in {path}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
