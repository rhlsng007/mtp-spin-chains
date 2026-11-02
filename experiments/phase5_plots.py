"""
Phase 5 figures, validation checks and summary (reads saved data only, plus a
small live re-simulation of saved Phase 4 points as a cross-phase check).

Figures (figures/phase5/):
    mitigation_comparison        deliverable: before/after mitigation (exact limit and finite shots)
    zne_extrapolation_curves     E versus scale factor with the three extrapolations
    zne_extrapolators_by_channel combined mitigation per channel and extrapolator
    mitigation_error_budget      gate error and ZNE residual against the error budget lambda
    mitigation_vs_field          energies along the field grid at one realistic strength
    readout_mitigation           confusion matrix and calibration-shot convergence
Summary: results/phase5/phase5_summary.txt

Error measure: delta = |E - E_ideal| / |E0|, with E_ideal the energy of the
noiseless circuit (the best mitigation can do) and E0 the exact ground energy.

Run from the project root:  python experiments\\phase5_plots.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import CFG, RESULTS_DIR
from noise.channels import make_noise_model
from noise.data import load_phase2
from noise.mitigation import EXTRAPOLATORS, error_budget, extrapolate, extrapolation_coefficients, simulate_folded
from noise.shots import EnergyEstimator
from models.hamiltonians import hamiltonian_terms
from utils.io import load_results, result_exists
from utils.plotting import COLORS, SERIES_COLORS, save_figure, set_thesis_style

SUBDIR = "phase5"
FIGDIR = "phase5"
MCFG = CFG.mitigation
METHOD_STYLE = {
    "unmitigated": dict(color=SERIES_COLORS["noisy"], label="unmitigated", marker="o"),
    "readout": dict(color=COLORS["orange"], label="readout corrected", marker="s"),
    "zne": dict(color=COLORS["blue"], label="ZNE", marker="^"),
    "combined": dict(color=SERIES_COLORS["mitigated"], label="readout + ZNE", marker="D"),
}
EXTRAP_STYLE = {"linear": dict(color=COLORS["sky"], ls="--"),
                "richardson": dict(color=COLORS["green"], ls="-"),
                "exponential": dict(color=COLORS["purple"], ls="-.")}
CHANNEL_LABEL = {"realistic": "realistic", "depolarizing": "depolarizing",
                 "amplitude_damping": "amplitude damping", "dephasing": "dephasing"}


# ---------------------------------------------------------------------------
# Loading and derived estimates
# ---------------------------------------------------------------------------
def _arrays(name):
    out = load_results(name, SUBDIR)
    return out[0] if isinstance(out, tuple) else out


def available_zne():
    pat = re.compile(r"zne_(hva|hea)_N(\d+)_p(\d+)_(\w+)\.npz$")
    keys = []
    for f in sorted((RESULTS_DIR / SUBDIR).glob("zne_*.npz")):
        m = pat.match(f.name)
        if m:
            keys.append((m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)))
    return keys


def zne_data(a, N, p, c):
    return _arrays(f"zne_{a}_N{N}_p{p}_{c}")


def fold_index(d, folding=None):
    return list(np.asarray(d["foldings"]).astype(str)).index(folding or MCFG.main_folding)


def estimates(d, folding=None):
    """Exact-limit (infinite shots, exact calibration) energies, each [S, F]."""
    sc, l = d["scales"], fold_index(d, folding)
    est = {"gate_only": d["E_true"][:, :, l, 0], "unmitigated": d["E_raw"][:, :, l, 0],
           "readout": d["E_ro"][:, :, l, 0]}
    for ex in EXTRAPOLATORS:
        est[f"zne_{ex}"] = extrapolate(sc, d["E_raw"][:, :, l, :], ex, asymptote=d["asym_raw"][None, :])
        est[f"combined_{ex}"] = extrapolate(sc, d["E_ro"][:, :, l, :], ex, asymptote=d["asym_ro"])
        est[f"idealcal_{ex}"] = extrapolate(sc, d["E_true"][:, :, l, :], ex, asymptote=d["asym_true"][None, :])
    return est


def deltas(d, folding=None):
    E_id, norm = d["E_ideal"][None, :], np.abs(d["E0"])[None, :]
    return {k: np.abs(v - E_id) / norm for k, v in estimates(d, folding).items()}


def shot_estimates(d, ex=None):
    """Finite-shot energies, each [S, F, R] (main folding)."""
    ex = ex or MCFG.main_extrapolator
    sc = d["scales"]
    return {"unmitigated": d["Es_raw"][:, :, 0, :], "readout": d["Es_ro"][:, :, 0, :],
            "zne": extrapolate(sc, d["Es_raw"], ex, axis=2, asymptote=d["asym_raw"][None, :, None]),
            "combined": extrapolate(sc, d["Es_ro"], ex, axis=2, asymptote=d["asym_ro"][:, :, None]),
            "combined_tensored": extrapolate(sc, d["Es_tens"], ex, axis=2, asymptote=d["asym_ro"][:, :, None])}


def shot_rms(d, ex=None):
    E_id, norm = d["E_ideal"][None, :, None], np.abs(d["E0"])[None, :]
    return {k: np.sqrt(np.nanmean((v - E_id) ** 2, axis=-1)) / norm for k, v in shot_estimates(d, ex).items()}


def budget(d, c):
    return error_budget(c, d["strengths"], int(d["n_1q"]), int(d["n_2q"]))


def field_index(d, h):
    return int(np.argmin(np.abs(d["h"] - h)))


def strength_index(d, s):
    return int(np.argmin(np.abs(np.log(np.maximum(d["strengths"], 1e-300)) - np.log(s))))


def loglog_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------
class Checks:
    def __init__(self):
        self.lines, self.n_pass, self.n_fail, self.n_note = [], 0, 0, 0

    def check(self, ok, text):
        self.lines.append(f"[{'PASS' if ok else 'FAIL'}] {text}")
        self.n_pass += bool(ok)
        self.n_fail += not ok
        print(self.lines[-1])

    def note(self, text):
        self.lines.append(f"[NOTE] {text}")
        self.n_note += 1
        print(self.lines[-1])


def run_checks(keys, ch: Checks, summary: list):
    tol = MCFG.exact_tol
    files = {k: zne_data(*k) for k in keys}

    # C1 zero noise reproduces the noiseless circuit and Phase 2
    e1 = max(np.max(np.abs(d["E_true"][0] - d["E_ideal"][:, None, None])) for d in files.values())
    e2 = max(np.max(np.abs(d["E_ideal"] - d["phase2_energy"])) for d in files.values())
    ch.check(e1 < tol, f"strength 0: every folded circuit gives the noiseless energy (max dev {e1:.1e})")
    ch.check(e2 < 1e-9, f"noiseless circuit energies reproduce Phase 2 (max dev {e2:.1e})")

    # C2 readout correction is exact at strength 0 (exact calibration, noiseless preparation)
    e3 = max(np.max(np.abs(d["E_ro"][0] - d["E_ideal"][:, None, None])) for d in files.values())
    ch.check(e3 < tol, f"strength 0: readout-corrected energies equal the noiseless energy (max dev {e3:.1e})")
    ro_bias = np.array([np.max(np.abs(d["E_raw"][0] - d["E_ideal"][:, None, None]) / np.abs(d["E0"])[:, None, None])
                        for d in files.values()])
    ch.note(f"readout error alone (strength 0): relative energy bias up to {ro_bias.max():.3f}")

    # C3 strength 0 is identical across channels
    groups = {}
    for (a, N, p, c), d in files.items():
        groups.setdefault((a, N, p), []).append(d)
    e4 = max(max(np.max(np.abs(x["E_raw"][0] - g[0]["E_raw"][0])) for x in g) for g in groups.values())
    ch.check(e4 == 0.0, f"strength 0 identical for all channels (max dev {e4:.1e})")

    # C4 variational principle for the physical (unmitigated) states
    worst = min(np.min(d["E_true"] - d["E0"][None, :, None, None]) for d in files.values())
    ch.check(worst > -tol, f"variational principle Tr(rho H) >= E0 at every scale factor (min {worst:.1e})")

    # C5 folding scales the first-order noise exactly for depolarizing noise
    for c in MCFG.channels:
        ratios = {3: [], 5: []}
        for (a, N, p, cc), d in files.items():
            if cc != c:
                continue
            for li in range(len(d["foldings"])):
                dE = d["E_true"][1, :, li, :] - d["E_ideal"][:, None]
                ok = np.abs(dE[:, 0]) > 1e-12 * np.abs(d["E0"])
                for k, lam in enumerate(d["scales"]):
                    if lam in ratios:
                        ratios[lam].extend(list(dE[ok, k] / dE[ok, 0] / lam))
        dev = max(np.max(np.abs(np.array(v) - 1)) for v in ratios.values())
        med = {lam: np.median(v) for lam, v in ratios.items()}
        text = (f"weakest noise ({c}): dE(lambda)/dE(1)/lambda = {med[3]:.4f} (3), {med[5]:.4f} (5), "
                f"max |dev| {dev:.2e}")
        if c == "depolarizing":
            ch.check(dev < MCFG.weak_fold_tol, text + " [first order exact for depolarizing]")
        else:
            ch.note(text)

    # C6 order of the extrapolation residual at weak noise (ideal readout). Folding scales the
    # first-order error by exactly lambda only for noise that commutes with the gates
    # (depolarizing); otherwise E(lambda) - E_ideal = s[(A - B)/2 + lambda (A + B)/2] + O(s^2)
    # and every extrapolator keeps the first-order folding bias s (A - B)/2.
    for c in MCFG.channels:
        sl = {"gate_only": [], "idealcal_linear": [], "idealcal_richardson": [], "idealcal_exponential": []}
        bias = []
        for (a, N, p, cc), d in files.items():
            if cc != c:
                continue
            dl = deltas(d)
            s = d["strengths"][1:4]
            for k in sl:
                for f in range(len(d["h"])):
                    y = dl[k][1:4, f]
                    if np.all(y > 1e-13):
                        sl[k].append(loglog_slope(s, y))
            ok = dl["gate_only"][1] > 1e-13
            bias.extend(list(dl["idealcal_richardson"][1][ok] / dl["gate_only"][1][ok]))
        if not sl["gate_only"]:
            continue
        med = {k: float(np.nanmedian(v)) for k, v in sl.items()}
        text = (f"weak-noise order ({c}): median log-log slopes unmitigated {med['gate_only']:.3f}, "
                f"linear {med['idealcal_linear']:.3f}, Richardson {med['idealcal_richardson']:.3f}, "
                f"exponential {med['idealcal_exponential']:.3f}")
        if c == "depolarizing":
            # Folding scales the first-order error exactly here, so Richardson must cancel it at
            # every point. Its second-order coefficient is not exactly proportional to lambda^2
            # (two errors in different folded copies), so a small s^2 term survives; where it
            # crosses the s^3 term the residual changes sign and single slopes scatter.
            ok = all(abs(med[k] - n) < MCFG.order_tol for k, n in
                     (("gate_only", 1), ("idealcal_linear", 2), ("idealcal_richardson", 3)))
            frac = np.mean(np.abs(np.array(sl["idealcal_richardson"]) - 3) < 2 * MCFG.order_tol)
            ch.check(ok, text + f" (expected 1, 2, 3; {100 * frac:.1f} % of single Richardson slopes "
                                f"within +-{2 * MCFG.order_tol})")
            ch.check(np.max(bias) < 1e-3, f"depolarizing: first-order error cancelled by Richardson at every "
                                          f"point (residual / unmitigated at p = {d['strengths'][1]:.0e}: "
                                          f"max {np.max(bias):.1e})")
        else:
            q = np.percentile(bias, [5, 50, 95])
            ch.note(text + f"; first-order folding bias: Richardson residual / unmitigated error at "
                           f"p = {d['strengths'][1]:.0e} median {q[1]:.3f} (5-95 %: {q[0]:.3f}-{q[2]:.3f})")

    # C7 mitigation helps everywhere inside the weak-noise regime
    bad, bad_ro, n = 0, 0, 0
    for (a, N, p, c), d in files.items():
        dl = deltas(d)
        lam = budget(d, c)
        reg = (lam > 0) & (lam <= MCFG.regime_lambda)
        best = dl[f"combined_{MCFG.main_extrapolator}"][reg]
        bad += np.sum(best >= dl["unmitigated"][reg])
        bad_ro += np.sum(best >= dl["readout"][reg])
        n += best.size
    ch.check(bad == 0, f"error budget lambda <= {MCFG.regime_lambda}: readout + ZNE beats unmitigated at "
                       f"{n - bad} of {n} points")
    ch.check(bad_ro == 0, f"error budget lambda <= {MCFG.regime_lambda}: readout + ZNE beats readout correction "
                          f"alone at {n - bad_ro} of {n} points")

    # C8 finite-shot estimates are unbiased; C9 variance amplification as predicted
    out3, tot, ratios0, ratiosR = 0, 0, [], []
    out3_r = 0
    c_rich = extrapolation_coefficients(files[keys[0]]["scales"], "richardson")
    for d in files.values():
        R = d["Es_raw"].shape[-1]
        l = fold_index(d)
        mean = d["Es_raw"].mean(-1)
        exact = d["E_raw"][:, :, l, :]
        sig = np.sqrt(d["var_raw"] / R)
        out3 += np.sum(np.abs(mean - exact) > 3 * sig)
        tot += mean.size
        rich = d["Es_raw"].transpose(0, 1, 3, 2) @ c_rich
        rich_exact = exact @ c_rich
        var_pred = d["var_raw"] @ c_rich ** 2
        out3_r += np.sum(np.abs(rich.mean(-1) - rich_exact) > 3 * np.sqrt(var_pred / R))
        ratios0.extend(list((d["Es_raw"][:, :, 0, :].var(-1, ddof=1) / d["var_raw"][:, :, 0]).ravel()))
        ratiosR.extend(list((rich.var(-1, ddof=1) / var_pred).ravel()))
    ch.check(out3 / tot < 0.01, f"unmitigated shot estimates unbiased: {100 * out3 / tot:.2f} % beyond 3 sigma "
                                f"(0.27 % expected)")
    ch.check(out3_r / (tot / 3) < 0.01, f"Richardson shot estimates unbiased: {100 * out3_r / (tot / 3):.2f} % "
                                        f"beyond 3 sigma")
    m0, mR = np.median(ratios0), np.median(ratiosR)
    ch.check(abs(m0 - 1) < MCFG.variance_ratio_tol and abs(mR - 1) < MCFG.variance_ratio_tol,
             f"shot variance matches theory: median empirical/predicted {m0:.3f} (unmitigated), "
             f"{mR:.3f} (Richardson, predicted sum_k c_k^2 Var_k)")
    # chi-square median of a sample variance with R - 1 dof is slightly below 1
    gam = float(np.sum(c_rich ** 2))
    summary.append(f"Richardson variance amplification sum c_k^2 = {gam:.4f} (std factor {np.sqrt(gam):.3f})")

    # C10 readout experiment
    for a in CFG.vqe.ansatze:
        for N in CFG.system.sizes_noisy:
            name = f"readout_{a}_N{N}"
            if not result_exists(name, SUBDIR):
                continue
            r = _arrays(name)
            e = max(abs(r["E_full_exact"] - r["E_ideal"]), abs(r["E_tens_exact"] - r["E_ideal"]))
            ch.check(e < tol and np.max(r["tvd_full_exact"]) < 1e-12,
                     f"readout {a} N={N}: exact full and tensored correction recover the ideal distribution "
                     f"and energy (energy dev {e:.1e})")
            meth = list(np.asarray(r["methods"]).astype(str))
            slopes = []
            for m in ("full_inverse", "tensored_inverse"):
                y = np.median(r["tvd"][0, :, meth.index(m)].mean(-1), axis=-1)
                slopes.append(loglog_slope(r["cal_shots"], y))
            ch.check(all(abs(s + 0.5) < MCFG.order_tol for s in slopes),
                     f"readout {a} N={N}: TVD after correction ~ M_cal^(-1/2) (slopes {slopes[0]:.3f} full, "
                     f"{slopes[1]:.3f} tensored)")

    # C11 cross-phase check against saved Phase 4 energies
    dev, n_pts = 0.0, 0
    for a in CFG.vqe.ansatze:
        for N in CFG.system.sizes_noisy:
            for p in CFG.system.depths(N, noisy=True):
                name = f"noisy_eval_{a}_N{N}_p{p}"
                if not result_exists(name, "phase4"):
                    continue
                e4 = load_results(name, "phase4")[0]
                h4, s4 = np.asarray(e4["h"]), np.asarray(e4["strengths"])
                chans = list(np.asarray(e4["channels"]).astype(str))
                fi = int(np.argmin(np.abs(h4 - CFG.tfim.h_critical)))
                si = len(s4) // 2
                theta = load_phase2(a, N, p, h_sel=[h4[fi]])["params"][0]
                est = EnergyEstimator(hamiltonian_terms("tfim", N, J=CFG.tfim.J, h=float(h4[fi]),
                                                        boundary=CFG.tfim.boundary), N)
                for ci, c in enumerate(chans):
                    rho = simulate_folded(a, N, p, theta, make_noise_model(c, s4[si]), 1)
                    dev = max(dev, abs(est.exact(rho) - e4["energy"][ci, si, fi]))
                    n_pts += 1
    if n_pts:
        ch.check(dev < 1e-10, f"scale-1 simulator reproduces {n_pts} saved Phase 4 energies (max dev {dev:.1e})")
    else:
        ch.note("Phase 4 files not found: cross-phase check skipped")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _budget_axis(ax, d, c):
    k = float(budget(d, c)[-1] / d["strengths"][-1])
    sec = ax.secondary_xaxis("top", functions=(lambda s: s * k, lambda l: l / k))
    sec.set_xlabel(r"error budget $\lambda$")


def fig_comparison(N, p, h, c="realistic"):
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 6.0), sharex=True)
    ex = MCFG.main_extrapolator
    for row, a in enumerate(CFG.vqe.ansatze):
        d = zne_data(a, N, p, c)
        f = field_index(d, h)
        s = d["strengths"][1:]
        dl, rms = deltas(d), shot_rms(d)
        vqe_err = abs(d["E_ideal"][f] - d["E0"][f]) / abs(d["E0"][f])
        for col, data in enumerate((dl, rms)):
            ax = axes[row, col]
            keys = {"unmitigated": "unmitigated", "readout": "readout",
                    "zne": f"zne_{ex}" if col == 0 else "zne",
                    "combined": f"combined_{ex}" if col == 0 else "combined"}
            for m, k in keys.items():
                ax.loglog(s, data[k][1:, f], ms=3.5, **METHOD_STYLE[m])
            if col == 0:
                ax.loglog(s, dl[f"idealcal_{ex}"][1:, f], color=SERIES_COLORS["mitigated"], ls=":", lw=1.4,
                          label="readout + ZNE, ideal calibration")
            else:
                sig = np.sqrt(d["var_raw"][1:, f, 0]) / abs(d["E0"][f])
                ax.loglog(s, sig, color="0.5", ls="--", lw=1.0, label="shot noise, one run")
            ax.axhline(vqe_err, color=SERIES_COLORS["ED"], ls=":", lw=1.0, label="noiseless VQE error")
            what = "exact limit" if col == 0 else f"{int(d['shots'])} shots per basis"
            ax.set_title(f"{a.upper()}, {what}", fontsize=10)
            if row == 1:
                ax.set_xlabel(r"noise strength $p$")
            if col == 0:
                ax.set_ylabel(r"$|E - E_{\rm ideal}| / |E_0|$")
            if row == 0:
                _budget_axis(ax, d, c)
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[0, 1].get_legend_handles_labels()
    extra = [(hh, ll) for hh, ll in zip(h2, l2) if ll not in l1]
    fig.legend(h1 + [e[0] for e in extra], l1 + [e[1] for e in extra], loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Mitigation, {CHANNEL_LABEL[c]} noise, N = {N}, p = {p}, h = {h:g}", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "mitigation_comparison", FIGDIR)
    plt.close(fig)


def fig_curves(N, p, h, a="hva", c="realistic"):
    d = zne_data(a, N, p, c)
    f = field_index(d, h)
    sc = d["scales"]
    l = fold_index(d)
    fig, axes = plt.subplots(1, len(MCFG.plot_strengths_curves), figsize=(6.3, 2.6), sharey=False)
    lam = np.linspace(0, sc.max() + 0.5, 200)
    for ax, s_t in zip(axes, MCFG.plot_strengths_curves):
        si = strength_index(d, s_t)
        y = d["E_ro"][si, f, l, :]
        ax.plot(sc, y, "o", color=SERIES_COLORS["noisy"], label="readout corrected", zorder=5)
        lin = np.polyfit(sc, y, 1)
        ax.plot(lam, np.polyval(lin, lam), label="linear", **EXTRAP_STYLE["linear"])
        quad = np.polyfit(sc, y, len(sc) - 1)
        ax.plot(lam, np.polyval(quad, lam), label="Richardson", **EXTRAP_STYLE["richardson"])
        asym = d["asym_ro"][si, f]
        dd = y - asym
        if np.all(dd * np.sign(dd[0]) > 0):
            b = np.polyfit(sc, np.log(np.abs(dd)), 1)
            ax.plot(lam, asym + np.sign(dd[0]) * np.exp(np.polyval(b, lam)), label="exponential",
                    **EXTRAP_STYLE["exponential"])
        ax.plot(0, d["E_ideal"][f], "*", color=SERIES_COLORS["ED"], ms=10, label="noiseless circuit", zorder=6)
        ax.axhline(d["E0"][f], color="0.5", lw=0.8, ls=":")
        ax.set_title(f"$p$ = {d['strengths'][si]:.1e}", fontsize=10)
        ax.set_xlabel(r"scale factor $\lambda$")
    axes[0].set_ylabel(r"$E$")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{a.upper()}, N = {N}, p = {p}, h = {h:g}, {CHANNEL_LABEL[c]}", fontsize=10)
    fig.tight_layout()
    save_figure(fig, "zne_extrapolation_curves", FIGDIR)
    plt.close(fig)


def fig_by_channel(N, p, h):
    chans = [c for c in MCFG.channels]
    fig, axes = plt.subplots(2, len(chans), figsize=(8.5, 4.6), sharex=True, sharey=True)
    for row, a in enumerate(CFG.vqe.ansatze):
        for col, c in enumerate(chans):
            ax = axes[row, col]
            d = zne_data(a, N, p, c)
            f = field_index(d, h)
            dl = deltas(d)
            s = d["strengths"][1:]
            ax.loglog(s, dl["readout"][1:, f], color=METHOD_STYLE["readout"]["color"], label="readout only")
            for ex in EXTRAPOLATORS:
                ax.loglog(s, dl[f"combined_{ex}"][1:, f], label=f"readout + {ex}", **EXTRAP_STYLE[ex])
            ax.loglog(s, dl["idealcal_richardson"][1:, f], color=EXTRAP_STYLE["richardson"]["color"], ls=":",
                      lw=1.3, label="Richardson, ideal readout (no SPAM)")
            if row == 0:
                ax.set_title(CHANNEL_LABEL[c], fontsize=10)
            if row == 1:
                ax.set_xlabel(r"$p$")
            if col == 0:
                ax.set_ylabel(f"{a.upper()}\n" + r"$|E - E_{\rm ideal}| / |E_0|$")
    hd, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"N = {N}, p = {p}, h = {h:g}", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, "zne_extrapolators_by_channel", FIGDIR)
    plt.close(fig)


def fig_budget(keys, h, channels=("depolarizing", "realistic")):
    """
    Residual against the error budget lambda (scale 1), ideal readout, all (ansatz, N, p).
    Rows: depolarizing (folding scales the noise exactly: Richardson ~ lambda^3,
    exponential ~ lambda^2) and realistic (first-order folding bias: ~ lambda).
    """
    ex_list = ("gate_only", "idealcal_richardson", "idealcal_exponential")
    titles = ("no mitigation (gate noise only)", "Richardson ZNE", "exponential ZNE")
    guides = {"depolarizing": (1, 3, 2), "realistic": (1, 1, 1)}
    colors = {4: COLORS["sky"], 6: COLORS["blue"], 8: COLORS["purple"]}
    markers = {"hva": "o", "hea": "^"}
    channels = [c for c in channels if any(k[3] == c for k in keys)]
    fig, axes = plt.subplots(len(channels), 3, figsize=(8.5, 2.7 * len(channels)), sharey=True,
                             squeeze=False)
    for row, c in enumerate(channels):
        pts = {k: ([], []) for k in ex_list}
        for (a, N, p, cc) in keys:
            if cc != c:
                continue
            d = zne_data(a, N, p, c)
            f = field_index(d, h)
            lam = budget(d, c)[1:]
            dl = deltas(d)
            for ax, k in zip(axes[row], ex_list):
                ax.loglog(lam, dl[k][1:, f], ls="none", marker=markers[a], ms=3, color=colors.get(N, "k"),
                          alpha=0.7)
                pts[k][0].extend(lam)
                pts[k][1].extend(dl[k][1:, f])
        for col, (ax, k) in enumerate(zip(axes[row], ex_list)):
            x, y = np.array(pts[k][0]), np.array(pts[k][1])
            ok = (x > 0) & (y > 0)
            n = guides[c][col]
            if ok.any():
                x0 = x[ok].min()
                y0 = np.median(y[ok & (x <= 1.5 * x0)])
                xx = np.logspace(np.log10(x0), 0, 20)
                ax.loglog(xx, y0 * (xx / x0) ** n, "k--", lw=0.8, label=rf"$\propto\lambda^{n}$")
            ax.axvline(1.0 / max(CFG.noise.zne_scale_factors), color="0.6", lw=0.8, ls=":")
            ax.legend(fontsize=7, loc="upper left")
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
            if row == len(channels) - 1:
                ax.set_xlabel(r"error budget $\lambda$ (scale 1)")
        axes[row, 0].set_ylabel(f"{CHANNEL_LABEL[c]}\n" + r"$|E - E_{\rm ideal}| / |E_0|$")
    handles = [plt.Line2D([], [], ls="none", marker="s", color=col, label=f"N = {N}") for N, col in colors.items()]
    handles += [plt.Line2D([], [], ls="none", marker=m, color="0.3", label=a.upper()) for a, m in markers.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"All depths, h = {h:g}, ideal readout; dotted line: $\\lambda$ = 1/max scale factor",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "mitigation_error_budget", FIGDIR)
    plt.close(fig)


def fig_vs_field(N, p, s_target, c="realistic"):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
    for ax, a in zip(axes, CFG.vqe.ansatze):
        d = zne_data(a, N, p, c)
        si = strength_index(d, s_target)
        h, E0 = d["h"], d["E0"]
        rel = lambda E: (E - E0) / np.abs(E0)
        ex = MCFG.main_extrapolator
        est = estimates(d)
        sh = shot_estimates(d)
        ax.plot(h, rel(d["E_ideal"]), color=SERIES_COLORS["ED"], ls=":", label="noiseless VQE")
        for m, k in (("unmitigated", "unmitigated"), ("readout", "readout"), ("zne", f"zne_{ex}"),
                     ("combined", f"combined_{ex}")):
            st = METHOD_STYLE[m]
            ax.plot(h, rel(est[k][si]), color=st["color"], marker=st["marker"], ms=3, label=st["label"])
        comb = sh["combined"][si]
        ax.errorbar(h, rel(comb.mean(-1)), yerr=comb.std(-1, ddof=1) / np.abs(E0), fmt="none",
                    ecolor=SERIES_COLORS["mitigated"], elinewidth=0.8, capsize=1.5)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.axvline(CFG.tfim.h_critical, color="0.7", lw=0.6, ls="--")
        ax.set_title(f"{a.upper()}", fontsize=10)
        ax.set_xlabel(r"$h$")
        ax.set_yscale("symlog", linthresh=1e-3)
    axes[0].set_ylabel(r"$(E - E_0) / |E_0|$")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"N = {N}, p = {p}, {CHANNEL_LABEL[c]} noise $p$ = {s_target:.0e} "
                 f"(error bars: {MCFG.shots} shots)", fontsize=10)
    fig.tight_layout()
    save_figure(fig, "mitigation_vs_field", FIGDIR)
    plt.close(fig)


def fig_readout(N_mat=4, N_conv=8, a="hva"):
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8))
    r4 = _arrays(f"readout_{a}_N{N_mat}")
    im = axes[0].imshow(np.log10(np.maximum(r4["A_true"], 1e-8)), cmap="viridis", vmin=-6, vmax=0)
    axes[0].set_title(f"confusion matrix, N = {N_mat}", fontsize=10)
    axes[0].set_xlabel("prepared")
    axes[0].set_ylabel("measured")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], fraction=0.046, label=r"$\log_{10} A_{ij}$")
    r = _arrays(f"readout_{a}_N{N_conv}")
    meth = list(np.asarray(r["methods"]).astype(str))
    M = r["cal_shots"]
    style = {"full_inverse": (COLORS["blue"], "-", "full, inverse"),
             "full_projected": (COLORS["blue"], "--", "full, projected"),
             "tensored_inverse": (COLORS["orange"], "-", "tensored, inverse"),
             "tensored_projected": (COLORS["orange"], "--", "tensored, projected")}
    for m, (col, ls, lab) in style.items():
        y = r["tvd"][0, :, meth.index(m)].mean(-1)            # mean over groups -> [cal, R]
        axes[1].loglog(M, np.median(y, -1), color=col, ls=ls, marker="o", ms=3, label=lab)
        e = np.sqrt(np.mean(r["dE"][1, :, meth.index(m)] ** 2, -1))
        axes[2].loglog(M, e, color=col, ls=ls, marker="o", ms=3, label=lab)
    axes[1].axhline(np.mean(r["tvd_raw_exact"]), color=SERIES_COLORS["noisy"], lw=1, label="uncorrected")
    axes[1].loglog(M, 0.5 * np.median(r["tvd"][0, 0, meth.index("full_inverse")].mean(-1)) * np.sqrt(M[0] / M),
                   "k:", lw=0.8, label=r"$\propto M_{\rm cal}^{-1/2}$")
    raw = np.sqrt(np.mean(r["dE"][1, 0, meth.index("raw")] ** 2))
    axes[2].axhline(raw, color=SERIES_COLORS["noisy"], lw=1, label="uncorrected")
    axes[1].set_title(f"distribution error, N = {N_conv}", fontsize=10)
    axes[2].set_title(f"energy error, {int(r['shots'])} shots", fontsize=10)
    for ax in axes[1:]:
        ax.set_xlabel(r"calibration shots $M_{\rm cal}$")
    axes[1].set_ylabel("total variation distance")
    axes[2].set_ylabel(r"RMS $|E - E_{\rm ideal}|/|E_0|$")
    axes[1].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    save_figure(fig, "readout_mitigation", FIGDIR)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary numbers
# ---------------------------------------------------------------------------
def summary_numbers(keys, summary):
    N, p, h = MCFG.plot_size, MCFG.plot_size // 2, MCFG.plot_field
    ex = MCFG.main_extrapolator
    for c in MCFG.channels:
        for a in CFG.vqe.ansatze:
            if (a, N, p, c) not in keys:
                continue
            d = zne_data(a, N, p, c)
            f = field_index(d, h)
            dl, rms = deltas(d), shot_rms(d)
            summary.append(f"\n{a.upper()} N={N} p={p} h={h:g}, {c} (noiseless VQE error "
                           f"{abs(d['E_ideal'][f] - d['E0'][f]) / abs(d['E0'][f]):.2e}; n_1q={int(d['n_1q'])}, "
                           f"n_2q={int(d['n_2q'])})")
            summary.append("  p          lambda   unmit     readout   ZNE-R     comb-lin  comb-R    comb-exp  "
                           "idealcal-R | shots: unmit  readout   comb-R    comb-R(tens)")
            lam = budget(d, c)
            for si, s in enumerate(d["strengths"]):
                summary.append(
                    f"  {s:9.2e}  {lam[si]:7.3f}  {dl['unmitigated'][si, f]:.2e}  {dl['readout'][si, f]:.2e}  "
                    f"{dl[f'zne_{ex}'][si, f]:.2e}  {dl['combined_linear'][si, f]:.2e}  "
                    f"{dl['combined_richardson'][si, f]:.2e}  {dl['combined_exponential'][si, f]:.2e}  "
                    f"{dl['idealcal_richardson'][si, f]:.2e}  | {rms['unmitigated'][si, f]:.2e}  "
                    f"{rms['readout'][si, f]:.2e}  {rms['combined'][si, f]:.2e}  {rms['combined_tensored'][si, f]:.2e}")
    # folding comparison and variational violations
    ratios, below, n = [], 0, 0
    for k in keys:
        d = zne_data(*k)
        if "local" not in list(np.asarray(d["foldings"]).astype(str)):
            continue
        g = deltas(d, "global")[f"idealcal_{ex}"][1:]
        lo = deltas(d, "local")[f"idealcal_{ex}"][1:]
        ok = (g > 1e-13) & (lo > 1e-13)
        ratios.extend(list((lo[ok] / g[ok]).ravel()))
        comb = estimates(d)[f"combined_{ex}"][1:]
        below += np.sum(comb < d["E0"][None, :] - 1e-12)
        n += comb.size
    if ratios:
        q = np.percentile(ratios, [5, 50, 95])
        summary.append(f"\nlocal / global folding residual ({ex}, ideal readout): median {q[1]:.2f} "
                       f"(5-95 %: {q[0]:.2f}-{q[2]:.2f})")
    summary.append(f"combined ({ex}) energies below E0 (variational principle violated by mitigation): "
                   f"{below} of {n}")


def main():
    set_thesis_style()
    keys = available_zne()
    if not keys:
        print("no Phase 5 results found; run experiments\\phase5_mitigation.py first")
        return
    ch, summary = Checks(), []
    run_checks(keys, ch, summary)
    N, p, h = MCFG.plot_size, MCFG.plot_size // 2, MCFG.plot_field
    have = lambda a, c: (a, N, p, c) in keys
    if all(have(a, "realistic") for a in CFG.vqe.ansatze):
        fig_comparison(N, p, h)
        fig_curves(N, p, h)
        fig_vs_field(N, p, MCFG.plot_strength)
        fig_budget(keys, h)
    if all(have(a, c) for a in CFG.vqe.ansatze for c in MCFG.channels):
        fig_by_channel(N, p, h)
    if result_exists("readout_hva_N4", SUBDIR) and result_exists(f"readout_hva_N{N}", SUBDIR):
        fig_readout(4, N)
    summary_numbers(keys, summary)
    head = [f"Phase 5 summary: {ch.n_pass} checks passed, {ch.n_fail} failed, {ch.n_note} notes", ""]
    text = "\n".join(head + ch.lines + [""] + summary) + "\n"
    out = RESULTS_DIR / SUBDIR / "phase5_summary.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\n{ch.n_pass} checks passed, {ch.n_fail} failed, {ch.n_note} notes; summary in {out}")


if __name__ == "__main__":
    main()
