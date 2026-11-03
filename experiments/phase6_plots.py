"""
Phase 6: validation checks, figures and summary for the ML datasets.

Run from the project root after experiments\\phase6_datasets.py:

    python experiments\\phase6_plots.py

Writes figures/phase6/*.pdf|png and results/phase6/phase6_summary.txt.
Checks against exact results (free fermions, ED, Phase 1, 2 and 5 data) and
exact statistical identities of classical shadows; missing optional inputs
(Phase 1 or Phase 5 files) are reported as notes, not failures.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
import numpy as np                                                      # noqa: E402
from scipy import stats                                                 # noqa: E402

from config import CFG, RESULTS_DIR                                     # noqa: E402
from analysis.ff_correlators import ff_observables                      # noqa: E402
from ml import dataset as D                                             # noqa: E402
from ml import features as F                                            # noqa: E402
from ml import shadows as S                                             # noqa: E402
from noise.data import field_indices                                    # noqa: E402
from utils.io import load_results, result_exists                       # noqa: E402
from utils.plotting import (COLORS, PALETTE, SERIES_COLORS, SINGLE_COLUMN,  # noqa: E402
                            save_figure, set_thesis_style)

FIG = "phase6"
OUT = RESULTS_DIR / "phase6"


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.lines, self.n_pass, self.n_fail, self.n_note = [], 0, 0, 0

    def check(self, cid: str, text: str, ok: bool, detail: str = "") -> None:
        tag = "PASS" if ok else "FAIL"
        self.n_pass += ok
        self.n_fail += (not ok)
        line = f"[{tag}] {cid} {text}" + (f": {detail}" if detail else "")
        self.lines.append(line)
        print(line)

    def note(self, text: str) -> None:
        self.n_note += 1
        line = f"[NOTE] {text}"
        self.lines.append(line)
        print(line)

    def section(self, title: str) -> None:
        self.lines.append("")
        self.lines.append(f"== {title} ==")
        print(f"\n== {title} ==")


def _s(a) -> str:
    return str(np.asarray(a).item()) if np.asarray(a).ndim == 0 else str(a)


def load_all() -> dict:
    """All Phase 6 datasets keyed by source and parameters."""
    data = {"ed": {}, "vqe": {}, "noisy": {}, "zne": {}}
    for src in data:
        for npz in sorted((OUT / src).glob("ds_*.npz")):
            arr, _ = load_results(npz.stem, subdir=f"phase6/{src}")
            N = int(arr["n_qubits"])
            if src == "ed":
                key = N
            elif src == "vqe":
                key = (_s(arr["ansatz"]), N, int(arr["depth"]))
            else:
                key = (_s(arr["ansatz"]), N, int(arr["depth"]), _s(arr["channel"]))
            data[src][key] = arr
    return data


def feature_variance(X: np.ndarray, N: int) -> np.ndarray:
    """Exact single-snapshot variance of each corr feature: 9 - ZZ^2 and 3 - X^2."""
    k = N * (N - 1) // 2
    return np.concatenate([9.0 - X[..., :k] ** 2, 3.0 - X[..., k:] ** 2], axis=-1)


def shadow_stats(X: np.ndarray, Xs: np.ndarray, N: int, M: int) -> tuple:
    """z-scores of the repetition mean and empirical / analytic variance ratios (flattened)."""
    R = Xs.shape[-2]
    v = feature_variance(X, N) / M
    z = (Xs.mean(axis=-2) - X) / np.sqrt(v / R)
    ratio = Xs.var(axis=-2, ddof=1) / v
    return z.ravel(), ratio.ravel()


def t_stats(X: np.ndarray, Xs: np.ndarray) -> np.ndarray:
    R = Xs.shape[-2]
    sd = Xs.std(axis=-2, ddof=1)
    return ((Xs.mean(axis=-2) - X) / (sd / np.sqrt(R))).ravel()


def m2_of(X, N):
    return F.magnetization_m2(F.unpack_features(X, N)[0])


def train_masks(h):
    train = D.train_field_mask(h)
    return train & (h < CFG.tfim.h_critical), train & (h > CFG.tfim.h_critical)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def run_checks(data: dict, rep: Report) -> dict:
    tol = CFG.ml
    J = CFG.tfim.J
    keep = {}
    every = [(src, key, arr) for src in data for key, arr in data[src].items()]

    rep.section("Inventory")
    total_mb = 0.0
    for src in data:
        files = sorted((OUT / src).glob("ds_*.npz"))
        mb = sum(f.stat().st_size for f in files) / 2 ** 20
        total_mb += mb
        rep.note(f"{src}: {len(files)} files, {mb:.1f} MB: " + ", ".join(f.stem for f in files))
    rep.note(f"total {total_mb:.1f} MB; R = {CFG.ml.shadow_repetitions} shadows of "
             f"M = {CFG.ml.n_shadow_snapshots} snapshots per field")
    if not every:
        rep.check("C0", "datasets present", False, "no files in results/phase6")
        return keep

    # C1 labels and splits ----------------------------------------------------------
    rep.section("Labels and splits (T6.3, T6.4)")
    bad = []
    for src, key, a in every:
        h = np.asarray(a["h"])
        rt, rs = np.asarray(a["rep_train"]), np.asarray(a["rep_test"])
        ok = (np.array_equal(a["y"], D.labels(h)) and np.array_equal(a["train_field"], D.train_field_mask(h))
              and not np.any(rt & rs) and np.all(rt | rs) and np.all(a["test_field"])
              and np.allclose(h, D.field_grid(noisy=src in ("noisy", "zne"))))
        if not ok:
            bad.append(f"{src}{key}")
    h0 = D.field_grid()
    y0 = D.labels(h0)
    rep.check("C1", "labels (0 below h_c, 1 above, -1 at h_c), training fields and repetition split",
              not bad, f"{len(every)} files; {np.sum(D.train_field_mask(h0))} training fields "
              f"({np.sum(D.train_field_mask(h0) & (y0 == 0))} ordered, {np.sum(D.train_field_mask(h0) & (y0 == 1))} "
              f"disordered); bad: {bad if bad else 'none'}")

    # C2-C4 exact features --------------------------------------------------------
    rep.section("Exact features")
    if data["ed"]:
        worst = 0.0
        for N, a in data["ed"].items():
            ff = np.array([F.corr_features(o["zz"], o["x"])
                           for o in (ff_observables(N, J, float(hv)) for hv in a["h"])])
            worst = max(worst, float(np.max(np.abs(a["X"] - ff))))
        rep.check("C2", "ED features against free-fermion correlators (all N, all fields)",
                  worst < tol.ff_tol, f"max |dX| = {worst:.2e} (tol {tol.ff_tol:g})")
        e = max(float(np.max(np.abs(a["energy"] - a["E0"]))) for a in data["ed"].values())
        rep.check("C3", "ED energy rebuilt from features equals the ED ground energy", e < tol.exact_tol,
                  f"max |dE| = {e:.2e}")
    else:
        rep.note("C2, C3 skipped: no ED datasets")
    if data["vqe"]:
        e = max(float(np.max(np.abs(a["energy"] - a["phase2_energy"]))) for a in data["vqe"].values())
        rep.check("C4", "VQE energy rebuilt from features equals the Phase 2 energy", e < tol.exact_tol,
                  f"max |dE| = {e:.2e} over {len(data['vqe'])} files")
    exact_sets = [(a["X"], int(a["n_qubits"])) for a in list(data["ed"].values()) + list(data["vqe"].values())]
    exact_sets += [(a["X_gate"], int(a["n_qubits"])) for a in data["noisy"].values()]
    exact_sets += [(a["X_gate_scales"], int(a["n_qubits"])) for a in data["zne"].values()]
    min_eig, m2_lo, m2_hi = np.inf, np.inf, -np.inf
    for X, N in exact_sets:
        zz, _ = F.unpack_features(X, N)
        min_eig = min(min_eig, float(np.min(np.linalg.eigvalsh(zz))))
        m2 = F.magnetization_m2(zz)
        m2_lo, m2_hi = min(m2_lo, float(m2.min())), max(m2_hi, float(m2.max()))
    rep.check("C5", "exact ZZ matrices positive semidefinite and 0 <= M^2 <= 1 (all states, all noise)",
              min_eig > -tol.exact_tol and m2_lo > -tol.exact_tol and m2_hi < 1 + tol.exact_tol,
              f"min eigenvalue {min_eig:.2e}, M^2 in [{m2_lo:.4f}, {m2_hi:.4f}]")
    worst, n_cmp = 0.0, 0
    for N, a in data["ed"].items():
        if not result_exists(f"ed_tfim_N{N}", "phase1"):
            continue
        p1, _ = load_results(f"ed_tfim_N{N}", subdir="phase1")
        if "M2" not in p1:
            continue
        m2_p1 = np.asarray(p1["M2"], dtype=float).reshape(-1)
        if m2_p1.size != a["h"].size:
            continue
        if "h" in p1 and not np.allclose(np.asarray(p1["h"]).reshape(-1), a["h"]):
            continue
        worst = max(worst, float(np.max(np.abs(m2_of(a["X"], N) - m2_p1))))
        n_cmp += 1
    if n_cmp:
        rep.check("C6", "ED M^2 equals the saved Phase 1 M^2", worst < tol.exact_tol,
                  f"max |dM^2| = {worst:.2e} over {n_cmp} sizes")
    else:
        rep.note("C6 skipped: no comparable results/phase1/ed_tfim_N{N} M2 arrays")

    # C7-C9 shadows ---------------------------------------------------------------
    rep.section("Classical shadows")
    zs, ratios, per_src = [], [], {}
    for src in ("ed", "vqe", "noisy"):
        zz_src, rr_src = [], []
        for a in data[src].values():
            N, M = int(a["n_qubits"]), int(a["snapshots"])
            X, Xs = (a["X"], a["Xs"]) if src != "noisy" else (a["X_gate"], a["Xs_gate"])
            z, r = shadow_stats(X, Xs, N, M)
            zz_src.append(z)
            rr_src.append(r)
        if zz_src:
            per_src[src] = (np.concatenate(zz_src), np.concatenate(rr_src))
            zs.append(per_src[src][0])
            ratios.append(per_src[src][1])
    if zs:
        z = np.concatenate(zs)
        r = np.concatenate(ratios)
        frac3 = float(np.mean(np.abs(z) > 3))
        rep.check("C7", "shadow estimates unbiased (z-scores of the repetition mean against exact values)",
                  frac3 < tol.z_frac_max and np.max(np.abs(z)) < tol.z_abs_max,
                  f"{z.size} values, |z| > 3: {100 * frac3:.2f} % (normal 0.27 %), max |z| = {np.max(np.abs(z)):.2f}, "
                  f"mean z = {z.mean():+.3f}")
        R = int(next(iter(data[next(s for s in per_src)].values()))["repetitions"])
        med_expected = stats.chi2.median(R - 1) / (R - 1)
        rep.check("C8", "shadow variance equals the analytic (9 - <ZZ>^2)/M and (3 - <X>^2)/M",
                  abs(r.mean() - 1) < tol.var_ratio_tol,
                  f"mean empirical/analytic = {r.mean():.4f}, median {np.median(r):.4f} "
                  f"(chi-square expectation {med_expected:.4f} for R = {R})")
        for src, (zz_, rr_) in per_src.items():
            rep.note(f"{src}: |z| > 3 in {100 * np.mean(np.abs(zz_) > 3):.2f} %, variance ratio mean {rr_.mean():.4f}")
        keep["z"], keep["ratio"] = z, r
    else:
        rep.note("C7, C8 skipped: no shadow data")
    if data["noisy"]:
        tm = np.concatenate([t_stats(a["X_meas"], a["Xs_meas"]) for a in data["noisy"].values()])
        tr = np.concatenate([t_stats(a["X_ro"], a["Xs_ro"]) for a in data["noisy"].values()])
        R = int(next(iter(data["noisy"].values()))["repetitions"])
        p_exp = 2 * stats.t.sf(3, R - 1)
        rep.check("C9", "readout shadows match the exact readout-biased and readout-corrected values",
                  np.mean(np.abs(tm) > 3) < tol.t_frac_max and np.mean(np.abs(tr) > 3) < tol.t_frac_max,
                  f"|t| > 3: meas {100 * np.mean(np.abs(tm) > 3):.2f} %, ro {100 * np.mean(np.abs(tr) > 3):.2f} % "
                  f"(t with {R - 1} dof: {100 * p_exp:.2f} %)")

    # C10-C13 noisy identities ----------------------------------------------------
    if data["noisy"]:
        rep.section("Noisy datasets")
        d_ro, d_vqe, match, bases_ok, n_vqe = 0.0, 0.0, 1.0, True, 0
        pur_ok, fid0 = True, 0.0
        for (an, N, p, c), a in data["noisy"].items():
            d_ro = max(d_ro, float(np.max(np.abs(a["X_ro"][0] - a["X_gate"][0]))))
            pur_ok &= bool(np.all(np.diff(a["purity"], axis=0) < 1e-12))
            fid0 = max(fid0, float(np.max(np.abs(a["fidelity_ideal"][0] - 1))))
            v = data["vqe"].get((an, N, p))
            if v is not None:
                n_vqe += 1
                d_vqe = max(d_vqe, float(np.max(np.abs(a["X_gate"][0] - v["X"]))))
                b0 = S.unpack_bits(a["bits_gate"][0], N)
                match = min(match, float(np.mean(np.all(b0 == S.unpack_bits(v["bits"], N), axis=-1))))
                bases_ok &= bool(np.array_equal(a["bases"], v["bases"]))
        rep.check("C10", "strength 0: readout-corrected = gate-only features (no SPAM without gate noise)",
                  d_ro < tol.exact_tol, f"max |dX| = {d_ro:.2e}")
        if n_vqe:
            rep.check("C11", "strength 0 reproduces the noiseless VQE dataset (features, bases, snapshots)",
                      d_vqe < tol.exact_tol and bases_ok and match >= tol.snapshot_match_min,
                      f"max |dX| = {d_vqe:.2e}, bases identical: {bases_ok}, identical snapshots {100 * match:.4f} % "
                      f"({n_vqe} files)")
        else:
            rep.note("C11 skipped: no matching noiseless VQE datasets at p = N/2")
        rep.check("C12", "purity falls monotonically with strength; fidelity to the noiseless state is 1 at 0",
                  pur_ok and fid0 < tol.exact_tol, f"max |F(0) - 1| = {fid0:.2e}")

        worst, worst_z, n5, n5z = 0.0, 0.0, 0, 0
        for (an, N, p, c), a in data["noisy"].items():
            name = f"zne_{an}_N{N}_p{p}_{c}"
            if not result_exists(name, "phase5"):
                continue
            p5, _ = load_results(name, subdir="phase5")
            if "E_true" not in p5 or not np.allclose(np.asarray(p5["strengths"]), a["strengths"]):
                continue
            idx = field_indices(np.asarray(a["h"]), np.asarray(p5["h"]))
            E5 = np.asarray(p5["E_true"])                                   # [s, f, folding, scale]
            worst = max(worst, float(np.max(np.abs(a["energy"][:, idx] - E5[:, :, 0, 0]))))
            n5 += 1
            zd = data["zne"].get((an, N, p, c))
            if zd is not None:
                for jw, js in enumerate(zd["strength_index"]):
                    worst_z = max(worst_z, float(np.max(np.abs(zd["energy_scales"][jw][:, idx] - E5[js, :, 0, :].T))))
                n5z += 1
        if n5:
            rep.check("C13", "noisy energies Tr(rho H) equal the Phase 5 E_true (scale 1; ZNE: every scale)",
                      worst < 1e-9 and worst_z < 1e-9,
                      f"max |dE| = {worst:.2e} ({n5} files), folded {worst_z:.2e} ({n5z} files)")
        else:
            rep.note("C13 skipped: no matching results/phase5/zne_* files")

    # C14, C15 ZNE ----------------------------------------------------------------
    if data["zne"]:
        rep.section("ZNE datasets")
        d1, ratios_z, n = 0.0, [], 0
        for key, z in data["zne"].items():
            a = data["noisy"].get(key)
            if a is None:
                continue
            n += 1
            for jw, js in enumerate(z["strength_index"]):
                d1 = max(d1, float(np.max(np.abs(z["Xs_ro_scales"][jw, 0] - a["Xs_ro"][js]))),
                         float(np.max(np.abs(z["X_gate_scales"][jw, 0] - a["X_gate"][js]))))
                X0 = a["X_gate"][0]
                e_raw = np.median(np.abs(a["X_gate"][js] - X0))
                e_zne = np.median(np.abs(z["X_zne_gate"][jw] - X0))
                ratios_z.append((key, float(z["lam"][jw]), e_zne / e_raw))
        if n:
            rep.check("C14", "ZNE scale 1 equals the noisy dataset (same snapshots and calibration)",
                      d1 < tol.exact_tol, f"max |dX| = {d1:.2e} ({n} files)")
            worst_ratio = max(r[2] for r in ratios_z)
            rep.check("C15", "ZNE (exact, gate noise) reduces the median feature error in the lambda window",
                      worst_ratio < 1.0, "; ".join(f"{k[0]} N={k[1]} lambda={lam:.3f}: {r:.3f}"
                                                   for k, lam, r in ratios_z))
        else:
            rep.note("C14, C15 skipped: ZNE files without noisy counterparts")

    # C16 physics of the training regions -----------------------------------------
    rep.section("Separability of the training regions")
    if data["ed"]:
        margins = []
        for N, a in sorted(data["ed"].items()):
            o, d = train_masks(a["h"])
            m2 = m2_of(a["X"], N)
            m2s = m2_of(a["Xs"].mean(axis=1), N)
            margins.append((N, float(m2[o].min() - m2[d].max()), float(m2s[o].min() - m2s[d].max()),
                            float(m2[np.isclose(a["h"], CFG.tfim.h_critical)][0])))
        rep.check("C16", "ED: M^2 in the ordered training region exceeds M^2 in the disordered region (every N)",
                  all(m[1] > 0 and m[2] > 0 for m in margins),
                  "; ".join(f"N={m[0]}: margin {m[1]:.3f} (shadow {m[2]:.3f}), M^2(h_c) {m[3]:.3f}" for m in margins))
    for (an, N, p, c), a in sorted(data["noisy"].items()):
        o, d = train_masks(a["h"])
        m2 = m2_of(a["X_gate"], N)                                        # (n_s, n_h)
        marg = m2[:, o].min(axis=1) - m2[:, d].max(axis=1)
        lost = np.flatnonzero(marg <= 0)
        lam = a["lam"]
        txt = (f"first lost at p = {a['strengths'][lost[0]]:.2e} (lambda = {lam[lost[0]]:.3f})" if lost.size
               else "kept at all strengths")
        rep.note(f"{an.upper()} N={N} p={p} {c}: M^2 margin {marg[0]:.3f} at p = 0, {marg[-1]:.3f} at "
                 f"p = {a['strengths'][-1]:.0e} (lambda = {lam[-1]:.2f}); {txt}")
        rb = float(np.max(np.abs(a["X_meas"][0] - a["X_gate"][0])))
        rr = float(np.max(np.abs(a["X_ro"][1:] - a["X_gate"][1:]))) if a["strengths"].size > 1 else 0.0
        rep.note(f"{an.upper()} N={N}: readout bias of features at p = 0 up to {rb:.3f}; after exact "
                 f"correction with the noisy-gate calibration the residual (SPAM) is up to {rr:.2e}")
    return keep


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def shade(ax):
    ax.axvspan(CFG.tfim.h_min - 0.05, CFG.ml.train_h_ordered_max, color=COLORS["sky"], alpha=0.12, lw=0)
    ax.axvspan(CFG.ml.train_h_disordered_min, CFG.tfim.h_max + 0.05, color=COLORS["orange"], alpha=0.12, lw=0)
    ax.axvline(CFG.tfim.h_critical, color="0.4", ls=":", lw=1)
    ax.set_xlim(CFG.tfim.h_min - 0.05, CFG.tfim.h_max + 0.05)


def fig_ed(data):
    if not data["ed"]:
        return
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.42))
    sizes = sorted(data["ed"])
    for k, N in enumerate(sizes):
        a = data["ed"][N]
        zz, x = F.unpack_features(a["X"], N)
        axes[0].plot(a["h"], F.magnetization_m2(zz), color=PALETTE[k % len(PALETTE)], label=f"N = {N}")
        axes[1].plot(a["h"], x.mean(axis=-1), color=PALETTE[k % len(PALETTE)])
    N = sizes[len(sizes) // 2]
    a = data["ed"][N]
    zz_s, x_s = F.unpack_features(a["Xs"], N)
    m2s, xs = F.magnetization_m2(zz_s), x_s.mean(axis=-1)
    axes[0].errorbar(a["h"], m2s.mean(1), m2s.std(1, ddof=1), fmt="o", ms=2.5, color="k", lw=0.8,
                     label=f"shadow, N = {N}")
    axes[1].errorbar(a["h"], xs.mean(1), xs.std(1, ddof=1), fmt="o", ms=2.5, color="k", lw=0.8)
    axes[0].set(xlabel="$h/J$", ylabel="$M^2$")
    axes[1].set(xlabel="$h/J$", ylabel=r"$\frac{1}{N}\sum_i\langle X_i\rangle$")
    for ax in axes:
        shade(ax)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    save_figure(fig, "phase6_ed_features", FIG)
    plt.close(fig)


def pick_noisy(data):
    keys = [k for k in data["noisy"] if k[0] == "hva" and k[3] == "realistic"] or list(data["noisy"])
    return max(keys, key=lambda k: k[1]) if keys else None


def fig_noisy(data):
    key = pick_noisy(data)
    if key is None:
        return
    a = data["noisy"][key]
    an, N, p, c = key
    s = np.asarray(a["strengths"])
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.42))
    cmap = plt.cm.viridis
    m2 = m2_of(a["X_gate"], N)
    for js in range(s.size):
        axes[0].plot(a["h"], m2[js], color=cmap(js / max(s.size - 1, 1)), lw=1.2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, s.size - 1))
    cb = fig.colorbar(sm, ax=axes[0], pad=0.02)
    cb.set_ticks([0, s.size - 1])
    cb.set_ticklabels(["0", f"{s[-1]:.0e}"])
    cb.set_label("noise strength", fontsize=8, labelpad=-6)
    axes[0].set(xlabel="$h/J$", ylabel="$M^2$ (exact, gate noise)", title=f"{an.upper()}, N = {N}, depth {p}")
    js = int(np.argmin(np.abs(s - CFG.mitigation.plot_strength)))
    axes[1].plot(a["h"], m2[0], color=SERIES_COLORS["HVA" if an == "hva" else "HEA"], label="noiseless")
    axes[1].plot(a["h"], m2[js], color=SERIES_COLORS["noisy"], label="gate noise")
    axes[1].plot(a["h"], m2_of(a["X_meas"][js], N), color=COLORS["purple"], ls="--", label="+ readout")
    sh = F.magnetization_m2(F.unpack_features(a["Xs_ro"][js], N)[0])
    axes[1].errorbar(a["h"], sh.mean(1), sh.std(1, ddof=1), fmt="o", ms=2.5, lw=0.8,
                     color=SERIES_COLORS["mitigated"], label="shadow, readout-corrected")
    axes[1].set(xlabel="$h/J$", ylabel="$M^2$", title=f"strength {s[js]:.1e}, $\\lambda$ = {a['lam'][js]:.2f}")
    axes[1].legend(fontsize=7, loc="lower left")
    for ax in axes:
        shade(ax)
    fig.tight_layout()
    save_figure(fig, "phase6_noisy_features", FIG)
    plt.close(fig)


def fig_shadow(keep):
    if "z" not in keep:
        return
    z, r = keep["z"], keep["ratio"]
    R = CFG.ml.shadow_repetitions
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.4))
    zc = np.clip(z, -6, 6)
    axes[0].hist(zc, bins=80, density=True, color=COLORS["blue"], alpha=0.7)
    g = np.linspace(-6, 6, 400)
    axes[0].plot(g, stats.norm.pdf(g), color="k", lw=1, label="$\\mathcal{N}(0,1)$")
    axes[0].set(xlabel="$z$ (repetition mean)", ylabel="density")
    axes[0].legend(fontsize=8)
    rc = np.clip(r, 0, 4)
    axes[1].hist(rc, bins=80, density=True, color=COLORS["orange"], alpha=0.7)
    g = np.linspace(1e-3, 4, 400)
    axes[1].plot(g, stats.chi2.pdf(g * (R - 1), R - 1) * (R - 1), color="k", lw=1,
                 label=f"$\\chi^2_{{{R - 1}}}/{R - 1}$")
    axes[1].set(xlabel="empirical / analytic variance", ylabel="density")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "phase6_shadow_validation", FIG)
    plt.close(fig)


def fig_zne(data):
    keys = [k for k in data["zne"] if k in data["noisy"]]
    if not keys:
        return
    key = max(keys, key=lambda k: (k[0] == "hva", k[1]))
    z, a = data["zne"][key], data["noisy"][key]
    an, N, p, c = key
    nw = z["strengths"].size
    fig, axes = plt.subplots(1, nw, figsize=(SINGLE_COLUMN, SINGLE_COLUMN * 0.4), squeeze=False)
    for jw in range(nw):
        ax = axes[0, jw]
        js = int(z["strength_index"][jw])
        ax.plot(a["h"], m2_of(a["X_gate"][0], N), color="k", label="noiseless")
        ax.plot(a["h"], m2_of(a["X_ro"][js], N), color=SERIES_COLORS["noisy"], label="unmitigated (ro)")
        ax.plot(a["h"], m2_of(z["X_zne_ro"][jw], N), color=SERIES_COLORS["mitigated"], label="ZNE (exact)")
        sh = F.magnetization_m2(F.unpack_features(z["Xs_zne_ro"][jw], N)[0])
        ax.errorbar(a["h"], sh.mean(1), sh.std(1, ddof=1), fmt="o", ms=2.5, lw=0.8,
                    color=COLORS["green"], alpha=0.6, label="ZNE (shadow)")
        ax.set(xlabel="$h/J$", ylabel="$M^2$",
               title=f"{an.upper()} N = {N}, $\\lambda$ = {z['lam'][jw]:.2f}")
        shade(ax)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    save_figure(fig, "phase6_zne_features", FIG)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    set_thesis_style()
    rep = Report()
    data = load_all()
    keep = run_checks(data, rep)
    fig_ed(data)
    fig_noisy(data)
    fig_shadow(keep)
    fig_zne(data)
    head = [f"Phase 6 summary: {rep.n_pass} checks passed, {rep.n_fail} failed, {rep.n_note} notes"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase6_summary.txt").write_text("\n".join(head + rep.lines) + "\n", encoding="utf-8")
    print(f"\n{head[0]}; summary in {OUT / 'phase6_summary.txt'}")


if __name__ == "__main__":
    main()
