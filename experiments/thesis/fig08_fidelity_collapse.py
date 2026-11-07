"""
Thesis Figure 8: loss of fidelity to the noiseless circuit state against the
error budget, for every ansatz, size, depth and field (Phase 4, experiment A).

    python experiments\\thesis\\fig08_fidelity_collapse.py

Panels: (a) depolarizing, (b) amplitude damping, (c) dephasing. Each point is
one state (HVA circles, HEA squares; N = 4, 6, 8; p = 1 .. N/2; 20 fields;
10 positive strengths). The error budget and the collapse are

    lambda = p (n_1q + n_2q),        1 - F_ideal ~ 1 - exp(-kappa lambda),

with kappa the median of -ln(F_ideal)/lambda over all points with lambda <= 1.

Validation (all 18 files, 3 channels, 11 strengths, 20 fields)
  * Zero noise reproduces Phase 2 (energy, F_GS) and Phase 3 (half-chain S).
  * Density matrices are valid: trace 1, Hermitian, no negative eigenvalues.
  * Gate counts: n_1q = N(p+1) (HVA), 2N(p+1) (HEA); n_2q = p(N-1).
  * Purity decreases monotonically with the strength (every channel).
  * No-error branch, exact lower bounds: depolarizing keeps the ideal state with
    probability (1-p) per gate, phase damping is a Z flip with probability
    q = (1 - sqrt(1-gamma))/2 on every qubit a gate touches, so
        F_ideal >= (1 - p)^(n_1q + n_2q),   F_ideal >= (1 - q)^(n_1q + 2 n_2q).
  * Weak noise: 1 - F_ideal and 1 - purity grow linearly in p (log-log slope 1
    within CFG.noise.slope_tol for p <= CFG.noise.weak_noise_max).
  * kappa medians and 5 to 95 % ranges reproduce the Phase 4 report.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from config import CFG
from experiments.thesis.figlib import FigureSpec, error_budget_phase4, run_figure
from utils.plotting import (CHANNEL_LABELS, COLORS, SERIES_COLORS, ANSATZ_MARKERS,
                            panel_labels, thesis_figure)

SPEC = FigureSpec(
    number=8,
    slug="fidelity_collapse",
    chapter="Noise robustness",
    phases=(2, 3, 4),
    title="Fidelity loss of noisy VQE states collapses onto the error budget",
    caption=("Infidelity to the noiseless circuit state against the error budget "
             "lambda = p (n_1q + n_2q) for (a) depolarizing, (b) amplitude-damping and (c) "
             "dephasing noise after every gate. Each marker is one state: both ansaetze (HVA "
             "circles, HEA squares), N = 4, 6, 8, every depth p <= N/2, 20 fields and 10 noise "
             "strengths. The black line is 1 - exp(-kappa lambda) with the median kappa; the grey "
             "band spans the 5 to 95 % range of kappa."),
)

ANSATZE = (("hva", "HVA"), ("hea", "HEA"))


def build(ctx):
    th, nz = CFG.thesis, CFG.noise
    sizes = list(CFG.system.sizes_noisy)
    keys = ("energy", "fidelity_gs", "fidelity_ideal", "purity", "S_A", "trace", "min_eig", "herm_err",
            "strengths", "channels", "h", "phase2_energy", "phase2_fidelity_gs", "phase3_S",
            "n_1q", "n_2q")
    files = {(a, N, p): ctx.load(f"noisy_eval_{a}_N{N}_p{p}", "phase4", require=keys)
             for a, _ in ANSATZE for N in sizes for p in CFG.system.depths(N, noisy=True)}
    report = ctx.load_json("phase4_report_data.json", "phase4")
    first = next(iter(files.values()))
    channels = [str(c) for c in first["channels"]]
    s = np.asarray(first["strengths"], dtype=float)
    ctx.param(sizes=sizes, depths="1..N/2", channels=channels, strengths=s.tolist(),
              lambda_definition="p (n_1q + n_2q)", kappa_window="lambda <= 1, p > 0")

    # -- validation ------------------------------------------------------------
    dE, dF, dS, tr, me, he, counts, mono = [], [], [], [], [], [], True, []
    lo_dep, lo_deph, slopes_F, slopes_P = [], [], [], []
    weak = (s > 0) & (s <= nz.weak_noise_max)
    iw = np.flatnonzero(weak)[[0, -1]]
    for (a, N, p), d in files.items():
        n1, n2 = int(d["n_1q"]), int(d["n_2q"])
        counts &= (n1 == (N if a == "hva" else 2 * N) * (p + 1)) and (n2 == p * (N - 1))
        dE.append(np.max(np.abs(d["energy"][:, 0] - d["phase2_energy"])))
        dF.append(np.max(np.abs(d["fidelity_gs"][:, 0] - d["phase2_fidelity_gs"])))
        dS.append(np.max(np.abs(d["S_A"][:, 0] - d["phase3_S"])))
        tr.append(np.max(np.abs(d["trace"] - 1.0)))
        me.append(np.min(d["min_eig"]))
        he.append(np.max(d["herm_err"]))
        mono.append(np.max(np.diff(d["purity"], axis=1)))
        F = np.asarray(d["fidelity_ideal"])
        for c, ch in enumerate(channels):
            if ch == "depolarizing":
                lo_dep.append(np.min(F[c] - ((1 - s) ** (n1 + n2))[:, None]))
            elif ch == "dephasing":
                q = 0.5 * (1 - np.sqrt(1 - s))
                lo_deph.append(np.min(F[c] - ((1 - q) ** (n1 + 2 * n2))[:, None]))
            x = s[iw]
            for k in range(F.shape[2]):
                if 1 - F[c, iw[0], k] > 1e-14:
                    slopes_F.append(np.log((1 - F[c, iw[1], k]) / (1 - F[c, iw[0], k])) / np.log(x[1] / x[0]))
                if 1 - d["purity"][c, iw[0], k] > 1e-14:
                    P = d["purity"][c]
                    slopes_P.append(np.log((1 - P[iw[1], k]) / (1 - P[iw[0], k])) / np.log(x[1] / x[0]))
    ctx.check_max("zero noise reproduces the Phase 2 energies", dE, th.identity_tol)
    ctx.check_max("zero noise reproduces the Phase 2 ground-space fidelities", dF, th.identity_tol)
    ctx.check_max("zero noise reproduces the Phase 3 half-chain entropies", dS, th.identity_tol)
    ctx.check_true("density matrices valid: |Tr rho - 1|, Hermiticity error, min eigenvalue",
                   max(tr) < 1e-12 and max(he) < 1e-12 and min(me) > -1e-12,
                   f"max |Tr - 1| {max(tr):.1e}, max herm err {max(he):.1e}, min eig {min(me):.1e}")
    ctx.check_true("gate counts n_1q = N(p+1) (HVA), 2N(p+1) (HEA); n_2q = p(N-1)", counts)
    ctx.check_max("purity decreases monotonically with the strength (all channels)", mono,
                  nz.monotone_tol)
    ctx.check_min("depolarizing: F_ideal >= (1-p)^(n_1q+n_2q) (no-error branch)", lo_dep, -1e-12)
    ctx.check_min("dephasing: F_ideal >= (1-q)^(n_1q+2n_2q), q = (1-sqrt(1-gamma))/2", lo_deph, -1e-12)
    sf, sp = np.median(slopes_F), np.median(slopes_P)
    ctx.check_close("weak noise: 1 - F_ideal linear in p (median log-log slope)", sf, 1.0,
                    atol=nz.slope_tol, detail=f"range {np.min(slopes_F):.3f} to {np.max(slopes_F):.3f}")
    ctx.check_close("weak noise: 1 - purity linear in p (median log-log slope)", sp, 1.0,
                    atol=nz.slope_tol, detail=f"range {np.min(slopes_P):.3f} to {np.max(slopes_P):.3f}")

    kap, pts = {}, {}
    for c, ch in enumerate(channels):
        lam_all, inf_all, kk, mark = [], [], [], []
        for (a, N, p), d in files.items():
            lam = error_budget_phase4(s, int(d["n_1q"]), int(d["n_2q"]))
            F = np.asarray(d["fidelity_ideal"])[c]                    # [strength, field]
            L = np.repeat(lam[:, None], F.shape[1], axis=1)
            sel = (L > 0)
            lam_all.append(L[sel]); inf_all.append(1 - F[sel]); mark.append(np.full(sel.sum(), a))
            w = sel & (L <= 1.0)
            kk.append(-np.log(F[w]) / L[w])
        kk = np.concatenate(kk)
        kap[ch] = (float(np.median(kk)), float(np.percentile(kk, 5)), float(np.percentile(kk, 95)), kk.size)
        pts[ch] = (np.concatenate(lam_all), np.concatenate(inf_all), np.concatenate(mark))
    mine = [v for ch in channels for v in kap[ch][:3]]
    theirs = [report["kappa"][ch][k] for ch in channels for k in ("median", "p05", "p95")]
    ctx.check_rounded("kappa median and 5 to 95 % range reproduce the Phase 4 report", mine, theirs)
    for ch in channels:
        m, lo, hi, n = kap[ch]
        ctx.note(f"{ch}: kappa median {m:.4f} (5-95 %: {lo:.4f}-{hi:.4f}) over {n} states with "
                 f"lambda <= 1 (report n = {report['kappa'][ch].get('n')})")

    # -- figure --------------------------------------------------------------------
    fig, axes = thesis_figure(ncols=3, aspect=0.36)
    lam_line = np.logspace(-3.5, 1.0, 100)
    for ax, ch in zip(axes, channels):
        L, inf, mark = pts[ch]
        for a, name in ANSATZE:
            sel = (mark == a) & (inf > 1e-16)
            ax.loglog(L[sel], inf[sel], ANSATZ_MARKERS[name], color=SERIES_COLORS[name], ms=1.6,
                      alpha=0.35, mew=0, label=name)
        m, lo, hi, _ = kap[ch]
        ax.fill_between(lam_line, 1 - np.exp(-lo * lam_line), 1 - np.exp(-hi * lam_line),
                        color=COLORS["grey"], alpha=0.25, lw=0)
        ax.loglog(lam_line, 1 - np.exp(-m * lam_line), color=COLORS["black"], lw=0.9)
        ax.text(0.96, 0.04, rf"$1-e^{{-\kappa\lambda}}$, $\kappa={m:.2f}$", transform=ax.transAxes,
                ha="right", va="bottom")
        ax.set_xlim(lam_line[0], lam_line[-1])
        ax.set_ylim(1e-4, 1.5)
        ax.set_xlabel(r"error budget $\lambda = p\,(n_{1q}+n_{2q})$")
        ax.set_ylabel(r"$1 - F_{\mathrm{ideal}}$")
        ax.set_title(CHANNEL_LABELS.get(ch, ch), loc="right")
        leg = ax.legend(loc="upper left", markerscale=2.5)
        for hnd in leg.legend_handles:
            hnd.set_alpha(1.0)
    panel_labels(axes)
    return fig


if __name__ == "__main__":
    raise SystemExit(run_figure(SPEC, build))
