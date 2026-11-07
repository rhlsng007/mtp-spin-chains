"""
Thesis Figure 11: readout correction and zero-noise extrapolation under the
hardware-like noise model (Phase 5).

    python experiments\\thesis\\fig11_error_mitigation.py

Panels (N = CFG.mitigation.plot_size, p = N/2, h = CFG.mitigation.plot_field,
channel "realistic", global folding with scales 1, 3, 5)
  (a) HVA, exact expectation values: error of the unmitigated, readout-corrected,
      ZNE-only and combined (readout + Richardson ZNE) energies
  (b) HEA, the same
  (c) 10,000 shots per basis and scale: RMS error over 50 repetitions

The error is measured against the noiseless circuit (the most mitigation can
recover), and Richardson extrapolation through lambda = 1, 3, 5 uses

    delta = |E - E_ideal| / |E0|,    E* = (15/8) E(1) - (5/4) E(3) + (3/8) E(5).

The dashed line is the noiseless VQE error (E_ideal - E0)/|E0|; below it the
mitigated energy is at VQE accuracy. The top axis is the error budget
lambda = p (n_1q + 10 n_2q).

Validation (all 72 ZNE files, 11 strengths, 20 fields)
  * Zero noise: every folded circuit gives E_ideal (folding is the identity),
    and E_ideal equals the Phase 2 energy.
  * Exact readout correction at zero noise recovers E_ideal; the unmitigated
    readout bias at the reference point equals the Phase 5 report.
  * Scale 1 is the same circuit for global and local folding.
  * Variational principle for every unmitigated noisy state (any scale).
  * Extrapolation weights: sum 1, kill the terms in lambda and lambda^2
    (Richardson), and sum c_k^2 equals the Phase 5 variance amplification
    (Richardson 167/32, linear 35/24).
  * Shot statistics: the spread of the 50 repetitions matches the analytic
    variance (median ratio within CFG.mitigation.variance_ratio_tol).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from config import CFG
from experiments.thesis.figlib import (FigureSpec, error_budget_phase5, extrapolation_weights, field_index,
                                       run_figure)
from utils.plotting import (COLORS, MITIGATION_COLORS, add_secondary_scale, label_zero_strength,
                            panel_labels, reference_line, thesis_figure, zero_strength_positions)

SPEC = FigureSpec(
    number=11,
    slug="error_mitigation",
    chapter="Error mitigation",
    phases=(2, 5),
    title="Readout correction and zero-noise extrapolation under hardware-like noise",
    caption=("Energy error relative to the noiseless circuit for N = 8, p = 4, h = J under the "
             "hardware-like noise model (two-qubit error 10 p, readout error p01 = 0.01, "
             "p10 = 0.02), against the gate-error strength p (bottom) and the error budget "
             "lambda (top). (a) HVA and (b) HEA with exact expectation values: unmitigated, readout-"
             "corrected, zero-noise extrapolation alone (Richardson, global folding, scales 1, 3, 5) "
             "and both combined. The dashed line is the noiseless VQE error. (c) RMS error over 50 "
             "repetitions with 10,000 shots per basis and scale (HVA solid, HEA dashed): shot noise "
             "sets a floor, amplified about 2.3 times by the extrapolation."),
)

ANSATZE = (("hva", "HVA"), ("hea", "HEA"))
METHODS = (("raw", "unmitigated"), ("readout", "readout corrected"), ("zne", "ZNE only"),
           ("combined", "readout + ZNE"))
FLOOR = 1e-16


def estimates(d, w, fold: int = 0) -> dict:
    """Exact energies [strength, field] of the four mitigation stages."""
    return {"raw": d["E_raw"][:, :, fold, 0], "readout": d["E_ro"][:, :, fold, 0],
            "zne": d["E_raw"][:, :, fold, :] @ w, "combined": d["E_ro"][:, :, fold, :] @ w}


def build(ctx):
    th, mt = CFG.thesis, CFG.mitigation
    sizes = list(CFG.system.sizes_noisy)
    keys = ("E_true", "E_raw", "E_ro", "Es_raw", "Es_ro", "var_raw", "E_ideal", "E0", "phase2_energy",
            "strengths", "scales", "foldings", "h", "n_1q", "n_2q")
    files = {(a, N, p, c): ctx.load(f"zne_{a}_N{N}_p{p}_{c}", "phase5", require=keys)
             for a, _ in ANSATZE for N in sizes for p in CFG.system.depths(N, noisy=True)
             for c in mt.channels}
    report = ctx.load_json("phase5_report_data.json", "phase5")
    first = next(iter(files.values()))
    s = np.asarray(first["strengths"], dtype=float)
    scales = np.asarray(first["scales"], dtype=float)
    folds = [str(f) for f in first["foldings"]]
    fg = folds.index(mt.main_folding)
    wR, wL = extrapolation_weights(scales, len(scales) - 1), extrapolation_weights(scales, 1)
    Np, ch = mt.plot_size, "realistic"
    ih = field_index(first["h"], mt.plot_field)
    ctx.param(plot_size=Np, plot_depth=Np // 2, field=mt.plot_field, channel=ch, scales=scales.tolist(),
              folding=mt.main_folding, extrapolator=mt.main_extrapolator, shots=int(first.get("shots", 0)),
              repetitions=int(first.get("repetitions", 0)), richardson_weights=wR.tolist())

    # -- validation ------------------------------------------------------------
    z_true, z_ideal, z_ro, same1, var = [], [], [], [], []
    ratios = []
    for (a, N, p, c), d in files.items():
        Ei = np.asarray(d["E_ideal"])
        z_true.append(np.max(np.abs(d["E_true"][0] - Ei[:, None, None])))
        z_ideal.append(np.max(np.abs(Ei - d["phase2_energy"])))
        z_ro.append(np.max(np.abs(d["E_ro"][0] - Ei[:, None, None])))
        same1.append(np.max(np.abs(d["E_true"][:, :, 0, 0] - d["E_true"][:, :, 1, 0])))
        var.append(np.min((d["E_true"] - np.asarray(d["E0"])[None, :, None, None]) /
                          np.abs(np.asarray(d["E0"]))[None, :, None, None]))
        v = np.asarray(d["var_raw"])
        emp = np.var(np.asarray(d["Es_raw"]), axis=-1, ddof=1)
        ok = v > 0
        ratios.append((emp[ok] / v[ok]).ravel())
    ctx.check_max("zero noise: every folded circuit gives E_ideal (both foldings, all scales)", z_true,
                  th.identity_tol)
    ctx.check_max("E_ideal equals the Phase 2 energy", z_ideal, th.identity_tol)
    ctx.check_max("zero noise: exact readout correction recovers E_ideal", z_ro, th.identity_tol)
    ctx.check_max("scale 1: global and local folding are the same circuit", same1, 1e-12)
    ctx.check_min("variational principle for every unmitigated noisy state", var, -th.variational_tol)
    mom = [abs(wR.sum() - 1), abs(wR @ scales), abs(wR @ scales ** 2), abs(wL.sum() - 1), abs(wL @ scales)]
    ctx.check_max("extrapolation weights: sum 1; Richardson kills lambda and lambda^2; linear kills lambda",
                  mom, 1e-12, detail=f"Richardson {np.round(wR, 6).tolist()}, linear {np.round(wL, 6).tolist()}")
    va = report.get("variance_amplification", {})
    ctx.check_close("sum c_k^2 = 167/32 (Richardson), 35/24 (linear), as in the Phase 5 report",
                    [wR @ wR, wL @ wL], [va.get("richardson", np.nan), va.get("linear", np.nan)], rtol=1e-12,
                    detail=f"exact {167 / 32:.6f}, {35 / 24:.6f}")
    rat = float(np.median(np.concatenate(ratios)))
    ctx.check_close("shot spread of 50 repetitions matches the analytic variance (median ratio)", rat, 1.0,
                    atol=mt.variance_ratio_tol, detail="chi-squared with 49 degrees of freedom has median 0.986")

    ref = report.get("reference_point", {}).get("data", {})
    est, delta, eps_vqe = {}, {}, {}
    for a, name in ANSATZE:
        d = files[a, Np, Np // 2, ch]
        Ei, E0 = float(d["E_ideal"][ih]), float(d["E0"][ih])
        est[a] = {k: v[:, ih] for k, v in estimates(d, wR, fg).items()}
        delta[a] = {k: np.abs(v - Ei) / abs(E0) for k, v in est[a].items()}
        eps_vqe[a] = (Ei - E0) / abs(E0)
    mine = [delta[a]["raw"][0] for a, _ in ANSATZE]
    theirs = [ref.get(f"{a}_{ch}", {}).get("strength_0", {}).get("unmitigated", np.nan) for a, _ in ANSATZE]
    ctx.check_close(f"readout bias at zero noise (reference point) equals the Phase 5 report", mine, theirs,
                    atol=th.identity_tol)

    dref = files["hva", Np, Np // 2, ch]
    n_1q, n_2q = int(dref["n_1q"]), int(dref["n_2q"])
    lam_per_p = float(error_budget_phase5(ch, 1.0, n_1q, n_2q))
    for a, name in ANSATZE:
        for k, label in METHODS:
            dd = delta[a][k]
            reach = [float(x) for x, y in zip(s, dd) if x > 0 and y < eps_vqe[a]]
            ctx.note(f"{name} {label}: delta at p = 1e-4 {dd[np.argmin(np.abs(s - 1e-4))]:.2e}, "
                     f"1e-3 {dd[np.argmin(np.abs(s - 1e-3))]:.2e}; at VQE accuracy "
                     f"({eps_vqe[a]:.2e}) up to p = {max(reach) if reach else 'never'}")
    ctx.note(f"lambda = {lam_per_p:g} p at N = {Np}, p = {Np // 2} (both ansaetze)")

    # -- shots -------------------------------------------------------------------------
    rms = {}
    for a, _ in ANSATZE:
        d = files[a, Np, Np // 2, ch]
        Ei, E0 = float(d["E_ideal"][ih]), float(d["E0"][ih])
        raw = np.asarray(d["Es_raw"])[:, ih, 0, :]
        ro = np.asarray(d["Es_ro"])[:, ih, 0, :]
        comb = np.einsum("skr,k->sr", np.asarray(d["Es_ro"])[:, ih], wR)          # [strength, rep]
        f = lambda e: np.sqrt(np.mean((e - Ei) ** 2, axis=-1)) / abs(E0)  # noqa: E731
        rms[a] = {"raw": f(raw), "readout": f(ro), "combined": f(comb)}
        ctx.note(f"{a.upper()} shots: RMS at p = {s[1]:g}: unmitigated {rms[a]['raw'][1]:.2e}, readout "
                 f"{rms[a]['readout'][1]:.2e}, combined {rms[a]['combined'][1]:.2e} "
                 f"(ratio {rms[a]['combined'][1] / rms[a]['readout'][1]:.2f}, sqrt(sum c^2) = {np.sqrt(wR @ wR):.2f})")

    # -- figure --------------------------------------------------------------------
    x, x0 = zero_strength_positions(s)
    fig, axes = thesis_figure(ncols=3, aspect=0.38)
    for ax, (a, name) in zip(axes[:2], ANSATZE):
        for k, label in METHODS:
            ax.loglog(x, np.maximum(delta[a][k], FLOOR), "o-", color=MITIGATION_COLORS[k], ms=2.5,
                      lw=1.0, label=label)
        reference_line(ax, eps_vqe[a], color=COLORS["black"], lw=0.7)
        ax.text(x[1], eps_vqe[a] * 1.6, "noiseless VQE error", va="bottom", ha="left")
        ax.set_ylim(1e-8, 2.0)
        ax.set_title(name, loc="right")
    for a, name in ANSATZE:
        ls = "-" if a == "hva" else "--"
        for k in ("raw", "readout", "combined"):
            axes[2].loglog(x, rms[a][k], ls, color=MITIGATION_COLORS[k], lw=1.0, marker="o" if a == "hva" else "s",
                           ms=2.2, mfc="none" if a == "hea" else None)
    axes[2].set_title(f"{int(first.get('shots', 0)):,} shots per basis", loc="right")
    for ax in axes:
        ax.set_xlabel("gate-error strength $p$")
        ax.set_ylabel(r"$\delta = |E - E_{\mathrm{ideal}}|/|E_0|$" if ax is not axes[2] else
                      r"RMS $\delta$ over repetitions")
        label_zero_strength(ax, x0)
        add_secondary_scale(ax, lam_per_p, r"$\lambda$")
    axes[0].legend(loc="lower right")
    panel_labels(axes)
    return fig


if __name__ == "__main__":
    raise SystemExit(run_figure(SPEC, build))
