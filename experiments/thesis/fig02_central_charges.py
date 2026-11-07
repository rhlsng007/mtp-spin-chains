"""
Thesis Figure 2: central charges from exact entanglement profiles at the
critical points (Phase 1 data, N = 4 to 14).

    python experiments\\thesis\\fig02_central_charges.py

Panels
  (a) TFIM at h = J, N = CFG.thesis.cc_size: S(l) against the chord length,
      open and periodic chains, with the Calabrese-Cardy fits
  (b) XXZ chain at Delta = 0 and 0.5, periodic, same N
  (c) fitted c against 1/N: TFIM periodic (-> 1/2), TFIM open, XXZ periodic (-> 1)

Calabrese-Cardy forms (fit window CFG.ent.cc_ell_min <= l <= N/2):

    open:     S(l) = (c/6) ln[(2N/pi) sin(pi l/N)] + s'
    periodic: S(l) = (c/3) ln[(N/pi) sin(pi l/N)] + s

Validation
  * Pure states: S(l) = S(N - l) for every profile.
  * TFIM periodic: c = 1/2 at N = cc_size (tolerance c_tol_tfim_periodic).
  * XXZ periodic, Delta = 0 and 0.5: c = 1 (tolerance c_tol_xxz_periodic).
  * The open TFIM fits reproduce the Phase 3 report values (same data, same
    estimator) to their printed digits.
Open XXZ chains are not fitted: their profiles alternate between even and odd
l (bond dimerization, Phase 1), which the single-parameter form cannot fit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from config import CFG
from experiments.thesis.figlib import FigureSpec, cc_fit, chord_log, run_figure
from utils.plotting import COLORS, panel_labels, reference_line, thesis_figure

SPEC = FigureSpec(
    number=2,
    slug="central_charges",
    chapter="Exact diagonalization baseline",
    phases=(1,),
    title="Central charges from Calabrese-Cardy fits of exact entanglement profiles",
    caption=("Entanglement entropy S(l) of a block of l sites against the chord length, from exact "
             "ground states at the critical points. (a) Transverse-field Ising chain at h = J, "
             "N = 14, open and periodic boundaries; lines are Calabrese-Cardy fits over "
             "2 <= l <= N/2 (filled markers). (b) Periodic XXZ chain at Delta = 0 and 0.5. "
             "(c) Fitted central charge against 1/N; dashed lines mark the exact values "
             "c = 1/2 (Ising) and c = 1 (Luttinger liquid). Error bars are least-squares "
             "standard errors."),
)

SERIES = (  # (profile key, boundary, label, colour, marker)
    ("tfim_periodic", "periodic", "TFIM, periodic", COLORS["black"], "o"),
    ("tfim_open", "open", "TFIM, open", COLORS["grey"], "s"),
    ("xxz_d0_periodic", "periodic", r"XXZ $\Delta=0$, periodic", COLORS["green"], "^"),
    ("xxz_d0p5_periodic", "periodic", r"XXZ $\Delta=0.5$, periodic", COLORS["purple"], "v"),
)
SHORT = {"tfim_periodic": "periodic", "tfim_open": "open",
         "xxz_d0_periodic": r"$\Delta=0$", "xxz_d0p5_periodic": r"$\Delta=0.5$"}
PLAIN = {"tfim_periodic": "TFIM periodic", "tfim_open": "TFIM open",
         "xxz_d0_periodic": "XXZ Delta=0 periodic", "xxz_d0p5_periodic": "XXZ Delta=0.5 periodic"}


def build(ctx):
    th, l0 = CFG.thesis, CFG.ent.cc_ell_min
    sizes = sorted(set(th.cc_fit_sizes) | {th.cc_size} | set(CFG.thesis.ed_sizes))
    keys = tuple(k for k, *_ in SERIES) + ("xxz_d0_open", "xxz_d0p5_open")
    prof = {N: ctx.load(f"ed_cc_profiles_N{N}", "phase1", require=keys) for N in sizes}
    report = ctx.load_json("phase3_report_data.json", "phase3")
    ctx.param(cc_size=th.cc_size, fit_sizes=list(th.cc_fit_sizes), ell_min=l0,
              tfim_field=CFG.tfim.h_critical, xxz_deltas=list(CFG.ed.xxz_cc_deltas))

    # -- validation ------------------------------------------------------------
    asym = max(float(np.max(np.abs(np.asarray(p[k]) - np.asarray(p[k])[::-1])))
               for p in prof.values() for k in keys)
    ctx.check_max("pure states: S(l) = S(N - l) for every profile", asym, th.identity_tol)

    fits = {k: {N: cc_fit(prof[N][k], N, b, l0) for N in th.cc_fit_sizes} for k, b, *_ in SERIES}
    big = th.cc_size
    c_t = cc_fit(prof[big]["tfim_periodic"], big, "periodic", l0)
    ctx.check_close(f"TFIM periodic, N = {big}: c = 1/2", c_t[0], 0.5, atol=th.c_tol_tfim_periodic,
                    detail=f"standard error {c_t[1]:.4f}")
    c_x = [cc_fit(prof[big][k], big, "periodic", l0)[0] for k in ("xxz_d0_periodic", "xxz_d0p5_periodic")]
    ctx.check_close(f"XXZ periodic, Delta = 0 and 0.5, N = {big}: c = 1", c_x, [1.0, 1.0],
                    atol=th.c_tol_xxz_periodic)

    rep_n = [N for N in sizes if f"N{N}" in report.get("ed", {})]
    mine = [cc_fit(prof[N]["tfim_open"], N, "open", l0)[0] for N in rep_n]
    theirs = [report["ed"][f"N{N}"]["c_open"] for N in rep_n]
    ctx.check_rounded("open TFIM fits reproduce the Phase 3 report (c_open)", mine, theirs)

    for k, *_ in SERIES:
        ctx.note(f"{PLAIN[k]}: c(N) = " + ", ".join(f"N={N}: {fits[k][N][0]:.4f}" for N in th.cc_fit_sizes))
    ctx.note(f"N = {big}: TFIM periodic c = {c_t[0]:.4f} +- {c_t[1]:.4f}; XXZ periodic "
             f"c = {c_x[0]:.4f} (Delta 0), {c_x[1]:.4f} (Delta 0.5)")

    # -- figure ------------------------------------------------------------------
    fig, (ax_t, ax_x, ax_c) = thesis_figure(ncols=3, aspect=0.36)

    def draw(ax, key, boundary, label, color, marker):
        N = big
        S = np.asarray(prof[N][key])
        ell = np.arange(1, N)
        x = chord_log(N, boundary)
        inside = (ell >= l0) & (ell <= N // 2)
        c, _, s0 = cc_fit(S, N, boundary, l0)
        scale = 6.0 if boundary == "open" else 3.0
        label = f"{SHORT[key]}: $c={c:.3f}$"
        ax.plot(x[inside], S[inside], marker, color=color, ms=3.5, label=label)
        ax.plot(x[~inside], S[~inside], marker, color=color, ms=3.5, mfc="none")
        xx = np.linspace(x.min(), x.max(), 50)
        ax.plot(xx, c / scale * xx + s0, color=color, lw=0.8)

    for key, b, label, color, marker in SERIES[:2]:
        draw(ax_t, key, b, label, color, marker)
    for key, b, label, color, marker in SERIES[2:]:
        draw(ax_x, key, b, label, color, marker)
    ax_t.set_xlabel(r"$\ln\!\left[\frac{aN}{\pi}\sin\frac{\pi\ell}{N}\right]$, $a=2$ open, $1$ periodic")
    ax_x.set_xlabel(r"$\ln\!\left[\frac{N}{\pi}\sin\frac{\pi\ell}{N}\right]$")
    for ax, loc in ((ax_t, "center right"), (ax_x, "lower right")):
        ax.set_ylabel(r"$S(\ell)$")
        ax.legend(loc=loc)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.12 * (hi - lo))          # room for the panel caption
    ax_t.text(0.04, 0.96, f"TFIM, $N={big}$, $h=J$", transform=ax_t.transAxes, ha="left", va="top")
    ax_x.text(0.04, 0.96, f"XXZ, periodic, $N={big}$", transform=ax_x.transAxes, ha="left", va="top")

    for key, _, label, color, marker in SERIES:
        n = np.array(th.cc_fit_sizes, dtype=float)
        c = np.array([fits[key][N][0] for N in th.cc_fit_sizes])
        e = np.array([fits[key][N][1] for N in th.cc_fit_sizes])
        ax_c.errorbar(1.0 / n, c, yerr=np.where(np.isfinite(e), e, 0.0), fmt=marker + "-",
                      color=color, ms=3.5, lw=0.8, label=label)
    reference_line(ax_c, 0.5)
    reference_line(ax_c, 1.0)
    ax_c.set_xlim(0.0, 1.0 / min(th.cc_fit_sizes) * 1.08)
    ax_c.set_xlabel(r"$1/N$")
    ax_c.set_ylabel(r"fitted central charge $c$")
    ax_c.legend(loc="center right", fontsize=8)
    panel_labels([ax_t, ax_x, ax_c])
    return fig


if __name__ == "__main__":
    raise SystemExit(run_figure(SPEC, build))
