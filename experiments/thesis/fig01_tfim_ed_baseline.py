"""
Thesis Figure 1: exact-diagonalization baseline of the open transverse-field
Ising chain (Phase 1 data, N = 4 to 14).

    python experiments\\thesis\\fig01_tfim_ed_baseline.py

Panels
  (a) gap E1 - E0 against h, with the thermodynamic-limit gap 2|h - J| (h > J)
  (b) Binder cumulant of the even-parity ground state; curves cross near h_c
  (c) half-chain entropy S(N/2), open (solid) and periodic (dashed) chains

Validation against exact results
  * Free fermions: E0 = -sum_k Lambda_k and E1 - E0 = 2 Lambda_min, with
    Lambda_k the singular values of the N x N matrix with h on the diagonal and
    J on the superdiagonal, at every N and field (tolerance CFG.thesis.ff_tol).
  * Closed form at h = J: E1 - E0 = 4 J sin(pi / (2 (2N + 1))).
  * Boundary CFT: N (E1 - E0) at h_c, fitted as a + b/N + c/N^2 over
    CFG.thesis.gap_fit_sizes, gives a = pi.
  * Ordered-phase cat state: S(N/2) = ln 2 at h = CFG.thesis.cat_field
    (open and periodic).
  * Area law: S(N/2) saturates with N deep in the paramagnet.
  * Binder crossings lie below h_c and approach it monotonically with N.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from config import CFG
from experiments.thesis.figlib import FigureSpec, run_figure
from utils.plotting import (COLORS, mark_critical, panel_labels, reference_line,
                            size_color, thesis_figure)

SPEC = FigureSpec(
    number=1,
    slug="tfim_ed_baseline",
    chapter="Exact diagonalization baseline",
    phases=(1,),
    title="Gap, Binder cumulant and half-chain entropy of the open TFIM",
    caption=("Exact diagonalization of the open transverse-field Ising chain, N = 4 to 14. "
             "(a) Gap E1 - E0; the black line is the thermodynamic-limit gap 2(h - J) of the "
             "paramagnet. (b) Binder cumulant of the even-parity ground state. (c) Half-chain "
             "entanglement entropy for open (solid) and periodic (dashed) chains; the dashed "
             "line marks ln 2, the value of the ordered-phase cat state. The dotted line marks "
             "h_c = J."),
)


def free_fermion_levels(n_qubits: int, h: float, J: float) -> tuple:
    """Exact E0 and E1 - E0 of the open TFIM from the single-particle spectrum."""
    m = np.diag(np.full(n_qubits, float(h))) + np.diag(np.full(n_qubits - 1, float(J)), 1)
    lam = np.linalg.svd(m, compute_uv=False)
    return -lam.sum(), 2.0 * lam.min()


def crossing(x, y1, y2, window) -> float:
    """First sign change of y1 - y2 inside window, by linear interpolation (NaN if none)."""
    d = np.asarray(y1) - np.asarray(y2)
    lo, hi = window
    for k in range(len(x) - 1):
        if lo <= x[k] and x[k + 1] <= hi and d[k] * d[k + 1] < 0:
            return float(x[k] - d[k] * (x[k + 1] - x[k]) / (d[k + 1] - d[k]))
    return float("nan")


def field_grid(ctx, arrays: dict, name: str) -> np.ndarray:
    if "h" in arrays:
        return np.asarray(arrays["h"], dtype=float)
    grid = CFG.tfim.h_grid()
    n = len(np.asarray(arrays["E0"]))
    if n != len(grid):
        raise KeyError(f"{name}: no 'h' array and {n} fields differ from CFG.tfim.h_grid()")
    ctx.note(f"{name}: no 'h' array saved; used CFG.tfim.h_grid() ({n} fields)")
    return grid


def build(ctx):
    th, J, hc = CFG.thesis, CFG.tfim.J, CFG.tfim.h_critical
    sizes = list(th.ed_sizes)
    keys = ("E0", "gap", "binder", "S_half")
    open_, pbc = {}, {}
    for N in sizes:
        a = ctx.load(f"ed_tfim_N{N}", "phase1", require=keys)
        a["h"] = field_grid(ctx, a, f"ed_tfim_N{N}")
        open_[N] = a
        b = ctx.load(f"ed_tfim_pbc_N{N}", "phase1", require=("S_half",))
        b["h"] = field_grid(ctx, b, f"ed_tfim_pbc_N{N}")
        pbc[N] = b
    h = open_[sizes[0]]["h"]
    ctx.param(model="tfim", J=J, h_critical=hc, sizes=sizes, n_fields=len(h),
              h_min=float(h.min()), h_max=float(h.max()), boundaries=["open", "periodic"])

    # -- validation against exact results -----------------------------------
    dE0, dgap = [], []
    for N in sizes:
        ff = np.array([free_fermion_levels(N, x, J) for x in open_[N]["h"]])
        dE0.append(np.max(np.abs(open_[N]["E0"] - ff[:, 0])))
        dgap.append(np.max(np.abs(open_[N]["gap"] - ff[:, 1])))
    ctx.check_max("E0 equals free fermions (all N, all h)", dE0, th.ff_tol)
    ctx.check_max("E1 - E0 equals free fermions (all N, all h)", dgap, th.ff_tol)

    ic = [int(np.flatnonzero(np.isclose(open_[N]["h"], hc))[0]) for N in sizes]
    gap_c = np.array([open_[N]["gap"][i] for N, i in zip(sizes, ic)])
    closed = 4.0 * J * np.sin(np.pi / (2.0 * (2.0 * np.array(sizes) + 1.0)))
    ctx.check_close("gap at h_c equals 4J sin(pi/(2(2N+1)))", gap_c, closed, atol=th.ff_tol)

    fit_n = np.array(th.gap_fit_sizes, dtype=float)
    fit_y = np.array([gap_c[sizes.index(int(n))] * n for n in fit_n])
    a_inf = float(np.polyfit(1.0 / fit_n, fit_y, 2)[-1])
    ctx.check_close("N (E1 - E0) at h_c extrapolates to pi (boundary CFT)", a_inf, np.pi,
                    rtol=th.gap_fit_rtol, detail=f"quadratic fit in 1/N, N = {list(th.gap_fit_sizes)}")
    ctx.note("N (E1 - E0) at h_c: " + ", ".join(f"N={N}: {N * g:.4f}" for N, g in zip(sizes, gap_c))
             + f"; extrapolated {a_inf:.5f} (pi = {np.pi:.5f})")

    icat = [int(np.flatnonzero(np.isclose(open_[N]["h"], th.cat_field))[0]) for N in sizes]
    s_cat_open = np.array([open_[N]["S_half"][i] for N, i in zip(sizes, icat)])
    s_cat_pbc = np.array([pbc[N]["S_half"][i] for N, i in zip(sizes, icat)])
    ctx.check_close(f"S(N/2) = ln 2 at h = {th.cat_field} (open, cat state)", s_cat_open,
                    np.log(2), atol=th.cat_entropy_tol)
    ctx.check_close(f"S(N/2) = ln 2 at h = {th.cat_field} (periodic, cat state)", s_cat_pbc,
                    np.log(2), atol=th.cat_entropy_tol)

    iar = int(np.flatnonzero(np.isclose(h, th.area_law_field))[0])
    s_par = [open_[N]["S_half"][iar] for N in sizes]
    ctx.check_close(f"area law: S(N/2) saturates at h = {th.area_law_field} (two largest N)",
                    s_par[-1], s_par[-2], atol=th.area_law_tol)

    bsizes = list(th.binder_sizes)
    cross = [crossing(h, open_[a]["binder"], open_[b]["binder"], th.binder_window)
             for a, b in zip(bsizes[:-1], bsizes[1:])]
    ok = all(np.isfinite(cross)) and all(np.diff(cross) > 0) and max(cross) < hc
    ctx.check_true("Binder crossings below h_c, approaching it monotonically", ok,
                   ", ".join(f"({a},{b}): {c:.3f}" for a, b, c in zip(bsizes[:-1], bsizes[1:], cross)))
    for N in sizes:
        s = open_[N]["S_half"]
        k = int(np.argmin(np.diff(s) / np.diff(h)))
        kp = int(np.argmax(pbc[N]["S_half"]))
        ctx.note(f"N={N}: steepest drop of open S(N/2) at h = {0.5 * (h[k] + h[k + 1]):.3f}; "
                 f"periodic S(N/2) peaks at h = {h[kp]:.2f}; S(N/2, h_c) open {s[ic[0]]:.4f}")

    # -- figure --------------------------------------------------------------
    fig, (ax_g, ax_b, ax_s) = thesis_figure(ncols=3, aspect=0.36)
    for N in sizes:
        c = size_color(N)
        ax_g.plot(h, open_[N]["gap"], color=c, label=f"$N={N}$")
        ax_b.plot(h, open_[N]["binder"], color=c)
        ax_s.plot(h, open_[N]["S_half"], color=c)
        ax_s.plot(h, pbc[N]["S_half"], color=c, ls="--")
    hh = np.linspace(hc, h.max(), 50)
    ax_g.plot(hh, 2.0 * (hh - J), color=COLORS["black"], lw=0.8, label=r"$2(h-J)$")
    ax_g.set_xlabel(r"$h/J$")
    ax_g.set_ylabel(r"$(E_1-E_0)/J$")
    ax_g.legend(loc="upper left")
    ax_b.set_xlabel(r"$h/J$")
    ax_b.set_ylabel(r"Binder cumulant $U$")
    ax_s.set_xlabel(r"$h/J$")
    ax_s.set_ylabel(r"$S(N/2)$")
    reference_line(ax_s, np.log(2))
    ax_s.text(1.95, np.log(2) + 0.02, r"$\ln 2$", ha="right", va="bottom")
    ax_s.plot([], [], color=COLORS["black"], label="open")
    ax_s.plot([], [], color=COLORS["black"], ls="--", label="periodic")
    ax_s.legend(loc="lower left")
    for ax in (ax_g, ax_b, ax_s):
        mark_critical(ax, hc)
        ax.set_xlim(h.min(), h.max())
    panel_labels([ax_g, ax_b, ax_s])
    return fig


if __name__ == "__main__":
    raise SystemExit(run_figure(SPEC, build))
