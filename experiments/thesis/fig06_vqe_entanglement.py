"""
Thesis Figure 6: half-chain entanglement of the optimized VQE states across the
transition, and the entanglement capacity of the circuits (Phase 3 data).

    python experiments\\thesis\\fig06_vqe_entanglement.py

Panels (N = CFG.thesis.ent_size)
  (a) HVA, p = N/2 and N: S(N/2) against h, with the exact even-parity ground
      state (the HVA's symmetry sector)
  (b) HEA, p = N/2 and N, with the exact even and symmetry-broken states
  (c) largest half-chain Renyi-2 entropy reachable at depth p (maximized over
      parameters) against p, with the rank bound min(p, N/2) ln 2

Bounds for a depth-p circuit on an open chain (one two-qubit gate per layer
crosses any cut), checked for every state, cut and Renyi index:

    S_alpha(l) <= min(p, l, N - l) ln 2,

the Fannes-Audenaert continuity bound on |S_VQE - S_ref| at the half cut, and
the Eckart-Young bound F_GS <= sum of the 2^p largest Schmidt weights.

Validation
  * Rank bound for S and S_2 (all N, p, h, cuts, both ansaetze).
  * The stored Fannes-Audenaert bounds equal T ln(d-1) + H2(T) wherever 1 - F
    is resolved (the Phase 3 bound was saved without its fidelity floor).
  * Fannes-Audenaert bound at the half cut, recomputed with 1 - F floored at
    CFG.ent.fidelity_resolution as in the Phase 3 check (S is not Lipschitz
    near F = 1, so infidelities below double precision still move S), and
    evaluated to the precision with which 1 - F is known (relative allowance
    INFIDELITY_ULP / (1 - F)); Eckart-Young bound on F_GS.
  * The ED even-state half-chain entropy equals the Phase 1 value.
  * Capacity: S2_max = p ln 2 for every p < N/2 (the rank bound is reached),
    and S2_max <= min(p, N/2) ln 2 for every p.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from config import CFG
from experiments.thesis.figlib import FigureSpec, run_figure
from utils.plotting import (COLORS, SERIES_COLORS, ANSATZ_MARKERS, mark_critical, panel_labels,
                            reference_line, thesis_figure)

SPEC = FigureSpec(
    number=6,
    slug="vqe_entanglement",
    chapter="Entanglement of VQE states",
    phases=(1, 3),
    title="Half-chain entanglement of VQE states and the entanglement capacity of the circuits",
    caption=("Half-chain entanglement entropy of the optimized states for N = 12 against the field, "
             "at depth p = N/2 (dashed) and p = N (solid). (a) Hamiltonian variational ansatz with "
             "the exact even-parity ground state (black). (b) Hardware-efficient ansatz with the "
             "exact even-parity (black solid) and symmetry-broken (black dotted, drawn where the "
             "two parity sectors form a degenerate ground space) ground states. "
             "(c) Largest half-chain Renyi-2 entropy the circuits reach at depth p, with the rank "
             "bound min(p, N/2) ln 2 (dashed) and the largest ground-state entropy over all fields "
             "(dotted)."),
)

ANSATZE = (("hva", "HVA"), ("hea", "HEA"))
LN2 = np.log(2.0)
# 1 - F is known only to a few units of double precision (2.2e-16). Near F = 1
# the Fannes-Audenaert bound B is steep, so it is evaluated only to relative
# accuracy (elasticity d ln B / d ln(1 - F) < 1/2, so this is conservative):
#     delta B / B <= INFIDELITY_ULP / (1 - F).
INFIDELITY_ULP = 4.0 * np.finfo(float).eps


def audenaert_bound(infidelity, n_qubits: int):
    """
    Fannes-Audenaert bound at the half cut, d = 2^(N/2), T = sqrt(1 - F):
    T ln(d - 1) + H2(T) for T < 1 - 1/d, ln d otherwise (H2 in nats).
    """
    d = 2.0 ** (n_qubits // 2)
    T = np.sqrt(np.clip(np.asarray(infidelity, dtype=float), 0.0, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        h2 = np.where((T > 0) & (T < 1), -T * np.log(T) - (1 - T) * np.log1p(-T), 0.0)
    return np.where(T < 1 - 1 / d, T * np.log(d - 1) + h2, np.log(d))


def build(ctx):
    th, tol = CFG.thesis, CFG.ent.bound_tol
    sizes, Np = list(th.vqe_sizes), th.ent_size
    ent = {a: {N: ctx.load(f"ent_{a}_N{N}", "phase3",
                           require=("S", "S2", "S_ref", "fidelity_gs", "audenaert",
                                    "rank_fidelity_bound", "depths", "h"))
               for N in sizes} for a, _ in ANSATZE}
    ed = {N: ctx.load(f"ent_ed_N{N}", "phase3", require=("S_even", "S_broken", "h", "ground_space"))
          for N in sizes}
    ph1 = {N: ctx.load(f"ed_tfim_N{N}", "phase1", require=("S_half",)) for N in sizes}
    mx = {a: {N: ctx.load(f"maxent_{a}_N{N}", "phase3", require=("S2_max", "depths")) for N in sizes}
          for a, _ in ANSATZE}
    h = np.asarray(ed[Np]["h"])
    ctx.param(plot_size=Np, sizes=sizes, depths_drawn=[Np // 2, Np], bound_tol=tol)

    # -- validation ------------------------------------------------------------
    rank_slack, fa_slack, ey_slack, fa_formula = [], [], [], []
    fa_raw, fa_zero, fa_worst = 0, 0, None
    floor = CFG.ent.fidelity_resolution
    for a, _ in ANSATZE:
        for N in sizes:
            d = ent[a][N]
            p = np.asarray(d["depths"], dtype=int)
            ell = np.arange(1, N)
            cap = np.minimum(np.minimum(p[:, None], ell[None, :]), N - ell[None, :]) * LN2  # [p, l]
            for key in ("S", "S2"):
                rank_slack.append(np.max(np.asarray(d[key]) - cap[:, None, :]))
            half = N // 2 - 1
            dS = np.abs(np.asarray(d["S"])[:, :, half] - np.asarray(d["S_ref"])[:, :, half])
            stored = np.asarray(d["audenaert"])
            infid = 1.0 - np.asarray(d["fidelity_gs"])
            T_sat = 1.0 - 1.0 / 2.0 ** (N // 2)
            resolved = (infid > 100 * floor) & (np.sqrt(np.clip(infid, 0, 1)) < T_sat)
            if resolved.any():
                ref = audenaert_bound(infid[resolved], N)
                rel = np.abs(stored[resolved] - ref) / ref
                fa_formula.append(np.max(rel - INFIDELITY_ULP / infid[resolved]))
            fa_zero += int(np.sum(stored == 0))
            x = np.maximum(infid, floor)              # Phase 3 floor on 1 - F
            bnd = audenaert_bound(x, N)
            allowed = bnd * (1.0 + INFIDELITY_ULP / x)
            excess = dS - allowed
            fa_slack.append(np.max(excess))
            fa_raw += int(np.sum(dS - stored > tol))
            k = np.unravel_index(np.argmax(dS / allowed), dS.shape)
            cand = (float(dS[k] / allowed[k]), a, N, int(p[k[0]]), float(d["h"][k[1]]), float(x[k]),
                    float(dS[k]), float(bnd[k]))
            if fa_worst is None or cand[0] > fa_worst[0]:
                fa_worst = cand
            ey_slack.append(np.max(np.asarray(d["fidelity_gs"]) - d["rank_fidelity_bound"]))
    ctx.check_max("rank bound S_alpha(l) <= min(p, l, N-l) ln 2 (S and S_2, all states, cuts)",
                  rank_slack, tol)
    ctx.check_max("stored Fannes-Audenaert bound equals T ln(d-1) + H2(T) where 1 - F is resolved",
                  fa_formula, 1e-8, detail="relative difference beyond the double-precision allowance of 1 - F; "
                         "points with 1 - F > 100 x floor and T < 1 - 1/d")
    ex, a_w, n_w, p_w, h_w, x_w, ds_w, b_w = fa_worst
    ctx.check_max("Fannes-Audenaert bound on |S_VQE - S_ref| at the half cut "
                  "(1 - F floored at the Phase 3 resolution, known to double precision)",
                  fa_slack, tol,
                  detail=f"closest point {a_w.upper()} N={n_w} p={p_w} h={h_w:g}: 1-F={x_w:.2e}, "
                         f"|dS|={ds_w:.3e}, bound={b_w:.3e}")
    ctx.note(f"Fannes-Audenaert: {fa_zero} states with F_GS = 1 to double precision (stored bound 0); "
             f"{fa_raw} states exceed the unfloored stored bound by more than {tol:g}, all resolved by "
             f"the Phase 3 floor 1 - F >= {floor:g}")
    ctx.check_max("Eckart-Young bound F_GS <= sum of the 2^p largest Schmidt weights", ey_slack, tol)
    dev = [np.max(np.abs(np.asarray(ed[N]["S_even"])[:, N // 2 - 1] - ph1[N]["S_half"])) for N in sizes]
    ctx.check_max("ED even-state S(N/2) equals the Phase 1 value", dev, th.identity_tol)

    reach, over = [], []
    for a, _ in ANSATZE:
        for N in sizes:
            p = np.asarray(mx[a][N]["depths"], dtype=int)
            s2 = np.asarray(mx[a][N]["S2_max"])
            low = p < N // 2
            if low.any():
                reach.append(np.max(np.abs(s2[low] - p[low] * LN2)))
            over.append(np.max(s2 - np.minimum(p, N // 2) * LN2))
    ctx.check_max("capacity: S2_max = p ln 2 for every p < N/2", reach, th.capacity_tol)
    ctx.check_max("capacity: S2_max <= min(p, N/2) ln 2 for every p", over, tol)

    ih = int(np.argmin(np.abs(h - CFG.tfim.h_critical)))
    for a, name in ANSATZE:
        d = ent[a][Np]
        p = list(np.asarray(d["depths"], dtype=int))
        k = p.index(Np)
        ctx.note(f"{name} N={Np}, p=N: S(N/2) at h_c {d['S'][k, ih, Np // 2 - 1]:.4f}, at h={h[0]:g} "
                 f"{d['S'][k, 0, Np // 2 - 1]:.4f}; max |S - S_ref| (half cut) "
                 f"{np.max(np.abs(d['S'][k, :, Np // 2 - 1] - d['S_ref'][k, :, Np // 2 - 1])):.2e}")
    ctx.note(f"ED N={Np}: S(N/2) even at h_c {ed[Np]['S_even'][ih, Np // 2 - 1]:.4f}; symmetry broken "
             f"at h={h[0]:g} {ed[Np]['S_broken'][0, Np // 2 - 1]:.2e}")
    for a, name in ANSATZE:
        p = np.asarray(mx[a][Np]["depths"], dtype=int)
        short = [int(q) for q, s in zip(p, mx[a][Np]["S2_max"]) if s < min(q, Np // 2) * LN2 - 1e-3]
        ctx.note(f"{name} N={Np}: maximal S2 below the rank bound at p = {short or 'none'}")

    # -- figure --------------------------------------------------------------------
    fig, axes = thesis_figure(ncols=3, aspect=0.36)
    half = Np // 2 - 1
    for ax, (a, name) in zip(axes[:2], ANSATZE):
        d = ent[a][Np]
        p = list(np.asarray(d["depths"], dtype=int))
        ax.plot(h, ed[Np]["S_even"][:, half], color=SERIES_COLORS["ED"], lw=0.9, label="ED, even parity")
        if a == "hea":
            twofold = np.asarray(ed[Np]["ground_space"]).all(axis=1)    # both sectors degenerate
            ax.plot(h[twofold], ed[Np]["S_broken"][twofold, half], ":", color=SERIES_COLORS["ED"],
                    lw=0.9, label="ED, symmetry broken")
        for q, ls in ((Np // 2, "--"), (Np, "-")):
            ax.plot(h, d["S"][p.index(q), :, half], ls, color=SERIES_COLORS[name], label=f"{name}, $p={q}$")
        reference_line(ax, LN2, color=COLORS["grey"], ls=":")
        mark_critical(ax)
        ax.set_xlim(h.min(), h.max())
        ax.set_xlabel(r"$h/J$")
        ax.set_ylabel(r"$S(N/2)$")
        ax.set_title(f"{name}, $N={Np}$", loc="right")
        ax.legend(loc="lower left" if a == "hva" else "upper right")
    ax = axes[2]
    pp = np.arange(1, Np + 1)
    ax.plot(pp, np.minimum(pp, Np // 2) * LN2, "--", color=COLORS["black"], lw=0.8,
            label=r"$\min(p, N/2)\ln 2$")
    for a, name in ANSATZE:
        p = np.asarray(mx[a][Np]["depths"], dtype=int)
        ax.plot(p, mx[a][Np]["S2_max"], ANSATZ_MARKERS[name], color=SERIES_COLORS[name], ms=3.5,
                mfc="none" if a == "hea" else None, label=f"{name}, maximal $S_2$")
    smax = float(np.max(ed[Np]["S_even"][:, half]))
    reference_line(ax, smax, color=COLORS["grey"], ls=":")
    ax.text(Np, smax + 0.08, "largest ground-state $S$", ha="right", va="bottom")
    ax.set_xlabel("depth $p$")
    ax.set_ylabel(r"$S_2(N/2)$, maximized")
    ax.set_title(f"$N={Np}$", loc="right")
    ax.legend(loc="center right")
    panel_labels(axes)
    return fig


if __name__ == "__main__":
    raise SystemExit(run_figure(SPEC, build))
