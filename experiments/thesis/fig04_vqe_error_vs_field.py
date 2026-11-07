"""
Thesis Figure 4: VQE accuracy across the phase transition, and the
symmetry-breaking floor of the hardware-efficient ansatz (Phase 2 data).

    python experiments\\thesis\\fig04_vqe_error_vs_field.py

Panels
  (a) HVA: relative error against h for N in CFG.thesis.vqe_plot_sizes,
      p = N/2 (dashed) and p = N (solid)
  (b) HEA: the same, with the parity-splitting bound of the p = N states (dotted)
  (c) parity <P> = <prod X_i> of the optimized HEA and HVA states at p = N

Parity-splitting bound. With E_even <= E_odd the lowest energies of the two
spin-flip sectors and w_odd = (1 - <P>)/2 the weight of a state in the odd
sector, every state satisfies

    E - E0 >= w_odd (E_odd - E_even),

so eps >= w_odd (E_odd - E_even)/|E0|.

Validation (every file, every field)
  * The even sector holds the ground state: E_even <= E_odd (Phase 1 sectors).
  * Every HVA and HEA state obeys the parity-splitting bound.
  * The HVA conserves parity exactly: <P> = 1.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from config import CFG
from experiments.thesis.figlib import FigureSpec, run_figure
from utils.plotting import (COLORS, SERIES_COLORS, mark_critical, panel_labels, size_color,
                            thesis_figure)

SPEC = FigureSpec(
    number=4,
    slug="vqe_error_vs_field",
    chapter="Noiseless VQE",
    phases=(1, 2),
    title="VQE energy error across the transition and the parity-splitting floor of the HEA",
    caption=("Relative energy error of the optimized circuits against the field for N = 8 and 12 "
             "at depth p = N/2 (dashed) and p = N (solid). (a) Hamiltonian variational ansatz. "
             "(b) Hardware-efficient ansatz; dotted lines are the parity-splitting lower bound "
             "w_odd (E_odd - E_even)/|E0| of the p = N states, which the symmetry-broken HEA "
             "states sit on in the ordered phase. (c) Spin-flip parity of the p = N states: the "
             "HVA conserves it, the HEA breaks it for h < J."),
)

ANSATZE = ("hva", "hea")
FLOOR = 1e-16


def sector_gaps(ed: dict) -> tuple:
    """(E_even, E_odd) per field from the Phase 1 sector energies."""
    labels = np.asarray(ed["sector_labels"]).ravel()
    se = np.asarray(ed["sector_energies"])
    return se[:, int(np.flatnonzero(labels == 1)[0]), 0], se[:, int(np.flatnonzero(labels == -1)[0]), 0]


def build(ctx):
    th = CFG.thesis
    sizes, plot_sizes = list(th.vqe_sizes), list(th.vqe_plot_sizes)
    keys = ("h", "energy", "E0", "rel_error", "parity", "fidelity", "fidelity_gs")
    data = {a: {N: {p: ctx.load(f"vqe_{a}_N{N}_p{p}", "phase2", require=keys)
                    for p in range(1, N + 1)} for N in sizes} for a in ANSATZE}
    ed = {N: ctx.load(f"ed_tfim_N{N}", "phase1", require=("sector_energies", "sector_labels", "E0"))
          for N in sizes}
    h = np.asarray(data["hva"][sizes[0]][1]["h"])
    ctx.param(sizes=sizes, plot_sizes=plot_sizes, depths="N/2 and N in the figure, all in the checks")

    # -- validation ------------------------------------------------------------
    order = [np.max(e - o) for e, o in (sector_gaps(ed[N]) for N in sizes)]
    ctx.check_max("even sector holds the ground state: E_even <= E_odd", order, th.identity_tol)

    slack, worst = [], None
    bound = {}
    for a in ANSATZE:
        for N in sizes:
            e_even, e_odd = sector_gaps(ed[N])
            for p, d in data[a][N].items():
                b = 0.5 * (1.0 - np.asarray(d["parity"])) * (e_odd - e_even) / np.abs(d["E0"])
                bound[a, N, p] = b
                s = float(np.min(np.asarray(d["rel_error"]) - b))
                slack.append(s)
                if worst is None or s < worst[0]:
                    worst = (s, a, N, p)
    ctx.check_min("parity-splitting bound eps >= w_odd (E_odd - E_even)/|E0| (all states)", slack,
                  -th.variational_tol,
                  detail=f"tightest: {worst[1].upper()} N={worst[2]} p={worst[3]}")
    par = max(np.max(np.abs(np.asarray(d["parity"]) - 1.0))
              for N in sizes for d in data["hva"][N].values())
    ctx.check_max("HVA conserves parity: <P> = 1", par, 1e-10)

    for N in sizes:
        d = data["hea"][N][N]
        b = bound["hea", N, N]
        sel = (b > 1e-12) & (np.abs(np.asarray(d["parity"])) < 0.5)      # symmetry-broken states
        if sel.any():
            r = np.asarray(d["rel_error"])[sel] / b[sel]
            ctx.note(f"HEA N={N}, p=N: eps / parity bound min {r.min():.3f}, median {np.median(r):.3f} "
                     f"over the {sel.sum()} symmetry-broken states (|<P>| < 0.5)")
        ctx.note(f"HEA N={N}, p=N, h={h[0]:g}: <P> = {d['parity'][0]:.3f}, F = {d['fidelity'][0]:.3f}, "
                 f"F_GS = {d['fidelity_gs'][0]:.4f}")

    # -- figure --------------------------------------------------------------------
    fig, axes = thesis_figure(ncols=3, aspect=0.36)
    for ax, a, name in zip(axes[:2], ANSATZE, ("HVA", "HEA")):
        for N in plot_sizes:
            c = size_color(N)
            for p, ls in ((N // 2, "--"), (N, "-")):
                ax.semilogy(h, np.maximum(data[a][N][p]["rel_error"], FLOOR), ls, color=c,
                            label=f"$N={N}$, $p={p}$")
            if a == "hea":
                ax.semilogy(h, np.maximum(bound[a, N, N], FLOOR), ":", color=c, lw=0.9)
        ax.set_xlabel(r"$h/J$")
        ax.set_ylabel(r"$\epsilon$")
        ax.set_ylim(1e-9, 1.0)
        ax.set_title(name, loc="right")
        mark_critical(ax)
        ax.set_xlim(h.min(), h.max())
    axes[0].legend(loc="lower left")
    axes[1].plot([], [], ":", color=COLORS["black"], lw=0.9, label="parity bound, $p=N$")
    axes[1].legend(loc="upper left", handles=axes[1].get_legend_handles_labels()[0][-1:])
    ax = axes[2]
    for N in plot_sizes:
        ax.plot(h, data["hea"][N][N]["parity"], "-", color=size_color(N), label=f"$N={N}$, HEA")
    ax.plot(h, data["hva"][plot_sizes[-1]][plot_sizes[-1]]["parity"], "-", color=SERIES_COLORS["HVA"],
            lw=0.8, label="HVA (all $N$)")
    ax.set_xlabel(r"$h/J$")
    ax.set_ylabel(r"parity $\langle P\rangle$, $p=N$")
    ax.set_ylim(-0.05, 1.08)
    mark_critical(ax)
    ax.set_xlim(h.min(), h.max())
    ax.legend(loc="center right")
    panel_labels(axes)
    return fig


if __name__ == "__main__":
    raise SystemExit(run_figure(SPEC, build))
