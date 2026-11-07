"""
Thesis Figure 3: accuracy of noiseless VQE against circuit depth (Phase 2 data).

    python experiments\\thesis\\fig03_vqe_error_vs_depth.py

Panels
  (a) HVA: relative energy error against depth p at h = CFG.thesis.vqe_field
  (b) HEA: the same
  (c) required depth p* (smallest p with error below CFG.vqe.target_rel_error)
      against h for the largest N in CFG.thesis.vqe_plot_sizes; crosses at p = N + 1 mark
      fields where no depth p <= N reaches the target

The relative error is

    eps = (E_VQE - E0) / |E0|.

Validation (every file: N = 4 to 12, p = 1 to N, all 39 fields)
  * Variational principle: E_VQE >= E0.
  * The stored error equals (E_VQE - E0)/|E0| from the stored energies.
  * The stored E0 equals the Phase 1 ED ground energy.
  * N = 4, p = 4: both ansaetze are exact at every field.
  * HVA: the error never increases with p (depth warm start).
  * Periodic HVA at p = N/2 is exact (Ho and Hsieh).
  * Fidelities lie in [0, 1] and F_GS >= F.
  * p*(h) equals the Phase 2 report (required_depth_vs_field) at every field,
    size and ansatz.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from matplotlib.ticker import MaxNLocator

from config import CFG
from experiments.thesis.figlib import FigureSpec, field_index, run_figure
from utils.plotting import (COLORS, SERIES_COLORS, ANSATZ_MARKERS, mark_critical, panel_labels, reference_line,
                            size_color, thesis_figure)

SPEC = FigureSpec(
    number=3,
    slug="vqe_error_vs_depth",
    chapter="Noiseless VQE",
    phases=(2,),
    title="VQE energy error against circuit depth, and the depth needed for 0.1 % accuracy",
    caption=("Relative energy error of the optimized circuits at the critical field h = J against "
             "the depth p, for N = 4 to 12: (a) Hamiltonian variational ansatz, (b) hardware-"
             "efficient ansatz. The dashed line marks the target error 10^-3. (c) Smallest depth "
             "reaching the target against the field for N = 12; crosses at p = N + 1 mark fields "
             "where no depth p <= N reaches it (the HVA in the ordered phase)."),
)

ANSATZE = ("hva", "hea")
FLOOR = 1e-16                     # errors below machine precision are drawn here


def build(ctx):
    th, tgt = CFG.thesis, CFG.vqe.target_rel_error
    sizes = list(th.vqe_sizes)
    keys = ("h", "energy", "E0", "rel_error", "fidelity", "fidelity_gs")
    data = {a: {N: {p: ctx.load(f"vqe_{a}_N{N}_p{p}", "phase2", require=keys)
                    for p in range(1, N + 1)} for N in sizes} for a in ANSATZE}
    ed = {N: ctx.load(f"ed_tfim_N{N}", "phase1", require=("E0", "h")) for N in sizes}
    pbc = ctx.load("vqe_hva_pbc_check", "phase2", require=("sizes", "fields", "rel_error", "fidelity"))
    report = ctx.load_json("phase2_report_data.json", "phase2")
    h = np.asarray(data["hva"][sizes[0]][1]["h"])
    ih = field_index(h, th.vqe_field)
    ctx.param(sizes=sizes, depths="1..N", field=th.vqe_field, target_rel_error=tgt,
              pstar_fields=list(th.vqe_pstar_fields), n_fields=len(h))

    # -- validation ------------------------------------------------------------
    var, cons, e0d, fid_lo, fid_hi, gs_minus = [], [], [], [], [], []
    for a in ANSATZE:
        for N in sizes:
            for p, d in data[a][N].items():
                E, E0 = np.asarray(d["energy"]), np.asarray(d["E0"])
                eps = (E - E0) / np.abs(E0)
                var.append(eps.min())
                cons.append(np.max(np.abs(eps - d["rel_error"])))
                e0d.append(np.max(np.abs(E0 - ed[N]["E0"])))
                fid_lo.append(min(np.min(d["fidelity"]), np.min(d["fidelity_gs"])))
                fid_hi.append(max(np.max(d["fidelity"]), np.max(d["fidelity_gs"])))
                gs_minus.append(np.max(np.asarray(d["fidelity"]) - d["fidelity_gs"]))
    ctx.check_min("variational principle: E_VQE >= E0 (all files, all fields)", var,
                  -th.variational_tol, detail="minimum relative error")
    ctx.check_max("stored error equals (E_VQE - E0)/|E0|", cons, 1e-12)
    ctx.check_max("stored E0 equals the Phase 1 ED ground energy", e0d, th.identity_tol)
    ctx.check_true("fidelities in [0, 1] and F_GS >= F",
                   min(fid_lo) >= -1e-12 and max(fid_hi) <= 1 + 1e-12 and max(gs_minus) <= 1e-12,
                   f"range [{min(fid_lo):.3g}, {max(fid_hi):.12g}], max(F - F_GS) {max(gs_minus):.2e}")
    if 4 in sizes:
        e4 = [np.max(data[a][4][4]["rel_error"]) for a in ANSATZE]
        ctx.check_max("N = 4, p = 4 exact at every field (HVA and HEA)", e4, th.exact_depth_tol)
    mono = max(np.max(np.asarray(data["hva"][N][p + 1]["rel_error"]) - data["hva"][N][p]["rel_error"])
               for N in sizes for p in range(1, N))
    ctx.check_max("HVA error non-increasing in p (depth warm start)", mono, th.variational_tol)
    ctx.check_max("periodic HVA at p = N/2 exact (Ho and Hsieh)", pbc["rel_error"], th.pbc_exact_tol,
                  detail=f"N = {[int(n) for n in pbc['sizes']]}, h = {[float(x) for x in pbc['fields']]}")

    # -- required depth p*(h) ---------------------------------------------------------
    pstar = {}
    for a in ANSATZE:
        for N in sizes:
            err = np.array([data[a][N][p]["rel_error"] for p in range(1, N + 1)])   # [p, h]
            ok = err < tgt
            pstar[a, N] = np.where(ok.any(axis=0), np.argmax(ok, axis=0) + 1.0, np.nan)
    agree, compared = True, 0
    for a in ANSATZE:
        for N in sizes:
            rep = report.get("sweeps", {}).get(f"{a}_N{N}", {}).get("required_depth_vs_field")
            if rep is None or len(rep) != len(h):
                agree = False
                continue
            r = np.array([np.nan if v is None else float(v) for v in rep])
            agree &= bool(np.array_equal(np.isnan(r), np.isnan(pstar[a, N]))
                          and np.allclose(r[~np.isnan(r)], pstar[a, N][~np.isnan(r)]))
            compared += 1
    ctx.check_true("p*(h) equals the Phase 2 report (required_depth_vs_field)",
                   agree and compared == 2 * len(sizes), f"{compared} sweeps compared, every field")
    for a in ANSATZE:
        for hf in th.vqe_pstar_fields:
            k = field_index(h, hf)
            ctx.note(f"{a.upper()} p* at h = {hf}: " + ", ".join(
                f"N={N}: {'-' if np.isnan(pstar[a, N][k]) else int(pstar[a, N][k])}" for N in sizes))
        win = np.abs(h - CFG.tfim.h_critical) <= 0.25 + 1e-9
        parts = []
        for N in sizes:
            ps, hw = pstar[a, N][win], h[win]
            if np.isnan(ps).any():
                parts.append(f"N={N}: not reached at h = {', '.join(f'{x:g}' for x in hw[np.isnan(ps)])}")
            else:
                parts.append(f"N={N}: {int(ps.max())} (at h = {hw[int(np.argmax(ps))]:g})")
        ctx.note(f"{a.upper()} hardest p* for 0.75 <= h <= 1.25: " + "; ".join(parts))
    for a in ANSATZE:
        ctx.note(f"{a.upper()} error at p = N, h = {th.vqe_field}: " + ", ".join(
            f"N={N}: {data[a][N][N]['rel_error'][ih]:.2e}" for N in sizes))

    # -- figure --------------------------------------------------------------------
    fig, axes = thesis_figure(ncols=3, aspect=0.36)
    for ax, a, name in zip(axes[:2], ANSATZE, ("HVA", "HEA")):
        for N in sizes:
            p = np.arange(1, N + 1)
            e = np.array([data[a][N][q]["rel_error"][ih] for q in p])
            ax.semilogy(p, np.maximum(e, FLOOR), ANSATZ_MARKERS[name] + "-", color=size_color(N),
                        ms=3, label=f"$N={N}$")
        reference_line(ax, tgt)
        ax.set_xlabel("depth $p$")
        ax.set_ylabel(r"$\epsilon = (E_{\mathrm{VQE}}-E_0)/|E_0|$")
        ax.set_ylim(FLOOR / 3, 1.0)
        ax.set_title(f"{name}, $h/J={th.vqe_field:g}$", loc="right")
    axes[0].legend(loc="lower right", ncol=1)
    ax = axes[2]
    Nc = max(th.vqe_plot_sizes)
    for a, name in zip(ANSATZE, ("HVA", "HEA")):
        ps = pstar[a, Nc]
        ax.plot(h, ps, ANSATZ_MARKERS[name] + ("-" if a == "hva" else "--"), color=SERIES_COLORS[name],
                ms=2.5, lw=0.9, mfc="none" if a == "hea" else None, label=name)
        miss = np.isnan(ps)
        if miss.any():
            ax.plot(h[miss], np.full(miss.sum(), Nc + 1.0), "x", color=SERIES_COLORS[name], ms=3)
    ax.plot([], [], "x", color=COLORS["black"], ms=3, label=f"not reached ($p \\leq {Nc}$)")
    ax.axhline(Nc, color=COLORS["grey"], lw=0.5, ls=":")
    ax.set_xlabel(r"$h/J$")
    ax.set_ylabel(r"required depth $p^*$")
    ax.set_xlim(h.min(), h.max())
    ax.set_ylim(0, Nc + 2.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    mark_critical(ax)
    ax.set_title(rf"$N={Nc}$, $\epsilon < 10^{{-3}}$", loc="right")
    ax.legend(loc="lower left")
    panel_labels(axes)
    return fig


if __name__ == "__main__":
    raise SystemExit(run_figure(SPEC, build))
