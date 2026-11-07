"""
Thesis Figure 12: the folding bias of zero-noise extrapolation depends on the
noise channel (Phase 5, ideal readout, so only gate noise is extrapolated).

    python experiments\\thesis\\fig12_folding_bias.py

Panels
  (a) HVA and (b) HEA at the reference point (N = CFG.mitigation.plot_size,
      p = N/2, h = CFG.mitigation.plot_field): unmitigated error (thin) and
      Richardson residual (thick) against p for the four channels
  (c) median folding-bias ratio |E_R - E_ideal| / |E(1) - E_ideal| at the
      weakest noise, over all sizes, depths and fields, global and local folding

First-order folding bias, with A (B) the effect of errors placed after
(before) each gate:

    E(lambda) - E_ideal = p [ (A - B)/2 + lambda (A + B)/2 ] + O(p^2).

Depolarizing noise commutes with every gate (A = B), so the constant term
vanishes and Richardson extrapolation removes the error to O(p^3); amplitude
damping and dephasing leave a residual of order p.

Validation (all 72 ZNE files)
  * Depolarizing weak-noise orders: unmitigated, linear and Richardson errors
    scale as p^1, p^2, p^3 (median log-log slopes within CFG.mitigation.order_tol).
  * Depolarizing has no first-order folding bias: median ratio below 1e-4.
  * Folding-bias medians and 5 to 95 % ranges (global folding, every channel)
    reproduce the Phase 5 report.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from config import CFG
from experiments.thesis.figlib import (FigureSpec, error_budget_phase5, extrapolation_weights, field_index,
                                       run_figure)
from utils.plotting import (CHANNEL_COLORS, CHANNEL_LABELS, COLORS, label_zero_strength, panel_labels,
                            thesis_figure, zero_strength_positions)

SPEC = FigureSpec(
    number=12,
    slug="folding_bias",
    chapter="Error mitigation",
    phases=(5,),
    title="The folding bias of zero-noise extrapolation depends on the noise channel",
    caption=("Zero-noise extrapolation with ideal readout, so only gate noise is extrapolated. "
             "(a) HVA and (b) HEA at N = 8, p = 4, h = J: unmitigated error (thin) and residual "
             "after Richardson extrapolation (thick) for the four noise models. Depolarizing noise "
             "commutes with the gates, so folding scales its first-order error exactly and the "
             "residual falls as p^3; amplitude damping and dephasing leave a residual of order p. "
             "(c) Median ratio of residual to unmitigated error at p = 1e-5 over all sizes, depths "
             "and fields (360 states per channel), for global (plain) and local (hatched) folding; "
             "hardware = the hardware-like model."),
)

ANSATZE = (("hva", "HVA"), ("hea", "HEA"))
FLOOR = 1e-16


def build(ctx):
    th, mt = CFG.thesis, CFG.mitigation
    sizes = list(CFG.system.sizes_noisy)
    channels = list(mt.channels)
    keys = ("E_true", "E_ideal", "E0", "strengths", "scales", "foldings", "h", "n_1q", "n_2q")
    files = {(a, N, p, c): ctx.load(f"zne_{a}_N{N}_p{p}_{c}", "phase5", require=keys)
             for a, _ in ANSATZE for N in sizes for p in CFG.system.depths(N, noisy=True) for c in channels}
    report = ctx.load_json("phase5_report_data.json", "phase5")
    first = next(iter(files.values()))
    s = np.asarray(first["strengths"], dtype=float)
    scales = np.asarray(first["scales"], dtype=float)
    folds = [str(f) for f in first["foldings"]]
    wR, wL = extrapolation_weights(scales, len(scales) - 1), extrapolation_weights(scales, 1)
    Np = mt.plot_size
    ih = field_index(first["h"], mt.plot_field)
    i_weak = 1                                          # weakest positive strength
    ctx.param(channels=channels, scales=scales.tolist(), weak_strength=float(s[i_weak]),
              readout="ideal (E_true)", reference=f"N={Np}, p={Np // 2}, h={mt.plot_field:g}")

    def errors(d, fold):
        Ei, E0 = np.asarray(d["E_ideal"]), np.abs(np.asarray(d["E0"]))
        E = np.asarray(d["E_true"])[:, :, fold, :]                       # [strength, field, scale]
        return {"raw": np.abs(E[..., 0] - Ei) / E0, "linear": np.abs(E @ wL - Ei) / E0,
                "richardson": np.abs(E @ wR - Ei) / E0}

    # -- validation ------------------------------------------------------------
    # weak-noise orders under depolarizing noise, from consecutive strengths with lambda <= regime_lambda
    slopes = {"raw": [], "linear": [], "richardson": []}
    for (a, N, p, c), d in files.items():
        if c != "depolarizing":
            continue
        lam = error_budget_phase5(c, s, int(d["n_1q"]), int(d["n_2q"]))
        e = errors(d, folds.index(mt.main_folding))
        ks = [k for k in range(1, len(s) - 1) if lam[k + 1] <= mt.regime_lambda]
        for name, v in e.items():
            for k in ks:
                y0, y1 = v[k], v[k + 1]
                ok = (y0 > 1e-12) & (y1 > 1e-12)        # above double-precision noise of the energies
                slopes[name] += list(np.log(y1[ok] / y0[ok]) / np.log(s[k + 1] / s[k]))
    med = {k: float(np.median(v)) if v else float("nan") for k, v in slopes.items()}
    ctx.check_close("depolarizing weak-noise orders p^1, p^2, p^3 (unmitigated, linear, Richardson)",
                    [med["raw"], med["linear"], med["richardson"]], [1.0, 2.0, 3.0], atol=mt.order_tol,
                    detail=f"medians {med['raw']:.3f}, {med['linear']:.3f}, {med['richardson']:.3f} over "
                           f"{len(slopes['raw'])}, {len(slopes['linear'])}, {len(slopes['richardson'])} slopes")

    ratio = {}
    for c in channels:
        for fname in folds:
            r = []
            for (a, N, p, cc), d in files.items():
                if cc != c:
                    continue
                e = errors(d, folds.index(fname))
                r.append(e["richardson"][i_weak] / e["raw"][i_weak])
            ratio[c, fname] = np.concatenate(r)
    g = mt.main_folding
    ctx.check_max("depolarizing: no first-order folding bias (median ratio)",
                  float(np.median(ratio["depolarizing", g])), 1e-4)
    rep = report.get("folding_bias_by_channel", {})
    mine = [f(ratio[c, g]) for c in channels for f in (np.median, lambda v: np.percentile(v, 5),
                                                        lambda v: np.percentile(v, 95))]
    theirs = [rep.get(c, {}).get(k, np.nan) for c in channels for k in ("folding_bias_ratio_median", "p5", "p95")]
    ctx.check_close("folding-bias medians and 5 to 95 % ranges reproduce the Phase 5 report",
                    mine, theirs, rtol=th.report_rtol,
                    detail=f"{ratio[channels[0], g].size} states per channel at p = {s[i_weak]:g}")
    for c in channels:
        lg = ratio[c, "local"] / np.where(ratio[c, g] > 0, ratio[c, g], np.nan)
        ctx.note(f"{c}: median folding-bias ratio {np.median(ratio[c, g]):.3e} (global), "
                 f"{np.median(ratio[c, 'local']):.3e} (local); local/global median {np.nanmedian(lg):.4f}")
    lgall = np.concatenate([ratio[c, "local"] / ratio[c, g] for c in channels])
    ctx.note(f"local/global residual ratio, all channels: median {np.nanmedian(lgall):.4f} "
             f"(report {report.get('local_over_global_residual', {}).get('median')})")

    # -- figure --------------------------------------------------------------------
    x, x0 = zero_strength_positions(s)
    fig, axes = thesis_figure(ncols=3, aspect=0.38)
    for ax, (a, name) in zip(axes[:2], ANSATZE):
        for c in channels:
            e = errors(files[a, Np, Np // 2, c], folds.index(g))
            ax.loglog(x, np.maximum(e["raw"][:, ih], FLOOR), "-", color=CHANNEL_COLORS[c], lw=0.6)
            ax.loglog(x, np.maximum(e["richardson"][:, ih], FLOOR), "o-", color=CHANNEL_COLORS[c], lw=1.2,
                      ms=2.5, label=CHANNEL_LABELS[c])
        ax.set_ylim(1e-14, 2.0)
        ax.set_xlabel("gate-error strength $p$")
        ax.set_ylabel(r"$|E - E_{\mathrm{ideal}}|/|E_0|$")
        ax.set_title(f"{name}, $N={Np}$, $p={Np // 2}$", loc="right")
        label_zero_strength(ax, x0)
    axes[0].legend(loc="lower right")
    hs = [axes[1].plot([], [], "-", color=COLORS["black"], lw=0.6, label="unmitigated")[0],
          axes[1].plot([], [], "o-", color=COLORS["black"], lw=1.2, ms=2.5, label="Richardson ZNE")[0]]
    axes[1].legend(handles=hs, loc="lower right")
    ax = axes[2]
    pos = np.arange(len(channels))
    for j, (fname, hatch) in enumerate(((g, None), ("local", "////"))):
        vals = [np.median(ratio[c, fname]) for c in channels]
        ax.bar(pos + (j - 0.5) * 0.38, vals, width=0.36, color=[CHANNEL_COLORS[c] for c in channels],
               hatch=hatch, edgecolor=COLORS["black"], lw=0.4, label=f"{fname} folding")
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 300)                        # headroom for the legend
    short = {"realistic": "hardware", "depolarizing": "depolar.", "amplitude_damping": "ampl. damp.",
             "dephasing": "dephasing"}
    ax.set_xticks(pos, [short.get(c, c) for c in channels], rotation=25, ha="right")
    ax.tick_params(axis="x", which="both", top=False)
    ax.set_ylabel("median residual / unmitigated")
    ax.set_title(f"$p = 10^{{{int(round(np.log10(s[i_weak])))}}}$, all states", loc="right")
    ax.legend(loc="upper left")
    panel_labels(axes)
    return fig


if __name__ == "__main__":
    raise SystemExit(run_figure(SPEC, build))
