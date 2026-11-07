"""
Thesis Figure 5: gradient variance at random initialization against system size
(barren plateaus; Phase 2 data, N = 4 to 14, 500 samples per point).

    python experiments\\thesis\\fig05_barren_plateaus.py

Panels
  (a) local cost <Z_c Z_{c+1}>: variance of a derivative inside its light cone
  (b) global cost <H>/N at h = J
Families: HEA with p = N (linear), HEA with fixed p = CFG.vqe.grad_var_fixed_depth,
HVA with p = N. Dashed lines are fits Var = A exp(-b N).

Validation
  * Parameter counts: n_HEA = 2N(p + 1), n_HVA = 2p for every point.
  * Depths: p = N for the linear families, p = grad_var_fixed_depth for the fixed one.
  * HEA with p = N approaches a unitary 2-design, for which the variance decays
    as 2^-N: fitted b = ln 2 within CFG.thesis.bp_fit_tol (local cost).
  * HVA: no exponential decay, |b| < CFG.thesis.bp_flat_max (its dynamics stay
    in the free-fermion Lie algebra).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from config import CFG
from experiments.thesis.figlib import FigureSpec, exp_decay_fit, run_figure
from utils.plotting import SERIES_COLORS, panel_labels, thesis_figure

SPEC = FigureSpec(
    number=5,
    slug="barren_plateaus",
    chapter="Noiseless VQE",
    phases=(2,),
    title="Gradient variance at random initialization against system size",
    caption=("Variance of a cost-function derivative over 500 uniformly random parameter draws, "
             "against N. (a) Local cost <Z_c Z_c+1> with c = N/2 - 1, derivative with respect to a "
             "parameter inside its light cone. (b) Energy per site at h = J. Hardware-efficient "
             "ansatz at depth p = N (solid) and p = 2 (dash-dotted), Hamiltonian variational "
             "ansatz at p = N. Thin dashed lines are fits A exp(-bN); the HEA at p = N decays "
             "with b close to ln 2, the 2-design value, while the HVA shows no exponential decay."),
)

FAMILIES = (  # (file, label, colour, line style, marker)
    ("bp_hea_linear", "HEA, $p=N$", SERIES_COLORS["HEA"], "-", "s"),
    ("bp_hea_fixed", "HEA, $p=2$", SERIES_COLORS["HEA"], "-.", "D"),
    ("bp_hva_linear", "HVA, $p=N$", SERIES_COLORS["HVA"], "-", "o"),
)
COSTS = (("local", "local cost"), ("energy", "energy cost"))


def build(ctx):
    th = CFG.thesis
    keys = ("sizes", "depths", "n_params", "local_var_center", "energy_var_center")
    data = {f: ctx.load(f, "phase2", require=keys) for f, *_ in FAMILIES}
    ctx.param(samples=int(np.asarray(data["bp_hea_linear"].get("n_samples", 0))),
              sizes=[int(n) for n in data["bp_hea_linear"]["sizes"]],
              fixed_depth=CFG.vqe.grad_var_fixed_depth, field=CFG.vqe.grad_var_field,
              statistic="variance of the derivative for the centre parameter")

    # -- validation ------------------------------------------------------------
    cnt_ok, dep_ok = True, True
    for f, *_ in FAMILIES:
        n, p, k = (np.asarray(data[f][x]).astype(int) for x in ("sizes", "depths", "n_params"))
        expect = 2 * p if "hva" in f else 2 * n * (p + 1)
        cnt_ok &= bool(np.all(k == expect))
        dep_ok &= bool(np.all(p == (CFG.vqe.grad_var_fixed_depth if "fixed" in f else n)))
    ctx.check_true("parameter counts n_HEA = 2N(p+1), n_HVA = 2p", cnt_ok)
    ctx.check_true("depths: p = N (linear), p = grad_var_fixed_depth (fixed)", dep_ok)

    fits = {}
    for f, *_ in FAMILIES:
        for cost, _ in COSTS:
            fits[f, cost] = exp_decay_fit(data[f]["sizes"], data[f][f"{cost}_var_center"])
            b, e, _ = fits[f, cost]
            ctx.note(f"{f}, {cost} cost: b = {b:.3f} +- {e:.3f}")
    b, e, _ = fits["bp_hea_linear", "local"]
    ctx.check_close("HEA p = N, local cost: decay rate b = ln 2 (2-design)", b, np.log(2),
                    atol=th.bp_fit_tol, detail=f"fit error {e:.3f}")
    ctx.check_max("HVA p = N, local cost: no exponential decay, |b| small",
                  abs(fits["bp_hva_linear", "local"][0]), th.bp_flat_max)

    # -- figure --------------------------------------------------------------------
    fig, axes = thesis_figure(ncols=2, aspect=0.42)
    for ax, (cost, title) in zip(axes, COSTS):
        for f, label, color, ls, marker in FAMILIES:
            n = np.asarray(data[f]["sizes"], dtype=float)
            v = np.asarray(data[f][f"{cost}_var_center"], dtype=float)
            b, _, A = fits[f, cost]
            ax.semilogy(n, v, marker + ls, color=color, ms=3.5, label=f"{label}: $b={b:.2f}$")
            nn = np.linspace(n.min(), n.max(), 20)
            ax.semilogy(nn, A * np.exp(-b * nn), "--", color=color, lw=0.6)
        ax.set_xlabel("$N$")
        ax.set_ylabel(r"$\mathrm{Var}(\partial_\theta C)$")
        ax.set_xticks(np.asarray(data["bp_hea_linear"]["sizes"], dtype=int))
        ax.set_title(title, loc="right")
        ax.legend(loc="lower left")
    panel_labels(axes)
    return fig


if __name__ == "__main__":
    raise SystemExit(run_figure(SPEC, build))
