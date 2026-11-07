"""
Thesis Figure 10: optimizing the circuit under noise (SPSA, 10,000 shots per
basis) against evaluating the noiseless optimum under the same noise
(Phase 4, experiments A and B).

    python experiments\\thesis\\fig10_training_under_noise.py

Panels (N = CFG.thesis.noise_size, p = N/2, h = CFG.thesis.noise_field)
  (a) depolarizing: relative energy error of experiment A (noiseless optimum
      under noise, solid), of the warm start trained under noise (dotted), and
      of the best of three cold starts (dashed); the grey line is the product-
      state bound eps_prod (below it the energy certifies entanglement)
  (b) amplitude damping: the same; the cold-trained HEA overtakes experiment A
  (c) amplitude damping: logarithmic negativity of the same states; the low
      energies of the noise-trained circuits come from abandoning entanglement

The energies are exact Tr(rho H) of the Polyak-averaged parameters, so

    eps = (Tr(rho H) - E0) / |E0| >= 0

holds for every trained state (variational principle for mixed states).

Validation (all 72 training files)
  * Variational principle: every trained state has eps >= 0.
  * Strength 0 is shared by the three channels (one noiseless run).
  * The experiment-A energies stored with the training runs equal the Phase 4
    experiment-A files at the same N, p and field.
  * Warm starts reproduce experiment A: |median(eps_warm - eps_A)| <= 2e-4 for
    every problem (Phase 4 result).
  * eps_A and the best cold eps at the strongest noise reproduce the Phase 4
    report (N = 8, h = 1, every channel).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from config import CFG
from experiments.thesis.figlib import FigureSpec, field_index, run_figure
from utils.plotting import (COLORS, SERIES_COLORS, ANSATZ_MARKERS, label_zero_strength, panel_labels,
                            reference_line, thesis_figure, zero_strength_positions)

SPEC = FigureSpec(
    number=10,
    slug="training_under_noise",
    chapter="Noise robustness",
    phases=(4,),
    title="Training under noise: lower energy by abandoning entanglement",
    caption=("Noisy VQE at N = 8, p = 4, h = J. Solid: noiseless optimum evaluated under noise "
             "(experiment A); dotted: warm start optimized under noise by SPSA with 10,000 shots per "
             "basis; dashed: best of three cold starts. (a) Depolarizing and (b) amplitude-damping "
             "noise; the grey line is the product-state bound eps_prod, below which the energy "
             "certifies entanglement. (c) Logarithmic negativity of the amplitude-damping states: "
             "the cold-trained HEA reaches a lower energy than experiment A by retreating to a "
             "nearly unentangled state."),
)

ANSATZE = (("hva", "HVA"), ("hea", "HEA"))
WARM_TOL = 2e-4


def build(ctx):
    th = CFG.thesis
    sizes = list(CFG.system.sizes_noisy)
    fields = list(CFG.noise.train_fields)
    tkeys = ("rel_error", "energy", "E_N", "purity", "evalA_rel_error", "E0", "strengths", "channels", "h")
    train = {(a, N, hf, st): ctx.load(f"noisy_train_{a}_N{N}_h{hf:.2f}_s{st}", "phase4", require=tkeys)
             for a, _ in ANSATZE for N in sizes for hf in fields for st in range(4)}
    evals = {(a, N): ctx.load(f"noisy_eval_{a}_N{N}_p{N // 2}", "phase4",
                              require=("rel_error", "E_N", "h", "strengths"))
             for a, _ in ANSATZE for N in sizes}
    report = ctx.load_json("phase4_report_data.json", "phase4")
    first = next(iter(train.values()))
    channels = [str(c) for c in first["channels"]]
    s = np.asarray(first["strengths"], dtype=float)
    Np, hf = th.noise_size, th.noise_field
    ctx.param(plot_size=Np, plot_depth=Np // 2, field=hf, train_fields=fields, starts="0 warm, 1-3 cold",
              train_shots=int(np.asarray(first.get("train_shots", CFG.noise.train_shots))))

    # -- validation ------------------------------------------------------------
    var, shared, evA, warm = [], [], [], []
    for (a, N, f, st), d in train.items():
        var.append(np.min((np.asarray(d["energy"]) - float(d["E0"])) / abs(float(d["E0"]))))
        r0 = np.asarray(d["rel_error"])[:, 0]
        shared.append(float(np.max(r0) - np.min(r0)))
        e = evals[a, N]
        evA.append(np.max(np.abs(np.asarray(d["evalA_rel_error"]) -
                                 np.asarray(e["rel_error"])[:, :, field_index(e["h"], f)])))
        if st == 0:
            warm.append(abs(float(np.median(np.asarray(d["rel_error"]) - d["evalA_rel_error"]))))
    ctx.check_min("variational principle: Tr(rho H) >= E0 for every trained state", var,
                  -th.variational_tol)
    ctx.check_max("strength 0 shared by the three channels", shared, 1e-15)
    ctx.check_max("stored experiment-A errors equal the Phase 4 experiment-A files", evA, th.identity_tol)
    ctx.check_max(f"warm starts reproduce experiment A: |median(eps_warm - eps_A)| <= {WARM_TOL:g}",
                  warm, WARM_TOL, detail=f"{len(warm)} problems")

    best = {}
    for a, _ in ANSATZE:
        cold = np.stack([np.asarray(train[a, Np, hf, st]["rel_error"]) for st in (1, 2, 3)])  # [start, ch, s]
        k = np.argmin(cold, axis=0)
        best[a] = {"rel_error": np.take_along_axis(cold, k[None], 0)[0],
                   "E_N": np.take_along_axis(np.stack([np.asarray(train[a, Np, hf, st]["E_N"])
                                                       for st in (1, 2, 3)]), k[None], 0)[0],
                   "purity": np.take_along_axis(np.stack([np.asarray(train[a, Np, hf, st]["purity"])
                                                          for st in (1, 2, 3)]), k[None], 0)[0]}
    mine, theirs = [], []
    for a, _ in ANSATZE:
        r = report.get("training", {}).get(f"{a}_N{Np}_h{float(hf)}", {})
        for c, ch in enumerate(channels):
            if ch in r:
                mine += [float(train[a, Np, hf, 0]["evalA_rel_error"][c, -1]), float(best[a]["rel_error"][c, -1])]
                theirs += [r[ch]["eps_A_pmax"], r[ch]["eps_cold_pmax"]]
    if not mine:
        ctx.check_true("Phase 4 report has training entries for the plotted problem", False)
    ctx.check_rounded(f"eps_A and best cold eps at p = {s[-1]:g} reproduce the Phase 4 report "
                      f"(N = {Np}, h = {hf:g})", mine, theirs)

    wit = report.get("energy_witness", {})
    eps_prod = {a: wit.get(f"{a}_h{float(hf)}", {}).get("eps_prod") for a, _ in ANSATZE}
    for a, name in ANSATZE:
        A = np.asarray(train[a, Np, hf, 0]["evalA_rel_error"])
        for c, ch in enumerate(channels):
            cross = np.flatnonzero(best[a]["rel_error"][c] < A[c])
            cross = [float(s[i]) for i in cross if s[i] > 0]
            ctx.note(f"{name} N={Np} h={hf:g} {ch}: eps_A {A[c, -1]:.3f}, best cold {best[a]['rel_error'][c, -1]:.3f} "
                     f"(purity {best[a]['purity'][c, -1]:.2f}, E_N {best[a]['E_N'][c, -1]:.3f}) at p = {s[-1]:g}; "
                     f"cold below A from p = {cross[0] if cross else 'never'}")
    ctx.note(f"product-state bound eps_prod at h = {hf:g}: {eps_prod}")

    # -- figure --------------------------------------------------------------------
    x, x0 = zero_strength_positions(s)
    fig, axes = thesis_figure(ncols=3, aspect=0.36)
    panels = ((axes[0], "depolarizing", "rel_error"), (axes[1], "amplitude_damping", "rel_error"),
              (axes[2], "amplitude_damping", "E_N"))
    for ax, ch, key in panels:
        c = channels.index(ch)
        for a, name in ANSATZE:
            col, m = SERIES_COLORS[name], ANSATZ_MARKERS[name]
            if key == "rel_error":
                A = np.asarray(train[a, Np, hf, 0]["evalA_rel_error"])[c]
                W = np.asarray(train[a, Np, hf, 0]["rel_error"])[c]
            else:
                A = np.asarray(evals[a, Np]["E_N"])[c, :, field_index(evals[a, Np]["h"], hf)]
                W = np.asarray(train[a, Np, hf, 0]["E_N"])[c]
            ax.plot(x, A, m + "-", color=col, ms=2.5, lw=1.0, label=f"{name}, A")
            ax.plot(x, W, ":", color=col, lw=1.0)
            ax.plot(x, best[a][key][c], m + "--", color=col, ms=2.5, lw=1.0, mfc="none")
        ax.set_xscale("log")
        ax.set_xlabel("noise strength $p$")
        if key == "rel_error":
            ax.set_yscale("log")
            ax.set_ylabel(r"$\epsilon = (\mathrm{Tr}\,\rho H - E_0)/|E_0|$")
            for a, name in ANSATZE:
                if eps_prod.get(a):
                    reference_line(ax, eps_prod[a], color=COLORS["grey"], lw=0.7, ls="--")
                    break
        else:
            ax.set_ylabel(r"$\mathcal{E}_N$, half chain")
        ax.set_title({"depolarizing": "depolarizing", "amplitude_damping": "amplitude damping"}[ch],
                     loc="right")
        label_zero_strength(ax, x0)
    handles = [axes[2].plot([], [], "-", color=COLORS["black"], label="A: noiseless optimum")[0],
               axes[2].plot([], [], ":", color=COLORS["black"], label="warm start, trained")[0],
               axes[2].plot([], [], "--", color=COLORS["black"], label="best cold start, trained")[0],
               axes[2].plot([], [], "--", color=COLORS["grey"], lw=0.7,
                            label=r"$\epsilon_{\mathrm{prod}}$ in (a), (b)")[0]]
    leg = axes[2].legend(handles=handles, loc="lower left")
    axes[2].add_artist(leg)
    hs = [axes[2].plot([], [], ANSATZ_MARKERS[n], color=SERIES_COLORS[n], ms=3, label=n)[0]
          for _, n in ANSATZE]
    axes[2].legend(handles=hs, loc="upper right")
    panel_labels(axes)
    return fig


if __name__ == "__main__":
    raise SystemExit(run_figure(SPEC, build))
