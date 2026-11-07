"""
Thesis Figure 16 (synthesis): the thresholds of Phases 4, 5 and 7 on one axis,
the error budget lambda = expected number of gate errors per circuit.

    python experiments\\thesis\\fig16_error_budget_window.py

Each row is one criterion for the N = 8, p = 4 circuits at h = J; markers are
the lambda at which the criterion is lost (HVA circles, HEA squares):

  Phase 4 (depolarizing, amplitude damping, dephasing; lambda = p (n_1q + n_2q))
    * energy witness lost: eps_A > eps_prod (no longer certifies entanglement)
    * hashing bound lost: S(A|B) > 0
    * entanglement sudden death: E_N = 0 (depolarizing)
  Phase 5 (hardware-like; lambda = p (n_1q + 10 n_2q))
    * readout + ZNE no longer at VQE accuracy
  Phase 7 (hardware-like, readout-corrected shadows, logistic regression)
    * 2-sigma drift of h_c^ML: trained on ED, on the noiseless circuit, on mixed noise

The shaded band is the window spanned by the Phase 4 witness and hashing
thresholds; the hatched band is the ZNE window of Phases 6 and 7.

Validation
  * Gate counts: Phase 4 budget 68 p (HVA) and 108 p (HEA); hardware-like
    budget 320 p for both (n_1q + 10 n_2q from the stored counts).
  * Witness and hashing thresholds reproduce the Phase 4 report.
  * Clean-trained drift onsets reproduce the Phase 7 results.
  * Unifying result: every clean-trained drift onset lies inside the Phase 4
    witness/hashing window.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from config import CFG
from experiments.thesis.fig09_mixed_state_entanglement import first_crossing
from experiments.thesis.fig11_error_mitigation import estimates, recovery_strength
from experiments.thesis.fig14_hc_vs_noise import PHASE7_ONSETS
from experiments.thesis.figlib import (FigureSpec, drift_onset, extrapolation_weights, field_index, run_figure,
                                       summarize_hc)
from utils.plotting import (ANSATZ_MARKERS, CHANNEL_COLORS, COLORS, MITIGATION_COLORS, MODE_COLORS,
                            SERIES_COLORS, thesis_figure)

SPEC = FigureSpec(
    number=16,
    slug="error_budget_window",
    chapter="Synthesis",
    phases=(4, 5, 6, 7),
    title="One error-budget window: where entanglement, mitigation and classification fail",
    caption=("Thresholds of Phases 4, 5 and 7 for the N = 8, p = 4 circuits at h = J on a common "
             "axis, the error budget lambda (expected number of gate errors per circuit). HVA: "
             "circles, HEA: squares. Phase 4: entanglement certificates lost under depolarizing "
             "(purple), amplitude-damping (light blue) and dephasing (green) noise. Phase 5: readout "
             "correction plus zero-noise extrapolation no longer reaches VQE accuracy. Phase 7: the "
             "learned critical field drifts by 2 sigma. The grey band is spanned by the Phase 4 "
             "witness and hashing thresholds; the hatched band is the ZNE window. Clean-trained "
             "classification fails inside the same window in which the entanglement certificates are "
             "lost; training on noisy data moves the threshold beyond it."),
)

ANSATZE = (("hva", "HVA"), ("hea", "HEA"))
CHANNELS4 = ("depolarizing", "amplitude_damping", "dephasing")


def build(ctx):
    th, ml, mt = CFG.thesis, CFG.ml, CFG.mitigation
    Np, hf = th.noise_size, th.noise_field
    ev = {a: ctx.load(f"noisy_eval_{a}_N{Np}_p{Np // 2}", "phase4",
                      require=("S_cond", "E_N", "rel_error", "strengths", "channels", "h", "n_1q", "n_2q"))
          for a, _ in ANSATZE}
    r4 = ctx.load_json("phase4_report_data.json", "phase4")
    z5 = {a: ctx.load(f"zne_{a}_N{Np}_p{Np // 2}_realistic", "phase5",
                      require=("E_raw", "E_ro", "E_ideal", "E0", "strengths", "scales", "foldings", "h", "n_1q", "n_2q"))
          for a, _ in ANSATZE}
    n7 = {a: ctx.load(f"noise_{a}_N{ml.study_size}_p{ml.study_size // 2}_{ml.noisy_channels[0]}_{ml.default_variant}",
                      "phase7/noise", require=("hc", "hc_rep", "lam", "strengths", "modes", "kinds", "feature_sets",
                                               "classifiers")) for a, _ in ANSATZE}
    zds = ctx.load(f"ds_zne_hva_N{ml.study_size}_p{ml.study_size // 2}_{ml.noisy_channels[0]}", "phase6/zne",
                   require=("lam",))
    ctx.param(size=Np, depth=Np // 2, field=hf, phase4_channels=list(CHANNELS4), phase5_channel="realistic",
              phase7_variant=ml.default_variant)

    # -- thresholds ---------------------------------------------------------------
    rows = []                                          # (row label, phase, {ansatz: [(lambda, colour)]})
    wit, hsh, death, mine, theirs = {}, {}, {}, [], []
    for a, _ in ANSATZE:
        d = ev[a]
        s = np.asarray(d["strengths"], dtype=float)
        ih = field_index(d["h"], hf)
        n4 = int(d["n_1q"]) + int(d["n_2q"])
        chans = [str(c) for c in d["channels"]]
        eps_prod = r4["energy_witness"][f"{a}_h{float(hf)}"]["eps_prod"]
        for ch in CHANNELS4:
            c = chans.index(ch)
            pw = first_crossing(s, np.asarray(d["rel_error"])[c, :, ih], level=eps_prod)
            ph = first_crossing(s, np.asarray(d["S_cond"])[c, :, ih])
            wit[a, ch], hsh[a, ch] = pw * n4, ph * n4
            rw = r4["energy_witness"][f"{a}_h{float(hf)}"].get(ch, {})
            rh = r4["hashing"][ch][f"{a}_N{Np}_p{Np // 2}"]
            if isinstance(rw, dict) and rw.get("p") is not None:
                mine.append(pw)
                theirs.append(rw["p"])
            mine.append(ph)
            theirs.append(rh["p"])
        c = chans.index("depolarizing")
        hit = np.flatnonzero(np.asarray(d["E_N"])[c, :, ih] <= th.zero_tol)
        death[a] = float(s[hit[0]]) * n4 if hit.size else float("nan")
    rec = {}
    for a, _ in ANSATZE:
        d = z5[a]
        s = np.asarray(d["strengths"], dtype=float)
        ih = field_index(d["h"], mt.plot_field)
        w = extrapolation_weights(np.asarray(d["scales"], dtype=float), len(d["scales"]) - 1)
        fold = [str(f) for f in d["foldings"]].index(mt.main_folding)
        Ei, E0 = float(d["E_ideal"][ih]), float(d["E0"][ih])
        comb = estimates(d, w, fold)["combined"][:, ih]
        pr = recovery_strength(s, np.abs(comb - Ei) / abs(E0), (Ei - E0) / abs(E0))
        rec[a] = pr * (int(d["n_1q"]) + 10 * int(d["n_2q"]))
    drift = {}
    for a, _ in ANSATZE:
        d = n7[a]
        K = {k: i for i, k in enumerate(d["kinds"])}
        FS = {f: i for i, f in enumerate(d["feature_sets"])}
        CL = {c: i for i, c in enumerate(d["classifiers"])}
        M = {m: i for i, m in enumerate(d["modes"])}
        st = summarize_hc(d["hc"], d["hc_rep"])
        kk, fc, lr = K[th.ml_main_kind], FS[th.ml_main_features], CL[th.ml_main_classifier]
        for m in ("clean_ed", "clean_vqe", "mixed"):
            k = drift_onset(st["mean"][M[m], :, kk, fc, lr], st["total"][M[m], :, kk, fc, lr], d["strengths"],
                            th.drift_sigmas)
            drift[a, m] = float(d["lam"][k]) if k >= 0 else float("nan")

    # -- validation ------------------------------------------------------------
    cnt = {a: (int(ev[a]["n_1q"]) + int(ev[a]["n_2q"]), int(z5[a]["n_1q"]) + 10 * int(z5[a]["n_2q"]),
               float(n7[a]["lam"][-1] / n7[a]["strengths"][-1])) for a, _ in ANSATZE}
    ctx.check_true("gate counts: Phase 4 budget 68 p (HVA), 108 p (HEA); hardware-like 320 p in Phases 5 and 7",
                   cnt["hva"][0] == 68 and cnt["hea"][0] == 108 and all(abs(v[1] - 320) < 1e-9 and
                                                                     abs(v[2] - 320) < 1e-6 for v in cnt.values()),
                   f"{cnt}")
    ctx.check_close("witness and hashing thresholds reproduce the Phase 4 report", mine, theirs,
                    rtol=th.hashing_rtol, detail=f"{len(mine)} strengths")
    ok = all((abs(drift[a, m] - PHASE7_ONSETS[a][m]) <= 0.006 * PHASE7_ONSETS[a][m] + 0.005)
             for a, _ in ANSATZE for m in ("clean_ed", "clean_vqe", "mixed"))
    ctx.check_true("drift onsets reproduce the Phase 7 results", ok,
                   ", ".join(f"{a.upper()} {m} {drift[a, m]:.2f}" for a, _ in ANSATZE for m in ("clean_ed", "clean_vqe",
                                                                                                "mixed")))
    window = [v for v in list(wit.values()) + list(hsh.values()) if np.isfinite(v)]
    lo, hi = min(window), max(window)
    inside = [lo <= drift[a, m] <= hi for a, _ in ANSATZE for m in ("clean_ed", "clean_vqe")]
    ctx.check_true("unifying result: clean-trained drift onsets lie inside the Phase 4 witness/hashing window",
                   all(inside), f"window lambda = {lo:.3f} to {hi:.3f}")
    ctx.note(f"Phase 4 window (witness and hashing, 3 channels, both ansaetze): lambda = {lo:.3f} to {hi:.3f}")
    for a, name in ANSATZE:
        ctx.note(f"{name}: witness lost at lambda = " + ", ".join(f"{ch} {wit[a, ch]:.3f}" for ch in CHANNELS4) +
                 "; hashing lost at " + ", ".join(f"{ch} {hsh[a, ch]:.3f}" for ch in CHANNELS4) +
                 f"; sudden death (depolarizing) {death[a]:.2f}; readout + ZNE at VQE accuracy up to "
                 f"{rec[a]:.3f}; drift onsets " + ", ".join(f"{m} {drift[a, m]:.2f}" for m in ("clean_ed", "clean_vqe",
                                                                                               "mixed")))
    zw = np.asarray(zds["lam"], dtype=float)
    ctx.note(f"ZNE window (Phase 6 datasets, N = {ml.study_size}): lambda = {zw.min():.3f} to {zw.max():.3f}")

    # -- figure --------------------------------------------------------------------
    labels = ["energy witness lost", "hashing bound lost", "entanglement sudden death",
              "readout + ZNE above VQE error", "drift: trained on ED", "drift: trained on noiseless circuit",
              "drift: trained on mixed noise"]
    fig, ax = thesis_figure(ncols=1, width="full", aspect=0.42)
    y = np.arange(len(labels))[::-1]
    ax.axvspan(lo, hi, color=COLORS["grey"], alpha=0.18, lw=0, zorder=0)
    ax.axvspan(zw.min(), zw.max(), facecolor="none", edgecolor=COLORS["grey"], hatch="////", lw=0, zorder=0)
    dy = {"hva": 0.12, "hea": -0.12}
    for a, name in ANSATZE:
        mk = ANSATZ_MARKERS[name]
        for ch in CHANNELS4:
            ax.plot(wit[a, ch], y[0] + dy[a], mk, color=CHANNEL_COLORS[ch], ms=4.5)
            ax.plot(hsh[a, ch], y[1] + dy[a], mk, color=CHANNEL_COLORS[ch], ms=4.5)
        ax.plot(death[a], y[2] + dy[a], mk, color=CHANNEL_COLORS["depolarizing"], ms=4.5)
        ax.plot(rec[a], y[3] + dy[a], mk, color=MITIGATION_COLORS["combined"], ms=4.5)
        for row, m in zip(y[4:], ("clean_ed", "clean_vqe", "mixed")):
            ax.plot(drift[a, m], row + dy[a], mk, color=MODE_COLORS[m], ms=4.5)
    for row in y[:-1]:
        ax.axhline(row - 0.5, color=COLORS["grey"], lw=0.3)
    ax.axhline(y[3] + 0.5, color=COLORS["black"], lw=0.6)
    ax.axhline(y[4] + 0.5, color=COLORS["black"], lw=0.6)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 10)
    ax.set_xlabel(r"error budget $\lambda$ (expected gate errors per circuit), $N=8$, $p=4$, $h=J$")
    for txt, row in (("Phase 4", y[1]), ("Phase 5", y[3]), ("Phase 7", y[5])):
        ax.text(1.01, row, txt, transform=ax.get_yaxis_transform(), ha="left", va="center", rotation=90)
    hs = [ax.plot([], [], ANSATZ_MARKERS[n], color=COLORS["black"], ms=4.5, label=f"{lab}: {n}")[0]
          for n, lab in (("HVA", "circles"), ("HEA", "squares"))]
    leg = ax.legend(handles=hs, loc="lower right")
    ax.add_artist(leg)
    hc_ = [ax.plot([], [], "s", color=CHANNEL_COLORS[ch], ms=4.5, label=lab)[0]
           for ch, lab in (("depolarizing", "depolarizing"), ("amplitude_damping", "amplitude damping"),
                           ("dephasing", "dephasing"))]
    ax.legend(handles=hc_, loc="upper left", title="Phase 4 channel")
    return fig


if __name__ == "__main__":
    raise SystemExit(run_figure(SPEC, build))
