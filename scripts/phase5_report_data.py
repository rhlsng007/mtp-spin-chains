"""
Compact Phase 5 numbers for the report: results/phase5/phase5_report_data.json.

For the reference point (CFG.mitigation.plot_size, p = N/2, h = plot_field) and every
channel and ansatz:
  * delta = |E - E_ideal| / |E0| of every method at selected strengths (exact limit and
    finite shots);
  * recovery strengths: the largest p (log-interpolated) at which a method's error stays
    below the noiseless VQE error, i.e. the noisy, mitigated energy is as good as the
    noiseless circuit's own error;
  * shot crossover: the smallest p above which readout + ZNE (finite shots) beats
    readout correction alone, i.e. where removing bias outweighs the amplified variance;
  * the first-order folding bias and the calibration (SPAM) effect at the weakest noise.
Plus channel-wide folding-bias statistics, variance amplification factors, local versus
global folding, variational-principle violations, and the readout experiment.

Run from the project root:  python scripts\\phase5_report_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np

from config import CFG, RESULTS_DIR
from noise.mitigation import variance_amplification
from phase5_plots import (MCFG, SUBDIR, _arrays, available_zne, budget, deltas, estimates, field_index,
                          shot_rms, strength_index, zne_data)
from utils.io import result_exists

METHODS_EXACT = ("unmitigated", "readout", "zne_richardson", "combined_linear", "combined_richardson",
                 "combined_exponential", "idealcal_richardson")
METHODS_SHOTS = ("unmitigated", "readout", "zne", "combined", "combined_tensored")
REPORT_STRENGTHS = (1e-4, 1e-3)


def last_below(s, y, thr):
    """Largest strength (log-interpolated) before y first exceeds thr; None if never below."""
    s, y = np.asarray(s, float), np.asarray(y, float)
    if not np.isfinite(y[0]) or y[0] >= thr:
        return None
    for i in range(1, len(s)):
        if not np.isfinite(y[i]) or y[i] >= thr:
            if not np.isfinite(y[i]) or y[i] <= 0 or y[i - 1] <= 0:
                return float(s[i - 1])
            t = (np.log(thr) - np.log(y[i - 1])) / (np.log(y[i]) - np.log(y[i - 1]))
            return float(np.exp(np.log(s[i - 1]) + t * (np.log(s[i]) - np.log(s[i - 1]))))
    return float(s[-1])


def first_better(s, a, b):
    """Smallest strength from which a < b holds for every larger strength; None if never."""
    s, a, b = np.asarray(s), np.asarray(a), np.asarray(b)
    ok = a < b
    for i in range(len(s)):
        if ok[i:].all():
            return float(s[i])
    return None


def reference_point(out):
    N, p, h = MCFG.plot_size, MCFG.plot_size // 2, MCFG.plot_field
    keys = set(available_zne())
    ref = {}
    for c in MCFG.channels:
        for a in CFG.vqe.ansatze:
            if (a, N, p, c) not in keys:
                continue
            d = zne_data(a, N, p, c)
            f = field_index(d, h)
            s = d["strengths"]
            dl, rms = deltas(d), shot_rms(d)
            eps_vqe = abs(d["E_ideal"][f] - d["E0"][f]) / abs(d["E0"][f])
            entry = {"noiseless_vqe_error": eps_vqe, "n_1q": int(d["n_1q"]), "n_2q": int(d["n_2q"]),
                     "lambda_per_unit_strength": float(budget(d, c)[-1] / s[-1]),
                     "strength_0": {m: float(dl[m][0, f]) for m in ("unmitigated", "readout")}}
            for st in REPORT_STRENGTHS:
                si = strength_index(d, st)
                entry[f"p={s[si]:.0e}"] = {
                    "lambda": float(budget(d, c)[si]),
                    "exact": {m: float(dl[m][si, f]) for m in METHODS_EXACT},
                    "shots": {m: float(rms[m][si, f]) for m in METHODS_SHOTS}}
            nz = slice(1, None)
            entry["recovery_strength_exact"] = {m: last_below(s[nz], dl[m][nz, f], eps_vqe)
                                                for m in ("unmitigated", "readout", "zne_richardson",
                                                          "combined_richardson", "combined_exponential")}
            entry["recovery_strength_shots"] = {m: last_below(s[nz], rms[m][nz, f], eps_vqe)
                                                for m in ("unmitigated", "readout", "combined")}
            entry["shot_crossover_zne_pays"] = first_better(s[nz], rms["combined"][nz, f], rms["readout"][nz, f])
            entry["weakest_noise"] = {
                "p": float(s[1]),
                "folding_bias_ratio": float(dl["idealcal_richardson"][1, f] / dl["gate_only"][1, f]),
                "spam_vs_ideal_calibration": float(dl["combined_richardson"][1, f] /
                                                   max(dl["idealcal_richardson"][1, f], 1e-300))}
            ref[f"{a}_{c}"] = entry
    out["reference_point"] = {"N": N, "p": p, "h": h, "data": ref}


def channel_statistics(out):
    stats = {}
    for c in MCFG.channels:
        bias, slopes = [], []
        for k in available_zne():
            if k[3] != c:
                continue
            d = zne_data(*k)
            dl = deltas(d)
            ok = dl["gate_only"][1] > 1e-13
            bias.extend(list(dl["idealcal_richardson"][1][ok] / dl["gate_only"][1][ok]))
        if bias:
            q = np.percentile(bias, [5, 50, 95])
            stats[c] = {"folding_bias_ratio_median": float(q[1]), "p5": float(q[0]), "p95": float(q[2]),
                        "n_points": len(bias)}
    out["folding_bias_by_channel"] = stats


def misc(out):
    ex = MCFG.main_extrapolator
    sc = CFG.noise.zne_scale_factors
    out["variance_amplification"] = {m: variance_amplification(sc, m) for m in ("linear", "richardson")}
    ratios, below, n = [], 0, 0
    for k in available_zne():
        d = zne_data(*k)
        if "local" in list(np.asarray(d["foldings"]).astype(str)):
            g = deltas(d, "global")[f"idealcal_{ex}"][1:]
            lo = deltas(d, "local")[f"idealcal_{ex}"][1:]
            ok = (g > 1e-13) & (lo > 1e-13)
            ratios.extend(list((lo[ok] / g[ok]).ravel()))
        comb = estimates(d)[f"combined_{ex}"][1:]
        below += int(np.sum(comb < d["E0"][None, :] - 1e-12))
        n += comb.size
    if ratios:
        q = np.percentile(ratios, [5, 50, 95])
        out["local_over_global_residual"] = {"median": float(q[1]), "p5": float(q[0]), "p95": float(q[2])}
    out["combined_below_E0"] = {"count": below, "of": n}


def readout(out):
    res = {}
    for a in CFG.vqe.ansatze:
        for N in CFG.system.sizes_noisy:
            name = f"readout_{a}_N{N}"
            if not result_exists(name, SUBDIR):
                continue
            r = _arrays(name)
            meth = list(np.asarray(r["methods"]).astype(str))
            entry = {"raw_energy_bias": float(abs(r["E_raw_exact"] - r["E_ideal"]) / abs(r["E0"])),
                     "raw_tvd": [float(x) for x in r["tvd_raw_exact"]]}
            for ci, M in enumerate(r["cal_shots"]):
                entry[f"M_cal={int(M)}"] = {
                    m: {"tvd_median": float(np.median(r["tvd"][0, ci, mi].mean(-1))),
                        "energy_rms_shots": float(np.sqrt(np.mean(r["dE"][1, ci, mi] ** 2))),
                        "negative_mass_median": float(np.median(r["neg_mass"][0, ci, mi].sum(-1)))}
                    for mi, m in enumerate(meth)}
            res[f"{a}_N{N}"] = entry
    out["readout"] = res


def main():
    out = {"readout_p01": MCFG.readout_p01, "readout_p10": MCFG.readout_p10,
           "two_qubit_factor": MCFG.two_qubit_factor, "strengths": [float(x) for x in MCFG.strengths()]}
    reference_point(out)
    channel_statistics(out)
    misc(out)
    readout(out)
    path = RESULTS_DIR / SUBDIR / "phase5_report_data.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["reference_point"]["data"].get("hva_realistic", {}), indent=2)[:3000])
    print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
