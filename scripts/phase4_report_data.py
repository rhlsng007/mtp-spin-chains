"""
Compact Phase 4 numbers for the report, computed from the saved results only.

Usage (from the project root, after experiments\\phase4_noise.py and phase4_plots.py):
    python scripts\\phase4_report_data.py

Writes results/phase4/phase4_report_data.json and prints a readable summary.
Crossing strengths are interpolated linearly in log10(p) between grid points;
None means no crossing inside the swept range (1e-4 to 5e-2), "<=0.0001" means
already crossed at the weakest strength.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import minimize

from config import CFG, RESULTS_DIR

D4 = RESULTS_DIR / "phase4"
ANS = ("hva", "hea")
H_MAIN = 1.0


def load(name):
    path = D4 / f"{name}.npz"
    return dict(np.load(path)) if path.exists() else None


def crossing(s, y, level=0.0):
    """First strength p > 0 where y(p) - level changes sign from <= 0 to > 0 (log-interpolated)."""
    m = s > 0
    s, d = s[m], np.asarray(y)[m] - level
    if d[0] > 0:
        return f"<={s[0]:g}"                   # already crossed at the weakest strength
    for k in range(1, d.size):
        if d[k - 1] <= 0 < d[k]:
            t = d[k - 1] / (d[k - 1] - d[k])
            return float(10 ** (np.log10(s[k - 1]) + t * (np.log10(s[k]) - np.log10(s[k - 1]))))
    return None


def r(x, n=4):
    if isinstance(x, str):
        return x
    return None if x is None or not np.isfinite(x) else float(f"{x:.{n}g}")


def budget(pc, n_gates):
    """Error budget at a crossing; strings ('<=...') and None pass through."""
    if pc is None or isinstance(pc, str):
        return pc
    return pc * n_gates


def best_product_energy(N, h, J):
    def e(th):
        return -J * np.sum(np.cos(th[:-1]) * np.cos(th[1:])) - h * np.sum(np.sin(th))
    return float(min(minimize(e, np.full(N, x0), method="L-BFGS-B").fun for x0 in (0.1, 0.5, 1.0, 1.4)))


def cc_fit(profile, N):
    ell = np.arange(CFG.ent.cc_ell_min, N // 2 + 1)
    x = np.log(2 * N / np.pi * np.sin(np.pi * ell / N))
    return float(6 * np.polyfit(x, profile[ell - 1], 1)[0])


def main():
    out = {"h_main": H_MAIN, "strengths": None, "kappa": {}, "hea_vs_hva": {}, "hashing": {},
           "sudden_death": {}, "energy_witness": {}, "shot_floor": {}, "central_charge": {},
           "training": {}, "noiseless": {}}
    E = {}
    for N in CFG.system.sizes_noisy:
        for a in ANS:
            for p in CFG.system.depths(N, noisy=True):
                d = load(f"noisy_eval_{a}_N{N}_p{p}")
                if d is not None:
                    E[a, N, p] = d
    if not E:
        sys.exit("no noisy_eval files found")
    any_d = next(iter(E.values()))
    s = any_d["strengths"]
    chans = [str(c) for c in any_d["channels"]]
    out["strengths"] = [float(x) for x in s]
    Nmax = max(N for (_, N, _) in E)

    # --- error-budget collapse: kappa = -ln F_ideal / lambda for lambda <= 1, all fields
    for ic, ch in enumerate(chans):
        k = []
        for (a, N, p), d in E.items():
            lam = s[1:] * (d["n_1q"] + d["n_2q"])
            F = d["fidelity_ideal"][ic, 1:, :]
            ok = lam <= 1.0
            k.append((-np.log(F[ok]) / lam[ok, None]).ravel())
        k = np.concatenate(k)
        out["kappa"][ch] = {"median": r(np.median(k)), "p05": r(np.percentile(k, 5)),
                            "p95": r(np.percentile(k, 95)), "n": int(k.size)}

    # --- noiseless reference values and HEA-vs-HVA crossover at N max, p = N/2, h_main
    pN = Nmax // 2
    dv, de = E.get(("hva", Nmax, pN)), E.get(("hea", Nmax, pN))
    if dv is not None and de is not None:
        ih = int(np.flatnonzero(np.isclose(dv["h"], H_MAIN))[0])
        for a, d in (("hva", dv), ("hea", de)):
            out["noiseless"][a] = {"rel_error": r(d["rel_error"][0, 0, ih]), "fidelity_gs": r(d["fidelity_gs"][0, 0, ih]),
                                   "S_A": r(d["S_A"][0, 0, ih]), "E_N": r(d["E_N"][0, 0, ih]),
                                   "n_1q": int(d["n_1q"]), "n_2q": int(d["n_2q"])}
        for ic, ch in enumerate(chans):
            diff = np.log(de["rel_error"][ic, :, ih]) - np.log(dv["rel_error"][ic, :, ih])
            out["hea_vs_hva"][ch] = r(crossing(s, diff))

    # --- hashing bound, sudden death, at h_main, every (ansatz, N, p)
    for ic, ch in enumerate(chans):
        hb, sd = {}, {}
        for (a, N, p), d in sorted(E.items()):
            ih = int(np.flatnonzero(np.isclose(d["h"], H_MAIN))[0])
            n_g = int(d["n_1q"] + d["n_2q"])
            ps = crossing(s, d["S_cond"][ic, :, ih])
            hb[f"{a}_N{N}_p{p}"] = {"p": r(ps), "lambda": r(budget(ps, n_g))}
            dead = np.flatnonzero((s > 0) & (d["E_N"][ic, :, ih] < 1e-6))
            sd[f"{a}_N{N}_p{p}"] = r(float(s[dead[0]])) if dead.size else None
        out["hashing"][ch], out["sudden_death"][ch] = hb, sd

    # --- energy witness (eps_A crosses eps_prod) at N max, p = N/2, training fields
    for h in CFG.noise.train_fields:
        for a in ANS:
            d = E.get((a, Nmax, pN))
            if d is None:
                continue
            ih = int(np.flatnonzero(np.isclose(d["h"], h))[0])
            E0 = float(d["E0"][ih])
            eps_prod = (best_product_energy(Nmax, h, CFG.tfim.J) - E0) / abs(E0)
            rec = {"eps_prod": r(eps_prod)}
            for ic, ch in enumerate(chans):
                eps = d["rel_error"][ic, :, ih]
                pw = "already above at p=0" if eps[0] >= eps_prod else crossing(s, eps, eps_prod)
                rec[ch] = {"p": r(pw), "lambda": r(budget(pw, float(d["n_1q"] + d["n_2q"])))}
            out["energy_witness"][f"{a}_h{h}"] = rec

    # --- shot floors sigma/|E0| at zero noise, N max, p = N/2, h_main
    for a in ANS:
        d = E.get((a, Nmax, pN))
        if d is None:
            continue
        ih = int(np.flatnonzero(np.isclose(d["h"], H_MAIN))[0])
        out["shot_floor"][a] = {str(int(M)): r(d["E_sigma"][0, 0, ih, k] / abs(d["E0"][ih]))
                                for k, M in enumerate(d["shots"])}

    # --- central charge from S and from I/2 at N max, p = N/2, cc_field, depolarizing
    ich = chans.index("depolarizing")
    for a in ANS:
        d = E.get((a, Nmax, pN))
        if d is None:
            continue
        ih = int(np.flatnonzero(np.isclose(d["h"], CFG.ent.cc_field))[0])
        out["central_charge"][a] = {f"{x:.1e}": {"from_S": r(cc_fit(d["S_A_profile"][ich, j, ih], Nmax)),
                                                  "from_I_half": r(cc_fit(0.5 * d["I_profile"][ich, j, ih], Nmax))}
                                    for j, x in enumerate(s)}

    # --- training: crossover where cold median beats experiment A; values at p_max
    n_starts = 1 + CFG.seeds.n_seeds_noisy
    for N in CFG.system.sizes_noisy:
        for a in ANS:
            for h in CFG.noise.train_fields:
                runs = [load(f"noisy_train_{a}_N{N}_h{h:.2f}_s{k}") for k in range(n_starts)]
                if any(x is None for x in runs):
                    continue
                rec = {"cold_floor": r(float(np.median([x["rel_error"][0, 0] for x in runs[1:]]))),
                       "warm_minus_A_median": r(float(np.median(runs[0]["rel_error"] - runs[0]["evalA_rel_error"])))}
                for ic, ch in enumerate(chans):
                    cold = np.median([x["rel_error"][ic] for x in runs[1:]], axis=0)
                    A = runs[0]["evalA_rel_error"][ic]
                    gap = np.log(A) - np.log(cold)          # > 0 where cold beats A
                    rec[ch] = {"crossover_p": r(crossing(s, gap)),
                               "eps_A_pmax": r(A[-1]), "eps_cold_pmax": r(cold[-1]),
                               "purity_cold_pmax": r(float(np.median([x["purity"][ic, -1] for x in runs[1:]]))),
                               "E_N_cold_pmax": r(max(0.0, float(np.median([x["E_N"][ic, -1] for x in runs[1:]]))))}
                out["training"][f"{a}_N{N}_h{h}"] = rec

    D4.mkdir(parents=True, exist_ok=True)
    (D4 / "phase4_report_data.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    # --- readable summary
    print("kappa (error-budget collapse, lambda <= 1, all fields/N/p):")
    for ch, v in out["kappa"].items():
        print(f"  {ch:>17s}: median {v['median']}, 5-95% [{v['p05']}, {v['p95']}] (n={v['n']})")
    print(f"HEA overtaken by HVA (N={Nmax}, p={pN}, h={H_MAIN}): {out['hea_vs_hva']}")
    print(f"hashing bound lost, h={H_MAIN} (p, lambda):")
    for ch, v in out["hashing"].items():
        print(f"  {ch}: " + "; ".join(f"{k} {x['p']} ({x['lambda']})" for k, x in v.items()))
    print(f"entanglement sudden death (E_N < 1e-6), h={H_MAIN}:")
    for ch, v in out["sudden_death"].items():
        print(f"  {ch}: " + "; ".join(f"{k} {x}" for k, x in v.items()))
    print("energy witness lost (eps_A > eps_prod):")
    for k, v in out["energy_witness"].items():
        print(f"  {k}: eps_prod {v['eps_prod']}; " + "; ".join(f"{ch} p={v[ch]['p']} lam={v[ch]['lambda']}" for ch in chans))
    print(f"shot floors sigma/|E0| at zero noise: {out['shot_floor']}")
    print("central charge (depolarizing):")
    for a, v in out["central_charge"].items():
        print(f"  {a}: " + "; ".join(f"p={k}: S {x['from_S']}, I/2 {x['from_I_half']}" for k, x in v.items()))
    print("training:")
    for k, v in out["training"].items():
        print(f"  {k}: cold floor {v['cold_floor']}, warm-A median {v['warm_minus_A_median']}; " +
              "; ".join(f"{ch}: x-over {v[ch]['crossover_p']}, eps A/cold at pmax {v[ch]['eps_A_pmax']}/{v[ch]['eps_cold_pmax']}, "
                        f"purity {v[ch]['purity_cold_pmax']}, E_N {v[ch]['E_N_cold_pmax']}" for ch in chans))
    print(f"\nwritten {D4 / 'phase4_report_data.json'}")


if __name__ == "__main__":
    main()
