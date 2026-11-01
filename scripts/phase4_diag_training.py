"""
Phase 4 diagnostic: what do circuits trained under noise converge to?

For each ansatz, channel and field it compares, at selected strengths,
    A     : the noiseless optimum evaluated under noise (experiment A)
    warm  : SPSA under noise started from the noiseless optimum
    cold  : SPSA under noise from small random angles (median of the cold starts)
with the best product state of the open TFIM, whose energy is

    E_prod = min over theta_i of  -J sum_i cos(theta_i) cos(theta_{i+1}) - h sum_i sin(theta_i)

(spins in the x-z plane, <Z_i> = cos theta_i, <X_i> = sin theta_i). Hypothesis:
under strong amplitude damping or dephasing, cold-trained circuits approach
this unentangled, noise-robust state (E_N about 0, eps about eps_prod).

Usage (from the project root):
    python scripts\\phase4_diag_training.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import minimize

from config import CFG, RESULTS_DIR

SHOW_STRENGTHS = (0.0, 1e-3, 1e-2, 5e-2)


def best_product_energy(N: int, h: float, J: float) -> float:
    def e(th):
        return -J * np.sum(np.cos(th[:-1]) * np.cos(th[1:])) - h * np.sum(np.sin(th))
    best = np.inf
    for start in (0.1, 0.5, 1.0, 1.4):
        r = minimize(e, np.full(N, start), method="L-BFGS-B")
        best = min(best, r.fun)
    return float(best)


def main():
    d4 = RESULTS_DIR / "phase4"
    N = max(CFG.system.sizes_noisy)
    n_starts = 1 + CFG.seeds.n_seeds_noisy
    for h in CFG.noise.train_fields:
        for a in CFG.vqe.ansatze:
            runs = [d4 / f"noisy_train_{a}_N{N}_h{h:.2f}_s{s}.npz" for s in range(n_starts)]
            if not all(r.exists() for r in runs):
                print(f"missing training files for {a} N={N} h={h}")
                continue
            R = [np.load(r) for r in runs]
            E0 = float(R[0]["E0"])
            eps_prod = (best_product_energy(N, h, CFG.tfim.J) - E0) / abs(E0)
            s = R[0]["strengths"]
            print(f"\n===== {a.upper()}  N={N}  h={h}   best product state: eps_prod = {eps_prod:.3e} =====")
            print(f"{'channel':>17s} {'p':>8s} | {'eps A':>9s} {'eps warm':>9s} {'eps cold':>9s} | "
                  f"{'purity w':>8s} {'purity c':>8s} | {'E_N warm':>8s} {'E_N cold':>8s} | {'S_A cold':>8s}")
            for ic, ch in enumerate(R[0]["channels"]):
                for target in SHOW_STRENGTHS:
                    j = int(np.argmin(np.abs(s - target)))
                    cold = lambda k: float(np.median([r[k][ic, j] for r in R[1:]]))
                    print(f"{ch:>17s} {s[j]:8.1e} | {R[0]['evalA_rel_error'][ic, j]:9.3e} "
                          f"{R[0]['rel_error'][ic, j]:9.3e} {cold('rel_error'):9.3e} | "
                          f"{R[0]['purity'][ic, j]:8.4f} {cold('purity'):8.4f} | "
                          f"{max(R[0]['E_N'][ic, j], 0):8.4f} {max(cold('E_N'), 0):8.4f} | {cold('S_A'):8.4f}")


if __name__ == "__main__":
    main()
