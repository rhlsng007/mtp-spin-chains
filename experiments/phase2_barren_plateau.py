"""
Phase 2, task T2.7: gradient variance at random initialization versus N.

Run from the project root:

    python experiments\\phase2_barren_plateau.py
    python experiments\\phase2_barren_plateau.py --sizes 4 6 8 10 --samples 200
    python experiments\\phase2_barren_plateau.py --recompute

For each circuit family and size N, theta is drawn uniformly from [0, 2 pi)^P
(CFG.vqe.grad_var_samples times) and the exact gradient of two costs is
evaluated at the same parameters:

    local   C = <Z_c Z_{c+1}>, c = N/2 - 1   (local observable at the centre)
    energy  C = <H_TFIM(h)> / N, h = CFG.vqe.grad_var_field   (energy density)

Circuit families:

    hea_linear   HEA, p = N   (depth grows with N: approaches a 2-design)
    hea_fixed    HEA, p = CFG.vqe.grad_var_fixed_depth   (shallow, local cost)
    hva_linear   HVA, p = N

Diagnostics stored per N and cost (results/phase2/bp_{family}.npz):

    var_center  Var of dC/dtheta for one fixed parameter inside the light cone of
                the local observable at every depth: the first-layer RY on qubit c
                (HEA) or the first ZZ angle gamma_1 (HVA). This is the McClean-type
                single-parameter diagnostic and the one used for the fits.
    var_mean    Var averaged over all parameters. For a shallow circuit most
                parameters lie outside the light cone of a local cost and have zero
                gradient, which adds an artificial 1/N dilution.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import CFG
from models.hamiltonians import build_hamiltonian, pauli_string_sparse
from utils.io import load_or_compute
from vqe.ansatz import make_ansatz
from vqe.gradients import fit_exponential_decay, fit_power_law
from vqe.optimize import ANSATZ_KEYS, point_rng

SUBDIR = "phase2"
EXPERIMENT_KEY = 6
COSTS = ("local", "energy")
FAMILIES = {
    "hea_linear": ("hea", lambda n: n),
    "hea_fixed": ("hea", lambda n: CFG.vqe.grad_var_fixed_depth),
    "hva_linear": ("hva", lambda n: n),
}


def cost_operators(n_sites: int) -> dict:
    c = n_sites // 2 - 1
    c_tfim = CFG.tfim
    h_tfim = build_hamiltonian("tfim", n_sites, J=c_tfim.J, h=CFG.vqe.grad_var_field,
                               boundary=c_tfim.boundary)
    return {"local": pauli_string_sparse(n_sites, {c: "Z", c + 1: "Z"}),
            "energy": h_tfim / n_sites}


def compute_family(family: str, sizes: list, n_samples: int) -> dict:
    ansatz_name, depth_of = FAMILIES[family]
    out = {"sizes": np.asarray(sizes), "depths": np.zeros(len(sizes), int),
           "n_params": np.zeros(len(sizes), int), "n_samples": n_samples}
    for cost in COSTS:
        for key in ("var_center", "var_mean", "var_first", "var_median", "mean_abs_grad"):
            out[f"{cost}_{key}"] = np.zeros(len(sizes))
    for a, n in enumerate(sizes):
        t0 = time.perf_counter()
        depth = depth_of(n)
        ansatz = make_ansatz(ansatz_name, n, depth)
        ops = cost_operators(n)
        rng = point_rng(EXPERIMENT_KEY, ANSATZ_KEYS[ansatz_name], n, depth)
        grads = {cost: np.empty((n_samples, ansatz.n_params)) for cost in COSTS}
        for s in range(n_samples):
            theta = ansatz.random_params(rng, 1.0, "uniform")
            for cost in COSTS:
                grads[cost][s] = ansatz.energy_and_grad(theta, ops[cost])[1]
        center = (n // 2 - 1) if ansatz_name == "hea" else 0   # first-layer RY_c, or gamma_1
        out["depths"][a] = depth
        out["n_params"][a] = ansatz.n_params
        for cost in COSTS:
            var = grads[cost].var(axis=0, ddof=1)
            out[f"{cost}_var_center"][a] = var[center]
            out[f"{cost}_var_mean"][a] = var.mean()
            out[f"{cost}_var_first"][a] = var[0]
            out[f"{cost}_var_median"][a] = np.median(var)
            out[f"{cost}_mean_abs_grad"][a] = np.abs(grads[cost]).mean()
        print(f"    {family:11s} N={n:2d} p={depth:2d}: Var(local) = {out['local_var_center'][a]:.3e}, "
              f"Var(energy) = {out['energy_var_center'][a]:.3e}  ({time.perf_counter() - t0:.1f} s)",
              flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(CFG.vqe.grad_var_sizes))
    parser.add_argument("--samples", type=int, default=CFG.vqe.grad_var_samples)
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    for family in args.families:
        d = load_or_compute(f"bp_{family}", lambda: compute_family(family, args.sizes, args.samples),
                            params={"experiment": "barren_plateau", "family": family,
                                    "sizes": args.sizes, "samples": args.samples},
                            subdir=SUBDIR, recompute=args.recompute)
        for cost in COSTS:
            e = fit_exponential_decay(d["sizes"], d[f"{cost}_var_center"])
            p = fit_power_law(d["sizes"], d[f"{cost}_var_center"])
            print(f"  {family:11s} {cost:6s}: Var ~ exp(-b N), b = {e['b']:.3f} +/- {e['b_err']:.3f}"
                  f"   |   Var ~ N^(-k), k = {p['k']:.2f} +/- {p['k_err']:.2f}")
    print(f"  reference: unitary 2-design, b = ln 2 = {np.log(2):.3f}")


if __name__ == "__main__":
    main()
