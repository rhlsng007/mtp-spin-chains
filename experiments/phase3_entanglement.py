"""
Phase 3, tasks T3.1 to T3.4: entanglement of the VQE states.

Run from the project root, after Phases 1 and 2:

    python experiments\\phase3_entanglement.py --workers 12
    python experiments\\phase3_entanglement.py --experiments states ed      # analysis of saved states only
    python experiments\\phase3_entanglement.py --sizes 4 6 8 --recompute

Experiments (results/phase3/, cached per file):

    ed       ent_ed_N{N}             ED even, odd and symmetry-broken entropy profiles over the
                                     39 fields, ground-space mask, Phase 1 S_half for the check
    states   ent_{ansatz}_N{N}       for every saved Phase 2 state (all p, all fields):
                                     S(ell), S_2(ell), Schmidt rank(ell); the same for its
                                     ground-space reference; F_GS; Fannes-Audenaert bound;
                                     Schmidt-rank fidelity bound            (T3.1, T3.2, T3.4)
    maxent   maxent_{ansatz}_N{N}    largest half-chain S_2 per depth (purity minimization),
                                     plus statistics of uniformly random parameters  (T3.3)
    pbc      pbc_hva_N{N}            periodic HVA at h_c for p = 1..N/2: states and profiles,
                                     for central charges free of open-boundary bias  (T3.2)
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import CFG
from analysis.vqe_entanglement import (audenaert_bound, maximize_entanglement, profile_data,
                                       projected_reference, rank_fidelity_bound,
                                       symmetry_broken_state)
from models.hamiltonians import build_hamiltonian
from utils.io import load_or_compute, load_results, result_exists
from vqe.ansatz import HVA, make_ansatz
from vqe.metrics import ed_reference, ground_space_mask, reference_energy
from vqe.optimize import ANSATZ_KEYS, optimize_point, point_rng, random_candidates

SUBDIR = "phase3"
PHASE2 = "phase2"
EXPERIMENT_KEYS = {"maxent": 8, "random": 9, "pbc": 10}
ALL_EXPERIMENTS = ("ed", "states", "maxent", "pbc")


def even_odd(ref: dict, i: int) -> tuple:
    """(phi_even, phi_odd) at field index i; the ground sector of the TFIM is the even one."""
    gs = int(ref["ground_sector"][i])
    return ref["states"][i, gs], ref["states"][i, 1 - gs]


# ---------------------------------------------------------------------------
# ED references
# ---------------------------------------------------------------------------
def compute_ed(n: int) -> dict:
    h = CFG.tfim.h_grid()
    ref = ed_reference(n, h, verbose=False)
    tol = CFG.ent.rank_tol
    out = {k: [] for k in ("S_even", "S2_even", "S_odd", "S_broken", "S2_broken")}
    for i in range(len(h)):
        e, o = even_odd(ref, i)
        de, do, db = (profile_data(x, tol) for x in (e, o, symmetry_broken_state(e, o)))
        out["S_even"].append(de["S"])
        out["S2_even"].append(de["S2"])
        out["S_odd"].append(do["S"])
        out["S_broken"].append(db["S"])
        out["S2_broken"].append(db["S2"])
    arrays = {k: np.array(v) for k, v in out.items()}
    arrays.update({"h": h, "n_qubits": n, "ground_space": ground_space_mask(ref),
                   "E0": reference_energy(ref)})
    if result_exists(f"ed_tfim_N{n}", "phase1"):
        p1 = load_results(f"ed_tfim_N{n}", "phase1")[0]
        if len(p1["h"]) == len(h) and np.allclose(p1["h"], h):
            arrays["S_half_phase1"] = p1["S_half"]
    return arrays


def job_ed(n: int, recompute: bool) -> str:
    load_or_compute(f"ent_ed_N{n}", lambda: compute_ed(n),
                    params={"experiment": "ed", "N": n}, subdir=SUBDIR, recompute=recompute)
    return f"ED N={n} done"


# ---------------------------------------------------------------------------
# Entanglement of the saved Phase 2 states
# ---------------------------------------------------------------------------
def compute_states(ansatz_name: str, n: int) -> dict:
    tol = CFG.ent.rank_tol
    h = CFG.tfim.h_grid()
    ref = ed_reference(n, h, verbose=False)
    mask = ground_space_mask(ref)
    depths = CFG.system.depths(n)
    shape_p = (len(depths), len(h), n - 1)
    keys_prof = ("S", "S2", "rank", "S_ref", "S2_ref")
    out = {k: np.zeros(shape_p, dtype=int if k == "rank" else float) for k in keys_prof}
    for k in ("fidelity_gs", "audenaert", "rank_fidelity_bound"):
        out[k] = np.zeros((len(depths), len(h)))
    dim = 2 ** (n // 2)
    for a, p in enumerate(depths):
        name = f"vqe_{ansatz_name}_N{n}_p{p}"
        if not result_exists(name, PHASE2):
            raise FileNotFoundError(f"results/{PHASE2}/{name} missing: run Phase 2 first")
        states = load_results(name, PHASE2)[0]["states"]
        for i in range(len(h)):
            psi = states[i]
            d = profile_data(psi, tol)
            phi, f_gs = projected_reference(psi, ref["states"][i], mask[i])
            dr = profile_data(phi, tol)
            for k in ("S", "S2", "rank"):
                out[k][a, i] = d[k]
            out["S_ref"][a, i] = dr["S"]
            out["S2_ref"][a, i] = dr["S2"]
            out["fidelity_gs"][a, i] = f_gs
            out["audenaert"][a, i] = audenaert_bound(f_gs, dim)
            out["rank_fidelity_bound"][a, i] = rank_fidelity_bound(phi, 2 ** p)
    out.update({"h": h, "depths": np.array(depths), "n_qubits": n})
    return out


def job_states(ansatz_name: str, n: int, recompute: bool) -> str:
    t0 = time.perf_counter()
    d = load_or_compute(f"ent_{ansatz_name}_N{n}", lambda: compute_states(ansatz_name, n),
                        params={"experiment": "states", "ansatz": ansatz_name, "N": n},
                        subdir=SUBDIR, recompute=recompute)
    half = n // 2 - 1
    dS = np.abs(d["S"][-1, :, half] - d["S_ref"][-1, :, half])
    print(f"  ent_{ansatz_name}_N{n}: at p = N, max |S - S_ref| (half chain) = {dS.max():.2e} "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)
    return f"states {ansatz_name} N={n} done"


# ---------------------------------------------------------------------------
# Maximal entanglement per depth (T3.3)
# ---------------------------------------------------------------------------
def compute_maxent(ansatz_name: str, n: int) -> dict:
    c = CFG.ent
    depths = CFG.system.depths(n)
    ak = ANSATZ_KEYS[ansatz_name]
    out = {k: np.zeros(len(depths)) for k in
           ("S2_max", "S_at_S2_max", "random_S_mean", "random_S_max", "random_S2_mean",
            "n_iter_best")}
    out["S2_starts"] = np.zeros((len(depths), c.maxent_starts))
    for a, p in enumerate(depths):
        ansatz = make_ansatz(ansatz_name, n, p)
        # statistics of uniformly random parameters
        rng = point_rng(EXPERIMENT_KEYS["random"], ak, n, p)
        s_vals, s2_vals = [], []
        for _ in range(c.random_samples):
            d = profile_data(ansatz.state(ansatz.random_params(rng, 1.0, "uniform")), c.rank_tol)
            s_vals.append(d["S"][n // 2 - 1])
            s2_vals.append(d["S2"][n // 2 - 1])
        out["random_S_mean"][a] = np.mean(s_vals)
        out["random_S_max"][a] = np.max(s_vals)
        out["random_S2_mean"][a] = np.mean(s2_vals)
        # purity minimization from uniform starts
        rng = point_rng(EXPERIMENT_KEYS["maxent"], ak, n, p)
        runs = [maximize_entanglement(ansatz, ansatz.random_params(rng, 1.0, "uniform"),
                                      c.maxent_maxiter, c.maxent_gtol)
                for _ in range(c.maxent_starts)]
        best = max(runs, key=lambda r: r["S2"])
        out["S2_starts"][a] = [r["S2"] for r in runs]
        out["S2_max"][a] = best["S2"]
        out["S_at_S2_max"][a] = best["S"]
        out["n_iter_best"][a] = best["n_iter"]
    out.update({"depths": np.array(depths), "n_qubits": n})
    return out


def job_maxent(ansatz_name: str, n: int, recompute: bool) -> str:
    t0 = time.perf_counter()
    d = load_or_compute(f"maxent_{ansatz_name}_N{n}", lambda: compute_maxent(ansatz_name, n),
                        params={"experiment": "maxent", "ansatz": ansatz_name, "N": n},
                        subdir=SUBDIR, recompute=recompute)
    ratio = d["S2_max"] / (np.minimum(d["depths"], n // 2) * np.log(2))
    print(f"  maxent_{ansatz_name}_N{n}: S2_max / bound = "
          + " ".join(f"{r:.3f}" for r in ratio) + f"  ({time.perf_counter() - t0:.0f} s)",
          flush=True)
    return f"maxent {ansatz_name} N={n} done"


# ---------------------------------------------------------------------------
# Periodic HVA at h_c (T3.2 without open-boundary bias)
# ---------------------------------------------------------------------------
def compute_pbc(n: int) -> dict:
    c = CFG.vqe
    h = float(CFG.ent.cc_field)
    H = build_hamiltonian("tfim", n, J=CFG.tfim.J, h=h, boundary="periodic")
    ref = ed_reference(n, np.array([h]), boundary="periodic", verbose=False)
    e0 = float(reference_energy(ref)[0])
    phi_even = ref["states"][0, int(ref["ground_sector"][0])]
    depths = np.arange(1, n // 2 + 1)
    energy, fid, n_iter = np.zeros(len(depths)), np.zeros(len(depths)), np.zeros(len(depths), int)
    S, S2 = np.zeros((len(depths), n - 1)), np.zeros((len(depths), n - 1))
    states = np.zeros((len(depths), 2 ** n), dtype=complex)
    prev = None
    for a, p in enumerate(depths):
        ansatz = HVA(n, int(p), boundary="periodic")
        cands = [] if prev is None else [("depth", HVA.pad(prev, int(p)))]
        cands += random_candidates(ansatz, c.n_random_inits,
                                   point_rng(EXPERIMENT_KEYS["pbc"], n, int(p)))
        best = optimize_point(ansatz, H, cands).best
        prev = best.x
        psi = ansatz.state(best.x)
        d = profile_data(psi, CFG.ent.rank_tol)
        energy[a], n_iter[a] = best.energy, best.n_iter
        fid[a] = abs(np.vdot(phi_even, psi)) ** 2
        S[a], S2[a], states[a] = d["S"], d["S2"], psi
    ed = profile_data(phi_even, CFG.ent.rank_tol)
    return {"h": h, "n_qubits": n, "depths": depths, "energy": energy, "E0": e0,
            "rel_error": (energy - e0) / abs(e0), "fidelity": fid, "n_iter": n_iter,
            "S": S, "S2": S2, "S_ed": ed["S"], "S2_ed": ed["S2"], "states": states}


def job_pbc(n: int, recompute: bool) -> str:
    d = load_or_compute(f"pbc_hva_N{n}", lambda: compute_pbc(n),
                        params={"experiment": "pbc", "N": n, "h": CFG.ent.cc_field},
                        subdir=SUBDIR, recompute=recompute)
    print(f"  pbc_hva_N{n}: eps at p = N/2 = {d['rel_error'][-1]:.1e}, "
          f"F = {d['fidelity'][-1]:.10f}", flush=True)
    return f"pbc N={n} done"


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------
def build_jobs(experiments, sizes, ansatze, recompute) -> list:
    jobs = []
    for n in sizes:
        if "ed" in experiments:
            jobs.append((2.0 ** n, job_ed, (n, recompute)))
        for a in ansatze:
            if "states" in experiments:
                jobs.append((2.0 ** n * n, job_states, (a, n, recompute)))
            if "maxent" in experiments:
                weight = n * n * (n + 1) if a == "hea" else n * n
                jobs.append((2.0 ** n * weight, job_maxent, (a, n, recompute)))
        if "pbc" in experiments and "hva" in ansatze and n in CFG.ent.pbc_sizes:
            jobs.append((2.0 ** n * n, job_pbc, (n, recompute)))
    return sorted(jobs, key=lambda j: -j[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(CFG.system.sizes_noiseless))
    parser.add_argument("--ansatze", nargs="+", default=list(CFG.vqe.ansatze), choices=["hva", "hea"])
    parser.add_argument("--experiments", nargs="+", default=list(ALL_EXPERIMENTS),
                        choices=list(ALL_EXPERIMENTS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    print(f"Phase 3: sizes {args.sizes}, ansatze {args.ansatze}, experiments {args.experiments}, "
          f"{args.workers} worker(s)")
    jobs = build_jobs(args.experiments, args.sizes, args.ansatze, args.recompute)
    t0 = time.perf_counter()
    if args.workers <= 1:
        for _, fn, fargs in jobs:
            fn(*fargs)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fn, *fargs) for _, fn, fargs in jobs]
            for fut in as_completed(futures):
                print(f"[done] {fut.result()}", flush=True)
    print(f"Finished in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
