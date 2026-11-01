"""
Phase 4: noisy density-matrix simulation of the HVA and HEA (N = 4, 6, 8).

Experiment A ("eval", T4.5 first part, plus T4.2 and T4.6):
    the noiseless Phase 2 optimum at depth p (p = 1..N/2) and every field of
    the noisy grid is run under each channel and noise strength. All metrics
    are computed from the exact density matrix; in addition, finite-shot
    energy estimates (CFG.noise.shots_list, CFG.noise.shot_repetitions each)
    are drawn from the same density matrix.
    File: results/phase4/noisy_eval_{ansatz}_N{N}_p{p}

Experiment B ("train", T4.5 second part, T4.3):
    SPSA optimization under noise with finite shots (CFG.noise.train_shots)
    at p = N/2 and the fields CFG.noise.train_fields. Start 0 is warm (the
    Phase 2 optimum, gain x spsa_warm_factor); starts 1..n are cold (normal,
    scale CFG.vqe.init_scale). The SPSA gain is calibrated once per problem at
    the first cold point with exact noiseless energies and shared by every
    start, channel and strength; initial points, SPSA directions and shot
    streams do not depend on channel or strength (common random numbers).
    File: results/phase4/noisy_train_{ansatz}_N{N}_h{h:.2f}_s{start}

Usage (from the project root):
    python experiments\\phase4_noise.py --workers 12
    python experiments\\phase4_noise.py --experiments eval --sizes 4 6
    python experiments\\phase4_noise.py --experiments train --workers 12
Each file is cached; interrupted runs resume. --recompute forces a rerun.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")       # one BLAS thread per worker process

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import CFG, RESULTS_DIR
from noise.channels import make_noise_model
from noise.data import exact_reference, load_phase2, load_phase3_half_entropy
from noise.density import gate_counts, simulate_density_matrix, simulate_statevector
from noise.evaluate import PROFILE_KEYS, SCALAR_KEYS, evaluate_state
from noise.shots import EnergyEstimator
from utils.io import save_results
from vqe.optimize import ANSATZ_KEYS
from vqe.spsa import calibrate_gain, spsa_minimize

SUBDIR = "phase4"
EXP_KEYS = {"eval": 11, "train": 12}        # Phase 3 used 8, 9, 10
TRAIN_SCALARS = ("energy", "rel_error", "fidelity", "fidelity_gs", "fidelity_ideal",
                 "purity", "S2_total", "S_A", "S2_A", "I", "I2", "E_N", "S_cond")
HISTORY_STRIDE = 10


def _exists(name: str) -> bool:
    return (RESULTS_DIR / SUBDIR / f"{name}.npz").exists()


def _seed(*keys) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([CFG.seeds.master_seed, *map(int, keys)]))


def eval_name(ansatz, N, p):
    return f"noisy_eval_{ansatz}_N{N}_p{p}"


def train_name(ansatz, N, h, start):
    return f"noisy_train_{ansatz}_N{N}_h{h:.2f}_s{start}"


# ---------------------------------------------------------------------------
# Experiment A: noiseless optimum evaluated under noise
# ---------------------------------------------------------------------------
def run_eval(ansatz: str, N: int, p: int) -> str:
    t0 = time.time()
    hs = CFG.tfim.h_grid(noisy=True)
    P2 = load_phase2(ansatz, N, p, hs)
    P3 = load_phase3_half_entropy(ansatz, N, p, hs)
    refs = [exact_reference(N, h, P2["ground_space"][i]) for i, h in enumerate(hs)]
    ests = [EnergyEstimator(r.terms, N) for r in refs]
    chans, strengths = CFG.noise.channels, CFG.noise.strengths()
    shots, R = CFG.noise.shots_list, CFG.noise.shot_repetitions
    nc, ns, nh = len(chans), len(strengths), len(hs)

    out = {k: np.full((nc, ns, nh), np.nan) for k in SCALAR_KEYS}
    out.update({k: np.full((nc, ns, nh, N - 1), np.nan) for k in PROFILE_KEYS})
    out["E_shots"] = np.zeros((nc, ns, nh, len(shots), R))
    out["E_sigma"] = np.zeros((nc, ns, nh, len(shots)))
    out["estimator_err"] = np.zeros((nc, ns, nh))
    out["repro_state_err"] = np.zeros(nh)
    akey = ANSATZ_KEYS[ansatz]

    for ih in range(nh):
        theta = P2["params"][ih]
        psi0 = np.asarray(P2["states"][ih], dtype=complex)
        for ic, ch in enumerate(chans):
            for js, s in enumerate(strengths):
                rho = simulate_density_matrix(ansatz, N, p, theta, make_noise_model(ch, s))
                if js == 0 and ic == 0:
                    out["repro_state_err"][ih] = np.abs(rho - np.outer(psi0, psi0.conj())).max()
                m = evaluate_state(rho, refs[ih], psi0, N, profiles=True)
                for k in SCALAR_KEYS + PROFILE_KEYS:
                    out[k][ic, js, ih] = m[k]
                probs = ests[ih].probabilities(rho)
                out["estimator_err"][ic, js, ih] = abs(ests[ih].exact(rho, probs) - m["energy"])
                for k, M in enumerate(shots):
                    rng = _seed(EXP_KEYS["eval"], akey, N, p, ic, js, ih, k)
                    out["E_shots"][ic, js, ih, k] = ests[ih].sample(rho, M, rng, probs, repetitions=R)
                    out["E_sigma"][ic, js, ih, k] = np.sqrt(ests[ih].variance(rho, M, probs))

    gc = gate_counts(ansatz, N, p)
    out.update({
        "h": hs, "strengths": strengths, "channels": np.array(chans), "shots": np.array(shots),
        "E0": np.array([r.E0 for r in refs]),
        "phase2_energy": P2["energy"], "phase2_fidelity": P2["fidelity"],
        "phase2_fidelity_gs": P2["fidelity_gs"],
        "phase3_S": np.full(nh, np.nan) if P3 is None else P3["S"],
        "phase3_S2": np.full(nh, np.nan) if P3 is None else P3["S2"],
        "n_1q": gc["n_1q"], "n_2q": gc["n_2q"], "n_qubits": N, "depth": p,
        "seconds": time.time() - t0,
    })
    name = eval_name(ansatz, N, p)
    save_results(name, out, {"experiment": "eval", "ansatz": ansatz, "N": N, "p": p,
                             "exp_key": EXP_KEYS["eval"]}, SUBDIR)
    return f"{name}: {time.time() - t0:.0f} s"


# ---------------------------------------------------------------------------
# Experiment B: SPSA training under noise
# ---------------------------------------------------------------------------
def spsa_settings(ansatz: str) -> tuple[float, float]:
    v = CFG.vqe
    return (v.spsa_target_hva, v.spsa_c_hva) if ansatz == "hva" else (v.spsa_target_hea, v.spsa_c_hea)


def cold_point(ansatz, N, p, ihf, k, n_params):
    rng = _seed(EXP_KEYS["train"], ANSATZ_KEYS[ansatz], N, p, ihf, 1000 + k)
    return rng.normal(0.0, CFG.vqe.init_scale, n_params)


def problem_gain(ansatz, N, p, ihf, H, n_params) -> tuple[float, float]:
    """SPSA gain calibrated once per problem at the first cold point (exact, noiseless)."""
    target, c = spsa_settings(ansatz)
    v = CFG.vqe
    x0 = cold_point(ansatz, N, p, ihf, 0, n_params)

    def f0(x):
        psi = simulate_statevector(ansatz, N, p, x)
        return float(np.vdot(psi, H @ psi).real)

    return calibrate_gain(f0, x0, _seed(EXP_KEYS["train"], ANSATZ_KEYS[ansatz], N, p, ihf, 2000),
                          target, c, v.spsa_iterations, v.spsa_alpha,
                          v.spsa_A_fraction * v.spsa_iterations, v.spsa_calibration_steps)


def run_train(ansatz: str, N: int, ihf: int, start: int) -> str:
    t0 = time.time()
    v = CFG.vqe
    p = N // 2
    h = float(CFG.noise.train_fields[ihf])
    P2 = load_phase2(ansatz, N, p, [h])
    theta_opt = P2["params"][0]
    psi0 = np.asarray(P2["states"][0], dtype=complex)
    ref = exact_reference(N, h, P2["ground_space"][0])
    est = EnergyEstimator(ref.terms, N)
    n_params = theta_opt.size
    _, c = spsa_settings(ansatz)
    gain, mag = problem_gain(ansatz, N, p, ihf, ref.H, n_params)
    if start == 0:
        x0, g = np.array(theta_opt, dtype=float), gain * v.spsa_warm_factor
    else:
        x0, g = cold_point(ansatz, N, p, ihf, start - 1, n_params), gain
    shots = CFG.noise.train_shots

    chans, strengths = CFG.noise.channels, CFG.noise.strengths()
    nc, ns = len(chans), len(strengths)
    n_hist = int(np.ceil(v.spsa_iterations / HISTORY_STRIDE))
    out = {k: np.full((nc, ns), np.nan) for k in TRAIN_SCALARS}
    out.update({"evalA_energy": np.full((nc, ns), np.nan), "evalA_rel_error": np.full((nc, ns), np.nan),
                "initial_rel_error": np.full((nc, ns), np.nan),
                "history": np.full((nc, ns, n_hist), np.nan), "n_fev": np.zeros((nc, ns), int),
                "params": np.full((nc, ns, n_params), np.nan)})

    for ic, ch in enumerate(chans):
        for js, s in enumerate(strengths):
            if js == 0 and ic > 0:                       # zero noise: identical for all channels
                for k in out:
                    out[k][ic, 0] = out[k][0, 0]
                continue
            nm = make_noise_model(ch, s)
            rng = _seed(EXP_KEYS["train"], ANSATZ_KEYS[ansatz], N, p, ihf, start)

            def f(x):
                rho = simulate_density_matrix(ansatz, N, p, x, nm)
                return est.estimate(rho, shots, rng)

            res = spsa_minimize(f, x0, rng, maxiter=v.spsa_iterations, c=c, alpha=v.spsa_alpha,
                                gamma=v.spsa_gamma, A=v.spsa_A_fraction * v.spsa_iterations,
                                gain=g, average_last=v.spsa_average_last)
            m = evaluate_state(simulate_density_matrix(ansatz, N, p, res.x, nm), ref, psi0, N,
                               profiles=False)
            for k in TRAIN_SCALARS:
                out[k][ic, js] = m[k]
            rhoA = simulate_density_matrix(ansatz, N, p, theta_opt, nm)
            EA = float(np.real(np.sum(ref.H.multiply(rhoA.T))))
            out["evalA_energy"][ic, js] = EA
            out["evalA_rel_error"][ic, js] = (EA - ref.E0) / abs(ref.E0)
            rho_init = simulate_density_matrix(ansatz, N, p, x0, nm)
            out["initial_rel_error"][ic, js] = (float(np.real(np.sum(ref.H.multiply(rho_init.T)))) - ref.E0) / abs(ref.E0)
            out["history"][ic, js] = (res.history[::HISTORY_STRIDE] - ref.E0) / abs(ref.E0)
            out["n_fev"][ic, js] = res.n_fev
            out["params"][ic, js] = res.x

    out.update({"h": h, "E0": ref.E0, "strengths": strengths, "channels": np.array(chans),
                "start": start, "warm": int(start == 0), "gain": g, "calibration_magnitude": mag,
                "train_shots": -1 if shots is None else shots, "n_qubits": N, "depth": p,
                "phase2_rel_error": (P2["energy"][0] - ref.E0) / abs(ref.E0),
                "seconds": time.time() - t0})
    name = train_name(ansatz, N, h, start)
    save_results(name, out, {"experiment": "train", "ansatz": ansatz, "N": N, "p": p, "h": h,
                             "start": start, "exp_key": EXP_KEYS["train"]}, SUBDIR)
    return f"{name}: {time.time() - t0:.0f} s"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", default=["eval", "train"], choices=["eval", "train"])
    ap.add_argument("--sizes", nargs="+", type=int, default=list(CFG.system.sizes_noisy))
    ap.add_argument("--ansatze", nargs="+", default=list(CFG.vqe.ansatze))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    jobs = []           # (cost estimate, function, args, name)
    if "eval" in args.experiments:
        for N in args.sizes:
            for a in args.ansatze:
                for p in CFG.system.depths(N, noisy=True):
                    jobs.append((4.0 ** N * p, run_eval, (a, N, p), eval_name(a, N, p)))
    if "train" in args.experiments:
        n_starts = 1 + CFG.seeds.n_seeds_noisy
        for N in args.sizes:
            for a in args.ansatze:
                for ihf, h in enumerate(CFG.noise.train_fields):
                    for s in range(n_starts):
                        jobs.append((50 * 4.0 ** N * N, run_train, (a, N, ihf, s), train_name(a, N, h, s)))
    todo = [j for j in jobs if args.recompute or not _exists(j[3])]
    todo.sort(key=lambda j: -j[0])                  # longest first, for load balance
    print(f"Phase 4: {len(jobs)} jobs, {len(jobs) - len(todo)} cached, {len(todo)} to run "
          f"on {args.workers} worker(s)", flush=True)
    t0 = time.time()
    if args.workers <= 1:
        for k, (_, fn, fa, _) in enumerate(todo, 1):
            print(f"[{k}/{len(todo)}] {fn(*fa)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fn, *fa): nm for _, fn, fa, nm in todo}
            for k, fut in enumerate(as_completed(futs), 1):
                print(f"[{k}/{len(todo)}] {fut.result()}  (elapsed {time.time() - t0:.0f} s)", flush=True)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
