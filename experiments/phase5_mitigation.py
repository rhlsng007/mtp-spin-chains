"""
Phase 5 experiments: readout mitigation and zero-noise extrapolation.

Experiment "zne" (T5.2, T5.3; experiment-A style, as decided in Phase 4)
    The Phase 2 optimum of every (ansatz, N, p <= N/2) is evaluated at every
    noisy-grid field (20 fields), for the four Phase 5 channels and the 11
    Phase 5 strengths, at scale factors 1, 3, 5 with global and local folding.
    For every circuit the exact density matrix gives three exact-limit energies:
        E_true : Tr(rho H), ideal readout
        E_raw  : measured through the readout model (unmitigated)
        E_ro   : readout-corrected with the exact calibrated matrix (which
                 contains the SPAM error of the noisy calibration X gates)
    and the fidelity with the noiseless circuit state. For the main folding,
    finite-shot energies (CFG.mitigation.shots per basis and per scale factor,
    shot_repetitions repetitions, each with its own calibration run) are drawn:
        Es_raw, Es_ro (full calibration, 2^N circuits), Es_tens (tensored, 2 circuits)
    Extrapolation is done when plotting, from these saved scaled energies.
    One file per (ansatz, N, p, channel): results/phase5/zne_{ansatz}_N{N}_p{p}_{channel}.

Experiment "readout" (T5.1)
    Zero gate noise, p = N/2, h = CFG.mitigation.readout_field. Calibration
    shots are swept; the corrected distributions (full / tensored, inverse /
    projected) are compared with the ideal ones (total-variation distance) and
    the corrected energies with the noiseless circuit energy, with exact and
    finite-shot measured distributions.
    One file per (ansatz, N): results/phase5/readout_{ansatz}_N{N}.

Seeds: SeedSequence([2026, 13, ...]) for zne, ([2026, 14, ...]) for readout.
Measurement streams depend on (ansatz, N, p, field) only and calibration
streams on N only, so they are shared by every channel and strength (common
random numbers), as in Phase 4.

Run from the project root:
    python experiments\\phase5_mitigation.py --workers 12
    python experiments\\phase5_mitigation.py --experiments readout
Options: --experiments zne readout, --sizes, --ansatze, --channels, --depths, --recompute.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker process

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import CFG
from models.hamiltonians import hamiltonian_terms
from noise.data import load_phase2
from noise.density import gate_sequence, simulate_statevector
from noise.mitigation import (
    PHASE5_CHANNELS, ReadoutEnergy, ReadoutModel, calibration_matrices, correct_probabilities,
    fold_gates, inverse_transpose, kron_all, make_mitigation_noise_model, negative_mass,
    noisy_gate_counts, prepared_one_probability, project_to_simplex, sample_full_calibration,
    sample_tensored_calibration, simulate_gates, total_variation,
)
from utils.io import result_exists, save_results
from vqe.optimize import ANSATZ_KEYS

SUBDIR = "phase5"
EXP_KEYS = {"zne": 13, "readout": 14}
READOUT_METHODS = ("raw", "full_inverse", "full_projected", "tensored_inverse", "tensored_projected")


def _seed(*keys) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([CFG.seeds.master_seed, *keys]))


def _terms(N: int, h: float):
    return hamiltonian_terms("tfim", N, J=CFG.tfim.J, h=float(h), boundary=CFG.tfim.boundary)


# ---------------------------------------------------------------------------
# Experiment "zne"
# ---------------------------------------------------------------------------
def zne_name(ansatz, N, p, channel):
    return f"zne_{ansatz}_N{N}_p{p}_{channel}"


def run_zne(ansatz: str, N: int, p: int, channel: str, recompute: bool = False):
    name = zne_name(ansatz, N, p, channel)
    if result_exists(name, SUBDIR) and not recompute:
        return name, "cached", 0.0
    t0 = time.time()
    cfg = CFG.mitigation
    key = EXP_KEYS["zne"]
    d2 = load_phase2(ansatz, N, p)
    hs = np.asarray(d2["h"], dtype=float)
    strengths = cfg.strengths()
    scales = np.asarray(CFG.noise.zne_scale_factors, dtype=float)
    folds = tuple(cfg.foldings)
    main_l = folds.index(cfg.main_folding)
    R, M, Mcal = cfg.shot_repetitions, cfg.shots, cfg.calibration_shots
    nF, nS, nK, nL = len(hs), len(strengths), len(scales), len(folds)
    readout = ReadoutModel.from_config(N)
    akey = ANSATZ_KEYS[ansatz]

    pipes = [ReadoutEnergy(_terms(N, h), N, readout) for h in hs]
    G = pipes[0].n_groups
    base = [gate_sequence(ansatz, N, p, th) for th in d2["params"]]
    folded = [[[fold_gates(b, lam, m) for lam in scales] for m in folds] for b in base]
    psis = [simulate_statevector(ansatz, N, p, th) for th in d2["params"]]
    E_ideal = np.array([pp.energy(pp.ideal_probabilities(np.outer(ps, ps.conj())))
                        for pp, ps in zip(pipes, psis)])

    shape = (nS, nF, nL, nK)
    E_true, E_raw, E_ro, fid = (np.full(shape, np.nan) for _ in range(4))
    Es_raw, Es_ro, Es_tens = (np.full((nS, nF, nK, R), np.nan) for _ in range(3))
    var_raw, var_ro = np.full((nS, nF, nK), np.nan), np.full((nS, nF, nK), np.nan)
    asym_true = np.array([pp.mixed_state_energy(readout=False) for pp in pipes])
    asym_raw = np.array([pp.mixed_state_energy(readout=True) for pp in pipes])
    asym_ro = np.full((nS, nF), np.nan)
    calib_f = np.full(nS, np.nan)

    for si, s in enumerate(strengths):
        nm = make_mitigation_noise_model(channel, s)
        nm_cal = nm if cfg.calibration_gate_noise else None
        C = calibration_matrices(readout, nm_cal)
        A = kron_all(C)
        calib_f[si] = prepared_one_probability(nm_cal)
        rng_cal = _seed(key, 0, N)                       # device-level stream (common random numbers)
        A_hat = sample_full_calibration(A, Mcal, rng_cal, R)
        C_hat = sample_tensored_calibration(C, Mcal, rng_cal, R)
        invT_exact = inverse_transpose(A)
        invT_hat = inverse_transpose(A_hat)
        for f, pipe in enumerate(pipes):
            w_ro = pipe.weights_from_inverse_T(invT_exact)
            w_full = pipe.weights_from_inverse_T(invT_hat)
            w_tens = pipe.weights_tensored(C_hat)
            asym_ro[si, f] = pipe.mixed_state_energy(w_ro, readout=True)
            Q_main = np.empty((nK, G, 2 ** N))
            rho_cache, rho_scale1 = None, None
            for li in range(nL):
                for ki in range(nK):
                    if nm.is_noiseless and rho_cache is not None:
                        rho = rho_cache                  # folding is exact without noise
                    elif scales[ki] == 1 and rho_scale1 is not None:
                        rho = rho_scale1                 # scale 1 is the same circuit for every folding
                    else:
                        rho = simulate_gates(folded[f][li][ki], N, nm)
                        rho_cache = rho
                        if scales[ki] == 1:
                            rho_scale1 = rho
                    P = pipe.ideal_probabilities(rho)
                    Q = pipe.measured_probabilities(P)
                    E_true[si, f, li, ki] = pipe.energy(P)
                    E_raw[si, f, li, ki] = pipe.energy(Q)
                    E_ro[si, f, li, ki] = pipe.energy(Q, w_ro)
                    fid[si, f, li, ki] = float(np.real(np.vdot(psis[f], rho @ psis[f])))
                    if li == main_l:
                        Q_main[ki] = Q
            var_raw[si, f] = pipe.shot_variance(Q_main, M)
            var_ro[si, f] = pipe.shot_variance(Q_main, M, w_ro)
            rng = _seed(key, akey, N, p, f)               # same stream for every channel and strength
            counts = pipe.sample_counts(Q_main, M, rng, R)          # (R, K, G, d)
            Es_raw[si, f] = pipe.energy_from_counts(counts, M).T
            Es_ro[si, f] = pipe.energy_from_counts(counts, M, w_full).T
            Es_tens[si, f] = pipe.energy_from_counts(counts, M, w_tens).T

    counts_gates = noisy_gate_counts(base[0], make_mitigation_noise_model(channel, 1e-3))
    arrays = dict(
        h=hs, strengths=strengths, scales=scales, foldings=np.array(folds), channel=np.array(channel),
        E0=np.asarray(d2["E0"], float), phase2_energy=np.asarray(d2["energy"], float), E_ideal=E_ideal,
        E_true=E_true, E_raw=E_raw, E_ro=E_ro, fidelity_ideal=fid,
        Es_raw=Es_raw, Es_ro=Es_ro, Es_tens=Es_tens, var_raw=var_raw, var_ro=var_ro,
        asym_true=asym_true, asym_raw=asym_raw, asym_ro=asym_ro, calib_f=calib_f,
        n_1q=counts_gates["n_1q"], n_2q=counts_gates["n_2q"], n_qubits=N, depth=p,
        shots=M, calibration_shots=Mcal, repetitions=R,
        readout_p01=cfg.readout_p01, readout_p10=cfg.readout_p10,
        seconds=time.time() - t0,
    )
    params = dict(experiment="zne", ansatz=ansatz, N=N, p=p, channel=channel)
    save_results(name, arrays, params=params, subdir=SUBDIR)
    return name, "done", time.time() - t0


# ---------------------------------------------------------------------------
# Experiment "readout"
# ---------------------------------------------------------------------------
def readout_name(ansatz, N):
    return f"readout_{ansatz}_N{N}"


def _corrections(q, A_hat, C_hat):
    """Corrected distributions for every method; q (R, G, d), A_hat (R, d, d), C_hat (R, N, 2, 2)."""
    R, G, d = q.shape
    full_inv = np.linalg.solve(A_hat, np.swapaxes(q, 1, 2))           # (R, d, G)
    full_inv = np.swapaxes(full_inv, 1, 2)
    tens_inv = np.stack([correct_probabilities(q[r], local_mats=C_hat[r]) for r in range(R)])
    proj = lambda x: project_to_simplex(x.reshape(-1, d)).reshape(x.shape)
    return {"raw": q, "full_inverse": full_inv, "full_projected": proj(full_inv),
            "tensored_inverse": tens_inv, "tensored_projected": proj(tens_inv)}


def run_readout(ansatz: str, N: int, recompute: bool = False):
    name = readout_name(ansatz, N)
    if result_exists(name, SUBDIR) and not recompute:
        return name, "cached", 0.0
    t0 = time.time()
    cfg = CFG.mitigation
    key = EXP_KEYS["readout"]
    p = N // 2
    h = cfg.readout_field
    d2 = load_phase2(ansatz, N, p, h_sel=[h])
    theta = d2["params"][0]
    psi = simulate_statevector(ansatz, N, p, theta)
    readout = ReadoutModel.from_config(N)
    pipe = ReadoutEnergy(_terms(N, h), N, readout)
    P = pipe.ideal_probabilities(np.outer(psi, psi.conj()))          # (G, d)
    Q = pipe.measured_probabilities(P)
    C = calibration_matrices(readout, None)                           # zero gate noise
    A = kron_all(C)
    E_ideal = float(pipe.energy(P))
    E0 = float(d2["E0"][0])
    E_raw_exact = float(pipe.energy(Q))
    E_full_exact = float(pipe.energy(Q, pipe.weights_from_inverse_T(inverse_transpose(A))))
    E_tens_exact = float(pipe.energy(Q, pipe.weights_tensored(C)))
    p_full_exact = correct_probabilities(Q, A)
    R, M = cfg.shot_repetitions, cfg.shots
    cal = tuple(cfg.readout_cal_shots)
    nM = len(READOUT_METHODS)
    tvd = np.full((2, len(cal), nM, R, pipe.n_groups), np.nan)
    neg = np.zeros_like(tvd)
    dE = np.full((2, len(cal), nM, R), np.nan)
    akey = ANSATZ_KEYS[ansatz]
    rng_meas = _seed(key, akey, N, 0)
    q_shots = pipe.sample_counts(Q, M, rng_meas, R) / M                # (R, G, d), shared by all M_cal
    for ci, Mcal in enumerate(cal):
        rng = _seed(key, 0, N, ci + 1)
        A_hat = sample_full_calibration(A, Mcal, rng, R)
        C_hat = sample_tensored_calibration(C, Mcal, rng, R)
        for si, q in enumerate((np.broadcast_to(Q, (R,) + Q.shape), q_shots)):
            corr = _corrections(np.ascontiguousarray(q), A_hat, C_hat)
            for mi, m in enumerate(READOUT_METHODS):
                pc = corr[m]
                tvd[si, ci, mi] = total_variation(pc, P[None])
                neg[si, ci, mi] = negative_mass(pc)
                dE[si, ci, mi] = (pipe.energy(pc) - E_ideal) / abs(E0)
    arrays = dict(
        h=h, depth=p, n_qubits=N, cal_shots=np.array(cal), methods=np.array(READOUT_METHODS),
        sources=np.array(("exact", "shots")), A_true=readout.matrix(), C_true=C,
        E0=E0, E_ideal=E_ideal, phase2_energy=float(d2["energy"][0]),
        E_raw_exact=E_raw_exact, E_full_exact=E_full_exact, E_tens_exact=E_tens_exact,
        tvd_raw_exact=total_variation(Q, P), tvd_full_exact=total_variation(p_full_exact, P),
        tvd=tvd, neg_mass=neg, dE=dE, shots=M, repetitions=R,
        readout_p01=cfg.readout_p01, readout_p10=cfg.readout_p10, seconds=time.time() - t0,
    )
    save_results(name, arrays, params=dict(experiment="readout", ansatz=ansatz, N=N, p=p, h=h),
                 subdir=SUBDIR)
    return name, "done", time.time() - t0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _job(task):
    kind, args, recompute = task
    if kind == "zne":
        return run_zne(*args, recompute=recompute)
    return run_readout(*args, recompute=recompute)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", default=["zne", "readout"], choices=["zne", "readout"])
    ap.add_argument("--sizes", nargs="+", type=int, default=list(CFG.system.sizes_noisy))
    ap.add_argument("--ansatze", nargs="+", default=list(CFG.vqe.ansatze))
    ap.add_argument("--channels", nargs="+", default=list(CFG.mitigation.channels), choices=PHASE5_CHANNELS)
    ap.add_argument("--depths", nargs="+", type=int, default=None, help="default: 1..N/2")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    tasks = []
    if "readout" in args.experiments:
        for N in args.sizes:
            for a in args.ansatze:
                tasks.append(("readout", (a, N), args.recompute))
    if "zne" in args.experiments:
        for N in args.sizes:
            depths = args.depths or CFG.system.depths(N, noisy=True)
            for p in depths:
                if p > N // 2:
                    continue
                for a in args.ansatze:
                    for c in args.channels:
                        tasks.append(("zne", (a, N, p, c), args.recompute))
    # largest jobs first (cost ~ 4^N * depth) for load balance
    tasks.sort(key=lambda t: -(4 ** t[1][1]) * (t[1][2] if t[0] == "zne" else 0.1))
    print(f"Phase 5: {len(tasks)} jobs, {args.workers} workers")
    t0 = time.time()
    if args.workers <= 1:
        for i, t in enumerate(tasks, 1):
            name, status, sec = _job(t)
            print(f"[{i}/{len(tasks)}] {name}: {status} ({sec:.1f} s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_job, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                name, status, sec = fut.result()
                print(f"[{i}/{len(tasks)}] {name}: {status} ({sec:.1f} s)", flush=True)
    print(f"total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
