"""
Phase 2, tasks T2.4 to T2.6: noiseless VQE for the open TFIM chain.

Run from the project root (after Phase 1):

    python experiments\\phase2_vqe_sweep.py --quick              # N = 4, 6, 8 only
    python experiments\\phase2_vqe_sweep.py --workers 4          # everything, 4 processes
    python experiments\\phase2_vqe_sweep.py --experiments sweep  # only the main sweeps
    python experiments\\phase2_vqe_sweep.py --sizes 4 6 --ansatze hva --recompute

Experiments (all saved to results/phase2/, every file is cached, so an
interrupted run resumes where it stopped):

    sweep      vqe_{ansatz}_N{N}_p{p}         every N, every depth p = 1..N, all 39 fields;
                                              warm start + random restarts at each field
    warmcold   vqe_warmcold_{ansatz}_N{N}_p{p} N = 8, p = N/2: warm start only vs cold
                                              random restarts only (T2.5 validation)
    depthscan  vqe_hva_depthscan_N{N}         HVA at the anchor fields up to p = 2N
    pbc        vqe_hva_pbc_check              periodic HVA at p = N/2 (exact, known result)

Each sweep file holds, per field value: energy, E0, energy_error, rel_error,
fidelity (even-parity ED state), fidelity_sectors, fidelity_gs (ground space),
parity, n_iter, n_fev, total_fev, success, best_source (0 warm, 1 depth,
2 random), params, seconds, warm_energy, random_best_energy and states.
"""

import os

# Small matrices: one BLAS thread per process is fastest and avoids
# oversubscription when several worker processes run in parallel.
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
from models.hamiltonians import build_hamiltonian
from utils.io import load_or_compute, load_results, result_exists
from vqe.ansatz import HVA, make_ansatz
from vqe.metrics import ed_reference, evaluate_sweep, reference_energy
from vqe.optimize import ANSATZ_KEYS, field_sweep, optimize_point, point_rng, random_candidates

SUBDIR = "phase2"
EXPERIMENT_KEYS = {"sweep": 1, "cold": 2, "warm": 3, "depthscan": 4, "pbc": 5}
ALL_EXPERIMENTS = ("sweep", "warmcold", "depthscan", "pbc")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tfim_hamiltonians(n_sites: int, h_values, boundary: str = None) -> list:
    c = CFG.tfim
    boundary = c.boundary if boundary is None else boundary
    return [build_hamiltonian("tfim", n_sites, J=c.J, h=float(h), boundary=boundary)
            for h in h_values]


def anchor_mask(h_values) -> np.ndarray:
    return np.isin(np.round(h_values, 6), np.round(CFG.vqe.anchor_fields, 6))


def sweep_direction(ansatz_name: str) -> str:
    c = CFG.vqe
    return c.hva_sweep_direction if ansatz_name == "hva" else c.hea_sweep_direction


def sweep_name(ansatz_name: str, n_sites: int, depth: int) -> str:
    return f"vqe_{ansatz_name}_N{n_sites}_p{depth}"


def report(name: str, arrays: dict, seconds: float) -> None:
    rel = arrays["rel_error"]
    print(f"  {name:28s} max eps = {rel.max():.2e}  median eps = {np.median(rel):.2e}  "
          f"min F = {arrays['fidelity'].min():.4f}  min F_GS = {arrays['fidelity_gs'].min():.4f}  "
          f"median it = {np.median(arrays['n_iter']):.0f}  ({seconds:.0f} s)", flush=True)


# ---------------------------------------------------------------------------
# Main field sweeps (T2.4 - T2.6)
# ---------------------------------------------------------------------------
def compute_sweep(ansatz_name: str, n_sites: int, depth: int, prev_params=None) -> dict:
    c = CFG.vqe
    h = CFG.tfim.h_grid()
    ansatz = make_ansatz(ansatz_name, n_sites, depth)
    ref = ed_reference(n_sites, h, verbose=False)
    extra = None
    if ansatz_name == "hva" and c.hva_depth_warm_start and prev_params is not None:
        extra = np.array([HVA.pad(x, depth) for x in prev_params])
    anchors = anchor_mask(h)
    out = field_sweep(
        ansatz, tfim_hamiltonians(n_sites, h), sweep_direction(ansatz_name),
        rng_keys=(EXPERIMENT_KEYS["sweep"], ANSATZ_KEYS[ansatz_name], n_sites, depth),
        n_restarts_first=c.n_random_inits, n_restarts=c.n_random_inits_sweep,
        anchor_mask=anchors, n_restarts_anchor=c.n_random_inits,
        warm_start=c.use_warm_start, extra_inits=extra)
    metrics = evaluate_sweep(ansatz, out["params"], out["energy"], ref)
    if not c.save_states:
        metrics.pop("states")
    return {"h": h, "anchor": anchors, "n_qubits": n_sites, "depth": depth,
            "n_params": ansatz.n_params, **out, **metrics}


def run_sweep(ansatz_name: str, n_sites: int, depth: int, recompute: bool,
              prev_params=None) -> dict:
    name = sweep_name(ansatz_name, n_sites, depth)
    t0 = time.perf_counter()
    arrays = load_or_compute(
        name, lambda: compute_sweep(ansatz_name, n_sites, depth, prev_params),
        params={"experiment": "sweep", "model": "tfim", "ansatz": ansatz_name,
                "N": n_sites, "p": depth, "J": CFG.tfim.J, "boundary": CFG.tfim.boundary,
                "direction": sweep_direction(ansatz_name)},
        subdir=SUBDIR, recompute=recompute)
    report(name, arrays, time.perf_counter() - t0)
    return arrays


def job_hva_sweeps(n_sites: int, depths: list, recompute: bool) -> str:
    """HVA depths run in order, so depth p can start from the padded p-1 optimum."""
    prev = None
    for p in depths:
        name = sweep_name("hva", n_sites, p)
        if prev is None and p > 1 and result_exists(sweep_name("hva", n_sites, p - 1), SUBDIR):
            prev = load_results(sweep_name("hva", n_sites, p - 1), SUBDIR)[0]["params"]
        arrays = run_sweep("hva", n_sites, p, recompute, prev_params=prev)
        prev = arrays["params"]
    return f"HVA N={n_sites} done"


def job_hea_sweep(n_sites: int, depth: int, recompute: bool) -> str:
    run_sweep("hea", n_sites, depth, recompute)
    return f"HEA N={n_sites} p={depth} done"


# ---------------------------------------------------------------------------
# Warm start versus cold start (T2.5 validation)
# ---------------------------------------------------------------------------
def compute_warmcold(ansatz_name: str) -> dict:
    c = CFG.vqe
    n = c.warmcold_size
    depth = max(n // 2, 1)
    h = CFG.tfim.h_grid()
    ansatz = make_ansatz(ansatz_name, n, depth)
    ref = ed_reference(n, h, verbose=False)
    hams = tfim_hamiltonians(n, h)
    direction = sweep_direction(ansatz_name)
    ak = ANSATZ_KEYS[ansatz_name]
    runs = {
        # every point: n_random_inits independent random starts, no information passed on
        "cold": field_sweep(ansatz, hams, direction, (EXPERIMENT_KEYS["cold"], ak, n, depth),
                            n_restarts_first=c.n_random_inits, n_restarts=c.n_random_inits,
                            warm_start=False),
        # first point as above, then a single warm start per point
        "warm": field_sweep(ansatz, hams, direction, (EXPERIMENT_KEYS["warm"], ak, n, depth),
                            n_restarts_first=c.n_random_inits, n_restarts=0, warm_start=True),
    }
    arrays = {"h": h, "n_qubits": n, "depth": depth}
    for tag, out in runs.items():
        m = evaluate_sweep(ansatz, out["params"], out["energy"], ref)
        for key in ("energy", "n_iter", "n_fev", "total_fev", "n_candidates", "success",
                    "seconds", "params"):
            arrays[f"{tag}_{key}"] = out[key]
        for key in ("rel_error", "fidelity", "fidelity_gs"):
            arrays[f"{tag}_{key}"] = m[key]
    arrays["E0"] = reference_energy(ref)
    return arrays


def job_warmcold(ansatz_name: str, recompute: bool) -> str:
    n = CFG.vqe.warmcold_size
    name = f"vqe_warmcold_{ansatz_name}_N{n}_p{max(n // 2, 1)}"
    t0 = time.perf_counter()
    d = load_or_compute(name, lambda: compute_warmcold(ansatz_name),
                        params={"experiment": "warmcold", "ansatz": ansatz_name, "N": n},
                        subdir=SUBDIR, recompute=recompute)
    print(f"  {name}: max eps cold = {d['cold_rel_error'].max():.2e}, "
          f"warm = {d['warm_rel_error'].max():.2e}; mean iterations of the kept run: "
          f"cold {d['cold_n_iter'].mean():.0f}, warm {d['warm_n_iter'].mean():.0f} "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)
    return f"warm/cold {ansatz_name} done"


# ---------------------------------------------------------------------------
# HVA depth scan beyond p = N at the anchor fields
# ---------------------------------------------------------------------------
def compute_depthscan(n_sites: int) -> dict:
    c = CFG.vqe
    fields = np.asarray(c.anchor_fields, dtype=float)
    depths = np.arange(1, c.hva_depth_scan_factor * n_sites + 1)
    ref = ed_reference(n_sites, fields, verbose=False)
    e0 = reference_energy(ref)
    hams = tfim_hamiltonians(n_sites, fields)
    shape = (len(fields), len(depths))
    energy, fid, n_iter = np.zeros(shape), np.zeros(shape), np.zeros(shape, int)
    success = np.zeros(shape, bool)
    for j, H in enumerate(hams):
        prev = None
        for k, p in enumerate(depths):
            ansatz = HVA(n_sites, int(p))
            cands = [] if prev is None else [("depth", HVA.pad(prev, int(p)))]
            rng = point_rng(EXPERIMENT_KEYS["depthscan"], n_sites, int(p), j)
            cands += random_candidates(ansatz, c.n_random_inits, rng)
            best = optimize_point(ansatz, H, cands).best
            prev = best.x
            psi = ansatz.state(best.x)
            energy[j, k] = best.energy
            fid[j, k] = abs(np.vdot(ref["states"][j, ref["ground_sector"][j]], psi)) ** 2
            n_iter[j, k] = best.n_iter
            success[j, k] = best.success
    return {"fields": fields, "depths": depths, "energy": energy, "E0": e0,
            "rel_error": (energy - e0[:, None]) / np.abs(e0[:, None]),
            "fidelity": fid, "n_iter": n_iter, "success": success}


def job_depthscan(n_sites: int, recompute: bool) -> str:
    name = f"vqe_hva_depthscan_N{n_sites}"
    t0 = time.perf_counter()
    d = load_or_compute(name, lambda: compute_depthscan(n_sites),
                        params={"experiment": "depthscan", "N": n_sites},
                        subdir=SUBDIR, recompute=recompute)
    print(f"  {name}: eps at p = 2N: " + ", ".join(
        f"h={h:g}: {e:.1e}" for h, e in zip(d["fields"], d["rel_error"][:, -1]))
        + f" ({time.perf_counter() - t0:.0f} s)", flush=True)
    return f"depth scan N={n_sites} done"


# ---------------------------------------------------------------------------
# Periodic HVA at p = N/2 (exact for every h: Ho & Hsieh, SciPost Phys. 6, 029 (2019))
# ---------------------------------------------------------------------------
def compute_pbc() -> dict:
    c = CFG.vqe
    sizes = np.asarray(c.hva_depth_scan_sizes)
    fields = np.asarray(c.anchor_fields, dtype=float)
    rel = np.zeros((len(sizes), len(fields)))
    fid = np.zeros_like(rel)
    for a, n in enumerate(sizes):
        ref = ed_reference(int(n), fields, boundary="periodic", verbose=False)
        e0 = reference_energy(ref)
        hams = tfim_hamiltonians(int(n), fields, boundary="periodic")
        ansatz = HVA(int(n), int(n) // 2, boundary="periodic")
        for j, H in enumerate(hams):
            rng = point_rng(EXPERIMENT_KEYS["pbc"], int(n), j)
            best = optimize_point(ansatz, H, random_candidates(ansatz, c.n_random_inits, rng)).best
            rel[a, j] = (best.energy - e0[j]) / abs(e0[j])
            gs = ref["ground_sector"][j]
            fid[a, j] = abs(np.vdot(ref["states"][j, gs], ansatz.state(best.x))) ** 2
    return {"sizes": sizes, "fields": fields, "rel_error": rel, "fidelity": fid}


def job_pbc(recompute: bool) -> str:
    d = load_or_compute("vqe_hva_pbc_check", compute_pbc,
                        params={"experiment": "pbc", "depth": "N/2"},
                        subdir=SUBDIR, recompute=recompute)
    print(f"  vqe_hva_pbc_check: max eps = {d['rel_error'].max():.1e}, "
          f"min F = {d['fidelity'].min():.12f}", flush=True)
    return "periodic check done"


# ---------------------------------------------------------------------------
# Job scheduling
# ---------------------------------------------------------------------------
def build_jobs(experiments, sizes, ansatze, recompute) -> list:
    """List of (cost estimate, function, args). Largest jobs first for load balance."""
    jobs = []
    if "sweep" in experiments:
        for n in sizes:
            depths = CFG.system.depths(n)
            if "hva" in ansatze:
                jobs.append((2.0 ** n * sum(depths), job_hva_sweeps, (n, depths, recompute)))
            if "hea" in ansatze:
                for p in depths:
                    jobs.append((2.0 ** n * n * (p + 1) * 3, job_hea_sweep, (n, p, recompute)))
    if "warmcold" in experiments:
        for a in ansatze:
            jobs.append((2.0 ** CFG.vqe.warmcold_size * 50, job_warmcold, (a, recompute)))
    if "depthscan" in experiments and "hva" in ansatze:
        for n in CFG.vqe.hva_depth_scan_sizes:
            if n in sizes:
                jobs.append((2.0 ** n * n * n, job_depthscan, (n, recompute)))
    if "pbc" in experiments and "hva" in ansatze:
        jobs.append((1.0, job_pbc, (recompute,)))
    return sorted(jobs, key=lambda j: -j[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(CFG.system.sizes_noiseless))
    parser.add_argument("--ansatze", nargs="+", default=list(CFG.vqe.ansatze), choices=["hva", "hea"])
    parser.add_argument("--experiments", nargs="+", default=list(ALL_EXPERIMENTS),
                        choices=list(ALL_EXPERIMENTS))
    parser.add_argument("--workers", type=int, default=1, help="parallel processes")
    parser.add_argument("--quick", action="store_true", help="only N = 4, 6, 8")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    sizes = [n for n in args.sizes if n <= 8] if args.quick else args.sizes

    print(f"Phase 2 VQE: sizes {sizes}, ansatze {args.ansatze}, experiments {args.experiments}, "
          f"{args.workers} worker(s)")
    jobs = build_jobs(args.experiments, sizes, args.ansatze, args.recompute)
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
