"""
Phase 6: generate the labeled ML datasets (T6.1 to T6.4).

Run from the project root (after Phases 1, 2 and 5):

    python experiments\\phase6_datasets.py --workers 12
    python experiments\\phase6_datasets.py --experiments ed vqe --workers 12
    python experiments\\phase6_datasets.py --experiments noisy zne --sizes 8 --ansatze hva

One job per output file; files that exist are skipped unless --recompute.
Output (organized by source; the noise level is the leading array axis):

    results/phase6/ed/ds_ed_N{N}
    results/phase6/vqe/ds_vqe_{hva|hea}_N{N}_p{p}
    results/phase6/noisy/ds_noisy_{hva|hea}_N{N}_p{p}_{channel}
    results/phase6/zne/ds_zne_{hva|hea}_N{N}_p{p}_{channel}
    results/phase6/dataset_index.json
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")            # one BLAS thread per worker process

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG                                            # noqa: E402
from ml import dataset as D                                       # noqa: E402
from utils.io import result_exists, save_results                  # noqa: E402

EXPERIMENTS = ("ed", "vqe", "noisy", "zne")


def build_jobs(args) -> list[dict]:
    jobs = []
    sizes = set(args.sizes) if args.sizes else None
    keep = (lambda n: sizes is None or n in sizes)
    if "ed" in args.experiments:
        jobs += [dict(source="ed", N=N) for N in CFG.ml.ed_sizes if keep(N)]
    if "vqe" in args.experiments:
        for a in args.ansatze:
            for N in CFG.ml.vqe_sizes:
                if not keep(N):
                    continue
                for p in sorted({D.resolve_depth(r, N) for r in CFG.ml.vqe_depth_rules}):
                    jobs.append(dict(source="vqe", N=N, ansatz=a, depth=p))
    for src, channels in (("noisy", args.channels or CFG.ml.noisy_channels),
                          ("zne", args.channels or CFG.ml.zne_channels)):
        if src not in args.experiments:
            continue
        for a in args.ansatze:
            for N in CFG.ml.noisy_sizes:
                if keep(N):
                    for c in channels:
                        jobs.append(dict(source=src, N=N, ansatz=a,
                                         depth=D.resolve_depth(CFG.ml.noisy_depth_rule, N), channel=c))
    # largest jobs first: they set the wall time
    cost = {"ed": 1.0, "vqe": 0.5, "noisy": 40.0, "zne": 60.0}
    jobs.sort(key=lambda j: -cost[j["source"]] * 2.0 ** j["N"])
    return jobs


def job_name(job: dict) -> tuple:
    return D.dataset_name(job["source"], job["N"], job.get("ansatz"), job.get("depth"), job.get("channel"))


def run_job(job: dict) -> tuple:
    t0 = time.perf_counter()
    src, N = job["source"], job["N"]
    if src == "ed":
        arrays = D.ed_dataset(N)
    elif src == "vqe":
        arrays = D.vqe_dataset(job["ansatz"], N, job["depth"])
    elif src == "noisy":
        arrays = D.noisy_dataset(job["ansatz"], N, job["depth"], job["channel"])
    else:
        arrays = D.zne_dataset(job["ansatz"], N, job["depth"], job["channel"])
    seconds = time.perf_counter() - t0
    arrays["seconds"] = seconds
    name, sub = job_name(job)
    params = dict(job, snapshots=CFG.ml.n_shadow_snapshots, repetitions=CFG.ml.shadow_repetitions)
    save_results(name, arrays, params=params, subdir=sub)
    return name, seconds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=EXPERIMENTS)
    ap.add_argument("--sizes", nargs="+", type=int, default=None)
    ap.add_argument("--ansatze", nargs="+", default=list(CFG.vqe.ansatze), choices=("hva", "hea"))
    ap.add_argument("--channels", nargs="+", default=None,
                    choices=("realistic", "depolarizing", "amplitude_damping", "dephasing"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs(args)
    todo = [j for j in jobs if args.recompute or not result_exists(*job_name(j))]
    print(f"[phase6] {len(jobs)} jobs, {len(jobs) - len(todo)} cached, {len(todo)} to run "
          f"(R = {CFG.ml.shadow_repetitions} shadows of M = {CFG.ml.n_shadow_snapshots} snapshots per field)")
    t0 = time.perf_counter()
    failed = []
    if args.workers <= 1:
        for j in todo:
            try:
                name, sec = run_job(j)
                print(f"[done] {name} ({sec:.1f} s)")
            except Exception:
                failed.append(job_name(j)[0])
                traceback.print_exc()
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_job, j): j for j in todo}
            for k, fut in enumerate(as_completed(futs), 1):
                j = futs[fut]
                try:
                    name, sec = fut.result()
                    print(f"[done {k}/{len(todo)}] {name} ({sec:.1f} s, wall {time.perf_counter() - t0:.0f} s)")
                except Exception:
                    failed.append(job_name(j)[0])
                    print(f"[failed] {job_name(j)[0]}")
                    traceback.print_exc()
    path = D.write_index()
    print(f"[phase6] wall time {time.perf_counter() - t0:.0f} s; index written to {path}")
    if failed:
        print(f"[phase6] {len(failed)} job(s) failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
