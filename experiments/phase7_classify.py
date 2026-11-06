"""
Phase 7: machine-learning phase classification of the open TFIM (T7.1 to T7.6).

Run from the project root (after Phase 6):

    python experiments\\phase7_classify.py --workers 12
    python experiments\\phase7_classify.py --experiments clean ff --workers 12
    python experiments\\phase7_classify.py --experiments noise zne --sizes 8 --ansatze hva

Experiments (one job per output file; existing files are skipped unless --recompute):

    clean     T7.1, T7.2, T7.6  train and test within each clean dataset (ED N = 4..14,
                                noiseless VQE), exact and shadow features; VQE files also
                                hold "ed_*" arrays: trained on ED at the same N
    ff        T7.1 check        exact free-fermion features, N = 4..64 (no shadows)
    noise     T7.3, T7.4, T7.6  cross-noise experiments, one file per noisy dataset and
                                variant ("gate", "meas", "ro")
    zne       T7.3              unmitigated against ZNE-mitigated features in the lambda window
    shots     (Phase 6 note)    shadows rebuilt from 100 to 1000 snapshots, N = study_size
    learning  (Phase 6 note)    1 to 16 training shadows per training field, N = study_size

Cross-noise modes (axis "mode" of the noise files):

    clean_ed   train on ED shadows (same N), test on the noisy data at every strength
    clean_vqe  train on the noiseless circuit (strength 0, variant "gate"), test at every strength
    noisy      train and test at the same strength and variant
    mixed      train on all 11 strengths pooled, test at every strength
    loo        train on all strengths but the tested one (leave one strength out)

Output: results/phase7/{clean, ff, noise, zne, shots, learning}/...
Options: --experiments, --sizes, --ansatze, --channels, --variants, --workers, --recompute.
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG                                            # noqa: E402
from ml import classifiers as C                                   # noqa: E402
from ml import dataset as D                                       # noqa: E402
from ml import features as F                                      # noqa: E402
from utils.io import result_exists, save_results                  # noqa: E402

EXPERIMENTS = ("clean", "ff", "noise", "zne", "shots", "learning")
SUBDIR = "phase7"
KINDS = C.KINDS
ZNE_METHODS = ("ro", "zne_ro", "gate", "zne_gate")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def fill(store: dict, idx: tuple, res: dict, prefix: str = "", shape: tuple = ()) -> None:
    """
    store[prefix + key][idx] = res[key], allocating NaN arrays of shape + value shape.
    Trailing axes grow to the largest value seen (exact features have one test
    repetition, shadows several); unused entries stay NaN.
    """
    nlead = len(shape)
    for k, v in res.items():
        v = np.asarray(v, dtype=float)
        key = prefix + k
        if key not in store:
            store[key] = np.full(tuple(shape) + v.shape, np.nan)
        A = store[key]
        tail = A.shape[nlead:]
        if tail != v.shape:
            grown = tuple(max(a, b) for a, b in zip(tail, v.shape))
            if grown != tail:
                B = np.full(tuple(shape) + grown, np.nan)
                B[tuple(slice(0, n) for n in A.shape)] = A
                store[key] = A = B
        A[tuple(idx) + tuple(slice(0, n) for n in v.shape)] = v


def exact_m2(ds: dict, X=None) -> np.ndarray:
    N = int(ds["n_qubits"])
    X = np.asarray(ds["X"]) if X is None else np.asarray(X)
    return F.magnetization_m2(F.unpack_features(X, N)[0])


def meta_arrays(ds: dict) -> dict:
    return {"h": np.asarray(ds["h"]), "y": np.asarray(ds["y"]), "train_field": np.asarray(ds["train_field"]),
            "n_qubits": int(ds["n_qubits"]), "classifiers": np.array(C.CLASSIFIERS),
            "seeds": np.array(C.training_seeds()), "kinds": np.array(KINDS),
            "feature_sets": np.array(CFG.ml.feature_sets_phase7)}


def pca_fill(store, idx, shape, X_all, h, m2):
    fill(store, idx, C.pca_analysis(X_all, h, m2), prefix="pca_", shape=shape)


# ---------------------------------------------------------------------------
# clean: T7.1, T7.2 on ED and noiseless VQE
# ---------------------------------------------------------------------------
def run_clean(job: dict) -> dict:
    src, N = job["source"], job["N"]
    ds = D.load_dataset(src, N, job.get("ansatz"), job.get("depth"))
    h, y = np.asarray(ds["h"]), np.asarray(ds["y"])
    y_tr = y[np.asarray(ds["train_field"], dtype=bool)]
    ed = None
    if src == "vqe":
        try:
            ed = D.load_dataset("ed", N)
        except FileNotFoundError:
            ed = None
    fsets = CFG.ml.feature_sets_phase7
    shape = (len(KINDS), len(fsets))
    out = meta_arrays(ds)
    for ik, kind in enumerate(KINDS):
        for jf, fs in enumerate(fsets):
            tr, te = C.split_array(ds, C.dataset_array(ds, kind), kind, fs)
            fill(out, (ik, jf), C.evaluate([tr], y_tr, te, h, y), shape=shape)
            if ed is not None:
                tr_ed, _ = C.split_array(ed, C.dataset_array(ed, kind), kind, fs)
                fill(out, (ik, jf), C.evaluate([tr_ed], y_tr, te, h, y), prefix="ed_", shape=shape)
        pca_fill(out, (ik,), (len(KINDS),), C.dataset_array(ds, kind), h, exact_m2(ds))
    out["m2_exact"] = exact_m2(ds)
    out["m2_shadow"] = F.magnetization_m2(F.unpack_features(np.asarray(ds["Xs"]), N)[0])
    out["has_ed"] = ed is not None
    out.update(source=src, ansatz=job.get("ansatz", "ed"), depth=job.get("depth", 0))
    if src == "vqe":
        out["fidelity"] = np.asarray(ds["fidelity"])
    return out


# ---------------------------------------------------------------------------
# ff: exact free-fermion features up to N = 64
# ---------------------------------------------------------------------------
def ff_features(N: int, h) -> np.ndarray:
    from analysis.ff_correlators import ff_observables
    rows = []
    for hv in h:
        o = ff_observables(N, CFG.tfim.J, float(hv))
        rows.append(F.corr_features(o["zz"], o["x"]))
    return np.array(rows)


def run_ff(job: dict) -> dict:
    h = D.field_grid()
    y = D.labels(h)
    tf = D.train_field_mask(h)
    sizes = tuple(CFG.ml.ff_sizes)
    fsets = CFG.ml.feature_sets_phase7
    out = {"h": h, "y": y, "sizes": np.array(sizes), "classifiers": np.array(C.CLASSIFIERS),
           "feature_sets": np.array(fsets), "seeds": np.array(C.training_seeds())}
    shape = (len(sizes), len(fsets))
    for i, N in enumerate(sizes):
        X = ff_features(N, h)
        for jf, fs in enumerate(fsets):
            Xf = F.select_features(X, N, fs)
            fill(out, (i, jf), C.evaluate([Xf[tf]], y[tf], Xf, h, y), shape=shape)
        m2 = F.magnetization_m2(F.unpack_features(X, N)[0])
        pca_fill(out, (i,), (len(sizes),), X, h, m2)
        fill(out, (i,), {"m2": m2}, shape=(len(sizes),))
    return out


# ---------------------------------------------------------------------------
# noise: cross-noise experiments (T7.3, T7.4)
# ---------------------------------------------------------------------------
def run_noise(job: dict) -> dict:
    a, N, p, ch, v = job["ansatz"], job["N"], job["depth"], job["channel"], job["variant"]
    ds = D.load_dataset("noisy", N, a, p, ch)
    ed = D.load_dataset("ed", N)
    h, y = np.asarray(ds["h"]), np.asarray(ds["y"])
    y_tr = y[np.asarray(ds["train_field"], dtype=bool)]
    strengths = np.asarray(ds["strengths"])
    ns = strengths.size
    modes, fsets = CFG.ml.noise_modes, CFG.ml.feature_sets_phase7
    shape = (len(modes), ns, len(KINDS), len(fsets))
    out = meta_arrays(ds)
    out.update(modes=np.array(modes), variant=v, strengths=strengths, lam=np.asarray(ds["lam"]),
               ansatz=a, depth=p, channel=ch)
    for ik, kind in enumerate(KINDS):
        for jf, fs in enumerate(fsets):
            tr_ed, _ = C.split_array(ed, C.dataset_array(ed, kind), kind, fs)
            tr_vqe, _ = C.split_array(ds, C.dataset_array(ds, kind, "gate", 0), kind, fs)
            blocks = [C.split_array(ds, C.dataset_array(ds, kind, v, js), kind, fs) for js in range(ns)]
            trains = [b[0] for b in blocks]
            tests = [b[1] for b in blocks]
            for im, mode in enumerate(modes):
                if mode == "clean_ed":
                    res = C.evaluate([tr_ed], y_tr, tests, h, y)
                elif mode == "clean_vqe":
                    res = C.evaluate([tr_vqe], y_tr, tests, h, y)
                elif mode == "mixed":
                    res = C.evaluate(trains, y_tr, tests, h, y)
                elif mode == "noisy":
                    res = [C.evaluate([trains[js]], y_tr, tests[js], h, y) for js in range(ns)]
                elif mode == "loo":
                    res = [C.evaluate([trains[k] for k in range(ns) if k != js], y_tr, tests[js], h, y)
                           for js in range(ns)]
                else:
                    raise ValueError(f"unknown mode {mode!r}")
                for js in range(ns):
                    fill(out, (im, js, ik, jf), res[js], shape=shape)
    # PCA (T7.2) per strength and kind; the exact M^2 of the same data orients PC1
    for js in range(ns):
        m2 = exact_m2(ds, C.dataset_array(ds, "exact", v, js))
        for ik, kind in enumerate(KINDS):
            pca_fill(out, (js, ik), (ns, len(KINDS)), C.dataset_array(ds, kind, v, js), h, m2)
        fill(out, (js,), {"m2_exact": m2}, shape=(ns,))
    out["m2_ed"] = exact_m2(ed)
    out["m2_clean"] = exact_m2(ds, C.dataset_array(ds, "exact", "gate", 0))
    return out


# ---------------------------------------------------------------------------
# zne: does mitigation restore h_c^ML? (T7.3)
# ---------------------------------------------------------------------------
def run_zne(job: dict) -> dict:
    a, N, p, ch = job["ansatz"], job["N"], job["depth"], job["channel"]
    z = D.load_dataset("zne", N, a, p, ch)
    ds = D.load_dataset("noisy", N, a, p, ch)
    ed = D.load_dataset("ed", N)
    h, y = np.asarray(z["h"]), np.asarray(z["y"])
    y_tr = y[np.asarray(z["train_field"], dtype=bool)]
    win = np.asarray(z["strength_index"]).astype(int)
    modes = ("clean_ed", "clean_vqe", "noisy")
    fsets = CFG.ml.feature_sets_phase7
    shape = (len(modes), len(ZNE_METHODS), win.size, len(KINDS), len(fsets))
    out = meta_arrays(z)
    out.update(modes=np.array(modes), methods=np.array(ZNE_METHODS), strengths=np.asarray(z["strengths"]),
               strength_index=win, lam=np.asarray(z["lam"]), ansatz=a, depth=p, channel=ch)

    def array_for(method, kind, w):
        js = int(win[w])
        if method in ("ro", "gate"):
            return C.dataset_array(ds, kind, method, js)
        if kind == "shadow" and method == "zne_gate":
            return None                              # no gate-only ZNE shadows were generated
        return C.dataset_array(z, kind, method, w)

    for ik, kind in enumerate(KINDS):
        for jf, fs in enumerate(fsets):
            tr_ed, _ = C.split_array(ed, C.dataset_array(ed, kind), kind, fs)
            tr_vqe, _ = C.split_array(ds, C.dataset_array(ds, kind, "gate", 0), kind, fs)
            for iu, method in enumerate(ZNE_METHODS):
                for w in range(win.size):
                    A = array_for(method, kind, w)
                    if A is None:
                        continue
                    tr, te = C.split_array(z, A, kind, fs)
                    for im, mode in enumerate(modes):
                        block = {"clean_ed": tr_ed, "clean_vqe": tr_vqe, "noisy": tr}[mode]
                        fill(out, (im, iu, w, ik, jf), C.evaluate([block], y_tr, te, h, y), shape=shape)
    # exact feature distance to the noiseless circuit (how much ZNE removes)
    clean = np.asarray(ds["X_gate"])[0]
    dist = np.full((len(ZNE_METHODS), win.size), np.nan)
    for iu, method in enumerate(ZNE_METHODS):
        for w in range(win.size):
            dist[iu, w] = np.sqrt(np.mean((array_for(method, "exact", w) - clean) ** 2))
    out["feature_rms_to_clean"] = dist
    return out


# ---------------------------------------------------------------------------
# shots: rebuild the shadows from fewer snapshots
# ---------------------------------------------------------------------------
def run_shots(job: dict) -> dict:
    src, N = job["source"], job["N"]
    Ms = tuple(int(m) for m in CFG.ml.shots_list)
    ed = D.load_dataset("ed", N)
    if src == "ed":
        ds, strengths, variant = ed, [None], None
    else:
        ds = D.load_dataset("noisy", N, job["ansatz"], job["depth"], job["channel"])
        strengths, variant = list(CFG.ml.shots_strength_indices), CFG.ml.default_variant
    h, y = np.asarray(ds["h"]), np.asarray(ds["y"])
    y_tr = y[np.asarray(ds["train_field"], dtype=bool)]
    modes = ("clean_ed", "noisy")
    shape = (len(modes), len(strengths), len(Ms))
    out = meta_arrays(ds)
    out.update(modes=np.array(modes), shots=np.array(Ms), source=src,
               strength_index=np.array([-1 if s is None else s for s in strengths]))
    if src != "ed":
        out["strengths"] = np.asarray(ds["strengths"])[strengths]
        out["lam"] = np.asarray(ds["lam"])[strengths]
    rebuild_err = np.full(len(strengths), np.nan)
    for iM, M in enumerate(Ms):
        Xe = C.rebuild_shadows(ed, M)
        tr_ed, _ = C.split_array(ed, Xe, "shadow")
        for i, js in enumerate(strengths):
            Xs = C.rebuild_shadows(ds, M, js, variant)
            if M == int(ds["snapshots"]):
                ref = np.asarray(ds["Xs"]) if js is None else np.asarray(ds[f"Xs_{variant}"])[js]
                rebuild_err[i] = float(np.max(np.abs(Xs - ref)))
            tr, te = C.split_array(ds, Xs, "shadow")
            for im, mode in enumerate(modes):
                block = tr_ed if mode == "clean_ed" else tr
                fill(out, (im, i, iM), C.evaluate([block], y_tr, te, h, y), shape=shape)
    out["rebuild_err"] = rebuild_err
    return out


# ---------------------------------------------------------------------------
# learning: fewer training shadows per field
# ---------------------------------------------------------------------------
def run_learning(job: dict) -> dict:
    src, N = job["source"], job["N"]
    if src == "ed":
        ds, strengths, variant = D.load_dataset("ed", N), [None], None
    else:
        ds = D.load_dataset("noisy", N, job["ansatz"], job["depth"], job["channel"])
        strengths, variant = list(CFG.ml.learning_strength_indices), CFG.ml.default_variant
    h, y = np.asarray(ds["h"]), np.asarray(ds["y"])
    y_tr = y[np.asarray(ds["train_field"], dtype=bool)]
    n_train_reps = int(np.sum(np.asarray(ds["rep_train"], dtype=bool)))
    ks = tuple(k for k in CFG.ml.learning_reps if k <= n_train_reps)
    shape = (len(strengths), len(ks))
    out = meta_arrays(ds)
    out.update(reps_per_field=np.array(ks), n_train_rows=np.array(ks) * int(y_tr.size), source=src,
               strength_index=np.array([-1 if s is None else s for s in strengths]))
    if src != "ed":
        out["strengths"] = np.asarray(ds["strengths"])[strengths]
    for i, js in enumerate(strengths):
        tr, te = C.split_array(ds, C.dataset_array(ds, "shadow", variant, js), "shadow")
        for ik, k in enumerate(ks):
            fill(out, (i, ik), C.evaluate([tr], y_tr, te, h, y, rep_subset=k), shape=shape)
    return out


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
RUNNERS = {"clean": run_clean, "ff": run_ff, "noise": run_noise, "zne": run_zne,
           "shots": run_shots, "learning": run_learning}


def job_name(job: dict) -> tuple:
    e = job["experiment"]
    if e == "ff":
        return "ff_exact", f"{SUBDIR}/ff"
    if e == "clean":
        if job["source"] == "ed":
            return f"clean_ed_N{job['N']}", f"{SUBDIR}/clean"
        return f"clean_vqe_{job['ansatz']}_N{job['N']}_p{job['depth']}", f"{SUBDIR}/clean"
    tag = "ed" if job.get("source") == "ed" else f"{job['ansatz']}_N{job['N']}_p{job['depth']}_{job['channel']}"
    if e == "noise":
        tag += f"_{job['variant']}"
    if e in ("shots", "learning") and job.get("source") == "ed":
        tag = f"ed_N{job['N']}"
    return f"{e}_{tag}", f"{SUBDIR}/{e}"


def build_jobs(args) -> list:
    jobs = []
    keep = (lambda n: not args.sizes or n in args.sizes)
    noisy_p = (lambda N: D.resolve_depth(CFG.ml.noisy_depth_rule, N))
    channels = args.channels or list(CFG.ml.noisy_channels)
    if "clean" in args.experiments:
        jobs += [dict(experiment="clean", source="ed", N=N) for N in CFG.ml.ed_sizes if keep(N)]
        for a in args.ansatze:
            for N in CFG.ml.vqe_sizes:
                if keep(N):
                    for p in sorted({D.resolve_depth(r, N) for r in CFG.ml.vqe_depth_rules}):
                        jobs.append(dict(experiment="clean", source="vqe", N=N, ansatz=a, depth=p))
    if "ff" in args.experiments:
        jobs.append(dict(experiment="ff"))
    for a in args.ansatze:
        for N in CFG.ml.noisy_sizes:
            if not keep(N):
                continue
            base = dict(source="noisy", N=N, ansatz=a, depth=noisy_p(N))
            if "noise" in args.experiments:
                for c in channels:
                    for v in args.variants:
                        jobs.append(dict(base, experiment="noise", channel=c, variant=v))
            if "zne" in args.experiments:
                for c in channels:
                    if c in CFG.ml.zne_channels:
                        jobs.append(dict(base, experiment="zne", channel=c))
    for e in ("shots", "learning"):
        if e in args.experiments:
            N = CFG.ml.study_size
            jobs.append(dict(experiment=e, source="ed", N=N))
            for a in args.ansatze:
                jobs.append(dict(experiment=e, source="noisy", N=N, ansatz=a, depth=noisy_p(N),
                                 channel=CFG.ml.noisy_channels[0]))
    cost = {"noise": 50.0, "zne": 8.0, "shots": 6.0, "learning": 4.0, "ff": 30.0, "clean": 1.0}
    jobs.sort(key=lambda j: -cost[j["experiment"]] * 2.0 ** j.get("N", 8))
    return jobs


def run_job(job: dict) -> tuple:
    t0 = time.perf_counter()
    arrays = RUNNERS[job["experiment"]](job)
    seconds = time.perf_counter() - t0
    arrays["seconds"] = seconds
    name, sub = job_name(job)
    save_results(name, arrays, params=dict(job), subdir=sub)
    return name, seconds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=EXPERIMENTS)
    ap.add_argument("--sizes", nargs="+", type=int, default=None)
    ap.add_argument("--ansatze", nargs="+", default=list(CFG.vqe.ansatze), choices=("hva", "hea"))
    ap.add_argument("--channels", nargs="+", default=None,
                    choices=("realistic", "depolarizing", "amplitude_damping", "dephasing"))
    ap.add_argument("--variants", nargs="+", default=list(D.NOISY_VARIANTS), choices=D.NOISY_VARIANTS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs(args)
    todo = [j for j in jobs if args.recompute or not result_exists(*job_name(j))]
    print(f"[phase7] {len(jobs)} jobs, {len(jobs) - len(todo)} cached, {len(todo)} to run "
          f"({len(C.CLASSIFIERS)} classifiers x {CFG.ml.n_training_seeds} seeds per configuration)")
    t0 = time.perf_counter()
    failed = []
    if args.workers <= 1:
        for j in todo:
            try:
                name, sec = run_job(j)
                print(f"[done] {name} ({sec:.1f} s)", flush=True)
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
                    print(f"[done {k}/{len(todo)}] {name} ({sec:.1f} s, wall {time.perf_counter() - t0:.0f} s)",
                          flush=True)
                except Exception:
                    failed.append(job_name(j)[0])
                    print(f"[failed] {job_name(j)[0]}")
                    traceback.print_exc()
    print(f"[phase7] wall time {time.perf_counter() - t0:.0f} s")
    if failed:
        print(f"[phase7] {len(failed)} job(s) failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
