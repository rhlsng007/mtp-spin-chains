"""
Phase 6: labeled ML datasets of the open TFIM (T6.1 to T6.4).

Sources (T6.1), all on the fine field grid 0.1 to 2.0, step 0.05 (39 fields):

* "ed"    : exact even-parity ground state, N = 4..14.
* "vqe"   : noiseless Phase 2 optima (HVA, HEA), p = N/2 and p = N, N = 4..12.
* "noisy" : the Phase 2 optimum at p = N/2 evaluated under the Phase 5
            hardware-like noise (default "realistic": depolarizing p and 10p,
            amplitude damping and dephasing p, virtual RZ) at the 11 strengths
            of CFG.mitigation.strengths(), with readout error, N = 4, 6, 8.
* "zne"   : the same circuits folded globally at scales 1, 3, 5 and
            extrapolated (Richardson), only at strengths whose error budget
            lambda lies in CFG.ml.zne_lambda_window (Phase 5 decision).

The noisy step-0.1 grid of Phases 4 and 5 was set by the cost of training
under noise; here the parameters are fixed, so the fine grid is affordable
and every source shares the same fields (CFG.ml.noisy_fine_grid).

Features (T6.2): exact expectation values (infinite shots) and classical
shadows (M snapshots per repetition, R independent repetitions per field),
both reduced to the "corr" feature vector of ml.features. Noisy sources come
in three variants: "gate" (gate noise, ideal readout), "meas" (with readout
error, uncorrected) and "ro" (readout-corrected with a finite-shot tensored
calibration whose X gates carry gate noise, one calibration per repetition,
as in Phase 5).

Labels (T6.3): y = 0 for h < h_c (ordered), 1 for h > h_c (disordered),
-1 at h = h_c (excluded from accuracies).

Split (T6.4): training fields h < CFG.ml.train_h_ordered_max or
h > CFG.ml.train_h_disordered_min; test fields: all. Shadow repetitions are
split as well: the last round(test_fraction R) repetitions are test-only, so
no snapshot used for training is ever tested on.

Random numbers (common random numbers): snapshot bases, sampling uniforms
and readout uniforms come from SeedSequence([2026, 15, source, N, p, field,
repetition, scale]) with source 0 = ED, 1 = HVA, 2 = HEA (p = 0 for ED).
They do not depend on channel or strength, and the noiseless VQE dataset
uses the same stream as the noisy one at the same (ansatz, N, p), so noisy
strength 0 reproduces the noiseless snapshots exactly. Calibration shots:
SeedSequence([2026, 16, N]), redrawn per (channel, strength).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import CFG, RESULTS_DIR
from ml import features as F
from ml import shadows as S

EXP_SHADOW = 15
EXP_CALIBRATION = 16
SOURCE_KEYS = {"ed": 0, "hva": 1, "hea": 2}
SOURCES = ("ed", "vqe", "noisy", "zne")
NOISY_VARIANTS = ("gate", "meas", "ro")
SUBDIR = "phase6"


# ---------------------------------------------------------------------------
# Grid, labels and splits (T6.3, T6.4)
# ---------------------------------------------------------------------------
def field_grid(noisy: bool = False) -> np.ndarray:
    if noisy and not CFG.ml.noisy_fine_grid:
        return CFG.tfim.h_grid(noisy=True)
    return CFG.tfim.h_grid(noisy=False)


def labels(h) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    hc = CFG.tfim.h_critical
    y = np.where(h > hc, CFG.ml.label_disordered, CFG.ml.label_ordered).astype(np.int8)
    y[np.isclose(h, hc, atol=1e-9)] = CFG.ml.label_critical
    return y


def train_field_mask(h) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    return (h < CFG.ml.train_h_ordered_max - 1e-9) | (h > CFG.ml.train_h_disordered_min + 1e-9)


def repetition_split(n_reps: int) -> tuple:
    n_test = int(round(CFG.ml.test_fraction * n_reps))
    n_test = min(max(n_test, 1), n_reps - 1) if n_reps > 1 else 0
    test = np.zeros(n_reps, dtype=bool)
    test[n_reps - n_test:] = True
    return ~test, test


def resolve_depth(rule, n_qubits: int) -> int:
    if rule == "half":
        return n_qubits // 2
    if rule == "full":
        return n_qubits
    return int(rule)


def _sizes(R, M):
    return (CFG.ml.shadow_repetitions if R is None else int(R),
            CFG.ml.n_shadow_snapshots if M is None else int(M))


def common_arrays(h, n_qubits: int, n_reps: int, n_snap: int) -> dict:
    rep_train, rep_test = repetition_split(n_reps)
    train = train_field_mask(h)
    return {"h": np.asarray(h, dtype=float), "y": labels(h), "train_field": train,
            "test_field": np.ones_like(train), "rep_train": rep_train, "rep_test": rep_test,
            "feature_names": F.feature_names(n_qubits), "n_qubits": n_qubits,
            "snapshots": n_snap, "repetitions": n_reps, "h_critical": CFG.tfim.h_critical}


# ---------------------------------------------------------------------------
# Random streams
# ---------------------------------------------------------------------------
def snapshot_rng(source_key: int, n_qubits: int, depth: int, field_index: int,
                 rep: int, scale: int = 1) -> np.random.Generator:
    ss = np.random.SeedSequence([CFG.seeds.master_seed, EXP_SHADOW, int(source_key), int(n_qubits),
                                 int(depth), int(field_index), int(rep), int(scale)])
    return np.random.default_rng(ss)


def draw_snapshots(source_key: int, n_qubits: int, depth: int, field_index: int,
                   n_reps: int, n_snap: int, scale: int = 1) -> tuple:
    """bases (R, M, N) uint8, sampling uniforms u (R, M, N), readout uniforms r (R, M, N)."""
    out = [S.draw_stream(snapshot_rng(source_key, n_qubits, depth, field_index, k, scale), n_snap, n_qubits)
           for k in range(n_reps)]
    return tuple(np.stack([o[i] for o in out]) for i in range(3))


def calibration(n_qubits: int, noise_model, n_reps: int) -> tuple:
    """Exact per-qubit calibration matrices (N, 2, 2) and R finite-shot estimates (R, N, 2, 2)."""
    from noise.mitigation import ReadoutModel, calibration_matrices, sample_tensored_calibration
    readout = ReadoutModel.from_config(n_qubits)
    nm = noise_model if CFG.mitigation.calibration_gate_noise else None
    C = calibration_matrices(readout, nm)
    rng = np.random.default_rng(np.random.SeedSequence([CFG.seeds.master_seed, EXP_CALIBRATION, n_qubits]))
    return C, sample_tensored_calibration(C, CFG.mitigation.calibration_shots, rng, n_reps)


# ---------------------------------------------------------------------------
# Pure-state sources: ED and noiseless VQE
# ---------------------------------------------------------------------------
def pure_dataset(states, h, n_qubits: int, source_key: int, depth: int, R=None, M=None) -> dict:
    """Exact and shadow features of pure states (n_h, 2^N)."""
    R, M = _sizes(R, M)
    N = n_qubits
    h = np.asarray(h, dtype=float)
    nh, nf = h.size, F.n_features(N)
    out = common_arrays(h, N, R, M)
    X, Xs = np.empty((nh, nf)), np.empty((nh, R, nf))
    z, zs = np.empty((nh, N)), np.empty((nh, R, N))
    bases = np.empty((nh, R, M, N), dtype=np.uint8)
    bits = np.empty((nh, R, M, (N + 7) // 8), dtype=np.uint8)
    for ih in range(nh):
        psi = np.asarray(states[ih])
        obs = F.pure_observables(psi, N)
        X[ih], z[ih] = F.corr_features(obs["zz"], obs["x"]), obs["z"]
        B, u, _ = draw_snapshots(source_key, N, depth, ih, R, M)
        b = S.sample_pure(psi, B.reshape(-1, N), u.reshape(-1, N), CFG.ml.sampler_chunk_elements).reshape(R, M, N)
        sh = S.shadow_observables(B, b)
        Xs[ih], zs[ih] = F.corr_features(sh["zz"], sh["x"]), sh["z"]
        bases[ih], bits[ih] = B, S.pack_bits(b)
    out.update(X=X, Xs=Xs, z=z, zs=zs, bases=bases, bits=bits,
               energy=F.energy_from_features(X, N, h, CFG.tfim.J))
    return out


def ed_dataset(n_qubits: int, R=None, M=None) -> dict:
    from models.exact import solve_model
    h = field_grid()
    states, E0 = [], []
    for hv in h:
        res = solve_model("tfim", n_qubits, J=CFG.tfim.J, h=float(hv), boundary=CFG.tfim.boundary)
        states.append(np.asarray(res.ground_state, dtype=float))
        E0.append(res.ground_energy)
    out = pure_dataset(states, h, n_qubits, SOURCE_KEYS["ed"], 0, R, M)
    out.update(E0=np.array(E0), depth=0, source="ed", ansatz="ed")
    return out


def vqe_dataset(ansatz: str, n_qubits: int, depth: int, R=None, M=None) -> dict:
    from noise.data import load_phase2
    h = field_grid()
    d = load_phase2(ansatz, n_qubits, depth, h_sel=h)
    out = pure_dataset(d["states"], h, n_qubits, SOURCE_KEYS[ansatz], depth, R, M)
    out.update(params=d["params"], phase2_energy=d["energy"], E0=d["E0"], fidelity=d["fidelity"],
               fidelity_gs=d["fidelity_gs"], depth=depth, source="vqe", ansatz=ansatz)
    return out


# ---------------------------------------------------------------------------
# Noisy sources
# ---------------------------------------------------------------------------
def _noisy_point(rho, n_qubits, B, u, r, readout, Rm, C_exact, C_samp) -> dict:
    """Exact and shadow features of one density matrix in the three variants."""
    N = n_qubits
    obs = F.mixed_observables(rho, N)
    exact = {"gate": obs, "meas": F.readout_biased(obs, Rm), "ro": F.readout_corrected(obs, Rm, C_exact)}
    R, M = B.shape[:2]
    tabs = S.marginal_tables(S.outcome_table(rho, N), N)
    b_gate = S.sample_mixed(tabs, B.reshape(-1, N), u.reshape(-1, N)).reshape(R, M, N)
    b_meas = S.apply_readout(b_gate, r, np.asarray(readout.p01), np.asarray(readout.p10))
    aC, bC = F.readout_affine(C_samp)                                  # (R, N)
    shadow = {"gate": S.shadow_observables(B, b_gate), "meas": S.shadow_observables(B, b_meas),
              "ro": S.shadow_observables(B, b_meas, aC, bC)}
    out = {"b_gate": b_gate, "b_meas": b_meas, "obs": obs}
    for v in NOISY_VARIANTS:
        out[f"X_{v}"] = F.corr_features(exact[v]["zz"], exact[v]["x"])
        out[f"z_{v}"] = exact[v]["z"]
        out[f"Xs_{v}"] = F.corr_features(shadow[v]["zz"], shadow[v]["x"])
        out[f"zs_{v}"] = shadow[v]["z"]
    return out


def error_budget_for(ansatz: str, n_qubits: int, depth: int, channel: str, strengths) -> tuple:
    from noise.density import gate_sequence
    from noise.mitigation import error_budget, make_mitigation_noise_model, noisy_gate_counts
    n_par = 2 * depth if ansatz == "hva" else 2 * n_qubits * (depth + 1)
    gates = gate_sequence(ansatz, n_qubits, depth, np.zeros(n_par))
    cnt = noisy_gate_counts(gates, make_mitigation_noise_model(channel, 1e-3))
    lam = error_budget(channel, np.asarray(strengths, dtype=float), cnt["n_1q"], cnt["n_2q"])
    return np.asarray(lam, dtype=float), cnt["n_1q"], cnt["n_2q"]


def noisy_dataset(ansatz: str, n_qubits: int, depth: int, channel: str, thetas=None, h=None,
                  strengths=None, R=None, M=None) -> dict:
    """Phase 2 optima at the given depth under gate noise and readout error (fixed parameters)."""
    from noise.density import gate_sequence, simulate_statevector
    from noise.mitigation import ReadoutModel, make_mitigation_noise_model, simulate_gates
    R, M = _sizes(R, M)
    N, p = n_qubits, depth
    extra = {}
    if thetas is None:
        from noise.data import load_phase2
        h = field_grid(noisy=True)
        d = load_phase2(ansatz, N, p, h_sel=h)
        thetas = d["params"]
        extra = {"phase2_energy": d["energy"], "E0": d["E0"]}
    h = np.asarray(h, dtype=float)
    strengths = CFG.mitigation.strengths() if strengths is None else np.asarray(strengths, dtype=float)
    ns, nh, nf = strengths.size, h.size, F.n_features(N)
    readout = ReadoutModel.from_config(N)
    Rm = readout.matrices()
    models = [make_mitigation_noise_model(channel, s) for s in strengths]
    cals = [calibration(N, nm, R) for nm in models]
    lam, n1, n2 = error_budget_for(ansatz, N, p, channel, strengths)

    out = common_arrays(h, N, R, M)
    arr = {}
    for v in NOISY_VARIANTS:
        arr[f"X_{v}"] = np.empty((ns, nh, nf))
        arr[f"Xs_{v}"] = np.empty((ns, nh, R, nf))
        arr[f"z_{v}"] = np.empty((ns, nh, N))
        arr[f"zs_{v}"] = np.empty((ns, nh, R, N))
    nb = (N + 7) // 8
    bits_gate = np.empty((ns, nh, R, M, nb), dtype=np.uint8)
    bits_meas = np.empty((ns, nh, R, M, nb), dtype=np.uint8)
    bases = np.empty((nh, R, M, N), dtype=np.uint8)
    purity, fid = np.empty((ns, nh)), np.empty((ns, nh))
    for ih in range(nh):
        theta = np.asarray(thetas[ih], dtype=float)
        gates = gate_sequence(ansatz, N, p, theta)
        psi = simulate_statevector(ansatz, N, p, theta)
        B, u, r = draw_snapshots(SOURCE_KEYS[ansatz], N, p, ih, R, M)
        bases[ih] = B
        for js, nm in enumerate(models):
            rho = simulate_gates(gates, N, nm)
            pt = _noisy_point(rho, N, B, u, r, readout, Rm, cals[js][0], cals[js][1])
            for key in arr:
                arr[key][js, ih] = pt[key]
            bits_gate[js, ih], bits_meas[js, ih] = S.pack_bits(pt["b_gate"]), S.pack_bits(pt["b_meas"])
            purity[js, ih] = float(np.real(np.vdot(rho, rho)))
            fid[js, ih] = float(np.real(np.vdot(psi, rho @ psi)))
    out.update(arr)
    out.update(extra)
    out.update(bases=bases, bits_gate=bits_gate, bits_meas=bits_meas, purity=purity, fidelity_ideal=fid,
               energy=F.energy_from_features(arr["X_gate"], N, h, CFG.tfim.J),
               strengths=strengths, lam=lam, n_1q=n1, n_2q=n2, params=np.asarray(thetas),
               readout_matrices=Rm, cal_exact=np.stack([c[0] for c in cals]),
               cal_samples=np.stack([c[1] for c in cals]), depth=p, source="noisy",
               ansatz=ansatz, channel=channel)
    return out


def zne_window(lam, strengths) -> np.ndarray:
    lo, hi = CFG.ml.zne_lambda_window
    return np.flatnonzero((np.asarray(strengths) > 0) & (lam >= lo) & (lam <= hi))


def zne_dataset(ansatz: str, n_qubits: int, depth: int, channel: str, thetas=None, h=None,
                strengths=None, R=None, M=None) -> dict:
    """
    Readout-corrected features at the folding scales and their zero-noise extrapolation.

    The scale-1 snapshots use the same random stream as the noisy dataset, so
    they coincide with its "ro" shadows; scales 3 and 5 have their own streams
    (independent circuits). The calibration is that of the noisy dataset.
    """
    from noise.density import gate_sequence
    from noise.mitigation import (ReadoutModel, extrapolation_coefficients, fold_gates,
                                  make_mitigation_noise_model, simulate_gates)
    R, M = _sizes(R, M)
    N, p = n_qubits, depth
    extra = {}
    if thetas is None:
        from noise.data import load_phase2
        h = field_grid(noisy=True)
        d = load_phase2(ansatz, N, p, h_sel=h)
        thetas = d["params"]
        extra = {"phase2_energy": d["energy"], "E0": d["E0"]}
    h = np.asarray(h, dtype=float)
    all_s = CFG.mitigation.strengths() if strengths is None else np.asarray(strengths, dtype=float)
    lam_all, n1, n2 = error_budget_for(ansatz, N, p, channel, all_s)
    win = zne_window(lam_all, all_s)
    if win.size == 0:
        raise ValueError(f"no strength with lambda in {CFG.ml.zne_lambda_window} for {ansatz} N={N} p={p}")
    sw = all_s[win]
    scales = tuple(int(s) for s in CFG.noise.zne_scale_factors)
    folding, method = CFG.mitigation.main_folding, CFG.mitigation.main_extrapolator
    coef = extrapolation_coefficients(scales, method)
    nw, nk, nh, nf = sw.size, len(scales), h.size, F.n_features(N)
    readout = ReadoutModel.from_config(N)
    Rm = readout.matrices()
    models = [make_mitigation_noise_model(channel, s) for s in sw]
    cals = [calibration(N, nm, R) for nm in models]

    X_gate = np.empty((nw, nk, nh, nf))
    X_ro = np.empty((nw, nk, nh, nf))
    Xs_ro = np.empty((nw, nk, nh, R, nf))
    for ih in range(nh):
        gates = gate_sequence(ansatz, N, p, np.asarray(thetas[ih], dtype=float))
        for ik, lamk in enumerate(scales):
            B, u, r = draw_snapshots(SOURCE_KEYS[ansatz], N, p, ih, R, M, scale=lamk)
            fg = fold_gates(gates, lamk, folding)
            for jw, nm in enumerate(models):
                pt = _noisy_point(simulate_gates(fg, N, nm), N, B, u, r, readout, Rm, cals[jw][0], cals[jw][1])
                X_gate[jw, ik, ih], X_ro[jw, ik, ih], Xs_ro[jw, ik, ih] = pt["X_gate"], pt["X_ro"], pt["Xs_ro"]
    out = common_arrays(h, N, R, M)
    out.update(extra)
    energy_sc = F.energy_from_features(X_gate, N, h, CFG.tfim.J)
    out.update(X_gate_scales=X_gate, X_ro_scales=X_ro, Xs_ro_scales=Xs_ro, energy_scales=energy_sc,
               X_zne_gate=np.einsum("k,wkhf->whf", coef, X_gate),
               X_zne_ro=np.einsum("k,wkhf->whf", coef, X_ro),
               Xs_zne_ro=np.einsum("k,wkhrf->whrf", coef, Xs_ro),
               energy_zne=np.einsum("k,wkh->wh", coef, energy_sc),
               strengths=sw, strength_index=win, lam=lam_all[win], scales=np.array(scales),
               coefficients=coef, folding=folding, extrapolator=method, n_1q=n1, n_2q=n2,
               params=np.asarray(thetas), cal_exact=np.stack([c[0] for c in cals]),
               cal_samples=np.stack([c[1] for c in cals]), depth=p, source="zne",
               ansatz=ansatz, channel=channel)
    return out


# ---------------------------------------------------------------------------
# Names, loading and sample matrices
# ---------------------------------------------------------------------------
def dataset_name(source: str, n_qubits: int, ansatz: str | None = None, depth: int | None = None,
                 channel: str | None = None) -> tuple:
    """(file name, subdir) of a dataset; organized by source, noise level is an array axis."""
    if source == "ed":
        return f"ds_ed_N{n_qubits}", f"{SUBDIR}/ed"
    if source == "vqe":
        return f"ds_vqe_{ansatz}_N{n_qubits}_p{depth}", f"{SUBDIR}/vqe"
    if source in ("noisy", "zne"):
        return f"ds_{source}_{ansatz}_N{n_qubits}_p{depth}_{channel}", f"{SUBDIR}/{source}"
    raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")


def load_dataset(source: str, n_qubits: int, ansatz=None, depth=None, channel=None) -> dict:
    from utils.io import load_results
    name, sub = dataset_name(source, n_qubits, ansatz, depth, channel)
    arrays, _ = load_results(name, subdir=sub)
    return arrays


def sample_matrix(ds: dict, kind: str = "shadow", variant: str | None = None, strength: int | None = None,
                  part: str = "train", feature_set: str = "corr") -> tuple:
    """
    (X, y, h) rows for training or testing.

    kind "exact" uses one row per field; kind "shadow" one row per field and
    repetition. part "train": training fields (and training repetitions);
    "test": all fields (test repetitions only). variant: None for ED and VQE,
    "gate", "meas" or "ro" for noisy data, "zne_gate" (exact only) or "zne_ro"
    for ZNE data. strength: index along the strength axis (noisy and ZNE).
    Rows with y = -1 (h = h_c) are kept; drop them for accuracies.
    """
    key = ("X" if kind == "exact" else "Xs") + (f"_{variant}" if variant else "")
    if key not in ds:
        raise KeyError(f"{key!r} not in this dataset")
    A = np.asarray(ds[key])
    if strength is not None:
        A = A[strength]
    h, y = np.asarray(ds["h"]), np.asarray(ds["y"])
    fields = np.asarray(ds["train_field"]) if part == "train" else np.asarray(ds["test_field"])
    N = int(ds["n_qubits"])
    if kind == "exact":
        X = A[fields]
        hh, yy = h[fields], y[fields]
    else:
        reps = np.asarray(ds["rep_train"]) if part == "train" else np.asarray(ds["rep_test"])
        sub = A[fields][:, reps]                                    # (n_f, n_r, F)
        X = sub.reshape(-1, sub.shape[-1])
        hh = np.repeat(h[fields], reps.sum())
        yy = np.repeat(y[fields], reps.sum())
    return F.select_features(X, N, feature_set), yy, hh


def snapshots(ds: dict, field: int, rep: int, strength: int | None = None, measured: bool = True) -> tuple:
    """(bases (M, N), bits (M, N)) of one shadow; noisy data: measured (readout) or gate-only bits."""
    N = int(ds["n_qubits"])
    if "bits" in ds:
        packed = ds["bits"][field, rep]
    else:
        packed = ds["bits_meas" if measured else "bits_gate"][strength, field, rep]
    return np.asarray(ds["bases"][field, rep]), S.unpack_bits(packed, N)


def write_index() -> Path:
    """results/phase6/dataset_index.json: every dataset file with its main parameters."""
    root = RESULTS_DIR / SUBDIR
    entries = []
    for sub in SOURCES:
        for js in sorted((root / sub).glob("ds_*.json")):
            with open(js) as f:
                meta = json.load(f)
            npz = js.with_suffix(".npz")
            entries.append({"name": js.stem, "subdir": f"{SUBDIR}/{sub}", "source": sub,
                            "params": meta.get("params", {}), "git_commit": meta.get("git_commit"),
                            "timestamp": meta.get("timestamp"),
                            "megabytes": round(npz.stat().st_size / 2 ** 20, 2) if npz.exists() else None})
    path = root / "dataset_index.json"
    root.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"datasets": entries, "count": len(entries)}, f, indent=2)
    return path
