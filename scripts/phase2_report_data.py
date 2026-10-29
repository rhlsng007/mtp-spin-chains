"""
Collect every number needed for the Phase 2 report into one small JSON file.

Run from the project root, after the Phase 2 experiments:

    python scripts\\phase2_report_data.py

Writes results/phase2/phase2_report_data.json (a few hundred kB; the optimized
states and parameters are left out). Reads only saved results, computes nothing new.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import CFG, RESULTS_DIR
from utils.io import load_results, result_exists
from vqe.metrics import required_depth

SUBDIR = "phase2"
DIGITS = 4          # significant digits kept in the JSON


def r(x):
    """Round floats (and arrays of floats) to DIGITS significant digits for compact JSON."""
    a = np.asarray(x)
    if a.dtype == bool:
        return a.tolist()
    if np.issubdtype(a.dtype, np.integer):
        return a.tolist()
    a = a.astype(float)
    out = np.vectorize(lambda v: float(f"{v:.{DIGITS}g}") if np.isfinite(v) else None,
                       otypes=[object])(a) if a.size else a
    return out.tolist() if isinstance(out, np.ndarray) else out


def load(name):
    return load_results(name, SUBDIR) if result_exists(name, SUBDIR) else (None, None)


def sweep_section(ansatz: str, n: int) -> dict:
    depths = CFG.system.depths(n)
    files = [load(f"vqe_{ansatz}_N{n}_p{p}") for p in depths]
    if any(d is None for d, _ in files):
        return None
    arrays = [d for d, _ in files]
    meta = files[-1][1]
    h = arrays[0]["h"]
    stack = lambda key: np.array([a[key] for a in arrays])
    rel, fid, fgs = stack("rel_error"), stack("fidelity"), stack("fidelity_gs")
    n_iter, success = stack("n_iter"), stack("success")
    target = CFG.vqe.target_rel_error
    anchors = {}
    for hv in CFG.vqe.anchor_fields:
        i = int(np.argmin(np.abs(h - hv)))
        anchors[f"{hv:g}"] = {"rel_error": r(rel[:, i]), "fidelity": r(fid[:, i]),
                              "fidelity_gs": r(fgs[:, i]), "n_iter": r(n_iter[:, i])}
    last = arrays[-1]
    i_min = int(np.argmin(last["fidelity_gs"]))
    return {
        "depths": depths,
        "n_params": [int(a["n_params"]) for a in arrays],
        "at_anchor_fields_vs_depth": anchors,
        "at_p_equals_N_vs_field": {
            "rel_error": r(last["rel_error"]), "fidelity": r(last["fidelity"]),
            "fidelity_gs": r(last["fidelity_gs"]), "parity": r(last["parity"]),
            "n_iter": r(last["n_iter"]), "best_source": r(last["best_source"]),
        },
        "max_rel_error_vs_depth": r(rel.max(axis=1)),
        "median_rel_error_vs_depth": r(np.median(rel, axis=1)),
        "min_fidelity_gs_vs_depth": r(fgs.min(axis=1)),
        "median_iterations_vs_depth": r(np.median(n_iter, axis=1)),
        "fraction_at_max_iterations_vs_depth": r((n_iter >= CFG.vqe.max_iterations).mean(axis=1)),
        "fraction_success_vs_depth": r(success.mean(axis=1)),
        "required_depth_vs_field": r([required_depth(depths, rel[:, i], target)
                                      for i in range(len(h))]),
        "worst_point_at_p_equals_N": {
            "h": float(h[i_min]), "fidelity_gs": r(last["fidelity_gs"][i_min]),
            "fidelity": r(last["fidelity"][i_min]), "rel_error": r(last["rel_error"][i_min]),
            "parity": r(last["parity"][i_min]), "n_iter": int(last["n_iter"][i_min]),
        },
        "sweep_seconds_vs_depth": r([float(a["seconds"].sum()) for a in arrays]),
        "git_commit": meta.get("git_commit"),
        "versions": meta.get("versions"),
    }


def main():
    out = {
        "config": {"tfim": CFG.to_dict()["tfim"], "vqe": CFG.to_dict()["vqe"],
                   "sizes": list(CFG.system.sizes_noiseless)},
        "h_grid": r(CFG.tfim.h_grid()),
        "sweeps": {},
    }
    for ansatz in ("hva", "hea"):
        for n in CFG.system.sizes_noiseless:
            sec = sweep_section(ansatz, n)
            if sec is not None:
                out["sweeps"][f"{ansatz}_N{n}"] = sec

    n_wc = CFG.vqe.warmcold_size
    for ansatz in ("hva", "hea"):
        d, _ = load(f"vqe_warmcold_{ansatz}_N{n_wc}_p{max(n_wc // 2, 1)}")
        if d is not None:
            out.setdefault("warm_vs_cold", {})[ansatz] = {
                k: r(d[k]) for k in d if k.split("_", 1)[-1] in
                ("rel_error", "fidelity_gs", "n_iter", "n_fev", "total_fev", "n_candidates", "seconds")
            }

    for n in CFG.vqe.hva_depth_scan_sizes:
        d, _ = load(f"vqe_hva_depthscan_N{n}")
        if d is not None:
            out.setdefault("hva_depth_scan", {})[f"N{n}"] = {
                "fields": r(d["fields"]), "depths": r(d["depths"]),
                "rel_error": r(d["rel_error"]), "fidelity": r(d["fidelity"]),
                "n_iter": r(d["n_iter"])}

    d, _ = load("vqe_hva_pbc_check")
    if d is not None:
        out["hva_periodic_check"] = {k: r(d[k]) for k in ("sizes", "fields", "rel_error", "fidelity")}

    for fam in ("hea_linear", "hea_fixed", "hva_linear"):
        d, _ = load(f"bp_{fam}")
        if d is not None:
            out.setdefault("barren_plateau", {})[fam] = {k: r(d[k]) for k in d if k != "n_samples"}
            out["barren_plateau"][fam]["n_samples"] = int(d["n_samples"])

    summary = RESULTS_DIR / SUBDIR / "phase2_summary.txt"
    if summary.exists():
        out["summary_text"] = summary.read_text(encoding="utf-8")

    path = RESULTS_DIR / SUBDIR / "phase2_report_data.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[report data] {len(out['sweeps'])} sweep tables written to {path} "
          f"({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
