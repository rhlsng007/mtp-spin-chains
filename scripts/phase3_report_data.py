"""
Collect the numbers needed for the Phase 3 report into one small JSON file.

Run from the project root, after the Phase 3 experiments:

    python scripts\\phase3_report_data.py

Writes results/phase3/phase3_report_data.json (tens of kB). Reads only saved
results and computes nothing new except Calabrese-Cardy fits of saved profiles.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import CFG, RESULTS_DIR
from analysis.entanglement import calabrese_cardy_fit
from analysis.vqe_entanglement import LN2, page_entropy
from utils.io import load_results, result_exists

SUBDIR = "phase3"
ANSATZE = ("hva", "hea")


def r(x, digits=4):
    a = np.asarray(x, dtype=float)
    f = np.vectorize(lambda v: float(f"{v:.{digits}g}") if np.isfinite(v) else None, otypes=[object])
    out = f(a) if a.size else a
    return out.tolist() if isinstance(out, np.ndarray) else out


def load(name):
    return load_results(name, SUBDIR) if result_exists(name, SUBDIR) else (None, None)


def cc(profile, n, boundary="open"):
    fit = calabrese_cardy_fit(profile, n, boundary, ell_min=CFG.ent.cc_ell_min)
    return fit["c"], fit["c_err"]


def main():
    out = {"config_ent": CFG.to_dict()["ent"], "sizes": list(CFG.system.sizes_noiseless),
           "ed": {}, "vqe": {}, "maxent": {}, "periodic": {}}
    meta_any = None
    for n in CFG.system.sizes_noiseless:
        ed, meta = load(f"ent_ed_N{n}")
        if ed is None:
            continue
        meta_any = meta_any or meta
        half = n // 2 - 1
        h = ed["h"]
        i_c = int(np.argmin(np.abs(h - CFG.ent.cc_field)))
        c_ed, c_ed_err = cc(ed["S_even"][i_c], n)
        out["ed"][f"N{n}"] = {
            "S_even_half_vs_h": r(ed["S_even"][:, half]),
            "S_broken_half_vs_h": r(ed["S_broken"][:, half]),
            "ordered_fields": r(h[ed["ground_space"].all(axis=1)]),
            "S_even_profile_hc": r(ed["S_even"][i_c]),
            "c_open": r(c_ed), "c_open_err": r(c_ed_err),
        }
        for a in ANSATZE:
            d, meta = load(f"ent_{a}_N{n}")
            if d is None:
                continue
            meta_any = meta_any or meta
            c_vals = [cc(d["S"][k, i_c], n) for k in range(len(d["depths"]))]
            dS_half = np.abs(d["S"][:, :, half] - d["S_ref"][:, :, half])
            out["vqe"][f"{a}_N{n}"] = {
                "depths": d["depths"].tolist(),
                "c_open_vs_depth": r([c for c, _ in c_vals]),
                "c_open_err_vs_depth": r([e for _, e in c_vals]),
                "S_half_hc_vs_depth": r(d["S"][:, i_c, half]),
                "S_half_vs_h_at_pN": r(d["S"][-1, :, half]),
                "S_profile_hc_at_pN": r(d["S"][-1, i_c]),
                "max_abs_dS_half_vs_depth": r(dS_half.max(axis=1)),
                "median_abs_dS_half_vs_depth": r(np.median(dS_half, axis=1)),
                "fidelity_gs_hc_vs_depth": r(d["fidelity_gs"][:, i_c]),
                "rank_fidelity_bound_hc_vs_depth": r(d["rank_fidelity_bound"][:, i_c]),
                "h_of_max_S_half_vs_depth": r(d["h"][np.argmax(d["S"][:, :, half], axis=1)]),
                "max_mirror_asymmetry_at_pN": r(np.abs(d["S"][-1] - d["S"][-1][:, ::-1]).max()),
            }
            m, _ = load(f"maxent_{a}_N{n}")
            if m is not None:
                bound = np.minimum(m["depths"], n // 2) * LN2
                out["maxent"][f"{a}_N{n}"] = {
                    "depths": m["depths"].tolist(),
                    "S2_max_over_ln2": r(m["S2_max"] / LN2),
                    "S_at_S2_max_over_ln2": r(m["S_at_S2_max"] / LN2),
                    "ratio_to_bound": r(m["S2_max"] / bound),
                    "random_S_mean_over_ln2": r(m["random_S_mean"] / LN2),
                    "random_S_max_over_ln2": r(m["random_S_max"] / LN2),
                    "page_over_ln2": r(page_entropy(n) / LN2),
                }
        p, _ = load(f"pbc_hva_N{n}")
        if p is not None:
            out["periodic"][f"N{n}"] = {
                "depths": p["depths"].tolist(),
                "c_vs_depth": r([cc(s, n, "periodic")[0] for s in p["S"]]),
                "c_ed": r(cc(p["S_ed"], n, "periodic")[0]),
                "rel_error_vs_depth": r(p["rel_error"]),
                "fidelity_vs_depth": r(p["fidelity"]),
            }
    out["h_grid"] = r(CFG.tfim.h_grid())
    if meta_any:
        out["git_commit"] = meta_any.get("git_commit")
        out["versions"] = meta_any.get("versions")
    summary = RESULTS_DIR / SUBDIR / "phase3_summary.txt"
    if summary.exists():
        out["summary_text"] = summary.read_text(encoding="utf-8")
    path = RESULTS_DIR / SUBDIR / "phase3_report_data.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[report data] written to {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
