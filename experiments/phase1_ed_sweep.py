"""
Phase 1, task T1.5: exact-diagonalization sweeps for all models and sizes.

Run from the project root:

    python experiments/phase1_ed_sweep.py                 # everything
    python experiments/phase1_ed_sweep.py --sizes 4 6 8   # quick run
    python experiments/phase1_ed_sweep.py --models tfim --recompute
    (model choices: tfim tfim_pbc xxz cluster cc)

Results go to results/phase1/:

    ed_tfim_N{N}             TFIM field sweep (open, the VQE reference)
    ed_tfim_pbc_N{N}         TFIM field sweep, periodic chain (no states)
    ed_xxz_N{N}              XXZ anisotropy sweep
    ed_cluster_N{N}_h2_{v}   cluster-Ising h1 sweep at fixed h2
    ed_cc_profiles_N{N}      entropy profiles at criticality, open + periodic

Every sweep file contains, for each parameter value:
    E0, E1, E2, gap (full spectrum), gap_sector (inside ground sector),
    sector_energies, ground_sector, sector_labels,
    S_half, S2_half, S_profile, S2_profile, zz_corr, M2, M4, binder, Ms2,
    and (if CFG.ed.save_states) states[n_param, n_sectors, 2^N]: the lowest
    state of every symmetry sector, for fidelities in Phase 2 and features in
    Phase 6. Cluster files also hold string_order.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config import CFG
from models.exact import solve_model
from analysis.observables import (binder_cumulant, magnetization_moments,
                                  string_order_parameter, zz_correlation_matrix)
from analysis.entanglement import entropy_profiles
from utils.io import load_or_compute

SUBDIR = "phase1"

try:
    from tqdm import tqdm
except ImportError:                      # tqdm is optional
    def tqdm(iterable, **kwargs):
        return iterable


# ---------------------------------------------------------------------------
# Generic sweep
# ---------------------------------------------------------------------------
def sweep(model: str, n_sites: int, param_name: str, values: np.ndarray,
          fixed: dict, extra_observables: dict = None, save_states: bool = None) -> dict:
    """
    Diagonalize `model` at every value of `param_name` and collect observables.

    extra_observables: {name: f(psi) -> float} evaluated on each ground state.
    save_states: store sector states (default CFG.ed.save_states).
    """
    extra_observables = extra_observables or {}
    save_states = CFG.ed.save_states if save_states is None else save_states
    rows = {k: [] for k in ("E0", "E1", "E2", "gap", "gap_sector", "ground_sector",
                            "sector_energies", "S_half", "S2_half", "S_profile",
                            "S2_profile", "zz_corr", "M2", "M4", "binder", "Ms2", "states")}
    for name in extra_observables:
        rows[name] = []
    sector_labels = None

    for value in tqdm(values, desc=f"{model} N={n_sites}", leave=False):
        res = solve_model(model, n_sites, **fixed, **{param_name: float(value)})
        psi = res.ground_state
        sector_labels = res.sector_labels

        e = res.energies
        rows["E0"].append(res.ground_energy)
        rows["E1"].append(e[1])
        rows["E2"].append(e[2] if len(e) > 2 else np.nan)
        rows["gap"].append(res.gap)
        rows["gap_sector"].append(res.gap_sector)
        rows["ground_sector"].append(res.ground_sector)
        rows["sector_energies"].append(res.sector_energies)

        prof = entropy_profiles(psi, alphas=(1.0, 2.0))
        rows["S_profile"].append(prof[1.0])
        rows["S2_profile"].append(prof[2.0])
        rows["S_half"].append(prof[1.0][n_sites // 2 - 1])
        rows["S2_half"].append(prof[2.0][n_sites // 2 - 1])

        rows["zz_corr"].append(zz_correlation_matrix(psi))
        m2, m4 = magnetization_moments(psi)
        rows["M2"].append(m2)
        rows["M4"].append(m4)
        rows["binder"].append(binder_cumulant(m2, m4))
        rows["Ms2"].append(magnetization_moments(psi, staggered=True)[0])
        if save_states:
            rows["states"].append(res.sector_states)

        for name, fn in extra_observables.items():
            rows[name].append(fn(psi))

    if not save_states:
        rows.pop("states")
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    arrays[param_name] = np.asarray(values, dtype=float)
    arrays["sector_labels"] = np.asarray(sector_labels, dtype=np.int8)
    return arrays


# ---------------------------------------------------------------------------
# Model-specific sweeps
# ---------------------------------------------------------------------------
def run_tfim(n_sites: int, recompute: bool) -> dict:
    c = CFG.tfim
    fixed = {"J": c.J, "boundary": c.boundary}
    return load_or_compute(
        f"ed_tfim_N{n_sites}",
        lambda: sweep("tfim", n_sites, "h", c.h_grid(), fixed),
        params={"model": "tfim", "N": n_sites, **fixed},
        subdir=SUBDIR, recompute=recompute)


def run_tfim_periodic(n_sites: int, recompute: bool) -> dict:
    """
    Supplementary periodic-chain TFIM sweep (no stored states). With periodic
    boundaries the critical entropy (c/3) ln N + s1 exceeds the ln 2 cat-state
    plateau already at small N, so the half-chain entropy peak near h_c is
    visible; with open boundaries it is not (see phase1_plots.py).
    """
    c = CFG.tfim
    fixed = {"J": c.J, "boundary": "periodic"}
    return load_or_compute(
        f"ed_tfim_pbc_N{n_sites}",
        lambda: sweep("tfim", n_sites, "h", c.h_grid(), fixed, save_states=False),
        params={"model": "tfim", "N": n_sites, **fixed},
        subdir=SUBDIR, recompute=recompute)


def run_xxz(n_sites: int, recompute: bool) -> dict:
    c = CFG.xxz
    fixed = {"J": c.J, "boundary": c.boundary}
    return load_or_compute(
        f"ed_xxz_N{n_sites}",
        lambda: sweep("xxz", n_sites, "delta", c.delta_grid(), fixed),
        params={"model": "xxz", "N": n_sites, **fixed},
        subdir=SUBDIR, recompute=recompute)


def h2_tag(h2: float) -> str:
    """File-name-safe tag: -1.0 -> 'm1', 0.0 -> '0', 1.0 -> '1', 0.5 -> '0p5'."""
    return f"{h2:g}".replace("-", "m").replace(".", "p")


def run_cluster(n_sites: int, recompute: bool) -> None:
    c = CFG.cluster
    for h2 in c.h2_values:
        fixed = {"J": c.J, "h2": float(h2), "boundary": c.boundary}
        load_or_compute(
            f"ed_cluster_N{n_sites}_h2_{h2_tag(h2)}",
            lambda: sweep("cluster", n_sites, "h1", c.h1_grid(), fixed,
                          extra_observables={"string_order": string_order_parameter}),
            params={"model": "cluster", "N": n_sites, **fixed},
            subdir=SUBDIR, recompute=recompute)


def run_cc_profiles(n_sites: int, recompute: bool) -> dict:
    """
    Entropy profiles S(ell) at critical points for one size, open and periodic:
    TFIM at h = h_c and XXZ at each delta in CFG.ed.xxz_cc_deltas.
    Keys look like 'tfim_open' or 'xxz_d0p5_periodic'.
    """
    def compute():
        out = {}
        for boundary in CFG.ed.cc_boundaries:
            res = solve_model("tfim", n_sites, J=CFG.tfim.J, h=CFG.tfim.h_critical,
                              boundary=boundary)
            out[f"tfim_{boundary}"] = entropy_profiles(res.ground_state, (1.0,))[1.0]
            for d in CFG.ed.xxz_cc_deltas:
                res = solve_model("xxz", n_sites, J=CFG.xxz.J, delta=float(d), boundary=boundary)
                out[f"xxz_d{h2_tag(d)}_{boundary}"] = entropy_profiles(res.ground_state, (1.0,))[1.0]
        return out

    return load_or_compute(f"ed_cc_profiles_N{n_sites}", compute,
                           params={"N": n_sites, "boundaries": list(CFG.ed.cc_boundaries),
                                   "xxz_deltas": list(CFG.ed.xxz_cc_deltas)},
                           subdir=SUBDIR, recompute=recompute)


RUNNERS = {"tfim": run_tfim, "tfim_pbc": run_tfim_periodic, "xxz": run_xxz,
           "cluster": run_cluster, "cc": run_cc_profiles}


def main():
    parser = argparse.ArgumentParser(description="Phase 1 exact-diagonalization sweeps")
    parser.add_argument("--models", nargs="+",
                        default=["tfim", "tfim_pbc", "xxz", "cluster", "cc"],
                        choices=["tfim", "tfim_pbc", "xxz", "cluster", "cc"])
    parser.add_argument("--sizes", nargs="+", type=int, default=list(CFG.system.sizes_exact))
    parser.add_argument("--recompute", action="store_true", help="ignore cached results")
    args = parser.parse_args()

    t_start = time.time()
    for model in args.models:
        for n in args.sizes:
            t0 = time.time()
            RUNNERS[model](n, args.recompute)
            print(f"   {model:8s} N={n:2d}  {time.time() - t0:6.1f} s")
    print(f"Phase 1 sweep finished in {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
