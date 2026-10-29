"""
Comparison of VQE results with exact diagonalization (Phase 2, task T2.6).

Energy error (reference E0 = lowest energy over all symmetry sectors):

    delta E = E_VQE - E0,      epsilon = (E_VQE - E0) / |E0|

Fidelities, with |phi_s> the lowest state of symmetry sector s (Phase 1):

    F_s  = |<phi_s|psi>|^2,     F = F_{s0}  (s0 = reference ground sector, even parity)
    F_GS = sum_{s in GS} F_s

GS (the ground space) contains every sector whose lowest level lies within
CFG.vqe.ground_space_gap_ratio times the in-sector gap of E0. In the TFIM
ordered phase both parity states belong to it (their splitting falls
exponentially with N), so a symmetry-broken HEA state with F = 1/2 still
has F_GS close to 1. The HVA conserves parity, so for it F_GS = F.
"""

from __future__ import annotations

import numpy as np

from config import CFG
from models.exact import solve_model
from utils.io import load_results, result_exists


# ---------------------------------------------------------------------------
# Exact reference
# ---------------------------------------------------------------------------
def ed_reference(n_sites: int, h_values: np.ndarray, boundary: str = None,
                 verbose: bool = True) -> dict:
    """
    Sector-resolved ED data for the TFIM at every field value.

    Reuses results/phase1/ed_tfim_N{N} when it exists, holds the states and
    has the same field grid; otherwise diagonalizes directly (fast for N <= 14).
    """
    c = CFG.tfim
    boundary = c.boundary if boundary is None else boundary
    h_values = np.asarray(h_values, dtype=float)
    name = f"ed_tfim_N{n_sites}"
    if boundary == "open" and result_exists(name, "phase1"):
        d = load_results(name, "phase1")[0]
        if ("states" in d and len(d["h"]) == len(h_values)
                and np.allclose(d["h"], h_values)):
            return {"h": d["h"], "states": d["states"],
                    "sector_energies": d["sector_energies"],
                    "ground_sector": d["ground_sector"].astype(int)}
    if verbose:
        print(f"    [ED] diagonalizing TFIM N={n_sites} ({boundary}) at {len(h_values)} fields")
    states, energies, ground = [], [], []
    for h in h_values:
        res = solve_model("tfim", n_sites, J=c.J, h=float(h), boundary=boundary)
        states.append(res.sector_states)
        energies.append(res.sector_energies)
        ground.append(res.ground_sector)
    return {"h": h_values, "states": np.asarray(states),
            "sector_energies": np.asarray(energies), "ground_sector": np.asarray(ground)}


def reference_energy(ref: dict) -> np.ndarray:
    """Lowest energy over all sectors, for every field value."""
    return ref["sector_energies"][:, :, 0].min(axis=1)


def ground_space_mask(ref: dict, gap_ratio: float = None) -> np.ndarray:
    """(n_h, n_sectors) mask of sectors belonging to the ground space."""
    gap_ratio = CFG.vqe.ground_space_gap_ratio if gap_ratio is None else gap_ratio
    se = ref["sector_energies"]
    gs = ref["ground_sector"]
    rows = np.arange(len(gs))
    e0 = se[rows, gs, 0]
    gap = se[rows, gs, 1] - e0
    return (se[:, :, 0] - e0[:, None]) <= gap_ratio * gap[:, None]


# ---------------------------------------------------------------------------
# State diagnostics
# ---------------------------------------------------------------------------
def sector_overlaps(psi: np.ndarray, sector_states: np.ndarray) -> np.ndarray:
    """|<phi_s|psi>|^2 for every sector state."""
    return np.abs(np.conj(sector_states) @ psi) ** 2


def parity_expectation(psi: np.ndarray) -> float:
    """<psi| prod_i X_i |psi>; the global flip reverses the state vector."""
    return float(np.real(np.vdot(psi, psi[::-1])))


def evaluate_sweep(ansatz, params: np.ndarray, energies: np.ndarray, ref: dict) -> dict:
    """All Phase 2 metrics for a set of optimized parameters (one per field value)."""
    n_h = len(params)
    e0 = reference_energy(ref)
    mask = ground_space_mask(ref)
    states = np.array([ansatz.state(p) for p in params])
    overlaps = np.array([sector_overlaps(states[i], ref["states"][i]) for i in range(n_h)])
    rows = np.arange(n_h)
    return {
        "E0": e0,
        "energy_error": energies - e0,
        "rel_error": (energies - e0) / np.abs(e0),
        "fidelity": overlaps[rows, ref["ground_sector"]],
        "fidelity_sectors": overlaps,
        "fidelity_gs": (overlaps * mask).sum(axis=1),
        "ground_space": mask,
        "parity": np.array([parity_expectation(s) for s in states]),
        "states": states,
    }


def required_depth(depths, rel_errors, target: float) -> float:
    """Smallest depth with relative error below target (NaN if never reached)."""
    for p, e in zip(depths, rel_errors):
        if e < target:
            return float(p)
    return np.nan
