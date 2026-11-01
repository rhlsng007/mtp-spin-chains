"""
Inputs of the Phase 4 experiments: Phase 2 optima, Phase 3 entropies, exact references.

The noisy field grid (step 0.1) is a subset of the Phase 2 grid (step 0.05),
so the Phase 2 optimal parameters are reused directly (T4.5, experiment A).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CFG
from models.exact import solve_model
from models.hamiltonians import build_hamiltonian, hamiltonian_terms
from utils.io import load_results


def _arrays(name: str, subdir: str):
    out = load_results(name, subdir=subdir)
    if isinstance(out, tuple):          # (arrays, metadata)
        out = out[0]
    return out


def field_indices(h_all: np.ndarray, h_sel: np.ndarray) -> np.ndarray:
    idx = []
    for h in h_sel:
        hit = np.flatnonzero(np.isclose(h_all, h, atol=1e-9))
        if hit.size != 1:
            raise ValueError(f"field {h} not found exactly once in the Phase 2 grid")
        idx.append(int(hit[0]))
    return np.array(idx)


def load_phase2(ansatz: str, n_qubits: int, depth: int, h_sel=None) -> dict:
    """Phase 2 sweep data, restricted to the fields h_sel (default: noisy grid)."""
    h_sel = CFG.tfim.h_grid(noisy=True) if h_sel is None else np.atleast_1d(h_sel)
    a = _arrays(f"vqe_{ansatz}_N{n_qubits}_p{depth}", "phase2")
    idx = field_indices(np.asarray(a["h"]), h_sel)
    keys = ("h", "params", "states", "energy", "E0", "fidelity", "fidelity_gs", "ground_space")
    return {k: np.asarray(a[k])[idx] for k in keys}


def load_phase3_half_entropy(ansatz: str, n_qubits: int, depth: int, h_sel=None) -> dict | None:
    """Phase 3 half-chain S and S2 of the noiseless VQE state, or None if absent."""
    h_sel = CFG.tfim.h_grid(noisy=True) if h_sel is None else np.atleast_1d(h_sel)
    try:
        a = _arrays(f"ent_{ansatz}_N{n_qubits}", "phase3")
    except (FileNotFoundError, OSError, KeyError):
        return None
    depths = list(np.asarray(a["depths"]).astype(int))
    if depth not in depths:
        return None
    ip = depths.index(depth)
    idx = field_indices(np.asarray(a["h"]), h_sel)
    cut = n_qubits // 2 - 1
    return {"S": np.asarray(a["S"])[ip, idx, cut], "S2": np.asarray(a["S2"])[ip, idx, cut]}


@dataclass
class Reference:
    """Exact data for one field: Hamiltonian, energy, even ground state, ground space."""
    h: float
    H: object                   # scipy CSR matrix
    terms: list
    E0: float
    phi: np.ndarray             # even-parity ground state (Phase 2 fidelity reference)
    gs_states: list             # lowest states of the ground-space sectors
    gs_mask: np.ndarray


def exact_reference(n_qubits: int, h: float, ground_space=None) -> Reference:
    """
    ``ground_space`` (bool per sector) is taken from Phase 2 when given; otherwise
    the Phase 2 rule is applied: sector s belongs to the ground space if
    E_s - E0 < ratio * (gap inside the ground-state sector).
    """
    params = dict(J=CFG.tfim.J, h=float(h), boundary=CFG.tfim.boundary)
    res = solve_model("tfim", n_qubits, **params)
    states = [np.asarray(s, dtype=complex) for s in np.asarray(res.sector_states)]
    if ground_space is None:
        e = np.asarray(res.sector_energies, dtype=float)
        e = e[:, 0] if e.ndim == 2 else e
        ground_space = (e - res.ground_energy) < CFG.vqe.ground_space_gap_ratio * res.gap_sector
    mask = np.asarray(ground_space, dtype=bool)
    return Reference(h=float(h), H=build_hamiltonian("tfim", n_qubits, **params),
                     terms=hamiltonian_terms("tfim", n_qubits, **params),
                     E0=float(res.ground_energy),
                     phi=np.asarray(res.ground_state, dtype=complex),
                     gs_states=[s for s, m in zip(states, mask) if m], gs_mask=mask)


def energy(rho: np.ndarray, H) -> float:
    return float(np.real(np.sum(H.multiply(rho.T)))) if hasattr(H, "multiply") else float(np.real(np.trace(H @ rho)))
