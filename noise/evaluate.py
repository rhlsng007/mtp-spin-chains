"""
All Phase 4 metrics of one noisy state (T4.6).

    energy, rel_error     Tr(rho H) and (Tr(rho H) - E0) / |E0|
    fidelity              <phi_even| rho |phi_even>      (Phase 2 convention)
    fidelity_gs           Tr(P_GS rho)                   (ground-space fidelity)
    fidelity_ideal        <psi_0| rho |psi_0>, psi_0 = noiseless state of the same
                          ansatz, depth and parameters (Phase 3 decision: compare
                          per ansatz and depth)
    purity, S2_total, S_total
    S_A, S_B, S2_A, S2_B, I, I2, E_N, S_cond      half-chain cut, and profiles
"""

from __future__ import annotations

import numpy as np

from analysis.mixed_state import fidelity_pure, mixed_state_metrics, projector_weight
from noise.data import Reference, energy

SCALAR_KEYS = ("energy", "rel_error", "fidelity", "fidelity_gs", "fidelity_ideal",
               "purity", "S2_total", "S_total", "S_A", "S_B", "S2_A", "S2_B",
               "I", "I2", "E_N", "S_cond", "trace", "min_eig", "herm_err")
PROFILE_KEYS = ("S_A_profile", "S2_A_profile", "I_profile", "I2_profile",
                "E_N_profile", "S_cond_profile")


def evaluate_state(rho: np.ndarray, ref: Reference, psi_ideal: np.ndarray | None,
                   n_qubits: int, profiles: bool = True) -> dict:
    out = mixed_state_metrics(rho, n_qubits, profiles=profiles)
    E = energy(rho, ref.H)
    out["energy"] = E
    out["rel_error"] = (E - ref.E0) / abs(ref.E0)
    out["fidelity"] = fidelity_pure(rho, ref.phi)
    out["fidelity_gs"] = projector_weight(rho, ref.gs_states)
    out["fidelity_ideal"] = np.nan if psi_ideal is None else fidelity_pure(rho, psi_ideal)
    out["herm_err"] = float(np.abs(rho - rho.conj().T).max())
    return out
