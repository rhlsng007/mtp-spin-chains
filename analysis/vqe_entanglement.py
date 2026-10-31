"""
Entanglement of VQE states (Phase 3, tasks T3.1 to T3.4).

Builds on analysis/entanglement.py (Phase 1), which already handles complex
state vectors. This module adds what is specific to comparing VQE states with
exact ones.

References for a VQE state psi (TFIM, parity sectors even/odd):

    even         phi_even, the reference ground state of Phase 1
    broken       (phi_even + phi_odd) / sqrt 2, the symmetry-broken ED state
                 (the sign does not matter: the two signs differ by the global
                 flip P, a product of local unitaries)
    projected    P_GS psi / |P_GS psi|, the ED state of the ground space that is
                 closest to psi. For the HVA it is phi_even; for a symmetry-broken
                 HEA state in the ordered phase it is close to the broken state.
                 Its overlap with psi is exactly F_GS.

Rigorous bounds used as validation checks:

    Schmidt rank   Every layer of both ansaetze has exactly one two-qubit gate
                   (IsingZZ or CNOT, operator Schmidt rank 2) across any cut of
                   an open chain, and the initial states are products, so

                       rank <= 2^p   and   S_alpha(ell) <= min(p, ell, N - ell) ln 2.

    Fidelity       For a state of Schmidt rank r and any target phi,
                   |<phi|psi>|^2 <= sum of the r largest Schmidt weights of phi
                   (Eckart-Young), at every cut.

    Continuity     For pure states with overlap F, the reduced states differ in
                   trace distance by at most T = sqrt(1 - F), and the Fannes-
                   Audenaert inequality gives |S(rho) - S(sigma)| <= T ln(d - 1) + H2(T)
                   for T <= 1 - 1/d, with d the dimension of the smaller half.

Maximal entanglement (T3.3): the half-chain purity P = Tr rho_A^2 is minimized
(S_2 = -ln P maximized) with exact gradients. P = <psi| (rho_A (x) 1) |psi>, so
PurityOperator plays the role of a Hamiltonian in Ansatz.energy_and_grad; since
P is quartic in psi, the true gradient is twice the gradient returned there.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from analysis.entanglement import (entropy_from_spectrum, schmidt_spectrum)
from analysis.observables import n_sites_of

LN2 = float(np.log(2.0))


# ---------------------------------------------------------------------------
# Spectra, ranks, profiles
# ---------------------------------------------------------------------------
def schmidt_rank(psi: np.ndarray, ell: int, tol: float) -> int:
    """Number of Schmidt weights above tol at the cut after ell sites."""
    return int(np.sum(schmidt_spectrum(psi, ell) > tol))


def profile_data(psi: np.ndarray, rank_tol: float) -> dict:
    """S_vN(ell), S_2(ell) and the Schmidt rank for every cut ell = 1..N-1, one SVD each."""
    n = n_sites_of(psi)
    spectra = [schmidt_spectrum(psi, ell) for ell in range(1, n)]
    return {
        "S": np.array([entropy_from_spectrum(lam, 1.0) for lam in spectra]),
        "S2": np.array([entropy_from_spectrum(lam, 2.0) for lam in spectra]),
        "rank": np.array([int(np.sum(lam > rank_tol)) for lam in spectra]),
    }


def entropy_bound(depth: int, n_sites: int) -> np.ndarray:
    """min(p, ell, N - ell) ln 2 for ell = 1..N-1 (open-chain HVA and HEA)."""
    ell = np.arange(1, n_sites)
    return np.minimum(np.minimum(ell, n_sites - ell), depth) * LN2


def page_entropy(n_sites: int) -> float:
    """
    Average half-chain entropy of a Haar-random pure state (Page 1993),
    for subsystems of dimension m <= n:  sum_{k=n+1}^{mn} 1/k - (m-1)/(2n).
    """
    m = 2 ** (n_sites // 2)
    n = 2 ** (n_sites - n_sites // 2)
    k = np.arange(n + 1, m * n + 1)
    return float(np.sum(1.0 / k) - (m - 1) / (2 * n))


# ---------------------------------------------------------------------------
# Reference states
# ---------------------------------------------------------------------------
def symmetry_broken_state(phi_even: np.ndarray, phi_odd: np.ndarray) -> np.ndarray:
    psi = (phi_even + phi_odd) / np.sqrt(2.0)
    return psi / np.linalg.norm(psi)


def projected_reference(psi: np.ndarray, sector_states: np.ndarray, mask: np.ndarray) -> tuple:
    """
    (phi_ref, F_GS): the normalized projection of psi onto the ground space
    spanned by the sector states selected by mask, and its weight.
    """
    states = sector_states[mask]
    amps = np.conj(states) @ psi
    weight = float(np.sum(np.abs(amps) ** 2))
    phi = amps @ states
    return phi / np.linalg.norm(phi), weight


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------
def binary_entropy(t: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -t * np.log(t) - (1 - t) * np.log(1 - t)
    return np.nan_to_num(h)


def audenaert_bound(fidelity, dim: int) -> np.ndarray:
    """
    Largest |S_vN(rho_A) - S_vN(sigma_A)| allowed for two pure states with overlap F,
    with T = sqrt(1 - F) bounding the trace distance of the reduced states.
    For T > 1 - 1/d the trivial bound ln d is returned.
    """
    t = np.sqrt(np.clip(1.0 - np.asarray(fidelity, dtype=float), 0.0, 1.0))
    bound = t * np.log(dim - 1.0) + binary_entropy(t)
    return np.where(t <= 1.0 - 1.0 / dim, bound, np.log(dim))


def rank_fidelity_bound(phi: np.ndarray, rank: int) -> float:
    """Largest overlap of phi with any state of Schmidt rank <= rank, minimized over cuts."""
    n = n_sites_of(phi)
    best = 1.0
    for ell in range(1, n):
        lam = schmidt_spectrum(phi, ell)
        best = min(best, float(np.sum(lam[:rank])))
    return best


# ---------------------------------------------------------------------------
# Maximal entanglement of an ansatz (T3.3)
# ---------------------------------------------------------------------------
class PurityOperator:
    """
    Acts as (rho_A (x) 1) built from the vector it is applied to, so that
    <psi| op @ psi> = Tr rho_A^2 for A = the first ell sites.
    """

    def __init__(self, n_sites: int, ell: int = None):
        self.n = n_sites
        self.ell = n_sites // 2 if ell is None else ell

    def __matmul__(self, psi: np.ndarray) -> np.ndarray:
        m = psi.reshape(2 ** self.ell, 2 ** (self.n - self.ell))
        rho = m @ m.conj().T
        return (rho @ m).reshape(psi.shape)


def purity_and_grad(ansatz, theta: np.ndarray, op: PurityOperator) -> tuple:
    p, g = ansatz.energy_and_grad(theta, op)
    return p, 2.0 * g


def maximize_entanglement(ansatz, x0: np.ndarray, maxiter: int, gtol: float) -> dict:
    """Minimize the half-chain purity from x0; returns the state's S_2 and S_vN."""
    op = PurityOperator(ansatz.n_qubits)
    res = scipy_minimize(lambda x: purity_and_grad(ansatz, x, op), np.asarray(x0, float),
                         jac=True, method="L-BFGS-B",
                         options={"maxiter": maxiter, "gtol": gtol, "ftol": 1e-15})
    psi = ansatz.state(res.x)
    lam = schmidt_spectrum(psi, ansatz.n_qubits // 2)
    return {"S2": float(-np.log(res.fun)), "S": entropy_from_spectrum(lam, 1.0),
            "x": res.x, "n_iter": int(res.nit), "success": bool(res.success)}
