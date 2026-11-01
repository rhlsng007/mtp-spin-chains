"""
Mixed-state quantities for noisy VQE states (Phase 4, T4.6).

All bipartitions are contiguous cuts of the open chain: A = sites 0..ell-1,
B = sites ell..N-1 (wire 0 is the most significant bit), so reduced density
matrices and partial transposes are reshapes of the (2^N, 2^N) matrix.

For a cut ell:

    S(A), S(B), S(AB)            von Neumann entropies (natural log)
    S2(A), S2(B), S2(AB)         second Renyi entropies, S2 = -ln Tr rho^2
    I(A:B)   = S(A) + S(B) - S(AB)        mutual information
    I2(A:B)  = S2(A) + S2(B) - S2(AB)     Renyi-2 mutual information
    E_N      = ln || rho^{T_A} ||_1       logarithmic negativity
    S(A|B)   = S(AB) - S(B)               conditional entropy (< 0 certifies
                                          distillable entanglement, hashing bound)

For a pure state S(AB) = 0, I = 2 S(A) and E_N equals the Renyi-1/2 entropy
2 ln sum_k sqrt(lambda_k); these identities are checked in the tests.
"""

from __future__ import annotations

import numpy as np

EIG_CUTOFF = 1e-15        # eigenvalues below this are treated as zero in S = -sum lam ln lam


def entropy_from_eigenvalues(lam: np.ndarray) -> float:
    lam = np.asarray(lam, dtype=float)
    lam = lam[lam > EIG_CUTOFF]
    return float(-np.sum(lam * np.log(lam)))


def purity(rho: np.ndarray) -> float:
    """Tr rho^2 = sum_ij |rho_ij|^2 for Hermitian rho (exact, no eigendecomposition)."""
    return float(np.vdot(rho, rho).real)


def renyi2(rho: np.ndarray) -> float:
    return float(-np.log(purity(rho)))


def von_neumann(rho: np.ndarray) -> float:
    return entropy_from_eigenvalues(np.linalg.eigvalsh(rho))


def reduced_left(rho: np.ndarray, n_qubits: int, ell: int) -> np.ndarray:
    """rho_A for A = sites 0..ell-1."""
    da, db = 2 ** ell, 2 ** (n_qubits - ell)
    return np.einsum("ijkj->ik", rho.reshape(da, db, da, db))


def reduced_right(rho: np.ndarray, n_qubits: int, ell: int) -> np.ndarray:
    """rho_B for B = sites ell..N-1."""
    da, db = 2 ** ell, 2 ** (n_qubits - ell)
    return np.einsum("ijil->jl", rho.reshape(da, db, da, db))


def partial_transpose_left(rho: np.ndarray, n_qubits: int, ell: int) -> np.ndarray:
    """rho^{T_A}: transpose of the A indices."""
    da, db = 2 ** ell, 2 ** (n_qubits - ell)
    d = da * db
    return rho.reshape(da, db, da, db).transpose(2, 1, 0, 3).reshape(d, d)


def log_negativity(rho: np.ndarray, n_qubits: int, ell: int) -> float:
    ev = np.linalg.eigvalsh(partial_transpose_left(rho, n_qubits, ell))
    return float(np.log(np.sum(np.abs(ev))))


def fidelity_pure(rho: np.ndarray, phi: np.ndarray) -> float:
    """<phi| rho |phi> for a normalized pure reference phi."""
    phi = np.asarray(phi, dtype=complex)
    return float(np.vdot(phi, rho @ phi).real)


def projector_weight(rho: np.ndarray, states) -> float:
    """Tr(P rho) for P the projector onto orthonormal states (ground-space fidelity)."""
    return float(sum(fidelity_pure(rho, s) for s in states))


def cut_quantities(rho: np.ndarray, n_qubits: int, ell: int,
                   S_AB: float | None = None, S2_AB: float | None = None) -> dict:
    rA = reduced_left(rho, n_qubits, ell)
    rB = reduced_right(rho, n_qubits, ell)
    S_A, S_B = von_neumann(rA), von_neumann(rB)
    S2_A, S2_B = renyi2(rA), renyi2(rB)
    if S_AB is None:
        S_AB = von_neumann(rho)
    if S2_AB is None:
        S2_AB = renyi2(rho)
    return {
        "S_A": S_A, "S_B": S_B, "S2_A": S2_A, "S2_B": S2_B,
        "I": S_A + S_B - S_AB, "I2": S2_A + S2_B - S2_AB,
        "E_N": log_negativity(rho, n_qubits, ell),
        "S_cond": S_AB - S_B,
    }


PROFILE_KEYS = ("S_A", "S2_A", "I", "I2", "E_N", "S_cond")


def mixed_state_metrics(rho: np.ndarray, n_qubits: int, profiles: bool = True) -> dict:
    """
    Global and per-cut quantities. Scalars refer to the half-chain cut ell = N/2;
    with profiles=True, arrays '<key>_profile' hold the value at ell = 1..N-1.
    """
    N = n_qubits
    lam = np.linalg.eigvalsh(rho)
    S_AB = entropy_from_eigenvalues(lam)
    pur = purity(rho)
    out = {
        "trace": float(np.trace(rho).real),
        "min_eig": float(lam.min()),
        "purity": pur,
        "S2_total": float(-np.log(pur)),
        "S_total": S_AB,
    }
    half = cut_quantities(rho, N, N // 2, S_AB, -np.log(pur))
    for k, v in half.items():
        out[k] = v
    if profiles:
        prof = {k: np.zeros(N - 1) for k in PROFILE_KEYS}
        for ell in range(1, N):
            q = half if ell == N // 2 else cut_quantities(rho, N, ell, S_AB, -np.log(pur))
            for k in PROFILE_KEYS:
                prof[k][ell - 1] = q[k]
        for k in PROFILE_KEYS:
            out[k + "_profile"] = prof[k]
    return out
