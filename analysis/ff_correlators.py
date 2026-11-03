"""
Exact ground-state correlators of the open transverse-field Ising chain from
free fermions (Phase 6 validation of the ML features).

    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i        (open chain, N sites)

A Hadamard on every site maps H to H' = -J sum X_i X_{i+1} - h sum Z_i, so
<Z_i Z_j>_H = <X_i X_j>_H' and <X_i>_H = <Z_i>_H'. With the Jordan-Wigner
Majorana operators (0-based, S_q = prod_{p<q} Z_p)

    a_{2q} = S_q X_q,   a_{2q+1} = S_q Y_q,

one has Z_q = -i a_{2q} a_{2q+1} and X_q X_{q+1} = -i a_{2q+1} a_{2q+2}, so
H' = (i/4) a^T M a with a real antisymmetric 2N x 2N matrix M. The ground
state has the Majorana covariance Gamma_kl = i <a_k a_l> (k != l),

    Gamma = -U V^T        (M = U S V^T, the polar factor of M),

and Wick's theorem gives

    <X_i>      = -Gamma[2i, 2i+1]
    <Z_i Z_j>  = (-1)^(j-i) Pf( Gamma[2i+1 .. 2j, 2i+1 .. 2j] ),   i < j.

The free-fermion vacuum is the global ground state, which for h > 0 is the
even-parity state used as the ED reference throughout the project. When the
parity splitting is below machine precision (deep ordered phase, large N) the
vacuum parity is numerically arbitrary; the even and odd states then differ
in these correlators only at that same negligible level.
"""

from __future__ import annotations

import numpy as np


def pfaffian(A: np.ndarray) -> float:
    """Pfaffian of a real antisymmetric matrix (Parlett-Reid elimination with pivoting)."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    if n % 2:
        return 0.0
    pf = 1.0
    for k in range(0, n - 1, 2):
        kp = k + 1 + int(np.argmax(np.abs(A[k, k + 1:])))
        if kp != k + 1:                                   # swap rows and columns k+1 and kp
            A[[k + 1, kp], :] = A[[kp, k + 1], :]
            A[:, [k + 1, kp]] = A[:, [kp, k + 1]]
            pf = -pf
        if A[k, k + 1] == 0.0:
            return 0.0
        pf *= A[k, k + 1]
        if k + 2 < n:
            tau = A[k, k + 2:] / A[k, k + 1]
            col = A[k + 2:, k + 1].copy()
            A[k + 2:, k + 2:] += np.outer(tau, col) - np.outer(col, tau)
    return float(pf)


def majorana_matrix(n_qubits: int, J: float, h: float) -> np.ndarray:
    """M with H' = (i/4) a^T M a; a term i c a_k a_l (k < l) gives M_kl = 2c = -M_lk."""
    N = n_qubits
    M = np.zeros((2 * N, 2 * N))
    for q in range(N):                         # -h Z_q = i h a_{2q} a_{2q+1}
        M[2 * q, 2 * q + 1] += 2.0 * h
    for q in range(N - 1):                     # -J X_q X_{q+1} = i J a_{2q+1} a_{2q+2}
        M[2 * q + 1, 2 * q + 2] += 2.0 * J
    return M - M.T


def ground_covariance(n_qubits: int, J: float, h: float) -> np.ndarray:
    U, _, Vt = np.linalg.svd(majorana_matrix(n_qubits, J, h))
    return -(U @ Vt)


def ff_energy(n_qubits: int, J: float, h: float) -> float:
    """Ground energy (1/4) sum_kl M_kl Gamma_kl = -(1/2) sum of the singular values pairs."""
    M = majorana_matrix(n_qubits, J, h)
    return float(0.25 * np.sum(M * ground_covariance(n_qubits, J, h)))


def ff_observables(n_qubits: int, J: float, h: float) -> dict:
    """Exact <Z_i Z_j> (N x N, diagonal 1), <X_i> and <Z_i> (= 0) of the even ground state."""
    N = n_qubits
    G = ground_covariance(N, J, h)
    x = np.array([-G[2 * q, 2 * q + 1] for q in range(N)])
    zz = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            idx = np.arange(2 * i + 1, 2 * j + 1)
            val = (-1) ** (j - i) * pfaffian(G[np.ix_(idx, idx)])
            zz[i, j] = zz[j, i] = val
    return {"zz": zz, "x": x, "z": np.zeros(N)}
