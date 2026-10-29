"""
Exact state-vector kernels for the Phase 2 ansätze.

Why not call PennyLane inside the optimizer?
--------------------------------------------
The noiseless sweeps need on the order of 10^5 to 10^6 energy-and-gradient
evaluations (39 fields x up to 12 depths x 5 sizes x 2 ansätze x several
initializations x hundreds of optimizer steps). PennyLane's Python-level
overhead per circuit execution dominates at N <= 12, so the optimizer uses
the vectorised NumPy kernels below instead. They implement exactly the same
gates and conventions as PennyLane:

    RX(phi) = exp(-i phi X / 2),  RY(phi) = exp(-i phi Y / 2),
    RZ(phi) = exp(-i phi Z / 2),  IsingZZ(phi) = exp(-i phi Z Z / 2),

with wire 0 as the most significant bit (the Phase 1 convention), and
tests/test_phase2.py checks states, energies and gradients against
PennyLane (backprop and the parameter-shift rule) gate for gate.

All functions act on the last axis, so a stacked array of shape (2, 2^N)
(the forward state and the adjoint state of the adjoint-gradient method)
is updated in one call.

Single-qubit gates reshape the state to (..., 2^q, 2, 2^(N-q-1)) and mix
the two slices of the qubit axis, which avoids copying through index arrays.
Diagonal gates (RZ layers, the ZZ layer) are elementwise phases built from
the sign table s_q(z) = 1 - 2 b_q(z), and the CNOT chain is one precomputed
permutation of basis indices.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from models.hamiltonians import basis_indices, site_bit


# ---------------------------------------------------------------------------
# Cached lookup tables
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def flip_index(n_qubits: int, qubit: int) -> np.ndarray:
    """Index array z XOR m_q: fancy-indexing with it applies X_q."""
    out = basis_indices(n_qubits) ^ site_bit(n_qubits, qubit)
    out.setflags(write=False)
    return out


@lru_cache(maxsize=None)
def z_signs(n_qubits: int) -> np.ndarray:
    """(2^N, N) table of s_q(z) = +1 / -1 for every basis state and qubit."""
    idx = basis_indices(n_qubits)
    bits = (idx[:, None] >> (n_qubits - 1 - np.arange(n_qubits))[None, :]) & 1
    table = (1 - 2 * bits).astype(np.float64)
    table.setflags(write=False)
    return table


@lru_cache(maxsize=None)
def zz_diagonal(n_qubits: int, boundary: str = "open") -> np.ndarray:
    """Diagonal of sum_i Z_i Z_{i+1} (open: N-1 bonds, periodic: N bonds)."""
    s = z_signs(n_qubits)
    n_bonds = n_qubits - 1 if boundary == "open" else n_qubits
    diag = sum(s[:, i] * s[:, (i + 1) % n_qubits] for i in range(n_bonds))
    diag = np.asarray(diag, dtype=np.float64)
    diag.setflags(write=False)
    return diag


@lru_cache(maxsize=None)
def cnot_chain_permutation(n_qubits: int) -> tuple:
    """
    Index arrays (perm, inv) of the CNOT chain C(N-2,N-1) ... C(1,2) C(0,1),
    with C(0,1) applied first.

    A permutation unitary U|z> = |f(z)> acts as (U psi)(y) = psi(f^-1(y)),
    so U psi = psi[perm] and U^dagger psi = psi[inv].
    """
    idx = basis_indices(n_qubits)
    perm = idx.copy()
    for c in range(n_qubits - 1):
        mc, mt = site_bit(n_qubits, c), site_bit(n_qubits, c + 1)
        g = idx ^ np.where(idx & mc, mt, 0)
        perm = perm[g]
    inv = np.argsort(perm)
    perm.setflags(write=False)
    inv.setflags(write=False)
    return perm, inv


def integer_diagonal_phase(diag: np.ndarray, angle: float) -> np.ndarray:
    """
    exp(-i angle diag) for a diagonal with integer entries.

    Only the distinct values are exponentiated and then gathered, which is
    several times faster than np.exp over all 2^N entries.
    """
    lo = int(diag.min())
    values = np.arange(lo, int(diag.max()) + 1)
    return np.exp(-1j * angle * values)[diag.astype(np.int64) - lo]


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
def zero_state(n_qubits: int) -> np.ndarray:
    psi = np.zeros(2 ** n_qubits, dtype=np.complex128)
    psi[0] = 1.0
    return psi


def plus_state(n_qubits: int) -> np.ndarray:
    return np.full(2 ** n_qubits, 2.0 ** (-n_qubits / 2), dtype=np.complex128)


# ---------------------------------------------------------------------------
# Gates (all act on the last axis; return new arrays)
# ---------------------------------------------------------------------------
def _split(psi: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """View of psi with shape (..., 2^q, 2, 2^(N-q-1)); axis -2 is the qubit."""
    return psi.reshape(psi.shape[:-1] + (2 ** qubit, 2, 2 ** (n_qubits - qubit - 1)))


def _apply_1q(psi: np.ndarray, qubit: int, n_qubits: int, u00, u01, u10, u11) -> np.ndarray:
    """Apply the 2x2 matrix [[u00, u01], [u10, u11]] to one qubit."""
    v = _split(psi, qubit, n_qubits)
    a0, a1 = v[..., 0, :], v[..., 1, :]
    out = np.empty(v.shape, dtype=np.result_type(psi, u00, u01, u10, u11))
    out[..., 0, :] = u00 * a0 + u01 * a1
    out[..., 1, :] = u10 * a0 + u11 * a1
    return out.reshape(psi.shape)


def _kron2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Kronecker product of two matrices (faster than np.kron for small inputs)."""
    return (a[:, None, :, None] * b[None, :, None, :]).reshape(
        a.shape[0] * b.shape[0], a.shape[1] * b.shape[1])


def kron_all(mats) -> np.ndarray:
    """mats[0] (x) mats[1] (x) ..., combined pairwise (balanced tree)."""
    mats = list(mats)
    if not mats:
        return np.ones((1, 1))
    while len(mats) > 1:
        mats = [_kron2(mats[i], mats[i + 1]) if i + 1 < len(mats) else mats[i]
                for i in range(0, len(mats), 2)]
    return mats[0]


def _apply_halves(psi: np.ndarray, a: np.ndarray, b: np.ndarray, n_qubits: int) -> np.ndarray:
    """(A (x) B) psi = A Psi B^T, with Psi = psi reshaped to (2^k, 2^(N-k))."""
    k = n_qubits // 2
    shape = psi.shape
    m = psi.reshape(shape[:-1] + (2 ** k, 2 ** (n_qubits - k)))
    return (a @ m @ b.T).reshape(shape)


@lru_cache(maxsize=None)
def _hadamard_half(n: int) -> np.ndarray:
    h = kron_all([np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)] * n)
    h.setflags(write=False)
    return h


@lru_cache(maxsize=None)
def _sum_z_half(n: int) -> np.ndarray:
    """Diagonal of sum_q Z_q on n qubits."""
    idx = np.arange(2 ** n)
    pop = sum((idx >> k) & 1 for k in range(n))
    out = (n - 2 * pop).astype(np.float64)
    out.setflags(write=False)
    return out


def apply_product(psi: np.ndarray, mats: list, n_qubits: int) -> np.ndarray:
    """
    Apply the tensor product mats[0] (x) mats[1] (x) ... (x) mats[N-1].

    The product splits as A (x) B over the first k = N // 2 qubits and the
    rest, and with Psi = psi reshaped to (2^k, 2^(N-k)),

        (A (x) B) psi  =  A Psi B^T,

    i.e. two small matrix products instead of N passes over the state.
    """
    k = n_qubits // 2
    return _apply_halves(psi, kron_all(mats[:k]), kron_all(mats[k:]), n_qubits)


def rx_matrix(phi: float) -> np.ndarray:
    c, s = np.cos(phi / 2), np.sin(phi / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def ry_matrix(phi: float) -> np.ndarray:
    c, s = np.cos(phi / 2), np.sin(phi / 2)
    return np.array([[c, -s], [s, c]])


def hadamard_transform(psi: np.ndarray, n_qubits: int) -> np.ndarray:
    """H^(x)N psi (real, self-inverse)."""
    k = n_qubits // 2
    return _apply_halves(psi, _hadamard_half(k), _hadamard_half(n_qubits - k), n_qubits)


def _uniform_rx_half(phi: float, n: int) -> np.ndarray:
    """RX(phi)^(x)n = H^(x)n diag(exp(-i phi/2 sum Z)) H^(x)n."""
    h = _hadamard_half(n)
    return (h * np.exp(-0.5j * phi * _sum_z_half(n))) @ h


@lru_cache(maxsize=None)
def sum_z_diagonal(n_qubits: int) -> np.ndarray:
    """Diagonal of sum_q Z_q (= N - 2 * popcount)."""
    diag = z_signs(n_qubits).sum(axis=1)
    diag.setflags(write=False)
    return diag


def apply_x(psi: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    return psi[..., flip_index(n_qubits, qubit)]


def apply_rx(psi: np.ndarray, qubit: int, phi: float, n_qubits: int) -> np.ndarray:
    """RX(phi) = [[c, -i s], [-i s, c]] with c = cos(phi/2), s = sin(phi/2)."""
    c, s = np.cos(phi / 2), np.sin(phi / 2)
    return _apply_1q(psi, qubit, n_qubits, c, -1j * s, -1j * s, c)


def apply_rx_all(psi: np.ndarray, phi: float, n_qubits: int) -> np.ndarray:
    """prod_q RX_q(phi) = exp(-i (phi/2) sum_q X_q)."""
    k = n_qubits // 2
    return _apply_halves(psi, _uniform_rx_half(phi, k),
                         _uniform_rx_half(phi, n_qubits - k), n_qubits)


def apply_ry_layer(psi: np.ndarray, phis, n_qubits: int) -> np.ndarray:
    """prod_q RY_q(phi_q)."""
    return apply_product(psi, [ry_matrix(p) for p in phis], n_qubits)


def apply_ry(psi: np.ndarray, qubit: int, phi: float, n_qubits: int) -> np.ndarray:
    """RY(phi) = [[c, -s], [s, c]]."""
    c, s = np.cos(phi / 2), np.sin(phi / 2)
    return _apply_1q(psi, qubit, n_qubits, c, -s, s, c)


def rz_layer_phase(phis: np.ndarray, n_qubits: int) -> np.ndarray:
    """Diagonal of prod_q RZ_q(phi_q) = exp(-(i/2) sum_q phi_q s_q(z))."""
    return np.exp(-0.5j * (z_signs(n_qubits) @ np.asarray(phis, dtype=np.float64)))


def apply_cnot_chain(psi: np.ndarray, n_qubits: int, inverse: bool = False) -> np.ndarray:
    perm, inv = cnot_chain_permutation(n_qubits)
    return psi[..., inv if inverse else perm]


# ---------------------------------------------------------------------------
# Matrix elements for the adjoint method
# ---------------------------------------------------------------------------
def braket(a: np.ndarray, b: np.ndarray) -> complex:
    """<a|b>."""
    return complex(np.vdot(a, b))


def _halves(lam: np.ndarray, phi: np.ndarray, qubit: int, n_qubits: int) -> tuple:
    """(<lam_0|phi_1>, <lam_1|phi_0>) for the two values of the qubit."""
    vl, vp = _split(lam, qubit, n_qubits), _split(phi, qubit, n_qubits)
    return np.vdot(vl[:, 0, :], vp[:, 1, :]), np.vdot(vl[:, 1, :], vp[:, 0, :])


def braket_sum_x(lam: np.ndarray, phi: np.ndarray, n_qubits: int) -> complex:
    """<lam| sum_q X_q |phi> = <H lam| sum_q Z_q |H phi> (Hadamard basis)."""
    hl = hadamard_transform(lam, n_qubits)
    hp = hadamard_transform(phi, n_qubits)
    return braket(hl, sum_z_diagonal(n_qubits) * hp)


def braket_y(lam: np.ndarray, phi: np.ndarray, qubit: int, n_qubits: int) -> complex:
    """<lam| Y_q |phi>, with Y = [[0, -i], [i, 0]]."""
    a, b = _halves(lam, phi, qubit, n_qubits)
    return complex(-1j * (a - b))


def braket_z_all(lam: np.ndarray, phi: np.ndarray, n_qubits: int) -> np.ndarray:
    """Vector of <lam| Z_q |phi> for every qubit q."""
    return z_signs(n_qubits).T @ (np.conj(lam) * phi)
