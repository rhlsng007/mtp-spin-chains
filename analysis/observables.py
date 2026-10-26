"""
Observables of pure states (Phase 1, task T1.3).

Diagonal (Z-basis) observables are computed from the probabilities
|psi(z)|^2 and the table of spin values z_i = +/-1, which is much faster than
building operators. General Pauli strings use the same bit-mask action as
models/hamiltonians.py.

All functions take a normalized state vector psi of length 2^N in the
PennyLane ordering (site 0 = most significant bit).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from models.hamiltonians import basis_indices, pauli_masks, popcount_parity


def n_sites_of(psi: np.ndarray) -> int:
    n = int(round(np.log2(len(psi))))
    if 2 ** n != len(psi):
        raise ValueError(f"state length {len(psi)} is not a power of two")
    return n


@lru_cache(maxsize=None)
def z_table(n_sites: int) -> np.ndarray:
    """(2^N, N) array of z_i = 1 - 2 b_i for every basis state (cached)."""
    idx = basis_indices(n_sites)
    bits = (idx[:, None] >> (n_sites - 1 - np.arange(n_sites))[None, :]) & 1
    table = (1 - 2 * bits).astype(np.float64)
    table.setflags(write=False)
    return table


def probabilities(psi: np.ndarray) -> np.ndarray:
    return np.abs(psi) ** 2


# ---------------------------------------------------------------------------
# Z-basis observables
# ---------------------------------------------------------------------------
def expval_z(psi: np.ndarray) -> np.ndarray:
    """<Z_i> for every site."""
    return z_table(n_sites_of(psi)).T @ probabilities(psi)


def zz_correlation_matrix(psi: np.ndarray) -> np.ndarray:
    """C_ij = <Z_i Z_j> (diagonal equals 1)."""
    Z = z_table(n_sites_of(psi))
    return (Z * probabilities(psi)[:, None]).T @ Z


def connected_zz_matrix(psi: np.ndarray) -> np.ndarray:
    """<Z_i Z_j> - <Z_i><Z_j>."""
    z = expval_z(psi)
    return zz_correlation_matrix(psi) - np.outer(z, z)


def magnetization_moments(psi: np.ndarray, staggered: bool = False) -> tuple:
    """
    (<m^2>, <m^4>) of m = (1/N) sum_i s_i Z_i, with s_i = 1 or (-1)^i.

    <m^2> equals (1/N^2) sum_ij s_i s_j <Z_i Z_j>.
    """
    n = n_sites_of(psi)
    signs = (-1.0) ** np.arange(n) if staggered else np.ones(n)
    m = z_table(n) @ signs / n
    p = probabilities(psi)
    return float(p @ m ** 2), float(p @ m ** 4)


def squared_magnetization(psi: np.ndarray) -> float:
    return magnetization_moments(psi)[0]


def staggered_squared_magnetization(psi: np.ndarray) -> float:
    """Neel order parameter for the antiferromagnetic XXZ chain."""
    return magnetization_moments(psi, staggered=True)[0]


def binder_cumulant(m2: float, m4: float) -> float:
    """
    U_4 = 1 - <m^4> / (3 <m^2>^2): 2/3 when ordered, -> 0 when disordered.

    Returns NaN when <m^2> vanishes (e.g. uniform m in the XXZ ground state,
    which has exactly zero total magnetization by U(1) symmetry).
    """
    if m2 < 1e-14:
        return float("nan")
    return 1.0 - m4 / (3.0 * m2 ** 2)


def centred_correlator(corr_matrix: np.ndarray) -> np.ndarray:
    """
    C(r) for r = 1 .. N-1, with each pair placed symmetrically about the
    chain centre (sites i = (N - r) // 2 and i + r) to reduce edge effects.
    """
    n = corr_matrix.shape[0]
    out = np.empty(n - 1)
    for r in range(1, n):
        i = (n - r) // 2
        out[r - 1] = corr_matrix[i, i + r]
    return out


# ---------------------------------------------------------------------------
# General Pauli strings
# ---------------------------------------------------------------------------
def pauli_expectation(psi: np.ndarray, ops: dict) -> float:
    """
    <psi| P |psi> for a Pauli string, e.g. pauli_expectation(psi, {0: 'X', 1: 'X'}).
    """
    n = n_sites_of(psi)
    idx = basis_indices(n)
    flip, sign, n_y = pauli_masks(n, ops)
    phase = (1j ** n_y) * (1.0 - 2.0 * popcount_parity(idx & sign, n))
    value = np.sum(np.conj(psi[idx ^ flip]) * phase * psi)
    return float(np.real(value))


def string_order_sites(n_sites: int) -> tuple:
    """
    Default end points (a, b) of the cluster string operator.

    b - a must be even; we take the longest such string in the chain. It is
    the product of the bulk stabilizers Z X Z centred on a+1, a+3, ..., b-1,
    so it equals +1 in every open-chain cluster ground state at h1 = h2 = 0.
    """
    length = n_sites - 2 if n_sites % 2 == 0 else n_sites - 1
    a = (n_sites - 1 - length) // 2
    return a, a + length


def string_order_parameter(psi: np.ndarray, a: int = None, b: int = None) -> float:
    """
    S = < Z_a X_{a+1} X_{a+3} ... X_{b-1} Z_b >   (cluster SPT order).
    """
    n = n_sites_of(psi)
    if a is None or b is None:
        a, b = string_order_sites(n)
    if (b - a) % 2 or b <= a:
        raise ValueError("string order needs b > a with b - a even")
    ops = {a: "Z", b: "Z"}
    ops.update({k: "X" for k in range(a + 1, b, 2)})
    return pauli_expectation(psi, ops)
