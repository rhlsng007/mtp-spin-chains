"""
Free-fermion exact solutions used as independent checks of the ED code.

Open TFIM
---------
H = -J sum Z_i Z_{i+1} - h sum X_i maps (Hadamard on every site, then
Jordan-Wigner) onto a quadratic fermion Hamiltonian. Its normal-mode
energies are 2 s_k, where s_k are the singular values of the N x N
bidiagonal matrix

    B_ii = h,   B_{i,i+1} = J.

Hence

    E_0 = -sum_k s_k,
    E_1 - E_0 = 2 s_min                     (one fermion, odd parity),
    gap inside the even sector = 2 (s_1 + s_2)  (two lowest fermions).

In the ordered phase s_min is exponentially small: it is the Majorana
edge zero mode of the dual Kitaev chain.

Open XX chain (XXZ at delta = 0)
--------------------------------
H = J sum (X_i X_{i+1} + Y_i Y_{i+1}) is a hopping model with single-particle
energies eps_k = 4 J cos(pi k / (N + 1)), k = 1..N; the ground state fills
every negative level.
"""

from __future__ import annotations

import numpy as np


def tfim_mode_singular_values(n_sites: int, J: float = 1.0, h: float = 1.0) -> np.ndarray:
    """Singular values s_k of the bidiagonal matrix B (ascending)."""
    B = np.diag(np.full(n_sites, float(h))) + np.diag(np.full(n_sites - 1, float(J)), k=1)
    return np.sort(np.linalg.svd(B, compute_uv=False))


def tfim_ground_energy(n_sites: int, J: float = 1.0, h: float = 1.0) -> float:
    return float(-tfim_mode_singular_values(n_sites, J, h).sum())


def tfim_gap(n_sites: int, J: float = 1.0, h: float = 1.0) -> float:
    """E_1 - E_0 of the full spectrum."""
    return float(2 * tfim_mode_singular_values(n_sites, J, h)[0])


def tfim_even_sector_gap(n_sites: int, J: float = 1.0, h: float = 1.0) -> float:
    """Lowest excitation with the same spin-flip parity as the ground state."""
    s = tfim_mode_singular_values(n_sites, J, h)
    return float(2 * (s[0] + s[1]))


def xx_chain_single_particle(n_sites: int, J: float = 1.0) -> np.ndarray:
    k = np.arange(1, n_sites + 1)
    return 4.0 * J * np.cos(np.pi * k / (n_sites + 1))


def xx_chain_ground_energy(n_sites: int, J: float = 1.0) -> float:
    eps = xx_chain_single_particle(n_sites, J)
    return float(eps[eps < 0].sum())
