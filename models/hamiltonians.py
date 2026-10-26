"""
Sparse Hamiltonians for 1D spin chains (Phase 1, task T1.1).

Qubit-ordering convention (identical to PennyLane)
--------------------------------------------------
Site / wire 0 is the MOST significant bit of the basis-state index:

    |z_0 z_1 ... z_{N-1}>   <->   index = sum_i z_i * 2**(N-1-i)

so an exact state vector from this module can be compared entry by entry
with qml.state() from a PennyLane circuit on wires 0..N-1. Qiskit uses the
opposite (little-endian) ordering; convert before comparing in task T4.7.

Representation
--------------
A Hamiltonian is a list of terms (coefficient, {site: "X" | "Y" | "Z"}).
A Pauli string P maps a basis state to a single basis state:

    P |z> = phase(z) |z XOR flip_mask>

where flip_mask marks the X and Y sites, and Y = i X Z gives

    phase(z) = i^(n_Y) * (-1)^popcount(z AND (z_mask OR y_mask))

The sparse matrix is therefore built with vectorised bit operations rather
than Kronecker products: it is exact and takes milliseconds at N = 14.
The same term list can be converted to PennyLane observables in Phase 2
via `terms_to_pauli_words`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.sparse as sp

VALID_BOUNDARIES = ("open", "periodic")


# ---------------------------------------------------------------------------
# Bit-level helpers
# ---------------------------------------------------------------------------
def site_bit(n_sites: int, site: int) -> int:
    """Integer bit mask of `site` (site 0 = most significant bit)."""
    return 1 << (n_sites - 1 - site)


@lru_cache(maxsize=None)
def basis_indices(n_sites: int) -> np.ndarray:
    """All basis-state indices 0 .. 2^N - 1 (cached, read-only)."""
    idx = np.arange(2 ** n_sites, dtype=np.int64)
    idx.setflags(write=False)
    return idx


def popcount_parity(x: np.ndarray, n_bits: int) -> np.ndarray:
    """Parity (0 or 1) of the number of set bits in each entry of x."""
    x = np.asarray(x, dtype=np.int64)
    parity = np.zeros_like(x)
    for k in range(n_bits):
        parity ^= (x >> k) & 1
    return parity


def pauli_masks(n_sites: int, ops: dict) -> tuple:
    """
    Bit masks of a Pauli string.

    Returns (flip_mask, sign_mask, n_y): flip_mask marks X and Y sites,
    sign_mask marks Z and Y sites, n_y is the number of Y operators.
    """
    flip = sign = 0
    n_y = 0
    for site, op in ops.items():
        if not 0 <= site < n_sites:
            raise ValueError(f"site {site} outside chain of length {n_sites}")
        b = site_bit(n_sites, site)
        op = op.upper()
        if op == "X":
            flip |= b
        elif op == "Y":
            flip |= b
            sign |= b
            n_y += 1
        elif op == "Z":
            sign |= b
        elif op != "I":
            raise ValueError(f"unknown Pauli label '{op}'")
    return flip, sign, n_y


# ---------------------------------------------------------------------------
# Sparse matrices from Pauli terms
# ---------------------------------------------------------------------------
def terms_to_sparse(n_sites: int, terms: list) -> sp.csr_matrix:
    """
    Sum of Pauli strings as a CSR matrix.

    The matrix is real (float64) whenever every prefactor coeff * i^n_Y is
    real, which holds for all models in this thesis; otherwise complex128.
    """
    idx = basis_indices(n_sites)
    rows, cols, vals = [], [], []
    is_real = True
    for coeff, ops in terms:
        if coeff == 0:
            continue
        flip, sign, n_y = pauli_masks(n_sites, ops)
        prefactor = complex(coeff) * (1j ** n_y)
        if abs(prefactor.imag) > 1e-15:
            is_real = False
        signs = 1.0 - 2.0 * popcount_parity(idx & sign, n_sites)
        rows.append(idx ^ flip)
        cols.append(idx)
        vals.append(prefactor * signs)

    dim = 2 ** n_sites
    if not rows:
        return sp.csr_matrix((dim, dim), dtype=np.float64)
    data = np.concatenate(vals)
    data = data.real if is_real else data
    mat = sp.coo_matrix((data, (np.concatenate(rows), np.concatenate(cols))),
                        shape=(dim, dim))
    mat = mat.tocsr()
    mat.sum_duplicates()
    mat.eliminate_zeros()
    return mat


def pauli_string_sparse(n_sites: int, ops: dict, coeff: complex = 1.0) -> sp.csr_matrix:
    """Single Pauli string, e.g. pauli_string_sparse(4, {0: 'Z', 3: 'X'})."""
    return terms_to_sparse(n_sites, [(coeff, ops)])


def terms_to_pauli_words(n_sites: int, terms: list) -> list:
    """
    Convert terms to (coeff, word) pairs such as (-1.0, 'ZZII').

    Useful for building qml.Hamiltonian / qml.dot objects in Phase 2.
    """
    words = []
    for coeff, ops in terms:
        letters = ["I"] * n_sites
        for site, op in ops.items():
            letters[site] = op.upper()
        words.append((coeff, "".join(letters)))
    return words


# ---------------------------------------------------------------------------
# Lattice bonds
# ---------------------------------------------------------------------------
def bond_starts(n_sites: int, boundary: str, span: int = 1) -> list:
    """
    Starting sites i of the terms acting on i, i+1, ..., i+span.

    Open chains have N - span such terms; periodic chains have N
    (indices taken modulo N). Periodic chains require N > span + 1 so that
    no term wraps onto itself or duplicates another.
    """
    if boundary not in VALID_BOUNDARIES:
        raise ValueError(f"boundary must be one of {VALID_BOUNDARIES}, got '{boundary}'")
    if boundary == "open":
        return list(range(max(n_sites - span, 0)))
    if n_sites <= span + 1:
        raise ValueError(f"periodic chain needs N > {span + 1}, got N = {n_sites}")
    return list(range(n_sites))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def tfim_terms(n_sites: int, J: float = 1.0, h: float = 1.0, boundary: str = "open") -> list:
    """
    Transverse-field Ising model

        H = -J sum_i Z_i Z_{i+1} - h sum_i X_i
    """
    terms = [(-J, {i: "Z", (i + 1) % n_sites: "Z"}) for i in bond_starts(n_sites, boundary, 1)]
    terms += [(-h, {i: "X"}) for i in range(n_sites)]
    return terms


def xxz_terms(n_sites: int, J: float = 1.0, delta: float = 1.0, boundary: str = "open") -> list:
    """
    Antiferromagnetic XXZ chain

        H = J sum_i (X_i X_{i+1} + Y_i Y_{i+1} + delta Z_i Z_{i+1})
    """
    terms = []
    for i in bond_starts(n_sites, boundary, 1):
        j = (i + 1) % n_sites
        terms += [(J, {i: "X", j: "X"}), (J, {i: "Y", j: "Y"}), (J * delta, {i: "Z", j: "Z"})]
    return terms


def cluster_ising_terms(n_sites: int, J: float = 1.0, h1: float = 0.0, h2: float = 0.0,
                        boundary: str = "open") -> list:
    """
    Cluster-Ising model (Cong, Choi & Lukin, Nat. Phys. 15, 1273 (2019))

        H = -J sum_i Z_i X_{i+1} Z_{i+2} - h1 sum_i X_i - h2 sum_i X_i X_{i+1}
    """
    N = n_sites
    terms = [(-J, {i: "Z", (i + 1) % N: "X", (i + 2) % N: "Z"})
             for i in bond_starts(N, boundary, 2)]
    terms += [(-h1, {i: "X"}) for i in range(N)]
    terms += [(-h2, {i: "X", (i + 1) % N: "X"}) for i in bond_starts(N, boundary, 1)]
    return terms


MODEL_TERMS = {"tfim": tfim_terms, "xxz": xxz_terms, "cluster": cluster_ising_terms}


def hamiltonian_terms(model: str, n_sites: int, **params) -> list:
    """Pauli-term list of a model, e.g. hamiltonian_terms('tfim', 8, J=1, h=0.5)."""
    if model not in MODEL_TERMS:
        raise ValueError(f"unknown model '{model}', choose from {list(MODEL_TERMS)}")
    return MODEL_TERMS[model](n_sites, **params)


def build_hamiltonian(model: str, n_sites: int, **params) -> sp.csr_matrix:
    """Sparse Hamiltonian, e.g. build_hamiltonian('xxz', 10, J=1, delta=0.5)."""
    return terms_to_sparse(n_sites, hamiltonian_terms(model, n_sites, **params))


# ---------------------------------------------------------------------------
# Spin-flip symmetries
# ---------------------------------------------------------------------------
def symmetry_generators(model: str, n_sites: int, boundary: str = "open") -> tuple:
    """
    Bit masks of the Z2 spin-flip symmetries used to split the Hilbert space.

    Each generator is a product of X over a set of sites, which acts on a
    basis index as  index -> index XOR mask.

    tfim, xxz : global flip  P = prod_i X_i   (commutes with ZZ, XX, YY, X)
    cluster   : Z2 x Z2 sublattice flips  P_even = prod_{i even} X_i  and
                P_odd = prod_{i odd} X_i. Each Z X Z term has its two Z's on
                the same sublattice, so both flips commute with H. For a
                periodic chain with odd N this fails, and only the global
                flip P_even * P_odd is used.
    """
    all_sites = (1 << n_sites) - 1
    if model in ("tfim", "xxz"):
        return (all_sites,)
    if model == "cluster":
        if boundary == "periodic" and n_sites % 2 == 1:
            return (all_sites,)
        even = sum(site_bit(n_sites, i) for i in range(0, n_sites, 2))
        odd = sum(site_bit(n_sites, i) for i in range(1, n_sites, 2))
        return (even, odd)
    raise ValueError(f"unknown model '{model}'")


def apply_spin_flip(psi: np.ndarray, mask: int) -> np.ndarray:
    """(prod_{i in mask} X_i) |psi>. For the global flip this reverses psi."""
    idx = basis_indices(int(np.log2(len(psi))))
    return psi[idx ^ mask]
