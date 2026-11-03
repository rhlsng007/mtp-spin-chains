"""
Classical shadows from random single-qubit Pauli measurements (Phase 6, T6.2).

A snapshot measures every qubit q in a uniformly random basis b_q in {X, Y, Z}
(codes 0, 1, 2): the basis rotation U_b (H for X, H S^dagger for Y, identity
for Z, as in noise.shots) is applied and the qubit is read in Z, giving a bit
s_q and the eigenvalue v_q = 1 - 2 s_q. For a Pauli string P with support S
and letters t_q, the single-snapshot estimator

    o_hat = prod_{q in S} 3 [b_q = t_q] v_q

is unbiased, E[o_hat] = <P>, with E[o_hat^2] = 3^|S|. Hence

    Var(ZZ_hat) = 9 - <Z_i Z_j>^2,      Var(X_hat) = 3 - <X_i>^2

per snapshot, divided by M for the mean of M snapshots (checked in Phase 6).

Sampling. Both samplers draw the qubits in order 0, 1, ..., N-1 from the exact
conditional probabilities, with one uniform number per qubit and the rule
s_q = 1 if u_q >= P(s_q = 0 | earlier outcomes). For equal states and equal
random numbers they therefore return identical snapshots:

* sample_pure: state vectors of any size. The first k qubits come from a
  precomputed tree of all 6^k projected branches; the rest are rotated and
  projected one qubit at a time per snapshot (cost about 2 M 2^(N-k)).
* sample_mixed: density matrices, from the table of all 6^N outcome
  probabilities Tr(rho (x)_q Pi_{c_q}), with c = 2 b + s and
  Pi_{b,s} = U_b^dagger |s><s| U_b, and its marginals over trailing qubits
  (practical up to N = 8: 1.7 million entries).

Readout error flips the sampled bits with probability p01 (0 -> 1) or p10
(1 -> 0) using a separate uniform number per qubit and snapshot.
"""

from __future__ import annotations

import numpy as np

from ml.features import bits as basis_bits

BASES = ("X", "Y", "Z")
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_SDG = np.diag([1, -1j])
ROTATIONS = np.stack([_H, _H @ _SDG, np.eye(2, dtype=complex)])     # U_b, indexed by basis code


# ---------------------------------------------------------------------------
# Random numbers of a shadow: bases, sampling uniforms, readout uniforms
# ---------------------------------------------------------------------------
def draw_stream(rng: np.random.Generator, n_snapshots: int, n_qubits: int) -> tuple:
    """(bases uint8 (M, N), u (M, N), r (M, N)) drawn in a fixed order from rng."""
    bases = rng.integers(0, 3, size=(n_snapshots, n_qubits)).astype(np.uint8)
    u = rng.random((n_snapshots, n_qubits))
    r = rng.random((n_snapshots, n_qubits))
    return bases, u, r


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------
def _decide(p0, p1, u):
    p0 = np.clip(p0, 0.0, None)
    p1 = np.clip(p1, 0.0, None)
    tot = p0 + p1
    prob0 = np.where(tot > 0, p0 / np.where(tot > 0, tot, 1.0), 0.5)
    return (u >= prob0), p0, p1


def _branch_tree(psi: np.ndarray, n_qubits: int, depth: int) -> list:
    """
    Unnormalized projected states after measuring qubits 0..j-1 (j = 0..depth):
    level j has shape (6^j, 2^(N-j)), node index sum_q c_q 6^(j-1-q), c = 2 b + s.
    """
    levels = [psi.reshape(1, -1)]
    for _ in range(depth):
        X = levels[-1]
        n = X.shape[0]
        X = X.reshape(n, 2, -1)
        child = np.empty((n, 6, X.shape[2]), dtype=complex)
        for b in range(3):
            U = ROTATIONS[b]
            child[:, 2 * b] = U[0, 0] * X[:, 0] + U[0, 1] * X[:, 1]
            child[:, 2 * b + 1] = U[1, 0] * X[:, 0] + U[1, 1] * X[:, 1]
        levels.append(child.reshape(n * 6, -1))
    return levels


def sample_pure(psi: np.ndarray, bases: np.ndarray, u: np.ndarray, chunk_elements: int = 2 ** 22) -> np.ndarray:
    """
    Snapshot bits (M, N) uint8 of the pure state psi for bases (M, N) and uniforms (M, N).

    The first k qubits are read off a precomputed tree of all 6^k projected
    branches (3^k 2^N amplitudes, k as large as chunk_elements allows); the
    remaining qubits are rotated and projected snapshot by snapshot, in chunks.
    """
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    M, N = bases.shape
    d = psi.size
    if d != 2 ** N:
        raise ValueError("state size does not match the number of qubits")
    k = 0
    while k < N and 3 ** (k + 1) * d <= chunk_elements:
        k += 1
    tree = _branch_tree(psi, N, k)
    norms = [np.sum(np.abs(t) ** 2, axis=1) for t in tree]
    out = np.empty((M, N), dtype=np.uint8)
    idx = np.zeros(M, dtype=np.int64)
    for q in range(k):                                          # prefix from the tree
        idx = idx * 6 + 2 * bases[:, q].astype(np.int64)
        s, _, _ = _decide(norms[q + 1][idx], norms[q + 1][idx + 1], u[:, q])
        out[:, q] = s
        idx = idx + s
    if k == N:
        return out
    rest = d >> k
    step = max(1, int(chunk_elements // rest))
    for start in range(0, M, step):
        sl = slice(start, min(M, start + step))
        B = bases[sl]
        m = B.shape[0]
        X = tree[k][idx[sl]] / np.sqrt(norms[k][idx[sl]])[:, None]
        for q in range(k, N):
            X = X.reshape(m, 2, -1)
            U = ROTATIONS[B[:, q]]                                 # (m, 2, 2)
            x0, x1 = X[:, 0, :], X[:, 1, :]
            y0 = U[:, 0, 0, None] * x0 + U[:, 0, 1, None] * x1
            y1 = U[:, 1, 0, None] * x0 + U[:, 1, 1, None] * x1
            s, p0, p1 = _decide(np.sum(np.abs(y0) ** 2, axis=1), np.sum(np.abs(y1) ** 2, axis=1), u[sl, q])
            out[sl, q] = s
            keep = np.where(s[:, None], y1, y0)
            norm = np.sqrt(np.where(s, p1, p0))
            X = keep / np.where(norm > 0, norm, 1.0)[:, None]
    return out


def projector_weights() -> np.ndarray:
    """W[c, 2k + l] = Pi_c[l, k], so that Tr(Pi_c rho) = sum_{k,l} W[c, 2k + l] rho[k, l]."""
    W = np.empty((6, 4), dtype=complex)
    for b in range(3):
        U = ROTATIONS[b]
        for s in range(2):
            e = np.zeros(2)
            e[s] = 1.0
            Pi = U.conj().T @ np.outer(e, e) @ U
            W[2 * b + s] = Pi.T.reshape(4)                        # entry (k, l) -> Pi[l, k]
    return W


_W = projector_weights()


def outcome_table(rho: np.ndarray, n_qubits: int) -> np.ndarray:
    """All 6^N probabilities Tr(rho (x)_q Pi_{c_q}), flattened with qubit 0 most significant."""
    N = n_qubits
    T = np.asarray(rho).reshape((2,) * (2 * N))
    perm = [ax for q in range(N) for ax in (q, N + q)]            # interleave (k_q, l_q)
    T = T.transpose(perm).reshape((4,) * N)
    for _ in range(N):
        T = np.tensordot(T, _W, axes=([0], [1]))                  # contract the leading qubit
    return np.real(T).reshape(-1)


def marginal_tables(table: np.ndarray, n_qubits: int) -> list:
    """Tables of the first q + 1 qubits (q = 0..N-1), trailing qubits traced out."""
    tabs = [np.asarray(table)]
    for _ in range(n_qubits - 1):
        t = tabs[0].reshape(-1, 6)
        tabs.insert(0, t[:, 4] + t[:, 5])                        # Z outcomes sum to the identity
    return tabs


def sample_mixed(tables: list, bases: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Snapshot bits (M, N) from the marginal tables of a density matrix."""
    M, N = bases.shape
    out = np.empty((M, N), dtype=np.uint8)
    idx = np.zeros(M, dtype=np.int64)
    for q in range(N):
        idx = idx * 6 + 2 * bases[:, q].astype(np.int64)
        t = tables[q]
        s, _, _ = _decide(t[idx], t[idx + 1], u[:, q])
        out[:, q] = s
        idx = idx + s
    return out


def outcome_probabilities(table: np.ndarray, basis: np.ndarray, n_qubits: int) -> np.ndarray:
    """Distribution over the 2^N outcomes for one basis setting (for the tests)."""
    N = n_qubits
    w = 6 ** (N - 1 - np.arange(N))
    base = int(np.sum(2 * np.asarray(basis, dtype=np.int64) * w))
    off = basis_bits(N).astype(np.int64) @ w
    return table[base + off]


def apply_readout(bits: np.ndarray, r: np.ndarray, p01, p10) -> np.ndarray:
    """Flip bits: 0 -> 1 if r < p01, 1 -> 0 if r < p10 (per qubit p01, p10 allowed)."""
    flip = np.where(bits == 0, r < np.asarray(p01), r < np.asarray(p10))
    return (bits ^ flip).astype(np.uint8)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def shadow_observables(bases: np.ndarray, bits: np.ndarray, a=None, b=None) -> dict:
    """
    Shadow estimates of <Z_i Z_j>, <X_i>, <Z_i> from snapshots (..., M, N).

    With a, b (per qubit, broadcastable to (..., N)) each measured value v is
    replaced by (v - a) / b first (readout correction).
    """
    bases = np.asarray(bases)
    v = 1.0 - 2.0 * np.asarray(bits, dtype=float)
    if a is not None:
        a = np.asarray(a, dtype=float)[..., None, :]
        b = np.asarray(b, dtype=float)[..., None, :]
        v = (v - a) / b
    M, N = v.shape[-2], v.shape[-1]
    Zm = 3.0 * (bases == 2) * v
    Xm = 3.0 * (bases == 0) * v
    zz = np.swapaxes(Zm, -1, -2) @ Zm / M
    zz[..., np.arange(N), np.arange(N)] = 1.0
    return {"zz": zz, "x": Xm.mean(axis=-2), "z": Zm.mean(axis=-2)}


def single_snapshot_variance(obs: dict) -> dict:
    """Exact variance of one snapshot's estimator: 9 - <ZZ>^2, 3 - <X>^2, 3 - <Z>^2."""
    zz = 9.0 - obs["zz"] ** 2
    N = zz.shape[-1]
    zz[..., np.arange(N), np.arange(N)] = 0.0
    return {"zz": zz, "x": 3.0 - obs["x"] ** 2, "z": 3.0 - obs["z"] ** 2}


# ---------------------------------------------------------------------------
# Compact storage
# ---------------------------------------------------------------------------
def pack_bits(bits: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(bits, dtype=np.uint8), axis=-1)


def unpack_bits(packed: np.ndarray, n_qubits: int) -> np.ndarray:
    return np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=-1, count=n_qubits)
