"""
Observables and feature vectors for phase classification (Phase 6, T6.2).

Features are Z2-invariant, so the even-parity states (ED, HVA) and the
symmetry-broken HEA states of the ordered phase are described alike:

    "corr"    : <Z_i Z_j> for all i < j, then <X_i> for all i    (N(N-1)/2 + N values)
    "zz"      : <Z_i Z_j> for all i < j only
    "compact" : M^2, mean <X_i>, mean nearest-neighbour <Z_i Z_{i+1}>, end-to-end <Z_0 Z_{N-1}>

The raw (not connected) ZZ correlator is used on purpose: a symmetry-broken
state has <Z_i> != 0 and a near-zero connected correlator in the ordered
phase, so connected correlators would place it in the disordered phase.
<Z_i> is stored beside the features for diagnostics only.

Readout error acts on each measured bit independently. A qubit with true
eigenvalue v = +-1 is read with mean a + b v, where for a column-stochastic
2 x 2 matrix C[outcome, state]

    a = C[0, 1] - C[1, 0],      b = 1 - C[1, 0] - C[0, 1],

so a product of readout values on distinct qubits has mean prod_q (a_q + b_q O_q)
and the per-qubit correction v -> (v - a) / b is unbiased (the tensored linear
inverse of Phase 5).
"""

from __future__ import annotations

import numpy as np

FEATURE_SETS = ("corr", "zz", "compact")


# ---------------------------------------------------------------------------
# Exact observables of pure and mixed states
# ---------------------------------------------------------------------------
def bits(n_qubits: int) -> np.ndarray:
    """bits[b, w] = value of wire w in basis index b (wire 0 = most significant)."""
    b = np.arange(2 ** n_qubits)[:, None]
    return ((b >> (n_qubits - 1 - np.arange(n_qubits))[None, :]) & 1).astype(np.uint8)


def z_signs(n_qubits: int) -> np.ndarray:
    return 1.0 - 2.0 * bits(n_qubits)


def _flip_index(n_qubits: int, q: int) -> np.ndarray:
    return np.arange(2 ** n_qubits) ^ (1 << (n_qubits - 1 - q))


def _from_diagonal(p: np.ndarray, n_qubits: int) -> tuple:
    S = z_signs(n_qubits)
    z = p @ S
    zz = S.T @ (p[:, None] * S)
    return zz, z


def pure_observables(psi: np.ndarray, n_qubits: int) -> dict:
    """<Z_i Z_j>, <X_i>, <Z_i> of a state vector (wire 0 = most significant bit)."""
    psi = np.asarray(psi).reshape(-1)
    zz, z = _from_diagonal(np.abs(psi) ** 2, n_qubits)
    x = np.array([np.real(np.vdot(psi, psi[_flip_index(n_qubits, q)])) for q in range(n_qubits)])
    return {"zz": zz, "x": x, "z": z}


def mixed_observables(rho: np.ndarray, n_qubits: int) -> dict:
    """Same for a density matrix: Tr(rho X_q) = sum_b rho[b, b xor e_q]."""
    d = 2 ** n_qubits
    idx = np.arange(d)
    zz, z = _from_diagonal(np.real(np.diagonal(rho)), n_qubits)
    x = np.array([np.real(rho[idx, _flip_index(n_qubits, q)].sum()) for q in range(n_qubits)])
    return {"zz": zz, "x": x, "z": z}


# ---------------------------------------------------------------------------
# Readout error on expectation values
# ---------------------------------------------------------------------------
def readout_affine(mats) -> tuple:
    """(a, b) per qubit from column-stochastic matrices of shape (..., N, 2, 2)."""
    m = np.asarray(mats, dtype=float)
    c10, c01 = m[..., 1, 0], m[..., 0, 1]
    return c01 - c10, 1.0 - c10 - c01


def affine_observables(obs: dict, a, b) -> dict:
    """Expectations of per-qubit affine maps v -> a_q + b_q v applied to every measured value."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    zz, x, z = obs["zz"], obs["x"], obs["z"]
    bz = b * z
    zz_m = np.outer(a, a) + np.outer(a, bz) + np.outer(bz, a) + np.outer(b, b) * zz
    np.fill_diagonal(zz_m, 1.0)
    return {"zz": zz_m, "x": a + b * x, "z": a + b * z}


def readout_biased(obs: dict, readout_mats) -> dict:
    """Infinite-shot expectations measured through the readout matrices R_q."""
    a, b = readout_affine(readout_mats)
    return affine_observables(obs, a, b)


def readout_corrected(obs: dict, readout_mats, calibration_mats) -> dict:
    """Infinite-shot expectations after the per-qubit correction with calibration C_q."""
    aR, bR = readout_affine(readout_mats)
    aC, bC = readout_affine(calibration_mats)
    return affine_observables(obs, (aR - aC) / bC, bR / bC)


# ---------------------------------------------------------------------------
# Feature vectors
# ---------------------------------------------------------------------------
def n_features(n_qubits: int) -> int:
    return n_qubits * (n_qubits - 1) // 2 + n_qubits


def feature_names(n_qubits: int) -> np.ndarray:
    iu = np.triu_indices(n_qubits, k=1)
    names = [f"ZZ_{i}_{j}" for i, j in zip(*iu)] + [f"X_{q}" for q in range(n_qubits)]
    return np.array(names)


def corr_features(zz, x) -> np.ndarray:
    """Default feature vector: upper triangle of <Z_i Z_j> (row-major), then <X_i>."""
    zz, x = np.asarray(zz), np.asarray(x)
    N = x.shape[-1]
    iu = np.triu_indices(N, k=1)
    return np.concatenate([zz[..., iu[0], iu[1]], x], axis=-1)


def unpack_features(X, n_qubits: int) -> tuple:
    """Inverse of corr_features: (zz with unit diagonal, x)."""
    X = np.asarray(X)
    N = n_qubits
    iu = np.triu_indices(N, k=1)
    k = iu[0].size
    zz = np.zeros(X.shape[:-1] + (N, N))
    zz[..., iu[0], iu[1]] = X[..., :k]
    zz[..., iu[1], iu[0]] = X[..., :k]
    zz[..., np.arange(N), np.arange(N)] = 1.0
    return zz, X[..., k:k + N]


def magnetization_m2(zz) -> np.ndarray:
    zz = np.asarray(zz)
    N = zz.shape[-1]
    return zz.sum(axis=(-1, -2)) / N ** 2


def compact_features(X, n_qubits: int) -> np.ndarray:
    zz, x = unpack_features(X, n_qubits)
    N = n_qubits
    nn = zz[..., np.arange(N - 1), np.arange(1, N)].mean(axis=-1)
    return np.stack([magnetization_m2(zz), x.mean(axis=-1), nn, zz[..., 0, N - 1]], axis=-1)


def compact_names() -> np.ndarray:
    return np.array(["M2", "X_mean", "ZZ_nn_mean", "ZZ_end"])


def select_features(X, n_qubits: int, kind: str = "corr") -> np.ndarray:
    """Feature matrix of the requested set from stored "corr" features (last axis)."""
    X = np.asarray(X)
    if kind == "corr":
        return X
    if kind == "zz":
        return X[..., :n_qubits * (n_qubits - 1) // 2]
    if kind == "compact":
        return compact_features(X, n_qubits)
    raise ValueError(f"unknown feature set {kind!r}; expected one of {FEATURE_SETS}")


def energy_from_features(X, n_qubits: int, h, J: float = 1.0) -> np.ndarray:
    """Open-chain TFIM energy -J sum <Z_i Z_{i+1}> - h sum <X_i> (exact features only)."""
    zz, x = unpack_features(X, n_qubits)
    N = n_qubits
    nn = zz[..., np.arange(N - 1), np.arange(1, N)].sum(axis=-1)
    return -J * nn - np.asarray(h) * x.sum(axis=-1)
