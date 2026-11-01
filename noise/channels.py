"""
Noise channels and density-matrix kernels (Phase 4, T4.1).

Representation
--------------
A density matrix of N qubits is a C-contiguous (2^N, 2^N) complex128 matrix,
wire 0 being the most significant bit (PennyLane ordering, as everywhere in
this project). Gates and channels act on reshaped views of it, so a gate costs
two batched matrix products and a channel a few strided slice updates.

Channel conventions (identical to PennyLane, so default.mixed cross-checks
are exact):

* single-qubit depolarizing, probability p (qml.DepolarizingChannel):
  rho -> (1 - p) rho + (p/3) (X rho X + Y rho Y + Z rho Z)
* two-qubit depolarizing, probability p (15 non-identity Paulis):
  rho -> (1 - p) rho + (p/15) sum_{P != II} P rho P
* amplitude damping, gamma (qml.AmplitudeDamping):
  K0 = [[1, 0], [0, sqrt(1-gamma)]],  K1 = [[0, sqrt(gamma)], [0, 0]]
* dephasing = phase damping, gamma (qml.PhaseDamping):
  K0 = [[1, 0], [0, sqrt(1-gamma)]],  K1 = [[0, 0], [0, sqrt(gamma)]]

Amplitude and phase damping with the same gamma shrink the coherences by the
same factor sqrt(1 - gamma), so their strengths are directly comparable.

The fast kernels use the Pauli-twirl identities
    sum_{P in 1q Paulis} P rho P = 2 * 1_q (x) Tr_q rho
    sum_{P in 2q Paulis} P rho P = 4 * 1_qr (x) Tr_qr rho
so no Kraus sums are formed. ``apply_kraus`` is the slow generic reference
used in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CFG

CHANNELS = ("depolarizing", "amplitude_damping", "dephasing")

# ---------------------------------------------------------------------------
# Kraus operators (reference definitions)
# ---------------------------------------------------------------------------
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = (_I, _X, _Y, _Z)


def kraus_depolarizing(p: float) -> list[np.ndarray]:
    return [np.sqrt(1 - p) * _I] + [np.sqrt(p / 3) * P for P in (_X, _Y, _Z)]


def kraus_depolarizing_2q(p: float) -> list[np.ndarray]:
    ops = [np.sqrt(1 - p) * np.eye(4, dtype=complex)]
    for a in range(4):
        for b in range(4):
            if a == 0 and b == 0:
                continue
            ops.append(np.sqrt(p / 15) * np.kron(PAULIS[a], PAULIS[b]))
    return ops


def kraus_amplitude_damping(gamma: float) -> list[np.ndarray]:
    return [np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
            np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)]


def kraus_dephasing(gamma: float) -> list[np.ndarray]:
    return [np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
            np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=complex)]


# ---------------------------------------------------------------------------
# Matrix-view helpers
# ---------------------------------------------------------------------------
# rho is kept as a C-contiguous (2^N, 2^N) complex matrix. An operation on
# wires w_1 < w_2 < ... reshapes it (without copying) so that each wire is an
# axis of length 2 on the ket side and on the bra side.

def _side_shape(wires, n_qubits: int) -> list[int]:
    shape, prev = [], 0
    for w in wires:
        shape += [2 ** (w - prev), 2]
        prev = w + 1
    shape.append(2 ** (n_qubits - prev))
    return shape


def _view(rho: np.ndarray, wires, n_qubits: int) -> np.ndarray:
    side = _side_shape(wires, n_qubits)
    return rho.reshape(side + side)


def _block(n_side: int, wires_vals: dict) -> tuple:
    """Index into a _view with n_side axes per side (3 for one wire, 5 for two).

    wires_vals maps (wire_position, "ket" or "bra") -> 0 or 1.
    """
    idx = [slice(None)] * (2 * n_side)
    for (pos, side), val in wires_vals.items():
        idx[2 * pos + 1 + (n_side if side == "bra" else 0)] = val
    return tuple(idx)


def zero_state(n_qubits: int) -> np.ndarray:
    d = 2 ** n_qubits
    rho = np.zeros((d, d), dtype=complex)
    rho[0, 0] = 1.0
    return rho


def _permute_gate(U: np.ndarray, order) -> np.ndarray:
    """Reorder the qubits of a k-qubit gate matrix: new qubit j = old qubit order[j]."""
    k = len(order)
    Ut = U.reshape((2,) * (2 * k))
    perm = list(order) + [k + o for o in order]
    return Ut.transpose(perm).reshape(2 ** k, 2 ** k)


def apply_unitary(rho: np.ndarray, U: np.ndarray, wires, n_qubits: int) -> np.ndarray:
    """rho -> U rho U^dagger (also K rho K^dagger for a single Kraus operator).

    rho must be Hermitian. Then U rho U^dagger = U (U rho)^dagger, so both
    sides are applied as left products U @ (a, 2^k, b) blocks, which NumPy
    does efficiently; the bra-side layout (many tiny products) is avoided.
    """
    wires = list(wires)
    order = sorted(range(len(wires)), key=lambda j: wires[j])
    if order != list(range(len(wires))):
        U = _permute_gate(U, order)
        wires = [wires[j] for j in order]
    d = 2 ** n_qubits
    k = len(wires)
    if wires == list(range(wires[0], wires[0] + k)):        # contiguous block
        shape = (2 ** wires[0], 2 ** k, 2 ** (n_qubits - wires[-1] - 1) * d)
        M = np.matmul(U, rho.reshape(shape)).reshape(d, d)   # U rho
        M = np.ascontiguousarray(M.conj().T)                 # rho U^dagger
        return np.matmul(U, M.reshape(shape)).reshape(d, d)  # U rho U^dagger
    # non-contiguous wires: generic tensor contraction
    T = rho.reshape((2,) * (2 * n_qubits))
    Ut = U.reshape((2,) * (2 * k))
    ins = list(range(k, 2 * k))
    T = np.moveaxis(np.tensordot(Ut, T, axes=(ins, wires)), list(range(k)), wires)
    bra = [w + n_qubits for w in wires]
    T = np.moveaxis(np.tensordot(Ut.conj(), T, axes=(ins, bra)), list(range(k)), bra)
    return np.ascontiguousarray(T).reshape(d, d)


def apply_diagonal(rho: np.ndarray, diag: np.ndarray, wires, n_qubits: int) -> np.ndarray:
    """rho_ij -> d_i rho_ij conj(d_j) for a diagonal gate on the given wires."""
    wires = list(wires)
    k = len(wires)
    dk = np.asarray(diag).reshape((2,) * k)
    order = sorted(range(k), key=lambda j: wires[j])
    dk = dk.transpose(order)
    wires = sorted(wires)
    full = np.ones((2,) * n_qubits, dtype=complex)
    shape = [1] * n_qubits
    for w in wires:
        shape[w] = 2
    full = full * dk.reshape(shape)
    v = full.reshape(-1)
    rho *= v[:, None]
    rho *= v.conj()[None, :]
    return rho


def apply_kraus(rho: np.ndarray, kraus, wires, n_qubits: int) -> np.ndarray:
    """Generic (slow) channel sum_k K rho K^dagger; reference for the tests."""
    out = np.zeros_like(rho)
    for K in kraus:
        out += apply_unitary(rho.copy(), K, wires, n_qubits)
    return out


# ---------------------------------------------------------------------------
# Fast in-place channel kernels
# ---------------------------------------------------------------------------
def depolarize_1q(rho: np.ndarray, q: int, p: float, n_qubits: int) -> np.ndarray:
    if p == 0.0:
        return rho
    V = _view(rho, [q], n_qubits)
    s00 = _block(3, {(0, "ket"): 0, (0, "bra"): 0})
    s11 = _block(3, {(0, "ket"): 1, (0, "bra"): 1})
    tr = V[s00] + V[s11]                      # Tr_q rho (a copy)
    rho *= 1.0 - 4.0 * p / 3.0
    V[s00] += (2.0 * p / 3.0) * tr
    V[s11] += (2.0 * p / 3.0) * tr
    return rho


def depolarize_2q(rho: np.ndarray, q: int, r: int, p: float, n_qubits: int) -> np.ndarray:
    if p == 0.0:
        return rho
    V = _view(rho, sorted((q, r)), n_qubits)
    blocks = [_block(5, {(0, "ket"): a, (0, "bra"): a, (1, "ket"): b, (1, "bra"): b})
              for a in (0, 1) for b in (0, 1)]
    tr = sum(V[s] for s in blocks)            # Tr_qr rho
    rho *= 1.0 - 16.0 * p / 15.0
    for s in blocks:
        V[s] += (4.0 * p / 15.0) * tr
    return rho


def amplitude_damp(rho: np.ndarray, q: int, gamma: float, n_qubits: int) -> np.ndarray:
    if gamma == 0.0:
        return rho
    V = _view(rho, [q], n_qubits)
    s = {(i, j): _block(3, {(0, "ket"): i, (0, "bra"): j}) for i in (0, 1) for j in (0, 1)}
    V[s[0, 0]] += gamma * V[s[1, 1]]
    V[s[1, 1]] *= 1.0 - gamma
    f = np.sqrt(1.0 - gamma)
    V[s[0, 1]] *= f
    V[s[1, 0]] *= f
    return rho


def dephase(rho: np.ndarray, q: int, gamma: float, n_qubits: int) -> np.ndarray:
    if gamma == 0.0:
        return rho
    V = _view(rho, [q], n_qubits)
    f = np.sqrt(1.0 - gamma)
    V[_block(3, {(0, "ket"): 0, (0, "bra"): 1})] *= f
    V[_block(3, {(0, "ket"): 1, (0, "bra"): 0})] *= f
    return rho


# ---------------------------------------------------------------------------
# Noise model: which channel follows which gate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NoiseModel:
    """
    Channels inserted after every gate (T4.1).

    depolarizing     : p of the single-qubit depolarizing channel after 1q gates
    depolarizing_2q  : p of the two-qubit depolarizing channel after 2q gates
    amplitude_damping: gamma on every qubit a gate acts on (both qubits for 2q gates)
    dephasing        : gamma on every qubit a gate acts on
    noisy_rz         : if False, RZ gates are noiseless (virtual Z, as on IBM devices)
    """
    depolarizing: float = 0.0
    depolarizing_2q: float = 0.0
    amplitude_damping: float = 0.0
    dephasing: float = 0.0
    noisy_rz: bool = True

    @property
    def is_noiseless(self) -> bool:
        return (self.depolarizing == 0.0 and self.depolarizing_2q == 0.0
                and self.amplitude_damping == 0.0 and self.dephasing == 0.0)

    def after_gate(self, rho: np.ndarray, gate: str, wires, n_qubits: int) -> np.ndarray:
        if self.is_noiseless or (gate == "RZ" and not self.noisy_rz):
            return rho
        wires = list(wires)
        if len(wires) == 1:
            rho = depolarize_1q(rho, wires[0], self.depolarizing, n_qubits)
        else:
            rho = depolarize_2q(rho, wires[0], wires[1], self.depolarizing_2q, n_qubits)
        for w in wires:
            rho = amplitude_damp(rho, w, self.amplitude_damping, n_qubits)
            rho = dephase(rho, w, self.dephasing, n_qubits)
        return rho


def make_noise_model(channel: str, strength: float, cfg=None) -> NoiseModel:
    """One-parameter noise model used in the sweeps (T4.4)."""
    cfg = cfg if cfg is not None else CFG.noise
    s = float(strength)
    if channel == "depolarizing":
        p2 = min(s * cfg.two_qubit_factor, 15.0 / 16.0)
        return NoiseModel(depolarizing=s, depolarizing_2q=p2, noisy_rz=cfg.noisy_rz)
    if channel == "amplitude_damping":
        return NoiseModel(amplitude_damping=s, noisy_rz=cfg.noisy_rz)
    if channel == "dephasing":
        return NoiseModel(dephasing=s, noisy_rz=cfg.noisy_rz)
    raise ValueError(f"unknown channel {channel!r}; expected one of {CHANNELS}")
