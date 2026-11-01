"""
Noisy density-matrix simulation of the Phase 2 ansaetze (Phase 4, T4.1).

The circuits are rebuilt gate by gate from the Phase 2 parameter layouts:

* HVA, parameters (gamma_1, beta_1, ..., gamma_p, beta_p):
  Hadamard on every wire (prepares |+>^N), then per layer IsingZZ(2 gamma_l)
  on bonds (0,1), (1,2), ..., (N-2,N-1) and RX(2 beta_l) on every wire.
* HEA, parameters theta.reshape(p + 1, 2, N) indexed [layer, (RY, RZ), qubit]:
  rotation layer 0, then per layer the CNOT staircase CNOT(0,1), ..., CNOT(N-2,N-1)
  followed by a rotation layer; every rotation layer applies RY then RZ to each qubit.

Gate matrices follow PennyLane: RX(phi) = exp(-i phi X / 2), likewise RY, RZ,
IsingZZ(phi) = exp(-i phi Z Z / 2). With a noiseless model the output is
exactly |psi><psi| with psi = Phase 2 ``Ansatz.state(theta)`` (checked in the tests).
"""

from __future__ import annotations

import numpy as np

from config import CFG
from noise.channels import NoiseModel, apply_diagonal, apply_unitary, zero_state

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def rx(phi: float) -> np.ndarray:
    c, s = np.cos(phi / 2), np.sin(phi / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(phi: float) -> np.ndarray:
    c, s = np.cos(phi / 2), np.sin(phi / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz_diag(phi: float) -> np.ndarray:
    return np.array([np.exp(-0.5j * phi), np.exp(0.5j * phi)])


def isingzz_diag(phi: float) -> np.ndarray:
    a, b = np.exp(-0.5j * phi), np.exp(0.5j * phi)
    return np.array([a, b, b, a])


# A gate is (name, wires, matrix, is_diagonal); for diagonal gates "matrix" is the diagonal.
def gate_sequence(ansatz: str, n_qubits: int, depth: int, theta,
                  state_prep_gates: bool | None = None) -> list[tuple]:
    """Ordered gate list of the HVA or HEA circuit (open boundaries)."""
    N, p = n_qubits, depth
    theta = np.asarray(theta, dtype=float)
    if state_prep_gates is None:
        state_prep_gates = CFG.noise.noisy_state_prep
    gates = []
    if ansatz == "hva":
        if theta.size != 2 * p:
            raise ValueError(f"HVA needs {2 * p} parameters, got {theta.size}")
        for q in range(N):
            gates.append(("H", (q,), _H, False))
        for layer in range(p):
            g, b = theta[2 * layer], theta[2 * layer + 1]
            zz = isingzz_diag(2.0 * g)
            for i in range(N - 1):
                gates.append(("IsingZZ", (i, i + 1), zz, True))
            u = rx(2.0 * b)
            for q in range(N):
                gates.append(("RX", (q,), u, False))
        if not state_prep_gates:
            gates = [g_ for g_ in gates if g_[0] != "H"]
            gates = [("PREP+", tuple(range(N)), None, False)] + gates
    elif ansatz == "hea":
        if theta.size != 2 * N * (p + 1):
            raise ValueError(f"HEA needs {2 * N * (p + 1)} parameters, got {theta.size}")
        t = theta.reshape(p + 1, 2, N)

        def rotations(layer):
            for q in range(N):
                gates.append(("RY", (q,), ry(t[layer, 0, q]), False))
                gates.append(("RZ", (q,), rz_diag(t[layer, 1, q]), True))

        rotations(0)
        for layer in range(1, p + 1):
            for i in range(N - 1):
                gates.append(("CNOT", (i, i + 1), _CNOT, False))
            rotations(layer)
    else:
        raise ValueError(f"unknown ansatz {ansatz!r}")
    return gates


def gate_counts(ansatz: str, n_qubits: int, depth: int, noise_model: NoiseModel | None = None) -> dict:
    """Numbers of noisy one- and two-qubit gates (for error-budget plots)."""
    noisy_rz = CFG.noise.noisy_rz if noise_model is None else noise_model.noisy_rz
    N, p = n_qubits, depth
    if ansatz == "hva":
        n1 = (N if CFG.noise.noisy_state_prep else 0) + p * N
        n2 = p * (N - 1)
    else:
        n1 = (p + 1) * N * (2 if noisy_rz else 1)
        n2 = p * (N - 1)
    return {"n_1q": n1, "n_2q": n2, "n_total": n1 + n2}


def simulate_density_matrix(ansatz: str, n_qubits: int, depth: int, theta,
                            noise_model: NoiseModel | None = None) -> np.ndarray:
    """Output density matrix (2^N x 2^N, complex128) of the noisy circuit."""
    nm = noise_model if noise_model is not None else NoiseModel()
    N = n_qubits
    rho = zero_state(N)
    for name, wires, mat, is_diag in gate_sequence(ansatz, N, depth, theta):
        if name == "PREP+":                    # noiseless |+>^N preparation
            rho = np.full((2 ** N, 2 ** N), 2.0 ** (-N), dtype=complex)
            continue
        if is_diag:
            rho = apply_diagonal(rho, mat, wires, N)
        else:
            rho = apply_unitary(rho, mat, wires, N)
        rho = nm.after_gate(rho, name, wires, N)
    return rho


def simulate_statevector(ansatz: str, n_qubits: int, depth: int, theta) -> np.ndarray:
    """Noiseless state from the same gate list (independent of vqe.statevector)."""
    N = n_qubits
    psi = np.zeros((2,) * N, dtype=complex)
    psi[(0,) * N] = 1.0
    for name, wires, mat, is_diag in gate_sequence(ansatz, N, depth, theta):
        if name == "PREP+":
            psi = np.full((2,) * N, 2 ** (-N / 2), dtype=complex)
            continue
        wires = list(wires)
        k = len(wires)
        U = np.diag(mat) if is_diag else mat
        psi = np.tensordot(U.reshape((2,) * (2 * k)), psi, axes=(list(range(k, 2 * k)), wires))
        psi = np.moveaxis(psi, list(range(k)), wires)
    return psi.reshape(-1)
