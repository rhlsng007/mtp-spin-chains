"""
PennyLane versions of the Phase 2 circuits, used as an independent reference.

The optimizer runs on the NumPy kernels in vqe/statevector.py; these QNodes
build the identical circuits from PennyLane gates so that tests can compare
states, energies and gradients (backprop and parameter shift), and so that
Phase 4 can reuse the same circuits on default.mixed with noise channels.
PennyLane is imported lazily: the rest of the package works without it.
"""

from __future__ import annotations

import numpy as np

from models.hamiltonians import hamiltonian_terms, terms_to_pauli_words


def pauli_word_op(word: str):
    """'ZZIX' -> Z(0) @ Z(1) @ X(3) as a PennyLane operator."""
    import pennylane as qml
    table = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    factors = [table[c](i) for i, c in enumerate(word) if c != "I"]
    if not factors:
        return qml.Identity(0)
    return factors[0] if len(factors) == 1 else qml.prod(*factors)


def pennylane_hamiltonian(model: str, n_sites: int, **params):
    """qml.Hamiltonian of a Phase 1 model, e.g. pennylane_hamiltonian('tfim', 6, J=1, h=0.5)."""
    import pennylane as qml
    words = terms_to_pauli_words(n_sites, hamiltonian_terms(model, n_sites, **params))
    coeffs = [float(c) for c, _ in words]
    ops = [pauli_word_op(w) for _, w in words]
    return qml.Hamiltonian(coeffs, ops)


def state_qnode(ansatz, device: str = "default.qubit"):
    import pennylane as qml
    dev = qml.device(device, wires=ansatz.n_qubits)

    @qml.qnode(dev)
    def circuit(theta):
        ansatz.pennylane_ops(theta)
        return qml.state()

    return circuit


def energy_qnode(ansatz, hamiltonian, diff_method: str = "backprop",
                 device: str = "default.qubit"):
    import pennylane as qml
    dev = qml.device(device, wires=ansatz.n_qubits)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(theta):
        ansatz.pennylane_ops(theta)
        return qml.expval(hamiltonian)

    return circuit


def pennylane_state(ansatz, theta) -> np.ndarray:
    return np.asarray(state_qnode(ansatz)(np.asarray(theta, dtype=float)))


def pennylane_energy_and_grad(ansatz, hamiltonian, theta, diff_method: str = "parameter-shift"):
    """(energy, gradient) from PennyLane, using its own differentiation."""
    import pennylane as qml
    from pennylane import numpy as pnp
    circuit = energy_qnode(ansatz, hamiltonian, diff_method=diff_method)
    x = pnp.array(np.asarray(theta, dtype=float), requires_grad=True)
    energy = float(circuit(x))
    grad = np.asarray(qml.grad(circuit)(x), dtype=float)
    return energy, grad
