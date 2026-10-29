"""
Variational ansätze for the TFIM (Phase 2, tasks T2.1 and T2.2).

Hamiltonian variational ansatz (HVA), T2.1
------------------------------------------
    |psi(gamma, beta)> = prod_{l=1..p} exp(-i beta_l sum_i X_i)
                                       exp(-i gamma_l sum_i Z_i Z_{i+1}) |+>^N

Parameters theta = (gamma_1, beta_1, ..., gamma_p, beta_p): 2p in total.
Gate form (PennyLane): IsingZZ(2 gamma_l) on every bond, then RX(2 beta_l) on
every qubit. Both generators commute with the parity P = prod_i X_i and
|+>^N has P = +1, so the HVA state always lies in the even-parity sector,
the sector of the reference ground state of Phase 1. theta = 0 gives |+>^N,
and a layer with gamma = beta = 0 is the identity, so the optimum at depth p
padded with zeros is an exact starting point at depth p + 1.

Hardware-efficient ansatz (HEA), T2.2
-------------------------------------
    |psi(theta)> = R(theta_p) C ... R(theta_1) C R(theta_0) |0>^N
    R(theta_l)   = prod_q RZ_q(theta_{l,1,q}) RY_q(theta_{l,0,q})
    C            = CNOT(N-2, N-1) ... CNOT(1, 2) CNOT(0, 1)

p entangling layers and p + 1 rotation layers: 2N(p + 1) parameters,
stored as theta.reshape(p + 1, 2, N) with index [layer, (RY, RZ), qubit].
The HEA knows nothing about the Hamiltonian and breaks the parity symmetry.

Gradients
---------
energy_and_grad uses the adjoint method (reverse-mode differentiation of a
unitary circuit): one forward pass, then a single backward sweep that
un-applies the gates to |phi> = U_k...U_1|psi_0> and to
|lambda> = U_{k+1}^dagger ... U_L^dagger H |psi>. For a gate exp(-i theta G),

    dE/dtheta = 2 Im <lambda| G |phi>,

which is exact to machine precision and costs about three energy evaluations
for all parameters together.
"""

from __future__ import annotations

import numpy as np

from vqe import statevector as sv


class Ansatz:
    """Common interface. Subclasses define the circuit."""

    name = "base"

    def __init__(self, n_qubits: int, depth: int):
        if n_qubits < 2:
            raise ValueError("need at least two qubits")
        if depth < 0:
            raise ValueError("depth must be non-negative")
        self.n_qubits = int(n_qubits)
        self.depth = int(depth)

    # -- interface ---------------------------------------------------------
    @property
    def n_params(self) -> int:
        raise NotImplementedError

    def state(self, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def energy_and_grad(self, theta: np.ndarray, hamiltonian) -> tuple:
        raise NotImplementedError

    def shift_spec(self) -> list:
        """
        Per parameter, (R, omega): E(theta_k) is a trigonometric polynomial
        with frequencies omega * {1, ..., R}. Used by the generalized
        parameter-shift rule in vqe/gradients.py.
        """
        raise NotImplementedError

    def pennylane_ops(self, theta) -> None:
        """Queue the same circuit as PennyLane operations (inside a QNode)."""
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------
    def energy(self, theta: np.ndarray, hamiltonian) -> float:
        psi = self.state(theta)
        return float(np.real(np.vdot(psi, hamiltonian @ psi)))

    def check(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64).ravel()
        if theta.size != self.n_params:
            raise ValueError(f"{self.name}: expected {self.n_params} parameters, got {theta.size}")
        return theta

    def random_params(self, rng: np.random.Generator, scale: float,
                      distribution: str = "normal") -> np.ndarray:
        """
        'normal'  : N(0, scale^2), a small perturbation of the reference state
                    (|+>^N for the HVA, |0>^N for the HEA);
        'uniform' : U[0, 2 pi), the fully random initialization used for
                    barren plateaus (T2.7).
        """
        if distribution == "normal":
            return rng.normal(0.0, scale, self.n_params)
        if distribution == "uniform":
            return rng.uniform(0.0, 2 * np.pi, self.n_params)
        raise ValueError(f"unknown distribution '{distribution}'")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(N={self.n_qubits}, p={self.depth}, params={self.n_params})"


# ---------------------------------------------------------------------------
# HVA
# ---------------------------------------------------------------------------
class HVA(Ansatz):
    name = "hva"

    def __init__(self, n_qubits: int, depth: int, boundary: str = "open"):
        super().__init__(n_qubits, depth)
        self.boundary = boundary
        self.n_bonds = n_qubits - 1 if boundary == "open" else n_qubits
        self._zz = sv.zz_diagonal(n_qubits, boundary)

    @property
    def n_params(self) -> int:
        return 2 * self.depth

    def initial_state(self) -> np.ndarray:
        return sv.plus_state(self.n_qubits)

    def state(self, theta: np.ndarray) -> np.ndarray:
        theta = self.check(theta)
        psi = self.initial_state()
        for gamma, beta in theta.reshape(self.depth, 2):
            psi = psi * sv.integer_diagonal_phase(self._zz, gamma)
            psi = sv.apply_rx_all(psi, 2 * beta, self.n_qubits)
        return psi

    def energy_and_grad(self, theta: np.ndarray, hamiltonian) -> tuple:
        theta = self.check(theta)
        n = self.n_qubits
        psi = self.state(theta)
        h_psi = hamiltonian @ psi
        energy = float(np.real(np.vdot(psi, h_psi)))
        grad = np.zeros(self.n_params)
        stack = np.stack([psi, h_psi])            # [phi, lambda]
        layers = theta.reshape(self.depth, 2)
        for layer in range(self.depth - 1, -1, -1):
            gamma, beta = layers[layer]
            # X layer exp(-i beta sum X): generator sum_q X_q
            grad[2 * layer + 1] = 2 * np.imag(sv.braket_sum_x(stack[1], stack[0], n))
            stack = sv.apply_rx_all(stack, -2 * beta, n)
            # ZZ layer exp(-i gamma sum ZZ): generator diagonal zz
            grad[2 * layer] = 2 * np.imag(sv.braket(stack[1], self._zz * stack[0]))
            stack = stack * sv.integer_diagonal_phase(self._zz, -gamma)
        return energy, grad

    def shift_spec(self) -> list:
        # sum ZZ has integer eigenvalues spaced by 2 over n_bonds bonds;
        # sum X has eigenvalues N, N-2, ..., -N.
        return [(self.n_bonds, 2.0), (self.n_qubits, 2.0)] * self.depth

    def pennylane_ops(self, theta) -> None:
        import pennylane as qml
        n = self.n_qubits
        for q in range(n):
            qml.Hadamard(wires=q)
        for layer in range(self.depth):
            gamma, beta = theta[2 * layer], theta[2 * layer + 1]
            for i in range(self.n_bonds):
                qml.IsingZZ(2 * gamma, wires=[i, (i + 1) % n])
            for q in range(n):
                qml.RX(2 * beta, wires=q)

    @staticmethod
    def pad(theta: np.ndarray, new_depth: int) -> np.ndarray:
        """Append identity layers (gamma = beta = 0): same state at higher depth."""
        theta = np.asarray(theta, dtype=np.float64).ravel()
        out = np.zeros(2 * new_depth)
        out[:theta.size] = theta
        return out


# ---------------------------------------------------------------------------
# HEA
# ---------------------------------------------------------------------------
class HEA(Ansatz):
    name = "hea"

    @property
    def n_params(self) -> int:
        return 2 * self.n_qubits * (self.depth + 1)

    def initial_state(self) -> np.ndarray:
        return sv.zero_state(self.n_qubits)

    def _layers(self, theta: np.ndarray) -> np.ndarray:
        return self.check(theta).reshape(self.depth + 1, 2, self.n_qubits)

    def state(self, theta: np.ndarray) -> np.ndarray:
        n = self.n_qubits
        psi = self.initial_state()
        for layer, (ry, rz) in enumerate(self._layers(theta)):
            if layer > 0:
                psi = sv.apply_cnot_chain(psi, n)
            psi = sv.apply_ry_layer(psi, ry, n)
            psi = psi * sv.rz_layer_phase(rz, n)
        return psi

    def energy_and_grad(self, theta: np.ndarray, hamiltonian) -> tuple:
        n = self.n_qubits
        layers = self._layers(theta)
        psi = self.state(theta)
        h_psi = hamiltonian @ psi
        energy = float(np.real(np.vdot(psi, h_psi)))
        grad = np.zeros_like(layers)
        stack = np.stack([psi, h_psi])
        for layer in range(self.depth, -1, -1):
            ry, rz = layers[layer]
            # RZ_q(phi) = exp(-i phi Z_q / 2): dE/dphi = Im <lambda|Z_q|phi>
            grad[layer, 1] = np.imag(sv.braket_z_all(stack[1], stack[0], n))
            stack = stack * np.conj(sv.rz_layer_phase(rz, n))
            # RY on different qubits commute, so all derivatives use one point
            for q in range(n):
                grad[layer, 0, q] = np.imag(sv.braket_y(stack[1], stack[0], q, n))
            stack = sv.apply_ry_layer(stack, -ry, n)
            if layer > 0:
                stack = sv.apply_cnot_chain(stack, n, inverse=True)
        return energy, grad.ravel()

    def shift_spec(self) -> list:
        return [(1, 1.0)] * self.n_params

    def pennylane_ops(self, theta) -> None:
        import pennylane as qml
        n = self.n_qubits
        for layer in range(self.depth + 1):
            if layer > 0:
                for c in range(n - 1):
                    qml.CNOT(wires=[c, c + 1])
            base = 2 * n * layer
            for q in range(n):
                qml.RY(theta[base + q], wires=q)
            for q in range(n):
                qml.RZ(theta[base + n + q], wires=q)


ANSATZE = {"hva": HVA, "hea": HEA}


def make_ansatz(name: str, n_qubits: int, depth: int, **kwargs) -> Ansatz:
    if name not in ANSATZE:
        raise ValueError(f"unknown ansatz '{name}', choose from {list(ANSATZE)}")
    return ANSATZE[name](n_qubits, depth, **kwargs)
