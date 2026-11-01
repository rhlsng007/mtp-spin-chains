"""
Finite-shot energy estimation (Phase 4, T4.2).

The Hamiltonian terms from ``models.hamiltonians.hamiltonian_terms`` are split
greedily into qubit-wise commuting groups; each group is measured in one
product basis (for the TFIM: all ZZ terms in the Z basis, all X terms in the X
basis). For a group g with outcome distribution p_g(b) over bitstrings b the
term values are fixed numbers v_g(b), so

    E = c_0 + sum_g sum_b p_g(b) v_g(b)

and with M shots per basis the sample-mean estimator is unbiased with variance

    Var(E_hat) = sum_g Var_{p_g}(v_g) / M.

The probabilities are the exact diagonals of the rotated noisy density matrix,
so sampling costs one multinomial draw per basis. Basis rotations and readout
are ideal here; readout error is the subject of Phase 5 (T5.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from noise.channels import apply_unitary

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_SDG = np.diag([1, -1j])
BASIS_ROTATION = {"X": _H, "Y": _H @ _SDG}        # maps the eigenbasis of P to the Z basis


def _bits(n_qubits: int) -> np.ndarray:
    """bits[b, w] = value of wire w in basis index b (wire 0 = most significant)."""
    b = np.arange(2 ** n_qubits)[:, None]
    shifts = (n_qubits - 1 - np.arange(n_qubits))[None, :]
    return (b >> shifts) & 1


@dataclass
class MeasurementGroup:
    basis: dict                       # wire -> "X" | "Y" | "Z"
    values: np.ndarray                # v_g(b), length 2^N
    terms: list = field(default_factory=list)


class EnergyEstimator:
    """Exact and finite-shot energies of density matrices for a Pauli Hamiltonian."""

    def __init__(self, terms, n_qubits: int):
        self.n_qubits = N = n_qubits
        self.constant = 0.0
        groups: list[dict] = []
        members: list[list] = []
        for coeff, ops in terms:
            ops = {int(w): o for w, o in ops.items() if o != "I"}
            if not ops:
                self.constant += float(np.real(coeff))
                continue
            for g, mem in zip(groups, members):
                if all(g.get(w, o) == o for w, o in ops.items()):
                    g.update(ops)
                    mem.append((float(np.real(coeff)), ops))
                    break
            else:
                groups.append(dict(ops))
                members.append([(float(np.real(coeff)), ops)])
        bits = _bits(N)
        self.groups = []
        for g, mem in zip(groups, members):
            v = np.zeros(2 ** N)
            for coeff, ops in mem:
                parity = bits[:, sorted(ops)].sum(axis=1) % 2
                v += coeff * (1 - 2 * parity)
            self.groups.append(MeasurementGroup(basis=g, values=v, terms=mem))

    # -- probabilities ----------------------------------------------------
    def probabilities(self, rho: np.ndarray) -> list[np.ndarray]:
        out = []
        for g in self.groups:
            r = rho
            for w, o in sorted(g.basis.items()):
                if o in BASIS_ROTATION:
                    r = apply_unitary(r, BASIS_ROTATION[o], [w], self.n_qubits)   # returns a new array
            p = np.clip(np.real(np.diagonal(r)), 0.0, None)
            out.append(p / p.sum())
        return out

    # -- energies ---------------------------------------------------------
    def exact(self, rho: np.ndarray, probs=None) -> float:
        probs = self.probabilities(rho) if probs is None else probs
        return self.constant + float(sum(p @ g.values for p, g in zip(probs, self.groups)))

    def variance(self, rho: np.ndarray, shots: int, probs=None) -> float:
        """Variance of the M-shot estimator (M shots in every basis)."""
        probs = self.probabilities(rho) if probs is None else probs
        var = 0.0
        for p, g in zip(probs, self.groups):
            m = p @ g.values
            var += p @ (g.values - m) ** 2
        return float(var / shots)

    def sample(self, rho: np.ndarray, shots: int, rng: np.random.Generator,
               probs=None, repetitions: int | None = None):
        """M-shot estimate(s): a float, or an array of independent repetitions."""
        probs = self.probabilities(rho) if probs is None else probs
        reps = 1 if repetitions is None else repetitions
        est = np.full(reps, self.constant)
        for p, g in zip(probs, self.groups):
            counts = rng.multinomial(shots, p, size=reps)
            est += counts @ g.values / shots
        return float(est[0]) if repetitions is None else est

    def estimate(self, rho: np.ndarray, shots: int | None, rng: np.random.Generator | None = None) -> float:
        """Exact expectation if shots is None, else one M-shot estimate."""
        if shots is None:
            return self.exact(rho)
        return self.sample(rho, shots, rng)
