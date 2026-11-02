"""
Error mitigation (Phase 5): readout correction and zero-noise extrapolation.

Hardware-like noise model (decided at the end of Phase 4)
---------------------------------------------------------
``make_mitigation_noise_model`` reuses the Phase 4 ``NoiseModel`` with the
Phase 5 settings in ``CFG.mitigation``: two-qubit depolarizing error equal to
``two_qubit_factor`` (10) times the one-qubit error, virtual (noiseless) RZ,
and the channel "realistic", which combines depolarizing noise with amplitude
damping and dephasing on every qubit a gate touches. Readout error is a
separate ``ReadoutModel``. Measurement-basis rotations stay ideal, as in Phase 4.

T5.1 Readout mitigation
-----------------------
Each qubit is read through a 2 x 2 column-stochastic matrix (rows = outcome,
columns = state), R_q = [[1 - p01, p10], [p01, 1 - p10]], so a measured
distribution is q = (R_0 (x) ... (x) R_{N-1}) p. The confusion matrix is
estimated from calibration circuits: "full" prepares all 2^N basis states,
"tensored" only |0...0> and |1...1> (enough for uncorrelated readout). The
calibration X gates carry the one-qubit gate noise when
``CFG.mitigation.calibration_gate_noise`` is set, so the calibrated matrix
absorbs a state-preparation error (SPAM): A_cal = R (P_0 (x) ... ), with
P_q = [[1, 1 - f], [0, f]] and f the probability that X|0> ends in |1>.
Correction: p_hat = A_cal^{-1} q ("inverse", unbiased and linear) or its
Euclidean projection onto the probability simplex ("projected", physical but
biased). Energies use the linear form E = c + sum_g w_g . q_g with
w_g = A_cal^{-T} v_g, which equals c + sum_g v_g . (A_cal^{-1} q_g).

T5.2 Zero-noise extrapolation
-----------------------------
Unitary folding with odd scale factors lambda = 2k + 1: global folding
U -> U (U^dagger U)^k, local folding G -> G (G^dagger G)^k for every gate.
Noise follows every gate, so the number of noisy gates is multiplied by
lambda exactly (virtual RZ stays noiseless). Inverse gates keep the name of
the gate, so the noise model treats them like the original. Extrapolators
(values at lambda = 0): "linear" (least-squares line through all points; with
two points the plan's formula E* = (l2 E(l1) - l1 E(l2)) / (l2 - l1)),
"richardson" (polynomial through all points, Lagrange form), "exponential"
(E = E_inf + A exp(-b lambda) with known asymptote E_inf, the energy of the
maximally mixed state measured through the same pipeline; log-linear fit).
Linear and Richardson are linear in the data, so their variances are
sum_k c_k^2 Var(E_k).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CFG
from noise.channels import NoiseModel, apply_diagonal, apply_unitary, make_noise_model, zero_state
from noise.density import gate_sequence
from noise.shots import EnergyEstimator

PHASE5_CHANNELS = ("realistic", "depolarizing", "amplitude_damping", "dephasing")
FOLDINGS = ("global", "local")
EXTRAPOLATORS = ("linear", "richardson", "exponential")
_X = np.array([[0, 1], [1, 0]], dtype=complex)


# ---------------------------------------------------------------------------
# Hardware-like noise model
# ---------------------------------------------------------------------------
def make_mitigation_noise_model(channel: str, strength: float, cfg=None) -> NoiseModel:
    """Phase 5 gate-noise model: Phase 4 channels with the Phase 5 settings, or "realistic"."""
    cfg = cfg if cfg is not None else CFG.mitigation
    s = float(strength)
    if channel == "realistic":
        return NoiseModel(depolarizing=s,
                          depolarizing_2q=min(s * cfg.two_qubit_factor, 15.0 / 16.0),
                          amplitude_damping=s * cfg.realistic_damping_ratio,
                          dephasing=s * cfg.realistic_dephasing_ratio,
                          noisy_rz=cfg.noisy_rz)
    return make_noise_model(channel, s, cfg)        # uses cfg.two_qubit_factor and cfg.noisy_rz


def error_budget(channel: str, strength, n_1q: int, n_2q: int, cfg=None):
    """
    Expected number of error events per circuit run (the Phase 4 variable lambda).

    Depolarizing and realistic: s (n_1q + f n_2q), f the two-qubit factor (the
    depolarizing part dominates the realistic model). Amplitude damping and
    dephasing act on every qubit a gate touches: s (n_1q + 2 n_2q).
    """
    cfg = cfg if cfg is not None else CFG.mitigation
    s = np.asarray(strength, dtype=float)
    if channel in ("depolarizing", "realistic"):
        return s * (n_1q + cfg.two_qubit_factor * n_2q)
    return s * (n_1q + 2 * n_2q)


# ---------------------------------------------------------------------------
# Gate folding (T5.2)
# ---------------------------------------------------------------------------
def inverse_gate(gate: tuple) -> tuple:
    """Inverse of a (name, wires, matrix, is_diagonal) gate; the name is kept for the noise model."""
    name, wires, mat, is_diag = gate
    if name == "PREP+":
        raise ValueError("state preparation cannot be inverted")
    inv = np.conj(mat) if is_diag else np.conj(np.asarray(mat)).T
    return (name, wires, inv, is_diag)


def _check_scale(scale) -> int:
    k = (int(round(scale)) - 1) // 2
    if abs(scale - (2 * k + 1)) > 1e-12 or k < 0:
        raise ValueError(f"unitary folding needs an odd integer scale factor, got {scale}")
    return k


def fold_gates(gates: list, scale, method: str = "global") -> list:
    """Folded gate list with noise scale factor ``scale`` (1, 3, 5, ...)."""
    k = _check_scale(scale)
    prep = [g for g in gates if g[0] == "PREP+"]
    U = [g for g in gates if g[0] != "PREP+"]
    if method == "global":
        Udag = [inverse_gate(g) for g in reversed(U)]
        return prep + U + k * (Udag + U)
    if method == "local":
        out = list(prep)
        for g in U:
            out.append(g)
            gd = inverse_gate(g)
            out.extend(k * [gd, g])
        return out
    raise ValueError(f"unknown folding {method!r}; expected one of {FOLDINGS}")


def folded_gate_sequence(ansatz: str, n_qubits: int, depth: int, theta, scale, method="global") -> list:
    return fold_gates(gate_sequence(ansatz, n_qubits, depth, theta), scale, method)


def noisy_gate_counts(gates: list, noise_model: NoiseModel) -> dict:
    """Numbers of one- and two-qubit gates that are followed by noise."""
    n1 = n2 = 0
    for name, wires, _, _ in gates:
        if name == "PREP+" or (name == "RZ" and not noise_model.noisy_rz):
            continue
        if len(wires) == 1:
            n1 += 1
        else:
            n2 += 1
    return {"n_1q": n1, "n_2q": n2, "n_total": n1 + n2}


def simulate_gates(gates: list, n_qubits: int, noise_model: NoiseModel | None = None) -> np.ndarray:
    """Density matrix of an arbitrary gate list (same kernels as noise.density)."""
    nm = noise_model if noise_model is not None else NoiseModel()
    N = n_qubits
    rho = zero_state(N)
    for name, wires, mat, is_diag in gates:
        if name == "PREP+":
            rho = np.full((2 ** N, 2 ** N), 2.0 ** (-N), dtype=complex)
            continue
        if is_diag:
            rho = apply_diagonal(rho, mat, wires, N)
        else:
            rho = apply_unitary(rho, mat, wires, N)
        rho = nm.after_gate(rho, name, wires, N)
    return rho


def simulate_folded(ansatz: str, n_qubits: int, depth: int, theta, noise_model: NoiseModel | None,
                    scale=1, method: str = "global") -> np.ndarray:
    return simulate_gates(folded_gate_sequence(ansatz, n_qubits, depth, theta, scale, method),
                          n_qubits, noise_model)


# ---------------------------------------------------------------------------
# Readout model and calibration (T5.1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReadoutModel:
    """Independent asymmetric readout bit flips: p01 = P(1 | 0), p10 = P(0 | 1), per qubit."""
    p01: tuple
    p10: tuple

    @classmethod
    def uniform(cls, n_qubits: int, p01: float, p10: float) -> "ReadoutModel":
        return cls(tuple([float(p01)] * n_qubits), tuple([float(p10)] * n_qubits))

    @classmethod
    def from_config(cls, n_qubits: int, cfg=None) -> "ReadoutModel":
        cfg = cfg if cfg is not None else CFG.mitigation
        return cls.uniform(n_qubits, cfg.readout_p01, cfg.readout_p10)

    @property
    def n_qubits(self) -> int:
        return len(self.p01)

    def matrices(self) -> np.ndarray:
        """(N, 2, 2) column-stochastic matrices R_q[outcome, state]."""
        a, b = np.asarray(self.p01), np.asarray(self.p10)
        return np.stack([np.array([[1 - x, y], [x, 1 - y]]) for x, y in zip(a, b)])

    def matrix(self) -> np.ndarray:
        return kron_all(self.matrices())


def kron_all(mats) -> np.ndarray:
    out = np.ones((1, 1))
    for m in mats:
        out = np.kron(out, m)                      # wire 0 = most significant bit
    return out


def apply_local_matrices(vecs, mats) -> np.ndarray:
    """(M_0 (x) ... (x) M_{N-1}) applied to the last axis of vecs (length 2^N), without forming it."""
    mats = np.asarray(mats)
    N = mats.shape[0]
    v = np.asarray(vecs)
    lead = v.shape[:-1]
    for q in range(N):
        v = v.reshape(lead + (2 ** q, 2, 2 ** (N - q - 1)))
        v = np.einsum("ab,...ibj->...iaj", mats[q], v)
    return v.reshape(lead + (2 ** N,))


def prepared_one_probability(noise_model: NoiseModel | None) -> float:
    """f = P(qubit in |1>) after a noisy X gate on |0> (the calibration preparation)."""
    rho = apply_unitary(zero_state(1), _X, [0], 1)
    if noise_model is not None:
        rho = noise_model.after_gate(rho, "X", [0], 1)
    return float(np.real(rho[1, 1]))


def calibration_matrices(readout: ReadoutModel, noise_model: NoiseModel | None = None) -> np.ndarray:
    """
    Exact per-qubit calibration matrices C_q = R_q P_q (infinite calibration shots).

    Qubits prepared in 0 receive no gate (idle qubits are noiseless), qubits in 1
    an X gate followed by the gate noise, so P_q = [[1, 1 - f], [0, f]].
    """
    f = 1.0 if noise_model is None else prepared_one_probability(noise_model)
    P = np.array([[1.0, 1.0 - f], [0.0, f]])
    return np.stack([R @ P for R in readout.matrices()])


def full_calibration_exact(readout: ReadoutModel, noise_model: NoiseModel | None = None) -> np.ndarray:
    """2^N x 2^N calibrated confusion matrix A[measured, prepared] (infinite shots)."""
    return kron_all(calibration_matrices(readout, noise_model))


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.real(p), 0.0, None)
    return p / p.sum(axis=-1, keepdims=True)


def sample_full_calibration(A_exact: np.ndarray, shots: int, rng: np.random.Generator,
                            repetitions: int) -> np.ndarray:
    """(R, d, d) confusion matrices estimated from ``shots`` shots per basis-state circuit."""
    d = A_exact.shape[0]
    counts = rng.multinomial(shots, _normalize(A_exact.T), size=(repetitions, d))   # [r, prepared, measured]
    return np.transpose(counts, (0, 2, 1)) / shots


def sample_tensored_calibration(C_exact: np.ndarray, shots: int, rng: np.random.Generator,
                                repetitions: int) -> np.ndarray:
    """(R, N, 2, 2) per-qubit matrices from the two circuits |0...0> and |1...1>."""
    N = C_exact.shape[0]
    mats = np.empty((repetitions, N, 2, 2))
    bits = _bits(N)
    for col in (0, 1):
        dist = kron_all([C[:, col][:, None] for C in C_exact])[:, 0]
        counts = rng.multinomial(shots, _normalize(dist), size=repetitions)          # (R, 2^N)
        ones = counts @ bits / shots                                                  # (R, N): P(read 1)
        mats[:, :, 1, col] = ones
        mats[:, :, 0, col] = 1.0 - ones
    return mats


def tensored_calibration_exact(C_exact: np.ndarray) -> np.ndarray:
    return np.array(C_exact, dtype=float)


def _bits(n_qubits: int) -> np.ndarray:
    b = np.arange(2 ** n_qubits)[:, None]
    return (b >> (n_qubits - 1 - np.arange(n_qubits))[None, :]) & 1


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of each row of v onto the probability simplex."""
    v = np.atleast_2d(np.asarray(v, dtype=float))
    u = -np.sort(-v, axis=1)
    css = np.cumsum(u, axis=1) - 1.0
    k = np.arange(1, v.shape[1] + 1)
    cond = u - css / k > 0
    r = cond.shape[1] - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = css[np.arange(v.shape[0]), r] / (r + 1)
    out = np.maximum(v - theta[:, None], 0.0)
    return out


def inverse_transpose(A: np.ndarray) -> np.ndarray:
    """A^{-T} for a matrix or a batch of matrices (reuse it for many weight vectors)."""
    return np.swapaxes(np.linalg.inv(A), -1, -2)


def correct_probabilities(q, A=None, method: str = "inverse", local_mats=None) -> np.ndarray:
    """
    Readout-corrected distributions. Pass the full matrix A, or per-qubit
    local_mats (tensored calibration). "inverse": A^{-1} q (may be negative);
    "projected": the nearest point of the probability simplex.
    """
    q = np.asarray(q, dtype=float)
    if local_mats is not None:
        p = apply_local_matrices(q, np.linalg.inv(local_mats))
    else:
        p = np.linalg.solve(A, q.T).T if q.ndim > 1 else np.linalg.solve(A, q)
    if method == "inverse":
        return p
    if method == "projected":
        return project_to_simplex(p).reshape(p.shape)
    raise ValueError(f"unknown correction method {method!r}")


def total_variation(p, q) -> np.ndarray:
    return 0.5 * np.sum(np.abs(np.asarray(p) - np.asarray(q)), axis=-1)


def negative_mass(p) -> np.ndarray:
    return -np.sum(np.minimum(np.asarray(p), 0.0), axis=-1)


# ---------------------------------------------------------------------------
# Energies measured through a noisy readout
# ---------------------------------------------------------------------------
class ReadoutEnergy:
    """
    Energy of a Pauli Hamiltonian measured group by group (Phase 4 grouping)
    through a readout model, with optional readout correction.

    probs arrays have shape (..., G, 2^N); weights (..., G, 2^N).
    """

    def __init__(self, terms, n_qubits: int, readout: ReadoutModel):
        self.estimator = EnergyEstimator(terms, n_qubits)
        self.n_qubits = n_qubits
        self.readout = readout
        self.constant = self.estimator.constant
        self.values = np.stack([g.values for g in self.estimator.groups])   # (G, d)
        self.R_mats = readout.matrices()

    @property
    def n_groups(self) -> int:
        return self.values.shape[0]

    def ideal_probabilities(self, rho: np.ndarray) -> np.ndarray:
        return np.stack(self.estimator.probabilities(rho))

    def measured_probabilities(self, probs: np.ndarray) -> np.ndarray:
        return apply_local_matrices(probs, self.R_mats)

    def energy(self, probs: np.ndarray, weights: np.ndarray | None = None):
        w = self.values if weights is None else weights
        return self.constant + np.sum(w * probs, axis=(-2, -1))

    def weights_full(self, A: np.ndarray) -> np.ndarray:
        """w_g = A^{-T} v_g for a full matrix A (d, d) or a batch (R, d, d)."""
        A = np.asarray(A)
        AT = np.swapaxes(A, -1, -2)
        rhs = np.broadcast_to(self.values.T, A.shape[:-2] + self.values.T.shape)
        return np.swapaxes(np.linalg.solve(AT, rhs), -1, -2)

    def weights_from_inverse_T(self, inv_T: np.ndarray) -> np.ndarray:
        """w_g = M v_g for a precomputed M = A^{-T}, shape (d, d) or (R, d, d)."""
        return np.swapaxes(np.asarray(inv_T) @ self.values.T, -1, -2)

    def weights_tensored(self, local_mats: np.ndarray) -> np.ndarray:
        """w_g = (C_0^{-T} (x) ...) v_g for per-qubit matrices (N, 2, 2) or (R, N, 2, 2)."""
        m = np.asarray(local_mats)
        inv_T = np.swapaxes(np.linalg.inv(m), -1, -2)
        if m.ndim == 3:
            return apply_local_matrices(self.values, inv_T)
        return np.stack([apply_local_matrices(self.values, mi) for mi in inv_T])

    def mixed_state_energy(self, weights: np.ndarray | None = None, readout: bool = True) -> float:
        """Energy of I / 2^N through the pipeline (the exponential-extrapolation asymptote)."""
        d = 2 ** self.n_qubits
        u = np.full((self.n_groups, d), 1.0 / d)
        q = self.measured_probabilities(u) if readout else u
        return float(self.energy(q, weights))

    def shot_variance(self, probs: np.ndarray, shots: int, weights: np.ndarray | None = None):
        """Variance of the M-shot estimator (M shots per group) of sum_g w_g . probs_g."""
        w = self.values if weights is None else weights
        m = np.sum(w * probs, axis=-1, keepdims=True)
        return np.sum(probs * (w - m) ** 2, axis=(-2, -1)) / shots

    def sample_counts(self, probs: np.ndarray, shots: int, rng: np.random.Generator,
                      repetitions: int) -> np.ndarray:
        """Counts of shape (R, ..., G, d) for probs of shape (..., G, d)."""
        p = _normalize(probs)
        return rng.multinomial(shots, p, size=(repetitions,) + p.shape[:-1])

    def energy_from_counts(self, counts: np.ndarray, shots: int, weights: np.ndarray | None = None):
        """counts (R, ..., G, d); weights (G, d) or per-repetition (R, G, d)."""
        w = self.values if weights is None else np.asarray(weights)
        if w.ndim == 3:                               # per repetition
            extra = counts.ndim - 3
            w = w.reshape(w.shape[:1] + (1,) * extra + w.shape[1:])
        return self.constant + np.sum(w * counts, axis=(-2, -1)) / shots


# ---------------------------------------------------------------------------
# Extrapolation (T5.2)
# ---------------------------------------------------------------------------
def richardson_coefficients(scales) -> np.ndarray:
    """c_k with E(0) = sum_k c_k E(lambda_k) exact for polynomials of degree n - 1."""
    lam = np.asarray(scales, dtype=float)
    c = np.ones_like(lam)
    for k in range(lam.size):
        for j in range(lam.size):
            if j != k:
                c[k] *= lam[j] / (lam[j] - lam[k])
    return c


def polynomial_coefficients(scales, order: int) -> np.ndarray:
    """Least-squares polynomial fit of the given order, evaluated at 0, as weights."""
    lam = np.asarray(scales, dtype=float)
    if order + 1 > lam.size:
        raise ValueError("order too high for the number of scale factors")
    V = np.vander(lam, order + 1, increasing=True)
    return np.linalg.pinv(V)[0]


def linear_coefficients(scales) -> np.ndarray:
    return polynomial_coefficients(scales, 1)


def linear_two_point(l1: float, e1, l2: float, e2):
    """The plan's formula E* = (l2 E(l1) - l1 E(l2)) / (l2 - l1)."""
    return (l2 * np.asarray(e1) - l1 * np.asarray(e2)) / (l2 - l1)


def extrapolation_coefficients(scales, method: str) -> np.ndarray:
    if method == "linear":
        return linear_coefficients(scales)
    if method == "richardson":
        return richardson_coefficients(scales)
    raise ValueError(f"{method!r} is not a linear extrapolator")


def extrapolate(scales, values, method: str = "richardson", axis: int = -1, asymptote=0.0):
    """
    Zero-noise value from energies at the given scale factors along ``axis``.

    "exponential" fits log|E - E_inf| linearly in lambda (asymptote E_inf known,
    broadcast against the other axes); points where E - E_inf changes sign or
    vanishes give NaN.
    """
    y = np.moveaxis(np.asarray(values, dtype=float), axis, -1)
    if method in ("linear", "richardson"):
        return y @ extrapolation_coefficients(scales, method)
    if method == "exponential":
        a = np.broadcast_to(np.asarray(asymptote, dtype=float), y.shape[:-1])[..., None]
        d = y - a
        sgn = np.sign(d[..., :1])
        ok = np.all(d * sgn > 0, axis=-1)
        z = np.log(np.where(ok[..., None], d * sgn, 1.0))
        intercept = z @ linear_coefficients(scales)
        out = a[..., 0] + sgn[..., 0] * np.exp(intercept)
        return np.where(ok, out, np.nan)
    raise ValueError(f"unknown extrapolator {method!r}; expected one of {EXTRAPOLATORS}")


def extrapolation_variance(scales, variances, method: str, axis: int = -1):
    """Variance of a linear extrapolator for independent estimates: sum_k c_k^2 Var_k."""
    c = extrapolation_coefficients(scales, method)
    v = np.moveaxis(np.asarray(variances, dtype=float), axis, -1)
    return v @ c ** 2


def variance_amplification(scales, method: str) -> float:
    """Gamma^2 = sum_k c_k^2: factor on the variance at equal shots per scale factor."""
    return float(np.sum(extrapolation_coefficients(scales, method) ** 2))
