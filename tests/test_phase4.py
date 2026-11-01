"""
Phase 4 tests: channels, noisy simulator, mixed-state metrics, shots, SPSA,
and zero-noise reproduction of Phases 2 and 3.

Run from the project root:  pytest -q tests/test_phase4.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root on the import path

import numpy as np
import pytest
from scipy.optimize import minimize

from analysis.mixed_state import (log_negativity, mixed_state_metrics, partial_transpose_left,
                                  purity, reduced_left, von_neumann)
from config import CFG, RESULTS_DIR
from models.exact import solve_model
from models.hamiltonians import build_hamiltonian, hamiltonian_terms
from noise.channels import (NoiseModel, amplitude_damp, apply_kraus, apply_unitary, dephase,
                            depolarize_1q, depolarize_2q, kraus_amplitude_damping,
                            kraus_dephasing, kraus_depolarizing, kraus_depolarizing_2q,
                            make_noise_model)
from noise.density import (gate_counts, gate_sequence, simulate_density_matrix,
                           simulate_statevector)
from noise.shots import EnergyEstimator
from vqe.ansatz import make_ansatz
from vqe.spsa import calibrate_gain, spsa_minimize

RNG = np.random.default_rng(4242)


def random_rho(n, rng=RNG):
    d = 2 ** n
    v = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    r = v @ v.conj().T
    return r / np.trace(r)


def n_params(ansatz, N, p):
    return 2 * p if ansatz == "hva" else 2 * N * (p + 1)


def tfim(N, h):
    kw = dict(J=CFG.tfim.J, h=h, boundary="open")
    return build_hamiltonian("tfim", N, **kw), hamiltonian_terms("tfim", N, **kw), \
        solve_model("tfim", N, **kw).ground_energy


# ---------------------------------------------------------------------------
# T4.1 channels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", [0.0, 1e-3, 0.2, 0.75])
def test_kraus_completeness(p):
    for K in (kraus_depolarizing(p), kraus_amplitude_damping(p), kraus_dephasing(p)):
        assert np.allclose(sum(k.conj().T @ k for k in K), np.eye(2), atol=1e-14)
    K2 = kraus_depolarizing_2q(min(p, 15 / 16))
    assert np.allclose(sum(k.conj().T @ k for k in K2), np.eye(4), atol=1e-14)


@pytest.mark.parametrize("p", [1e-4, 0.03, 0.4])
def test_fast_kernels_match_kraus(p):
    N = 4
    rho = random_rho(N)
    for q in range(N):
        assert np.allclose(depolarize_1q(rho.copy(), q, p, N), apply_kraus(rho, kraus_depolarizing(p), [q], N), atol=1e-14)
        assert np.allclose(amplitude_damp(rho.copy(), q, p, N), apply_kraus(rho, kraus_amplitude_damping(p), [q], N), atol=1e-14)
        assert np.allclose(dephase(rho.copy(), q, p, N), apply_kraus(rho, kraus_dephasing(p), [q], N), atol=1e-14)
    for q, r in [(0, 1), (1, 2), (2, 3), (0, 2), (0, 3)]:
        assert np.allclose(depolarize_2q(rho.copy(), q, r, p, N),
                           apply_kraus(rho, kraus_depolarizing_2q(p), [q, r], N), atol=1e-14)


def test_unitary_wire_order():
    """CNOT with control below target equals the dense embedding."""
    N = 3
    rho = random_rho(N)
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    U = np.zeros((8, 8))
    for i in range(8):
        b = [(i >> (2 - w)) & 1 for w in range(3)]
        if b[2]:                       # control wire 2, target wire 1
            b[1] ^= 1
        U[b[0] * 4 + b[1] * 2 + b[2], i] = 1
    assert np.allclose(apply_unitary(rho, cnot, [2, 1], N), U @ rho @ U.T, atol=1e-14)


@pytest.mark.parametrize("p", [1e-3, 0.1, 0.5])
def test_single_qubit_closed_forms(p):
    zero = np.array([[1, 0], [0, 0]], dtype=complex)
    one = np.array([[0, 0], [0, 1]], dtype=complex)
    plus = np.full((2, 2), 0.5, dtype=complex)
    r = depolarize_1q(zero.copy(), 0, p, 1)                  # Bloch vector shrinks by 1 - 4p/3
    assert np.isclose(r[0, 0] - r[1, 1], 1 - 4 * p / 3, atol=1e-15)
    assert np.isclose(amplitude_damp(one.copy(), 0, p, 1)[1, 1], 1 - p, atol=1e-15)
    assert np.isclose(dephase(plus.copy(), 0, p, 1)[0, 1], np.sqrt(1 - p) / 2, atol=1e-15)


@pytest.mark.parametrize("p", [0.0, 0.01, 0.2, 0.5])
def test_werner_state(p):
    """Bell state + two-qubit depolarizing = Werner state: F = 1 - 4p/5, E_N = ln(2F), S_A = ln 2."""
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho = depolarize_2q(np.outer(bell, bell.conj()), 0, 1, p, 2)
    F = np.vdot(bell, rho @ bell).real
    assert np.isclose(F, 1 - 4 * p / 5, atol=1e-14)
    assert np.isclose(log_negativity(rho, 2, 1), np.log(max(2 * F, 1.0)), atol=1e-13)
    assert np.isclose(von_neumann(reduced_left(rho, 2, 1)), np.log(2), atol=1e-13)


# ---------------------------------------------------------------------------
# Noisy simulator
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ansatz", ["hva", "hea"])
@pytest.mark.parametrize("N,p", [(4, 1), (4, 2), (4, 4), (6, 3)])
def test_noiseless_matches_phase2_ansatz(ansatz, N, p):
    """Gate list reproduces the Phase 2 Ansatz.state (convention check)."""
    a = make_ansatz(ansatz, N, p)
    theta = RNG.normal(size=a.n_params)
    psi = np.asarray(a.state(theta), dtype=complex)
    rho = simulate_density_matrix(ansatz, N, p, theta)
    assert np.abs(rho - np.outer(psi, psi.conj())).max() < 1e-12
    assert np.abs(abs(np.vdot(simulate_statevector(ansatz, N, p, theta), psi)) - 1) < 1e-12


@pytest.mark.parametrize("ansatz", ["hva", "hea"])
def test_gate_counts(ansatz):
    for N, p in [(4, 2), (6, 3), (8, 4)]:
        seq = gate_sequence(ansatz, N, p, np.zeros(n_params(ansatz, N, p)))
        n1 = sum(len(g[1]) == 1 for g in seq)
        n2 = sum(len(g[1]) == 2 for g in seq)
        gc = gate_counts(ansatz, N, p)
        assert (gc["n_1q"], gc["n_2q"]) == (n1, n2)


@pytest.mark.parametrize("channel", ["depolarizing", "amplitude_damping", "dephasing"])
@pytest.mark.parametrize("ansatz", ["hva", "hea"])
def test_density_matrix_valid(channel, ansatz):
    N, p = 4, 2
    rho = simulate_density_matrix(ansatz, N, p, RNG.normal(size=n_params(ansatz, N, p)),
                                  make_noise_model(channel, 0.05))
    assert np.isclose(np.trace(rho).real, 1, atol=1e-13)
    assert np.abs(rho - rho.conj().T).max() < 1e-14
    assert np.linalg.eigvalsh(rho).min() > -1e-13


@pytest.mark.parametrize("ansatz", ["hva", "hea"])
def test_full_depolarization_gives_maximally_mixed(ansatz):
    """p = 3/4 (1q) and 15/16 (2q) are completely depolarizing; the last layer is 1q on every wire."""
    N, p = 4, 2
    rho = simulate_density_matrix(ansatz, N, p, RNG.normal(size=n_params(ansatz, N, p)),
                                  NoiseModel(depolarizing=0.75, depolarizing_2q=15 / 16))
    assert np.allclose(rho, np.eye(2 ** N) / 2 ** N, atol=1e-14)
    m = mixed_state_metrics(rho, N)
    assert np.isclose(m["purity"], 2.0 ** -N) and abs(m["E_N"]) < 1e-13 and abs(m["I"]) < 1e-12


@pytest.mark.parametrize("ansatz", ["hva", "hea"])
def test_full_amplitude_damping_gives_ground_state(ansatz):
    """gamma = 1 resets every wire to |0>: a pure state (purity need not fall for non-unital noise)."""
    N, p = 4, 2
    rho = simulate_density_matrix(ansatz, N, p, RNG.normal(size=n_params(ansatz, N, p)),
                                  NoiseModel(amplitude_damping=1.0))
    target = np.zeros((2 ** N, 2 ** N))
    target[0, 0] = 1
    assert np.allclose(rho, target, atol=1e-14)


@pytest.mark.parametrize("channel", ["depolarizing", "amplitude_damping", "dephasing"])
@pytest.mark.parametrize("ansatz", ["hva", "hea"])
def test_variational_principle_and_purity_monotone(channel, ansatz):
    N, p, h = 4, 2, 1.0
    H, _, E0 = tfim(N, h)
    theta = RNG.normal(scale=0.5, size=n_params(ansatz, N, p))
    pur = []
    for s in CFG.noise.strengths():
        rho = simulate_density_matrix(ansatz, N, p, theta, make_noise_model(channel, s))
        assert np.real(np.trace(H @ rho)) >= E0 - 1e-12
        pur.append(purity(rho))
    assert np.all(np.diff(pur) <= CFG.noise.monotone_tol)


@pytest.mark.parametrize("channel", ["depolarizing", "amplitude_damping", "dephasing"])
def test_weak_noise_linear_response(channel):
    """1 - F and 1 - purity are linear in the strength at weak noise (log-log slope 1)."""
    N, p = 4, 2
    theta = RNG.normal(size=n_params("hea", N, p))
    psi = simulate_statevector("hea", N, p, theta)
    out = []
    for s in (1e-6, 2e-6):
        rho = simulate_density_matrix("hea", N, p, theta, make_noise_model(channel, s))
        out.append((1 - np.vdot(psi, rho @ psi).real, 1 - purity(rho)))
    out = np.array(out)
    assert np.allclose(out[1] / out[0], 2.0, rtol=1e-4)


# ---------------------------------------------------------------------------
# T4.6 metrics
# ---------------------------------------------------------------------------
def test_pure_state_identities():
    N = 6
    psi = RNG.normal(size=2 ** N) + 1j * RNG.normal(size=2 ** N)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    m = mixed_state_metrics(rho, N)
    lam = np.linalg.svd(psi.reshape(2 ** (N // 2), -1), compute_uv=False) ** 2
    assert abs(m["S_total"]) < 1e-10 and np.isclose(m["purity"], 1)
    assert np.isclose(m["I"], 2 * m["S_A"], atol=1e-10)
    assert np.isclose(m["E_N"], 2 * np.log(np.sum(np.sqrt(lam))), atol=1e-10)
    assert np.isclose(m["S_cond"], -m["S_A"], atol=1e-10)
    assert np.allclose(m["I_profile"], 2 * m["S_A_profile"], atol=1e-10)


def test_product_state_has_no_correlations():
    a, b = random_rho(2), random_rho(2)
    m = mixed_state_metrics(np.kron(a, b), 4)
    assert abs(m["I"]) < 1e-12 and abs(m["E_N"]) < 1e-12


def test_partial_transpose_is_involution_and_trace_preserving():
    rho = random_rho(4)
    pt = partial_transpose_left(rho, 4, 2)
    assert np.isclose(np.trace(pt), 1) and np.allclose(partial_transpose_left(pt, 4, 2), rho)


# ---------------------------------------------------------------------------
# T4.2 shots
# ---------------------------------------------------------------------------
def test_estimator_exact_and_unbiased():
    N, h = 4, 0.8
    H, terms, _ = tfim(N, h)
    est = EnergyEstimator(terms, N)
    assert sorted(set(g.basis.values()).pop() for g in est.groups) == ["X", "Z"]
    rho = simulate_density_matrix("hea", N, 2, RNG.normal(size=24), make_noise_model("depolarizing", 0.02))
    E = np.real(np.trace(H @ rho))
    assert np.isclose(est.exact(rho), E, atol=1e-13)
    M, R = 1000, 4000
    s = est.sample(rho, M, np.random.default_rng(7), repetitions=R)
    sigma = np.sqrt(est.variance(rho, M))
    assert abs(s.mean() - E) < 4 * sigma / np.sqrt(R)
    assert abs(s.std() / sigma - 1) < 0.06


def test_estimator_general_paulis():
    from functools import reduce
    P = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
         "Z": np.diag([1, -1])}
    terms = [(0.3, {0: "X", 1: "Y"}), (-0.7, {2: "Y"}), (0.5, {}), (1.1, {0: "Z", 2: "Z"})]
    Hm = sum(c * reduce(np.kron, [P[o.get(q, "I")] for q in range(3)]) for c, o in terms)
    rho = random_rho(3)
    assert np.isclose(EnergyEstimator(terms, 3).exact(rho), np.trace(Hm @ rho).real, atol=1e-13)


# ---------------------------------------------------------------------------
# T4.3 SPSA
# ---------------------------------------------------------------------------
def test_spsa_noisy_quadratic():
    rng = np.random.default_rng(3)
    a = np.linspace(0.5, 2.0, 10)
    f = lambda x: float(np.sum(a * x ** 2) + 0.01 * rng.normal())
    res = spsa_minimize(f, np.ones(10), np.random.default_rng(4), maxiter=1000, a=0.2, c=0.1)
    assert np.linalg.norm(res.x) < 0.1


def _hva_problem(N=4, p=2, h=1.0):
    H, terms, E0 = tfim(N, h)
    f = lambda x: float(np.real(np.vdot(simulate_statevector("hva", N, p, x),
                                        H @ simulate_statevector("hva", N, p, x))))
    return f, E0


def test_spsa_warm_start_stays_at_optimum():
    """Calibrated at a cold point, the warm gain does not destroy a noiseless optimum."""
    f, E0 = _hva_problem()
    x_opt = minimize(f, np.full(4, 0.1), method="L-BFGS-B").x
    gain, _ = calibrate_gain(f, np.random.default_rng(0).normal(0, 0.1, 4), np.random.default_rng(1),
                             CFG.vqe.spsa_target_hva, CFG.vqe.spsa_c_hva, 500)
    res = spsa_minimize(f, x_opt, np.random.default_rng(2), maxiter=500, c=CFG.vqe.spsa_c_hva,
                        gain=gain * CFG.vqe.spsa_warm_factor)
    assert f(res.x) - f(x_opt) < 1e-4 * abs(E0)


def test_spsa_cold_start_reaches_small_problem_optimum():
    f, E0 = _hva_problem(p=1)
    e_opt = min(minimize(f, x0, method="L-BFGS-B").fun for x0 in ([0.1, 0.1], [0.5, 0.3], [-0.4, 0.6]))
    x0 = np.random.default_rng(5).normal(0, 0.1, 2)
    res = spsa_minimize(f, x0, np.random.default_rng(6), maxiter=500, a=CFG.vqe.spsa_target_hva,
                        c=CFG.vqe.spsa_c_hva)
    assert (f(res.x) - e_opt) / abs(E0) < 1e-3


# ---------------------------------------------------------------------------
# Validation: zero noise reproduces Phases 2 and 3 (needs the saved results)
# ---------------------------------------------------------------------------
needs_phase2 = pytest.mark.skipif(not (RESULTS_DIR / "phase2" / "vqe_hva_N4_p2.npz").exists(),
                                  reason="Phase 2 results not present")


@needs_phase2
@pytest.mark.parametrize("ansatz", ["hva", "hea"])
def test_zero_noise_reproduces_phase2_and_phase3(ansatz):
    from noise.data import exact_reference, load_phase2, load_phase3_half_entropy
    from noise.evaluate import evaluate_state
    N, p = 4, 2
    hs = CFG.tfim.h_grid(noisy=True)
    P2 = load_phase2(ansatz, N, p, hs)
    P3 = load_phase3_half_entropy(ansatz, N, p, hs)
    for i, h in enumerate(hs):
        ref = exact_reference(N, h, P2["ground_space"][i])
        assert np.array_equal(exact_reference(N, h).gs_mask, P2["ground_space"][i].astype(bool))
        assert np.isclose(ref.E0, P2["E0"][i], atol=1e-10)
        psi = np.asarray(P2["states"][i], dtype=complex)
        rho = simulate_density_matrix(ansatz, N, p, P2["params"][i], make_noise_model("depolarizing", 0.0))
        assert np.abs(rho - np.outer(psi, psi.conj())).max() < 1e-12
        m = evaluate_state(rho, ref, psi, N, profiles=False)
        assert abs(m["energy"] - P2["energy"][i]) < 1e-10
        assert abs(m["fidelity"] - P2["fidelity"][i]) < 1e-10
        assert abs(m["fidelity_gs"] - P2["fidelity_gs"][i]) < 1e-10
        if P3 is not None:
            assert abs(m["S_A"] - P3["S"][i]) < 1e-10 and abs(m["S2_A"] - P3["S2"][i]) < 1e-10


# ---------------------------------------------------------------------------
# PennyLane default.mixed cross-check of the channel conventions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("channel", ["depolarizing", "amplitude_damping", "dephasing"])
@pytest.mark.parametrize("ansatz", ["hva", "hea"])
def test_pennylane_default_mixed(channel, ansatz):
    qml = pytest.importorskip("pennylane")
    N, p, s = 4, 2, 0.03
    theta = RNG.normal(size=n_params(ansatz, N, p))
    nm = make_noise_model(channel, s)
    kraus_2q = kraus_depolarizing_2q(nm.depolarizing_2q)
    qchannel = getattr(qml, "QubitChannel", None)
    if channel == "depolarizing" and qchannel is None:
        pytest.skip("qml.QubitChannel not available for the two-qubit depolarizing channel")
    pl_gate = {"H": lambda w, m: qml.Hadamard(wires=w[0]),
               "CNOT": lambda w, m: qml.CNOT(wires=list(w))}
    seq = gate_sequence(ansatz, N, p, theta)
    angles = iter(_angles(ansatz, N, p, theta))
    dev = qml.device("default.mixed", wires=N)

    @qml.qnode(dev)
    def circuit():
        for name, wires, mat, _ in seq:
            if name in pl_gate:
                pl_gate[name](wires, mat)
            else:
                getattr(qml, name)(next(angles), wires=list(wires) if len(wires) > 1 else wires[0])
            if channel == "depolarizing":
                if len(wires) == 1:
                    qml.DepolarizingChannel(s, wires=wires[0])
                else:
                    qchannel(kraus_2q, wires=list(wires))
            else:
                op = qml.AmplitudeDamping if channel == "amplitude_damping" else qml.PhaseDamping
                for w in wires:
                    op(s, wires=w)
        return qml.density_matrix(wires=list(range(N)))

    ours = simulate_density_matrix(ansatz, N, p, theta, nm)
    assert np.abs(np.asarray(circuit()) - ours).max() < 1e-10


def _angles(ansatz, N, p, theta):
    """PennyLane rotation angles in gate_sequence order (H and CNOT take none)."""
    out = []
    if ansatz == "hva":
        for layer in range(p):
            out += [2 * theta[2 * layer]] * (N - 1) + [2 * theta[2 * layer + 1]] * N
    else:
        t = np.asarray(theta).reshape(p + 1, 2, N)
        for layer in range(p + 1):
            for q in range(N):
                out += [t[layer, 0, q], t[layer, 1, q]]
    return out
