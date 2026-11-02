"""
Phase 5 tests: hardware-like noise model, gate folding, readout calibration and
correction, extrapolators, and the energy pipeline. Every test checks an exact
identity or a known analytic result.

Run from the project root:  pytest -q tests\\test_phase5.py
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root, so the file also runs alone

import numpy as np
import pytest

from config import CFG
from models.hamiltonians import hamiltonian_terms
from noise.channels import NoiseModel, apply_unitary, make_noise_model, zero_state
from noise.data import energy as tr_energy
from noise.density import gate_sequence, simulate_density_matrix, simulate_statevector
from noise.mitigation import (
    FOLDINGS, PHASE5_CHANNELS, ReadoutEnergy, ReadoutModel, apply_local_matrices,
    calibration_matrices, correct_probabilities, extrapolate, extrapolation_variance, fold_gates,
    folded_gate_sequence, full_calibration_exact, inverse_transpose, kron_all, linear_coefficients,
    linear_two_point, make_mitigation_noise_model, noisy_gate_counts, prepared_one_probability,
    project_to_simplex, richardson_coefficients, sample_full_calibration,
    sample_tensored_calibration, simulate_folded, simulate_gates, variance_amplification,
)
from noise.shots import EnergyEstimator
from models.hamiltonians import build_hamiltonian

ANSATZE = ("hva", "hea")
SCALES = (1, 3, 5)


def _theta(ansatz, N, p, seed=0, scale=0.7):
    rng = np.random.default_rng(seed)
    n = 2 * p if ansatz == "hva" else 2 * N * (p + 1)
    return rng.normal(0.0, scale, n)


def _terms(N, h=1.0, J=None):
    J = CFG.tfim.J if J is None else J
    return hamiltonian_terms("tfim", N, J=J, h=h, boundary="open")


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------
def test_realistic_noise_model_parameters():
    s = 1e-3
    nm = make_mitigation_noise_model("realistic", s)
    cfg = CFG.mitigation
    assert nm.depolarizing == s
    assert nm.depolarizing_2q == pytest.approx(cfg.two_qubit_factor * s)
    assert nm.amplitude_damping == pytest.approx(cfg.realistic_damping_ratio * s)
    assert nm.dephasing == pytest.approx(cfg.realistic_dephasing_ratio * s)
    assert nm.noisy_rz == cfg.noisy_rz is False


@pytest.mark.parametrize("channel", PHASE5_CHANNELS)
def test_phase5_channels_use_phase5_settings(channel):
    nm = make_mitigation_noise_model(channel, 2e-3)
    assert nm.noisy_rz is False
    if channel == "depolarizing":
        assert nm.depolarizing_2q == pytest.approx(10 * 2e-3)
    assert make_mitigation_noise_model(channel, 0.0).is_noiseless


def test_phase4_model_unchanged():
    """Phase 4 settings are not touched by Phase 5 (reproducibility of Phase 4)."""
    nm = make_noise_model("depolarizing", 1e-3)
    assert nm.depolarizing_2q == pytest.approx(CFG.noise.two_qubit_factor * 1e-3)
    assert CFG.noise.two_qubit_factor == 1.0 and CFG.noise.noisy_rz is True


# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0, 2, 4, 1.5, -1])
def test_fold_rejects_bad_scales(bad):
    with pytest.raises(ValueError):
        fold_gates(gate_sequence("hva", 4, 1, _theta("hva", 4, 1)), bad)


@pytest.mark.parametrize("ansatz", ANSATZE)
def test_scale_one_is_the_original_circuit(ansatz):
    N, p = 4, 2
    th = _theta(ansatz, N, p)
    for method in FOLDINGS:
        assert len(fold_gates(gate_sequence(ansatz, N, p, th), 1, method)) == len(gate_sequence(ansatz, N, p, th))
    for nm in (make_noise_model("depolarizing", 3e-3), make_mitigation_noise_model("realistic", 3e-3)):
        r0 = simulate_density_matrix(ansatz, N, p, th, nm)
        r1 = simulate_folded(ansatz, N, p, th, nm, 1, "global")
        assert np.array_equal(r0, r1)


@pytest.mark.parametrize("ansatz,method,scale", list(itertools.product(ANSATZE, FOLDINGS, (3, 5))))
def test_noiseless_folding_is_identity(ansatz, method, scale):
    """U (U^dagger U)^k = U and G (G^dagger G)^k = G: folded circuits prepare the same state."""
    N, p = 4, 2
    th = _theta(ansatz, N, p)
    psi = simulate_statevector(ansatz, N, p, th)
    rho = simulate_folded(ansatz, N, p, th, None, scale, method)
    assert np.max(np.abs(rho - np.outer(psi, psi.conj()))) < 1e-12


@pytest.mark.parametrize("ansatz,method,scale", list(itertools.product(ANSATZE, FOLDINGS, SCALES)))
def test_folding_multiplies_noisy_gate_counts(ansatz, method, scale):
    N, p = 6, 2
    th = _theta(ansatz, N, p)
    for nm in (make_mitigation_noise_model("realistic", 1e-3), make_noise_model("depolarizing", 1e-3)):
        c1 = noisy_gate_counts(gate_sequence(ansatz, N, p, th), nm)
        cl = noisy_gate_counts(folded_gate_sequence(ansatz, N, p, th, scale, method), nm)
        assert cl["n_1q"] == scale * c1["n_1q"] and cl["n_2q"] == scale * c1["n_2q"]
    # virtual RZ: HEA has (p + 1) N noisy one-qubit gates, HVA N (p + 1)
    c = noisy_gate_counts(gate_sequence(ansatz, N, p, th), make_mitigation_noise_model("realistic", 1e-3))
    assert c["n_1q"] == N * (p + 1) and c["n_2q"] == p * (N - 1)


@pytest.mark.parametrize("method", FOLDINGS)
def test_product_circuit_depolarizing_analytic(method):
    """
    Depth-0 HEA (RY then virtual RZ on each qubit) under depolarizing noise:
    every folded copy adds one depolarizing channel per qubit and the channel
    commutes with unitaries, so Bloch vectors shrink by c^lambda, c = 1 - 4p/3.
    Then <X_i> -> c^lambda <X_i>, <Z_i Z_j> -> c^(2 lambda) <Z_i Z_j>, and
    exponential extrapolation of a one-term-type observable is exact.
    """
    N, p_dep = 4, 0.02
    th = _theta("hea", N, 0, seed=3, scale=1.0)
    psi = simulate_statevector("hea", N, 0, th)
    nm = make_mitigation_noise_model("depolarizing", p_dep)
    c = 1.0 - 4.0 * p_dep / 3.0
    est_zz = EnergyEstimator(_terms(N, h=0.0), N)
    est_x = EnergyEstimator(_terms(N, h=1.0, J=0.0), N)
    ideal = np.outer(psi, psi.conj())
    ezz0, ex0 = est_zz.exact(ideal), est_x.exact(ideal)
    ex = []
    for lam in SCALES:
        rho = simulate_folded("hea", N, 0, th, nm, lam, method)
        assert est_zz.exact(rho) == pytest.approx(c ** (2 * lam) * ezz0, abs=1e-12)
        assert est_x.exact(rho) == pytest.approx(c ** lam * ex0, abs=1e-12)
        ex.append(est_x.exact(rho))
    assert extrapolate(SCALES, ex, "exponential", asymptote=0.0) == pytest.approx(ex0, abs=1e-12)


@pytest.mark.parametrize("ansatz,method", list(itertools.product(ANSATZE, FOLDINGS)))
def test_first_order_noise_scales_with_lambda_depolarizing(ansatz, method):
    """
    To first order in p every gate error contributes once per folded copy; for
    depolarizing noise (unitarily covariant) the contribution of an error in the
    inverse copy equals that of the original, so dE(lambda) / dE(1) -> lambda.
    """
    N, p = 4, 2
    th = _theta(ansatz, N, p, seed=5)
    nm = make_mitigation_noise_model("depolarizing", 1e-7)
    est = EnergyEstimator(_terms(N), N)
    psi = simulate_statevector(ansatz, N, p, th)
    e_ideal = est.exact(np.outer(psi, psi.conj()))
    d1 = est.exact(simulate_folded(ansatz, N, p, th, nm, 1, method)) - e_ideal
    for lam in (3, 5):
        dl = est.exact(simulate_folded(ansatz, N, p, th, nm, lam, method)) - e_ideal
        assert dl / d1 == pytest.approx(lam, rel=1e-4)


def test_folding_bias_single_qubit_dephasing():
    """
    One RX(theta) gate followed by dephasing, observable Y. The error after RX^dagger acts on
    |0>, where dephasing does nothing, so to first order only the (lambda + 1)/2 forward copies
    contribute: dE(lambda) = s A (lambda + 1)/2. The zero-noise limit of folding is then
    E_ideal + s A / 2, so Richardson keeps half of the unmitigated error (folding bias).
    """
    from noise.density import rx
    theta, s = 0.9, 1e-7
    gates = [("RX", (0,), rx(theta), False)]
    nm = make_mitigation_noise_model("dephasing", s)
    est = EnergyEstimator([(1.0, {0: "Y"})], 1)
    e_id = est.exact(simulate_gates(gates, 1, None))
    assert e_id == pytest.approx(-np.sin(theta), abs=1e-14)
    dE = {lam: est.exact(simulate_gates(fold_gates(gates, lam), 1, nm)) - e_id for lam in SCALES}
    for lam in SCALES:
        assert dE[lam] / dE[1] == pytest.approx((lam + 1) / 2, rel=1e-5)
    rich = extrapolate(SCALES, [e_id + dE[lam] for lam in SCALES], "richardson")
    assert (rich - e_id) / dE[1] == pytest.approx(0.5, rel=1e-5)
    # depolarizing commutes with RX: no folding bias
    nm = make_mitigation_noise_model("depolarizing", s)
    dE = {lam: est.exact(simulate_gates(fold_gates(gates, lam), 1, nm)) - e_id for lam in SCALES}
    rich = extrapolate(SCALES, [e_id + dE[lam] for lam in SCALES], "richardson")
    assert abs(rich - e_id) < 1e-12


@pytest.mark.parametrize("ansatz,method,channel",
                         list(itertools.product(ANSATZE, FOLDINGS, ("amplitude_damping", "dephasing", "realistic"))))
def test_folding_bias_is_half_a_final_noise_layer(ansatz, method, channel):
    """
    Channels applied to every touched qubit after each noisy gate (amplitude damping, dephasing)
    telescope along each qubit, so the first-order zero-noise limit of folding is the ideal circuit
    followed by half a layer of that noise; the depolarizing part of "realistic" contributes nothing.
    """
    from noise.channels import amplitude_damp, dephase
    N, p, s = 4, 2, 1e-7
    th = _theta(ansatz, N, p, seed=11)
    nm = make_mitigation_noise_model(channel, s)
    est = EnergyEstimator(_terms(N), N)
    rho_id = simulate_folded(ansatz, N, p, th, None, 1)
    e_id = est.exact(rho_id)
    layer = rho_id.copy()
    for q in range(N):
        layer = amplitude_damp(layer, q, nm.amplitude_damping, N)
        layer = dephase(layer, q, nm.dephasing, N)
    pred = 0.5 * (est.exact(layer) - e_id)
    E = [est.exact(simulate_folded(ansatz, N, p, th, nm, lam, method)) for lam in SCALES]
    assert extrapolate(SCALES, E, "richardson") - e_id == pytest.approx(pred, rel=1e-4)


@pytest.mark.parametrize("channel", PHASE5_CHANNELS)
def test_folded_states_are_valid_density_matrices(channel):
    N, p = 4, 2
    th = _theta("hea", N, p)
    nm = make_mitigation_noise_model(channel, 5e-3)
    for method in FOLDINGS:
        rho = simulate_folded("hea", N, p, th, nm, 5, method)
        assert abs(np.trace(rho) - 1) < 1e-12
        assert np.max(np.abs(rho - rho.conj().T)) < 1e-14
        assert np.linalg.eigvalsh(rho).min() > -1e-12


# ---------------------------------------------------------------------------
# Readout model and calibration
# ---------------------------------------------------------------------------
def test_readout_matrix_structure_and_ordering():
    N = 3
    ro = ReadoutModel([0.1, 0.0, 0.0], [0.0, 0.0, 0.0])
    A = ro.matrix()
    assert np.allclose(A.sum(axis=0), 1.0)
    # a flip on wire 0 (most significant bit) moves |000> to |100> = index 4
    assert A[4, 0] == pytest.approx(0.1) and A[1, 0] == 0.0 and A[2, 0] == 0.0


def test_local_matrices_equal_kronecker_product():
    rng = np.random.default_rng(2)
    N = 4
    mats = rng.random((N, 2, 2))
    v = rng.random((3, 2 ** N))
    assert np.allclose(apply_local_matrices(v, mats), v @ kron_all(mats).T, atol=1e-14)


def test_calibration_without_gate_noise_equals_readout_matrix():
    ro = ReadoutModel.from_config(4)
    assert np.allclose(full_calibration_exact(ro, None), ro.matrix(), atol=1e-15)
    assert np.allclose(full_calibration_exact(ro, make_mitigation_noise_model("realistic", 0.0)),
                       ro.matrix(), atol=1e-15)


@pytest.mark.parametrize("channel", PHASE5_CHANNELS)
def test_calibration_matches_density_matrix_simulation(channel):
    """The product-form calibration equals simulating every calibration circuit."""
    N = 3
    ro = ReadoutModel.from_config(N)
    nm = make_mitigation_noise_model(channel, 0.02)
    A = full_calibration_exact(ro, nm)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    for j in range(2 ** N):
        rho = zero_state(N)
        for q in range(N):
            if (j >> (N - 1 - q)) & 1:
                rho = apply_unitary(rho, X, [q], N)
                rho = nm.after_gate(rho, "X", [q], N)
        col = ro.matrix() @ np.real(np.diag(rho))
        assert np.allclose(A[:, j], col, atol=1e-14)


def test_prepared_one_probability_values():
    s = 0.03
    assert prepared_one_probability(make_mitigation_noise_model("depolarizing", s)) == pytest.approx(1 - 2 * s / 3)
    assert prepared_one_probability(make_mitigation_noise_model("amplitude_damping", s)) == pytest.approx(1 - s)
    assert prepared_one_probability(make_mitigation_noise_model("dephasing", s)) == pytest.approx(1.0)
    assert prepared_one_probability(None) == 1.0


def test_inversion_recovers_ideal_distribution():
    N = 4
    rng = np.random.default_rng(4)
    ro = ReadoutModel(tuple(rng.uniform(0, 0.05, N)), tuple(rng.uniform(0, 0.08, N)))
    p = rng.dirichlet(np.ones(2 ** N), size=2)
    q = apply_local_matrices(p, ro.matrices())
    assert np.allclose(correct_probabilities(q, ro.matrix()), p, atol=1e-13)
    assert np.allclose(correct_probabilities(q, local_mats=ro.matrices()), p, atol=1e-13)
    assert np.allclose(correct_probabilities(q, ro.matrix(), "projected"), p, atol=1e-13)


def test_readout_analytic_energy():
    """
    Independent flips: <Z>_meas = a + b <Z>, <Z_i Z_j>_meas = a^2 + a b (<Z_i> + <Z_j>) + b^2 <Z_i Z_j>,
    a = p10 - p01, b = 1 - p01 - p10 (X terms: the same, after the basis rotation).
    """
    N, p = 4, 2
    cfg = CFG.mitigation
    a, b = cfg.readout_p10 - cfg.readout_p01, 1 - cfg.readout_p01 - cfg.readout_p10
    th = _theta("hea", N, p, seed=7)
    psi = simulate_statevector("hea", N, p, th)
    rho = np.outer(psi, psi.conj())
    pipe = ReadoutEnergy(_terms(N, h=0.8), N, ReadoutModel.from_config(N))
    e_meas = pipe.energy(pipe.measured_probabilities(pipe.ideal_probabilities(rho)))

    def ev(ops):
        return EnergyEstimator([(1.0, ops)], N).exact(rho)

    expected = 0.0
    for i in range(N - 1):
        zi, zj, zz = ev({i: "Z"}), ev({i + 1: "Z"}), ev({i: "Z", i + 1: "Z"})
        expected += -CFG.tfim.J * (a * a + a * b * (zi + zj) + b * b * zz)
    for i in range(N):
        expected += -0.8 * (a + b * ev({i: "X"}))
    assert e_meas == pytest.approx(expected, abs=1e-12)


def test_spam_bias_single_qubit():
    """Calibration with a noisy X gate: <Z>_corr = p0 - p1 (2 - f) / f (first-order bias -(1-f)(1-<Z>))."""
    s = 0.02
    nm = make_mitigation_noise_model("realistic", s)
    ro = ReadoutModel.uniform(1, 0.01, 0.02)
    f = prepared_one_probability(nm)
    pipe = ReadoutEnergy([(1.0, {0: "Z"})], 1, ro)
    rho = np.array([[0.7, 0.2], [0.2, 0.3]], dtype=complex)
    P = pipe.ideal_probabilities(rho)
    Q = pipe.measured_probabilities(P)
    w = pipe.weights_from_inverse_T(inverse_transpose(full_calibration_exact(ro, nm)))
    p0, p1 = 0.7, 0.3
    assert pipe.energy(Q, w) == pytest.approx(p0 - p1 * (2 - f) / f, abs=1e-14)


def test_tensored_equals_full_in_exact_limit():
    N = 4
    ro = ReadoutModel.from_config(N)
    nm = make_mitigation_noise_model("realistic", 4e-3)
    C = calibration_matrices(ro, nm)
    pipe = ReadoutEnergy(_terms(N), N, ro)
    w_full = pipe.weights_from_inverse_T(inverse_transpose(kron_all(C)))
    w_tens = pipe.weights_tensored(C)
    assert np.allclose(w_full, w_tens, atol=1e-12)


def test_sampled_calibrations_are_stochastic_and_unbiased():
    N = 3
    ro = ReadoutModel.from_config(N)
    nm = make_mitigation_noise_model("realistic", 1e-2)
    C = calibration_matrices(ro, nm)
    A = kron_all(C)
    rng = np.random.default_rng(11)
    A_hat = sample_full_calibration(A, 20000, rng, 400)
    assert np.allclose(A_hat.sum(axis=1), 1.0)
    se = np.sqrt(A * (1 - A) / 20000 / 400)
    assert np.all(np.abs(A_hat.mean(axis=0) - A) <= 5 * se + 1e-15)
    C_hat = sample_tensored_calibration(C, 20000, rng, 400)
    assert np.allclose(C_hat.sum(axis=2), 1.0)
    se = np.sqrt(C * (1 - C) / 20000 / 400)
    assert np.all(np.abs(C_hat.mean(axis=0) - C) <= 5 * se + 1e-15)


def test_project_to_simplex():
    v = np.array([[0.5, 0.7, -0.1, -0.1], [0.25, 0.25, 0.25, 0.25]])
    out = project_to_simplex(v)
    assert np.allclose(out[0], [0.4, 0.6, 0.0, 0.0])
    assert np.allclose(out[1], v[1])
    rng = np.random.default_rng(1)
    w = project_to_simplex(rng.normal(size=(20, 16)))
    assert np.allclose(w.sum(axis=1), 1.0) and np.all(w >= 0)


# ---------------------------------------------------------------------------
# Extrapolators
# ---------------------------------------------------------------------------
def test_richardson_coefficients_135():
    c = richardson_coefficients(SCALES)
    assert np.allclose(c, [15 / 8, -5 / 4, 3 / 8])
    assert c.sum() == pytest.approx(1.0)
    assert variance_amplification(SCALES, "richardson") == pytest.approx(167 / 32)


def test_richardson_exact_for_quadratics_and_linear_for_lines():
    lam = np.array(SCALES, float)
    quad = 1.3 - 0.4 * lam + 0.05 * lam ** 2
    assert extrapolate(lam, quad, "richardson") == pytest.approx(1.3, abs=1e-13)
    line = -2.0 + 0.3 * lam
    assert extrapolate(lam, line, "linear") == pytest.approx(-2.0, abs=1e-13)
    assert extrapolate(lam, line, "richardson") == pytest.approx(-2.0, abs=1e-13)


def test_two_point_formula_matches_linear_and_richardson():
    e1, e3 = -4.1, -3.7
    star = linear_two_point(1.0, e1, 3.0, e3)
    assert star == pytest.approx((3 * e1 - 1 * e3) / 2)
    assert extrapolate([1, 3], [e1, e3], "linear") == pytest.approx(star)
    assert extrapolate([1, 3], [e1, e3], "richardson") == pytest.approx(star)
    assert np.allclose(linear_coefficients([1, 3]), richardson_coefficients([1, 3]))


def test_exponential_extrapolation():
    lam = np.array(SCALES, float)
    y = 0.2 - 3.0 * np.exp(-0.15 * lam)
    assert extrapolate(lam, y, "exponential", asymptote=0.2) == pytest.approx(-2.8, abs=1e-12)
    assert np.isnan(extrapolate(lam, [1.0, -1.0, 0.5], "exponential", asymptote=0.0))


def test_extrapolate_broadcasts_over_axes():
    rng = np.random.default_rng(0)
    vals = rng.normal(size=(4, 3, 5))
    out = extrapolate(SCALES, vals, "richardson", axis=1)
    assert out.shape == (4, 5)
    assert np.allclose(out, np.einsum("akb,k->ab", vals, richardson_coefficients(SCALES)))
    asym = rng.normal(size=(4, 5)) - 10
    e = extrapolate(SCALES, vals, "exponential", axis=1, asymptote=asym)
    assert e.shape == (4, 5)


def test_extrapolation_variance_empirical():
    rng = np.random.default_rng(8)
    sig = np.array([0.01, 0.02, 0.03])
    samples = rng.normal(0.0, sig, size=(200000, 3))
    emp = extrapolate(SCALES, samples, "richardson").var()
    pred = extrapolation_variance(SCALES, sig ** 2, "richardson")
    assert emp == pytest.approx(pred, rel=0.02)


def test_pennylane_extrapolators_agree():
    qml = pytest.importorskip("pennylane")
    mod = None
    for cand in ("noise", "transforms"):
        m = getattr(qml, cand, None)
        if m is not None and hasattr(m, "richardson_extrapolate") and hasattr(m, "poly_extrapolate"):
            mod = m
            break
    if mod is None:
        pytest.skip("PennyLane extrapolation functions not found")
    x = np.array(SCALES, float)
    y = np.array([-9.1, -8.2, -7.6])
    assert float(mod.richardson_extrapolate(x, y)) == pytest.approx(extrapolate(x, y, "richardson"), abs=1e-10)
    assert float(mod.poly_extrapolate(x, y, 1)) == pytest.approx(extrapolate(x, y, "linear"), abs=1e-10)


# ---------------------------------------------------------------------------
# Energy pipeline
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ansatz", ANSATZE)
def test_pipeline_with_ideal_readout_is_tr_rho_h(ansatz):
    N, p = 4, 2
    th = _theta(ansatz, N, p)
    rho = simulate_folded(ansatz, N, p, th, make_mitigation_noise_model("realistic", 2e-3), 3)
    pipe = ReadoutEnergy(_terms(N, 1.2), N, ReadoutModel.uniform(N, 0.0, 0.0))
    P = pipe.ideal_probabilities(rho)
    H = build_hamiltonian("tfim", N, J=CFG.tfim.J, h=1.2, boundary="open")
    assert pipe.energy(P) == pytest.approx(tr_energy(rho, H), abs=1e-12)
    assert pipe.energy(pipe.measured_probabilities(P)) == pytest.approx(tr_energy(rho, H), abs=1e-12)
    assert pipe.shot_variance(P, 1000) == pytest.approx(pipe.estimator.variance(rho, 1000), rel=1e-12)


def test_readout_correction_exact_at_zero_gate_noise():
    N, p = 4, 2
    th = _theta("hva", N, p)
    psi = simulate_statevector("hva", N, p, th)
    ro = ReadoutModel.from_config(N)
    pipe = ReadoutEnergy(_terms(N), N, ro)
    P = pipe.ideal_probabilities(np.outer(psi, psi.conj()))
    Q = pipe.measured_probabilities(P)
    w = pipe.weights_from_inverse_T(inverse_transpose(full_calibration_exact(ro, None)))
    assert abs(pipe.energy(Q) - pipe.energy(P)) > 1e-2          # readout bias is visible
    assert pipe.energy(Q, w) == pytest.approx(pipe.energy(P), abs=1e-12)


def test_counts_energy_and_unbiasedness():
    N, p = 4, 1
    th = _theta("hea", N, p)
    rho = simulate_folded("hea", N, p, th, make_mitigation_noise_model("realistic", 1e-3), 1)
    ro = ReadoutModel.from_config(N)
    pipe = ReadoutEnergy(_terms(N), N, ro)
    Q = pipe.measured_probabilities(pipe.ideal_probabilities(rho))
    w = pipe.weights_from_inverse_T(inverse_transpose(ro.matrix()))
    rng = np.random.default_rng(3)
    R, M = 4000, 2000
    counts = pipe.sample_counts(Q, M, rng, R)                            # (R, G, d)
    e = pipe.energy_from_counts(counts, M, w)
    loop = np.array([pipe.constant + np.sum(w * c) / M for c in counts])
    assert np.allclose(e, loop, atol=1e-12)
    se = np.sqrt(pipe.shot_variance(Q, M, w) / R)
    assert abs(e.mean() - pipe.energy(Q, w)) < 4 * se
    assert e.var() == pytest.approx(pipe.shot_variance(Q, M, w), rel=0.1)
    # per-repetition weights broadcast correctly
    wr = np.broadcast_to(w, (R,) + w.shape)
    assert np.allclose(pipe.energy_from_counts(counts, M, wr), e, atol=1e-12)


def test_mixed_state_energy_is_trace_over_dimension():
    N = 4
    ro = ReadoutModel.from_config(N)
    pipe = ReadoutEnergy(_terms(N), N, ro)
    assert pipe.mixed_state_energy(readout=False) == pytest.approx(0.0, abs=1e-14)   # TFIM is traceless
    w = pipe.weights_from_inverse_T(inverse_transpose(ro.matrix()))
    assert pipe.mixed_state_energy(w, readout=True) == pytest.approx(0.0, abs=1e-12)
    rho = np.eye(2 ** N, dtype=complex) / 2 ** N
    assert pipe.mixed_state_energy(readout=True) == pytest.approx(
        pipe.energy(pipe.measured_probabilities(pipe.ideal_probabilities(rho))), abs=1e-13)


def test_simulate_gates_matches_density_module():
    N, p = 4, 2
    th = _theta("hva", N, p)
    nm = make_mitigation_noise_model("amplitude_damping", 1e-2)
    assert np.array_equal(simulate_gates(gate_sequence("hva", N, p, th), N, nm),
                          simulate_density_matrix("hva", N, p, th, nm))
