"""
Phase 2 validation tests. Run from the project root with:  pytest -q tests\\test_phase2.py

Every test compares the VQE code with an independent exact result:
dense Kronecker-product circuits, PennyLane (state vectors, expectation
values, backprop and parameter-shift gradients), finite differences,
closed-form limits, exact diagonalization, and two known exactness results
for the TFIM HVA (N = 2 at p = 1; periodic chains at p = N/2).
The PennyLane tests are skipped automatically if PennyLane is missing.
"""
import sys
from functools import reduce
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from scipy.linalg import expm

from config import CFG
from models.exact import solve_model
from models.hamiltonians import build_hamiltonian, pauli_string_sparse
from vqe import statevector as sv
from vqe.ansatz import HEA, HVA, make_ansatz
from vqe.gradients import (finite_difference_gradient, fit_exponential_decay,
                           gradient_variance, parameter_shift_gradient, shift_points)
from vqe.metrics import ed_reference, evaluate_sweep, ground_space_mask, parity_expectation
from vqe.optimize import field_sweep, minimize, optimize_point, point_rng, random_candidates

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
P0, P1 = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])


def op(n, q, m):
    return reduce(np.kron, [m if k == q else I2 for k in range(n)])


def tfim(n, h, boundary="open"):
    return build_hamiltonian("tfim", n, J=1.0, h=h, boundary=boundary)


def rel_error(energy, n, h, boundary="open"):
    e0 = solve_model("tfim", n, J=1.0, h=h, boundary=boundary).sector_energies[:, 0].min()
    return (energy - e0) / abs(e0)


# ---------------------------------------------------------------------------
# Dense reference circuits (Kronecker products and matrix exponentials)
# ---------------------------------------------------------------------------
def hva_dense(n, p, theta, boundary="open"):
    nb = n - 1 if boundary == "open" else n
    zz = sum(op(n, i, Z) @ op(n, (i + 1) % n, Z) for i in range(nb))
    xs = sum(op(n, i, X) for i in range(n))
    psi = np.ones(2 ** n, complex) / 2 ** (n / 2)
    for g, b in np.reshape(theta, (p, 2)):
        psi = expm(-1j * b * xs) @ (expm(-1j * g * zz) @ psi)
    return psi


def hea_dense(n, p, theta):
    th = np.reshape(theta, (p + 1, 2, n))
    psi = np.zeros(2 ** n, complex)
    psi[0] = 1
    for layer in range(p + 1):
        if layer > 0:
            for c in range(n - 1):
                psi = (op(n, c, P0) + op(n, c, P1) @ op(n, c + 1, X)) @ psi
        for q in range(n):
            psi = expm(-0.5j * th[layer, 0, q] * op(n, q, Y)) @ psi
        for q in range(n):
            psi = expm(-0.5j * th[layer, 1, q] * op(n, q, Z)) @ psi
    return psi


# ---------------------------------------------------------------------------
# T2.1 / T2.2  Circuits
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n, p", [(2, 1), (3, 2), (4, 3), (5, 2)])
def test_hva_state_matches_dense_circuit(n, p):
    a = HVA(n, p)
    theta = point_rng(901, n, p).uniform(0, 2 * np.pi, a.n_params)
    assert np.allclose(a.state(theta), hva_dense(n, p, theta), atol=1e-12)


def test_periodic_hva_state_matches_dense_circuit():
    a = HVA(4, 2, boundary="periodic")
    theta = point_rng(902).uniform(0, 2 * np.pi, a.n_params)
    assert np.allclose(a.state(theta), hva_dense(4, 2, theta, "periodic"), atol=1e-12)


@pytest.mark.parametrize("n, p", [(2, 1), (3, 2), (4, 3), (5, 1)])
def test_hea_state_matches_dense_circuit(n, p):
    a = HEA(n, p)
    theta = point_rng(903, n, p).uniform(0, 2 * np.pi, a.n_params)
    assert np.allclose(a.state(theta), hea_dense(n, p, theta), atol=1e-12)


def test_parameter_counts():
    assert HVA(8, 5).n_params == 10
    assert HEA(8, 5).n_params == 2 * 8 * 6


def test_cnot_chain_permutation_is_unitary_inverse():
    n = 6
    psi = point_rng(904).normal(size=2 ** n) + 0j
    out = sv.apply_cnot_chain(sv.apply_cnot_chain(psi, n), n, inverse=True)
    assert np.allclose(out, psi)


def test_reference_states_at_zero_parameters():
    n, h = 6, 0.7
    H = tfim(n, h)
    # <+|H|+> = -h N and <0|H|0> = -J (N - 1)
    assert np.isclose(HVA(n, 3).energy(np.zeros(6), H), -h * n)
    assert np.isclose(HEA(n, 2).energy(np.zeros(HEA(n, 2).n_params), H), -(n - 1))


def test_hva_identity_padding_preserves_state():
    a2, a4 = HVA(6, 2), HVA(6, 4)
    theta = point_rng(905).normal(size=4)
    assert np.allclose(a2.state(theta), a4.state(HVA.pad(theta, 4)))


def test_hva_conserves_parity_and_hea_does_not():
    n = 6
    th = point_rng(906).uniform(0, 2 * np.pi, HVA(n, 3).n_params)
    assert np.isclose(parity_expectation(HVA(n, 3).state(th)), 1.0, atol=1e-12)
    th = point_rng(907).uniform(0, 2 * np.pi, HEA(n, 2).n_params)
    assert abs(parity_expectation(HEA(n, 2).state(th))) < 0.99


# ---------------------------------------------------------------------------
# T2.3  Gradients
# ---------------------------------------------------------------------------
def test_shift_rule_is_exact_on_trigonometric_polynomials():
    rng = point_rng(908)
    for order in (1, 2, 5, 11):
        a, b = rng.normal(size=order), rng.normal(size=order)
        k = np.arange(1, order + 1)
        f = lambda x: 0.3 + np.sum(a * np.cos(k * x) + b * np.sin(k * x))
        x, c = shift_points(order)
        assert np.isclose(sum(ci * f(xi) for xi, ci in zip(x, c)), np.sum(k * b), atol=1e-12)


@pytest.mark.parametrize("name", ["hva", "hea"])
def test_adjoint_gradient_matches_finite_differences(name):
    n, p = 5, 3
    a = make_ansatz(name, n, p)
    H = tfim(n, 0.8)
    theta = point_rng(909).uniform(0, 2 * np.pi, a.n_params)
    _, g = a.energy_and_grad(theta, H)
    assert np.allclose(g, finite_difference_gradient(a, theta, H), atol=1e-7)


@pytest.mark.parametrize("name, boundary", [("hva", "open"), ("hva", "periodic"), ("hea", "open")])
def test_adjoint_gradient_matches_parameter_shift(name, boundary):
    n, p = 5, 2
    kwargs = {"boundary": boundary} if name == "hva" else {}
    a = make_ansatz(name, n, p, **kwargs)
    H = tfim(n, 0.6, boundary)
    theta = point_rng(910).uniform(0, 2 * np.pi, a.n_params)
    _, g = a.energy_and_grad(theta, H)
    assert np.allclose(g, parameter_shift_gradient(a, theta, H), atol=1e-11)


def test_adjoint_energy_equals_direct_energy():
    a = HEA(6, 3)
    H = tfim(6, 1.2)
    theta = point_rng(911).uniform(0, 2 * np.pi, a.n_params)
    assert np.isclose(a.energy_and_grad(theta, H)[0], a.energy(theta, H))


# ---------------------------------------------------------------------------
# PennyLane cross-checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name, n, p", [("hva", 4, 2), ("hva", 5, 3), ("hea", 4, 2), ("hea", 5, 1)])
def test_states_match_pennylane(name, n, p):
    pytest.importorskip("pennylane")
    from vqe.pennylane_ref import pennylane_state
    a = make_ansatz(name, n, p)
    theta = point_rng(912, n, p).uniform(0, 2 * np.pi, a.n_params)
    assert np.allclose(a.state(theta), pennylane_state(a, theta), atol=1e-10)


@pytest.mark.parametrize("name", ["hva", "hea"])
def test_energy_and_gradient_match_pennylane_parameter_shift(name):
    pytest.importorskip("pennylane")
    from vqe.pennylane_ref import pennylane_energy_and_grad, pennylane_hamiltonian
    n, p, h = 4, 2, 0.9
    a = make_ansatz(name, n, p)
    theta = point_rng(913).uniform(0, 2 * np.pi, a.n_params)
    H_pl = pennylane_hamiltonian("tfim", n, J=1.0, h=h, boundary="open")
    e_pl, g_pl = pennylane_energy_and_grad(a, H_pl, theta, diff_method="parameter-shift")
    e, g = a.energy_and_grad(theta, tfim(n, h))
    assert np.isclose(e, e_pl, atol=1e-10)
    assert np.allclose(g, g_pl, atol=1e-8)


def test_gradient_matches_pennylane_backprop():
    pytest.importorskip("pennylane")
    from vqe.pennylane_ref import pennylane_energy_and_grad, pennylane_hamiltonian
    a = HEA(4, 3)
    theta = point_rng(914).uniform(0, 2 * np.pi, a.n_params)
    H_pl = pennylane_hamiltonian("tfim", 4, J=1.0, h=1.3, boundary="open")
    _, g_pl = pennylane_energy_and_grad(a, H_pl, theta, diff_method="backprop")
    assert np.allclose(a.energy_and_grad(theta, tfim(4, 1.3))[1], g_pl, atol=1e-10)


# ---------------------------------------------------------------------------
# Optimization against exact results
# ---------------------------------------------------------------------------
def test_variational_principle_at_random_parameters():
    n = 6
    for h in (0.3, 1.0, 1.8):
        H = tfim(n, h)
        e0 = solve_model("tfim", n, J=1.0, h=h).ground_energy
        for name in ("hva", "hea"):
            a = make_ansatz(name, n, 3)
            for theta in point_rng(915, int(10 * h)).uniform(0, 2 * np.pi, (5, a.n_params)):
                assert a.energy(theta, H) >= e0 - 1e-12


@pytest.mark.parametrize("h", [0.3, 1.0, 2.0])
def test_two_site_hva_is_exact_at_depth_one(h):
    """Even sector of N = 2 is two-dimensional: one HVA layer suffices."""
    a = HVA(2, 1)
    best = optimize_point(a, tfim(2, h), random_candidates(a, 4, point_rng(916))).best
    assert rel_error(best.energy, 2, h) < 1e-12


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("h", [0.5, 1.0, 1.5])
def test_periodic_hva_is_exact_at_half_depth(n, h):
    """Known result: the periodic TFIM HVA prepares the exact ground state at p = N/2."""
    a = HVA(n, n // 2, boundary="periodic")
    H = tfim(n, h, "periodic")
    best = optimize_point(a, H, random_candidates(a, CFG.vqe.n_random_inits, point_rng(917, n))).best
    assert rel_error(best.energy, n, h, "periodic") < 1e-9


@pytest.mark.parametrize("h", [0.5, 1.0, 1.5])
def test_open_hva_reaches_unit_fidelity_at_n4(h):
    """Plan check: at N = 4 the HVA reaches near-unit fidelity at modest depth."""
    n = 4
    ref = solve_model("tfim", n, J=1.0, h=h)
    a = HVA(n, n)
    best = optimize_point(a, tfim(n, h), random_candidates(a, CFG.vqe.n_random_inits,
                                                           point_rng(918))).best
    fid = abs(np.vdot(ref.ground_state, a.state(best.x))) ** 2
    assert fid > 1 - 1e-6


def test_adam_and_lbfgs_find_the_same_minimum():
    a = HVA(4, 2)
    H = tfim(4, 0.8)
    x0 = a.random_params(point_rng(919), CFG.vqe.init_scale)
    e1 = minimize(a, H, x0, method="L-BFGS-B").energy
    e2 = minimize(a, H, x0, method="Adam", maxiter=3000).energy
    assert abs(e1 - e2) < 1e-6


def test_hea_breaks_symmetry_in_ordered_phase():
    """
    Deep in the ordered phase the HEA finds a symmetry-broken state:
    overlap 1/2 with the even cat state but full ground-space fidelity.
    """
    n, h = 6, 0.3
    ref = ed_reference(n, np.array([h]), verbose=False)
    a = HEA(n, 2)        # shallow on purpose: relative error ~ 4e-4 at p = 2
    best = optimize_point(a, tfim(n, h), random_candidates(a, 4, point_rng(920))).best
    m = evaluate_sweep(a, best.x[None, :], np.array([best.energy]), ref)
    assert m["ground_space"][0].all()
    assert abs(m["fidelity"][0] - 0.5) < 0.05
    assert m["fidelity_gs"][0] > 0.99
    assert m["rel_error"][0] < 1e-3


def test_ground_space_excludes_odd_sector_in_paramagnet():
    ref = ed_reference(8, np.array([0.3, 1.8]), verbose=False)
    mask = ground_space_mask(ref)
    assert mask[0].all() and mask[1].tolist() == [True, False]


def test_warm_started_sweep_is_exact_at_n4():
    """
    At p = N = 4 roughly half of all random starts end in a local minimum
    (T2.4). Once the first point is found, warm starting alone (T2.5, no
    further restarts) carries the exact solution over the whole field grid.
    The step matters: with steps of 0.5 instead of 0.05 the warm start loses
    the exact branch when it crosses into the ordered phase.
    """
    n, p = 4, 4
    h = CFG.tfim.h_grid()
    a = HVA(n, p)
    out = field_sweep(a, [tfim(n, x) for x in h], "descending", (921,),
                      n_restarts_first=CFG.vqe.n_random_inits, n_restarts=0)
    errors = np.array([rel_error(e, n, x) for e, x in zip(out["energy"], h)])
    assert errors.max() < 1e-10
    order = np.arange(len(h))[::-1]
    assert (out["best_source"][order[1:]] == 0).all()   # later points kept the warm start


def test_point_rng_is_reproducible_and_key_dependent():
    a = point_rng(1, 2, 3).normal(size=5)
    assert np.allclose(a, point_rng(1, 2, 3).normal(size=5))
    assert not np.allclose(a, point_rng(1, 2, 4).normal(size=5))


# ---------------------------------------------------------------------------
# T2.7  Barren plateaus (small, fast version of the full experiment)
# ---------------------------------------------------------------------------
def test_hea_gradient_variance_decays_like_two_design():
    sizes = [4, 6, 8]
    var = []
    for n in sizes:
        c = n // 2 - 1
        obs = pauli_string_sparse(n, {c: "Z", c + 1: "Z"})
        var.append(gradient_variance(HEA(n, n), obs, 150, point_rng(922, n))["var_mean"])
    fit = fit_exponential_decay(sizes, var)
    assert abs(fit["b"] - np.log(2)) < 0.15
