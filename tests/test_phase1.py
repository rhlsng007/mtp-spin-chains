"""
Phase 1 validation tests. Run from the project root with:  pytest -q

Every test compares the ED code with an independent exact result:
Kronecker-product matrices, dense diagonalization, free-fermion solutions,
closed-form limits, or conformal field theory predictions.
"""
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import scipy.sparse as sp

from config import make_grid
from models.hamiltonians import (apply_spin_flip, build_hamiltonian, hamiltonian_terms,
                                 pauli_string_sparse, symmetry_generators)
from models.exact import (dense_spectrum, lowest_eigenpairs, solve, solve_model,
                          symmetry_sectors)
from analysis import free_fermion as ff
from analysis.entanglement import (calabrese_cardy_fit, chord_log, entanglement_entropy,
                                   entropy_from_spectrum, entropy_profile, half_chain_entropy,
                                   reduced_density_matrix, von_neumann_entropy_dm)
from analysis.observables import (binder_cumulant, expval_z, magnetization_moments,
                                  pauli_expectation, string_order_parameter,
                                  zz_correlation_matrix)

LN2 = np.log(2.0)
PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0, -1.0])}
MODEL_PARAMS = {"tfim": {"J": 1.0, "h": 0.7},
                "xxz": {"J": 1.0, "delta": 0.6},
                "cluster": {"J": 1.0, "h1": 0.4, "h2": 0.3}}


def kron_string(word: str) -> np.ndarray:
    out = np.array([[1.0]])
    for letter in word:
        out = np.kron(out, PAULI[letter])
    return out


def random_state(n: int, seed: int = 0, complex_: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(2 ** n) + (1j * rng.standard_normal(2 ** n) if complex_ else 0)
    return psi / np.linalg.norm(psi)


def ghz(n: int) -> np.ndarray:
    psi = np.zeros(2 ** n)
    psi[0] = psi[-1] = 1 / np.sqrt(2)
    return psi


# ---------------------------------------------------------------------------
# T1.1  Hamiltonian construction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("word", ["ZI", "IZ", "XYZ", "YIX", "ZZYX", "YYYY"])
def test_pauli_strings_match_kronecker_products(word):
    ops = {i: c for i, c in enumerate(word) if c != "I"}
    assert np.allclose(pauli_string_sparse(len(word), ops).toarray(), kron_string(word))


@pytest.mark.parametrize("model", ["tfim", "xxz", "cluster"])
@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_hamiltonian_matches_kronecker_sum(model, boundary):
    n = 5
    terms = hamiltonian_terms(model, n, boundary=boundary, **MODEL_PARAMS[model])
    ref = sum(c * kron_string("".join(ops.get(i, "I") for i in range(n))) for c, ops in terms)
    H = build_hamiltonian(model, n, boundary=boundary, **MODEL_PARAMS[model])
    assert np.allclose(H.toarray(), ref)


@pytest.mark.parametrize("model", ["tfim", "xxz", "cluster"])
@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_hamiltonians_are_real_symmetric(model, boundary):
    H = build_hamiltonian(model, 6, boundary=boundary, **MODEL_PARAMS[model])
    assert H.dtype == np.float64
    assert abs(H - H.T).max() < 1e-14


def test_bond_counts():
    count = lambda terms, k: sum(len(ops) == k for _, ops in terms)
    assert count(hamiltonian_terms("tfim", 6, J=1, h=1, boundary="open"), 2) == 5
    assert count(hamiltonian_terms("tfim", 6, J=1, h=1, boundary="periodic"), 2) == 6
    assert count(hamiltonian_terms("cluster", 6, J=1, h1=0, h2=0, boundary="open"), 3) == 4


@pytest.mark.parametrize("model", ["tfim", "xxz", "cluster"])
def test_spin_flips_commute_with_hamiltonian(model):
    n = 8
    H = build_hamiltonian(model, n, **MODEL_PARAMS[model])
    v = random_state(n, complex_=False)
    for mask in symmetry_generators(model, n):
        lhs = apply_spin_flip(H @ apply_spin_flip(v, mask), mask)
        assert np.allclose(lhs, H @ v)


def test_global_flip_is_vector_reversal():
    v = random_state(5)
    assert np.allclose(apply_spin_flip(v, 2 ** 5 - 1), v[::-1])
    assert np.allclose(apply_spin_flip(v, 2 ** 5 - 1), kron_string("XXXXX") @ v)


# ---------------------------------------------------------------------------
# T1.2  Diagonalization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("generators", [(63,), (0b101010, 0b010101)])
def test_sector_bases_are_orthonormal_and_complete(generators):
    sectors = symmetry_sectors(6, generators)
    total = sp.csr_matrix((64, 64))
    for s in sectors:
        V = s.isometry
        assert np.allclose((V.T @ V).toarray(), np.eye(V.shape[1]))
        total = total + V @ V.T
    assert np.allclose(total.toarray(), np.eye(64))


@pytest.mark.parametrize("model", ["tfim", "xxz", "cluster"])
def test_sector_spectra_combine_to_full_spectrum(model):
    n = 6
    H = build_hamiltonian(model, n, **MODEL_PARAMS[model])
    pieces = [dense_spectrum(s.isometry.T @ H @ s.isometry)
              for s in symmetry_sectors(n, symmetry_generators(model, n))]
    assert np.allclose(np.sort(np.concatenate(pieces)), dense_spectrum(H), atol=1e-10)


@pytest.mark.parametrize("model", ["tfim", "xxz", "cluster"])
def test_sparse_lanczos_agrees_with_dense(model):
    H = build_hamiltonian(model, 10, **MODEL_PARAMS[model])
    w_sparse, v = lowest_eigenpairs(H, 3, method="sparse")
    w_dense = dense_spectrum(H)[:3]
    assert np.allclose(w_sparse, w_dense, atol=1e-10)
    assert np.linalg.norm(H @ v[:, 0] - w_sparse[0] * v[:, 0]) < 1e-9


@pytest.mark.parametrize("model", ["tfim", "xxz", "cluster"])
def test_symmetry_resolved_solver_agrees_with_dense(model):
    n = 8
    res = solve_model(model, n, **MODEL_PARAMS[model])
    H = build_hamiltonian(model, n, **MODEL_PARAMS[model])
    ev = dense_spectrum(H)
    assert np.isclose(res.ground_energy, ev[0], atol=1e-10)
    assert np.allclose(res.energies[:3], ev[:3], atol=1e-10)
    psi = res.ground_state
    assert np.isclose(np.linalg.norm(psi), 1.0)
    assert np.linalg.norm(H @ psi - ev[0] * psi) < 1e-9


def test_sector_states_are_eigenstates_of_flip_at_n14():
    res = solve_model("tfim", 14, J=1.0, h=0.4)
    for labels, psi in zip(res.sector_labels, res.sector_states):
        assert np.allclose(apply_spin_flip(psi, 2 ** 14 - 1), labels[0] * psi, atol=1e-10)


def test_symmetry_sectors_fix_the_cat_state():
    """At N = 14, h = 0.1 the parity splitting is below machine precision."""
    res = solve_model("tfim", 14, J=1.0, h=0.1)
    assert abs(res.sector_energies[1, 0] - res.sector_energies[0, 0]) < 1e-10
    assert res.ground_labels == (1,)
    assert abs(half_chain_entropy(res.ground_state) - LN2) < 1e-3


# ---------------------------------------------------------------------------
# Free-fermion and closed-form checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [4, 6, 8, 10, 12, 14])
@pytest.mark.parametrize("h", [0.1, 0.5, 1.0, 1.5, 2.0])
def test_tfim_matches_free_fermions(n, h):
    res = solve_model("tfim", n, J=1.0, h=h)
    assert np.isclose(res.ground_energy, ff.tfim_ground_energy(n, 1.0, h), atol=1e-9)
    assert np.isclose(res.gap, ff.tfim_gap(n, 1.0, h), atol=1e-9)
    assert np.isclose(res.gap_sector, ff.tfim_even_sector_gap(n, 1.0, h), atol=1e-9)


@pytest.mark.parametrize("n", [4, 6, 8, 10, 12])
def test_xx_chain_matches_free_fermions(n):
    res = solve_model("xxz", n, J=1.0, delta=0.0)
    assert np.isclose(res.ground_energy, ff.xx_chain_ground_energy(n), atol=1e-9)


@pytest.mark.parametrize("n", [4, 8, 12])
def test_tfim_limits(n):
    assert np.isclose(solve_model("tfim", n, J=1.0, h=0.0).ground_energy, -(n - 1))
    assert np.isclose(solve_model("tfim", n, J=0.0, h=1.3).ground_energy, -1.3 * n)
    h = 50.0   # second-order perturbation theory: E0 = -N h - (N-1) J^2 / (4h)
    assert np.isclose(solve_model("tfim", n, J=1.0, h=h).ground_energy,
                      -n * h - (n - 1) / (4 * h), atol=1e-4)


def test_heisenberg_small_chains():
    assert np.isclose(solve_model("xxz", 2, J=1.0, delta=1.0).ground_energy, -3.0)
    assert np.isclose(solve_model("xxz", 4, J=1.0, delta=1.0).ground_energy, -(3 + 2 * np.sqrt(3)))


@pytest.mark.parametrize("n", [6, 8, 10, 12])
def test_xxz_ground_state_symmetries(n):
    res = solve_model("xxz", n, J=1.0, delta=0.7)
    psi = res.ground_state
    assert abs(expval_z(psi).sum()) < 1e-10                 # total S^z = 0
    assert magnetization_moments(psi)[0] < 1e-12            # exactly, not just on average
    assert res.ground_labels == ((-1) ** (n // 2),)         # Marshall sign rule


@pytest.mark.parametrize("n", [6, 8, 10])
def test_cluster_state_limit(n):
    res = solve_model("cluster", n, J=1.0, h1=0.0, h2=0.0)
    assert np.isclose(res.ground_energy, -(n - 2))
    assert np.allclose(res.sector_energies[:, 0], -(n - 2))   # fourfold degenerate (edge modes)
    for psi in res.sector_states:
        assert np.isclose(string_order_parameter(psi), 1.0)


def test_cluster_string_order_vanishes_in_paramagnet():
    psi = solve_model("cluster", 12, J=1.0, h1=3.0, h2=0.0).ground_state
    assert abs(string_order_parameter(psi)) < 0.05


# ---------------------------------------------------------------------------
# T1.3  Observables
# ---------------------------------------------------------------------------
def test_pauli_expectation_matches_dense():
    psi = random_state(4, seed=3)
    for word in ("XZIY", "YYII", "ZXZI", "IIXX"):
        ops = {i: c for i, c in enumerate(word) if c != "I"}
        ref = np.real(psi.conj() @ kron_string(word) @ psi)
        assert np.isclose(pauli_expectation(psi, ops), ref)


def test_zz_matrix_matches_pauli_expectation():
    psi = random_state(5, seed=4)
    C = zz_correlation_matrix(psi)
    assert np.allclose(np.diag(C), 1.0)
    assert np.isclose(C[1, 3], pauli_expectation(psi, {1: "Z", 3: "Z"}))


def test_magnetization_limits():
    n = 6
    m2, m4 = magnetization_moments(ghz(n))
    assert np.isclose(m2, 1.0) and np.isclose(binder_cumulant(m2, m4), 2 / 3)
    plus = np.full(2 ** n, 2 ** (-n / 2))
    assert np.isclose(magnetization_moments(plus)[0], 1 / n)


def test_tfim_magnetization_extremes():
    psi0 = solve_model("tfim", 10, J=1.0, h=0.0).ground_state
    assert np.isclose(magnetization_moments(psi0)[0], 1.0)
    psi_inf = solve_model("tfim", 10, J=0.0, h=1.0).ground_state
    assert np.isclose(magnetization_moments(psi_inf)[0], 0.1)


# ---------------------------------------------------------------------------
# T1.4  Entanglement
# ---------------------------------------------------------------------------
def test_entropy_of_simple_states():
    product = np.zeros(16)
    product[5] = 1.0
    assert entanglement_entropy(product, 2) < 1e-12
    assert np.allclose(entropy_profile(ghz(6)), LN2)
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    two_bells = np.kron(np.kron(bell, np.eye(2)[0]), np.eye(2)[0])  # (0,1) Bell, 2 and 3 product
    assert np.isclose(entanglement_entropy(two_bells, 1), LN2)
    assert entanglement_entropy(two_bells, 2) < 1e-12


def test_renyi_entropies():
    flat = np.full(8, 1 / 8)
    for alpha in (0.5, 1, 2, 3, np.inf):
        assert np.isclose(entropy_from_spectrum(flat, alpha), np.log(8))
    lam = np.array([0.7, 0.2, 0.1])
    s = [entropy_from_spectrum(lam, a) for a in (0.5, 1, 2, np.inf)]
    assert all(a >= b for a, b in zip(s, s[1:]))         # S_alpha decreases with alpha


def test_svd_entropy_matches_partial_trace():
    psi = random_state(7, seed=5)
    rho = reduced_density_matrix(psi, [0, 1, 2])
    assert np.isclose(np.trace(rho).real, 1.0)
    assert np.isclose(von_neumann_entropy_dm(rho), entanglement_entropy(psi, 3))
    rho_b = reduced_density_matrix(psi, [3, 4, 5, 6])
    assert np.isclose(von_neumann_entropy_dm(rho_b), von_neumann_entropy_dm(rho))
    rho_nc = reduced_density_matrix(psi, [0, 4])       # non-contiguous subsystem
    assert np.isclose(np.trace(rho_nc).real, 1.0)
    assert np.all(np.linalg.eigvalsh(rho_nc) > -1e-12)


@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_calabrese_cardy_fit_recovers_synthetic_c(boundary):
    n, c_true, const = 16, 0.73, 0.21
    pref = 6 if boundary == "open" else 3
    S = c_true / pref * chord_log(np.arange(1, n), n, boundary) + const
    fit = calabrese_cardy_fit(S, n, boundary)
    assert np.isclose(fit["c"], c_true) and np.isclose(fit["const"], const)


# ---------------------------------------------------------------------------
# Physics validation (plan, Section 3)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def tfim_scan(n: int, boundary: str = "open") -> tuple:
    """Fine grid around the critical point; returns (h, S_half, binder, gap)."""
    h = make_grid(0.6, 1.4, 0.05)
    S, U, g = [], [], []
    for x in h:
        res = solve_model("tfim", n, J=1.0, h=x, boundary=boundary)
        S.append(half_chain_entropy(res.ground_state))
        U.append(binder_cumulant(*magnetization_moments(res.ground_state)))
        g.append(res.gap)
    return h, np.array(S), np.array(U), np.array(g)


def critical_profile(model: str, n: int, boundary: str, **params) -> np.ndarray:
    return entropy_profile(solve_model(model, n, boundary=boundary, **params).ground_state)


def test_tfim_central_charge_periodic():
    fit = calabrese_cardy_fit(critical_profile("tfim", 14, "periodic", J=1.0, h=1.0), 14, "periodic")
    assert abs(fit["c"] - 0.5) < 0.01


def test_tfim_central_charge_open():
    """Open chains carry boundary corrections at N <= 14 (c ~ 0.56); see Phase 1 notes."""
    fit = calabrese_cardy_fit(critical_profile("tfim", 14, "open", J=1.0, h=1.0), 14, "open")
    assert abs(fit["c"] - 0.5) < 0.08


@pytest.mark.parametrize("delta", [0.0, 0.5])
def test_xxz_central_charge_periodic(delta):
    fit = calabrese_cardy_fit(critical_profile("xxz", 14, "periodic", J=1.0, delta=delta),
                              14, "periodic")
    assert abs(fit["c"] - 1.0) < 0.05


def test_tfim_gap_follows_boundary_cft():
    """N * gap at h_c -> pi v x with v = 2J: x = 1/2 (full gap), x = 2 (even sector)."""
    sizes = np.array([10, 12, 14])
    full = [n * solve_model("tfim", int(n), J=1.0, h=1.0).gap for n in sizes]
    even = [n * solve_model("tfim", int(n), J=1.0, h=1.0).gap_sector for n in sizes]
    a_full = np.polyfit(1 / sizes, full, 1)[1]
    a_even = np.polyfit(1 / sizes, even, 1)[1]
    assert abs(a_full / np.pi - 1) < 0.01
    assert abs(a_even / (4 * np.pi) - 1) < 0.01


def test_tfim_gap_closes_at_criticality():
    gaps = [solve_model("tfim", n, J=1.0, h=1.0).gap for n in (6, 8, 10, 12, 14)]
    assert np.all(np.diff(gaps) < 0)


def test_tfim_binder_crossing_near_hc():
    h, _, u10, _ = tfim_scan(10)
    _, _, u14, _ = tfim_scan(14)
    diff = u10 - u14
    k = np.flatnonzero(np.diff(np.sign(diff)) != 0)[0]
    h_cross = h[k] - diff[k] * (h[k + 1] - h[k]) / (diff[k + 1] - diff[k])
    assert abs(h_cross - 1.0) < 0.1


def test_tfim_entropy_signatures_of_criticality():
    """
    Open chain, even-parity state: the ordered phase sits on the ln 2 cat-state
    plateau, so the transition shows up as the steepest drop of S(N/2).
    Periodic chain: the critical entropy exceeds ln 2 and S(N/2) peaks near h_c.
    """
    h, S_open, _, _ = tfim_scan(14, "open")
    assert abs(h[np.argmin(np.gradient(S_open, h))] - 1.0) <= 0.1
    h, S_pbc, _, _ = tfim_scan(14, "periodic")
    assert 0.8 <= h[np.argmax(S_pbc)] <= 1.1
    s_crit = [half_chain_entropy(solve_model("tfim", n, J=1.0, h=1.0).ground_state)
              for n in (8, 10, 12, 14)]
    assert np.all(np.diff(s_crit) > 0)               # log growth at criticality
