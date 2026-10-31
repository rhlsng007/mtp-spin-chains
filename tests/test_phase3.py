"""
Phase 3 validation tests. Run from the project root with:  pytest -q tests\\test_phase3.py

Each test compares the Phase 3 code with an exact or independent result:
Phase 1 entropy routines, exact limits (product, GHZ and cluster states, the
Page formula), rigorous inequalities (Schmidt-rank, Eckart-Young and
Fannes-Audenaert bounds) and the periodic HVA, which is exact at p = N/2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from config import CFG
from analysis.entanglement import calabrese_cardy_fit, entropy_from_spectrum, entropy_profile, schmidt_spectrum
from analysis.vqe_entanglement import (LN2, PurityOperator, audenaert_bound, entropy_bound,
                                       maximize_entanglement, page_entropy, profile_data,
                                       projected_reference, purity_and_grad, rank_fidelity_bound,
                                       schmidt_rank, symmetry_broken_state)
from models.exact import solve_model
from models.hamiltonians import build_hamiltonian
from vqe.ansatz import HEA, HVA, make_ansatz
from vqe.metrics import ed_reference, evaluate_sweep
from vqe.optimize import optimize_point, point_rng, random_candidates

TOL = CFG.ent.rank_tol


def random_state(n, seed):
    rng = point_rng(950, seed)
    psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return psi / np.linalg.norm(psi)


def ghz(n):
    psi = np.zeros(2 ** n)
    psi[0] = psi[-1] = 1 / np.sqrt(2)
    return psi


# ---------------------------------------------------------------------------
# Entropies, ranks, simple states
# ---------------------------------------------------------------------------
def test_profile_data_matches_phase1_routines():
    psi = random_state(7, 1)
    d = profile_data(psi, TOL)
    assert np.allclose(d["S"], entropy_profile(psi))
    assert np.allclose(d["S2"], entropy_profile(psi, 2.0))
    assert d["rank"].tolist() == [2, 4, 8, 8, 4, 2]


def test_ranks_of_product_and_ghz_states():
    product = np.zeros(2 ** 6, complex)
    product[13] = 1.0
    assert profile_data(product, TOL)["rank"].tolist() == [1] * 5
    assert profile_data(ghz(6), TOL)["rank"].tolist() == [2] * 5
    assert np.allclose(profile_data(ghz(6), TOL)["S"], LN2)


def test_hva_depth_one_gives_the_cluster_state():
    """IsingZZ(pi/2) on every bond of |+>^N is the linear cluster state: S = ln 2 at every cut."""
    psi = HVA(8, 1).state(np.array([np.pi / 4, 0.0]))
    assert np.allclose(entropy_profile(psi), LN2, atol=1e-12)


def test_page_formula():
    assert np.isclose(page_entropy(2), 1 / 3)                    # m = n = 2: 1/3 + 1/4 - 1/4
    assert abs(page_entropy(12) - (6 * LN2 - 0.5)) < 1e-3        # large-m limit


# ---------------------------------------------------------------------------
# Rigorous bounds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["hva", "hea"])
@pytest.mark.parametrize("p", [1, 2, 3, 5])
def test_schmidt_rank_and_entropy_bound(name, p):
    n = 8
    a = make_ansatz(name, n, p)
    for k in range(3):
        psi = a.state(a.random_params(point_rng(951, p, k), 1.0, "uniform"))
        d = profile_data(psi, TOL)
        ell = np.arange(1, n)
        assert np.all(d["rank"] <= 2 ** np.minimum(np.minimum(ell, n - ell), p))
        assert np.all(d["S"] <= entropy_bound(p, n) + 1e-12)
        assert np.all(d["S2"] <= entropy_bound(p, n) + 1e-12)


def test_eckart_young_bound_is_attained_by_truncation():
    n, r, ell = 6, 3, 3
    phi = random_state(n, 2)
    u, s, vh = np.linalg.svd(phi.reshape(2 ** ell, 2 ** (n - ell)))
    trunc = (u[:, :r] * s[:r]) @ vh[:r]
    trunc = trunc.ravel() / np.linalg.norm(trunc)
    lam = schmidt_spectrum(phi, ell)
    assert np.isclose(abs(np.vdot(phi, trunc)) ** 2, lam[:r].sum())
    assert rank_fidelity_bound(phi, r) <= lam[:r].sum() + 1e-12


def test_rank_fidelity_bound_holds_for_low_rank_states():
    n = 8
    phi = random_state(n, 3)
    for p in (1, 2, 3):
        a = HEA(n, p)
        for k in range(5):
            psi = a.state(a.random_params(point_rng(952, p, k), 1.0, "uniform"))
            assert abs(np.vdot(phi, psi)) ** 2 <= rank_fidelity_bound(phi, 2 ** p) + 1e-12


def test_fannes_audenaert_bound_holds_for_random_pairs():
    n = 6
    dim = 2 ** (n // 2)
    for k in range(40):
        psi = random_state(n, 100 + k)
        eps = 10.0 ** point_rng(953, k).uniform(-6, 0)
        phi = psi + eps * random_state(n, 200 + k)
        phi /= np.linalg.norm(phi)
        f = abs(np.vdot(psi, phi)) ** 2
        ds = abs(entropy_profile(psi)[n // 2 - 1] - entropy_profile(phi)[n // 2 - 1])
        assert ds <= audenaert_bound(f, dim) + 1e-12


def test_audenaert_bound_limits():
    assert audenaert_bound(1.0, 8) == 0.0
    vals = audenaert_bound(np.array([0.999999, 0.99, 0.9, 0.5]), 8)
    assert np.all(np.diff(vals) > 0)
    assert np.isclose(audenaert_bound(0.0, 8), np.log(8))


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
def test_symmetry_broken_reference_deep_in_ordered_phase():
    res = solve_model("tfim", 10, J=1.0, h=0.1)
    even, odd = res.sector_states
    assert abs(entropy_profile(even)[4] - LN2) < 1e-3
    assert entropy_profile(symmetry_broken_state(even, odd))[4] < 1e-2


def test_projected_reference_weight_equals_ground_space_fidelity():
    n, h = 6, 0.4
    ref = ed_reference(n, np.array([h]), verbose=False)
    a = HEA(n, 2)
    best = optimize_point(a, build_hamiltonian("tfim", n, J=1.0, h=h),
                          random_candidates(a, 3, point_rng(954))).best
    m = evaluate_sweep(a, best.x[None, :], np.array([best.energy]), ref)
    phi, w = projected_reference(a.state(best.x), ref["states"][0], m["ground_space"][0])
    assert np.isclose(w, m["fidelity_gs"][0])
    assert np.isclose(abs(np.vdot(phi, a.state(best.x))) ** 2, w)


# ---------------------------------------------------------------------------
# Maximal entanglement (T3.3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["hva", "hea"])
def test_purity_operator_and_exact_gradient(name):
    n = 6
    a = make_ansatz(name, n, 3)
    op = PurityOperator(n)
    theta = a.random_params(point_rng(955), 1.0, "uniform")
    p, g = purity_and_grad(a, theta, op)
    lam = schmidt_spectrum(a.state(theta), n // 2)
    assert np.isclose(p, np.sum(lam ** 2))
    fd = np.array([(np.sum(schmidt_spectrum(a.state(theta + 1e-6 * e), n // 2) ** 2)
                    - np.sum(schmidt_spectrum(a.state(theta - 1e-6 * e), n // 2) ** 2)) / 2e-6
                   for e in np.eye(a.n_params)])
    assert np.allclose(g, fd, atol=1e-7)


@pytest.mark.parametrize("name", ["hva", "hea"])
@pytest.mark.parametrize("p", [1, 2])
def test_maximal_entanglement_reaches_the_bound(name, p):
    n = 6
    a = make_ansatz(name, n, p)
    best = max((maximize_entanglement(a, a.random_params(point_rng(956, p, k), 1.0, "uniform"),
                                      300, 1e-10) for k in range(4)), key=lambda r: r["S2"])
    assert abs(best["S2"] - p * LN2) < 1e-6
    assert best["S2"] <= p * LN2 + 1e-9


# ---------------------------------------------------------------------------
# Central charge from the periodic HVA
# ---------------------------------------------------------------------------
def test_periodic_hva_at_half_depth_reproduces_the_ed_central_charge():
    n, h = 8, 1.0
    H = build_hamiltonian("tfim", n, J=1.0, h=h, boundary="periodic")
    a = HVA(n, n // 2, boundary="periodic")
    best = optimize_point(a, H, random_candidates(a, CFG.vqe.n_random_inits, point_rng(957))).best
    ed = solve_model("tfim", n, J=1.0, h=h, boundary="periodic").ground_state
    c_vqe = calabrese_cardy_fit(entropy_profile(a.state(best.x)), n, "periodic")["c"]
    c_ed = calabrese_cardy_fit(entropy_profile(ed), n, "periodic")["c"]
    assert abs(c_vqe - c_ed) < 1e-6
    assert abs(c_ed - 0.5) < 0.02
