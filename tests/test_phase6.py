"""
Phase 6 tests: features, classical shadows, readout, labels and splits.

Run from the project root:  pytest -q tests\\test_phase6.py
None of these tests needs saved results.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG                                                     # noqa: E402
from analysis.ff_correlators import ff_energy, ff_observables, pfaffian    # noqa: E402
from ml import dataset as D                                                 # noqa: E402
from ml import features as F                                                # noqa: E402
from ml import shadows as S                                                 # noqa: E402
from models.exact import solve_model                                       # noqa: E402
from models.hamiltonians import build_hamiltonian                          # noqa: E402
from noise.channels import apply_unitary                                   # noqa: E402

RNG = np.random.default_rng(20261103)


def random_state(N, rng=RNG):
    psi = rng.normal(size=2 ** N) + 1j * rng.normal(size=2 ** N)
    return psi / np.linalg.norm(psi)


def random_rho(N, rank=3, rng=RNG):
    G = rng.normal(size=(2 ** N, rank)) + 1j * rng.normal(size=(2 ** N, rank))
    rho = G @ G.conj().T
    return rho / np.trace(rho)


def pauli_string(ops, N):
    P = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
         "Z": np.diag([1.0, -1.0])}
    out = np.ones((1, 1))
    for q in range(N):
        out = np.kron(out, P[ops.get(q, "I")])
    return out


# ---------------------------------------------------------------------------
# Free-fermion reference
# ---------------------------------------------------------------------------
def test_pfaffian_small_and_squared_determinant():
    A = RNG.normal(size=(4, 4))
    A = A - A.T
    assert pfaffian(A) == pytest.approx(A[0, 1] * A[2, 3] - A[0, 2] * A[1, 3] + A[0, 3] * A[1, 2], abs=1e-12)
    B = RNG.normal(size=(10, 10))
    B = B - B.T
    assert pfaffian(B) ** 2 == pytest.approx(np.linalg.det(B), rel=1e-9)


@pytest.mark.parametrize("N", [4, 6, 8])
@pytest.mark.parametrize("h", [0.3, 1.0, 1.7])
def test_free_fermion_matches_ed(N, h):
    res = solve_model("tfim", N, J=CFG.tfim.J, h=h, boundary="open")
    obs = F.pure_observables(res.ground_state, N)
    ff = ff_observables(N, CFG.tfim.J, h)
    assert np.max(np.abs(obs["zz"] - ff["zz"])) < CFG.ml.ff_tol
    assert np.max(np.abs(obs["x"] - ff["x"])) < CFG.ml.ff_tol
    assert ff_energy(N, CFG.tfim.J, h) == pytest.approx(res.ground_energy, abs=1e-10)


# ---------------------------------------------------------------------------
# Exact observables and features
# ---------------------------------------------------------------------------
def test_observables_match_pauli_matrices():
    N = 4
    psi = random_state(N)
    rho = random_rho(N)
    o_p, o_m = F.pure_observables(psi, N), F.mixed_observables(rho, N)
    for i in range(N):
        assert o_p["x"][i] == pytest.approx(np.real(np.vdot(psi, pauli_string({i: "X"}, N) @ psi)), abs=1e-12)
        assert o_m["x"][i] == pytest.approx(np.real(np.trace(rho @ pauli_string({i: "X"}, N))), abs=1e-12)
        assert o_m["z"][i] == pytest.approx(np.real(np.trace(rho @ pauli_string({i: "Z"}, N))), abs=1e-12)
        for j in range(i + 1, N):
            Pz = pauli_string({i: "Z", j: "Z"}, N)
            assert o_m["zz"][i, j] == pytest.approx(np.real(np.trace(rho @ Pz)), abs=1e-12)
    o_pm = F.mixed_observables(np.outer(psi, psi.conj()), N)
    for k in o_p:
        assert np.allclose(o_p[k], o_pm[k], atol=1e-13)


def test_feature_roundtrip_energy_and_sets():
    N = 6
    psi = random_state(N)
    o = F.pure_observables(psi, N)
    X = F.corr_features(o["zz"], o["x"])
    assert X.shape == (F.n_features(N),) and len(F.feature_names(N)) == X.size
    zz, x = F.unpack_features(X, N)
    assert np.allclose(zz, o["zz"]) and np.allclose(x, o["x"])
    h = 0.7
    H = build_hamiltonian("tfim", N, J=CFG.tfim.J, h=h, boundary="open")
    assert F.energy_from_features(X, N, h) == pytest.approx(np.real(np.vdot(psi, H @ psi)), abs=1e-12)
    assert F.select_features(X, N, "zz").size == N * (N - 1) // 2
    c = F.select_features(X, N, "compact")
    assert c[0] == pytest.approx(F.magnetization_m2(o["zz"]))
    assert np.linalg.eigvalsh(o["zz"]).min() > -1e-12              # ZZ matrix is a Gram matrix


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------
def test_outcome_table_matches_rotated_diagonals():
    N = 3
    rho = random_rho(N)
    T = S.outcome_table(rho, N)
    assert T.size == 6 ** N and np.sum(T) == pytest.approx(3 ** N)
    for basis in [(0, 1, 2), (2, 2, 2), (1, 0, 1), (0, 0, 0)]:
        r = rho.copy()
        for q, b in enumerate(basis):
            r = apply_unitary(r, S.ROTATIONS[b], [q], N)
        assert np.allclose(np.real(np.diag(r)), S.outcome_probabilities(T, basis, N), atol=1e-13)


@pytest.mark.parametrize("N", [3, 5, 7])
def test_pure_and_mixed_samplers_agree_exactly(N):
    psi = random_state(N)
    B, u, _ = S.draw_stream(np.random.default_rng(N), 4000, N)
    b_tree = S.sample_pure(psi, B, u)
    b_loop = S.sample_pure(psi, B, u, chunk_elements=2 * 2 ** N)  # no prefix tree
    b_mix = S.sample_mixed(S.marginal_tables(S.outcome_table(np.outer(psi, psi.conj()), N), N), B, u)
    assert np.array_equal(b_tree, b_loop) and np.array_equal(b_tree, b_mix)


def test_sampler_frequencies_fixed_basis():
    N, M = 3, 200_000
    rho = random_rho(N)
    T = S.outcome_table(rho, N)
    basis = np.array([0, 1, 2])
    B = np.tile(basis.astype(np.uint8), (M, 1))
    u = np.random.default_rng(7).random((M, N))
    bits = S.sample_mixed(S.marginal_tables(T, N), B, u)
    idx = bits.astype(int) @ (2 ** (N - 1 - np.arange(N)))
    freq = np.bincount(idx, minlength=2 ** N) / M
    p = S.outcome_probabilities(T, basis, N)
    assert np.all(np.abs(freq - p) < 5 * np.sqrt(p * (1 - p) / M) + 1e-12)


# ---------------------------------------------------------------------------
# Shadow estimators
# ---------------------------------------------------------------------------
def test_shadow_product_state_limits():
    N, M = 4, 6000
    psi = np.zeros(2 ** N)
    psi[0] = 1.0                                                   # |0000>: Z = ZZ = 1, X = 0
    B, u, _ = S.draw_stream(np.random.default_rng(1), M, N)
    sh = S.shadow_observables(B, S.sample_pure(psi, B, u))
    nz = (B == 2).mean(axis=0)
    assert np.allclose(sh["z"], 3 * nz)                            # every Z-basis outcome is +1
    assert np.allclose(np.diag(sh["zz"]), 1.0)
    assert np.all(np.abs(sh["x"]) < 5 * np.sqrt(3 / M))


def test_shadow_unbiased_with_analytic_variance():
    N, M, R = 4, 2000, 200
    psi = random_state(N)
    obs = F.pure_observables(psi, N)
    B, u, _ = S.draw_stream(np.random.default_rng(11), M * R, N)
    b = S.sample_pure(psi, B, u)
    sh = S.shadow_observables(B.reshape(R, M, N), b.reshape(R, M, N))
    var = S.single_snapshot_variance(obs)
    iu = np.triu_indices(N, 1)
    for key, sel in (("zz", iu), ("x", slice(None))):
        est = sh[key][(slice(None),) + (sel if key == "zz" else (sel,))]
        exact = obs[key][sel]
        v = var[key][sel] / M
        z = (est.mean(axis=0) - exact) / np.sqrt(v / R)
        assert np.all(np.abs(z) < 5)
        ratio = est.var(axis=0, ddof=1) / v
        assert np.all(np.abs(ratio - 1) < 0.35)                    # R = 200: std of ratio about 0.1


def test_readout_bias_and_correction():
    N, M = 3, 400_000
    psi = random_state(N)
    obs = F.pure_observables(psi, N)
    p01, p10 = np.array([0.03, 0.05, 0.02]), np.array([0.06, 0.04, 0.08])
    R = np.stack([np.array([[1 - a, b], [a, 1 - b]]) for a, b in zip(p01, p10)])
    B, u, r = S.draw_stream(np.random.default_rng(3), M, N)
    bits = S.sample_pure(psi, B, u)
    meas = S.apply_readout(bits, r, p01, p10)
    assert abs(np.mean(meas[bits == 0]) - np.mean(p01)) < 0.01   # flip rates (pooled)
    biased = F.readout_biased(obs, R)
    sh_m = S.shadow_observables(B, meas)
    a, b = F.readout_affine(R)
    sh_c = S.shadow_observables(B, meas, a, b)
    tol_zz, tol_x = 6 * 3 / np.sqrt(M), 6 * np.sqrt(3 / M)
    assert np.max(np.abs(sh_m["zz"] - biased["zz"])) < tol_zz
    assert np.max(np.abs(sh_m["x"] - biased["x"])) < tol_x
    assert np.max(np.abs(sh_c["zz"] - obs["zz"])) < 1.2 * tol_zz
    assert np.max(np.abs(sh_c["x"] - obs["x"])) < 1.2 * tol_x
    corrected = F.readout_corrected(obs, R, R)                    # calibration = readout: exact recovery
    for k in obs:
        assert np.allclose(corrected[k], obs[k], atol=1e-13)


def test_pack_unpack_bits():
    for N in (4, 8, 9, 14):
        bits = (RNG.random((3, 5, N)) < 0.5).astype(np.uint8)
        assert np.array_equal(S.unpack_bits(S.pack_bits(bits), N), bits)


# ---------------------------------------------------------------------------
# Labels, splits and random streams
# ---------------------------------------------------------------------------
def test_labels_and_splits():
    h = D.field_grid()
    y = D.labels(h)
    assert h.size == 39
    assert np.all(y[h < 1.0 - 1e-9] == CFG.ml.label_ordered)
    assert np.all(y[h > 1.0 + 1e-9] == CFG.ml.label_disordered)
    assert np.sum(y == CFG.ml.label_critical) == 1
    train = D.train_field_mask(h)
    assert np.array_equal(train, (h < 0.6 - 1e-9) | (h > 1.4 + 1e-9))
    assert train.sum() == 10 + 12
    rt, rs = D.repetition_split(20)
    assert rt.sum() == 16 and rs.sum() == 4 and not np.any(rt & rs)


def test_streams_are_deterministic_and_distinct():
    a = D.draw_snapshots(1, 4, 2, 3, 2, 50)
    b = D.draw_snapshots(1, 4, 2, 3, 2, 50)
    c = D.draw_snapshots(1, 4, 2, 3, 2, 50, scale=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[1], c[1])
    assert not np.array_equal(a[1][0], a[1][1])                   # repetitions differ


# ---------------------------------------------------------------------------
# Small end-to-end noisy and ZNE datasets (random parameters, no saved data)
# ---------------------------------------------------------------------------
def _small_noisy(ansatz="hva"):
    N, p = 4, 2
    h = np.array([0.3, 1.0, 1.7])
    n_par = 2 * p if ansatz == "hva" else 2 * N * (p + 1)
    thetas = np.random.default_rng(5).normal(scale=0.4, size=(h.size, n_par))
    strengths = np.array([0.0, 1e-3, 2.15443469e-3])
    return N, p, h, thetas, strengths


@pytest.mark.parametrize("ansatz", ["hva", "hea"])
def test_noisy_dataset_limits(ansatz):
    from noise.density import simulate_statevector
    N, p, h, thetas, strengths = _small_noisy(ansatz)
    ds = D.noisy_dataset(ansatz, N, p, "realistic", thetas=thetas, h=h, strengths=strengths, R=3, M=200)
    pure = D.pure_dataset([simulate_statevector(ansatz, N, p, t) for t in thetas], h, N,
                          D.SOURCE_KEYS[ansatz], p, R=3, M=200)
    assert np.allclose(ds["X_gate"][0], pure["X"], atol=1e-12)    # strength 0 = noiseless
    assert np.allclose(ds["X_ro"][0], ds["X_gate"][0], atol=1e-12)  # no SPAM at strength 0
    bits0 = S.unpack_bits(ds["bits_gate"][0], N)
    assert np.mean(np.all(bits0 == S.unpack_bits(pure["bits"], N), axis=-1)) >= CFG.ml.snapshot_match_min
    assert np.array_equal(ds["bases"], pure["bases"])
    assert np.all(np.diff(ds["purity"], axis=0) < 1e-12)          # purity falls with strength
    assert np.allclose(ds["fidelity_ideal"][0], 1.0)
    assert ds["Xs_ro"].shape == (3, 3, 3, F.n_features(N))


def test_zne_scale_one_equals_noisy():
    N, p, h, thetas, strengths = _small_noisy("hva")
    ds = D.noisy_dataset("hva", N, p, "realistic", thetas=thetas, h=h, strengths=strengths, R=2, M=100)
    z = D.zne_dataset("hva", N, p, "realistic", thetas=thetas, h=h, strengths=strengths, R=2, M=100)
    for jw, js in enumerate(z["strength_index"]):
        assert np.allclose(z["Xs_ro_scales"][jw, 0], ds["Xs_ro"][js], atol=1e-12)
        assert np.allclose(z["X_gate_scales"][jw, 0], ds["X_gate"][js], atol=1e-12)
    c = z["coefficients"]
    assert np.sum(c) == pytest.approx(1.0) and np.dot(c, z["scales"]) == pytest.approx(0.0, abs=1e-12)


def test_sample_matrix_shapes():
    N, p, h, thetas, strengths = _small_noisy("hva")
    h = D.field_grid()[[0, 5, 18, 30, 38]]                         # 0.1, 0.35, 1.0, 1.6, 2.0
    thetas = np.random.default_rng(1).normal(scale=0.3, size=(h.size, 2 * p))
    ds = D.noisy_dataset("hva", N, p, "realistic", thetas=thetas, h=h, strengths=strengths[:2], R=5, M=50)
    X, y, hh = D.sample_matrix(ds, "shadow", "ro", strength=1, part="train")
    assert X.shape == (4 * 4, F.n_features(N)) and set(np.unique(y)) == {0, 1}
    Xt, yt, ht = D.sample_matrix(ds, "shadow", "ro", strength=1, part="test")
    assert Xt.shape[0] == 5 * 1 and np.sum(yt == -1) == 1
    Xe, ye, he = D.sample_matrix(ds, "exact", "gate", strength=0, part="train", feature_set="compact")
    assert Xe.shape == (4, 4)
