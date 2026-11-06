"""
Phase 7 tests: critical point, accuracies, seeds, classifiers, PCA, shadow rebuild.

Run from the project root:  pytest -q tests\\test_phase7.py
None of these tests needs saved results.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import CFG                                                     # noqa: E402
from analysis.ff_correlators import ff_observables                         # noqa: E402
from ml import classifiers as C                                            # noqa: E402
from ml import dataset as D                                                # noqa: E402
from ml import features as F                                               # noqa: E402
from ml import shadows as S                                                # noqa: E402
from models.exact import solve_model                                       # noqa: E402

H = D.field_grid()
Y = D.labels(H)
TF = D.train_field_mask(H)


def _experiment_module():
    spec = importlib.util.spec_from_file_location("phase7_classify", ROOT / "experiments" / "phase7_classify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Critical point and accuracies
# ---------------------------------------------------------------------------
def test_isotonic_matches_sklearn():
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.default_rng(1)
    for _ in range(200):
        P = np.clip(1 / (1 + np.exp(-(H - 1) / 0.1)) + 0.2 * rng.normal(size=H.size), 0, 1)
        assert np.allclose(C.isotonic(P), IsotonicRegression().fit_transform(H, P), atol=1e-14)


def test_critical_point_exact_for_piecewise_linear_curve():
    for hc in (0.62, 0.87, 1.0, 1.33):
        P = np.clip(0.5 + 2.0 * (H - hc), 0.0, 1.0)              # linear near the crossing
        assert C.critical_point(H, P) == pytest.approx(hc, abs=1e-12)
    assert np.isnan(C.critical_point(H, np.full(H.size, 0.2)))
    assert np.isnan(C.critical_point(H, np.full(H.size, 0.8)))
    P = 1 / (1 + np.exp(-(H - 0.9) / 0.05))
    assert C.n_crossings(P) == 1
    P_bad = P.copy()
    P_bad[3] = 0.9                                                 # an isolated spike in the ordered phase
    assert C.n_crossings(P_bad) == 3
    assert abs(C.critical_point(H, P_bad) - 0.9) < 0.05            # isotonic fit is robust to it


def test_accuracies_exclude_critical_field():
    p = (H > 0.9).astype(float)                                   # wrong on 0.95 only (and h_c excluded)
    a = C.accuracies(p, H, Y)
    n_lab = np.sum(Y != CFG.ml.label_critical)
    assert a["all"] == pytest.approx((n_lab - 1) / n_lab)
    assert a["deep"] == 1.0
    masks = C.field_masks(H, Y)
    assert masks["window"].sum() + masks["deep"].sum() == n_lab
    assert a["window"] == pytest.approx(1 - 1 / masks["window"].sum())


def test_summarize_hc():
    hc = np.array([0.80, 0.82, 0.84, np.nan])
    rep = np.array([[0.79, 0.81], [0.81, 0.83], [0.83, 0.85], [np.nan, np.nan]])
    s = C.summarize_hc(hc, rep)
    assert s["mean"] == pytest.approx(0.82)
    assert s["seed_std"] == pytest.approx(0.02)
    assert s["n_valid"] == 3
    assert s["shot_se"] == pytest.approx(np.std([0.79, 0.81], ddof=1) / np.sqrt(2))
    assert s["total"] == pytest.approx(np.hypot(s["seed_std"], s["shot_se"]))


# ---------------------------------------------------------------------------
# Seeds and training rows
# ---------------------------------------------------------------------------
def test_bootstrap_and_subsets():
    a, b = C.bootstrap_indices(0, 22, 16), C.bootstrap_indices(0, 22, 16)
    assert np.array_equal(a, b) and a.shape == (22, 16) and a.min() >= 0 and a.max() < 16
    assert not np.array_equal(a, C.bootstrap_indices(1, 22, 16))
    s = C.subset_indices(3, 22, 16, 4)
    assert s.shape == (22, 4) and all(len(set(r)) == 4 for r in s)
    assert C.model_seed(0) != C.model_seed(1)


def test_training_rows_keep_fields():
    nf, nr, nfeat = 5, 4, 3
    B = np.arange(nf * nr * nfeat, dtype=float).reshape(nf, nr, nfeat)
    yf = np.array([0, 0, 1, 1, 1])
    idx = np.array([[0, 0, 1, 3]] * nf)
    X, y = C.training_rows([B, B + 1000], yf, idx)
    assert X.shape == (2 * nf * nr, nfeat) and np.array_equal(y, np.tile(np.repeat(yf, nr), 2))
    assert np.array_equal(X[:4], B[0, [0, 0, 1, 3]])               # repetitions resampled within field 0
    Xe, ye = C.training_rows([B[:, 0]], yf)
    assert Xe.shape == (nf, nfeat) and np.array_equal(ye, yf)


def test_split_array_matches_sample_matrix():
    rng = np.random.default_rng(0)
    N, R = 4, 20
    rep_train, rep_test = D.repetition_split(R)
    ds = {"Xs": rng.normal(size=(H.size, R, F.n_features(N))), "X": rng.normal(size=(H.size, F.n_features(N))),
          "h": H, "y": Y, "train_field": TF, "test_field": np.ones_like(TF), "rep_train": rep_train,
          "rep_test": rep_test, "n_qubits": N}
    for kind in ("exact", "shadow"):
        for fs in ("corr", "compact"):
            tr, te = C.split_array(ds, C.dataset_array(ds, kind), kind, fs)
            Xtr, _, _ = D.sample_matrix(ds, kind, part="train", feature_set=fs)
            Xte, _, _ = D.sample_matrix(ds, kind, part="test", feature_set=fs)
            assert np.allclose(tr.reshape(-1, tr.shape[-1]), Xtr)
            assert np.allclose(te.reshape(-1, te.shape[-1]), Xte)


# ---------------------------------------------------------------------------
# Classifiers on exact free-fermion data
# ---------------------------------------------------------------------------
def _ff_X(N):
    return np.array([F.corr_features(**{k: v for k, v in ff_observables(N, CFG.tfim.J, float(h)).items()
                                         if k != "z"}) for h in H])


def test_ed_and_free_fermion_classifiers_agree():
    N = 6
    X_ff = _ff_X(N)
    X_ed = []
    for h in H:
        o = F.pure_observables(solve_model("tfim", N, J=CFG.tfim.J, h=float(h), boundary="open").ground_state, N)
        X_ed.append(F.corr_features(o["zz"], o["x"]))
    X_ed = np.array(X_ed)
    r_ff = C.evaluate([X_ff[TF]], Y[TF], X_ff, H, Y, seeds=[0, 1])
    r_ed = C.evaluate([X_ed[TF]], Y[TF], X_ed, H, Y, seeds=[0, 1])
    assert np.max(np.abs(r_ff["P"] - r_ed["P"])) < CFG.ml.identity_tol
    lr = C.CLASSIFIERS.index("logistic")
    assert r_ff["hc"][lr, 0] == r_ff["hc"][lr, 1]                 # exact logistic: seed independent
    assert 0.6 < r_ff["hc"][lr, 0] < 1.0                           # finite-size pseudo-critical point
    assert r_ff["acc_deep"][lr, 0] == 1.0


def test_logistic_hc_increases_with_N():
    hc = []
    for N in (4, 8, 16):
        X = _ff_X(N)
        hc.append(C.evaluate([X[TF]], Y[TF], X, H, Y, seeds=[0], classifiers=("logistic",))["hc"][0, 0])
    assert np.all(np.diff(hc) > 0) and hc[-1] < CFG.tfim.h_critical


def test_evaluate_many_tests_equals_single():
    X = _ff_X(4)
    a = C.evaluate([X[TF]], Y[TF], [X, X + 0.01], H, Y, seeds=[0])
    b = C.evaluate([X[TF]], Y[TF], X + 0.01, H, Y, seeds=[0])
    assert np.array_equal(a[1]["P"], b["P"]) and np.array_equal(a[1]["hc"], b["hc"])


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def test_pca_tracks_magnetization_exactly():
    N = 8
    X = _ff_X(N)
    m2 = F.magnetization_m2(F.unpack_features(X, N)[0])
    res = C.pca_analysis(X, H, m2)
    assert res["r_m2"] >= CFG.ml.pca_corr_min
    assert np.corrcoef(res["pc1"], m2)[0, 1] > 0                   # orientation: PC1 follows M^2
    assert res["evr"][0] > 0.9
    assert 0.6 < res["h_mid"] < 1.0


# ---------------------------------------------------------------------------
# Shadow rebuild
# ---------------------------------------------------------------------------
def test_rebuild_pure_shadows():
    N, R, M = 4, 3, 60
    h = H[[0, 18, 38]]
    states = [solve_model("tfim", N, J=CFG.tfim.J, h=float(v), boundary="open").ground_state for v in h]
    ds = D.pure_dataset(states, h, N, D.SOURCE_KEYS["ed"], 0, R=R, M=M)
    assert np.array_equal(C.rebuild_shadows(ds, M), ds["Xs"])
    m = 10
    sh = S.shadow_observables(ds["bases"][:, :, :m], S.unpack_bits(ds["bits"][:, :, :m], N))
    assert np.array_equal(C.rebuild_shadows(ds, m), F.corr_features(sh["zz"], sh["x"]))


def test_rebuild_noisy_shadows_all_variants():
    N, p = 4, 2
    h = np.array([0.3, 1.7])
    thetas = np.random.default_rng(5).normal(scale=0.4, size=(h.size, 2 * p))
    ds = D.noisy_dataset("hva", N, p, "realistic", thetas=thetas, h=h, strengths=np.array([0.0, 1e-3]), R=2, M=40)
    for v in D.NOISY_VARIANTS:
        for js in (0, 1):
            assert np.allclose(C.rebuild_shadows(ds, 40, js, v), ds[f"Xs_{v}"][js], atol=1e-12)
    with pytest.raises(ValueError):
        C.rebuild_shadows(ds, 40, 0, None)


# ---------------------------------------------------------------------------
# Experiment helper
# ---------------------------------------------------------------------------
def test_fill_grows_trailing_axes():
    mod = _experiment_module()
    store = {}
    mod.fill(store, (0,), {"a": np.ones((1, 3))}, shape=(2,))
    mod.fill(store, (1,), {"a": 2 * np.ones((4, 3))}, shape=(2,))
    A = store["a"]
    assert A.shape == (2, 4, 3)
    assert np.all(A[0, 0] == 1) and np.all(np.isnan(A[0, 1:])) and np.all(A[1] == 2)
