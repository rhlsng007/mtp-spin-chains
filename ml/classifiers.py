"""
Phase 7: supervised and unsupervised phase classification (T7.1 to T7.6).

Classifiers (T7.1). Both act on standardized features (StandardScaler fitted
on the training rows only):

* "logistic": L2-regularized logistic regression,

      p(y = 1 | x) = sigma(w . x + b),     sigma(t) = 1 / (1 + e^-t),

  with inverse regularization strength CFG.ml.logistic_C.
* "mlp": a small multilayer perceptron (hidden layers CFG.ml.mlp_hidden, ReLU,
  Adam with learning rate CFG.ml.learning_rate, at most CFG.ml.epochs epochs,
  L2 penalty CFG.ml.mlp_alpha, mini-batches of CFG.ml.mlp_batch_size). Early
  stopping is off by default: the training regions are separable, so the
  validation accuracy reaches 1 within a few epochs and would stop a network
  whose outputs are still near 1/2.

Predicted critical point. The test rows of every field are averaged into the
classifier output curve P(h) = mean of p(y = 1 | x) over the test shadows of
field h. h_c^ML is the field where P crosses 1/2:

      P(h_c^ML) = 1/2.

P is first made monotone by isotonic regression (identical to P wherever P is
already increasing), then interpolated linearly between the last field below
1/2 and the first field above it. The number of crossings of the raw curve is
kept as a diagnostic. Each test shadow also gives its own curve and crossing
(hc_rep); their spread measures the shot noise of the test data.

Seeds (T7.6). Seed s fixes (i) a bootstrap of the training shadows, which
resamples the training repetitions of every training field with replacement
(so every training field stays represented), and (ii) the MLP initialization
and early-stopping split. Both depend on s only, not on the dataset, noise
strength or variant (common random numbers):
SeedSequence([2026, 17, s]) for the MLP, [2026, 17, s, 1] for the bootstrap
and [2026, 17, s, 2] for learning-curve subsets. Exact features (one row per
field) are never bootstrapped, so the logistic regression on exact features
is identical for every seed.

PCA (T7.2). Principal components of the centred, unscaled "corr" features of
all fields and repetitions, without labels. PC1 is oriented so that it
correlates positively with M^2. Two label-free transition markers: the field
of steepest change of the field-averaged PC1, and the field where PC1 crosses
the midpoint between its averages over the two deep regions.

Shadow rebuild (shots test). Shadow features are recomputed from the first m
stored snapshots of every shadow, for the pure (ED, VQE) and noisy datasets,
with the same per-repetition readout correction as in Phase 6.
"""

from __future__ import annotations

import warnings

import numpy as np

from config import CFG
from ml import features as F
from ml import shadows as S

EXP_TRAIN = 17
CLASSIFIERS = tuple(CFG.ml.classifiers)
KINDS = ("exact", "shadow")


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
def training_seeds() -> list:
    return list(range(CFG.ml.n_training_seeds))


def model_seed(seed: int) -> int:
    ss = np.random.SeedSequence([CFG.seeds.master_seed, EXP_TRAIN, int(seed)])
    return int(ss.generate_state(1)[0] % (2 ** 31 - 1))


def bootstrap_indices(seed: int, n_fields: int, n_reps: int) -> np.ndarray:
    """(n_fields, n_reps) repetition indices drawn with replacement within each field."""
    rng = np.random.default_rng(np.random.SeedSequence([CFG.seeds.master_seed, EXP_TRAIN, int(seed), 1]))
    return rng.integers(0, n_reps, size=(n_fields, n_reps))


def subset_indices(seed: int, n_fields: int, n_reps: int, k: int) -> np.ndarray:
    """(n_fields, k) distinct repetition indices per field (learning curve)."""
    rng = np.random.default_rng(np.random.SeedSequence([CFG.seeds.master_seed, EXP_TRAIN, int(seed), 2]))
    return np.stack([rng.permutation(n_reps)[:k] for _ in range(n_fields)])


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------
def make_classifier(name: str, seed: int = 0, n_rows: int | None = None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "logistic":
        clf = LogisticRegression(C=CFG.ml.logistic_C, max_iter=CFG.ml.logistic_max_iter)
    elif name == "mlp":
        early = bool(CFG.ml.mlp_early_stopping)
        clf = MLPClassifier(hidden_layer_sizes=tuple(CFG.ml.mlp_hidden), activation="relu", solver="adam",
                            learning_rate_init=CFG.ml.learning_rate, max_iter=CFG.ml.epochs,
                            alpha=CFG.ml.mlp_alpha,
                            batch_size=min(CFG.ml.mlp_batch_size, n_rows) if n_rows else CFG.ml.mlp_batch_size,
                            early_stopping=early, validation_fraction=CFG.ml.mlp_validation_fraction,
                            n_iter_no_change=CFG.ml.mlp_patience, random_state=model_seed(seed))
    else:
        raise ValueError(f"unknown classifier {name!r}; expected one of {CLASSIFIERS}")
    return make_pipeline(StandardScaler(), clf)


def _fit(name: str, X_train, y_train, seed: int):
    from sklearn.exceptions import ConvergenceWarning
    X_train = np.asarray(X_train, dtype=float)
    model = make_classifier(name, seed, n_rows=X_train.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, np.asarray(y_train).astype(int))
    return model


def fit_predict(name: str, X_train, y_train, X_test, seed: int = 0) -> np.ndarray:
    """p(y = 1 | x) for every test row (any leading shape, features last)."""
    model = _fit(name, X_train, y_train, seed)
    X_test = np.asarray(X_test, dtype=float)
    p = model.predict_proba(X_test.reshape(-1, X_test.shape[-1]))[:, 1]
    return p.reshape(X_test.shape[:-1])


# ---------------------------------------------------------------------------
# Training rows from field blocks
# ---------------------------------------------------------------------------
def training_rows(blocks: list, y_fields, rep_idx=None) -> tuple:
    """
    Stack training rows from blocks of shape (n_fields, n_reps, F) (shadows) or
    (n_fields, F) (exact). rep_idx (n_fields, k) selects repetitions per field
    (bootstrap or subset) and is applied to every block alike.
    """
    X, y = [], []
    y_fields = np.asarray(y_fields)
    for B in blocks:
        B = np.asarray(B)
        if B.ndim == 2:
            X.append(B)
            y.append(y_fields)
            continue
        if rep_idx is not None:
            B = np.take_along_axis(B, np.asarray(rep_idx)[:, :, None], axis=1)
        X.append(B.reshape(-1, B.shape[-1]))
        y.append(np.repeat(y_fields, B.shape[1]))
    return np.concatenate(X), np.concatenate(y)


# ---------------------------------------------------------------------------
# Output curves, critical point, accuracy
# ---------------------------------------------------------------------------
def isotonic(P) -> np.ndarray:
    """Least-squares increasing fit of P on equally weighted points (pool adjacent violators)."""
    vals, wts, cnt = [], [], []
    for v in np.asarray(P, dtype=float):
        vals.append(v)
        wts.append(1.0)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            c = cnt[-2] + cnt[-1]
            del vals[-1], wts[-1], cnt[-1]
            vals[-1], wts[-1], cnt[-1] = v, w, c
    return np.repeat(vals, cnt)


def critical_point(h, P) -> float:
    """Field where the isotonic fit of P(h) crosses 1/2 (NaN without a crossing); h ascending."""
    h, P = np.asarray(h, dtype=float), np.asarray(P, dtype=float)
    if not np.all(np.isfinite(P)):
        return float("nan")
    d = isotonic(P) - CFG.ml.decision_threshold
    above, below = np.flatnonzero(d > 0), np.flatnonzero(d < 0)
    if above.size == 0 or below.size == 0:
        return float("nan")
    k, j = above[0], below[-1]
    return float(h[j] + (-d[j]) * (h[k] - h[j]) / (d[k] - d[j]))


def n_crossings(P) -> int:
    s = np.sign(np.asarray(P) - CFG.ml.decision_threshold)
    s = s[s != 0]
    return int(np.sum(s[1:] != s[:-1]))


def field_masks(h, y) -> dict:
    h, y = np.asarray(h), np.asarray(y)
    labeled = y != CFG.ml.label_critical
    lo, hi = CFG.ml.train_h_ordered_max, CFG.ml.train_h_disordered_min
    window = (h >= lo - 1e-9) & (h <= hi + 1e-9)
    return {"all": labeled, "window": labeled & window, "deep": labeled & ~window}


def accuracies(p, h, y) -> dict:
    """Accuracy of the rows of p (n_fields, n_reps) or (n_fields,) against the true labels."""
    p = np.asarray(p)
    p2 = p[:, None] if p.ndim == 1 else p
    correct = (p2 >= CFG.ml.decision_threshold) == (np.asarray(y)[:, None] == CFG.ml.label_disordered)
    out = {}
    for key, m in field_masks(h, y).items():
        out[key] = float(np.mean(correct[m])) if np.any(m) else float("nan")
    return out


def evaluate(train_blocks: list, y_train_fields, tests, h, y, seeds=None,
             classifiers=CLASSIFIERS, rep_subset: int | None = None) -> list:
    """
    Train every classifier for every seed once and evaluate on every test set.

    train_blocks: list of (n_train_fields, n_reps, F) shadow blocks or
    (n_train_fields, F) exact blocks (several blocks are pooled, e.g. mixed
    noise levels). tests: one array or a list of arrays (n_fields, n_reps_test, F)
    or (n_fields, F). Returns one dict per test set with leading axes
    [classifier, seed]: P (field curve), P_rep (per test repetition; exact:
    one), hc, hc_rep, ncross, acc, acc_window, acc_deep.
    """
    single = not isinstance(tests, (list, tuple))
    tests = [np.asarray(tests)] if single else [np.asarray(t) for t in tests]
    seeds = training_seeds() if seeds is None else list(seeds)
    nc, ns = len(classifiers), len(seeds)
    outs = []
    for T in tests:
        nh, n_rt = T.shape[0], (T.shape[1] if T.ndim == 3 else 1)
        outs.append({"P": np.full((nc, ns, nh), np.nan), "P_rep": np.full((nc, ns, n_rt, nh), np.nan),
                     "hc": np.full((nc, ns), np.nan), "hc_rep": np.full((nc, ns, n_rt), np.nan),
                     "ncross": np.zeros((nc, ns)), "acc": np.full((nc, ns), np.nan),
                     "acc_window": np.full((nc, ns), np.nan), "acc_deep": np.full((nc, ns), np.nan)})
    shadow_train = np.asarray(train_blocks[0]).ndim == 3
    for js, s in enumerate(seeds):
        rep_idx = None
        if shadow_train:
            nf, nr = np.asarray(train_blocks[0]).shape[:2]
            if rep_subset is not None:
                rep_idx = subset_indices(s, nf, nr, rep_subset)
            elif CFG.ml.bootstrap_training:
                rep_idx = bootstrap_indices(s, nf, nr)
        Xtr, ytr = training_rows(train_blocks, y_train_fields, rep_idx)
        for ic, name in enumerate(classifiers):
            model = _fit(name, Xtr, ytr, s)
            for T, out in zip(tests, outs):
                p = model.predict_proba(T.reshape(-1, T.shape[-1]))[:, 1].reshape(T.shape[:-1])
                p2 = p[:, None] if p.ndim == 1 else p
                P = p2.mean(axis=1)
                out["P"][ic, js] = P
                out["P_rep"][ic, js] = p2.T
                out["hc"][ic, js] = critical_point(h, P)
                out["ncross"][ic, js] = n_crossings(P)
                if T.ndim == 3:
                    out["hc_rep"][ic, js] = [critical_point(h, p2[:, r]) for r in range(p2.shape[1])]
                a = accuracies(p2, h, y)
                out["acc"][ic, js], out["acc_window"][ic, js], out["acc_deep"][ic, js] = a["all"], a["window"], a["deep"]
    return outs[0] if single else outs


def summarize_hc(hc, hc_rep=None) -> dict:
    """
    Mean over seeds (last axis of hc), seed spread (sample standard deviation),
    and the shot-noise standard error of h_c^ML from the test repetitions
    (standard deviation of the per-repetition crossings / sqrt(number), averaged
    over seeds). NaN (no crossing) entries are ignored; all-NaN gives NaN.
    """
    hc = np.asarray(hc, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(hc, axis=-1)
        seed_std = np.nanstd(hc, axis=-1, ddof=1) if hc.shape[-1] > 1 else np.zeros_like(mean)
        out = {"mean": mean, "seed_std": seed_std, "n_valid": np.sum(np.isfinite(hc), axis=-1)}
        if hc_rep is not None:
            hr = np.asarray(hc_rep, dtype=float)
            n = np.sum(np.isfinite(hr), axis=-1)
            sd = np.nanstd(hr, axis=-1, ddof=1)
            se = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), np.nan)
            out["shot_se"] = np.nanmean(se, axis=-1)
            out["total"] = np.sqrt(np.nan_to_num(seed_std) ** 2 + np.nan_to_num(out["shot_se"]) ** 2)
    return out


# ---------------------------------------------------------------------------
# Splitting a dataset array into training blocks and test sets
# ---------------------------------------------------------------------------
def split_array(ds: dict, A, kind: str, feature_set: str = "corr") -> tuple:
    """
    (train block, test array) from a field-major array: shadows (39, R, F),
    exact (39, F). Train: training fields and repetitions; test: all fields,
    test repetitions (exact: all fields).
    """
    N = int(ds["n_qubits"])
    A = F.select_features(np.asarray(A), N, feature_set)
    tf = np.asarray(ds["train_field"], dtype=bool)
    if kind == "exact":
        return A[tf], A
    rt, rs = np.asarray(ds["rep_train"], dtype=bool), np.asarray(ds["rep_test"], dtype=bool)
    return A[tf][:, rt], A[:, rs]


def dataset_array(ds: dict, kind: str, variant: str | None = None, strength: int | None = None):
    key = ("X" if kind == "exact" else "Xs") + (f"_{variant}" if variant else "")
    A = np.asarray(ds[key])
    return A[strength] if strength is not None else A


# ---------------------------------------------------------------------------
# PCA (T7.2)
# ---------------------------------------------------------------------------
def pca_analysis(X_all, h, m2_exact, n_components: int | None = None) -> dict:
    """
    PCA of all rows (no labels). X_all: (n_fields, n_reps, F) or (n_fields, F).
    m2_exact (n_fields,) orients PC1 and gives the correlation check.
    """
    from sklearn.decomposition import PCA
    X_all = np.asarray(X_all, dtype=float)
    nc = n_components or CFG.ml.pca_components
    X3 = X_all[:, None, :] if X_all.ndim == 2 else X_all
    nh, nr, nf = X3.shape
    rows = X3.reshape(-1, nf)
    pca = PCA(n_components=min(nc, nf, rows.shape[0])).fit(rows)
    scores = pca.transform(rows).reshape(nh, nr, -1)
    pc1 = scores[..., 0].mean(axis=1)
    m2 = np.asarray(m2_exact, dtype=float)
    r = float(np.corrcoef(pc1, m2)[0, 1]) if np.std(m2) > 0 and np.std(pc1) > 0 else float("nan")
    sign = -1.0 if (np.isfinite(r) and r < 0) else 1.0
    scores *= sign
    pc1 = sign * pc1
    r = abs(r) if np.isfinite(r) else r
    pc1_std = scores[..., 0].std(axis=1, ddof=1) if nr > 1 else np.zeros(nh)
    evr = np.full(nc, np.nan)
    evr[:pca.explained_variance_ratio_.size] = pca.explained_variance_ratio_
    h = np.asarray(h, dtype=float)
    grad = np.gradient(pc1, h)
    h_steep = float(h[int(np.argmax(np.abs(grad)))])
    masks = field_masks(h, np.zeros_like(h, dtype=int))
    deep = masks["deep"]
    lo = deep & (h < CFG.tfim.h_critical)
    hi = deep & (h > CFG.tfim.h_critical)
    mid = 0.5 * (pc1[lo].mean() + pc1[hi].mean())
    # PC1 falls from the ordered to the disordered side; crossing of the midpoint
    h_mid = critical_point(h, -(pc1 - mid) + CFG.ml.decision_threshold)
    pc2 = sign * scores[..., 1].mean(axis=1) if scores.shape[-1] > 1 else np.full(nh, np.nan)
    return {"pc1": pc1, "pc1_std": pc1_std, "pc2": pc2, "evr": evr, "r_m2": r,
            "h_steep": h_steep, "h_mid": h_mid, "components": sign * pca.components_[0]}


# ---------------------------------------------------------------------------
# Shadow rebuild from stored snapshots (shots test)
# ---------------------------------------------------------------------------
def rebuild_shadows(ds: dict, n_snapshots: int, strength: int | None = None, variant: str | None = None):
    """
    Shadow "corr" features (39, R, F) from the first n_snapshots of every stored
    shadow. Pure datasets: variant None. Noisy datasets: variant "gate", "meas"
    or "ro" at the given strength index.
    """
    N = int(ds["n_qubits"])
    m = int(n_snapshots)
    B = np.asarray(ds["bases"])[:, :, :m]
    if "bits" in ds:
        bits = S.unpack_bits(np.asarray(ds["bits"])[:, :, :m], N)
        sh = S.shadow_observables(B, bits)
    else:
        if variant not in ("gate", "meas", "ro"):
            raise ValueError("noisy datasets need variant 'gate', 'meas' or 'ro'")
        key = "bits_gate" if variant == "gate" else "bits_meas"
        bits = S.unpack_bits(np.asarray(ds[key])[strength, :, :, :m], N)
        if variant == "ro":
            a, b = F.readout_affine(np.asarray(ds["cal_samples"])[strength])     # (R, N)
            sh = S.shadow_observables(B, bits, a, b)
        else:
            sh = S.shadow_observables(B, bits)
    return F.corr_features(sh["zz"], sh["x"])
