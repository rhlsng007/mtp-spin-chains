"""
VQE optimization (Phase 2, tasks T2.3 to T2.5).

T2.3  minimize()      : exact-expectation VQE with L-BFGS-B (default) or Adam,
                        exact adjoint gradients from the ansatz.
T2.4  optimize_point(): several initializations per point, best one kept.
T2.5  field_sweep()   : warm start along the field axis; the optimum at the
                        previous field value is one of the initial points at
                        the next one.

Every random draw comes from point_rng(...), a Generator seeded by a
SeedSequence built from the master seed and integer keys (ansatz, N, p,
field index, restart). Results are therefore identical however the jobs
are split across processes or in which order they are run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from config import CFG

ANSATZ_KEYS = {"hva": 1, "hea": 2}


# ---------------------------------------------------------------------------
# Reproducible randomness
# ---------------------------------------------------------------------------
def point_rng(*keys: int) -> np.random.Generator:
    """Independent Generator for the given integer keys (order-independent of the run)."""
    entropy = [CFG.seeds.master_seed] + [int(k) for k in keys]
    return np.random.default_rng(np.random.SeedSequence(entropy))


# ---------------------------------------------------------------------------
# Single optimization
# ---------------------------------------------------------------------------
@dataclass
class OptResult:
    x: np.ndarray
    energy: float
    n_iter: int
    n_fev: int
    success: bool
    message: str
    label: str = ""
    seconds: float = 0.0
    history: list = field(default_factory=list)


def minimize(ansatz, hamiltonian, x0: np.ndarray, method: str = None,
             maxiter: int = None, record_history: bool = False, label: str = "") -> OptResult:
    """
    Minimize E(theta) = <psi(theta)|H|psi(theta)> from x0.

    L-BFGS-B stops when the largest gradient component falls below
    CFG.vqe.tolerance (gtol) or the relative energy decrease per step falls
    below CFG.vqe.lbfgs_ftol. Adam runs with learning rate CFG.vqe.adam_lr
    and stops on the same gradient criterion.
    """
    c = CFG.vqe
    method = (method or c.optimizer_noiseless).upper()
    maxiter = c.max_iterations if maxiter is None else maxiter
    history = []
    t0 = time.perf_counter()

    if method == "L-BFGS-B":
        def fun(x):
            e, g = ansatz.energy_and_grad(x, hamiltonian)
            if record_history:
                history.append(e)
            return e, g

        res = scipy_minimize(fun, np.asarray(x0, dtype=np.float64), jac=True,
                             method="L-BFGS-B",
                             options={"maxiter": maxiter, "ftol": c.lbfgs_ftol,
                                      "gtol": c.tolerance, "maxcor": c.lbfgs_maxcor})
        out = OptResult(x=res.x, energy=float(res.fun), n_iter=int(res.nit),
                        n_fev=int(res.nfev), success=bool(res.success),
                        message=str(res.message), label=label)
    elif method == "ADAM":
        out = _adam(ansatz, hamiltonian, x0, maxiter, history if record_history else None)
        out.label = label
    else:
        raise ValueError(f"unknown optimizer '{method}' (use L-BFGS-B or Adam)")

    out.seconds = time.perf_counter() - t0
    out.history = history
    return out


def _adam(ansatz, hamiltonian, x0, maxiter, history) -> OptResult:
    """Adam (Kingma & Ba 2015) on the exact gradient; returns the best iterate."""
    c = CFG.vqe
    b1, b2, eps = c.adam_beta1, c.adam_beta2, 1e-12
    x = np.asarray(x0, dtype=np.float64).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    best_x, best_e = x.copy(), np.inf
    success, it = False, 0
    for it in range(1, maxiter + 1):
        e, g = ansatz.energy_and_grad(x, hamiltonian)
        if history is not None:
            history.append(e)
        if e < best_e:
            best_e, best_x = e, x.copy()
        if np.max(np.abs(g)) < c.tolerance:
            success = True
            break
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x = x - c.adam_lr * (m / (1 - b1 ** it)) / (np.sqrt(v / (1 - b2 ** it)) + eps)
    msg = "gradient below tolerance" if success else "maximum iterations reached"
    return OptResult(x=best_x, energy=float(best_e), n_iter=it, n_fev=it,
                     success=success, message=msg)


# ---------------------------------------------------------------------------
# Several initializations at one point (T2.4)
# ---------------------------------------------------------------------------
@dataclass
class PointResult:
    best: OptResult
    candidate_labels: list
    candidate_energies: np.ndarray
    total_fev: int
    seconds: float


def optimize_point(ansatz, hamiltonian, candidates: list, method: str = None) -> PointResult:
    """
    Run one optimization per (label, x0) candidate and keep the lowest energy.

    On (near) ties the earlier candidate wins, so a warm start listed first
    is preferred over a random restart that reaches the same minimum.
    """
    if not candidates:
        raise ValueError("no initial points given")
    t0 = time.perf_counter()
    results = [minimize(ansatz, hamiltonian, x0, method=method, label=label)
               for label, x0 in candidates]
    energies = np.array([r.energy for r in results])
    tie = CFG.vqe.tie_tolerance
    best_index = int(np.flatnonzero(energies <= energies.min() + tie)[0])
    return PointResult(best=results[best_index],
                       candidate_labels=[r.label for r in results],
                       candidate_energies=energies,
                       total_fev=int(sum(r.n_fev for r in results)),
                       seconds=time.perf_counter() - t0)


def random_candidates(ansatz, n: int, rng: np.random.Generator, prefix: str = "random",
                      distribution: str = None, scale: float = None) -> list:
    distribution = distribution or CFG.vqe.init_distribution
    scale = CFG.vqe.init_scale if scale is None else scale
    return [(f"{prefix}{k}", ansatz.random_params(rng, scale, distribution)) for k in range(n)]


# ---------------------------------------------------------------------------
# Field sweep with warm start (T2.5)
# ---------------------------------------------------------------------------
SOURCE_CODES = {"warm": 0, "depth": 1, "random": 2}


def source_code(label: str) -> int:
    for prefix, code in SOURCE_CODES.items():
        if label.startswith(prefix):
            return code
    return -1


def sweep_order(n_points: int, direction: str) -> np.ndarray:
    if direction == "ascending":
        return np.arange(n_points)
    if direction == "descending":
        return np.arange(n_points)[::-1]
    raise ValueError(f"direction must be 'ascending' or 'descending', got '{direction}'")


def field_sweep(ansatz, hamiltonians: list, direction: str, rng_keys: tuple,
                n_restarts_first: int, n_restarts: int, anchor_mask: np.ndarray = None,
                n_restarts_anchor: int = None, warm_start: bool = True,
                extra_inits: np.ndarray = None, method: str = None,
                progress: bool = False) -> dict:
    """
    Optimize the ansatz at every field value, visiting them in `direction`.

    Initial points at each field value, in this order of preference:
      1. 'warm'   : optimum of the previously visited field value (T2.5);
      2. 'depth'  : extra_inits[i], e.g. the depth p - 1 optimum padded with an
                    identity layer (HVA only), if given;
      3. 'random' : n_restarts small random initializations (T2.4). The first
                    visited point gets n_restarts_first of them (there is no warm
                    start yet), anchor points get n_restarts_anchor.

    Returns arrays indexed in the ORIGINAL field order.
    """
    n_h = len(hamiltonians)
    anchor_mask = np.zeros(n_h, bool) if anchor_mask is None else np.asarray(anchor_mask)
    n_restarts_anchor = n_restarts_first if n_restarts_anchor is None else n_restarts_anchor

    out = {
        "energy": np.full(n_h, np.nan),
        "params": np.zeros((n_h, ansatz.n_params)),
        "n_iter": np.zeros(n_h, int),
        "n_fev": np.zeros(n_h, int),
        "total_fev": np.zeros(n_h, int),
        "success": np.zeros(n_h, bool),
        "best_source": np.zeros(n_h, int),
        "n_candidates": np.zeros(n_h, int),
        "warm_energy": np.full(n_h, np.nan),
        "random_best_energy": np.full(n_h, np.nan),
        "seconds": np.zeros(n_h),
    }
    previous = None
    order = sweep_order(n_h, direction)
    for step, i in enumerate(order):
        candidates = []
        if warm_start and previous is not None:
            candidates.append(("warm", previous))
        if extra_inits is not None:
            candidates.append(("depth", extra_inits[i]))
        if step == 0 or not warm_start:
            n_rand = n_restarts_first
        elif anchor_mask[i]:
            n_rand = n_restarts_anchor
        else:
            n_rand = n_restarts
        rng = point_rng(*rng_keys, i)
        candidates += random_candidates(ansatz, n_rand, rng)

        res = optimize_point(ansatz, hamiltonians[i], candidates, method=method)
        b = res.best
        out["energy"][i] = b.energy
        out["params"][i] = b.x
        out["n_iter"][i] = b.n_iter
        out["n_fev"][i] = b.n_fev
        out["total_fev"][i] = res.total_fev
        out["success"][i] = b.success
        out["best_source"][i] = source_code(b.label)
        out["n_candidates"][i] = len(candidates)
        out["seconds"][i] = res.seconds
        labels = res.candidate_labels
        e = res.candidate_energies
        if "warm" in labels:
            out["warm_energy"][i] = e[labels.index("warm")]
        rand = [k for k, lab in enumerate(labels) if lab.startswith("random")]
        if rand:
            out["random_best_energy"][i] = e[rand].min()
        previous = b.x
        if progress:
            print(f"    {ansatz.name} N={ansatz.n_qubits} p={ansatz.depth} "
                  f"point {step + 1}/{n_h}: E = {b.energy:.10f} ({b.label}, {b.n_iter} it)")
    return out
