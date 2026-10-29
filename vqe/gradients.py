"""
Gradients for validation and barren-plateau statistics (Phase 2, T2.3 and T2.7).

Generalized parameter-shift rule
--------------------------------
If a parameter enters only through gates exp(-i theta G), E(theta) is a
trigonometric polynomial whose frequencies are the eigenvalue differences
of G. Write E(theta) = f(omega theta) with f having integer frequencies
1..R. Then (Wierichs, Izaac, Wang & Lin, Quantum 6, 677 (2022))

    f'(0) = sum_{mu=1}^{2R} f(x_mu) (-1)^(mu-1) / (4 R sin^2(x_mu / 2)),
    x_mu  = (2 mu - 1) pi / (2R),

and dE/dtheta = omega f'(0). For R = 1 this is the familiar two-term rule
with shifts +/- pi/2. The HEA has (R, omega) = (1, 1) for every parameter;
the HVA's shared ZZ and X parameters have (N - 1, 2) and (N, 2).

The rule is exact and needs only energy evaluations, as on hardware. The
optimizer itself uses the adjoint gradient (equal to machine precision,
see tests/test_phase2.py) because it is far cheaper.

Barren plateaus
---------------
gradient_variance() samples theta ~ U[0, 2 pi)^P and returns Var[dE/dtheta_k]
for every k. For a circuit that forms an approximate unitary 2-design the
variance decays exponentially with N (McClean et al., Nat. Commun. 9, 4812
(2018)), even for a local observable once the depth grows with N (Cerezo et
al., Nat. Commun. 12, 1791 (2021)). The TFIM HVA lives in the free-fermion
Lie algebra of dimension O(N^2), so its variance decays only polynomially
(Larocca et al., Quantum 6, 824 (2022); Wiersema et al., PRX Quantum 1,
020319 (2020)).
"""

from __future__ import annotations

import numpy as np


def shift_points(order: int) -> tuple:
    """Shift points x_mu and coefficients of the generalized parameter-shift rule."""
    mu = np.arange(1, 2 * order + 1)
    x = (2 * mu - 1) * np.pi / (2 * order)
    coeff = (-1.0) ** (mu - 1) / (4 * order * np.sin(x / 2) ** 2)
    return x, coeff


def parameter_shift_gradient(ansatz, theta: np.ndarray, hamiltonian) -> np.ndarray:
    """Exact gradient from energy evaluations only (generalized shift rule)."""
    theta = np.asarray(theta, dtype=np.float64).ravel()
    grad = np.zeros_like(theta)
    for k, (order, omega) in enumerate(ansatz.shift_spec()):
        x, coeff = shift_points(order)
        total = 0.0
        for xm, cm in zip(x, coeff):
            shifted = theta.copy()
            shifted[k] += xm / omega
            total += cm * ansatz.energy(shifted, hamiltonian)
        grad[k] = omega * total
    return grad


def finite_difference_gradient(ansatz, theta: np.ndarray, hamiltonian,
                               step: float = 1e-6) -> np.ndarray:
    """Central differences (validation only)."""
    theta = np.asarray(theta, dtype=np.float64).ravel()
    grad = np.zeros_like(theta)
    for k in range(theta.size):
        e = np.zeros_like(theta)
        e[k] = step
        grad[k] = (ansatz.energy(theta + e, hamiltonian)
                   - ansatz.energy(theta - e, hamiltonian)) / (2 * step)
    return grad


def gradient_samples(ansatz, observable, n_samples: int, rng: np.random.Generator,
                     distribution: str = "uniform") -> np.ndarray:
    """(n_samples, n_params) array of exact gradients at random parameters."""
    out = np.empty((n_samples, ansatz.n_params))
    for s in range(n_samples):
        theta = ansatz.random_params(rng, 1.0, distribution)
        out[s] = ansatz.energy_and_grad(theta, observable)[1]
    return out


def gradient_variance(ansatz, observable, n_samples: int, rng: np.random.Generator,
                      distribution: str = "uniform") -> dict:
    """
    Var[dE/dtheta_k] over random initializations, for every parameter k.

    Returns the per-parameter variance and mean, plus the variance of the
    first parameter (the usual McClean et al. diagnostic) and the mean
    variance over all parameters (less noisy for the same sample count).
    """
    g = gradient_samples(ansatz, observable, n_samples, rng, distribution)
    var = g.var(axis=0, ddof=1)
    return {"var": var, "mean": g.mean(axis=0), "var_first": float(var[0]),
            "var_mean": float(var.mean()), "var_median": float(np.median(var))}


def fit_exponential_decay(sizes, variances) -> dict:
    """
    Least-squares fit of ln Var = a - b N.

    b > 0 is the decay rate; Var proportional to 2^(-N) gives b = ln 2.
    Also returns the residual-based standard error of b.
    """
    n = np.asarray(sizes, dtype=float)
    y = np.log(np.asarray(variances, dtype=float))
    A = np.column_stack([np.ones_like(n), -n])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(len(n) - 2, 1)
    cov = np.linalg.inv(A.T @ A) * (resid @ resid) / dof
    return {"a": float(coef[0]), "b": float(coef[1]), "b_err": float(np.sqrt(cov[1, 1]))}


def fit_power_law(sizes, variances) -> dict:
    """Least-squares fit of ln Var = a - k ln N (polynomial decay)."""
    n = np.log(np.asarray(sizes, dtype=float))
    y = np.log(np.asarray(variances, dtype=float))
    A = np.column_stack([np.ones_like(n), -n])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(len(n) - 2, 1)
    cov = np.linalg.inv(A.T @ A) * (resid @ resid) / dof
    return {"a": float(coef[0]), "k": float(coef[1]), "k_err": float(np.sqrt(cov[1, 1]))}
