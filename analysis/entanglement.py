"""
Entanglement of pure states (Phase 1, task T1.4).

For a bipartition into the first ell sites (A) and the rest (B), the state
vector reshaped into a 2^ell x 2^(N-ell) matrix has singular values sigma_k,
and the eigenvalues of rho_A are lambda_k = sigma_k^2. No reduced density
matrix has to be formed. Natural logarithms are used throughout, which the
Calabrese-Cardy prefactors c/6 and c/3 assume.

reduced_density_matrix() handles arbitrary (non-contiguous) subsystems and
is used for validation here and for mixed-state measures in Phase 4.
"""

from __future__ import annotations

import numpy as np

from analysis.observables import n_sites_of


# ---------------------------------------------------------------------------
# Spectra and entropies
# ---------------------------------------------------------------------------
def schmidt_spectrum(psi: np.ndarray, ell: int) -> np.ndarray:
    """Eigenvalues of rho_A for A = sites 0 .. ell-1 (descending, sum to 1)."""
    n = n_sites_of(psi)
    if not 0 < ell < n:
        raise ValueError(f"cut position ell must satisfy 0 < ell < {n}")
    sigma = np.linalg.svd(psi.reshape(2 ** ell, 2 ** (n - ell)), compute_uv=False)
    lam = sigma ** 2
    return lam / lam.sum()


def entropy_from_spectrum(lam: np.ndarray, alpha: float = 1.0) -> float:
    """
    Renyi entropy S_alpha of a probability vector.

    alpha = 1 gives von Neumann, alpha = np.inf gives the min-entropy.
    """
    lam = np.clip(np.asarray(lam, dtype=float), 0.0, None)
    lam = lam[lam > 0]
    if alpha == 1:
        return float(-np.sum(lam * np.log(lam)))
    if np.isinf(alpha):
        return float(-np.log(lam.max()))
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return float(np.log(np.sum(lam ** alpha)) / (1.0 - alpha))


def entanglement_entropy(psi: np.ndarray, ell: int, alpha: float = 1.0) -> float:
    return entropy_from_spectrum(schmidt_spectrum(psi, ell), alpha)


def half_chain_entropy(psi: np.ndarray, alpha: float = 1.0) -> float:
    return entanglement_entropy(psi, n_sites_of(psi) // 2, alpha)


def entropy_profile(psi: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """S(ell) for ell = 1 .. N-1."""
    n = n_sites_of(psi)
    return np.array([entanglement_entropy(psi, ell, alpha) for ell in range(1, n)])


def entropy_profiles(psi: np.ndarray, alphas=(1.0, 2.0)) -> dict:
    """Several Renyi profiles from a single set of SVDs."""
    n = n_sites_of(psi)
    spectra = [schmidt_spectrum(psi, ell) for ell in range(1, n)]
    return {a: np.array([entropy_from_spectrum(l, a) for l in spectra]) for a in alphas}


# ---------------------------------------------------------------------------
# General reduced density matrix
# ---------------------------------------------------------------------------
def reduced_density_matrix(psi: np.ndarray, sites) -> np.ndarray:
    """
    rho_A = Tr_B |psi><psi| for any set of kept sites (in the order given,
    first site most significant).
    """
    n = n_sites_of(psi)
    sites = list(sites)
    rest = [i for i in range(n) if i not in sites]
    tensor = psi.reshape([2] * n).transpose(sites + rest)
    m = tensor.reshape(2 ** len(sites), 2 ** len(rest))
    return m @ m.conj().T


def von_neumann_entropy_dm(rho: np.ndarray, alpha: float = 1.0) -> float:
    """Entropy of a density matrix from its eigenvalues."""
    return entropy_from_spectrum(np.linalg.eigvalsh(rho), alpha)


# ---------------------------------------------------------------------------
# Calabrese-Cardy fit
# ---------------------------------------------------------------------------
def chord_log(ell: np.ndarray, n_sites: int, boundary: str = "open") -> np.ndarray:
    """
    Scaling variable of the Calabrese-Cardy formula.

    open:      ln[(2N/pi) sin(pi ell / N)],   S = (c/6) x + s1'
    periodic:  ln[(N/pi)  sin(pi ell / N)],   S = (c/3) x + s1
    """
    ell = np.asarray(ell, dtype=float)
    base = np.sin(np.pi * ell / n_sites) / np.pi
    if boundary == "open":
        return np.log(2 * n_sites * base)
    if boundary == "periodic":
        return np.log(n_sites * base)
    raise ValueError(f"unknown boundary '{boundary}'")


def calabrese_cardy_fit(profile: np.ndarray, n_sites: int, boundary: str = "open",
                        ell_min: int = 2, parity: str = None) -> dict:
    """
    Least-squares fit of S(ell), ell = 1..N-1, to the Calabrese-Cardy form.

    ell_min : keep ell_min <= ell <= N/2 (drop the smallest blocks; ell > N/2
              duplicates ell < N/2 exactly for a pure state)
    parity  : None (all ell), "even" or "odd" -- open-boundary XXZ profiles
              oscillate between even and odd ell, which biases a joint fit

    Returns c, its standard error, the constant, and the points used.
    """
    profile = np.asarray(profile, dtype=float)
    ell = np.arange(1, n_sites)
    # S(ell) = S(N - ell) for any pure state, so ell > N/2 adds no information
    mask = (ell >= ell_min) & (ell <= n_sites // 2)
    if parity == "even":
        mask &= ell % 2 == 0
    elif parity == "odd":
        mask &= ell % 2 == 1
    elif parity is not None:
        raise ValueError("parity must be None, 'even' or 'odd'")

    x = chord_log(ell[mask], n_sites, boundary)
    y = profile[mask]
    prefactor = 6.0 if boundary == "open" else 3.0
    nan_result = {"c": np.nan, "c_err": np.nan, "const": np.nan,
                  "ell": ell[mask], "x": x, "S": y, "n_points": int(mask.sum())}
    if np.unique(np.round(x, 12)).size < 2:
        return nan_result

    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, const = coef
    dof = len(x) - 2
    if dof > 0:
        resid = y - A @ coef
        cov = (resid @ resid / dof) * np.linalg.inv(A.T @ A)
        slope_err = float(np.sqrt(max(cov[0, 0], 0.0)))
    else:
        slope_err = np.nan
    return {"c": float(prefactor * slope), "c_err": float(prefactor * slope_err),
            "const": float(const), "ell": ell[mask], "x": x, "S": y,
            "n_points": int(mask.sum())}
