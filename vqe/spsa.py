"""
Simultaneous perturbation stochastic approximation (Phase 4, T4.3).

Iteration k (k = 0, 1, ...), with Delta_k a vector of independent +-1 entries:

    g_k = [f(theta_k + c_k Delta_k) - f(theta_k - c_k Delta_k)] / (2 c_k) * Delta_k
    theta_{k+1} = theta_k - a_k g_k
    a_k = a / (k + 1 + A)^alpha,   c_k = c / (k + 1)^gamma

with Spall's asymptotically optimal exponents alpha = 0.602, gamma = 0.101 and
A = 10 % of the iteration budget. Two function evaluations per step, whatever
the number of parameters.

Calibration (Spall 1998; the same rule as Qiskit's SPSA): before the run the
mean magnitude of the gradient estimate at theta_0 is measured over a few
random directions, and a is chosen so that the first step moves each parameter
by about ``target_step``:

    a = target_step * (A + 1)^alpha / mean |g_0|

Important: at a stationary point the symmetric difference cancels to third
order in c, so calibrating there gives a near-zero |g| and an exploding gain.
The Phase 4 experiments therefore calibrate once per problem (ansatz, N, h) at
the standard cold initial point, with exact noiseless energies, and reuse that
gain for every start and every noise strength of the problem, so that all runs
of a problem share one learning-rate schedule (``calibrate_gain`` + ``gain=``).

The returned parameters are the average of the last ``average_last`` iterates
(Polyak-Ruppert averaging), which suppresses the shot-noise jitter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SPSAResult:
    x: np.ndarray                 # averaged final parameters
    x_last: np.ndarray            # last iterate
    history: np.ndarray           # mean of f(theta+) and f(theta-) at every iteration
    n_fev: int
    a: float                      # learning-rate scale actually used
    calibration_magnitude: float  # mean |g| at theta_0 (nan if not calibrated)


def default_A(maxiter: int) -> float:
    return 0.1 * maxiter


def calibrate_gain(f, x0, rng: np.random.Generator, target_step: float, c: float,
                   maxiter: int, alpha: float = 0.602, A: float | None = None,
                   steps: int = 20) -> tuple[float, float]:
    """Gain a giving a first step of about target_step per parameter; returns (a, mean |g|)."""
    x = np.array(x0, dtype=float)
    A = default_A(maxiter) if A is None else A
    mags = []
    for _ in range(steps):
        d = rng.choice((-1.0, 1.0), size=x.size)
        mags.append(abs(f(x + c * d) - f(x - c * d)) / (2 * c))
    mag = float(np.mean(mags))
    return target_step * (A + 1) ** alpha / max(mag, 1e-12), mag


def spsa_minimize(f, x0, rng: np.random.Generator, maxiter: int = 500,
                  a: float = 0.2, c: float = 0.1, alpha: float = 0.602,
                  gamma: float = 0.101, A: float | None = None,
                  gain: float | None = None, calibration_steps: int = 20,
                  average_last: int = 50, max_step: float | None = None) -> SPSAResult:
    """
    Minimize a (possibly noisy) scalar function f.

    If ``gain`` is given it is used as the gain a in a_k directly; otherwise
    the gain is calibrated at x0 so that the first step is about ``a`` per
    parameter (do not do this at a stationary point, see the module notes).
    ``max_step`` optionally clips each update in the infinity norm (off by default).
    """
    x = np.array(x0, dtype=float)
    n = x.size
    A = default_A(maxiter) if A is None else A
    n_fev = 0

    def direction():
        return rng.choice((-1.0, 1.0), size=n)

    mag = np.nan
    if gain is None:
        gain, mag = calibrate_gain(f, x, rng, a, c, maxiter, alpha, A, calibration_steps)
        n_fev += 2 * calibration_steps

    history = np.zeros(maxiter)
    keep = max(1, min(average_last, maxiter))
    tail = np.zeros((keep, n))
    for k in range(maxiter):
        ak = gain / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma
        d = direction()
        fp = f(x + ck * d)
        fm = f(x - ck * d)
        n_fev += 2
        history[k] = 0.5 * (fp + fm)
        step = ak * (fp - fm) / (2 * ck) * d
        if max_step is not None:
            step = np.clip(step, -max_step, max_step)
        x = x - step
        if k >= maxiter - keep:
            tail[k - (maxiter - keep)] = x
    return SPSAResult(x=tail.mean(axis=0), x_last=x.copy(), history=history,
                      n_fev=n_fev, a=float(gain), calibration_magnitude=mag)
