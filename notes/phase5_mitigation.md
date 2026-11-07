# Phase 5: error mitigation

**Status.** Tag `phase5-complete` (merge 06622d4); results at commit 36a8c0d. 311
tests at the time (83 for Phase 5); `phase5_plots.py` 27 checks, 7 notes. Report:
"Phase 5: Error Mitigation of Noisy VQE States for the Transverse-Field Ising
Chain" (Claude Doc).

## Parameters

- Hardware-like ("realistic") noise model: one-qubit depolarizing p, two-qubit
  depolarizing 10p, amplitude damping and dephasing p on every touched qubit,
  virtual (noiseless) RZ; readout error p01 = 0.01, p10 = 0.02, uncorrelated.
  Also depolarizing, amplitude damping and dephasing alone.
- Strengths p = 0 and 10 log-spaced values 1e-5 to 1e-2; N = 4, 6, 8, p = 1 to
  N/2, 20 fields (72 files, 1,932 s).
- Error budget λ = p (n_1q + 10 n_2q) (depolarizing, realistic) or
  p (n_1q + 2 n_2q) (damping channels); N = 8, p = 4: n_1q = 40, n_2q = 28,
  λ = 320 p.
- Readout correction: linear inverse of the calibration matrix (exact, tensored or
  full; 1,000 to 100,000 calibration shots); calibration X gates carry gate noise.
- Zero-noise extrapolation: global and local gate folding at scales 1, 3, 5;
  Richardson (main), linear least squares, exponential.
- Shots: 10,000 per basis and scale, 50 repetitions, each with its own calibration.
- Error measure δ = |E − E_ideal| / |E0| against the noiseless circuit; "at VQE
  accuracy" means δ below the noiseless VQE error.

## Key observations

- **Richardson weights** for λ = 1, 3, 5 and the variance amplification

  $$
  E^* = \tfrac{15}{8}E(1) - \tfrac{5}{4}E(3) + \tfrac{3}{8}E(5), \qquad
  \sum_k c_k^2 = \tfrac{167}{32} \ (\text{linear fit: } \tfrac{35}{24}).
  $$

- **Readout** alone biases the energy by 3.3 % (N = 8, p = 4, h = 1); exact
  calibration removes it to 1e-14. ZNE alone stays at this bias: folding does not
  scale measurement error.
- **Combined mitigation.** At p = 1e-4: unmitigated 4.3e-2 (HVA) / 4.7e-2 (HEA),
  readout-corrected 1.1e-2 / 1.5e-2, readout + Richardson 2.0e-5 / 2.0e-5. At VQE
  accuracy up to p = 9.62e-4 (HVA, λ = 0.31) and 3.50e-4 (HEA, λ = 0.11); readout
  correction alone up to 2.4e-5 (HVA) and never (HEA). Extrapolation fails beyond
  λ ≈ 0.7.
- **Folding bias.** With A (B) the effect of errors after (before) each gate,

  $$
  E(\lambda) - E_{\text{ideal}} = p\left[\frac{A-B}{2} + \lambda\,\frac{A+B}{2}\right] + O(p^2).
  $$

  Depolarizing noise commutes with the gates (A = B): weak-noise orders 1.000,
  1.998, 2.998 (unmitigated, linear, Richardson); median residual / unmitigated at
  p = 1e-5 is 5e-7. Amplitude damping 9.1 %, dephasing 5.2 %, hardware-like
  1.1 %. Local and global folding agree (ratio 1.0000). The half-final-layer
  identity holds to 7.4e-5 in 36 live cases.
- **Shots.** Floor σ/|E0| = 3.6e-3 at 10,000 shots; readout + Richardson amplifies
  the readout-corrected spread 2.04 (HEA) and 2.31 (HVA) times at p = 1e-5
  (Figure 11; Phase 5 report: 2.2 to 2.6 over p ≤ 1e-3), against the predicted
  √(167/32) = 2.28. No method reaches VQE accuracy
  at 10,000 shots: readout + ZNE needs about 9.5e4 (HVA) and 1.0e7 (HEA) shots per
  basis and scale factor.
- Mitigation works in the window λ ≈ 0.1 to 0.3, inside the Phase 4 window where
  the entanglement certificates still hold (λ ≈ 0.1 to 0.5).

## Failures and surprises

- **ZNE alone fails** under readout error (bias untouched by folding).
- **Non-variational mitigated energies.** 250 of 14,400 combined energies fall
  below E0 (249 HEA, all λ ≤ 1; most negative −0.8 % under amplitude damping).
  Mitigated energies are not variational bounds.
- **SPAM.** Noisy calibration gates leave a residual: under depolarizing noise
  SPAM is the whole combined residual (ratio to ideal calibration 4,300 HVA /
  590 HEA); under amplitude damping it cancels most of the folding bias (0.05),
  mechanism not derived; reported as an observation.
- **Single Richardson slopes scatter** (65 % within ±0.3 of 3) because a small
  second-order term survives folding; orders are quoted as medians.
- **Simplex projection** of corrected distributions removes negative
  probabilities but raises the energy error by 10 to 30 %; energies use the linear
  inverse.
- Open: literature check of the half-final-layer folding-bias form before claiming
  it as new.

## Thesis figures

- Figure 11 (`fig11_error_mitigation.py`): unmitigated, readout, ZNE, combined,
  exact and with shots; 12 checks.
- Figure 12 (`fig12_folding_bias.py`): residual by channel, folding-bias ratios;
  6 checks.
