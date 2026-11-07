# Phase 7: machine-learning phase classification

**Status.** Tag `phase7-complete`; results at commit 87d4b42. 364 tests at the time
(14 for Phase 7); `phase7_plots.py` all checks passed. 57 jobs in 197 s
(12 workers), 118 files in `results/phase7/`, including three CSV tables.

## Parameters

- Classifiers: logistic regression (C = 1, at most 5,000 iterations) and an MLP
  (hidden layers 32, 16; L2 1e-3; batch 64; 300 epochs; early stopping off);
  5 training seeds; features standardized on the training rows only.
- Kinds: exact features and classical shadows; feature sets "corr" (main) and
  "compact".
- h_c^ML: the field where the isotonic fit of the field-averaged output P(h)
  crosses 1/2 (NaN without a crossing). Uncertainty: seed spread and shot error
  from the 4 test shadows, in quadrature.
- Experiments: clean (ED N = 4 to 14, VQE N = 4 to 12), exact free fermions
  (N = 4 to 64), noise (N = 4, 6, 8, p = N/2, three variants, five training modes),
  ZNE (two window strengths), shots (100 to 1,000 snapshots), learning curves
  (1 to 16 shadows per training field).
- Training modes under noise: ED (`clean_ed`), noiseless circuit (`clean_vqe`),
  train = test noise (`noisy`), all 11 strengths pooled (`mixed`), all but the
  tested strength (`loo`). Default variant: readout-corrected shadows.
- Drift onset: first strength with

  $$
  |h_c^{ML}(p) - h_c^{ML}(0)| > 2\sqrt{\sigma_{\text{tot}}^2(p) + \sigma_{\text{tot}}^2(0)} .
  $$

## Key observations

- **Validation.** Classifiers on ED features equal those on free-fermion
  correlators (1.1e-14); zero-noise data reproduce the noiseless classifiers
  (5.6e-15; ED-trained 2.1e-14); readout correction is exact for exact features at
  p = 0; shadow and exact h_c^ML agree within 3σ + 0.02 (12 of 12); PC1 correlates
  with M² at 0.9975 to 0.9999; deep-field accuracy 100 %.
- **Finite-size drift.** h_c^ML (logistic regression, exact) = 0.761, 0.798,
  0.826, 0.845, 0.860, 0.871 for N = 4 to 14 and 0.878, 0.901, 0.917, 0.928, 0.936
  for free fermions at N = 16 to 64. A linear fit in 1/N over N ≥ 16 extrapolates
  to 0.955, below h_c: a fixed-threshold hypothesis, not derived.
- **Transfer to VQE.** An ED-trained classifier works on HEA states (0.74 to 0.87)
  but not on the HVA at p = N/2 (0.59 to 0.66: its ordered-phase M² is about 0.65);
  the HVA at p = N recovers (0.80 to 0.86).
- **Central result: noise** (N = 8, λ = 320 p, readout-corrected shadows,
  logistic regression). Classifiers trained on clean data drift from λ = 0.15
  (trained on ED) and 0.32 (noiseless circuit) and find no transition at λ = 3.2
  (accuracy 0.526 = 20/38: everything called disordered). Trained at the test noise
  they keep h_c^ML within 0.07 (HVA 0.872 → 0.802, HEA 0.837 → 0.796) and accuracy
  0.93 / 0.91 at λ = 3.2, where the fidelity to the noiseless state is only

  $$
  F \approx e^{-\kappa\lambda} = e^{-0.73 \times 3.2} \approx 0.10 .
  $$

  Mixed training: accuracy 0.91 to 0.94 throughout, drift onset λ = 0.69 (HVA)
  and 1.49 (HEA).
- **λ collapses the sizes.** Clean-trained drift onsets at N = 6 lie at λ = 0.17
  to 0.81, at N = 8 at 0.15 and 0.32. At N = 4 (λ at most 0.72) only the
  ED-trained HVA drifts (λ = 0.33), and it starts from an already misplaced value
  (h_c^ML = 0.623 against 0.815 when trained on the circuit; the p = N/2 transfer
  failure), sliding to 0.432 at p = 1e-2; every other mode at N = 4 stays within
  2σ.
- **ZNE** (window λ = 0.07 to 0.17): exact feature error cut to 0.7 to 5.6 % of
  the unmitigated error; h_c^ML within 0.004 of the noiseless circuit at all 12
  points (HVA N = 8: 0.850 → 0.870 and 0.827 → 0.868, noiseless 0.870).
- **Statistics.** σ_tot = 0.006 to 0.02, far below the noise-induced shifts (0.04 to
  0.7); learning curves are flat (22 training rows within 0.015 of 352).

## Failures and surprises

- **MLP early stopping** stopped untrained networks on separable data (seed spread
  of h_c^ML 0.12 to 0.21); turned off (spread 0.015 to 0.021).
- **Accuracy has a finite-size ceiling:** fields between h_c^ML and 1 always count
  as errors. Noise levels are compared by the drift of h_c^ML instead.
- **Leave-one-out** interpolates between noise levels but does not extrapolate: at
  the strongest strength h_c^ML = 0.982 (HVA) and 0.741 (HEA).
- **Snapshots:** 100 per shadow bias the ED h_c^ML by +0.025 (0.845 against 0.820).
- **Definition found in Phase 8.** "No crossing implies accuracy 20/38 or 18/38" is
  not an identity (accuracies at missing crossings range 0.526 to 0.592, since
  single shadows can fall below 1/2 while the averaged curve does not). Checked
  instead: h_c^ML is missing exactly when the isotonic fit never crosses 1/2 (all
  26,400 entries).
- **Transcription error found in Phase 8.** The Phase 7 report and THESIS_CONTEXT.md
  stated "N = 4: no 2-sigma drift in any mode". The Phase 7 summary itself, and
  the Figure 14 recomputation, give one drift: the ED-trained HVA at λ = 0.33.
  Corrected in the context file; the Phase 7 report text needs the same fix.
- T7.5 (QCNN, cluster-Ising) remains optional future work.

## Thesis figures

- Figure 13 (`fig13_ml_clean.py`): output curves, h_c^ML against 1/N, transfer to
  VQE; 10 checks.
- Figure 14 (`fig14_hc_vs_noise.py`), central figure: h_c^ML against λ, five modes,
  drift for N = 4, 6, 8; 10 checks.
- Figure 15 (`fig15_accuracy_and_zne.py`): accuracy against λ, ZNE recovery;
  7 checks.
- Figure 16 (`fig16_error_budget_window.py`): the common λ window (see the Phase 8
  note); 7 checks.
