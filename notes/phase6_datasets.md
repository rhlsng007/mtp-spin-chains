# Phase 6: machine-learning datasets

**Status.** Tag `phase6-complete` (merge 53189d3); datasets at commit 27b3527.
350 tests at the time (27 for Phase 6); `phase6_plots.py` 16 checks. Report: "Phase 6: Machine-Learning Datasets for Phase Classification of the Transverse-Field Ising Chain"
(Claude Doc). 38 datasets
in 170 s (12 workers), 78 files in `results/phase6/`.

## Parameters

- Sources: ED (N = 4 to 14), noiseless VQE (N = 4 to 12, p = N/2 and N, both
  ansätze), noisy VQE under the Phase 5 hardware-like model (N = 4, 6, 8,
  p = N/2, 11 strengths 0 and 1e-5 to 1e-2), and ZNE-mitigated noisy VQE inside
  the λ window 0.05 to 0.3 (two strengths per dataset; λ = 0.069 and 0.149 at
  N = 8).
- Noisy variants: gate noise only ("gate"), gate noise plus readout error ("meas"),
  readout-corrected with tensored calibration and the linear inverse ("ro").
- Features: ZZ correlators and transverse magnetizations ("corr"), and a compact
  set; exact values and classical-shadow estimates.
- Classical shadows: random Pauli bases, 1,000 snapshots per shadow, 20 shadows
  per field (repetitions 1 to 16 for training, 17 to 20 for testing); 39 fields;
  training fields h < 0.6 (ordered, label 0) and h > 1.4 (disordered, label 1);
  h = h_c unlabelled.
- Common random numbers: noisy strength 0 reuses the noiseless bases and snapshots.

## Key observations

- **Validation.** ED features equal free-fermion correlators (Pfaffians) to 1.4e-14;
  energies rebuilt from the features equal ED and Phase 2; the ZZ matrix is
  positive semidefinite; shadow means are unbiased and their spread matches the
  analytic variance,

  $$
  \sigma^2_{ZZ} = 9 - \langle Z_iZ_j\rangle^2, \qquad \sigma^2_{X} = 3 - \langle X_i\rangle^2 .
  $$

  Readout correction at strength 0 equals gate-only features exactly; noisy
  strength 0 reproduces the noiseless VQE features, bases and snapshots.
- **Order parameter.** The M² margin between the training regions grows with N
  (ED: 0.376 at N = 4 to 0.711 at N = 14). At h_c,

  $$
  M^2(h_c) \propto N^{-2\Delta_\sigma} = N^{-1/4},
  $$

  with local slopes −0.30 and −0.29 (N = 10 to 14), approaching −1/4 (two pairs of
  sizes, not a fit).
- **Noise compresses but separates.** For the HVA at N = 8, M² at h = 0.1 falls
  from 0.65 to about 0.2 at p = 1e-2 (λ = 3.2); the training regions stay separated
  in exact features at every strength (margin 0.039 for the HVA, 0.073 for the HEA
  at N = 8). The HVA starts with about half the ED margin (0.322 against 0.589):
  its ordered-phase M² is only 0.65 (Phase 2 limit). The HEA margin matches ED
  (0.599): its symmetry-broken states reproduce the ground-space correlators.
- **Readout.** Bias of 0.049 to 0.059 in the features at strength 0, removed exactly
  by the correction; with gate noise a SPAM residual grows to 1.5e-2 to 2.3e-2 at
  p = 1e-2.
- **ZNE** brings the exact features back to the noiseless curve (median remaining
  error 0.7 % to 4.1 % of the unmitigated), at about twice the shadow variance.

## Failures and surprises

- **Variance check statistic.** With 20 shadows the median of the variance ratio is
  expected below 1 (median of χ²₁₉/19 = 0.965), so the check uses the mean ratio.
- **SPAM residual** after readout correction (noisy calibration gates), not removed
  by design; recorded per dataset.
- **Repository housekeeping.** A commit first landed on `main` instead of the phase
  branch (moved to a branch and reset); the tag `phase6-complete` initially pointed
  at the wrong merge (corrected); Phase 3 and 4 results were found unbacked and
  backed up (combined archive, 747 files).

## Thesis figures

The datasets are the inputs of Figures 13 to 15 (Phase 7). Their identities are
re-checked there: λ = p(n_1q + 10 n_2q) stored identically in Phases 6 and 7, the
Phase 6 ZNE datasets use the Richardson weights, and zero-noise data reproduce the
noiseless classifiers.
