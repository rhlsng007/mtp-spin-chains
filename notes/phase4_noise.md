# Phase 4: noise robustness

**Status.** Tag `phase4-complete` (merge 99928f4); code at c50fd87 and c0291c3
(two commits with the same message); experiment B run on 77aaf99. 240 tests at the
time (60 for Phase 4, 6 PennyLane `default.mixed` cross-checks); `phase4_plots.py`
21 checks, 14 notes. Report: "Phase 4: Noise Robustness of VQE States for the
Transverse-Field Ising Chain" (Claude Doc). T4.7 (IBM calibration noise through
Qiskit) skipped by decision.

## Parameters

- Exact density-matrix simulation, N = 4, 6, 8, depths p = 1 to N/2, 20 fields
  (h = 0.1 to 2.0, step 0.1).
- Channels after every gate: depolarizing (two-qubit error equal to one-qubit),
  amplitude damping and dephasing on every qubit a gate touches; RZ and the HVA
  Hadamard layer noisy. Strengths p = 0 and 10 log-spaced values 1e-4 to 5e-2.
- Error budget λ = p (n_1q + n_2q); N = 8, p = 4: 68 noisy gates (HVA), 108 (HEA).
- Experiment A: the Phase 2 optimum evaluated under noise (164 s). Finite-shot
  energies with 1,000, 5,000, 10,000 shots per basis, 50 repetitions.
- Experiment B: SPSA under noise, 10,000 shots per basis, 500 iterations, at
  p = N/2 and h = 0.5, 1.0, 1.5; one warm and three cold starts; gain calibrated
  once per problem at a cold point; Polyak average of the last 50 iterates;
  21,417 s (5 h 57 min).
- Seeds: experiment keys 11 (eval) and 12 (train); common random numbers across
  channels and strengths.

## Key observations

- **Validation.** Zero noise reproduces the Phase 2 states (3.8e-15) and energies
  and the Phase 3 entropies; density matrices are valid to 1e-15; purity falls
  monotonically; weak-noise slopes of 1 − purity and 1 − F_ideal are 0.97 to 1.00.
  The no-error branch gives exact fidelity bounds, which every state obeys:

  $$
  F_{\text{ideal}} \ge (1-p)^{n_{1q}+n_{2q}} \ (\text{depolarizing}), \qquad
  F_{\text{ideal}} \ge (1-q)^{n_{1q}+2n_{2q}}, \ q = \tfrac{1-\sqrt{1-\gamma}}{2} \ (\text{dephasing}).
  $$

- **Error-budget collapse.** The infidelity of every ansatz, size and depth falls
  on one curve per channel,

  $$
  1 - F_{\text{ideal}} \approx 1 - e^{-\kappa\lambda}, \qquad
  \kappa = 0.734 \ (\text{depolarizing}), \ 0.359 \ (\text{amplitude damping}), \ 0.318 \ (\text{dephasing}),
  $$

  over 3,140 states with λ ≤ 1. Depolarizing is tight (5 to 95 %: 0.69 to 0.84);
  the damping channels are broad and state dependent.
- **The HEA's noiseless advantage is gone** from p = 1.0e-4 (depolarizing): more
  noisy gates (108 against 68).
- **Mixed-state subtlety.** Under depolarizing noise S(ρ_A) rises at all 339
  entangled states while E_N falls at all of them, reaching exactly 0 at p = 0.05
  (entanglement sudden death; N = 8, p = 4, h = 1: HVA S_A 0.340 → 2.427,
  E_N 0.615 → 0). The hashing bound (S(A|B) < 0) is lost at λ = 0.116 / 0.120
  (depolarizing, HVA / HEA), 0.214 / 0.350 (amplitude damping), 0.279 / 0.447
  (dephasing).
- **Energy witness** (ε < ε_prod = 0.044 at h = 1) lost at λ = 0.157 / 0.113
  (depolarizing), 0.259 / 0.314 (amplitude damping), 0.518 / 0.410 (dephasing):
  two independent certificates fail at λ ≈ 0.1 to 0.5.
- **Training under noise.** Warm starts equal experiment A. Under amplitude damping
  the cold-trained HEA (median of three starts) reaches ε = 0.049 at p = 0.05
  against 0.454 for experiment A, with purity 0.954 and E_N = 0.034: just above
  ε_prod, by retreating to a nearly pure, nearly unentangled state. Under
  dephasing the HVA gains nothing. Lower energy under noise comes from abandoning
  entanglement, so energy error is a misleading benchmark for noisy many-body VQE.
- Shot floor σ/|E0| = 1.1e-2, 5.0e-3, 3.5e-3 at 1,000, 5,000, 10,000 shots.

## Failures and surprises

- **E_N rising with noise.** 15 near-product, symmetry-broken HEA states
  (E_N(0) = 0.012 to 0.069) gain negativity under amplitude damping or dephasing
  (largest 0.015 → 0.129): mid-circuit noise spoils the cancellation by which the
  circuit undoes its own entanglement. The monotonicity check applies to
  E_N(0) ≥ 0.1, a threshold set after this diagnostic; stated as such.
- **Mutual information is not monotone** (falls at 311 to 319 of 339 points):
  noise inside the circuit is followed by gates crossing the cut.
- **SPSA gain calibration at an optimum** is forbidden: the symmetric difference
  cancels to third order there and the gain explodes. The gain is calibrated at a
  cold point and scaled by 0.1 for warm starts.
- **Cold-SPSA floors** at zero noise (HEA 0.042 to 0.057, at the product-state
  bound): cold SPSA from small angles barely leaves the product manifold, which
  partly explains the training crossover.
- **Definitions found in Phase 8.** Phase 4 reports "cold" values as the median of
  the three cold starts (not the best); Figure 10 now uses the median and
  reproduces all 24 report values.
- T4.7 skipped; the realistic noise model was built in Phase 5 instead.

## Thesis figures

- Figure 8 (`fig08_fidelity_collapse.py`): 1 − F against λ, three channels; 14 checks.
- Figure 9 (`fig09_mixed_state_entanglement.py`): S_A, E_N, S(A|B); 11 checks.
- Figure 10 (`fig10_training_under_noise.py`): experiment A against SPSA; 8 checks.
