# Phase 2: noiseless VQE

**Status.** Tag `phase2-complete` (merge 0f7015c); results at commit 5ab2a0b.
155 tests at the time (47 for Phase 2, 7 PennyLane cross-checks);
`phase2_plots.py` 20 checks. Report: "Phase 2: Noiseless VQE for the
Transverse-Field Ising Chain" (Claude Doc).

## Parameters

- Ansätze: Hamiltonian variational ansatz (HVA, 2p parameters) and
  hardware-efficient ansatz (HEA, RY–RZ layers with a CNOT staircase, 2N(p + 1)
  parameters).
- N = 4 to 12, depths p = 1 to N, 39 fields (h = 0.1 to 2.0, step 0.05).
- Optimizer L-BFGS-B with exact adjoint gradients (gradient tolerance 1e-8,
  ftol 1e-15, at most 1,000 iterations). At every field: warm start from the
  previous field, the depth p − 1 optimum padded with an identity layer (HVA), and
  2 random restarts; 8 restarts at the first field and at h = 0.5, 1.0, 1.5.
  Ties within 1e-10 go to the warm start.
- Sweep direction: HVA from h = 2.0 down, HEA from h = 0.1 up.
- Barren plateaus: 500 uniform parameter draws, N = 4 to 14, HEA at p = N and
  p = 2, HVA at p = N, local cost ⟨Z_c Z_c+1⟩ and energy per site at h = 1.
- Seeds `SeedSequence([2026, experiment, ansatz, N, p, field])`: results
  independent of worker count and job order.

## Key observations

- **Validation.** States agree with dense circuits (1e-12) and PennyLane (1e-10);
  adjoint gradients with the parameter-shift rule (1e-11). The variational
  principle holds for every state; the periodic HVA at p = N/2 is exact
  (ε ≤ 2.2e-14, Ho and Hsieh); N = 4 is exact at p = 4 for both ansätze.
- **Accuracy at h_c, p = N.** HVA ε ≈ 6e-6 for N = 6 to 12; HEA ε = 2.4e-8,
  2.1e-5, 1.2e-4, 5.0e-4 for N = 6, 8, 10, 12. At p = N/2 the HEA is the more
  accurate (2.5e-4 against 2.6e-3 at N = 8).
- **Required depth** p* (smallest p with ε < 1e-3) near h_c, as in the Phase 2
  report: HVA 5, 7, 8, 9 for N = 6 to 12; HEA 2, 4, 7, not reached (the hardest
  p* in 0.75 ≤ h ≤ 1.25 is in the Figure 3 notes of the manifest). The
  HVA's hardest region is the ordered phase, where it does not reach 1e-3 within
  p ≤ N for N ≥ 8; the HEA's hardest field lies on the ordered side of h_c.
- **Symmetry breaking.** In the ordered phase the HEA prepares symmetry-broken
  states (F = 1/2, F_GS ≥ 0.999) whose error sits on the parity-splitting floor,

  $$
  \epsilon \;\ge\; \frac{1 - \langle P\rangle}{2}\,\frac{E_{\text{odd}} - E_{\text{even}}}{|E_0|} ,
  $$

  with minimum ratio ε / bound 1.005, 1.023, 1.086, 1.193 for N = 6 to 12. The
  HVA conserves parity exactly (⟨P⟩ = 1).
- **Barren plateaus.** HEA at p = N: Var ∝ exp(−bN) with b = 0.690 ± 0.008, the
  2-design value ln 2 = 0.693; HEA at p = 2: b = 0.29; HVA: no exponential decay
  (variance levels off near 0.5; free-fermion Lie algebra).
- Warm starts (N = 8, p = 4): HVA 13.5 times cheaper per sweep at equal accuracy.

## Failures and surprises

- **HEA optimizer budget.** 44 %, 64 %, 77 %, 84 % of HEA runs stop at the
  iteration limit (N = 6 to 12), against at most 2 % for the HVA. The HEA error
  is not monotonic in depth at N = 12, and an independent Linux rerun reproduced
  HEA numbers only to two digits (HVA to every printed digit).
- **HEA warm start.** Cheaper but less reliable (79 % against 97 % of fields with
  ε < 1e-3): it carries the symmetry-broken branch into the paramagnet.
- **Apparent failure at N = 12, p = 12, h = 0.85** (⟨P⟩ = 0.15, F_GS = 0.57).
  Resolved in Phase 3: a symmetry-broken state on the parity floor
  (ε = 1.2 × bound), one grid point past the ground-space cutoff (splitting / gap
  = 0.108 > 0.1). No rerun.
- **Definition of p\*.** Phase 8 found that p* at exactly h = J (HVA 5, 6, 7, 7)
  differs from the "near h_c" values above; Figure 3 now plots p*(h) and checks it
  at every field against the saved `required_depth_vs_field`.

## Thesis figures

- Figure 3 (`fig03_vqe_error_vs_depth.py`): ε against p at h_c, p*(h); 11 checks.
- Figure 4 (`fig04_vqe_error_vs_field.py`): ε against h, parity floor, ⟨P⟩;
  6 checks.
- Figure 5 (`fig05_barren_plateaus.py`): gradient variance against N; 7 checks.
