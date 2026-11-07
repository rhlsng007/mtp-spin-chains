# Phase 3: entanglement of VQE states

**Status.** Tag `phase3-complete` (merge 9110ea2); results at commit 1441fc3.
180 tests at the time (25 for Phase 3); `phase3_plots.py` 12 checks, 9 notes.
Report: "Phase 3: Entanglement Analysis of VQE States" (Claude Doc).

## Parameters

- Inputs: the Phase 2 states (p = 1 to N, N = 4 to 12, 39 fields) and the
  Phase 1 sector states as references.
- Every VQE state is compared with the exact state of its own symmetry: the nearest
  ground-space state, with overlap F_GS (the even state for the HVA; close to the
  symmetry-broken state for a symmetry-broken HEA state).
- Entropies S and S_2 and Schmidt ranks at every cut; Calabrese–Cardy fits over
  2 ≤ ℓ ≤ N/2 (left half for the HEA, whose profiles are not mirror symmetric).
- Entanglement capacity: maximal half-chain S_2 over parameters (6 starts, at most
  500 iterations, gtol 1e-9) and 200 random parameter samples.
- Periodic HVA at h_c for p = 1 to N/2, N = 4 to 12. Rank tolerance 1e-14; bound
  tolerance 1e-9; fidelity resolution 1 − F ≥ 1e-14.

## Key observations

- **Bounds.** All 3,120 states obey the Schmidt-rank bound, the Eckart–Young bound
  and the Fannes–Audenaert bound,

  $$
  S_\alpha(\ell) \le \min(p, \ell, N - \ell)\,\ln 2, \qquad
  |S_{\text{VQE}} - S_{\text{ref}}| \le T\ln(d - 1) + H_2(T), \quad T = \sqrt{1 - F_{GS}} .
  $$

  Entropy errors lie 1 to 3 orders of magnitude below the Fannes–Audenaert bound.
- **Capacity is not the bottleneck.** Both ansätze reach S_2 = p ln 2 exactly for
  every p < N/2 (to 1.6e-13), while no ground state needs more than ln 2.
- **Half-chain entropy** at p = N, N = 12, h_c: ED 0.3965, HVA 0.3986, HEA 0.3791.
  Ordered phase (h = 0.1): HVA 0.7001 (slightly above ln 2; about 0.9 at p = 6),
  HEA 0 (symmetry broken, matching the exact symmetry-broken state, 6e-6).
- **Symmetry-broken HEA states** show the entropy peak that the open-chain even
  state lacks (h = 0.70, 0.80, 0.90 for N = 8, 10, 12).
- **Central charge at h_c** (open, N = 12): ED 0.564, HVA 0.661 (p = N/2) and 0.597
  (p = N), HEA 0.380 and 0.515. The HVA approaches from above (peak 0.75 at p = 8),
  the HEA from below. Shallow circuits obey an area law (HVA c = 0 for p ≤ 2). The
  periodic HVA reaches the exact value 0.502 at p = N/2.
- Random parameters: HEA mean S within 1 % of the Page value (2-design), HVA about
  a third of it (free-fermion manifold).

## Failures and surprises

- **Maximal-entanglement search** falls short of the rank bound in five cases with
  p ≥ N/2 (HEA N = 10, p = 9, 10; HEA N = 12, p = 11, 12; HVA N = 12, p = 6): purity
  is a global cost with barren plateaus. Still at least five times what any ground
  state needs.
- **Fidelity alone misleads.** The HVA at p = N has ε ≈ 6e-6 but its central charge
  is 6 % high: entanglement structure is not guaranteed by energy accuracy.
- **Fannes–Audenaert near F = 1.** S is not Lipschitz there; Phase 3 floored
  1 − F at 1e-14 in its check. Phase 8 found that the saved bound had been stored
  without that floor: 11 states have F_GS = 1 to double precision (stored bound 0)
  and 9 of them exceed it by up to 6.6e-9. Figure 6 now recomputes the bound with
  the floor and at the precision of 1 − F; the formula reproduces every stored bound
  to 9e-16.
- The N = 12, h = 0.85 HEA point of Phase 2 was explained here (see the Phase 2
  note).

## Thesis figures

- Figure 6 (`fig06_vqe_entanglement.py`): S(N/2) against h for both ansätze,
  capacity S_2,max(p); 10 checks.
- Figure 7 (`fig07_central_charge_vs_depth.py`): c(p), open and periodic; 9 checks.
