# Phase 1: exact diagonalization

**Status.** Tag `phase1-complete`; results at commit 6051c9d. `tests/test_phase1.py`
(100 tests) passes. Full sweep under a minute, about 35 MB in `results/phase1/`
(85 files).

## Parameters

- Models: TFIM (J = 1, h = 0.1 to 2.0, step 0.05, 39 fields), XXZ (Δ = −0.5 to
  2.0, step 0.1), cluster-Ising (h2 ∈ {−1, 0, 1}); open chains, plus periodic TFIM
  and XXZ for central charges.
- Sizes N = 4, 6, 8, 10, 12, 14.
- Sector-resolved diagonalization in the Z2 spin-flip sectors (cluster model: both
  sublattice flips), 3 eigenpairs per sector; dense `eigh` up to dimension 512,
  `eigsh` (tol 0, fixed start vector) above.
- Calabrese–Cardy fits over 2 ≤ ℓ ≤ N/2 at h = J (TFIM) and Δ = 0, 0.5 (XXZ).

## Key observations

- **Free fermions.** E0 and E1 − E0 equal the free-fermion solution at every N and
  field to 1e-9 (thesis check: below 1e-12). At h_c the gap has the closed form

  $$
  E_1 - E_0 = 4J\sin\frac{\pi}{2(2N+1)}, \qquad N(E_1 - E_0) \to \pi ,
  $$

  the boundary-CFT value (quadratic fit in 1/N over N = 8 to 14: π to 2e-5).
- **Binder crossings** h* = 0.921, 0.952, 0.967, 0.976 for the pairs (6, 8) to
  (12, 14): below h_c and converging to it.
- **Central charges** (periodic, N = 14): TFIM c = 0.5020 ± 0.0003; XXZ c = 1.028
  (Δ = 0) and 1.032 (Δ = 0.5). Open chains are biased at these sizes (TFIM 0.561).
- **Entanglement.** The ordered phase sits on the cat-state plateau S(N/2) = ln 2
  (to 2e-4 at N = 4, 6e-6 for N ≥ 6); the paramagnet obeys an area law
  (S(N/2) = 0.0888 at h = 2, the same for N = 12 and 14 to 2e-6).
- **No entropy peak at h_c on open chains.** S(h_c) = 0.411 < ln 2 at N = 14; the
  transition shows as the steepest drop of S(N/2) at h = 0.80 to 0.95 (N = 6 to
  14). With

  $$
  S(h_c) \approx \frac{c}{6}\ln N + s', \qquad c = \tfrac12 ,
  $$

  the critical value would exceed ln 2 only for N of about 400. Periodic chains do
  peak (h = 0.55 to 0.90).
- XXZ open chains split by the parity of N/2 (bond dimerization); no feature at
  the BKT point Δ = 1 at these sizes. Cluster model: string order 1 at
  h1 = h2 = 0 (fourfold degenerate, edge modes), 0.01 at h1 = 1.6.

## Failures and surprises

- **Unresolved degeneracy.** At N = 14, h = 0.1 the parity splitting is 3e-14; a
  solver that ignores the symmetry returns a random mixture of the two cat states
  with S(N/2) = 0.48 instead of ln 2. Fixed by diagonalizing inside each Z2 sector
  and choosing the lowest state over all sectors (symmetric sector on exact ties).
- **Open-chain central charges.** Biased at N ≤ 14 (TFIM 0.561; the XX chain
  alternates between 1.18 for even ℓ and 0.54 for odd ℓ). An oscillation-corrected
  fit did not help; central charges are taken from periodic chains.

## Thesis figures

- Figure 1 (`python experiments\thesis\fig01_tfim_ed_baseline.py`): gap, Binder
  cumulant, half-chain entropy; 11 checks.
- Figure 2 (`python experiments\thesis\fig02_central_charges.py`): Calabrese–Cardy
  fits and c against 1/N; 7 checks.
