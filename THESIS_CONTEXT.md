# MTP Thesis Context

Upload this file, together with `config.py`, at the start of every new chat about the thesis.

## 1. Thesis

**Title:** Variational Quantum Simulation of 1D Spin Chains: Entanglement, Noise Robustness, and Machine-Learning-Based Phase Classification

**Author:** Rahul Singh

**Core research question:** Can a variational quantum algorithm faithfully prepare ground states of 1D spin chains across a quantum phase transition? How does noise degrade the entanglement structure of those states? Can machine learning still identify the phase from noisy, imperfect quantum data?

**Central original contribution (target):** classification accuracy and the ML-predicted critical point as functions of noise strength, including train-clean / test-noisy experiments.

## 2. Preferences for the assistant

- Write every equation in display form on its own line, never embedded inside a sentence.
- Give complete, runnable code files and say exactly which folder each file goes in.
- Every phase ends with validation checks against known exact results.

## 3. Environment (completed in Phase 0)

| Item | Value |
|---|---|
| OS | Windows, PowerShell |
| Project path | `C:\Projects\mtp-spin-chains` |
| Python | 3.12.10, virtual environment in `.venv` |
| Activate | `.venv\Scripts\Activate.ps1` |
| Key packages | PennyLane, NumPy 2.x, SciPy, PyTorch 2.14.1 (CPU only), scikit-learn, matplotlib, pandas, tqdm, pytest |
| Version control | Git, private GitHub repo `mtp-spin-chains` |
| Status | `scripts\check_env.py`: 11/11 passed |

Scripts are always run from the project root, e.g. `python experiments\phase1_ed_sweep.py`.

## 4. Project structure

```
mtp-spin-chains/
├── config.py              # all parameters and seeds (single source of truth)
├── models/                # hamiltonians.py, exact.py           (Phase 1, done)
├── analysis/              # entanglement.py, observables.py,
│                          # free_fermion.py                      (Phase 1 done; Phase 3)
├── vqe/                   # ansatz.py, optimize.py               (Phase 2)
├── noise/                 # channels.py, mitigation.py           (Phase 4, 5)
├── ml/                    # dataset.py, classifiers.py           (Phase 6, 7)
├── utils/                 # io.py, seeding.py, plotting.py       (Phase 0, done)
├── experiments/           # one script per thesis figure
├── scripts/check_env.py   # environment verification
├── tests/                 # test_phase0.py, test_phase1.py
├── results/               # .npz data + .json metadata (gitignored)
└── figures/               # .pdf for LaTeX, .png for slides
```

## 5. Existing utilities (Phase 0)

**`config.py`** exposes a frozen global `CFG` object:

- `CFG.tfim.h_grid(noisy=False)`: 0.1 to 2.0, step 0.05 (39 points); noisy step 0.1. `CFG.tfim.J = 1.0`, `CFG.tfim.h_critical = 1.0`, `CFG.tfim.boundary = "open"`.
- `CFG.xxz.delta_grid(noisy=False)`: −0.5 to 2.0, step 0.1; noisy step 0.25.
- `CFG.cluster.h1_grid()`: 0 to 1.6, step 0.1; `CFG.cluster.h2_values = (-1.0, 0.0, 1.0)`.
- `CFG.system.sizes_exact = (4, 6, 8, 10, 12, 14)`, `sizes_noiseless = (4, ..., 12)`, `sizes_noisy = (4, 6, 8)`, `CFG.system.depths(N, noisy)`.
- `CFG.ed` (added in Phase 1): `k_per_sector=3`, `dense_max_dim=512`, `eigsh_tol=0` (machine precision), `degeneracy_tol=1e-9`, `eigsh_seed=0`, `save_states=True`, `cc_ell_min=2`, `cc_boundaries=("open","periodic")`, `xxz_cc_deltas=(0.0, 0.5)`. `CFG.cluster.boundary = "open"` was also added.
- `CFG.vqe`, `CFG.noise.strengths()` (0 plus 10 log-spaced values from 1e-4 to 5e-2), `CFG.ml`, `CFG.seeds.seeds(noisy)` (master seed 2026).
- Paths: `RESULTS_DIR`, `FIGURES_DIR`. Helper `make_grid(start, stop, step)` avoids floating-point drift.

**`utils/io.py`:** `save_results(name, arrays, params, subdir)`, `load_results(name, subdir)`, `result_exists(...)`, `load_or_compute(name, compute_fn, params, subdir, recompute=False)`. Metadata JSON stores params, full config, timestamp, git commit, package versions.

**`utils/seeding.py`:** `set_seed(seed)` returns a `np.random.Generator`; `child_seeds(master, n)`.

**`utils/plotting.py`:** `set_thesis_style()`, `save_figure(fig, name, subdir)`, colour maps `COLORS` (Okabe–Ito) and `SERIES_COLORS` with fixed colours for `ED`, `HVA`, `HEA`, `noisy`, `mitigated`.

**Conventions:** never hard-code parameters; always save raw data, then plot from saved data in a separate script; always seed; always use the thesis plot style.

## 6. Models and key formulas

**Transverse-field Ising model (open boundaries):**

$$
H_{\text{TFIM}} = -J\sum_{i=1}^{N-1} Z_i Z_{i+1} - h\sum_{i=1}^{N} X_i
$$

Critical point:

$$
h_c = J, \qquad c = \tfrac{1}{2}
$$

Squared order parameter:

$$
M^2 = \frac{1}{N^2}\sum_{i,j=1}^{N}\langle Z_i Z_j\rangle
$$

**XXZ chain (antiferromagnetic):**

$$
H_{\text{XXZ}} = J\sum_{i=1}^{N-1}\left(X_iX_{i+1} + Y_iY_{i+1} + \Delta Z_iZ_{i+1}\right)
$$

Gapless Luttinger liquid with central charge 1 for −1 < Δ ≤ 1; gapped Néel phase for Δ > 1 (BKT-type transition at Δ = 1).

**Cluster-Ising model (QCNN benchmark):**

$$
H_{\text{CI}} = -J\sum_{i} Z_i X_{i+1} Z_{i+2} - h_1\sum_i X_i - h_2\sum_i X_i X_{i+1}
$$

**Entanglement:**

$$
\rho_A = \operatorname{Tr}_B |\psi\rangle\langle\psi|, \qquad S_{\text{vN}} = -\sum_k \lambda_k \ln \lambda_k, \qquad S_\alpha = \frac{1}{1-\alpha}\ln\operatorname{Tr}\rho_A^\alpha
$$

**Calabrese–Cardy scaling, open boundaries:**

$$
S(\ell) = \frac{c}{6}\ln\!\left[\frac{2N}{\pi}\sin\!\left(\frac{\pi\ell}{N}\right)\right] + s_1'
$$

**Calabrese–Cardy scaling, periodic boundaries:**

$$
S(\ell) = \frac{c}{3}\ln\!\left[\frac{N}{\pi}\sin\!\left(\frac{\pi\ell}{N}\right)\right] + s_1
$$

**Mixed-state measures (Phase 4):**

$$
I(A\!:\!B) = S(\rho_A) + S(\rho_B) - S(\rho_{AB}), \qquad \mathcal{E}_N = \ln\left\|\rho^{T_A}\right\|_1
$$

## 7. Full implementation plan

| Phase | Content | Status |
|---|---|---|
| 0 | Environment, project structure, utilities | **Done** |
| 1 | Exact diagonalization baseline and entanglement | **Done** |
| 2 | Noiseless VQE: HVA and HEA, warm start, random restarts, barren plateaus | **Next** |
| 3 | Entanglement of VQE states, central-charge extraction vs depth | |
| 4 | Noisy density-matrix simulation (N = 4–8), shots, SPSA, noise sweeps | |
| 5 | Error mitigation: readout correction, ZNE with gate folding | |
| 6 | ML datasets: correlators, classical shadows, clean and noisy | |
| 7 | ML: logistic regression, MLP, PCA, optional QCNN, cross-noise tests | |
| 8 | Figures, reproducibility, writing | |

## 8. Phase 1: completed

**Status:** done. `tests/test_phase1.py` (100 tests) and `tests/test_phase0.py` all pass. The full sweep (`python experiments\phase1_ed_sweep.py`, N = 4 to 14, all models) takes under a minute and writes about 35 MB to `results/phase1/`. Figures: `python experiments\phase1_plots.py`, which also writes `results/phase1/phase1_summary.txt`.

**API (for Phase 2 onward):**

- `models.hamiltonians`: `hamiltonian_terms(model, N, **params)` gives a list of `(coeff, {site: "X"|"Y"|"Z"})`; `terms_to_pauli_words` gives `(coeff, "ZZII")` pairs for building PennyLane Hamiltonians; `build_hamiltonian(model, N, **params)` gives a CSR matrix; models `"tfim"` (J, h), `"xxz"` (J, delta), `"cluster"` (J, h1, h2), all with `boundary`. `symmetry_generators(model, N)` gives the spin-flip bit masks; `apply_spin_flip(psi, mask)`.
- `models.exact`: `solve_model(model, N, **params)` gives an `ExactResult` with `.ground_state`, `.ground_energy`, `.energies` (merged lowest levels), `.gap` (E1 − E0 over the full spectrum), `.gap_sector` (gap inside the ground-state sector), `.sector_states` (lowest state of each sector), `.sector_energies`, `.sector_labels`, `.ground_labels`. Also `lowest_eigenpairs`, `dense_spectrum`.
- `analysis.observables`: `zz_correlation_matrix`, `expval_z`, `magnetization_moments(psi, staggered)`, `binder_cumulant`, `centred_correlator`, `pauli_expectation(psi, ops)`, `string_order_parameter`.
- `analysis.entanglement`: `schmidt_spectrum`, `entropy_from_spectrum(lam, alpha)`, `entanglement_entropy`, `half_chain_entropy`, `entropy_profile(s)`, `reduced_density_matrix(psi, sites)`, `von_neumann_entropy_dm`, `chord_log`, `calabrese_cardy_fit(profile, N, boundary, ell_min, parity)` (fits ℓ ≤ N/2 only).
- `analysis.free_fermion`: exact open-TFIM energies and gaps, and the open XX-chain ground energy.

**Conventions fixed in Phase 1:**

- Qubit ordering matches PennyLane: wire 0 is the most significant bit. Qiskit is little-endian; convert in T4.7.
- All Hamiltonians are real, so states are stored as float64.
- Every model is diagonalized inside its Z2 spin-flip sectors (TFIM and XXZ: global flip; cluster: the two sublattice flips). This is essential: at N = 14 and h = 0.1 the TFIM parity splitting is 3e-14, and an unresolved solver returns a random mixture with S(N/2) = 0.48 instead of the cat-state value ln 2.
- The reference ground state is the lowest state over all sectors, with the fully symmetric sector preferred on exact ties. For the TFIM this is the even-parity state, which is what the parity-conserving HVA prepares. The XXZ ground state has parity (−1)^(N/2) (Marshall sign rule).
- Saved sweep files: `results/phase1/ed_tfim_N{N}`, `ed_tfim_pbc_N{N}`, `ed_xxz_N{N}`, `ed_cluster_N{N}_h2_{m1|0|1}`, `ed_cc_profiles_N{N}`. Each sweep file holds E0, E1, E2, gap, gap_sector, sector_energies, ground_sector, sector_labels, S_half, S2_half, S_profile, S2_profile, zz_corr, M2, M4, binder, Ms2 (and string_order for the cluster model). Open-chain files also hold `states[n_param, n_sectors, 2^N]`, the lowest state of every sector.
- For Phase 2: the HVA fidelity uses the even-parity state; the HEA breaks parity, so also report the ground-space fidelity (overlap with the near-degenerate sector states in the ordered phase).

**Key results (numbers in `phase1_summary.txt`):**

- TFIM energies and gaps agree with the free-fermion solution to 1e-9 for all N and h.
- Boundary-CFT gap scaling at the critical field: N times the full gap extrapolates to 3.138 (π, error 0.11 %); N times the even-sector gap extrapolates to 12.596 (4π, error 0.23 %).
- Binder crossings h* = 0.921, 0.952, 0.967, 0.976 for size pairs (6,8) to (12,14), converging to the critical field.
- Central charge, periodic chains at N = 14: TFIM c = 0.5020 ± 0.0003; XXZ c = 1.028 (Δ = 0) and 1.032 (Δ = 0.5). Open chains are biased at these sizes: TFIM c = 0.561; the XX chain oscillates strongly between even ℓ (1.18) and odd ℓ (0.54). An oscillation-corrected fit did not help. Use periodic chains for central charges.
- The half-chain entropy of the open-chain even-parity state does not peak at the critical field. The ordered phase sits on the ln 2 plateau, and S(h_c) = 0.411 < ln 2 at N = 14 (it would exceed ln 2 only for N of about 400). The transition appears as the steepest drop of S(N/2): h = 0.80 to 0.95 for N = 6 to 14. For periodic chains the entropy peak is visible (h = 0.55 to 0.90 for N = 6 to 14).
- XXZ open chains show an N/2-odd / N/2-even split in S(N/2), from bond dimerization. No sharp feature at the BKT point Δ = 1 is expected at these sizes.
- Cluster model: string order equals 1 at h1 = h2 = 0 with a fourfold-degenerate ground state (edge modes), and falls to 0.01 at h1 = 1.6 (h2 = 0). At h2 = ±1, h1 = 0 the chain is critical (string order 0.09).
