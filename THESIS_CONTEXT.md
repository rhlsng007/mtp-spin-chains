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
- Phase reports are written as Claude Docs (exported to Word), in the layout of the Phase 1 report: Summary, Introduction, Models or Methods, Validation, Results, Discussion and implications, Reproducibility, References. References written from memory are flagged for checking.

## 3. Environment (completed in Phase 0)

| Item | Value |
|---|---|
| OS | Windows 11, PowerShell |
| Hardware | Intel Core Ultra 7 255H (16 cores: 6 performance, 8 efficiency, 2 low-power), 23 GB RAM |
| Project path | `C:\Projects\mtp-spin-chains` |
| Python | 3.12.10, virtual environment in `.venv`; base interpreter `C:\Users\RAHUL SINGH\AppData\Local\Programs\Python\Python312\python.exe` |
| Activate | `.venv\Scripts\Activate.ps1` |
| Key packages | PennyLane 0.45.1, NumPy 2.5.4, SciPy 1.18.1, PyTorch 2.14.1 (CPU only), scikit-learn 1.9.1, matplotlib 3.11.2, pandas, tqdm, pytest |
| Version control | Git, private GitHub repo `mtp-spin-chains` |
| Status | `scripts\check_env.py`: 11/11 passed |

Scripts are always run from the project root, e.g. `python experiments\phase1_ed_sweep.py`.

**Long runs on this laptop (settled in Phase 2):**

- Parallel experiment scripts: use `--workers 12` (the low-power cores add little).
- Python is exempted from Windows power throttling (`powercfg /powerthrottling disable /path <base interpreter>`); without it a lone background worker drops to an efficiency core.
- Sleep is disabled on mains power (`powercfg /change standby-timeout-ac 0`); Modern Standby otherwise pauses all workers. Check for pauses with Kernel-Power events 506/507.
- The Desktop is redirected by OneDrive; use `$HOME\Downloads` in commands.

**Git workflow (settled in Phase 2):** one branch per phase (`phaseN-...`); commit the code before running experiments, so every result records the commit that produced it; merge with `--no-ff`; tag `phaseN-complete`. Tags are lightweight, so push them explicitly (`git push origin <tag>`); `--follow-tags` skips them. Tags so far: `phase1-complete` (6051c9d), `phase2-complete` (0f7015c), `phase3-complete` (9110ea2).

## 4. Project structure

```
mtp-spin-chains/
├── config.py              # all parameters and seeds (single source of truth)
├── models/                # hamiltonians.py, exact.py           (Phase 1, done)
├── analysis/              # entanglement.py, observables.py,
│                          # free_fermion.py                      (Phase 1, done)
│                          # vqe_entanglement.py                  (Phase 3, done)
├── vqe/                   # ansatz.py, statevector.py, optimize.py, gradients.py,
│                          # metrics.py, pennylane_ref.py         (Phase 2, done)
├── noise/                 # channels.py, mitigation.py           (Phase 4, 5)
├── ml/                    # dataset.py, classifiers.py           (Phase 6, 7)
├── utils/                 # io.py, seeding.py, plotting.py       (Phase 0, done)
├── experiments/           # one script per thesis figure
├── scripts/               # check_env.py, phase2_report_data.py, phase3_report_data.py
├── tests/                 # test_phase0.py ... test_phase3.py (180 tests)
├── results/               # .npz data + .json metadata (gitignored; back up separately)
└── figures/               # .pdf tracked in git, .png gitignored
```

`results/` is gitignored, so all result data exist only locally. Keep a backup zip of `results\phase1` to `results\phase3` outside the repo; regenerating Phase 2 takes about 3 hours, Phase 3 about 6 minutes.

## 5. Existing utilities (Phase 0)

**`config.py`** exposes a frozen global `CFG` object:

- `CFG.tfim.h_grid(noisy=False)`: 0.1 to 2.0, step 0.05 (39 points); noisy step 0.1. `CFG.tfim.J = 1.0`, `CFG.tfim.h_critical = 1.0`, `CFG.tfim.boundary = "open"`.
- `CFG.xxz.delta_grid(noisy=False)`: −0.5 to 2.0, step 0.1; noisy step 0.25.
- `CFG.cluster.h1_grid()`: 0 to 1.6, step 0.1; `CFG.cluster.h2_values = (-1.0, 0.0, 1.0)`.
- `CFG.system.sizes_exact = (4, 6, 8, 10, 12, 14)`, `sizes_noiseless = (4, ..., 12)`, `sizes_noisy = (4, 6, 8)`, `CFG.system.depths(N, noisy)` (1..N noiseless, 1..N/2 noisy).
- `CFG.ed` (added in Phase 1): `k_per_sector=3`, `dense_max_dim=512`, `eigsh_tol=0` (machine precision), `degeneracy_tol=1e-9`, `eigsh_seed=0`, `save_states=True`, `cc_ell_min=2`, `cc_boundaries=("open","periodic")`, `xxz_cc_deltas=(0.0, 0.5)`. `CFG.cluster.boundary = "open"` was also added.
- `CFG.vqe` (extended in Phase 2, see Section 9), `CFG.ent` (added in Phase 3, see Section 10), `CFG.noise.strengths()` (0 plus 10 log-spaced values from 1e-4 to 5e-2), `CFG.ml`, `CFG.seeds.seeds(noisy)` (master seed 2026).
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

**Hamiltonian variational ansatz (Phase 2):**

$$
|\psi(\boldsymbol{\gamma},\boldsymbol{\beta})\rangle = \prod_{l=1}^{p} e^{-i\beta_l \sum_i X_i}\, e^{-i\gamma_l \sum_i Z_i Z_{i+1}}\, |+\rangle^{\otimes N}, \qquad n_{\text{HVA}} = 2p
$$

**Hardware-efficient ansatz (Phase 2):**

$$
|\psi(\boldsymbol{\theta})\rangle = R(\boldsymbol{\theta}_p)\, C \cdots R(\boldsymbol{\theta}_1)\, C\, R(\boldsymbol{\theta}_0)\, |0\rangle^{\otimes N}, \qquad n_{\text{HEA}} = 2N(p+1)
$$

$$
R(\boldsymbol{\theta}_l) = \prod_{q} R_Z(\theta_{l,q}^{z})\, R_Y(\theta_{l,q}^{y}), \qquad C = \mathrm{CNOT}_{N-1,N} \cdots \mathrm{CNOT}_{1,2}
$$

**VQE metrics (Phase 2):**

$$
\epsilon = \frac{E_{\text{VQE}} - E_0}{|E_0|}, \qquad F = |\langle \phi_{\text{even}} | \psi \rangle|^2, \qquad F_{\text{GS}} = \sum_{s \in \text{GS}} |\langle \phi_s | \psi \rangle|^2
$$

**Parity-splitting bound for symmetry-broken states (Phase 2):**

$$
\epsilon \;\ge\; w_{\text{odd}}\, \frac{E_{\text{odd}} - E_{\text{even}}}{|E_0|}, \qquad w_{\text{odd}} = \frac{1 - \langle P \rangle}{2}
$$

**Entanglement bounds for depth-p circuits on open chains (Phase 3):** one two-qubit gate per layer crosses any cut, so

$$
S_\alpha(\ell) \le \min(p,\ \ell,\ N-\ell)\,\ln 2, \qquad F \le \sum_{k=1}^{2^p} \lambda_k(\phi_{\text{ref}})
$$

**Fannes–Audenaert continuity (Phase 3), d = 2^(N/2):**

$$
|S_{\text{VQE}} - S_{\text{ref}}| \le T\ln(d-1) + H_2(T), \qquad T = \sqrt{1 - F_{\text{GS}}}
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
| 2 | Noiseless VQE: HVA and HEA, warm start, random restarts, barren plateaus | **Done** |
| 3 | Entanglement of VQE states, central-charge extraction vs depth | **Done** |
| 4 | Noisy density-matrix simulation (N = 4–8), shots, SPSA, noise sweeps | **Next** |
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
- All Hamiltonians are real, so exact states are stored as float64.
- Every model is diagonalized inside its Z2 spin-flip sectors (TFIM and XXZ: global flip; cluster: the two sublattice flips). This is essential: at N = 14 and h = 0.1 the TFIM parity splitting is 3e-14, and an unresolved solver returns a random mixture with S(N/2) = 0.48 instead of the cat-state value ln 2.
- The reference ground state is the lowest state over all sectors, with the fully symmetric sector preferred on exact ties. For the TFIM this is the even-parity state, which is what the parity-conserving HVA prepares. The XXZ ground state has parity (−1)^(N/2) (Marshall sign rule).
- Saved sweep files: `results/phase1/ed_tfim_N{N}`, `ed_tfim_pbc_N{N}`, `ed_xxz_N{N}`, `ed_cluster_N{N}_h2_{m1|0|1}`, `ed_cc_profiles_N{N}`. Each sweep file holds E0, E1, E2, gap, gap_sector, sector_energies, ground_sector, sector_labels, S_half, S2_half, S_profile, S2_profile, zz_corr, M2, M4, binder, Ms2 (and string_order for the cluster model). Open-chain files also hold `states[n_param, n_sectors, 2^N]`, the lowest state of every sector.
- Fidelity convention (resolved in Phase 2): the HVA uses the even-parity state; for the HEA both the even-sector fidelity and the ground-space fidelity are reported.

**Key results (numbers in `phase1_summary.txt`):**

- TFIM energies and gaps agree with the free-fermion solution to 1e-9 for all N and h.
- Boundary-CFT gap scaling at the critical field: N times the full gap extrapolates to 3.138 (π, error 0.11 %); N times the even-sector gap extrapolates to 12.596 (4π, error 0.23 %).
- Binder crossings h* = 0.921, 0.952, 0.967, 0.976 for size pairs (6,8) to (12,14), converging to the critical field.
- Central charge, periodic chains at N = 14: TFIM c = 0.5020 ± 0.0003; XXZ c = 1.028 (Δ = 0) and 1.032 (Δ = 0.5). Open chains are biased at these sizes: TFIM c = 0.561; the XX chain oscillates strongly between even ℓ (1.18) and odd ℓ (0.54). An oscillation-corrected fit did not help. Use periodic chains for central charges.
- The half-chain entropy of the open-chain even-parity state does not peak at the critical field. The ordered phase sits on the ln 2 plateau, and S(h_c) = 0.411 < ln 2 at N = 14 (it would exceed ln 2 only for N of about 400). The transition appears as the steepest drop of S(N/2): h = 0.80 to 0.95 for N = 6 to 14. For periodic chains the entropy peak is visible (h = 0.55 to 0.90 for N = 6 to 14).
- XXZ open chains show an N/2-odd / N/2-even split in S(N/2), from bond dimerization. No sharp feature at the BKT point Δ = 1 is expected at these sizes.
- Cluster model: string order equals 1 at h1 = h2 = 0 with a fourfold-degenerate ground state (edge modes), and falls to 0.01 at h1 = 1.6 (h2 = 0). At h2 = ±1, h1 = 0 the chain is critical (string order 0.09).

## 9. Phase 2: completed

**Status:** done; merged into `main` at 0f7015c, tag `phase2-complete`. Results were produced at commit 5ab2a0b. `pytest -q`: 155 tests pass in 53 s (47 for Phase 2, including 7 PennyLane cross-checks). `experiments\phase2_plots.py`: 20 checks passed, 0 failed. Report: the Claude Doc "Phase 2: Noiseless VQE for the Transverse-Field Ising Chain" (https://claude.ai/artifact/Ffk8v1ieS1dpBmM57NNMVU).

**How to run (from the project root):**

- `python experiments\phase2_vqe_sweep.py --workers 12` (full; about 2 to 3 hours; `--quick` for N = 4 to 8 in about 10 minutes; `--experiments sweep warmcold depthscan pbc`; `--sizes`, `--ansatze`, `--recompute`). Cached per file, so interrupted runs resume.
- `python experiments\phase2_barren_plateau.py` (about 7 minutes; `--families hea_linear hea_fixed hva_linear`).
- `python experiments\phase2_plots.py` (7 figures in `figures/phase2/` and `results/phase2/phase2_summary.txt`).
- `python scripts\phase2_report_data.py` (compact numbers in `results/phase2/phase2_report_data.json`).

**API (for Phase 3 onward):**

- `vqe.ansatz`: `HVA(N, p, boundary="open")`, `HEA(N, p)`, `make_ansatz(name, N, p)`. Methods: `.n_params`, `.state(theta)`, `.energy(theta, H)`, `.energy_and_grad(theta, H)` (exact adjoint gradient), `.random_params(rng, scale, distribution)` (`"normal"` or `"uniform"`), `.shift_spec()`, `.pennylane_ops(theta)` (queues the same circuit inside a QNode), `HVA.pad(theta, new_depth)` (identity layers). Parameter layout: HVA `(γ1, β1, ..., γp, βp)`; HEA `theta.reshape(p + 1, 2, N)` with index `[layer, (RY, RZ), qubit]`.
- `vqe.statevector`: NumPy kernels with PennyLane gate conventions (`apply_rx_all`, `apply_ry_layer`, `rz_layer_phase`, `apply_cnot_chain`, `zz_diagonal`, `z_signs`, ...). States are complex128.
- `vqe.optimize`: `point_rng(*keys)`, `minimize(ansatz, H, x0, method, maxiter)` returning `OptResult`, `optimize_point(ansatz, H, candidates)` returning `PointResult` (best of several initial points), `random_candidates(...)`, `field_sweep(...)` (warm start plus restarts), `ANSATZ_KEYS = {"hva": 1, "hea": 2}`.
- `vqe.gradients`: `parameter_shift_gradient` (generalized shift rule), `finite_difference_gradient`, `gradient_variance`, `fit_exponential_decay`, `fit_power_law`.
- `vqe.metrics`: `ed_reference(N, h_values, boundary)` (reuses the Phase 1 files), `reference_energy`, `ground_space_mask`, `sector_overlaps`, `parity_expectation`, `evaluate_sweep`, `required_depth`.
- `vqe.pennylane_ref`: `pennylane_hamiltonian(model, N, **params)`, `state_qnode`, `energy_qnode(ansatz, H, diff_method, device)` (use `device="default.mixed"` for Phase 4), `pennylane_state`, `pennylane_energy_and_grad`.

**`CFG.vqe` fields added in Phase 2:** `optimizer_noiseless="L-BFGS-B"`, `max_iterations=1000`, `tolerance=1e-8` (gradient stop), `lbfgs_ftol=1e-15`, `lbfgs_maxcor=20`, `adam_lr=0.02`, `n_random_inits=8`, `n_random_inits_sweep=2`, `init_scale=0.1`, `init_distribution="normal"`, `tie_tolerance=1e-10`, `anchor_fields=(0.5, 1.0, 1.5)`, `use_warm_start=True`, `hva_sweep_direction="descending"`, `hea_sweep_direction="ascending"`, `hva_depth_warm_start=True`, `warmcold_size=8`, `ansatze=("hva","hea")`, `ground_space_gap_ratio=0.1`, `target_rel_error=1e-3`, `hva_depth_scan_sizes=(4,6,8,10)`, `hva_depth_scan_factor=2`, `save_states=True`, `grad_var_sizes=(4,...,14)`, `grad_var_samples=500`, `grad_var_fixed_depth=2`, `grad_var_field=1.0`. SPSA settings are kept for Phase 4.

**Conventions fixed in Phase 2:**

- Gates as in PennyLane: RX(φ) = exp(−iφX/2), likewise RY, RZ, IsingZZ. The HVA ZZ layer is IsingZZ(2γ) on every bond, the X layer RX(2β) on every qubit.
- At every field: warm start from the previous field, plus (HVA only) the depth p − 1 optimum padded with an identity layer, plus 2 random restarts; 8 random restarts at the first field and at the anchor fields. Energies within 1e-10 count as equal and the warm start wins. The depth warm start makes the HVA energy non-increasing in p.
- Sweep directions: HVA from h = 2.0 downward (it starts at the paramagnet), HEA from h = 0.1 upward (it starts at a ferromagnet).
- Seeds: `SeedSequence([2026, experiment, ansatz, N, p, field index])`, experiment keys sweep 1, cold 2, warm 3, depthscan 4, pbc 5, barren plateau 6. Results do not depend on worker count or job order.
- ε is relative to the lowest energy over all sectors. A sector belongs to the ground space if its lowest level is within 0.1 times the in-sector gap of E0. The required depth p* is the smallest p with ε < 1e-3.
- Barren-plateau diagnostic: variance over 500 uniform draws of the derivative with respect to a parameter inside the light cone of the observable (first-layer RY on qubit c = N/2 − 1 for the HEA, γ1 for the HVA); costs ⟨Z_c Z_{c+1}⟩ and ⟨H⟩/N at h = 1.

**Saved files in `results/phase2/`:**

- `vqe_{hva|hea}_N{N}_p{p}` (N = 4 to 12, p = 1 to N): arrays over the 39 fields: `h`, `anchor`, `energy`, `E0`, `energy_error`, `rel_error`, `fidelity`, `fidelity_sectors[n_h, 2]`, `fidelity_gs`, `ground_space[n_h, 2]`, `parity`, `n_iter`, `n_fev`, `total_fev`, `success`, `best_source` (0 warm, 1 depth, 2 random), `n_candidates`, `warm_energy`, `random_best_energy`, `seconds`, `params[n_h, n_params]`, `states[n_h, 2^N]` (complex128, the optimized states for Phase 3); scalars `n_qubits`, `depth`, `n_params`.
- `vqe_warmcold_{hva|hea}_N8_p4`: `cold_*` and `warm_*` versions of energy, rel_error, fidelity, fidelity_gs, n_iter, n_fev, total_fev, n_candidates, success, seconds, params; plus `h`, `E0`.
- `vqe_hva_depthscan_N{N}` (N = 4 to 10): `fields`, `depths` (1 to 2N), `energy[n_f, n_p]`, `E0`, `rel_error`, `fidelity`, `n_iter`, `success`.
- `vqe_hva_pbc_check`: `sizes`, `fields`, `rel_error`, `fidelity` (periodic HVA at p = N/2).
- `bp_{hea_linear|hea_fixed|hva_linear}`: `sizes`, `depths`, `n_params`, `n_samples`, and for each cost (`local`, `energy`): `var_center`, `var_mean`, `var_first`, `var_median`, `mean_abs_grad`.

**Key results (numbers in `phase2_summary.txt` and the report):**

- Validation: states agree with dense circuits (1e-12) and PennyLane (1e-10); adjoint gradients agree with the generalized parameter-shift rule (1e-11) and PennyLane parameter-shift and backprop. The periodic HVA at p = N/2 is exact (ε ≤ 2.2e-14, N = 4 to 10, Ho and Hsieh). N = 4: HVA and HEA exact at p = 4 at all fields. The variational principle holds everywhere.
- At p = N and h_c: HVA ε ≈ 6e-6 for N = 6 to 12; HEA ε = 2.4e-8, 2.1e-5, 1.2e-4, 5.0e-4 for N = 6, 8, 10, 12. Share of HEA runs at the iteration limit: 44 %, 64 %, 77 %, 84 %. HVA: at most 2 %.
- At p = N/2 and h_c the HEA is more accurate: ε = 2.5e-4, 5.7e-4, 7.6e-4 for N = 8, 10, 12, against 2.6e-3, 1.9e-3, 1.5e-3 for the HVA.
- Required depth p* (ε < 1e-3) near h_c: HVA 5, 7, 8, 9 for N = 6 to 12; HEA 2, 4, 7, not reached by p = 12. Paramagnet: 2 for both. Ordered phase: HEA 1; HVA not reached within p ≤ N for N ≥ 8.
- The HVA's hardest region is the ordered phase, not h_c. Depth for ε < 1e-6 on open chains: up to 1.6 N at N = 8 and 2 N at N = 10 (h = 0.5), against N/2 on periodic chains.
- The HEA's hardest field lies on the ordered side of h_c and moves toward it with N (h ≈ 0.6 at N = 8, 0.75 to 0.90 at N = 12), matching the Phase 1 finite-size shift.
- In the ordered phase the HEA prepares symmetry-broken states: F = 1/2, F_GS ≥ 0.999. Their ε respects the parity-splitting bound at every N (min ratio 1.005 to 1.193); median ε/bound 1.008, 1.044, 1.277, 3.143 for N = 6 to 12, where resolvable.
- One HEA failure: N = 12, p = 12, h = 0.85 ends with ⟨P⟩ = 0.15 and F_GS = 0.57 (run stopped at the iteration limit). The HEA error is not monotonic in depth at N = 12.
- Warm start (N = 8, p = 4): HVA 3.7 times fewer gradient evaluations per optimization, 13.5 times less per sweep, equal accuracy. HEA: cheaper but less reliable (79 % against 97 % of fields with ε < 1e-3), because the warm start carries the symmetry-broken branch into the paramagnet.
- Barren plateaus (N = 4 to 14): HEA at p = N has Var ∝ exp(−bN) with b = 0.690 ± 0.008 (local cost), against the 2-design value ln 2 = 0.693; HEA at p = 2 has b = 0.293 (the CNOT staircase has no local light cone); the HVA variance stays near 0.5, with no exponential decay (free-fermion Lie algebra).
- Reproducibility: an independent Linux run reproduced every HVA number to the last printed digit; HEA numbers agree in conclusions but differ beyond the second digit, since most HEA runs stop at the iteration limit.

**Decisions taken at the end of Phase 2:**

- Phase 3 uses the saved `states` from the p = N sweeps as its main input, for both ansätze, with the ED even-sector states as reference. HVA entanglement in the ordered phase is read with its known accuracy limit (F ≥ 0.967 at N = 12). For a clean ordered-phase comparison, the HVA depth-scan states at p = 2N (anchor fields, N ≤ 10) can be regenerated if needed.
- Entanglement comparisons are made per ansatz: HVA states are parity eigenstates (cat-state ln 2 plateau expected), HEA states in the ordered phase are symmetry broken (reduced entanglement expected).
- Both ansätze are kept for the noisy phases (4 and 5): at p ≤ N/2 the HEA is the more accurate one.

**Resolved in Phase 3:** the HEA point at N = 12, h = 0.85 (F_GS ≈ 0.52 to 0.57 for every p ≥ 3, not only p = 12) is a symmetry-broken state on the parity-splitting floor (⟨P⟩ = 0.15, ε = 3.1e-3 = 1.2 times the bound), one grid point past the ground-space cutoff (splitting / gap = 0.108 > 0.1). Not an optimizer failure; no rerun needed.

**Open, not blocking:**

- [ ] Optional housekeeping: move the legend in the top-right panel of `vqe_error_vs_depth`; drop the PDF `CreationDate` in `utils/plotting.py` so re-plotting does not mark PDFs as modified; add a `-dirty` flag to the git commit recorded by `utils/io.py`.
- [ ] Check the 14 references of the Phase 2 report before submission.
- [ ] Check the 9 references of the Phase 3 report before submission.

## 10. Phase 3: completed

**Status:** done; merged into `main` at 9110ea2, tag `phase3-complete`. Results produced at commit 1441fc3 (plot check refined in aa8a4e7). `pytest -q`: 180 tests pass in 70 s (25 for Phase 3). `experiments\phase3_plots.py`: 12 checks passed, 0 failed, 9 notes. Report: the Claude Doc "Phase 3: Entanglement Analysis of VQE States" (https://claude.ai/artifact/Ptt7MBm516srhNMd7FREQG).

**How to run (from the project root, after Phases 1 and 2):**

- `python experiments\phase3_entanglement.py --workers 12` (about 6 minutes; `--experiments ed states maxent pbc`; `--sizes`, `--ansatze`, `--recompute`). Cached per file.
- `python experiments\phase3_plots.py` (6 figures in `figures/phase3/` and `results/phase3/phase3_summary.txt`).
- `python scripts\phase3_report_data.py` (compact numbers in `results/phase3/phase3_report_data.json`).

**API (`analysis.vqe_entanglement`, for Phase 4 onward):** `profile_data(psi, rank_tol)` (S, S_2 and Schmidt rank at every cut), `schmidt_rank`, `entropy_bound(p, N)`, `page_entropy(N)`, `symmetry_broken_state(phi_even, phi_odd)`, `projected_reference(psi, sector_states, mask)` (nearest ground-space state and F_GS), `audenaert_bound(F, d)`, `rank_fidelity_bound(phi, rank)`, `PurityOperator(N, ell)` (acts as rho_A (x) 1 inside `Ansatz.energy_and_grad`; true purity gradient = 2 x returned gradient), `purity_and_grad`, `maximize_entanglement(ansatz, x0, maxiter, gtol)`. Reuses the Phase 1 routines in `analysis.entanglement`, which accept complex states.

**`CFG.ent` (added in Phase 3):** `rank_tol=1e-14`, `cc_field=1.0`, `cc_ell_min=2`, `maxent_starts=6`, `maxent_maxiter=500`, `maxent_gtol=1e-9`, `random_samples=200`, `pbc_sizes=(4,...,12)`, `bound_tol=1e-9`, `fidelity_resolution=1e-14` (1 − F floored at this in the Fannes–Audenaert check, since S is not Lipschitz near F = 1).

**Conventions fixed in Phase 3:**

- Every VQE state is compared with the exact state of its own symmetry: the nearest ground-space state P_GS psi / |P_GS psi| (= phi_even for the HVA; close to the symmetry-broken state (phi_even + phi_odd)/sqrt 2 for a symmetry-broken HEA state). Its overlap with psi is F_GS.
- Calabrese–Cardy fits use the Phase 1 window 2 ≤ ℓ ≤ N/2. HEA profiles are not mirror symmetric (up to 0.03 at p = N; the CNOT staircase runs one way), so S(ℓ) ≠ S(N − ℓ) for the HEA; fits use the left half as in Phase 1.
- Experiment seed keys: maxent 8, random samples 9, periodic HVA 10.

**Saved files in `results/phase3/`:**

- `ent_ed_N{N}`: `h`, `S_even`, `S2_even`, `S_odd`, `S_broken`, `S2_broken` (each [n_h, N − 1]), `ground_space`, `E0`, `S_half_phase1`.
- `ent_{hva|hea}_N{N}`: `S`, `S2`, `rank`, `S_ref`, `S2_ref` (each [n_p, n_h, N − 1]); `fidelity_gs`, `audenaert`, `rank_fidelity_bound` ([n_p, n_h]); `h`, `depths`.
- `maxent_{hva|hea}_N{N}`: `depths`, `S2_max`, `S_at_S2_max`, `S2_starts`, `random_S_mean`, `random_S_max`, `random_S2_mean`, `n_iter_best`.
- `pbc_hva_N{N}`: periodic HVA at h_c for p = 1..N/2: `energy`, `E0`, `rel_error`, `fidelity`, `S`, `S2`, `S_ed`, `S2_ed`, `states`.

**Key results:**

- Capacity is not the bottleneck: both ansaetze reach the rank bound p ln 2 exactly for every p < N/2 (all N), while the ground state never needs more than ln 2. At h_c, N = 12, rank 2 (p = 1) already allows F = 0.9989, but the optimized states reach 0.73 (HVA) and 0.50 (HEA); from p = 3 the rank allows infidelities below 1e-10.
- For p ≥ N/2 the maximal-entanglement search falls short in five cases (HEA N = 10, p = 9, 10: 0.92; HEA N = 12, p = 11, 12: 0.92, 0.90; HVA N = 12, p = 6: 0.83): purity is a global cost with barren plateaus. Still at least 5 times what any ground state needs.
- Random parameters: HEA mean S within 1 % of the Page value (2-design, consistent with the barren plateau); HVA about a third of it (free-fermion manifold).
- Half-chain entropy at p = N, N = 12: ED 0.3965 at h_c, HVA 0.3986, HEA 0.3791. Ordered phase (h = 0.1): ED even ln 2, HVA 0.7001 (slightly above ln 2; about 0.9 at p = 6), HEA 0 (symmetry broken, matching the ED symmetry-broken state, 6e-6).
- The symmetry-broken HEA states show the entropy peak the open-chain even state lacks: at h = 0.70, 0.80, 0.90 (N = 8, 10, 12), against 0.80, 0.85, 0.85 for the symmetry-matched exact reference.
- Central charge at h_c: shallow circuits give c ≈ 0 (area law; HVA c = 0 for p ≤ 2). Open chains, N = 12: ED 0.564, HVA 0.661 (p = N/2) and 0.597 (p = N, 6 % high despite ε ≈ 6e-6), HEA 0.380 and 0.515. HVA approaches from above (peak 0.75 at p = 8), HEA from below (within 1 % for N ≤ 10). Periodic HVA, N = 12: c(p) = 0, 0.157, 0.314, 0.472, 0.564, 0.502 (exact 0.502 at p = N/2).
- All 3,120 states obey the Schmidt-rank, Eckart–Young and Fannes–Audenaert bounds; entropy errors lie 1 to 3 orders of magnitude below the Fannes–Audenaert bound.

**Decisions taken at the end of Phase 3:**

- The ground-space cutoff stays at 0.1 times the in-sector gap, so the Phase 2 numbers are unchanged; the N = 12, h = 0.85 point is reported with its explanation.
- Phase 4 tracks S(N/2), purity, mutual information and logarithmic negativity as the primary entanglement metrics, compared per ansatz and per depth with the noiseless Phase 3 values (not with ED). The central charge is reported only at the deepest noisy depth p = N/2, as a secondary quantity, since at p ≤ N/2 it is set by depth (HVA 0 to 0.69, HEA up to 0.54).
- Noisy entropies are compared with the noiseless value of the same ansatz and depth, and with the matching symmetry reference.
