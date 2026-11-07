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
- Screen-off also triggers Modern Standby (found in Phase 4: a 30-minute screen timeout caused a 7-minute pause). For long runs set `powercfg /change monitor-timeout-ac 0` and the lid action on mains to "do nothing" (`powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0`, then `powercfg /setactive SCHEME_CURRENT`); restore the screen timeout afterwards.
- Measured noisy-simulation cost (Phase 4): about 0.25 s per N = 8 density-matrix energy evaluation with 12 workers under full load, about 5 times the single-core speed (memory-bandwidth contention and efficiency cores). Budget long noisy runs from this rate. Progress lines appear only when a whole job finishes; check liveness with `Get-Process python | Select-Object Id, CPU`.
- Phase 5 ZNE run (72 jobs, 12 workers): 1,932 s; the longest jobs are HEA N = 8, p = 4 (1,200 to 1,370 s each), so the wall time is set by the N = 8 jobs.
- Phase 8 thesis figures (16 scripts, 4 in parallel): about 8 s; inventory 10 to 30 s.
- Phase 7 classification run (57 jobs, 12 workers): 197 s; the longest jobs are the 18 noise jobs (89 to 101 s each); `phase7_plots.py` about 1 minute.
- Phase 6 dataset run (38 jobs, 12 workers): 170 s; the longest jobs are the N = 8 noisy and ZNE runs (120 to 170 s each) and ED N = 14 (96 s).
- Archives: Windows `tar` (`tar -a -c -f out.zip <folders>`) instead of `Compress-Archive` for large folders; check the entry count with `tar -t -f out.zip`.
- The Desktop is redirected by OneDrive; use `$HOME\Downloads` in commands.

**Git workflow (settled in Phase 2):** one branch per phase (`phaseN-...`; create it before the first commit of the phase, since a commit made on `main` by mistake has to be moved with `git branch phaseN-...`, `git checkout phaseN-...`, `git branch -f main HEAD~1`); commit the code before running experiments, so every result records the commit that produced it; merge with `--no-ff`; tag `phaseN-complete`. Tags are lightweight, so push them explicitly (`git push origin <tag>`); `--follow-tags` skips them. Tags so far: `phase1-complete` (6051c9d), `phase2-complete` (0f7015c), `phase3-complete` (9110ea2), `phase4-complete` (99928f4), `phase5-complete` (06622d4), `phase6-complete` (53189d3), `phase7-complete`.

## 4. Project structure

```
mtp-spin-chains/
├── config.py              # all parameters and seeds (single source of truth)
├── models/                # hamiltonians.py, exact.py           (Phase 1, done)
├── analysis/              # entanglement.py, observables.py,
│                          # free_fermion.py                      (Phase 1, done)
│                          # vqe_entanglement.py                  (Phase 3, done)
│                          # mixed_state.py                       (Phase 4, done)
│                          # ff_correlators.py                    (Phase 6, done)
├── vqe/                   # ansatz.py, statevector.py, optimize.py, gradients.py,
│                          # metrics.py, pennylane_ref.py         (Phase 2, done)
│                          # spsa.py                              (Phase 4, done)
├── noise/                 # channels.py, density.py, shots.py,
│                          # data.py, evaluate.py                 (Phase 4, done)
│                          # mitigation.py                        (Phase 5, done)
├── ml/                    # features.py, shadows.py, dataset.py  (Phase 6, done)
│                          # classifiers.py                       (Phase 7, done)
├── utils/                 # io.py, seeding.py, plotting.py       (Phase 0, done)
├── experiments/           # one script per thesis figure
├── scripts/               # check_env.py, phase2_report_data.py, phase3_report_data.py,
│                          # phase4_diag_negativity.py, phase4_diag_training.py,
│                          # phase4_report_data.py, phase5_report_data.py
├── tests/                 # test_phase0.py ... test_phase8.py, test_phase8_notes.py (404 tests)
├── notes/                 # Results notes per phase (Phase 8, T8.3)
├── results/               # .npz data + .json metadata (gitignored; back up separately)
└── figures/               # .pdf tracked in git, .png gitignored
```

`results/` is gitignored, so all result data exist only locally. Only the `phaseN_summary.txt` files are tracked (`.gitignore` excludes `results/**/*.npz` and `results/**/*.json`). Keep a backup zip of all result folders outside the repo and off the laptop; the latest is `mtp_results_phase1-7_backup_2026-11-06.zip` (865 files, 359.6 MB; file counts per phase: 85, 182, 62, 182, 158, 78, 118). Regenerating Phase 2 takes about 3 hours, Phase 3 about 6 minutes, Phase 4 about 6 hours (experiment A alone 3 minutes), Phase 5 about 33 minutes, Phase 6 about 3 minutes (needs Phases 1, 2 and 5), Phase 7 about 4 minutes (needs Phase 6).

## 5. Existing utilities (Phase 0)

**`config.py`** exposes a frozen global `CFG` object:

- `CFG.tfim.h_grid(noisy=False)`: 0.1 to 2.0, step 0.05 (39 points); noisy step 0.1. `CFG.tfim.J = 1.0`, `CFG.tfim.h_critical = 1.0`, `CFG.tfim.boundary = "open"`.
- `CFG.xxz.delta_grid(noisy=False)`: −0.5 to 2.0, step 0.1; noisy step 0.25.
- `CFG.cluster.h1_grid()`: 0 to 1.6, step 0.1; `CFG.cluster.h2_values = (-1.0, 0.0, 1.0)`.
- `CFG.system.sizes_exact = (4, 6, 8, 10, 12, 14)`, `sizes_noiseless = (4, ..., 12)`, `sizes_noisy = (4, 6, 8)`, `CFG.system.depths(N, noisy)` (1..N noiseless, 1..N/2 noisy).
- `CFG.ed` (added in Phase 1): `k_per_sector=3`, `dense_max_dim=512`, `eigsh_tol=0` (machine precision), `degeneracy_tol=1e-9`, `eigsh_seed=0`, `save_states=True`, `cc_ell_min=2`, `cc_boundaries=("open","periodic")`, `xxz_cc_deltas=(0.0, 0.5)`. `CFG.cluster.boundary = "open"` was also added.
- `CFG.vqe` (extended in Phases 2 and 4, see Sections 9 and 11), `CFG.ent` (added in Phase 3, see Section 10), `CFG.noise` (extended in Phase 4, see Section 11; `CFG.noise.strengths()` gives 0 plus 10 log-spaced values from 1e-4 to 5e-2), `CFG.mitigation` (added in Phase 5, see Section 12; `CFG.mitigation.strengths()` gives 0 plus 10 log-spaced values from 1e-5 to 1e-2), `CFG.ml` (extended in Phases 6 and 7, see Sections 13 and 14), `CFG.seeds.seeds(noisy)` (master seed 2026; `n_seeds_noisy = 3`).
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
I(A\!:\!B) = S(\rho_A) + S(\rho_B) - S(\rho_{AB}), \qquad \mathcal{E}_N = \ln\left\|\rho^{T_A}\right\|_1, \qquad S(A|B) = S(\rho_{AB}) - S(\rho_B)
$$

S(A|B) < 0 certifies distillable entanglement (hashing bound). For pure states:

$$
I = 2S(\rho_A), \qquad \mathcal{E}_N = 2\ln\sum_k \sqrt{\lambda_k}
$$

**Noise channels (Phase 4, PennyLane conventions), applied after every gate:**

$$
\mathcal{D}_p(\rho) = (1-p)\,\rho + \frac{p}{3}\left(X\rho X + Y\rho Y + Z\rho Z\right), \qquad \mathcal{D}^{(2)}_p(\rho) = (1-p)\,\rho + \frac{p}{15}\sum_{P \neq II} P\rho P
$$

$$
K^{\text{AD}}_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K^{\text{AD}}_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}, \qquad K^{\text{PD}}_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K^{\text{PD}}_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\gamma} \end{pmatrix}
$$

**Noisy gate counts (Phase 4, open chains, noisy RZ and noisy Hadamard layer):**

$$
n^{\text{HVA}}_{1q} = N(p+1), \quad n^{\text{HVA}}_{2q} = p(N-1), \qquad n^{\text{HEA}}_{1q} = 2N(p+1), \quad n^{\text{HEA}}_{2q} = p(N-1)
$$

**Error budget and fidelity collapse (Phase 4):**

$$
\lambda = p\,(n_{1q} + n_{2q}), \qquad 1 - \langle\psi_0|\rho|\psi_0\rangle \approx 1 - e^{-\kappa\lambda}
$$

κ is channel-specific (median over all fields, N and p with λ ≤ 1): 0.734 depolarizing (5–95 %: 0.69–0.84), 0.359 amplitude damping (0.02–0.52), 0.318 dephasing (0.05–0.35). For one depolarizing event a product-state qubit loses 2p/3 of fidelity, a pure two-qubit block 4p/5, an entangled qubit up to p.

**Shot-noise variance (Phase 4), M shots in each measurement basis g:**

$$
\operatorname{Var}(\hat{E}) = \frac{1}{M}\sum_g \operatorname{Var}_{p_g}(v_g)
$$

**SPSA (Phase 4):**

$$
\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - a_k\,\frac{f(\boldsymbol{\theta}_k + c_k\boldsymbol{\Delta}_k) - f(\boldsymbol{\theta}_k - c_k\boldsymbol{\Delta}_k)}{2c_k}\,\boldsymbol{\Delta}_k, \qquad a_k = \frac{a}{(k+1+A)^{0.602}}, \qquad c_k = \frac{c}{(k+1)^{0.101}}
$$

**Product-state bound and energy entanglement witness (Phase 4):** every separable state, pure or mixed, satisfies

$$
\operatorname{Tr}(\rho H) \;\ge\; E_{\text{prod}} = \min_{\{\theta_i\}} \left[-J\sum_{i}\cos\theta_i\cos\theta_{i+1} - h\sum_i \sin\theta_i\right]
$$

so ε < ε_prod certifies entanglement. N = 8, open: ε_prod = 3.07e-3 (h = 0.5), 4.43e-2 (h = 1.0), 7.17e-2 (h = 1.5).

**Readout model and correction (Phase 5):** independent asymmetric flips, p01 = P(1|0), p10 = P(0|1); calibration X gates are noisy, f = P(|1⟩ after the noisy X):

$$
q = \left(R_0 \otimes \cdots \otimes R_{N-1}\right) p, \qquad R_q = \begin{pmatrix} 1-p_{01} & p_{10} \\ p_{01} & 1-p_{10} \end{pmatrix}, \qquad A_{\text{cal}} = \bigotimes_q R_q \begin{pmatrix} 1 & 1-f \\ 0 & f \end{pmatrix}
$$

$$
\hat{E} = c + \sum_g w_g \cdot q_g, \qquad w_g = A_{\text{cal}}^{-T} v_g
$$

**Unitary folding and extrapolation (Phase 5):** noise follows inverse gates too, so noisy gate counts scale exactly by λ:

$$
U \to U\,(U^\dagger U)^k \ \text{(global)}, \qquad G \to G\,(G^\dagger G)^k \ \text{(local)}, \qquad \lambda = 2k+1 \in \{1, 3, 5\}
$$

$$
E^{*}_{\text{R}} = \tfrac{15}{8}E(1) - \tfrac{5}{4}E(3) + \tfrac{3}{8}E(5), \qquad \operatorname{Var}(\hat{E}^{*}) = \sum_k c_k^2 \operatorname{Var}\big(\hat{E}(\lambda_k)\big), \qquad \sum_k c_k^2 = \tfrac{167}{32}\ \text{(Richardson)},\ \tfrac{35}{24}\ \text{(linear)}
$$

**Folding bias (Phase 5):** with A (B) the first-order effect of errors placed after (before) each gate,

$$
E(\lambda) - E_{\text{ideal}} = p\left[\frac{A-B}{2} + \lambda\,\frac{A+B}{2}\right] + O(p^2)
$$

A = B for depolarizing noise (commutes with the gates). For amplitude damping and dephasing the sum telescopes along each qubit (trivial on |0⟩ before the first gate), so ZNE converges to the circuit followed by half a final noise layer:

$$
E_{\text{ZNE}} - E_{\text{ideal}} = \frac{1}{2}\left[\operatorname{Tr}\!\big(H\,\mathcal{N}_p^{\otimes N}(\rho_{\text{ideal}})\big) - E_{\text{ideal}}\right] + O(p^2)
$$

**Free-fermion correlators of the open TFIM (Phase 6 reference).** After a Hadamard on every site and the Jordan–Wigner Majoranas a_{2q} = S_q X_q, a_{2q+1} = S_q Y_q (S_q = ∏_{p<q} Z_p):

$$
H' = \frac{i}{4}\,a^{T} M a, \qquad \Gamma = -UV^{T} \quad (M = USV^{T}), \qquad \langle X_i\rangle = -\Gamma_{2i,\,2i+1}, \qquad \langle Z_iZ_j\rangle = (-1)^{j-i}\operatorname{Pf}\!\left(\Gamma_{[2i+1,\,2j]}\right)
$$

**Classical shadows (Phase 6):** random single-qubit Pauli bases b_q, outcomes v_q = ±1; for a Pauli string P of weight k with letters t_q,

$$
\hat{o} = \prod_{q \in \operatorname{supp} P} 3\,[b_q = t_q]\, v_q, \qquad \mathbb{E}[\hat{o}] = \langle P\rangle, \qquad \mathbb{E}[\hat{o}^2] = 3^{k}
$$

$$
\operatorname{Var}(\widehat{ZZ}) = 9 - \langle Z_iZ_j\rangle^2, \qquad \operatorname{Var}(\hat{X}) = 3 - \langle X_i\rangle^2 \qquad \text{(per snapshot; divide by } M\text{)}
$$

**Readout on expectation values (Phase 6):** for a column-stochastic C_q[outcome, state] each measured value has mean a_q + b_q v, and the per-qubit correction is unbiased for products on distinct qubits (the tensored linear inverse of Phase 5):

$$
a_q = C_{01} - C_{10}, \qquad b_q = 1 - C_{10} - C_{01}, \qquad v \;\to\; \frac{v - a_q}{b_q}
$$

**Critical finite-size scaling of M² (Phase 6 observation):** with Δ_σ = 1/8,

$$
M^2(h_c) \propto N^{-2\Delta_\sigma} = N^{-1/4}
$$

**ML critical point (Phase 7):** P(h) = mean classifier output over the test shadows of field h; P is made monotone by isotonic regression before the crossing is interpolated:

$$
P\!\left(h_c^{\text{ML}}\right) = \frac{1}{2}, \qquad p(y = 1 \mid x) = \sigma(w \cdot x + b) \ \text{(logistic regression)}
$$

**Uncertainty of h_c^ML (Phase 7):** seed spread (5 seeds: bootstrap of the training shadows within each field, MLP initialization) and the shot-noise error from the 4 test shadows:

$$
\sigma_{\text{tot}} = \sqrt{\sigma_{\text{seed}}^2 + \sigma_{\text{shot}}^2}, \qquad \sigma_{\text{shot}} = \frac{\operatorname{sd}_r\!\left(h_{c,r}^{\text{ML}}\right)}{\sqrt{n_{\text{test}}}}
$$

**Fidelity at the strongest Phase 7 noise (N = 8, p = 4, realistic, lambda = 320 p):**

$$
F \approx e^{-\kappa\lambda} = e^{-0.73 \times 3.2} \approx 0.10
$$

## 7. Full implementation plan

| Phase | Content | Status |
|---|---|---|
| 0 | Environment, project structure, utilities | **Done** |
| 1 | Exact diagonalization baseline and entanglement | **Done** |
| 2 | Noiseless VQE: HVA and HEA, warm start, random restarts, barren plateaus | **Done** |
| 3 | Entanglement of VQE states, central-charge extraction vs depth | **Done** |
| 4 | Noisy density-matrix simulation (N = 4–8), shots, SPSA, noise sweeps | **Done** |
| 5 | Error mitigation: readout correction, ZNE with gate folding | **Done** |
| 6 | ML datasets: correlators, classical shadows, clean and noisy | **Done** |
| 7 | ML: logistic regression, MLP, PCA, optional QCNN, cross-noise tests | **Done** (QCNN skipped) |
| 8 | Figures, reproducibility, writing | **Done** (thesis writing follows) |

## 8. Phase 1: completed

**Status:** done. `tests/test_phase1.py` (100 tests) and `tests/test_phase0.py` all pass. The full sweep (`python experiments\phase1_ed_sweep.py`, N = 4 to 14, all models) takes under a minute and writes about 35 MB to `results/phase1/`. Figures: `python experiments\phase1_plots.py`, which also writes `results/phase1/phase1_summary.txt`.

**API (for Phase 2 onward):**

- `models.hamiltonians`: `hamiltonian_terms(model, N, **params)` gives a list of `(coeff, {site: "X"|"Y"|"Z"})`; `terms_to_pauli_words` gives `(coeff, "ZZII")` pairs for building PennyLane Hamiltonians; `build_hamiltonian(model, N, **params)` gives a CSR matrix; models `"tfim"` (J, h), `"xxz"` (J, delta), `"cluster"` (J, h1, h2), all with `boundary`. `symmetry_generators(model, N)` gives the spin-flip bit masks; `apply_spin_flip(psi, mask)`.
- `models.exact`: `solve_model(model, N, **params)` gives an `ExactResult` with `.ground_state`, `.ground_energy`, `.energies` (merged lowest levels), `.gap` (E1 − E0 over the full spectrum), `.gap_sector` (gap inside the ground-state sector), `.sector_states` (lowest state of each sector), `.sector_energies`, `.sector_labels`, `.ground_labels`. Also `lowest_eigenpairs`, `dense_spectrum`.
- `analysis.observables`: `zz_correlation_matrix`, `expval_z`, `magnetization_moments(psi, staggered)`, `binder_cumulant`, `centred_correlator`, `pauli_expectation(psi, ops)`, `string_order_parameter`.
- `analysis.entanglement`: `schmidt_spectrum`, `entropy_from_spectrum(lam, alpha)`, `entanglement_entropy`, `half_chain_entropy`, `entropy_profile(s)`, `reduced_density_matrix(psi, sites)`, `von_neumann_entropy_dm`, `chord_log`, `calabrese_cardy_fit(profile, N, boundary, ell_min, parity)` (fits ℓ ≤ N/2 only).
- `analysis.free_fermion`: exact open-TFIM energies and gaps, and the open XX-chain ground energy.

**Conventions fixed in Phase 1:**

- Qubit ordering matches PennyLane: wire 0 is the most significant bit. Qiskit is little-endian; convert if Qiskit is ever used (T4.7 was skipped in Phase 4).
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

- [x] Done in Phase 8 (thesis Figure 3 replaces the figure; deterministic PDFs; `-dirty` commit flag). Former item: move the legend in the top-right panel of `vqe_error_vs_depth`; drop the PDF `CreationDate` in `utils/plotting.py` so re-plotting does not mark PDFs as modified; add a `-dirty` flag to the git commit recorded by `utils/io.py`.
- [ ] Check the 14 references of the Phase 2 report before submission.
- [ ] Check the 9 references of the Phase 3 report before submission.
- [ ] Check the references of the Phase 4 report before submission.
- [ ] Check the 12 references of the Phase 5 report before submission, and search the literature for the half-final-layer form of the folding bias before claiming it as new.

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

## 11. Phase 4: completed

**Status:** done; merged into `main` at 99928f4, tag `phase4-complete`. Code at c50fd87 and c0291c3 (two commits with the same message); experiment B run on 77aaf99; negativity checks refined in db34cdd; training diagnostic 60e842e; report-data script 4c9a105; final figures and summary committed just before the merge. `pytest -q`: 240 tests pass in 47 s (60 for Phase 4, including 6 PennyLane `default.mixed` cross-checks). `experiments\phase4_plots.py`: 21 checks passed, 0 failed, 14 notes. Report: the Claude Doc "Phase 4: Noise Robustness of VQE States for the Transverse-Field Ising Chain" (https://claude.ai/artifact/QfeNxKJwW1LT9AygWiE5FT). T4.7 (IBM calibration noise through Qiskit) was skipped by decision; a realistic noise model is built in Phase 5 instead.

**How to run (from the project root, after Phases 1 to 3):**

- `python experiments\phase4_noise.py --experiments eval --workers 12` (experiment A, 18 files, 164 s).
- `python experiments\phase4_noise.py --experiments train --workers 12` (experiment B, 72 jobs, 21,417 s = 5 h 57 min; HVA N = 8 jobs about 8,100 s, HEA N = 8 jobs about 13,100 s, N = 6 and 4 jobs 17 to 125 s). Cached per file; an unfinished job restarts from zero. Options: `--sizes`, `--ansatze`, `--recompute`.
- `python experiments\phase4_plots.py` (9 figures in `figures/phase4/` and `results/phase4/phase4_summary.txt`).
- `python scripts\phase4_diag_negativity.py` (points where E_N rises with noise) and `python scripts\phase4_diag_training.py` (trained states against the product-state bound).
- `python scripts\phase4_report_data.py` (crossing strengths, κ fits, shot floors, central charges, training crossovers; `results/phase4/phase4_report_data.json`).

**API (for Phase 5 onward):**

- `noise.channels`: `CHANNELS = ("depolarizing", "amplitude_damping", "dephasing")`; Kraus references `kraus_depolarizing`, `kraus_depolarizing_2q`, `kraus_amplitude_damping`, `kraus_dephasing`; matrix kernels `apply_unitary(rho, U, wires, N)` (rho must be Hermitian), `apply_diagonal`, `apply_kraus` (slow reference), in-place `depolarize_1q`, `depolarize_2q`, `amplitude_damp`, `dephase`; `NoiseModel(depolarizing, depolarizing_2q, amplitude_damping, dephasing, noisy_rz)` with `.after_gate(rho, gate, wires, N)`; `make_noise_model(channel, strength)` (one-parameter sweeps; two-qubit depolarizing = `two_qubit_factor` × p).
- `noise.density`: `gate_sequence(ansatz, N, p, theta)` (ordered gate list rebuilt from the Phase 2 parameter layouts), `gate_counts`, `simulate_density_matrix(ansatz, N, p, theta, noise_model)` (complex (2^N, 2^N) matrix), `simulate_statevector`. ρ is a C-contiguous matrix; gates act on reshaped views (no 2N-axis tensors).
- `noise.shots`: `EnergyEstimator(terms, N)` groups Pauli terms greedily into qubit-wise commuting bases (TFIM: Z and X); `.probabilities(rho)`, `.exact`, `.variance(rho, shots)`, `.sample(rho, shots, rng, probs, repetitions)`, `.estimate(rho, shots or None, rng)`. Basis rotations and readout are ideal (readout error is Phase 5).
- `noise.data`: `load_phase2(ansatz, N, p, h_sel)`, `load_phase3_half_entropy`, `exact_reference(N, h, ground_space)` giving `Reference(h, H, terms, E0, phi, gs_states, gs_mask)`, `energy(rho, H)`, `field_indices`.
- `noise.evaluate`: `evaluate_state(rho, ref, psi_ideal, N, profiles)`; keys `SCALAR_KEYS` (energy, rel_error, fidelity, fidelity_gs, fidelity_ideal, purity, S2_total, S_total, S_A, S_B, S2_A, S2_B, I, I2, E_N, S_cond, trace, min_eig, herm_err) and `PROFILE_KEYS` (the same per cut ℓ = 1..N − 1 for S_A, S2_A, I, I2, E_N, S_cond).
- `analysis.mixed_state`: `purity`, `renyi2`, `von_neumann`, `reduced_left`, `reduced_right`, `partial_transpose_left`, `log_negativity`, `fidelity_pure`, `projector_weight`, `cut_quantities`, `mixed_state_metrics(rho, N, profiles)`. Cuts are contiguous: A = sites 0..ℓ − 1.
- `vqe.spsa`: `spsa_minimize(f, x0, rng, maxiter, a, c, alpha, gamma, A, gain, calibration_steps, average_last, max_step)` returning `SPSAResult(x, x_last, history, n_fev, a, calibration_magnitude)`; `calibrate_gain(f, x0, rng, target_step, c, maxiter, ...)`; `default_A(maxiter)`.

**Config fields added in Phase 4:**

- `CFG.noise`: `two_qubit_factor=1.0`, `noisy_rz=True`, `noisy_state_prep=True`, `shot_repetitions=50`, `train_shots=10000`, `train_fields=(0.5, 1.0, 1.5)`, `weak_noise_max=1e-3`, `slope_tol=0.15`, `monotone_tol=1e-12` (existing: `channels`, `strengths()`, `shots_list=(1000, 5000, 10000)`, `zne_scale_factors`).
- `CFG.vqe`: `spsa_alpha=0.602`, `spsa_gamma=0.101`, `spsa_A_fraction=0.1`, `spsa_calibration_steps=20`, `spsa_average_last=50`, `spsa_target_hva=0.02`, `spsa_c_hva=0.05`, `spsa_target_hea=0.1`, `spsa_c_hea=0.1`, `spsa_warm_factor=0.1` (existing: `spsa_iterations=500`, generic `spsa_a=0.2`, `spsa_c=0.1`).

**Conventions fixed in Phase 4:**

- Noise follows every gate: one-qubit depolarizing after one-qubit gates, two-qubit depolarizing after two-qubit gates; amplitude damping and dephasing (phase damping) on every qubit a gate touches. One strength parameter per channel; amplitude and phase damping with equal γ shrink coherences by the same factor. RZ gates and the HVA Hadamard layer are noisy; two-qubit error equals one-qubit error (`two_qubit_factor = 1`).
- HVA ZZ bonds are applied in the order (0,1), (1,2), ...; HEA CNOT staircase CNOT(0,1), ..., CNOT(N−2,N−1); each HEA rotation layer applies RY then RZ per qubit.
- Experiment A evaluates the Phase 2 optimum at every noisy-grid field (step 0.1, 20 fields), every depth p ≤ N/2, three channels and 11 strengths. All metrics come from the exact density matrix; finite-shot energies (1,000, 5,000, 10,000 shots per basis, 50 repetitions) are drawn from the same ρ.
- Experiment B (SPSA under noise, 10,000 shots per basis, 500 iterations) runs at p = N/2 and h = 0.5, 1.0, 1.5 with one warm start (Phase 2 optimum) and three cold starts (normal, scale 0.1). The SPSA gain is calibrated once per problem (ansatz, N, h) at the first cold point with exact noiseless energies and shared by every start, channel and strength; warm starts use 0.1 times that gain. Calibrating at an optimum is forbidden: the symmetric difference cancels to third order there, so the gain explodes. Initial points, SPSA directions and shot streams do not depend on channel or strength (common random numbers). Strength 0 is run once and shared by the channels. Final metrics are exact Tr(ρH) of the Polyak-averaged parameters (last 50 iterates).
- Seeds: `SeedSequence([2026, experiment, ansatz, N, p, ...])`, experiment keys eval 11, train 12.
- Negativity check: E_N must decrease monotonically over the whole sweep for every state with E_N(0) ≥ 0.1 (all channels), and must end below its zero-noise value at every entangled point under depolarizing noise. The 0.1 threshold was set after the diagnostic (all exceptions had E_N(0) < 0.07) and must be stated as such in the report.

**Saved files in `results/phase4/`:**

- `noisy_eval_{hva|hea}_N{N}_p{p}` (N = 4, 6, 8; p = 1..N/2): `SCALAR_KEYS` arrays [3 channels, 11 strengths, 20 fields]; profiles [3, 11, 20, N − 1]; `E_shots[3, 11, 20, 3, 50]`, `E_sigma[3, 11, 20, 3]`, `estimator_err`, `repro_state_err[20]`; `h`, `strengths`, `channels`, `shots`, `E0`, `phase2_energy`, `phase2_fidelity`, `phase2_fidelity_gs`, `phase3_S`, `phase3_S2`, `n_1q`, `n_2q`, `n_qubits`, `depth`, `seconds`.
- `noisy_train_{hva|hea}_N{N}_h{h:.2f}_s{start}` (start 0 warm, 1 to 3 cold): arrays [3, 11] of `energy`, `rel_error`, `fidelity`, `fidelity_gs`, `fidelity_ideal`, `purity`, `S2_total`, `S_A`, `S2_A`, `I`, `I2`, `E_N`, `S_cond`, `evalA_energy`, `evalA_rel_error`, `initial_rel_error`, `n_fev`; `history[3, 11, 50]` (relative error every 10 iterations), `params[3, 11, n_params]`; scalars `h`, `E0`, `start`, `warm`, `gain`, `calibration_magnitude`, `train_shots`, `phase2_rel_error`, `seconds`.

**Key results (N = 8, p = N/2, h = h_c unless stated; numbers in `phase4_summary.txt`):**

- Validation: zero noise reproduces the Phase 2 states (3.8e-15), energies (2.3e-13), F and F_GS (1.4e-14) and the Phase 3 half-chain S and S2 (4.2e-14). Density matrices valid to 1e-15; variational principle holds everywhere; purity falls monotonically for all three channels; weak-noise log-log slopes of 1 − purity and 1 − F_ideal are 0.97 to 1.00; shot estimates unbiased (0.30 % beyond 3σ) with spread matching the analytic variance (median ratio 0.993). PennyLane `default.mixed` agrees with the simulator for all channels and both ansätze.
- The HEA's noiseless accuracy advantage (ε = 2.5e-4 against 2.6e-3) is gone from p = 1.0e-4 (depolarizing), 3.9e-4 (dephasing) and 5.6e-4 (amplitude damping): 108 noisy gates against 68. Depolarizing at p = 7.9e-4: HVA ε = 1.6e-2, HEA 3.2e-2.
- Error-budget collapse: the infidelity to the noiseless state of every ansatz, size and depth falls on one curve 1 − exp(−κλ) per channel, κ = 0.73 (depolarizing, tight), 0.36 (amplitude damping), 0.32 (dephasing; both broad and state dependent). Fidelity loss is set by the gate count, not by the ansatz design.
- Mixed-state subtlety: under depolarizing noise S(ρ_A) rises at all 339 entangled points while E_N falls at all of them, reaching exactly 0 by p = 0.05 (entanglement sudden death) with S2(A) = 2.1 (HVA) and 2.7 (HEA). Cleanest case: the symmetry-broken HEA at h = 0.5, where S(ρ_A) rises from about 0.04 to 2.6 with E_N ≈ 0 throughout. Hashing bound lost (S(A|B) > 0, log-interpolated) at N = 8, p = 4, h = 1: depolarizing 1.7e-3 (HVA, λ = 0.12) and 1.1e-3 (HEA, λ = 0.12); amplitude damping 3.1e-3 / 3.2e-3 (λ = 0.21 / 0.35); dephasing 4.1e-3 / 4.1e-3 (λ = 0.28 / 0.45). Under depolarizing noise λ = 0.06 to 0.12 for every N and p (outlier HEA N = 8, p = 1: 0.017). Sudden death (E_N = 0) at p = 0.05 for HVA N = 8, p = 3, 4 and HEA N = 6, p = 3, N = 8, p = 1, 3, 4.
- Mutual information is not monotone in noise strength (falls at 311 to 319 of 339 points): noise inside the circuit is followed by gates crossing the cut, which turn it into classical correlations; data processing only covers noise after the last such gate.
- E_N can rise with noise in near-product states: 14 points (amplitude damping) and 1 (dephasing), all symmetry-broken HEA states in the ordered phase (F = 0.50, F_GS = 1.00, E_N(0) = 0.012 to 0.069), largest 0.015 → 0.129 (HEA N = 6, p = 3, h = 0.3). Mechanism: mid-circuit noise spoils the cancellation by which the optimized circuit undoes its own entanglement. Every state with E_N(0) ≥ 0.1 loses entanglement monotonically.
- Central charge (secondary, 3-point fits, depolarizing): from S(ℓ) it rises from 0.692 (HVA) and 0.539 (HEA) to 0.86 / 0.93 at p = 1e-4 and 19.3 / 21.7 at 0.05 (volume-law noise term). From I(ℓ)/2 the HEA stays within 2 % (0.530) up to p = 1.6e-3; the HVA drifts to 0.512 (−26 %) by then; both reach 0 beyond p ≈ 2e-2. I/2 is the noise-robust estimator.
- Shots: single-run shot-noise floor σ/|E0| = 1.1e-2, 5.0e-3, 3.5e-3 at 1,000, 5,000, 10,000 shots per basis (1/√M, both ansätze); noise-induced errors below p ≈ 1e-3 are invisible in one 1,000-shot run.
- Energy witness (ε < ε_prod) lost at N = 8, p = 4, h = 1: depolarizing 2.3e-3 (HVA, λ = 0.16) and 1.0e-3 (HEA, λ = 0.11); amplitude damping 3.8e-3 / 2.9e-3; dephasing 7.6e-3 / 3.8e-3. Same error budget as the hashing bound: two independent certificates fail at λ ≈ 0.1 to 0.5. In the ordered phase (h = 0.5) the HVA never certifies by energy (noiseless ε = 0.045 > ε_prod = 0.003).
- Training under noise: warm starts equal experiment A (median difference at most 2e-4 per problem). Under depolarizing noise cold training only meets experiment A where both are poor (HEA h = 1, p = 0.05: 0.820 against 0.831). Under amplitude damping and dephasing, cold-trained HEA circuits overtake it beyond p ≈ 3e-3 to 9e-3 (N = 8): at p = 0.05, h = 1, amplitude damping ε = 0.049 against 0.454, with purity 0.95 (against 0.13), E_N = 0.03, S_A = 0.04, just above ε_prod = 0.044. The trained circuits retreat to a nearly pure, nearly unentangled state near the best product state. The HVA gains less (amplitude damping, h = 1: 0.33 → 0.19, purity 0.33) and nothing under dephasing: its RX rotations cannot steer the damping fixed point |0⟩ toward X, and its |+⟩ start lies where Z-dephasing hurts most.
- Lower energy under noise comes from abandoning entanglement: the noise-trained HEA at h = 1 has 10 times lower ε than experiment A but E_N = 0.03 against a noiseless 0.54. Energy error is a misleading benchmark for noisy many-body VQE.
- Cold-SPSA floors at zero noise: HVA 0.144, 0.036, 0.009 and HEA 0.045, 0.042, 0.057 (h = 0.5, 1.0, 1.5). The HEA floor sits at the product-state bound, so cold SPSA from small angles barely leaves the product manifold even without noise; the training crossover partly reflects that near-product states are found anyway and happen to be robust.

**Decisions taken at the end of Phase 4:**

- T4.7 skipped. Phase 5 builds a realistic noise model in the existing framework: `two_qubit_factor` about 10, `noisy_rz = False` (virtual Z), plus readout error.
- Phase 5 mitigation runs should be experiment-A style (fixed parameters evaluated under noise: minutes) rather than SPSA training (hours). The error-budget collapse (fidelity about exp(−κλ)) supports exponential extrapolation in ZNE, since gate folding scales λ directly.
- Phases 6 and 7 classify phases from correlators and shadows, not energies, since noise-aware optima can have good energies without the physics.
- Report the 0.1 negativity threshold, the mutual-information non-monotonicity and the HEA near-product mechanism as findings, with the diagnostics as evidence.

## 12. Phase 5: completed

**Status:** done; merged into `main` at 06622d4, tag `phase5-complete`. Results produced at commit 36a8c0d; folding-bias check and report-data refinements in 402a486; final figures and summary in 7dd63da. `pytest -q`: 311 tests pass (83 for Phase 5, including the PennyLane extrapolator cross-check). `experiments\phase5_plots.py`: 27 checks passed, 0 failed, 7 notes. Report: the Claude Doc "Phase 5: Error Mitigation of Noisy VQE States for the Transverse-Field Ising Chain" (https://claude.ai/artifact/TrgXg2ap3k1FubLRHScbJg).

**How to run (from the project root, after Phases 1 to 4):**

- `python experiments\phase5_mitigation.py --experiments readout` (6 files, seconds).
- `python experiments\phase5_mitigation.py --experiments zne --workers 12` (72 files, 1,932 s). Cached per file. Options: `--sizes`, `--ansatze`, `--channels`, `--depths`, `--recompute`.
- `python experiments\phase5_plots.py` (6 figures in `figures/phase5/`, 27 checks, `results/phase5/phase5_summary.txt`; about 1 minute because check C12 runs live N = 8 simulations).
- `python scripts\phase5_report_data.py` (`results/phase5/phase5_report_data.json`).

**API (`noise.mitigation`, for Phase 6 onward):**

- Noise: `PHASE5_CHANNELS = ("realistic", "depolarizing", "amplitude_damping", "dephasing")`; `make_mitigation_noise_model(channel, strength)` (Phase 5 settings; "realistic" = depolarizing p and 10p plus amplitude damping and dephasing p on every touched qubit; virtual RZ); `error_budget(channel, strength, n_1q, n_2q)`.
- Folding: `inverse_gate`, `fold_gates(gates, scale, method)` (`FOLDINGS = ("global", "local")`, odd scales only), `folded_gate_sequence(ansatz, N, p, theta, scale, method)`, `noisy_gate_counts(gates, noise_model)`, `simulate_gates(gates, N, noise_model)`, `simulate_folded(ansatz, N, p, theta, noise_model, scale, method)`.
- Readout: `ReadoutModel(p01, p10)` with `.uniform`, `.from_config(N)`, `.matrices()`, `.matrix()`; `kron_all`, `apply_local_matrices(vecs, mats)` (tensor product without forming it); `prepared_one_probability(noise_model)`; `calibration_matrices(readout, noise_model)` (per-qubit C_q = R_q P_q), `full_calibration_exact`; `sample_full_calibration(A, shots, rng, R)`, `sample_tensored_calibration(C, shots, rng, R)`; `inverse_transpose`, `correct_probabilities(q, A, method, local_mats)` ("inverse" or "projected"), `project_to_simplex`, `total_variation`, `negative_mass`.
- `ReadoutEnergy(terms, N, readout)`: `.ideal_probabilities(rho)` (G, 2^N), `.measured_probabilities(p)`, `.energy(probs, weights)`, `.weights_full(A)`, `.weights_from_inverse_T(invT)`, `.weights_tensored(mats)`, `.mixed_state_energy(weights, readout)` (exponential asymptote), `.shot_variance(probs, shots, weights)`, `.sample_counts(probs, shots, rng, R)`, `.energy_from_counts(counts, shots, weights)` (weights may be per repetition).
- Extrapolation: `EXTRAPOLATORS = ("linear", "richardson", "exponential")`; `extrapolate(scales, values, method, axis, asymptote)` (exponential returns NaN where E − E∞ changes sign); `richardson_coefficients`, `linear_coefficients`, `polynomial_coefficients`, `linear_two_point`, `extrapolation_coefficients`, `extrapolation_variance`, `variance_amplification`.
- Plot helpers reused by `scripts\phase5_report_data.py`: `experiments.phase5_plots.estimates(d)`, `deltas(d)`, `shot_estimates(d)`, `shot_rms(d)`, `available_zne()`, `zne_data(a, N, p, c)`.

**`CFG.mitigation` (added in Phase 5):** `channels=("realistic","depolarizing","amplitude_damping","dephasing")`, `two_qubit_factor=10.0`, `noisy_rz=False`, `realistic_damping_ratio=1.0`, `realistic_dephasing_ratio=1.0`, `p_min=1e-5`, `p_max=1e-2`, `n_points=10`, `readout_p01=0.01`, `readout_p10=0.02`, `calibration_shots=10000`, `calibration_gate_noise=True`, `foldings=("global","local")`, `main_folding="global"`, `extrapolators=("linear","richardson","exponential")`, `main_extrapolator="richardson"`, `shots=10000`, `shot_repetitions=50`, `readout_field=1.0`, `readout_cal_shots=(1000,10000,100000)`, `plot_size=8`, `plot_field=1.0`, `plot_strength=1e-3`, `plot_strengths_curves=(1e-4,1e-3,4.64159e-3)`, tolerances `weak_fold_tol=0.02`, `order_tol=0.15`, `regime_lambda=0.2`, `variance_ratio_tol=0.1`, `exact_tol=1e-10`. ZNE scale factors come from `CFG.noise.zne_scale_factors = (1, 3, 5)`. `CFG.noise` (Phase 4) is unchanged.

**Conventions fixed in Phase 5:**

- Error measure δ = |E − E_ideal| / |E0|, with E_ideal the noiseless circuit energy (the most mitigation can recover); "at VQE accuracy" means δ below the noiseless VQE error.
- Error budget: λ = p(n_1q + 10 n_2q) for depolarizing and realistic noise, λ = p(n_1q + 2 n_2q) for amplitude damping and dephasing; N = 8, p = 4: n_1q = 40, n_2q = 28, λ = 320p (realistic).
- Inverse gates keep the gate name, so they get the same noise (virtual RZ stays noiseless). Idle qubits and basis rotations are noiseless; readout is uncorrelated.
- Calibration X gates carry gate noise (SPAM); qubits prepared in 0 receive no gate. Calibration streams depend on N only and are redrawn per (channel, strength) with common random numbers; each of the 50 shot repetitions has its own calibration.
- Energies use the linear readout inverse (unbiased); simplex projection is reported only for distributions.
- The exponential asymptote is the energy of I/2^N measured through the same pipeline (raw, corrected or ideal).
- Seeds: `SeedSequence([2026, 13, ...])` (ZNE: measurement `[13, akey, N, p, field]`, calibration `[13, 0, N]`), `([2026, 14, ...])` (readout).

**Saved files in `results/phase5/`:**

- `zne_{hva|hea}_N{N}_p{p}_{channel}` (72): `E_true`, `E_raw`, `E_ro`, `fidelity_ideal` [11 strengths, 20 fields, 2 foldings, 3 scales]; `Es_raw`, `Es_ro`, `Es_tens` [11, 20, 3, 50] (global folding); `var_raw`, `var_ro` [11, 20, 3]; `asym_true`, `asym_raw` [20], `asym_ro` [11, 20], `calib_f` [11]; `h`, `strengths`, `scales`, `foldings`, `channel`, `E0`, `phase2_energy`, `E_ideal`, `n_1q`, `n_2q`, `n_qubits`, `depth`, `shots`, `calibration_shots`, `repetitions`, `readout_p01`, `readout_p10`, `seconds`.
- `readout_{hva|hea}_N{N}` (6): `tvd`, `neg_mass` [2 sources (exact, shots), 3 calibration shots, 5 methods, 50, 2 groups], `dE` [2, 3, 5, 50]; `A_true`, `C_true`, `E_ideal`, `E0`, `E_raw_exact`, `E_full_exact`, `E_tens_exact`, `tvd_raw_exact`, `tvd_full_exact`, `cal_shots`, `methods`, `sources`.
- `phase5_summary.txt`, `phase5_report_data.json`.

**Key results (N = 8, p = 4, h = 1, realistic noise unless stated):**

- Readout error alone biases the energy by 3.3 % (up to 5.7 % over all points); exact calibration removes it to 1e-14. ZNE alone stays at this bias (folding does not scale measurement error).
- At p = 1e-4: unmitigated 4.3e-2 (HVA) / 4.7e-2 (HEA), readout-corrected 1.1e-2 / 1.5e-2, readout + Richardson ZNE 2.0e-5 / 2.0e-5 (exact limit). Readout + ZNE stays at VQE accuracy up to p = 9.6e-4 (HVA, λ = 0.31) and 3.5e-4 (HEA, λ = 0.11); readout correction alone up to 2.4e-5 (HVA) and never (HEA). Extrapolation fails beyond λ ≈ 0.7 (p ≥ 2e-3); at p = 1e-2 the combined error is 0.45 / 0.56.
- Linear extrapolation is one to two orders worse than Richardson; exponential is comparable at weak noise and degrades more gracefully under depolarizing noise (HVA, p = 1e-2: 0.12 against 0.37).
- Depolarizing: folding scales the first-order error exactly; weak-noise orders 1.000, 1.998, 2.998 (unmitigated, linear, Richardson); exponential order 2. A small second-order term survives folding, so single three-point Richardson slopes scatter (65 % within ±0.3 of 3).
- Folding bias (median of Richardson residual / unmitigated error, p = 1e-5): amplitude damping 9.1 % (5 to 95 %: 0.05 to 19 %), dephasing 5.2 %, realistic 1.1 %, depolarizing 5e-7. The half-final-layer identity holds to 7.4e-5 in 36 live cases (both ansätze, N = 4, 6, 8, both foldings). Local and global folding agree (residual ratio median 1.00, 5 to 95 %: 0.97 to 1.24).
- SPAM from noisy calibration (ratio of combined residual to ideal-calibration residual, p = 1e-5, HVA / HEA): depolarizing 4,300 / 590 (SPAM is the whole residual), realistic 0.35 / 0.39, amplitude damping 0.05 / 0.06 (SPAM cancels most of the folding bias; mechanism not derived), dephasing 1.0 (no SPAM).
- 250 of 14,400 combined energies fall below E0 (not variational): 249 HEA, all λ ≤ 1, none under dephasing; most negative −0.8 % (amplitude damping), −0.18 % (realistic), −0.10 % (depolarizing).
- Shots (10,000 per basis): single-run floor σ/|E0| = 3.6e-3; readout-corrected about 3.8e-3; readout + Richardson 0.8 to 1.0e-2 (factor 2.2 to 2.6, predicted 2.28). The combined RMS is flat near 1e-2 from p = 1e-5 to 1e-3 (shot-limited). ZNE beats readout correction alone only above p = 1e-4 (realistic, depolarizing), 1e-3 (amplitude damping), 2.2e-3 (dephasing). No method reaches VQE accuracy at 10,000 shots; readout + ZNE needs about 9.5e4 (HVA) and 1.0e7 (HEA) shots per basis and scale factor.
- Readout calibration: TVD 3.8e-2 (Z) / 8.3e-2 (X) uncorrected; after correction ∝ M_cal^(−1/2) (1.2e-2, 3.9e-3, 1.2e-3 at 1e3, 1e4, 1e5 shots per circuit); tensored (2 circuits) equals full (256). Projection removes negative probabilities but raises the energy error by 10 to 30 %.
- Mitigation works in the same window (λ ≈ 0.1 to 0.3) in which Phase 4 found the energy witness and the hashing bound still holding (λ ≈ 0.1 to 0.5).

**Decisions taken at the end of Phase 5:**

- Phase 6 noisy datasets use the Phase 5 realistic model (two-qubit factor 10, virtual RZ) with readout error, and provide a readout-corrected variant (tensored calibration, linear inverse). The Phase 4 one-parameter channels stay available for channel-dependence studies.
- ZNE is not applied to the Phase 6 datasets by default: at 1,000 to 10,000 shots it adds more variance than it removes bias below p ≈ 1e-4. A ZNE-mitigated dataset is generated only in the λ ≈ 0.05 to 0.3 window, to test whether mitigation restores classification accuracy.
- Phases 6 and 7 classify from correlators and classical shadows, not energies (confirmed: mitigated energies are not variational and are shot-limited).
- Report the half-final-layer folding bias as a finding, subject to the literature check in the open items; report the amplitude-damping SPAM cancellation as an observation only.

## 13. Phase 6: completed

**Status:** done; merged into `main` at 53189d3, tag `phase6-complete`. Code at 27b3527 (datasets produced at this commit); final figures and summary in e36e370. `pytest -q`: 350 tests pass (27 for Phase 6; the suite had 323 before Phase 6, not the 311 recorded at the end of Phase 5). `experiments\phase6_plots.py`: 16 checks passed, 0 failed, 20 notes. Report: to be written (Claude Doc "Phase 6: Machine-Learning Datasets for Phase Classification of the Transverse-Field Ising Chain").

**How to run (from the project root, after Phases 1, 2 and 5):**

- `python experiments\phase6_datasets.py --workers 12` (38 files, 170 s). Cached per file. Options: `--experiments ed vqe noisy zne`, `--sizes`, `--ansatze`, `--channels` (realistic, depolarizing, amplitude_damping, dephasing), `--recompute`. Writes `results/phase6/dataset_index.json`.
- `python experiments\phase6_plots.py` (16 checks, 4 figures in `figures/phase6/`, `results/phase6/phase6_summary.txt`).

**API (for Phase 7):**

- `analysis.ff_correlators`: `pfaffian(A)` (Parlett–Reid with pivoting), `majorana_matrix(N, J, h)`, `ground_covariance`, `ff_energy`, `ff_observables(N, J, h)` giving `zz`, `x`, `z` of the even ground state.
- `ml.features`: `FEATURE_SETS = ("corr", "zz", "compact")`; `bits`, `z_signs`, `pure_observables(psi, N)`, `mixed_observables(rho, N)` (dicts `zz` (N, N), `x`, `z`); `readout_affine(mats)`, `affine_observables`, `readout_biased(obs, R)`, `readout_corrected(obs, R, C)`; `n_features`, `feature_names`, `corr_features(zz, x)`, `unpack_features(X, N)`, `magnetization_m2`, `compact_features`, `compact_names`, `select_features(X, N, kind)`, `energy_from_features(X, N, h, J)`.
- `ml.shadows`: `BASES = ("X", "Y", "Z")` (codes 0, 1, 2), `ROTATIONS`; `draw_stream(rng, M, N)` (bases, sampling uniforms u, readout uniforms r); `sample_pure(psi, bases, u, chunk_elements)` (prefix tree of 6^k branches, then per-snapshot projection), `outcome_table(rho, N)` (all 6^N probabilities), `marginal_tables`, `sample_mixed(tables, bases, u)`, `outcome_probabilities`, `apply_readout(bits, r, p01, p10)`, `shadow_observables(bases, bits, a, b)` (optional readout correction), `single_snapshot_variance(obs)`, `pack_bits`, `unpack_bits`.
- `ml.dataset`: `SOURCE_KEYS = {"ed": 0, "hva": 1, "hea": 2}`, `SOURCES`, `NOISY_VARIANTS = ("gate", "meas", "ro")`; `field_grid(noisy)`, `labels(h)`, `train_field_mask(h)`, `repetition_split(R)`, `resolve_depth(rule, N)`, `snapshot_rng`, `draw_snapshots(source_key, N, p, field_index, R, M, scale)`, `calibration(N, noise_model, R)`; generators `pure_dataset`, `ed_dataset(N)`, `vqe_dataset(ansatz, N, p)`, `noisy_dataset(ansatz, N, p, channel, thetas, h, strengths, R, M)`, `zne_dataset(...)`, `error_budget_for`, `zne_window`; access `dataset_name(source, N, ansatz, depth, channel)` (returns name and subdir), `load_dataset(...)`, `sample_matrix(ds, kind, variant, strength, part, feature_set)` returning (X, y, h), `snapshots(ds, field, rep, strength, measured)`, `write_index()`.

**`CFG.ml` (extended in Phase 6):** existing `train_h_ordered_max=0.6`, `train_h_disordered_min=1.4`, `test_fraction=0.2` (now the share of shadow repetitions kept for testing), `n_shadow_snapshots=1000`, `mlp_hidden=(32,16)`, `learning_rate=1e-3`, `epochs=300`, `n_training_seeds=5`; new `shadow_repetitions=20`, `label_ordered=0`, `label_disordered=1`, `label_critical=-1`, `ed_sizes=(4,...,14)`, `vqe_sizes=(4,...,12)`, `vqe_depth_rules=("half","full")`, `noisy_sizes=(4,6,8)`, `noisy_depth_rule="half"`, `noisy_channels=("realistic",)`, `noisy_fine_grid=True`, `zne_lambda_window=(0.05,0.3)`, `zne_channels=("realistic",)`, `sampler_chunk_elements=2**22`, tolerances `exact_tol=1e-10`, `ff_tol=1e-8`, `z_frac_max=0.01`, `z_abs_max=6.0`, `t_frac_max=0.02`, `var_ratio_tol=0.1`, `snapshot_match_min=0.9999`.

**Conventions fixed in Phase 6:**

- Model: open TFIM only. Every source uses the fine grid (39 fields, step 0.05), noisy data included: the parameters are fixed (experiment-A style), so the coarse noisy grid of Phases 4 and 5 (set by the cost of training under noise) is not needed.
- Sources: ED even-parity ground state (N = 4 to 14); noiseless Phase 2 states at p = N/2 and p = N (N = 4 to 12, both ansätze); noisy: the Phase 2 optimum at p = N/2 under the Phase 5 "realistic" model at the 11 strengths of `CFG.mitigation.strengths()`, with readout error (N = 4, 6, 8); ZNE: global folding, scales 1, 3, 5, Richardson, only at strengths with λ in [0.05, 0.3]. Density matrices are not saved.
- Features ("corr"): upper triangle of ⟨Z_iZ_j⟩ (row-major), then ⟨X_i⟩; N(N − 1)/2 + N values. Raw, not connected, correlators, so symmetry-broken HEA states are not placed in the disordered phase. ⟨Z_i⟩ is stored for diagnostics only. "compact": M², mean X, mean nearest-neighbour ZZ, end-to-end ZZ.
- Shadows: R = 20 independent shadows of M = 1000 snapshots per field (one sample each). Both samplers draw qubits in order 0..N − 1 with s_q = 1 if u_q ≥ P(s_q = 0 | earlier outcomes), so pure and mixed samplers give identical snapshots for identical states and random numbers. Bases stored as uint8 [field, rep, M, N], bits packed along the qubit axis.
- Noisy variants: "gate" (gate noise, ideal readout), "meas" (readout error, uncorrected), "ro" (per-qubit correction with a 10,000-shot tensored calibration whose X gates carry gate noise, one calibration per repetition). Exact (infinite-shot) values are stored for all three; the exact "ro" uses the exact calibration.
- Labels: y = 0 for h < h_c, 1 for h > h_c, −1 at h = h_c (excluded from accuracies). Split: training fields h < 0.6 or h > 1.4 (10 ordered, 12 disordered); test fields all 39. Repetitions 17 to 20 are test-only, so no training snapshot is ever tested on.
- Seeds: snapshots `SeedSequence([2026, 15, source, N, p, field, rep, scale])` (source 0 ED, 1 HVA, 2 HEA; p = 0 for ED), independent of channel and strength (common random numbers); noiseless VQE and noisy data share the stream, so noisy strength 0 reproduces the noiseless snapshots. Calibration `SeedSequence([2026, 16, N])`, redrawn per (channel, strength). ZNE scale 1 uses the noisy stream; scales 3 and 5 have their own.

**Saved files in `results/phase6/` (by source; noise strength is the leading axis):**

- `ed/ds_ed_N{N}` (6): `X` [39, F], `Xs` [39, 20, F], `z`, `zs`, `energy`, `E0`, `bases` [39, 20, 1000, N], `bits` (packed).
- `vqe/ds_vqe_{hva|hea}_N{N}_p{p}` (20): as ED plus `params`, `phase2_energy`, `E0`, `fidelity`, `fidelity_gs`.
- `noisy/ds_noisy_{hva|hea}_N{N}_p{p}_realistic` (6): `X_{gate|meas|ro}` [11, 39, F], `Xs_{gate|meas|ro}` [11, 39, 20, F], `z_*`, `zs_*`, `bases` [39, 20, 1000, N], `bits_gate`, `bits_meas` [11, 39, 20, 1000, bytes], `energy`, `purity`, `fidelity_ideal` [11, 39], `strengths`, `lam`, `n_1q`, `n_2q`, `params`, `readout_matrices`, `cal_exact` [11, N, 2, 2], `cal_samples` [11, 20, N, 2, 2], `phase2_energy`, `E0`.
- `zne/ds_zne_{hva|hea}_N{N}_p{p}_realistic` (6): `X_gate_scales`, `X_ro_scales` [2, 3, 39, F], `Xs_ro_scales` [2, 3, 39, 20, F], `energy_scales`, `X_zne_gate`, `X_zne_ro` [2, 39, F], `Xs_zne_ro` [2, 39, 20, F], `energy_zne`, `strengths`, `strength_index`, `lam`, `scales`, `coefficients`, `cal_exact`, `cal_samples`.
- Every file: `h`, `y`, `train_field`, `test_field`, `rep_train`, `rep_test`, `feature_names`, `n_qubits`, `snapshots`, `repetitions`, `h_critical`, `depth`, `source`, `ansatz`, `seconds`. Also `dataset_index.json` and `phase6_summary.txt`. Total 164.4 MB.

**Key results:**

- Validation: ED features equal free-fermion correlators to 1.4e-14 (N = 4 to 14, all fields); ED M² equals Phase 1 to 2.2e-15; energies rebuilt from features equal ED (7.5e-14) and Phase 2 (1.2e-13); noisy Tr(ρH) equals the Phase 5 `E_true` to 8.0e-14 (folded scales 2.1e-13). Exact ZZ matrices are positive semidefinite (minimum eigenvalue 2.5e-3) with M² in [0.125, 0.997].
- Shadows: of 100,581 z-scores, 0.27 % lie beyond 3σ (normal 0.27 %), max |z| = 4.13; mean empirical/analytic variance 1.0008, median 0.9655 (χ²₁₉/19 median 0.9651). Readout t-statistics beyond 3: 0.77 % (uncorrected) and 0.75 % (corrected), expected 0.74 %.
- Identities: readout correction exact at strength 0; noisy strength 0 reproduces the noiseless snapshots 100 %; ZNE scale 1 equals the noisy dataset exactly; purity monotone in strength.
- ZNE in the window (λ = 0.07 and 0.15 at N = 8; 0.08 and 0.17 at N = 6; 0.07 and 0.16 at N = 4): exact gate-noise feature error reduced to 0.7 % to 4.1 % of the unmitigated error; shadow standard deviation amplified by √(167/32) ≈ 2.28.
- ED separability: M² margin between the training regions 0.376, 0.505, 0.589, 0.644, 0.683, 0.711 for N = 4 to 14 (shadow margins within 0.01). M²(h_c) = 0.554, 0.482, 0.440, 0.412, 0.390, 0.373; local log-log slopes −0.30 (N = 10 to 12) and −0.29 (N = 12 to 14), approaching the expected −1/4 (hand estimate, not a fit).
- Noise: the exact M² margin is never closed, even at p = 1e-2 (λ = 3.2 at N = 8): N = 8 margins 0.322 → 0.039 (HVA) and 0.599 → 0.073 (HEA). The HVA margin at p = N/2 is about half the HEA's (ordered phase is the HVA's hardest region, Phase 2).
- Readout: feature bias up to 0.049 to 0.059 at strength 0, removed exactly by correction; the residual after correction reaches 1.5e-2 to 2.3e-2 only at the strongest noise (SPAM from the noisy calibration X gates).

**Decisions taken at the end of Phase 6 (for Phase 7):**

- Default noisy variant "ro" (what an experiment would report); "gate" and "meas" isolate gate noise and readout. Classifiers train on `rep_train` shadows at the training fields and are tested on `rep_test` shadows at all fields; accuracies exclude y = −1.
- Exact features keep the training regions separable at every strength, so the central Phase 7 quantities are the drift of h_c^ML with noise and the accuracy of shadow-based classifiers, where shot noise (M² standard deviation of order 0.01 to 0.03 at N = 8) competes with margins of 0.04 to 0.07 at the strongest noise.
- Expect h_c^ML below 1 at small N from the finite-size crossover (Phase 1 Binder crossings 0.92 to 0.98); compare against ED at the same N, not against h_c = 1 alone.
- The ZNE dataset tests whether mitigation restores h_c^ML in the λ window, at the cost of about 2.3 times larger shadow noise.
- Channel dependence (`--channels depolarizing amplitude_damping dephasing`) and the cluster-Ising QCNN dataset (T7.5) are optional and not yet generated.

## 14. Phase 7: completed

**Status:** done; merged into `main` with tag `phase7-complete` (merge hash: `git rev-parse --short phase7-complete`). Code at 87d4b42 (results produced at this commit); results, figures, summary and tables in 9b4a766. `pytest -q`: 364 tests pass (14 for Phase 7). `experiments\phase7_plots.py`: 11 checks passed, 0 failed, 64 notes. Report: the Claude Doc "Phase 7: Machine-Learning Phase Classification of the Transverse-Field Ising Chain under Noise" (https://claude.ai/artifact/9zsruXJk6he4vvVPiGg7YA). T7.5 (QCNN on the cluster-Ising model) skipped by decision; optional future work.

**How to run (from the project root, after Phase 6):**

- `python experiments\phase7_classify.py --workers 12` (57 jobs, 197 s). Cached per file. Options: `--experiments clean ff noise zne shots learning`, `--sizes`, `--ansatze`, `--channels`, `--variants`, `--recompute`.
- `python experiments\phase7_plots.py` (11 checks, 9 figures in `figures/phase7/`, `results/phase7/tables/*.csv`, `results/phase7/phase7_summary.txt`; about 1 minute).

**API (`ml.classifiers`, for Phase 8):**

- Constants: `EXP_TRAIN = 17`, `CLASSIFIERS = CFG.ml.classifiers` ("logistic", "mlp"), `KINDS = ("exact", "shadow")`.
- Seeds: `training_seeds()`, `model_seed(s)`, `bootstrap_indices(s, n_fields, n_reps)`, `subset_indices(s, n_fields, n_reps, k)`.
- Models: `make_classifier(name, seed, n_rows)` (StandardScaler + LogisticRegression or MLPClassifier), `fit_predict(name, X_train, y_train, X_test, seed)`.
- Data: `training_rows(blocks, y_fields, rep_idx)`, `split_array(ds, A, kind, feature_set)` (train block, test array), `dataset_array(ds, kind, variant, strength)`, `rebuild_shadows(ds, n_snapshots, strength, variant)`.
- Evaluation: `isotonic(P)` (pool adjacent violators), `critical_point(h, P)` (NaN without a crossing), `n_crossings(P)`, `field_masks(h, y)` ("all", "window" 0.6 to 1.4, "deep"), `accuracies(p, h, y)`, `evaluate(train_blocks, y_train_fields, tests, h, y, seeds, classifiers, rep_subset)` (trains once per seed and classifier, tests on one or many sets; keys `P`, `P_rep`, `hc`, `hc_rep`, `ncross`, `acc`, `acc_window`, `acc_deep` with leading axes [classifier, seed]), `summarize_hc(hc, hc_rep)` (`mean`, `seed_std`, `n_valid`, `shot_se`, `total`).
- PCA: `pca_analysis(X_all, h, m2_exact)` (`pc1`, `pc1_std`, `pc2`, `evr`, `r_m2`, `h_steep`, `h_mid`, `components`).
- Plot helpers reusable in Phase 8: `experiments.phase7_plots.load_all()`, `hc_stats(a, prefix)`, `strength_axis(strengths)`, `label_zero(ax, x0)`.

**`CFG.ml` (extended in Phase 7):** `classifiers=("logistic","mlp")`, `feature_sets_phase7=("corr","compact")`, `logistic_C=1.0`, `logistic_max_iter=5000`, `mlp_alpha=1e-3`, `mlp_batch_size=64`, `mlp_early_stopping=False`, `mlp_validation_fraction=0.15`, `mlp_patience=30`, `bootstrap_training=True`, `decision_threshold=0.5`, `noise_modes=("clean_ed","clean_vqe","noisy","mixed","loo")`, `default_variant="ro"`, `pca_components=3`, `ff_sizes=(4,...,14,16,24,32,48,64)`, `study_size=8`, `shots_list=(100,250,500,1000)`, `shots_strength_indices=(0,4,7,10)`, `learning_reps=(1,2,4,8,16)`, `learning_strength_indices=(0,7,10)`, tolerances `identity_tol=1e-8`, `pca_corr_min=0.995`, `hc_sigma=3.0`, `hc_floor=0.02`. Existing `mlp_hidden=(32,16)`, `learning_rate=1e-3`, `epochs=300`, `n_training_seeds=5` are used by the MLP.

**Conventions fixed in Phase 7:**

- Features standardized on the training rows only. Training: fields h < 0.6 or h > 1.4, repetitions 1 to 16; testing: all 39 fields, repetitions 17 to 20; accuracies exclude h = h_c.
- h_c^ML from the field-averaged output curve, isotonic fit, linear interpolation; NaN = no crossing (every field assigned to one phase; accuracy 20/38 = 0.526 means "all disordered").
- MLP early stopping off: on separable data validation accuracy saturates within a few epochs and stopped an untrained network (seed spread of h_c^ML 0.12 to 0.21 with it, 0.015 to 0.021 without).
- Seeds: `SeedSequence([2026, 17, s])` (MLP), `[2026, 17, s, 1]` (bootstrap), `[2026, 17, s, 2]` (learning-curve subsets); independent of dataset, strength and variant (common random numbers). Exact features are not bootstrapped (exact logistic results are seed independent).
- PCA on centred, unscaled "corr" features of all fields and repetitions; PC1 oriented to correlate positively with the exact M^2.
- Cross-noise modes: `clean_ed` (train ED shadows, same N), `clean_vqe` (train strength 0, variant gate), `noisy` (train = test strength), `mixed` (all 11 strengths pooled), `loo` (all but the tested strength). Default variant "ro".
- Noise compared across sizes by lambda (N = 8, p = 4: lambda = 320 p), not by p.
- Report h_c^ML always beside the ED value at the same N, never against h_c = 1 alone.

**Saved files in `results/phase7/` (axes [classifier 2, seed 5] innermost):**

- `clean/clean_ed_N{N}` (6), `clean/clean_vqe_{hva|hea}_N{N}_p{p}` (20): `P` [kind 2, feature set 2, 2, 5, 39], `P_rep` [..., 4, 39], `hc`, `ncross`, `acc`, `acc_window`, `acc_deep` [2, 2, 2, 5], `hc_rep` [..., 4]; VQE files also `ed_*` (trained on ED at the same N); `pca_pc1`, `pca_pc1_std`, `pca_pc2` [kind, 39], `pca_evr` [kind, 3], `pca_r_m2`, `pca_h_steep`, `pca_h_mid` [kind], `pca_components` [kind, F]; `m2_exact` [39], `m2_shadow` [39, 20]; `h`, `y`, `train_field`, `n_qubits`, `classifiers`, `seeds`, `kinds`, `feature_sets`, `has_ed`, `source`, `ansatz`, `depth`, `fidelity` (VQE).
- `ff/ff_exact`: [size 11, feature set 2, 2, 5, ...]; `pca_*` [size, ...]; `m2` [size, 39]; `sizes`.
- `noise/noise_{hva|hea}_N{N}_p{p}_realistic_{gate|meas|ro}` (18): [mode 5, strength 11, kind 2, feature set 2, 2, 5, ...]; `pca_*` [strength, kind, ...]; `m2_exact` [strength, 39]; `m2_ed`, `m2_clean` [39]; `modes`, `variant`, `strengths`, `lam`.
- `zne/zne_{hva|hea}_N{N}_p{p}_realistic` (6): [mode 3 (clean_ed, clean_vqe, noisy), method 4 (ro, zne_ro, gate, zne_gate; zne_gate exact only), window strength 2, kind, feature set, 2, 5, ...]; `feature_rms_to_clean` [method, window]; `strengths`, `strength_index`, `lam`.
- `shots/shots_{ed_N8|hva_N8_p4_realistic|hea_N8_p4_realistic}`: [mode 2 (clean_ed, noisy), strength (1 or 4), snapshots 4, 2, 5, ...]; `shots`, `strength_index`, `rebuild_err`.
- `learning/learning_*` (same three sources): [strength, shadows per field 5, 2, 5, ...]; `reps_per_field`, `n_train_rows`.
- `tables/clean_accuracy.csv`, `tables/noise_accuracy.csv`, `tables/zne_accuracy.csv` (h_c^ML mean, seed std, shot SE, total sigma, valid seeds, accuracy all / window / deep), `phase7_summary.txt`. 118 files in total.

**Key results:**

- Validation: classifiers on ED features equal those on free-fermion correlators (1.1e-14); strength-0 noisy data reproduce the noiseless VQE classifiers (5.6e-15, ED-trained 2.1e-14); ro equals gate at p = 0 (exactly); shadow and exact h_c^ML agree (12 of 12 within 3 sigma + 0.02); PC1 correlates with M^2 at 0.9975 to 0.9999; shadow rebuild and live rerun exact; deep-field accuracy 100 %. The ED and free-fermion numbers equal an independent sandbox run to every printed digit.
- Clean: h_c^ML (LR, exact) = 0.761, 0.798, 0.826, 0.845, 0.860, 0.871 for N = 4 to 14; free fermions 0.878, 0.901, 0.917, 0.928, 0.936 for N = 16, 24, 32, 48, 64; 1/N extrapolation 0.954 (below h_c; fixed-threshold hypothesis, not derived). Shadows (LR): 0.781 to 0.883, sigma_tot 0.003 to 0.008. MLP 0.01 to 0.04 higher, seed spread 0.015 to 0.021. PCA midpoint 0.729 to 0.868, PC1 explains 99 % of the variance.
- VQE: an ED-trained classifier transfers to HEA states (h_c^ML 0.740 to 0.874, accuracy 0.88 to 0.95) but not to HVA at p = N/2 (0.59 to 0.66, accuracy 0.80 to 0.83: ordered-phase M^2 about 0.65); HVA at p = N recovers (0.80 to 0.86).
- Noise (N = 8, ro, LR shadows): clean-trained classifiers drift by 2 sigma from lambda = 0.15 (train ED) and 0.32 (train noiseless circuit) and give no crossing at p = 1e-2 (everything disordered). Train = test noise keeps accuracy 0.93 (HVA) and 0.91 (HEA) at lambda = 3.2 (F about 0.1), h_c^ML 0.872 to 0.802 and 0.837 to 0.796. Mixed training: accuracy 0.91 to 0.94 throughout, drift onset lambda = 0.69 (HVA) and 1.49 (HEA). Leave-one-out fails only at the strongest strength (extrapolation): 0.982 (HVA), 0.741 (HEA). N = 4 (lambda at most about 0.7): only the ED-trained HVA drifts, from lambda = 0.33, starting from an already misplaced 0.623 (the p = N/2 transfer failure; corrected in Phase 8, the earlier text said no mode drifts); N = 6 clean-trained drift from lambda = 0.17 to 0.81.
- ZNE (window lambda = 0.07 to 0.17): exact feature error cut to 0.7 to 5.6 % of the unmitigated error; h_c^ML within 0.004 of the noiseless circuit at all 12 points (N = 8: HVA 0.850 / 0.827 to 0.870 / 0.868, HEA 0.804 / 0.790 to 0.816 / 0.815).
- Statistics: sigma_tot 0.006 to 0.02, far below the shifts (0.04 to 0.7), so more shadows are not needed. Learning curves flat (22 rows within 0.015 of 352). 100 snapshots bias ED h_c^ML by +0.025 (0.845 against 0.820). Shot-SE exponent -0.56 (ED); noisy estimates (-0.27 to -0.35) rest on 4 test shadows.
- Common window: clean-trained classification fails at lambda about 0.15 to 0.3, inside the window in which Phase 4 lost the energy witness and the hashing bound (0.1 to 0.5) and Phase 5 found ZNE effective (0.1 to 0.3).

**Decisions taken at the end of Phase 7 (for Phase 8):**

- Central thesis figure: `fig4_hc_vs_noise` (lambda on the top axis), supported by `fig5_accuracy_vs_noise` and `fig8_zne`.
- Report the common error-budget window of Phases 4, 5 and 7 as a unifying result.
- Use the drift of h_c^ML, not the accuracy against h < 1 / h > 1 labels, to compare noise levels: the accuracy has a finite-size ceiling (fields between h_c^ML and 1 count as errors).
- The finite-size convergence of h_c^ML below h_c is presented as a hypothesis (fixed-threshold classifier), with the free-fermion series as evidence.
- T7.5 (QCNN, cluster-Ising) stays optional future work; the Phase 1 cluster-Ising ED files would support it.

## 15. Phase 8: completed

**Status:** done; merged into `main` at 7e2004f, tag `phase8-complete`. Last commits before the merge: the Results notes (1288dc2) and this context update (64e3565). `pytest -q`: 404 tests pass (31 in `test_phase8.py`, 9 in `test_phase8_notes.py`). `python experiments\make_thesis_figures.py`: 16 thesis figures, 146 checks passed, 0 failed, about 8 s; a rerun leaves the committed PDFs and the manifest byte-identical. Report: the Claude Doc "Phase 8: Results Consolidation".

**How to run (from the project root, after Phases 1 to 7):**

- `python scripts\phase8_inventory.py` (about 10 to 30 s): inventory of all results in `results/phase8/inventory.md` and `.json`, with provenance checks (file counts, metadata, stored keys, commit ancestry).
- `python experiments\make_thesis_figures.py` (`--only fig03 fig14`, `--workers`, `--list`): runs every `experiments/thesis/figNN_*.py` in its own process and writes `figures/thesis/MANIFEST.md` and `results/phase8/figure_checks.json`.
- `python experiments\thesis\figNN_<slug>.py` (`--no-save`, `--quiet`, `--latex`): one figure, its checks and notes.

**API:**

- `experiments.thesis.figlib`: `FigureSpec(number, slug, chapter, phases, title, caption, width)`; `Context` (`load(name, subdir, require)`, `load_json`, `load_table`, `param`, `note`, `check_close`, `check_max`, `check_min`, `check_true`, `check_rounded`); `run_figure(spec, build)` (thesis style, style audit, save, sidecar, exit code); helpers `field_index`, `chord_log`, `cc_fit`, `exp_decay_fit`, `extrapolation_weights(scales, degree)`, `error_budget_phase4`, `error_budget_phase5`, `loglog_slope`, `summarize_hc` (the Phase 7 estimator), `table_value` (Phase 7 CSV cells), `drift_onset`, `isotonic_ends`, `crossing_consistent`.
- Reusable from the figure scripts: `fig09.first_crossing` (log-interpolated crossing), `fig11.recovery_strength`, `fig11.estimates`, `fig04.sector_gaps`, `fig14.PHASE7_ONSETS`.
- `utils.plotting` (Phase 0 API unchanged): `thesis_rc`, `thesis_style()`, `set_thesis_figure_style`, `thesis_figure(ncols, nrows, width, aspect)`, `save_thesis_figure`, `panel_labels`, `mark_critical`, `reference_line`, `add_secondary_scale`, `zero_strength_positions`, `label_zero_strength`, `audit_style`, `size_color(N)`; colour maps `SERIES_COLORS`, `CHANNEL_COLORS`, `CHANNEL_LABELS`, `MITIGATION_COLORS`, `MODE_COLORS`, `MODE_LABELS`, `ANSATZ_MARKERS`; `THESIS_WIDTHS`, `THESIS_FONT_SIZES`.
- `utils.io`: `format_commit`, `strip_dirty`; recorded commits end in `-dirty` when tracked files had uncommitted changes.

**`CFG.thesis` (added in Phase 8):** `figure_subdir="thesis"`, `results_subdir="phase8"`, `formats=("pdf","png")`, `png_dpi=300`, `strict=True`; Figure 1: `ed_sizes`, `binder_sizes`, `binder_window`, `ff_tol=1e-9`, `gap_fit_sizes=(8,10,12,14)`, `gap_fit_rtol=1e-4`, `cat_field=0.1`, `cat_entropy_tol=1e-3`, `area_law_field=2.0`, `area_law_tol=1e-4`; shared `identity_tol=1e-10`, `regression_tol=1e-4`; Figures 2 to 7: `cc_size=14`, `cc_fit_sizes`, `c_tol_tfim_periodic=0.01`, `c_tol_xxz_periodic=0.05`, `vqe_sizes`, `vqe_plot_sizes=(8,12)`, `vqe_field=1.0`, `vqe_pstar_fields`, `variational_tol=1e-10`, `exact_depth_tol=1e-9`, `pbc_exact_tol=1e-12`, `bp_fit_tol=0.03`, `bp_flat_max=0.1`, `ent_size=12`, `ent_cc_sizes=(8,10,12)`, `capacity_tol=1e-6`, `pbc_entropy_tol=1e-6`; Figures 8 to 12: `noise_size=8`, `noise_field=1.0`, `negativity_threshold=0.1`, `zero_tol=1e-10`, `hashing_rtol=0.01`, `report_rtol=1e-3`; Figures 13 to 16: `ml_main_kind="shadow"`, `ml_main_features="corr"`, `ml_main_classifier="logistic"`, `drift_sigmas=2.0`, `table_tol=5e-5`, `zne_feature_ratio_max=0.1`.

**Conventions fixed in Phase 8:**

- Thesis figures are drawn at printed size (6.3 in full width, 3.1 in half) with constrained layout and saved without cropping; Computer Modern text and math (`cmr10`, DejaVu Serif fallback); font sizes 8, 9 and 10 pt only; panel labels as left titles; fixed colours for ED (black), HVA (blue), HEA (orange), each system size (log-N viridis, N = 4 to 14), noise channel, mitigation stage and training mode. `audit_style` checks width, font sizes and series colours in every figure; labels beginning with "HVA", "HEA" or "ED" must use their series colour.
- PDFs carry no creation date and PNGs no timestamp: a rerun on unchanged data leaves them byte-identical. Only PDFs and `MANIFEST.md` are tracked; PNGs, sidecars and the inventory are regenerated.
- Every figure validates its data against exact results (free fermions, closed forms, bounds, identities between phases) and against the phase reports and tables to their printed digits. A failed check gives exit code 1.
- Error budget per noise model: Phase 4 λ = p(n_1q + n_2q); hardware-like model (Phases 5 to 7) λ = p(n_1q + 10 n_2q); Phase 5 damping channels λ = p(n_1q + 2 n_2q). The common axis of Figure 16 is the expected number of gate errors per circuit.
- Definitions recovered from the reports: p* near h_c is the hardest field near h_c; Phase 4 "cold" values are the median of the three cold starts; the Phase 3 Fannes–Audenaert bound uses 1 − F ≥ 1e-14 (the stored bound has no floor); a missing h_c^ML means the isotonic fit never crosses 1/2 (accuracy need not be 20/38).

**Saved files:**

- `figures/thesis/fig01_tfim_ed_baseline` to `fig16_error_budget_window` (`.pdf` tracked, `.png` ignored); `figures/thesis/MANIFEST.md` (tracked: caption, command, parameters, inputs with commits, checks and notes of every figure; no timestamps).
- `results/phase8/`: `inventory.md`, `inventory.json`, `figures/figNN_*.json` (sidecars with run time, code commit and input hashes), `figure_checks.json`. All regenerated; nothing to back up beyond Phases 1 to 7.
- `notes/README.md` and `notes/phase1_*.md` to `phase8_*.md`: the Results notes (T8.3), one per phase with status, parameters, key observations, failures and thesis figures.

**Thesis figures:** 1 TFIM ED baseline, 2 central charges, 3 VQE error against depth and p*(h), 4 error against field and parity floor, 5 barren plateaus, 6 VQE entanglement and capacity, 7 central charge against depth, 8 fidelity collapse, 9 mixed-state entanglement, 10 training under noise, 11 error mitigation, 12 folding bias, 13 clean classification, 14 h_c^ML against noise (central), 15 accuracy and ZNE, 16 the common error-budget window.

**Key results:**

- Inventory: 865 result files (85, 182, 62, 182, 158, 78, 118 per phase), every one with metadata, matching stored keys and a producing commit in the current history.
- Reproduction: the figures recompute the phase results from the saved data and match the reports and tables to their printed digits (κ, hashing and witness thresholds, sudden death, training values, folding-bias medians, recovery strengths, Phase 7 h_c^ML, σ_tot and accuracies, drift onsets).
- Unifying result (Figure 16, N = 8, p = 4, h = J): the clean-trained classifiers drift at λ = 0.15 (ED) and 0.32 (noiseless circuit), inside the window λ = 0.113 to 0.518 where Phase 4 loses the energy witness and the hashing bound; readout + ZNE reaches VQE accuracy up to λ = 0.31 (HVA) and 0.11 (HEA); the ZNE window of the classifier inputs is λ = 0.069 to 0.149; noise-aware (mixed) training moves the threshold to 0.69 and 1.49.
- Findings of the consolidation (no result file changed): the stored Fannes–Audenaert bound lacks its fidelity floor (9 states with F_GS = 1 exceed it by up to 6.6e-9); p* at exactly h = J differs from the hardest p* near h_c; Phase 4 cold values are medians; missing crossings do not imply accuracy 20/38 (0.526 to 0.592); the symmetry-broken reference has no meaning in the paramagnet; colours were reused across phases in the synthesis figure; the Phase 7 text "N = 4: no 2-sigma drift in any mode" was wrong (the ED-trained HVA drifts at λ = 0.33).

**Decisions taken at the end of Phase 8:**

- The thesis uses only `figures/thesis/` (16 figures); captions start from `MANIFEST.md`, the Results chapter from `notes/`. Phase figures in `figures/phaseN/` remain working figures.
- State the two error-budget definitions once in the text where Figure 16 is introduced.
- Present the finite-size drift of h_c^ML below h_c as a hypothesis (fixed-threshold classifier; free-fermion extrapolation 0.955).
- Still open before submission: correct the N = 4 sentence in the Phase 7 report; check the references of the Phase 2 to 7 reports; the literature check of the half-final-layer folding bias. T7.5 (QCNN, cluster-Ising) stays optional future work.
