"""
Central configuration for the MTP thesis project.

Every experiment script imports its parameters from here, so that any
figure in the thesis can be regenerated from a single source of truth.
Change a value here, rerun the experiment script, and the figure updates.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helper for building parameter grids without floating-point drift
# ---------------------------------------------------------------------------
def make_grid(start: float, stop: float, step: float, decimals: int = 6) -> np.ndarray:
    """
    Inclusive, evenly spaced grid from start to stop.

    np.arange accumulates floating-point error (e.g. 0.30000000000000004),
    which breaks dictionary lookups and file names. Using linspace with an
    explicit point count and rounding avoids this.
    """
    n_points = int(round((stop - start) / step)) + 1
    return np.round(np.linspace(start, stop, n_points), decimals)


# ---------------------------------------------------------------------------
# Physical models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TFIMConfig:
    J: float = 1.0                       # Ising coupling (energy unit)
    h_critical: float = 1.0              # exact critical field for J = 1
    h_min: float = 0.1
    h_max: float = 2.0
    h_step_noiseless: float = 0.05
    h_step_noisy: float = 0.1
    boundary: str = "open"               # "open" or "periodic"

    def h_grid(self, noisy: bool = False) -> np.ndarray:
        step = self.h_step_noisy if noisy else self.h_step_noiseless
        return make_grid(self.h_min, self.h_max, step)


@dataclass(frozen=True)
class XXZConfig:
    J: float = 1.0                       # antiferromagnetic coupling
    delta_min: float = -0.5
    delta_max: float = 2.0
    delta_step_noiseless: float = 0.1
    delta_step_noisy: float = 0.25
    boundary: str = "open"

    def delta_grid(self, noisy: bool = False) -> np.ndarray:
        step = self.delta_step_noisy if noisy else self.delta_step_noiseless
        return make_grid(self.delta_min, self.delta_max, step)


@dataclass(frozen=True)
class ClusterIsingConfig:
    J: float = 1.0
    h1_min: float = 0.0
    h1_max: float = 1.6
    h2_values: tuple = (-1.0, 0.0, 1.0)
    h1_step: float = 0.1
    boundary: str = "open"

    def h1_grid(self) -> np.ndarray:
        return make_grid(self.h1_min, self.h1_max, self.h1_step)


# ---------------------------------------------------------------------------
# System sizes and circuits
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SystemConfig:
    sizes_exact: tuple = (4, 6, 8, 10, 12, 14)   # exact diagonalization
    sizes_noiseless: tuple = (4, 6, 8, 10, 12)   # state-vector VQE
    sizes_noisy: tuple = (4, 6, 8)               # density-matrix VQE

    @staticmethod
    def depths(n_qubits: int, noisy: bool = False) -> list:
        """Circuit depths p to scan: 1..N noiseless, 1..N/2 noisy."""
        p_max = n_qubits // 2 if noisy else n_qubits
        return list(range(1, max(p_max, 1) + 1))


# ---------------------------------------------------------------------------
# Exact diagonalization (Phase 1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EDConfig:
    k_per_sector: int = 3               # lowest eigenpairs kept in each symmetry sector
    dense_max_dim: int = 512            # sectors up to this dimension use dense eigh
    eigsh_tol: float = 0.0              # 0 = machine precision in scipy's eigsh
    degeneracy_tol: float = 1e-9        # levels closer than this count as degenerate
    eigsh_seed: int = 0                 # fixed Lanczos start vector -> reproducible
    save_states: bool = True            # store lowest state of every sector (Phases 2, 6)
    cc_ell_min: int = 2                 # Calabrese-Cardy fit uses ell_min <= ell <= N/2
    cc_boundaries: tuple = ("open", "periodic")
    xxz_cc_deltas: tuple = (0.0, 0.5)   # gapless XXZ points used for c = 1 checks


# ---------------------------------------------------------------------------
# VQE optimisation (Phase 2; SPSA settings are used in Phase 4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VQEConfig:
    # Optimizer (T2.3)
    optimizer_noiseless: str = "L-BFGS-B"   # "L-BFGS-B" or "Adam"
    optimizer_noisy: str = "SPSA"
    max_iterations: int = 1000
    tolerance: float = 1e-8             # stop when max |dE/dtheta| < tolerance
    lbfgs_ftol: float = 1e-15           # relative energy-decrease stop (tiny: gtol decides)
    lbfgs_maxcor: int = 20              # L-BFGS memory (stored correction pairs)
    adam_lr: float = 0.02
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # Initializations (T2.4)
    n_random_inits: int = 8             # random restarts per point (first point, anchors, scans)
    n_random_inits_sweep: int = 2       # random restarts at the other warm-started points
    init_scale: float = 0.1             # std of random initial parameters
    init_distribution: str = "normal"   # "normal" (small, around the reference state) or "uniform"
    tie_tolerance: float = 1e-10        # energies this close count as equal (earlier init wins)
    anchor_fields: tuple = (0.5, 1.0, 1.5)  # fields that always get n_random_inits restarts

    # Warm start (T2.5)
    use_warm_start: bool = True         # reuse parameters from previous field value
    hva_sweep_direction: str = "descending"  # HVA starts at |+>: begin deep in the paramagnet
    hea_sweep_direction: str = "ascending"   # HEA starts at |0...0>: begin deep in the ferromagnet
    hva_depth_warm_start: bool = True   # also start from the depth p-1 optimum + identity layer
    warmcold_size: int = 8              # warm-versus-cold comparison at N = 8, p = N/2

    # Sweeps and analysis (T2.6)
    ansatze: tuple = ("hva", "hea")
    ground_space_gap_ratio: float = 0.1  # sector in ground space if E_s - E0 < ratio * gap
    target_rel_error: float = 1e-3      # threshold defining the required depth p*(h)
    hva_depth_scan_sizes: tuple = (4, 6, 8, 10)
    hva_depth_scan_factor: int = 2      # HVA depth scan at anchor fields up to p = factor * N
    save_states: bool = True            # store optimized states (Phase 3 entanglement)

    # Barren plateaus (T2.7)
    grad_var_sizes: tuple = (4, 6, 8, 10, 12, 14)
    grad_var_samples: int = 500         # random parameter draws per (ansatz, N, depth)
    grad_var_fixed_depth: int = 2       # shallow comparison depth (besides p = N)
    grad_var_field: float = 1.0         # TFIM cost function evaluated at h = h_c

    # SPSA (Phase 4, T4.3)
    spsa_iterations: int = 500
    spsa_a: float = 0.2                 # generic default of vqe.spsa (target first step)
    spsa_c: float = 0.1                 # generic default perturbation scale
    spsa_alpha: float = 0.602           # a_k = a / (k + 1 + A)^alpha  (Spall)
    spsa_gamma: float = 0.101           # c_k = c / (k + 1)^gamma      (Spall)
    spsa_A_fraction: float = 0.1        # A = fraction * spsa_iterations
    spsa_calibration_steps: int = 20    # random directions in the gain calibration
    spsa_average_last: int = 50         # Polyak averaging over the last iterates
    # Per-ansatz settings tuned in Phase 4 (HVA: 2p parameters that each move a
    # whole layer, so a 0.05 rad first step already diverges at h = 1):
    spsa_target_hva: float = 0.02       # first-step size per parameter, cold start
    spsa_c_hva: float = 0.05
    spsa_target_hea: float = 0.1
    spsa_c_hea: float = 0.1
    spsa_warm_factor: float = 0.1       # warm starts use factor * cold gain


# ---------------------------------------------------------------------------
# Entanglement of VQE states (Phase 3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EntanglementConfig:
    rank_tol: float = 1e-14             # Schmidt weights above this count toward the rank
    cc_field: float = 1.0               # field of the central-charge analysis (h_c)
    cc_ell_min: int = 2                 # Calabrese-Cardy fits use ell_min <= ell <= N/2 (as Phase 1)
    maxent_starts: int = 6              # starts for maximizing the half-chain S_2 per (ansatz, N, p)
    maxent_maxiter: int = 500
    maxent_gtol: float = 1e-9
    random_samples: int = 200           # uniform random parameter draws per (ansatz, N, p)
    pbc_sizes: tuple = (4, 6, 8, 10, 12)  # periodic HVA at h_c, p = 1..N/2
    bound_tol: float = 1e-9             # slack allowed in the rigorous-bound checks
    fidelity_resolution: float = 1e-14  # 1 - F below this is rounding; floor used in the
                                        # Fannes-Audenaert check (S is not Lipschitz near F = 1)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NoiseConfig:
    p_min: float = 1e-4
    p_max: float = 5e-2
    n_points: int = 10
    channels: tuple = ("depolarizing", "amplitude_damping", "dephasing")
    shots_list: tuple = (1000, 5000, 10000)
    zne_scale_factors: tuple = (1, 3, 5)

    # Phase 4 additions
    two_qubit_factor: float = 1.0       # p_2q = factor * p (IBM devices: about 10)
    noisy_rz: bool = True               # False = virtual RZ (noiseless), as on IBM hardware
    noisy_state_prep: bool = True       # the HVA Hadamard layer carries gate noise
    shot_repetitions: int = 50          # independent finite-shot estimates per point (T4.2)
    train_shots: int = 10000            # shots per basis per energy in SPSA training (T4.5 B)
    train_fields: tuple = (0.5, 1.0, 1.5)   # fields of the optimize-under-noise runs
    weak_noise_max: float = 1e-3        # strengths used in the linear-response slope check
    slope_tol: float = 0.15             # |slope - 1| allowed in that check
    monotone_tol: float = 1e-12         # slack in the purity-monotonicity check

    def strengths(self) -> np.ndarray:
        """Logarithmically spaced noise strengths, plus the noiseless point 0."""
        grid = np.logspace(np.log10(self.p_min), np.log10(self.p_max), self.n_points)
        return np.concatenate(([0.0], np.round(grid, 8)))


# ---------------------------------------------------------------------------
# Error mitigation (Phase 5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MitigationConfig:
    """
    Phase 5 settings. The Phase 4 fields in NoiseConfig are left untouched so the
    Phase 4 results stay reproducible; Phase 5 uses its own hardware-like model
    (decided at the end of Phase 4): two-qubit error 10 times the one-qubit
    error, virtual (noiseless) RZ, and readout error. The ZNE scale factors are
    CFG.noise.zne_scale_factors = (1, 3, 5).
    """
    # Gate noise. "realistic" = depolarizing p (1q) and two_qubit_factor * p (2q),
    # plus amplitude damping and dephasing with gamma = ratio * p on every qubit a
    # gate touches. The other three channels are the Phase 4 one-parameter models
    # with the Phase 5 two-qubit factor and virtual RZ.
    channels: tuple = ("realistic", "depolarizing", "amplitude_damping", "dephasing")
    two_qubit_factor: float = 10.0
    noisy_rz: bool = False
    realistic_damping_ratio: float = 1.0
    realistic_dephasing_ratio: float = 1.0
    p_min: float = 1e-5                 # strength grid: 0 plus n_points log-spaced values
    p_max: float = 1e-2
    n_points: int = 10

    # Readout error (T5.1): independent, asymmetric bit flips at measurement
    readout_p01: float = 0.01           # P(read 1 | qubit in 0)
    readout_p10: float = 0.02           # P(read 0 | qubit in 1), larger: T1 decay during readout
    calibration_shots: int = 10000      # shots per calibration circuit
    calibration_gate_noise: bool = True # calibration X gates carry the 1q gate noise (SPAM)

    # Zero-noise extrapolation (T5.2)
    foldings: tuple = ("global", "local")
    main_folding: str = "global"        # folding used for the finite-shot statistics
    extrapolators: tuple = ("linear", "richardson", "exponential")
    main_extrapolator: str = "richardson"

    # Finite-shot statistics (T5.3)
    shots: int = 10000                  # shots per measurement basis and per scale factor
    shot_repetitions: int = 50          # independent repetitions (each with its own calibration)

    # Readout experiment (T5.1)
    readout_field: float = 1.0
    readout_cal_shots: tuple = (1000, 10000, 100000)

    # Figures
    plot_size: int = 8
    plot_field: float = 1.0
    plot_strength: float = 1e-3         # strength of the field-sweep figure (on the grid)
    plot_strengths_curves: tuple = (1e-4, 1e-3, 4.64159e-3)   # E-versus-scale panels

    # Validation tolerances
    weak_fold_tol: float = 0.02         # |ratio - scale| / scale at the weakest noise (depolarizing)
    order_tol: float = 0.15             # |log-log slope - expected order|
    regime_lambda: float = 0.2          # error budget below which mitigation must help everywhere
    variance_ratio_tol: float = 0.1     # |median empirical / predicted variance - 1|
    exact_tol: float = 1e-10            # exact-limit identities

    def strengths(self) -> np.ndarray:
        grid = np.logspace(np.log10(self.p_min), np.log10(self.p_max), self.n_points)
        return np.concatenate(([0.0], np.round(grid, 10)))


# ---------------------------------------------------------------------------
# Machine learning
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MLConfig:
    train_h_ordered_max: float = 0.6     # train only deep in the ordered phase ...
    train_h_disordered_min: float = 1.4  # ... and deep in the disordered phase
    test_fraction: float = 0.2           # Phase 6: share of shadow repetitions kept for testing only
    n_shadow_snapshots: int = 1000       # snapshots M per classical shadow
    mlp_hidden: tuple = (32, 16)
    learning_rate: float = 1e-3
    epochs: int = 300
    n_training_seeds: int = 5

    # Phase 6: datasets (T6.1 to T6.4)
    shadow_repetitions: int = 20         # independent shadows R per field (one sample each)
    label_ordered: int = 0               # h < h_c
    label_disordered: int = 1            # h > h_c
    label_critical: int = -1             # h = h_c: no label, excluded from accuracies
    ed_sizes: tuple = (4, 6, 8, 10, 12, 14)
    vqe_sizes: tuple = (4, 6, 8, 10, 12)
    vqe_depth_rules: tuple = ("half", "full")   # p = N/2 (the noisy depth) and p = N
    noisy_sizes: tuple = (4, 6, 8)
    noisy_depth_rule: str = "half"       # Phase 4/5 depth p = N/2
    noisy_channels: tuple = ("realistic",)      # Phase 5 decision; Phase 4 channels optional
    noisy_fine_grid: bool = True         # fixed parameters: noisy data on the step-0.05 grid too
    zne_lambda_window: tuple = (0.05, 0.3)      # ZNE dataset only where it can help (Phase 5)
    zne_channels: tuple = ("realistic",)
    sampler_chunk_elements: int = 2 ** 22       # amplitudes held at once by the pure-state sampler

    # Phase 6 validation tolerances
    exact_tol: float = 1e-10             # identities (energies, strength-0 limits, PSD floor)
    ff_tol: float = 1e-8                 # ED correlators against free fermions
    z_frac_max: float = 0.01             # share of shadow z-scores beyond 3 sigma (expected 0.27 %)
    z_abs_max: float = 6.0               # no shadow z-score beyond this
    t_frac_max: float = 0.02             # share of |t| > 3 (t with R - 1 dof, expected about 0.7 %)
    var_ratio_tol: float = 0.1           # |median empirical / analytic shadow variance - 1|
    snapshot_match_min: float = 0.9999   # noisy strength 0 against noiseless VQE snapshots

    # Phase 7: classification (T7.1 to T7.6)
    classifiers: tuple = ("logistic", "mlp")
    feature_sets_phase7: tuple = ("corr", "compact")
    logistic_C: float = 1.0              # inverse L2 strength (standardized features)
    logistic_max_iter: int = 5000
    mlp_alpha: float = 1e-3              # MLP L2 penalty
    mlp_batch_size: int = 64
    mlp_early_stopping: bool = False     # off: on separable data validation accuracy saturates at 1
                                         # within a few epochs and stops an untrained network
    mlp_validation_fraction: float = 0.15
    mlp_patience: int = 30               # epochs without improvement before stopping
    bootstrap_training: bool = True      # seed-dependent bootstrap of the training shadows
    decision_threshold: float = 0.5      # p(y = 1 | x) = 1/2 defines h_c^ML
    noise_modes: tuple = ("clean_ed", "clean_vqe", "noisy", "mixed", "loo")
    default_variant: str = "ro"          # Phase 6 decision: readout-corrected shadows
    pca_components: int = 3
    ff_sizes: tuple = (4, 6, 8, 10, 12, 14, 16, 24, 32, 48, 64)   # free-fermion exact features
    study_size: int = 8                  # N of the shots test, learning curve and central figure
    shots_list: tuple = (100, 250, 500, 1000)
    shots_strength_indices: tuple = (0, 4, 7, 10)          # p = 0, 1e-4, 1e-3, 1e-2
    learning_reps: tuple = (1, 2, 4, 8, 16)                # training shadows per training field
    learning_strength_indices: tuple = (0, 7, 10)

    # Phase 7 validation tolerances
    identity_tol: float = 1e-8           # classifier outputs of identical inputs
    pca_corr_min: float = 0.995          # |corr(PC1, exact M^2)| for ED exact features
    hc_sigma: float = 3.0                # shadow vs exact h_c^ML agreement, in combined sigma ...
    hc_floor: float = 0.02               # ... plus this floor (grid step 0.05)



# ---------------------------------------------------------------------------
# Thesis figures and results consolidation (Phase 8)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThesisConfig:
    # Output: figures/<figure_subdir>/figNN_<slug>.pdf/.png; sidecars and
    # inventory in results/<results_subdir>/
    figure_subdir: str = "thesis"
    results_subdir: str = "phase8"
    formats: tuple = ("pdf", "png")
    png_dpi: int = 300
    strict: bool = True                  # a failed check gives exit code 1
    # Figure 1: exact-diagonalization baseline of the TFIM (Phase 1 data)
    ed_sizes: tuple = (4, 6, 8, 10, 12, 14)
    binder_sizes: tuple = (6, 8, 10, 12, 14)   # consecutive pairs give crossings
    binder_window: tuple = (0.5, 1.5)    # crossings searched only in this field range
    ff_tol: float = 1e-9                 # ED E0 and E1 - E0 against free fermions
    gap_fit_sizes: tuple = (8, 10, 12, 14)   # N*gap(h_c) = a + b/N + c/N^2 -> a = pi
    gap_fit_rtol: float = 1e-4
    cat_field: float = 0.1               # ordered-phase plateau S(N/2) = ln 2
    cat_entropy_tol: float = 1e-3      # largest deviation is N = 4 (2e-4); N >= 6: 6e-6
    area_law_field: float = 2.0          # paramagnet: S(N/2) saturates with N
    area_law_tol: float = 1e-4


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SeedConfig:
    master_seed: int = 2026
    n_seeds_noiseless: int = 5
    n_seeds_noisy: int = 3

    def seeds(self, noisy: bool = False) -> list:
        n = self.n_seeds_noisy if noisy else self.n_seeds_noiseless
        return [self.master_seed + k for k in range(n)]


# ---------------------------------------------------------------------------
# Single global configuration object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    tfim: TFIMConfig = field(default_factory=TFIMConfig)
    xxz: XXZConfig = field(default_factory=XXZConfig)
    cluster: ClusterIsingConfig = field(default_factory=ClusterIsingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    ed: EDConfig = field(default_factory=EDConfig)
    vqe: VQEConfig = field(default_factory=VQEConfig)
    ent: EntanglementConfig = field(default_factory=EntanglementConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    thesis: ThesisConfig = field(default_factory=ThesisConfig)

    def to_dict(self) -> dict:
        """Plain dictionary, stored alongside every saved result."""
        return asdict(self)


CFG = Config()


if __name__ == "__main__":
    print("TFIM h grid (noiseless):", CFG.tfim.h_grid())
    print("TFIM h grid (noisy):    ", CFG.tfim.h_grid(noisy=True))
    print("XXZ delta grid:         ", CFG.xxz.delta_grid())
    print("Cluster h1 grid:        ", CFG.cluster.h1_grid())
    print("Noise strengths:        ", CFG.noise.strengths())
    print("Depths for N=8:         ", CFG.system.depths(8))
    print("Depths for N=8 (noisy): ", CFG.system.depths(8, noisy=True))
    print("Seeds:                  ", CFG.seeds.seeds())
    print("ED settings:            ", CFG.ed)
    print("VQE settings:           ", CFG.vqe)
    print("Entanglement settings:  ", CFG.ent)
    print("Noise settings:         ", CFG.noise)
    print("Mitigation settings:    ", CFG.mitigation)
    print("Mitigation strengths:   ", CFG.mitigation.strengths())
    print("ML settings:            ", CFG.ml)
