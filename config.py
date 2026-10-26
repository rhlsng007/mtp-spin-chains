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
# VQE optimisation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VQEConfig:
    optimizer_noiseless: str = "L-BFGS-B"
    optimizer_noisy: str = "SPSA"
    max_iterations: int = 1000
    tolerance: float = 1e-8
    n_random_inits: int = 8             # random restarts per point
    use_warm_start: bool = True         # reuse parameters from previous field value
    init_scale: float = 0.1             # std of random initial parameters
    spsa_iterations: int = 500
    spsa_a: float = 0.2                 # SPSA learning-rate scale
    spsa_c: float = 0.1                 # SPSA perturbation scale


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

    def strengths(self) -> np.ndarray:
        """Logarithmically spaced noise strengths, plus the noiseless point 0."""
        grid = np.logspace(np.log10(self.p_min), np.log10(self.p_max), self.n_points)
        return np.concatenate(([0.0], np.round(grid, 8)))


# ---------------------------------------------------------------------------
# Machine learning
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MLConfig:
    train_h_ordered_max: float = 0.6     # train only deep in the ordered phase ...
    train_h_disordered_min: float = 1.4  # ... and deep in the disordered phase
    test_fraction: float = 0.2
    n_shadow_snapshots: int = 1000
    mlp_hidden: tuple = (32, 16)
    learning_rate: float = 1e-3
    epochs: int = 300
    n_training_seeds: int = 5


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
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)

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
