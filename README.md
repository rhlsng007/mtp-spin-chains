# Variational Quantum Simulation of 1D Spin Chains

MTP thesis code: entanglement, noise robustness, and machine-learning-based phase classification.

## 1. Setup

Requires Python 3.10 or newer.

**Option A: venv**

```bash
cd mtp-spin-chains
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Option B: conda**

```bash
conda create -n mtp python=3.11 -y
conda activate mtp
pip install -r requirements.txt
```

If you have an NVIDIA GPU and want CUDA PyTorch, install torch first using the command from https://pytorch.org, then run `pip install -r requirements.txt`.

## 2. Verify the environment

```bash
python scripts/check_env.py
pytest -q
```

Every check compares against an exact analytical result, for example the noisy Bell-state correlator `<ZZ> = 1 - 4p/3`. All checks must pass before starting Phase 1.

## 3. Version control

```bash
git init
git add -A
git commit -m "Phase 0: project setup"
```

Commit before every large run. Each saved result records the current git commit, so every number in the thesis can be traced to the exact code that produced it.

## 4. Project structure

| Folder | Purpose | Phase |
|---|---|---|
| `config.py` | All parameters and seeds (single source of truth) | 0 |
| `utils/` | Saving/loading results, seeding, plotting style | 0 |
| `models/` | Hamiltonians and exact diagonalization | 1 |
| `analysis/` | Entanglement entropies, correlators, CFT fits | 1, 3 |
| `vqe/` | HVA and HEA ansätze, optimisation loop | 2 |
| `noise/` | Noise channels and error mitigation | 4, 5 |
| `ml/` | Dataset generation and classifiers | 6, 7 |
| `experiments/` | One script per thesis figure | all |
| `results/` | Saved raw data (`.npz` + `.json` metadata) | all |
| `figures/` | Final plots (`.pdf` for LaTeX, `.png` for slides) | all |
| `tests/` | Unit tests | all |

## 5. Conventions

1. **Never hard-code parameters** in experiment scripts. Import them from `config.py`.
2. **Always save raw data** with `utils.io.save_results` or `load_or_compute`, then plot from the saved data in a separate step. Expensive simulations are never rerun just to restyle a plot.
3. **Always seed** with `utils.seeding.set_seed` and pass the returned generator to functions that need randomness.
4. **Always plot** with `utils.plotting.set_thesis_style` and `save_figure`, so all figures share one style.
5. **Run all scripts from the project root**, e.g. `python experiments/phase1_ed_sweep.py`.
