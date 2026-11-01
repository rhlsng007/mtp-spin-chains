"""
Phase 4 diagnostic: points where the half-chain log negativity at the strongest
noise is not below its zero-noise value, and where E_N rises anywhere along
the strength sweep.

Usage (from the project root, after experiments\\phase4_noise.py --experiments eval):
    python scripts\\phase4_diag_negativity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import CFG, RESULTS_DIR

THRESHOLD = 0.01          # "entangled at zero noise", as in phase4_plots.py
RISE_TOL = 1e-9

np.set_printoptions(precision=4, suppress=True, linewidth=200)
files = sorted((RESULTS_DIR / "phase4").glob("noisy_eval_*.npz"))
if not files:
    sys.exit("no noisy_eval files in results/phase4")

for ch in CFG.noise.channels:
    print(f"\n===== {ch} =====")
    n_end = n_rise = n_ent = 0
    for f in files:
        d = np.load(f)
        ic = list(d["channels"]).index(ch)
        s = d["strengths"]
        EN = d["E_N"][ic]                      # [n_s, n_h]
        ent = EN[0] > THRESHOLD
        n_ent += int(ent.sum())
        rises = (np.diff(EN, axis=0) > RISE_TOL).any(axis=0) & ent
        end = (EN[-1] >= EN[0]) & ent
        n_rise += int(rises.sum())
        n_end += int(end.sum())
        for ih in np.flatnonzero(rises):
            j = int(np.argmax(EN[:, ih]))
            print(f"{f.stem:24s} h={d['h'][ih]:.1f}  {'END>=START' if end[ih] else 'rises only':10s}  "
                  f"E_N(0)={EN[0, ih]:.4f}  max {EN[j, ih]:.4f} at p={s[j]:.1e}  E_N(pmax)={EN[-1, ih]:.4f}  "
                  f"S_A(0)={d['S_A'][ic, 0, ih]:.4f}  F={d['fidelity'][ic, 0, ih]:.3f}  "
                  f"F_GS={d['fidelity_gs'][ic, 0, ih]:.3f}  eps(0)={d['rel_error'][ic, 0, ih]:.1e}")
    print(f"-- {ch}: {n_ent} entangled points; E_N(pmax) >= E_N(0) at {n_end}; "
          f"E_N rises somewhere along the sweep at {n_rise}")
