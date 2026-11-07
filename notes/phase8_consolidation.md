# Phase 8: results consolidation

**Status.** Branch `phase8-consolidation`. 395 tests (31 for Phase 8). 16 thesis
figures, 146 checks, all passing; full regeneration in about 8 s
(`python experiments\make_thesis_figures.py`); a rerun leaves the committed PDFs
and the manifest byte-identical.

## Parameters

- One script per thesis figure in `experiments/thesis/` (T8.1), each regenerated
  by one command; `experiments/make_thesis_figures.py` runs all of them and
  writes `figures/thesis/MANIFEST.md` (inputs with commits, parameters, checks,
  notes; no timestamps).
- Thesis style (T8.2, `utils/plotting.py`): printed width 6.3 in (full) or
  3.1 in (half), Computer Modern text and math, font sizes 8, 9, 10 pt only, fixed
  colours for ED, HVA, HEA, system size (log-N viridis), noise channel, mitigation
  stage and training mode. Every figure is audited automatically (width, font
  sizes, series colours). PDFs carry no creation date.
- Provenance: results record the code commit, with "-dirty" for uncommitted
  tracked changes (`utils/io.py`); `scripts/phase8_inventory.py` checks file
  counts, metadata, stored keys and commit ancestry of all 865 result files.
- Settings in `CFG.thesis` (four patches; figure sizes, fields and tolerances).

## Key observations

- **Inventory.** File counts per phase 85, 182, 62, 182, 158, 78, 118, as at the
  Phase 7 backup; every result has metadata, its stored keys match, and every
  producing commit is in the current history (0 provenance problems).
- **Reproduction.** The figures recompute the phase results from the saved data
  and match the phase reports to their printed digits: κ medians and ranges,
  hashing and witness thresholds, sudden death, training values, folding-bias
  medians, recovery strengths, every compared Phase 7 table cell and the drift
  onsets.
- **Unifying result** (Figure 16, checked automatically). At N = 8, p = 4, h = J
  the clean-trained classifiers start to drift inside the window in which Phase 4
  loses its entanglement certificates:

  $$
  0.113 \;\le\; \lambda_{\text{drift}}^{\text{ED}} = 0.15, \ \lambda_{\text{drift}}^{\text{noiseless}} = 0.32 \;\le\; 0.518 .
  $$

  Readout + ZNE reaches VQE accuracy up to λ = 0.31 (HVA) and 0.11 (HEA), and its
  window for the classifier inputs is λ = 0.069 to 0.149. Certified entanglement,
  accurate mitigated energies and clean-trained phase classification all fail
  once a circuit holds about 0.1 to 0.5 expected gate errors; noise-aware training
  moves the classification threshold to λ = 0.69 to 1.49 and beyond.

## Failures and surprises found during consolidation

Each was resolved without changing a result file:

1. **Fannes–Audenaert bound stored without its fidelity floor** (Phase 3): 9 states
   with F_GS = 1 exceeded the stored bound by up to 6.6e-9. Bound recomputed with
   the floor and at the double precision of 1 − F.
2. **p\* "near h_c"** (Phase 2) is the hardest field near h_c, not p* at h = J;
   Figure 3 plots p*(h) and checks it field by field.
3. **"Cold" in Phase 4** is the median of three cold starts, not the best.
4. **Missing crossings** do not imply accuracy 20/38 (Phase 7); replaced by an
   exact consistency check of the isotonic estimator.
5. **Symmetry-broken reference in the paramagnet** (Figure 6) has no meaning there
   (no degeneracy); drawn only where the ground space is twofold.
6. **Colours across phases** (Figure 16): Phase 5 and 7 rows reused colours of the
   Phase 4 channels; now drawn in the hardware-like colour.
7. **Phase 7 text: drift at N = 4.** The report said no mode drifts at N = 4;
   the Phase 7 summary and Figure 14 both give λ = 0.33 for the ED-trained HVA
   (a transcription error in the text, not in the data).

Housekeeping closed: the PDF creation date (deterministic PDFs) and the "-dirty"
commit flag, both open since Phase 2.

## Thesis figures

All 16; see `figures/thesis/MANIFEST.md` for each caption, command, inputs and
checks.
