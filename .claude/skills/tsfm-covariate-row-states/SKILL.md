---
name: tsfm-covariate-row-states
description: Compare a TSFM's target-row hidden state (T) with the mean of its covariate-row states (F) and their concatenation (TF) across H standardizations, then rank-average ensembles of the cells, and report everything sorted by payout. Use for requests to use the feature/covariate series' hidden states, to sweep H standardizations on the same states, to ensemble readouts or standardizations, or to extend E222/E223/E224 to another TSFM or dataset.
---

# TSFM covariate-row states, standardization grid and ensembles

Invoke with `/tsfm-covariate-row-states`. See [repository instructions](../../../CLAUDE.md).
This is the E222 → E223 → E224 workflow, distilled. The reference code is
`experiments/e222_c2_featmean_ridge.py`, `experiments/e223_e224_featmean_norms.py`,
`src/data_mani/e222_featmean_embeddings.py` and `src/data_mani/e224_featmean_embeddings.py`.

## 1. The read-out: one forward pass, two blocks

A TSFM given one asset as a group (the target series plus F covariate series)
produces a hidden state for EVERY row. The E212 extractor keeps only the target
row; the covariate rows are thrown away. Keep both from the same pass and store
width `2d`: `[:d]` target row (byte-identical to the plain extractor, so parity
against the existing H must be exactly 0), `[d:]` mean over the F covariate rows
at the same output token.

Where each model puts its rows (read from the libraries, assert the shapes):

| model | row layout | token read |
|---|---|---|
| Chronos-2 | per sample: target row, then its F covariate rows (group_ids repeat) | `-patches + (h-1)//16` |
| t0 | all B target rows first, then B×F covariate rows, sample-major | `n - hp - 1 + j` |
| Timestral | per sample `v_all = 1+F` consecutive rows, target first | `n - hp + j` |

Wrap the existing adapter rather than forking the extractor: the context, label
wall, batching, causal audits and cache protocol stay the same. Do add the new
file's sha to the manifest fingerprint so the cache cannot be confused with the
plain one, and run `--smoke` (audits must pass) before the full extraction.

## 2. Arms × standardizations, one common sample

- Arms: **T** (`[:d]`), **F** (`[d:]`), **TF** (all `2d`, one shared ridge penalty).
- Standardizations: `rolling` (63/126/252 by validation MSE), `pooled`, `asset`,
  `robust`, `cross_section`, `l2` via `src.analysis.tsfm_momentum_study.normalize`.
  The user's per-model default (see `/tsfm-embedding-standardization`) is the
  reference row, not the only row.
- Head: per-asset dual ridge on design C, alpha by validation MSE, fresh final fit.
- Every cell of one TSFM is scored on ONE common sample (same dates, same assets).
  Never compare cells across TSFMs as if paired unless the sample is shared too.

## 3. Ensembles of the cells

Equal-weight average of per-date ranks, scored with `matrix_scores` on that same sample:

1. **Pre-specified**: all T, all F, all TF, all cells, and per standardization the
   T+F *rank blend* (prediction-level), which is the direct counterpart of TF
   (feature-level).
2. **Exhaustive 2–3 member screen** over all cells. Its top rows are chosen on the
   window they are scored on: label them in-sample and read them as an upper bound.
3. **Split-half check**: choose the best blend and the best single cell on the first
   half of dates, score both on the second half, report the paired NW t. This is
   the number to believe.
4. Paired Newey-West (lag 20) deltas against the best single cell, and a heatmap of
   mean per-date rank correlation of the cells' predictions.

## 4. Report conventions

- **Sort every table by payout descending**: the ranked table and both ensemble tables
  by `payout`, the paired-difference table by `d_payout`, the split-half table by
  `H2_payout`, and the summary by best payout.
- One report accumulates versions (v2 adds a study, v3 the next); add a TSFM's
  section, never a separate document.
- Deliver the PDF. Run the `report` stage on `lovelace` (its google-chrome prints);
  on deepfour the snap chromium fails on the NFS home and leaves a 660-byte PDF.
- No frozen protocol or config dumps (see CLAUDE.md § Report presentation).

## 5. What E222/E223 found (Chronos-2, reversal187, 297 dates)

- Under `rolling`, F is worse than T (−0.010 payout). Under every other
  standardization F beats T. F pooled − F rolling is +0.017 (t +3.39), the only
  difference that clears |t| ≥ 2. The default for T is not the default for F.
- Best single cell: cross_section TF, payout 0.0719. TF ≥ T everywhere, never significant.
- Ensembles: all 18 cells give 0.0740 (t +0.36). The best in-sample 3-blend gives
  0.0839 (t +1.72). Split-half: the H1-picked blend beats the H1-picked single by
  +0.020 in H2 (t +2.35), but both fall from about 0.107 to 0.044 and 0.024.

## 6. Operational traps met here

- Register every study before launch (CLAUDE.md § Register every experiment).
- The /home quota is shared by every session. A full quota kills fits with exit 1
  and no traceback. Probe with `dd if=/dev/urandom`, and keep embeddings on /scratch.
- Run the fits as one cluster job per cell (`fit --tsfm --norm --arm`) when no
  single host has room for a wide pool. A finished cell is skipped through its
  `COMPLETE.json`.
