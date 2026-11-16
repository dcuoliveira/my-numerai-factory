# E38 — TimesFM zero-shot univariate forecast of each symbol's target

- **State:** done (1/1 arms)
- **Runner:** src/analysis/timesfm_zeroshot.py (scored by timesfm_score.py)
- **Models / datasets:** `timesfm` (TimesFM 1.0 200M, `google/timesfm-1.0-200m-pytorch`, v1.2.6 torch backend); univariate per-symbol forecast of `target_binned_return_20` on `crypto/v2.0`
- **Artifacts on disk:** 2.7 MB across 1 forecast run (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/timesfm/e38_target.json

## Main question
First TIME-SERIES foundation model on the board and the first univariate/per-symbol one. Zero-shot (no training): one symbol at a time, TimesFM forecasts that coin's own `target_binned_return_20` history and the forecast is the signal. A foundation-model cousin of the mostly-null E13 AR study, so a WEAK signal is the expected outcome.

## What was tried
1 arm. Bespoke inference outside the row-wise MODELS pipeline (TimesFM is a sequence forecaster). Leak wall at the 20-period embargo: context is only the symbol's targets resolved by t (dates ≤ t−20 on the global grid); forecast horizon 20, signal = mean of the horizon. Scored on `tabpfn/e19_fenccausal`'s 510 dates. Run on the GPU.

## Trained parameters
`config JSON on disk is a run descriptor, not a tuned config.` From `e38_target.json`: data_version `crypto/v2.0`, n_features 1, target `target_binned_return_20`, model `timesfm-1.0-200m`, note "zero-shot univariate, one symbol at a time". No cv_scheme / train_window / refit_every / n_iter / fixed_params are recorded — the model is used zero-shot with no fitting.

## What was found
NUMBERS-ONLY — no written finding; from the metric JSON. **The signal is statistically zero**, exactly the weak outcome the idea predicted:
- CORR mean 0.011949, MMC mean 0.012541, n_periods 510. (pnl mean is NaN in the JSON — no payout series was formed for this bespoke arm.)
This CORR ≈ 0.012 is the figure E39's framing comparison cites as "statistically zero" for TimesFM-on-the-target's-own-history. No t-stat against a baseline is recorded.
