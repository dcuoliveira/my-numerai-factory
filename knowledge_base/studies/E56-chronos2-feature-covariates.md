# E56 — Chronos-2 with the feature cross-section as covariates

- **State:** done (1/1 arm)
- **Runner:** src/analysis/chronos2_covariates.py
- **Models / datasets:** model `chronos2` (amazon/chronos-2, ~120M, zero-shot); run `e56_fenccausal`; config data_version `crypto/v2.0`, scored on tabpfn/e19_fenccausal's dates
- **Artifacts on disk:** 1.92 MB across 1 forecast run (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/chronos2/e56_fenccausal.json

## Main question
Can a time-series foundation model that ingests EXOGENOUS COVARIATES finally use the feature cross-section, closing the gap between univariate TS-FMs (which scored ~0: TimesFM E38 +0.012, TEMPO E40 -0.017) and the tabular FM that worked (TabPFN E19 CORR 0.145)? E56 feeds the 44 fenccausal features as Chronos-2 covariates.

## What was tried
One arm. Per scored era T: a single `Chronos2Pipeline.predict_df` panel call over all symbols (group attention shares cross-series info); context = every symbol's (target+features) up to the 20-period leak wall T-20; features for eras T-19..T passed as known-future covariates; prediction_length 20; the era-T median is the signal. Zero-shot. Scored on e19_fenccausal's 510 dates. Intended paired Newey-West comparisons vs E38/E40 (does covariate support beat univariate?) and vs E19 (does a TS-FM with covariates reach the tabular edge?) — those paired tests are not in the metrics JSON read.

## Trained parameters
From results/metrics/crypto/chronos2/e56_fenccausal.json: model `chronos2`, data_version `crypto/v2.0`, target `target_binned_return_20`, n_features 1 (as recorded; the 44 fenccausal covariates are fed at call time per the idea, not counted in this field), n_periods 510. Zero-shot foundation model: the config records no cv_scheme, embargo, train_window, refit_every, n_splits, seed, or n_iter, and there are 0 refits / no best_params (nothing is tuned).

## What was found
No written finding; from the metrics JSON: CORR mean 0.002516 (sharpe 0.0215) and MMC mean 0.011477 over 510 periods; the payout/pnl field is not populated. The CORR is essentially zero — far below the tabular-FM edge the idea cites for TabPFN (E19 CORR 0.145) and in the same near-null band as the univariate TS-FMs the idea cites (E38/E40). Direction: the covariate support did NOT reach the tabular cross-sectional edge. (The E38/E40/E19 comparison values come from the study idea, not re-measured here.)
