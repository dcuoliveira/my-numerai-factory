# E06 — Cumulative lags, volatility group only

- **State:** done (11/11 arms)
- **Runner:** experiments/e05_cumlags_lgb_reg_crypto.sh (SCOPE=volatility)
- **Models / datasets:** lgb; crypto/v2.0-cumlag{0..10} restricted to the volatility group via `feature_include ^feature_volatility_`
- **Artifacts on disk:** 34.4 MB across 11 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/lgb/e06_volcumlag10.json
- Note: `payout` = 0.1*CORR_mean + MMC_mean from stored summaries.

## Main question
E05 narrowed to the volatility group and its lags, to separate "lags help" from
"more columns help": the number of base features is held fixed to one group while only
the lag depth grows.

## What was tried
11 arms, volatility group with lag depth 0..10 (volcumlag0..10), selected by
`feature_include ^feature_volatility_` on the cumlagN datasets. Same pinned booster and
cadence. 510 periods, 20-day label.

## Trained parameters
From results/metrics/crypto/lgb/e06_volcumlag10.json: model lgb, regression,
target_binned_return_20, data_version crypto/v2.0-cumlag10, n_features 22,
feature_include `^feature_volatility_`, cv_scheme rolling, n_splits 2, train_window 130,
refit_every 130, embargo 20, seed 42, n_iter 50 (nominal). LightGBM booster PINNED via `fixed_params` (every searched bound has size 1, so `n_iter=50` is nominal): colsample_bytree 0.070213, learning_rate 0.002572, max_depth 8, n_estimators 4327, num_leaves 255, reg_lambda 1.757511, subsample 0.816153, subsample_freq 1.

## What was found
No written finding; from the metrics (510 periods): again depth 0 is best and lags hurt.
`volcumlag0` pays 0.067608 (CORR 0.136523) — the top arm — and the lagged arms sit
below it (volcumlag1 0.067420, volcumlag3 0.066483, volcumlag10 0.064651), falling with
depth. Within the volatility group too, adding lagged history degrades the booster; no
paired t-stats in available artifacts (the study idea cites E06 t −2.18..−2.30 for the
lag arms).
