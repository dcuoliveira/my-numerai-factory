# E05 — Cumulative feature lags, all 22 features

- **State:** done (11/11 arms)
- **Runner:** experiments/e05_cumlags_lgb_reg_crypto.sh
- **Models / datasets:** lgb; crypto/v2.0-cumlag{1..10} (raw 22 plus every lag 1..p) vs cumlag0 = raw crypto/v2.0
- **Artifacts on disk:** 49.6 MB across 11 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/lgb/e05_cumlag3.json
- Note: `payout` = 0.1*CORR_mean + MMC_mean from stored summaries.

## Main question
Raw 22 plus every lag 1..p of all of them. A tree can split on both the contemporaneous
and lagged columns directly, so if lags help a booster it is for more history to
condition on, not the representational fix ridge got. Does any lag depth beat the
no-lag baseline?

## What was tried
11 arms, lag depth p = 0..10 (cumlag0 is raw/no lags; cumlag3 has n_features 88 = 22x4).
Same pinned booster/cadence as E01-E04. 510 periods, 20-day label.

## Trained parameters
From results/metrics/crypto/lgb/e05_cumlag3.json: model lgb, regression,
target_binned_return_20, data_version crypto/v2.0-cumlag3, n_features 88, cv_scheme
rolling, n_splits 2, train_window 130, refit_every 130, embargo 20, seed 42,
n_iter 50 (nominal). LightGBM booster PINNED via `fixed_params` (every searched bound has size 1, so `n_iter=50` is nominal): colsample_bytree 0.070213, learning_rate 0.002572, max_depth 8, n_estimators 4327, num_leaves 255, reg_lambda 1.757511, subsample 0.816153, subsample_freq 1. cumlag0 is the same booster on crypto/v2.0 (22 feat).

## What was found
No written finding; from the metrics (510 periods): lags do not help — depth 0 wins.
`cumlag0` (no lags) pays 0.083272 at CORR 0.149754, the best of the 11. Every lag arm
is lower and the payout falls roughly monotonically with depth: cumlag1 0.079724,
cumlag3 0.076671, cumlag10 0.073777. Adding lagged columns strictly degrades the booster
here; no paired t-stats in available artifacts (the study idea notes LightGBM's best
lag depth was zero).
