# E02 — Group blend vs group concat, on LightGBM

- **State:** done (12/12 arms)
- **Runner:** experiments/e02_group_blends_lgb_reg_crypto.sh
- **Models / datasets:** lgb on crypto/v2.0 (raw 22); feature groups selected by `feature_include` regex (momentum, volatility, bollinger, oscillator, liquidity, size, price)
- **Artifacts on disk:** 39.7 MB across 12 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/lgb/e02_concat_quintet.json
- Note: `payout` below = 0.1*CORR_mean + MMC_mean from the stored summaries (the repo's ranking convention).

## Main question
Re-asks the ridge group-blend study for a model class whose answer should differ:
fit each feature group as its own LightGBM and rank-average the predictions (blend),
or fit one model on the union of the groups (concat)?

## What was tried
12 arms: per-group single-model fits (grp_momentum, grp_volatility, grp_bollinger,
grp_oscillator, grp_liquidity, grp_size, grp_price), two concat arms over the union
(concat_trio, concat_quintet), two rank-average blend arms (blend_trio, blend_quintet),
and raw (all 22). Groups are carved by a `feature_include` regex on crypto/v2.0.
concat_quintet selects 5 groups (n_features 14). The blend arms carry no target in their
config (target = None; run_ensemble writes none) and are scored on the 20-day label.

## Trained parameters
From results/metrics/crypto/lgb/e02_concat_quintet.json: model lgb, task regression,
target target_binned_return_20, data_version crypto/v2.0, n_features 14,
feature_include `^(feature_close_(avg|ewa)_|feature_market_cap_(avg|ewa)_|feature_bollinger_|feature_momentum_|feature_volatility_)`,
cv_scheme rolling, n_splits 2, train_window 130, refit_every 130, embargo 20, seed 42,
n_iter 50 (nominal). LightGBM booster PINNED via `fixed_params` (every searched bound has size 1, so `n_iter=50` is nominal): colsample_bytree 0.070213, learning_rate 0.002572, max_depth 8, n_estimators 4327, num_leaves 255, reg_lambda 1.757511, subsample 0.816153, subsample_freq 1.

## What was found
No written finding; from the metrics (510 periods, 20-day label): the concat arms edge
out everything. Best payout `concat_trio` 0.088635 (CORR 0.157138, MMC 0.072921),
then `concat_quintet` 0.085235 (CORR 0.154381). raw 0.083272 (CORR 0.149754).
The two rank-average blends are below raw and below concat: blend_trio 0.076659,
blend_quintet 0.067443. Single-group arms are all weaker than raw, the volatility group
strongest (0.067608) and liquidity weakest (0.009135). On these numbers the concatenated
union beats the fixed rank-average of per-group models for LightGBM; no paired t-stats
were computed in available artifacts.
