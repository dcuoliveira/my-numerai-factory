# E03 — Group blend vs concat at leaf 50, with the one-hot block

- **State:** done (12/12 arms)
- **Runner:** experiments/e03_group_blends_onehot_lgb_reg_crypto.sh
- **Models / datasets:** lgb on crypto/v2.0-onehot (adds the 740 ticker one-hot dummies); groups by `feature_include` regex
- **Artifacts on disk:** 46.2 MB across 12 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/lgb/e03_concat_quintet.json
- Note: `payout` = 0.1*CORR_mean + MMC_mean from stored summaries.

## Main question
E02 repeated at a LightGBM leaf size (min_child_samples 50) small enough that the model
can actually split on a one-hot ticker column — at leaf 313 it discards all 740 before
training. Same blend-vs-concat question, now with the one-hot block in play.

## What was tried
Same 12-arm shape as E02 (per-group fits, concat_trio/quintet, blend_trio/quintet, raw)
but on crypto/v2.0-onehot and at min_child_samples 50. The group regex now also admits
`feature_sym_` (the one-hot dummies); concat_quintet has n_features 756. 510 periods,
20-day label. Blend arms carry target None.

## Trained parameters
From results/metrics/crypto/lgb/e03_concat_quintet.json: model lgb, regression,
target_binned_return_20, data_version crypto/v2.0-onehot, n_features 756,
feature_include `(^feature_close_(avg|ewa)_)|(^feature_market_cap_(avg|ewa)_)|(^feature_(relative_strength_index|sharpe_ratio)_)|(^feature_momentum_)|(^feature_volatility_)|(^feature_sym_)`,
cv_scheme rolling, n_splits 2, train_window 130, refit_every 130, embargo 20, seed 42,
n_iter 50 (nominal). Booster pinned as E01 but with **min_child_samples 50**.

## What was found
No written finding; from the metrics (510 periods): concat and raw are essentially tied
and ahead of the blends. Best payout `concat_trio` 0.090742 (CORR 0.143209, MMC 0.076421),
`concat_quintet` 0.089960 (CORR 0.147887), raw 0.088361 (CORR 0.148333). Rank-average
blends trail: blend_trio 0.082662, blend_quintet 0.076915. Per-group arms are stronger
here than in E02 (volatility 0.083207, size 0.075615) but still below raw. Dropping the
leaf to 50 and adding one-hots did not make the group-blend beat the concatenated union
or raw; no paired t-stats in available artifacts.
