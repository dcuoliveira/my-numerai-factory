# E120 — Training window of the best standalone LightGBM (concat_trio), 1 week to 6 months

- **State:** done (7/7 arms)
- **Runner:** `experiments/e120_train_window_lgb_crypto.sh`
- **Models / datasets:** lgb (pinned booster, `--fixed-params`) on the group-CONCAT subset of `crypto/v2.0-onehot` (market-cap + momentum + volatility + per-symbol one-hot, 748 feat); reproduces `lgb/e03_concat_trio`
- **Artifacts on disk:** 29.18 MB across 7 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §(table row, line 111 — result written in the status cell with full numbers); memory: none; metrics: `results/metrics/crypto/lgb/e120_w130.json`

## Main question
The LightGBM twin of E117/E119: how much history should each refit use? Uses the payout-best standalone LightGBM `e03_concat_trio` (the canonical pinned booster), varying only the rolling window (5/10/15/20/22/65/130 periods). Because the booster is PINNED there is no search and no `--tune-window` — this isolates "how much history," not freshness (E116) and not model capacity.

## What was tried
Seven window arms (w5…w130), the model pinned via `--fixed-params` (n_iter=1), cleanly paired NW lag 20 vs the w130 control. Held constant: data group-CONCAT subset of `crypto/v2.0-onehot` (748 feat), refit cadence 130, embargo 20, seed 42, target `target_binned_return_20`, era_step 1, cv rolling/purged, tree count 4327. w130 reproduces `e03_concat_trio`. 4 refits per arm; n=510 scored periods.

## Trained parameters
From `results/metrics/crypto/lgb/e120_w130.json`: model `lgb`, data_version `crypto/v2.0-onehot`, n_features 748, n_iter 1 (pinned), refit_every 130, embargo 20, train_window 130, seed 42, target `target_binned_return_20`. fixed_params (identical across all 4 refits, by design pinned): `colsample_bytree` 0.070213, `learning_rate` 0.002572, `max_depth` 8, `min_child_samples` 50, `n_estimators` 4327, `num_leaves` 255, `reg_lambda` 1.757511, `subsample` 0.816153, `subsample_freq` 1. Only `train_window` swept across the other six arms.

## What was found
FINDINGS-BACKED. EXPERIMENTS.md states: "The deployed 6-month window is clearly best — more history is strictly better, monotonic, every shorter window significantly worse" (NW lag 20 vs w130):
- w5 payout 0.03223 (t −3.46)
- w10 0.02625 (t −3.98)
- w15 0.02945 (t −4.01)
- w20 0.03644 (t −3.62)
- w22 0.04018 (t −3.45)
- w65 0.06244 (t −2.91)
- **w130 0.09074 (control)**

Payout climbs monotonically with window; p.sharpe 0.27→1.24; dd(pay) ~8→~1. A pinned 4327-tree booster on a week–month of data overfits badly (sharper than E117's ridge, being higher-capacity). w130 reproduces `e03_concat_trio` to 0.00e+00. Payout is still rising at the 130 ceiling → a >130 (e.g. 260 / 1yr) arm is the natural follow-up. (The metrics JSON summary-rebuilt payouts — w130 CORR 0.143209 / MMC 0.076421 / 0.1·c+m 0.090742 — agree with the written 0.09074 to ranking precision.) Ran 7/7 in ~10 min.
