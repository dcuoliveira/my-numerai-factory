# E91 — cointegration block vs raw v2.0: crypto/v2.0-coint-fillmask (26f) vs v2.0 (22f)

- **State:** done (5/5 arms)
- **Runner:** experiments/e91_coint_vs_raw_crypto.sh (shared mommask_core.sh, VARIANT=coint)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on `crypto/v2.0-coint-fillmask` (26f)
- **Artifacts on disk:** 24.1 MB across 5 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md — no E91 row exists; memory: e90-e94-statarb-block-sweep.md; metrics: results/metrics/crypto/mlp_ktz/e91_mlp_ktz_coint.json

## Main question
Does a cointegration / stat-arb reversion block help the raw panel? Block-vs-raw test on a clean superset of raw.

## What was tried
`crypto/v2.0-coint-fillmask` (26f) = raw 22 + 4 `feature_coint_` features: residual z-score at 60d and 120d, half-life, and the ADF (Dickey-Fuller) statistic — a rolling-OLS cointegration residual block. fillmask = missing cells pre-filled to 0.5 (== the loader's own fill). 5 models from scratch at the shared protocol, reported 3-way vs raw v2.0 (`e72_*_base`) and vs the vol block (`e74_*_vol`), both reused.

## Trained parameters
From results/metrics/crypto/mlp_ktz/e91_mlp_ktz_coint.json: model mlp_ktz; data_version `crypto/v2.0-coint-fillmask`; n_features 26; cv_scheme rolling, inner_cv purged; train_window 130, refit_every 130, embargo 20, min_train 0, era_step 1; n_splits 5, val_frac 0.3; seed 42; n_iter 50; search_type random; metric corr; target target_binned_return_20; fixed_params none. mlp refit-0 best_params: activation identity, alpha 0.1, batch_size 1024, hidden_layer_sizes [32], learning_rate_init 0.00875. Same protocol across all 5 arms; model class varied. (Note: cointegration features have low coverage — they need long history — so `fillmask` matters here.)

## What was found
MEMORY-BACKED (e90-e94-statarb-block-sweep.md). **Least-bad / null — null on lgb/mlp, mild negative on ridge (t −3.2), catboost and rf. No help.** Consistent with the standing program conclusion that recombining the price-derived 22 does not beat raw v2.0. From the metrics (own 523-period windows, standalone): best lgb_coint 0.07419 / CORR 0.14806 / MMC 0.05939, then catboost 0.07076, rf 0.07028, ridge 0.07009, mlp 0.06512.
