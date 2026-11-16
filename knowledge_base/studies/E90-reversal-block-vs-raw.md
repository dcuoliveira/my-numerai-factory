# E90 — reversal block vs raw v2.0: crypto/v2.0-rev (36f) vs v2.0 (22f)

- **State:** done (5/5 arms)
- **Runner:** experiments/e90_rev_vs_raw_crypto.sh (shared mommask_core.sh, VARIANT=rev)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on `crypto/v2.0-rev` (36f)
- **Artifacts on disk:** 26.6 MB across 5 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md — no E90 row exists; memory: e90-e94-statarb-block-sweep.md, report-calmar-convention.md (reporting convention only); metrics: results/metrics/crypto/mlp_ktz/e90_mlp_ktz_rev.json

## Main question
Does a reversal / mean-reversion block help the raw panel? Classic block-vs-raw test on a clean superset of raw.

## What was tried
`crypto/v2.0-rev` (36f) = raw 22 + 14 `feature_rev_` features: short-term reversal (str_1/7/14d), MAX/lottery effect (max1/max5_28d), idiosyncratic vol (ivol_28d), return skew (skew_28d), Amihud illiquidity (amihud_28d), and mean-reversion oscillators (rsi_14d, bbz_20d, stoch_14d, cci_20d, dist_ma_20/50d). All per-symbol (cross-sectional, scoreable). 5 models from scratch at the shared protocol, reported 3-way vs raw v2.0 (`e72_*_base`) and vs the vol block (`e74_*_vol`), both reused.

## Trained parameters
From results/metrics/crypto/mlp_ktz/e90_mlp_ktz_rev.json: model mlp_ktz; data_version `crypto/v2.0-rev`; n_features 36; cv_scheme rolling, inner_cv purged; train_window 130, refit_every 130, embargo 20, min_train 0, era_step 1; n_splits 5, val_frac 0.3; seed 42; n_iter 50; search_type random; metric corr; target target_binned_return_20; fixed_params none. mlp refit-0 best_params: activation identity, alpha 0.01, batch_size 1024, hidden_layer_sizes [32], learning_rate_init 0.0375. Same protocol across all 5 arms; model class varied.

## What was found
MEMORY-BACKED (e90-e94-statarb-block-sweep.md). **Mixed-negative — hurts ridge (t −3.0) and catboost (t −4.1) vs raw, null on lgb/rf/mlp.** The reversal block overlaps raw's own RSI/Bollinger, so it adds no new signal and dilutes the linear/catboost fits. Standing program conclusion: recombining the price-derived 22 (momentum/vol/carver/interactions/reversal/coint/beta/BAB) does not beat raw v2.0. From the metrics (own 523-period windows, standalone): best rf_rev 0.07084 / CORR 0.14876, then lgb 0.06919, catboost 0.06734, ridge 0.06448, mlp 0.06376.
