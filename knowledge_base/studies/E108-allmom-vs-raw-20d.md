# E108 — all momentum/trend families in one dataset vs raw: v2.0-allmom-fillmask (57f) vs v2.0

- **State:** done (5/5 arms)
- **Runner:** experiments/e108_allmom_vs_raw_crypto.sh (reuses mommask_core.sh, VARIANT=allmom)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on crypto/v2.0-allmom-fillmask (57f), compared vs raw v2.0 (22f, e72_*_base)
- **Artifacts on disk:** 26.4 MB across 5 forecast runs (plus 5 metrics + 5 hyperparams + 5 per_period) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none; memory: e108-e111-allmom-pipeline.md, e112-allmom-encodings.md; metrics: results/metrics/crypto/mlp_ktz/e108_mlp_ktz_allmom.json, results/metrics/crypto/lgb_ktz/e108_lgb_ktz_allmom.json

## Main question
Does pooling EVERY momentum/trend family into one dataset beat the raw 22 features? Prior: momentum was null-to-negative vs raw (E64–E67), but a tree given all variants at once might pick among them — against the expectation that the raw 22 already absorb most of it.

## What was tried
crypto/v2.0-allmom-fillmask (57f), assembled with build_combo (leak-safe union of already-binned blocks onto the raw v2.0 spine, no recomputation): raw 22 + feature_mom_ 17 (litmom: cumret horizons, 12-1, MAX, vol-scaled, Tzotchev robust trend) + feature_xsmom_ 3 (cross-sectional momentum mean/var/skew, day-level) + feature_tf_ 15 (Carver/Clenow trend: EWMAC, breakout, Bollinger-z, Clenow). Block coverage: mom 64.9%, xsmom 94.6%, tf 78.7% (rest neutral-filled). 5 models from scratch at the shared protocol, n_iter=50, reported 3-way (CORR/MMC/payout/c.sharpe/calmar/n) vs raw v2.0 and vs the vol block (e74_*_vol).

## Trained parameters
From results/metrics/crypto/lgb_ktz/e108_lgb_ktz_allmom.json config: model lgb_ktz; data_version crypto/v2.0-allmom-fillmask; cv_scheme rolling; train_window 130; refit_every 130; embargo 20; era_step 1; inner_cv purged; n_splits 5; val_frac 0.3; n_iter 50; search_type random; metric corr; target target_binned_return_20; task regression; n_features 57; seed 42; fixed_params null; search_jobs 4. The mlp_ktz twin (e108_mlp_ktz_allmom.json) is identical except search_jobs 8. CORR.mean: lgb 0.14708 (MMC 0.05683), mlp 0.11715 (MMC 0.04211), both n=526.

## What was found
MEMORY-BACKED. No written EXPERIMENTS.md finding; the memory records allmom as feeding the E111 pipeline and states (via E111's 20d validation) that "allmom still trails raw after residualisation (ridge −0.018 t−3.3, mlp −0.016 t−2.7)" and "allmom < raw" overall. The E112 memory confirms E108 base arms were reused as the comparison base (e.g. allmom lgb 0.07337 payout on 435 days; book 0.10468). Direction: allmom is null-to-negative vs raw, matching the E64–E67 prior — the raw 22 already absorb the momentum signal. Per-arm paired deltas vs raw are not in a dedicated EXPERIMENTS.md section; the numbers quoted here are the metric means from the two JSONs opened plus the memory's E111-stage comparison.
