# E85 — combo + causal per-symbol means (1506f) vs combo (753f) + v2.0

- **State:** done (5/5 arms)
- **Runner:** experiments/e85_combo_fenccausal_crypto.sh (shared mommask_core.sh, VARIANT=fenc)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on `crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal` (1506f)
- **Artifacts on disk:** 26.4 MB across 5 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §E85 row (status "registered", no finding cell); memory: e82-e85-combo-fenccausal.md; metrics: results/metrics/crypto/catboost_ktz/e85_catboost_ktz_fenc.json

## Main question
Does a causal per-symbol expanding-mean twin of every kitchen-sink feature help over the kitchen sink? This applies the fenccausal/IA_01 idea (best MLP slot) to the whole 753f stack rather than just the raw 22.

## What was tried
`crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal` (1506f), a clean superset of the E82 combo (753f): appends a `_symmean` column for every combo feature (raw 22→44, vol 22→44, bei 12→24, tf 15→30, ix 682→1364). Each `_symmean` = mean of that symbol's own past values (a row never enters its own encoding), leak-safe. Biggest feature set run to date. 5 models from scratch at the shared protocol, reported 3-way vs combo (`e82_*_combo`) and v2.0 (`e72_*_base`), both reused.

## Trained parameters
From results/metrics/crypto/catboost_ktz/e85_catboost_ktz_fenc.json:
model catboost_ktz; data_version `crypto/v2.0-vol-infl-carver-ix-fillmask-fenccausal`; n_features 1506; cv_scheme rolling, inner_cv purged; train_window 130, refit_every 130, embargo 20, min_train 0, era_step 1; n_splits 5, val_frac 0.3; seed 42; n_iter 50; search_type random; metric corr; target target_binned_return_20; fixed_params none. catboost refit-0 best_params: depth 2, iterations 2000, l2_leaf_reg 10.0, learning_rate 0.005, min_data_in_leaf 100, rsm 0.2. Same protocol across all 5 arms; model class varied.

## What was found
MEMORY-BACKED (e82-e85-combo-fenccausal.md). **Every arm is BELOW its E82 combo twin:** lgb 0.129 < 0.140, rf 0.128 < 0.134, catboost 0.120 < 0.128, ridge 0.095 < 0.104, mlp 0.099 ≈ (CORR values). Doubling to 1506f drowned it further, including the mlp — fenccausal did not rescue it, buried among 1364 interaction-symmeans of noise. **No blend with the book helps either:** the E85 mlp/ridge arms ARE decorrelated from the submitted model (per-date rank-corr 0.43/0.47) but that is decorrelated noise, not signal (CORR ~0.10, too weak). Priced sub(cu_sa_ia_bo_02)+E85 rank-averages at weights 0.1–0.2 were all negative, none |t|>2, MMC never improved; best is 0.9 sub + 0.1 mlp payout −0.0014 (t −0.89). Contrast IA_01, which lifts CU_SA_IA_BO_01 because it is decorrelated AND strong (0.158). **The combo/fenccausal line is a dead end.** From the metrics (own 521-period windows, standalone): best lgb 0.03474 / CORR 0.12906, then rf 0.03463, catboost 0.03408, ridge 0.02621, mlp 0.02261. Scratch blends on disk: lgb/e85x_*.
