# E64 — KTZ-prior model bake-off: litmom vs base across 5 model classes

- **State:** done (10/10 arms + 10 scouts = 20 runs)
- **Runner:** experiments/e64_ktz_litmom_bakeoff_crypto.sh
- **Models / datasets:** models ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz; base crypto/v2.0 (22f) vs litmom crypto/v2.0-litmom (here the OLD 22+10 = 32f)
- **Artifacts on disk:** 56.40 MB across 20 forecast runs (10 full + 10 scout) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none; memory: e64-ktz-litmom-bakeoff.md, litmom-momentum-feature-pipeline.md, e65-litmom-v2-rerun.md, e28-e29-prefix-collision.md, e67-old-vs-new-momentum.md; metrics: results/metrics/crypto/lgb_ktz/e64_lgb_ktz_litmom.json (+ full 10-arm board summarised)

## Main question
Does the literature-momentum block help ACROSS function classes, not just the two E63 tried? A clean finding would be CONSISTENCY: litmom beating matched base on numerai_corr at paired Newey-West |t|>=2 (lag 20) across a linear model, two boosters, a forest and a net.

## What was tried
10 arms = {ridge (linear control, no ktz variant), lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz} x {base v2.0 22f, litmom v2.0-litmom 32f}, each tuned FROM SCRATCH under its KTZ priors, paired WITHIN model. Cadence rolling 130/130, embargo auto=20, 5-fold, val_frac 0.3, seed 42, target_binned_return_20, metric corr. Per-model n_iter {ridge 50, lgb_ktz 50, catboost_ktz 12, rf_ktz 40, mlp_ktz 80}, identical across a model's two arms. NOTE (memory): E64 ran on the OLD 22+10 litmom — the parquet was later rebuilt to 22+17 and re-run as E65.

## Trained parameters
From the metrics JSONs: cv_scheme rolling, embargo 20, train_window 130, refit_every 130, n_splits 5, seed 42, target target_binned_return_20; litmom arms n_features 32, base arms 22. Representative best-payout arm e64_lgb_ktz_litmom (n_iter 50), best_params (refit 0): {colsample_bytree 0.11, learning_rate 0.00123, max_depth -1, min_child_samples 100, n_estimators 4000, num_leaves 2, reg_lambda 10.0, subsample 0.8, subsample_freq 1}.

## What was found
MEMORY-BACKED (e64-ktz-litmom-bakeoff.md / e65-litmom-v2-rerun.md): a NULL across all five classes at paired |t|<2. Recorded paired deltas (litmom − base): ridge ΔCORR −0.0052 (t −1.66), lgb_ktz +0.0064 (t +1.62), catboost −0.0015 (t −0.66), rf_ktz +0.0025 (t +0.89), mlp pending at report time. Prior expectation (E63 EDA) was a small incremental IC ~0.015, so a near-null is itself the finding — adding the 10-feature momentum block does not clear the bar for any function class.

Consistent with the metrics board read here (CORR, n=520, litmom vs base): ridge 0.13903 vs 0.14423 (worse), lgb_ktz 0.15415 vs 0.14771 (better), catboost_ktz 0.14965 vs 0.15113 (~tie), rf_ktz 0.15217 vs 0.14962 (slightly better), mlp_ktz 0.12570 vs 0.14276 (worse) — mixed signs, no consistent direction. E65 re-ran this on the expanded 22+17 litmom.
