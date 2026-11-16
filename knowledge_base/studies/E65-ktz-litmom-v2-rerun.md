# E65 — E64 re-run on the updated litmom (22+17) at uniform n_iter=50

- **State:** done (10/10 arms + 10 scouts = 20 runs)
- **Runner:** experiments/e65_ktz_litmom_v2_bakeoff_crypto.sh
- **Models / datasets:** models ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz; base crypto/v2.0 (22f) vs litmom crypto/v2.0-litmom (now 22+17 = 39f)
- **Artifacts on disk:** 57.12 MB across 20 forecast runs (10 full + 10 scout) (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §none; memory: e65-litmom-v2-rerun.md, e79-fillmask-equals-loader.md, e73-litmom-mommask.md (+ MEMORY.md index line for the final verdict); metrics: results/metrics/crypto/rf_ktz/e65_rf_ktz_litmom.json (+ full 10-arm board summarised)

## Main question
Re-run of E64 after litmom grew from 22+10 (32f) to 22+17 (39f, adding the 7-feature Tzotchev robust-trend block) and the search budget was made uniform at n_iter=50. Does the added trend block move any model's verdict — does litmom now beat matched base on numerai_corr at paired |t|>=2?

## What was tried
10 arms = {ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz} x {base v2.0 (22f), litmom v2.0-litmom (39f)}, base RE-RUN at 50 so the within-model pairing is exact. Same cadence as E64 (rolling 130/130, embargo auto=20, 5-fold, val_frac 0.3, seed 42, target_binned_return_20). Uniform n_iter=50 for every model (catboost_ktz's 12-cell grid clamps 50->12, a full sweep). The 7 added features: feature_mom_trend_{7,14,30,60,120,240}d + feature_mom_trend_robust.

## Trained parameters
From the metrics JSONs: cv_scheme rolling, embargo 20, train_window 130, refit_every 130, n_splits 5, seed 42, n_iter 50 (uniform), target target_binned_return_20; litmom arms n_features 39, base arms 22. Representative best-payout arm e65_rf_ktz_litmom, best_params (refit 0): {bootstrap True, max_depth 3, max_features 0.1, min_samples_leaf 300, n_estimators 100}.

## What was found
MEMORY-BACKED: still null-to-negative — the added trend block did not rescue litmom. Per the MEMORY.md index line, E65 is "DONE, still null-to-negative (lgb_ktz now HURTS MMC/payout at |t|>2)".

From the metrics board read here (CORR / MMC, n=520, litmom 39f vs base 22f):
- ridge:        CORR 0.13654 vs 0.14423 (worse); MMC 0.05538 vs 0.05919
- lgb_ktz:      CORR 0.14555 vs 0.14771 (worse); MMC 0.05265 vs 0.05952 (litmom MMC lower by ~0.0069 — the "HURTS MMC" case)
- catboost_ktz: CORR 0.14840 vs 0.15113 (worse); MMC 0.05613 vs 0.05926
- rf_ktz:       CORR 0.15483 vs 0.14962 (better); MMC 0.05894 vs 0.06230
- mlp_ktz:      CORR 0.13246 vs 0.14259 (worse); MMC 0.05623 vs 0.05931

Four of five models are flat-to-worse on CORR with litmom; only rf_ktz is higher on CORR (but lower on MMC). Direction confirms E63/E64: the literature-momentum block adds no robust signal over the raw 22, consistent with it being redundant with the base's own momentum proxies (see E66/E67) and with the EDA's ~0.015 incremental-IC ceiling.
