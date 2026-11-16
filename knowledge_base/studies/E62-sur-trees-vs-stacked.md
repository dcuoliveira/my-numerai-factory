# E62 — SUR cross-symbol factor-GLS on trees (lgb/rf/catboost) vs stacked

- **State:** partial (55/108 arms) — deepsix rebooted uncleanly on 2026-09-26; the 48 wide-dataset arms + 5 sur_catboost classifier arms never produced a result
- **Runner:** experiments/e62_sur_trees_crypto.sh
- **Models / datasets:** models sur_lgb/sur_rf/sur_catboost vs lgb_ktz/rf_ktz/catboost_ktz(reg)/catboost(clf); 25 complete pairs on the 5 narrow datasets (raw, tenc, interact, fenccausal, cumlag3)
- **Artifacts on disk:** 242.78 MB across 55 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §"E62 — factor-augmented trees do not beat their plain twins; the one apparent edge is search budget"; memory: e62-sur-trees.md; metrics: results/metrics/crypto/sur_catboost/e62_sur_catboost_reg_fenccausal.json (+ 55-arm board summarised)

## Main question
Does E61's cross-symbol factor idea, which failed for the ridge, help the tree families? A result is SUR beating the plain-tree twin on numerai_corr at paired |t|>=2; prior is the E61 null (factor picked in CV, dies out-of-sample).

## What was tried
108 planned arms = 3 families x 2 tasks x 9 datasets x {sur, stacked}. A tree has no coefficients to partial, so SurAugment adds the k common cross-symbol factor scores as EXTRA FEATURES at fit and predicts with them ZEROED (future factor scores are unknowable). n_factors=0 is an exact plain-tree control; factors come from a cheap stage-1 ridge's residuals. Matched protocol: rolling 130/130, embargo 20, 5 folds, 50 draws, seed 42. Symbol+date routed via WANTS_SYMBOL/WANTS_PERIOD. Only 55 arms finished (25 complete SUR-vs-stacked pairs on the 5 narrow datasets); all 5 sur_catboost classifiers crashed at refit 0 (sklearn `clone` rejects CatBoost when l2_leaf_reg is np.float64).

## Trained parameters
From the metrics JSON: cv_scheme rolling, embargo 20, train_window 130, refit_every 22, n_splits 5, seed 42, n_iter 50, target target_binned_return_20. Representative best-payout arm e62_sur_catboost_reg_tenc; best_params (refit 0 of e62_sur_catboost_reg_fenccausal): {estimator__depth 2, estimator__iterations 2000, estimator__l2_leaf_reg 10.0, estimator__learning_rate 0.005, estimator__min_data_in_leaf 100, estimator__rsm 0.1, n_factors 2}.

## What was found
FINDINGS-BACKED (EXPERIMENTS.md §E62): factor-augmented trees do NOT beat their plain twins; the one apparent edge is a search-budget artifact.
- No pair clears the Bonferroni bar (|t| >= 3.1 over 25 pairs). The largest of all 75 tests is CORR t +2.41 (catboost reg interact); the worst is lgb reg cumlag3 at payout t -2.21 (mmc t -2.23). lgb and rf lose payout in 15 of 20 pairs, every positive among them |t| <= 0.60.
- SUR arms are near-duplicates of their twins: within-date prediction correlation 0.93-0.96 (reg), 0.80-0.95 (clf). CV keeps k>0 in ~90% of refits — the E61 pattern.
- The CatBoost "edge" (sur_catboost reg looked better on all 5 datasets) is SEARCH BUDGET, not factors: catboost_ktz's grid has only 12 cells so the control's 50-draw search clamps to 12, while SUR searches 50 of the 60 cells the n_factors axis adds. With the base cell held fixed (58 refits), factors give CORR t +0.87, payout t -0.11. Pooled CORR t ~1.9, pooled payout t 1.09 (5 datasets ~ 2 independent looks, so 5/5 sign has null prob ~0.18).
- Against the book, the best arm (sur_catboost reg tenc, payout 0.072 on 449 periods) is far below the book's slot blends (0.106-0.117).

Verdict: modelling the cross-symbol error factor does nothing for trees, as for the ridge (E61). Do NOT resume E62; if ever revisited, fix the CatBoost float cast and match the candidate count. Report artifacts: results/reports/crypto/e62_sur_trees_{reg,clf}.html.
