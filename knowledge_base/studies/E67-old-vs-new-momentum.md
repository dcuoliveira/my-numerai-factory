# E67 — Old momentum proxies vs new literature momentum, momentum-only, from scratch

- **State:** done (16/10 arms — 10 model arms + ensemble/blend arms)
- **Runner:** `experiments/e67_old_vs_new_momentum_crypto.sh`
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz × {old = 8 base momentum proxies (8f), new = 17 `feature_mom_*` (17f)}, both selected from `crypto/v2.0-litmom`
- **Artifacts on disk:** 68.2 MB across 26 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** FINDINGS-BACKED
- **Sources read:** EXPERIMENTS.md §none dedicated — planning-table row (line 73) carries a DONE finding note; memory: e67-old-vs-new-momentum.md; metrics: results/metrics/crypto/mlp_ktz/e67_mlp_ktz_new.json (+ all full arms and blends)

## Main question
The clean head-to-head that E64–E66 kept confounded by the rest of the panel: race the two momentum blocks ALONE, each tuned from scratch with no other features. `old` = the 8 base momentum proxies (`feature_{momentum,relative_strength_index,sharpe_ratio,bollinger}_{20,60}d`); `new` = the 17 literature momentum features (`feature_mom_*`). Does the literature block beat the crude proxies at paired |t|≥2?

## What was tried
10 model arms = 5 models × {old (8f, `--feature-include '^feature_(momentum|relative_strength_index|sharpe_ratio|bollinger)_[0-9]+d$'`), new (17f, `--feature-include '^feature_mom_'`)}, both selected from `crypto/v2.0-litmom` so arms share rows/dates/seed and are verified disjoint. Levels are deliberately toy (8–17 features) and not comparable to the deployed book — read the paired delta (new − old). Plus 6 extra arms: an ensemble `e67_new_ens` of the new-momentum models, and `e67_bl_cusa_*` blends of the book slot `cu_sa_ia_bo` with the momentum models at various weights.

## Trained parameters
From the metrics JSONs: `cv_scheme=rolling`, `train_window=130`, `refit_every=130`, `n_splits=5`, `embargo=20`, `val_frac=0.3`, `metric=corr`, `search_type=random`, `n_iter=50`, `seed=42`, `target=target_binned_return_20`. `fixed_params` = null (ktz priors in the search grid). `n_features`: old 8, new 17; model arms n=520; blend arms n≈449. The blend arms (`e67_bl_*`, `e67_new_ens`) carry `data_version=null` (ensembles write no dataset into their config).

## What was found
FINDINGS-BACKED (EXPERIMENTS.md table row: "DONE; ensemble `e67_new_ens` built + priced vs `cu_sa_ia_bo_02` (momentum-only tops ~0.10 CORR / 0.04 payout, ~⅓ of the deployed blend; no blend beats it)"; MEMORY index: "ensemble `e67_new_ens` built (underperforms best member; ~1/3 of cu_sa_ia_bo_02)"). The metrics corroborate: NEW beats OLD on all 5 models, decisively. ΔCORR (new − old): ridge +0.04702, lgb +0.02767, catboost +0.03217, rf +0.03484, mlp +0.03996. Δpayout: ridge +0.02708, lgb +0.00365, catboost +0.01075, rf +0.01157, mlp +0.02125. Best momentum-only arm: `catboost_ktz_new` (CORR 0.102365 ≈ the "~0.10" in the finding; payout 0.042467), then `rf_ktz_new` (payout 0.039715). The new-momentum ensemble `e67_new_ens` (CORR 0.091971, payout 0.038554) does NOT beat its best member (catboost_ktz_new payout 0.042467). Book+momentum blends reach payout ~0.100 (`e67_bl_cusa_cat_82` 0.100350) but that is the book carrying the result. Takeaway: the literature momentum block is a clearly better momentum representation than the base proxies, but momentum-only tops out at ~⅓ of the deployed blend's payout and no blend beats the book. Paired Newey-West t-stats are not recorded in the artifacts.
