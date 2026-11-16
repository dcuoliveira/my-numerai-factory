# E118 — 34 causal model-representation blocks, one at a time, on the best ridge / LightGBM / CatBoost

- **State:** not started (0 arms; scout 0/105)
- **Runner:** `experiments/e118_fmx_crypto.sh`
- **Models / datasets:** ridge_glmnet (`w6r6_lobo_tenc`, allfeat_pp − tenc, 811f), lgb_ktz (`e29_tenc_onehot`, tenc_onehot, 763f), catboost_ktz (`e109_catboost_ktz_fenc`, fenccausal, 44f); representation blocks `feature_xNN_` added one at a time
- **Artifacts on disk:** 0.0 MB across 0 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** THIN
- **Sources read:** EXPERIMENTS.md §(table row, line 109 — no dedicated section); memory: none; metrics: none on disk

## Main question
Does adding any single causal, live-computable "model-representation" block (derived from the book's own models or from TabPFN/foundation models) ALONE improve the payout-best clean standalone run of three model families (ridge / LightGBM / CatBoost)? Each block is paired against the base re-run on its own `-fmx` dataset (the in-study "without").

## What was tried
Planned as 3 families × 34 blocks = 102 block arms + 3 `x00` base re-runs = 105 arms on the 1,701-date spine, 510 scored periods. Blocks `feature_xNN_`: TabPFN on fenccausal (x01 point, x02 distribution + bin probs, x03 estimator disagreement, x04 embedding PCs, x05 per-block heads + contexts, x07 attention familiarity, x08 rank uncertainty, x09 drift, x10 embedding neighbours, x11 residual learner, x12 crowd/meta-model label, x13 60-day term structure, x14 extremeness, x15 feature nowcasts, x16 self-context, x17 BTC-regime context, x18 pairwise ranking, x19 local drivers, x20 evidence age); x06 TimesFM/Chronos-2 on spot; x22–x35 the book's RF/CatBoost/LightGBM/MLP/block-ridge refit walk-forward from 2023-05 (predictions, tree/stage/seed spread, leaf PCs/support, SHAP, hidden PCs, member spread). Every block declared causal and live-computable (TabPFN context ends 20 dates — 60 for x13 — before each 20-date refresh block; source models searched + refit every 130 in their own window, embargo 20; spot as-of D−1; reductions fitted on past rows only; each column rank-binned per date). Builders `src/data_mani/{fmx_tabpfn,fmx_models,build_fmx}.py`. Each block paired NW lag 20 vs its `x00`.

## Trained parameters
config JSON not on disk; no runs exist. Intended baselines (from the idea): ridge_glmnet 811f payout 0.08779, lgb_ktz 763f payout 0.08999, catboost_ktz 44f payout 0.09303. No fitted parameters recorded in available artifacts.

## What was found
THIN. The study is NOT started — EXPERIMENTS.md marks it "running" / "105 expected" but the manifest and metrics directory hold 0 runs (scout 0/105), so no arm has produced a forecast, metric, or finding. Nothing is reclaimable (0 bytes) and no result is recorded in available artifacts.
