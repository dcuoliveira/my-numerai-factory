# E46 — FARM over the from-scratch ktz SEARCH-PICK models (lgb E29 + catboost E32-sel + mlp E34), per-family and pooled, vs 1/N

- **State:** done (50/48 arms)
- **Runner:** experiments/e46_ktz_searchpick_farm_crypto.sh
- **Models / datasets:** model `ktz_farm` (Lee-Lee FARM combiner over finished forecasts); members are lgb_ktz/e29_*, catboost_ktz/e32_sel_*, mlp_ktz2/e34_* over 9 book datasets (27 base forecasters)
- **Artifacts on disk:** 249.4 MB across 50 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** NUMBERS-ONLY
- **Sources read:** EXPERIMENTS.md §none; memory: none; metrics: results/metrics/crypto/ktz_farm/e46_all_equal.json, results/metrics/crypto/ktz_farm/e46_all_farm1_ridge.json

## Main question
Does a fitted factor-adjusted combination (Lee-Lee FARM) beat the plain 1/N rank-average of the same 27 best-tuned (model x dataset) forecasters? Prior from E27/E29/E36: 1/N wins on this panel, so a fitted combiner clearing |t|>=2 would be the surprise. Raw scores, ratios not NMR.

## What was tried
Pool = 27 base forecasters (the config each family's from-scratch KTZ search PICKED per dataset): lgb lgb_ktz/e29_<ds> (60-draw search), catboost catboost_ktz/e32_sel_<ds> (exhaustive 12-cell), mlp mlp_ktz2/e34_<ds> (80-draw), over the 9 book datasets (raw, tenc, interact, fenccausal, cumlag3, tenc_onehot, onehot_rollsel, allfeat, allfeatpp). Combiners: equal (the 1/N baseline each arm is measured against), farm1/farm2/farm3 x {ridge, l2boost, postalasso}, gr, ds — over four pools: per-family (lgb 9, cat 9, mlp 9) and the pooled union (all 27). Paired NW lag 20 vs each pool's own 1/N; each pool's best arm also ranked against the deployed book. 50 runs for the ~48 planned arms.

## Trained parameters
`ktz_farm` combiner configs (not model fits). From results/metrics/crypto/ktz_farm/e46_all_farm1_ridge.json: method farm1, beta_method ridge, update_every 22, lag 20, min_history 60, roll 252, members = the 27-forecaster pooled union (9 lgb_ktz/e29_* + 9 catboost_ktz/e32_sel_* + 9 mlp_ktz2/e34_*). e46_all_equal.json is the pooled 1/N baseline (method equal) over the same 27 members. The underlying members' own hyperparameters live in their source studies (E29/E32/E34), not in these FARM metrics JSONs.

## What was found
No written finding; from the two pooled arms opened, 1/N beats the FARM combiner on both CORR and MMC, consistent with the E27/E29/E36 prior. e46_all_equal (pooled 1/N): CORR mean 0.165213 (sharpe 1.749487), MMC mean 0.074538, n = 497. e46_all_farm1_ridge (pooled fitted FARM): CORR mean 0.153894 (sharpe 1.755850), MMC mean 0.068270, n = 497 — lower on both the paid metrics. Only 2 of 50 arms were opened (the pooled 1/N baseline and one pooled FARM arm); a full per-pool ranking and the paired |t| vs 1/N are not recorded in available artifacts, but the examined arms match the prior that 1/N wins.
