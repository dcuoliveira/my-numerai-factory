# E80 — inflation-expectations (breakeven) macro block vs v2.0 + momswap-macro

- **State:** done (5/5 arms)
- **Runner:** experiments/e80_infl_momswap_macro_crypto.sh (shared mommask_core.sh, VARIANT=infl)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on `crypto/v2.0-momswap-macro-infl` (61f)
- **Artifacts on disk:** 26.4 MB across 5 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §E80 row (status "registered", no finding cell); memory: e80-infl-breakeven-macro.md, inflation-breakeven-block.md, MEMORY.md index line; metrics: results/metrics/crypto/catboost_ktz/e80_catboost_ktz_infl.json

## Main question
Do market-implied inflation-expectations (break-even inflation, BEI) features help the panel? Direct sibling of E69 (macro). The 12 BEI features are day-level/broadcast (one macro value per date), so they are inert for ridge on a within-date rank metric (negative control) and can help only via tree/MLP interactions.

## What was tried
`crypto/v2.0-momswap-macro-infl` (61f) = momswap-macro (49f) + 12 BEI features: market-implied inflation expectations at 5y/10y/30y/5y5y-forward plus their 20d and 60d changes (`feature_macro_bei_*`), clean superset of momswap-macro. 5 models from scratch at the shared protocol, reported 3-way vs raw v2.0 (`e72_*_base`) and momswap-macro (`e70_*_base`, to isolate the 12 BEI features' marginal value), both reused. BEI sources (memory): US Treasury nominal par yields minus TIPS real yield curve; normalisation/attach = causal trailing percentile, per-column ffill, date-level broadcast, D-1 point-in-time.

## Trained parameters
From results/metrics/crypto/catboost_ktz/e80_catboost_ktz_infl.json:
model catboost_ktz; data_version `crypto/v2.0-momswap-macro-infl`; n_features 61; cv_scheme rolling, inner_cv purged; train_window 130, refit_every 130, embargo 20, min_train 0, era_step 1; n_splits 5, val_frac 0.3; seed 42; n_iter 50; search_type random; metric corr; target target_binned_return_20; fixed_params none. catboost refit-0 best_params: depth 2, iterations 2000, l2_leaf_reg 10.0, learning_rate 0.005, min_data_in_leaf 100, rsm 0.2. Same protocol across all 5 arms; model class varied.

## What was found
MEMORY-BACKED. The MEMORY.md index records E80 as "DONE: null (inflation adds nothing; E81 = same 12 BEI on raw, 34f)." The e80 memory file's prior (and the E69 macro sibling) is null-or-marginal: ~zero standalone within-era IC. Paired deltas vs the baselines are not recorded in the available artifacts. From the metrics (own ~520-period windows, standalone): the 5 arms cluster tightly — rf 0.07029 / CORR 0.14703, lgb 0.06698, ridge 0.07062, mlp 0.07167, catboost 0.06660 — consistent with "adds nothing" over the momswap-macro baseline.
