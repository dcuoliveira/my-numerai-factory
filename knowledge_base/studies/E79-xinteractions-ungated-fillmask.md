# E79 — xinteractions un-gated (fillmask == raw 219f), 3-way vs mommask + v2.0

- **State:** done (5/5 arms)
- **Runner:** experiments/e79_xinteractions_fillmask_crypto.sh (shared mommask_core.sh, VARIANT=fillmask)
- **Models / datasets:** ridge, lgb_ktz, catboost_ktz, rf_ktz, mlp_ktz on `crypto/v2.0-momswap-macro-xinteractions-fillmask` (219f)
- **Artifacts on disk:** 26.7 MB across 5 forecast runs (reclaimability is NOT decided here — see DELETION_LEDGER.md, computed by src.analysis.retention)
- **Provenance:** MEMORY-BACKED
- **Sources read:** EXPERIMENTS.md §E79 row (status "registered", no finding cell); memory: e79-fillmask-equals-loader.md; metrics: results/metrics/crypto/lgb_ktz/e79_lgb_ktz_fillmask.json

## Main question
Does cell-level fill (`build_mommask --mode fill`, replace only NaN cells with 0.5) beat whole-row gating (mommask) on the 219-feature macro-momentum interactions block, and does the un-gated 219f block beat raw v2.0? This run also supplies the proper un-gated 219f baseline, which did not exist because the dataset was rebuilt 73f→219f after E71 ran at 73f.

## What was tried
5 model classes tuned from scratch on `crypto/v2.0-momswap-macro-xinteractions-fillmask` (219f), reported as a clean 3-way: v2.0 (`e72_*_base`), mommask (`e76_*_mommask`, gated), fillmask (this run, un-gated). A key structural fact (memory): a `-fillmask` dataset, once loaded, is byte-identical to its raw parent, because the loader (`fill_missing_features`, neutral 0.5) already does the same fillna — so 4/5 fillmask datasets reproduced earlier runs exactly and were not re-run; only the 219f xinteractions fillmask was genuinely new.

## Trained parameters
From results/metrics/crypto/catboost_ktz/e79_catboost_ktz_fillmask.json (and the lgb winner):
model catboost_ktz; data_version `crypto/v2.0-momswap-macro-xinteractions-fillmask`; n_features 219; cv_scheme rolling, inner_cv purged; train_window 130, refit_every 130, embargo 20, min_train 0, era_step 1; n_splits 5, val_frac 0.3; seed 42; n_iter 50; search_type random; metric corr; target target_binned_return_20; fixed_params none. catboost refit-0 best_params: depth 4, iterations 2000, l2_leaf_reg 10.0, learning_rate 0.005, min_data_in_leaf 100, rsm 0.1. Same protocol across all 5 arms; only the model class varied.

## What was found
MEMORY-BACKED (e79-fillmask-equals-loader.md). fillmask-vs-mommask (= parent-vs-mommask, sign-flipped): **per-cell fill beats whole-row gating in 20/20 model×block cells, 15/20 at |t|≥2** (ΔCORR +0.01..+0.045). But **fillmask only TIES the raw block — it adds no signal over v2.0**; its value is correctness and making the fill explicit for the live path. From the metrics (own ~520-period windows, standalone, not paired): best arm lgb_ktz payout 0.07272 / CORR 0.15075 / MMC 0.05765 (n 520); then catboost 0.06472, rf 0.06167, mlp 0.05852, ridge 0.05301.
